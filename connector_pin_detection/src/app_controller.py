"""应用主控协调器"""
import os, cv2, numpy as np
from src.image_processing.preprocess import Preprocessor
from src.image_processing.pin_detection import PinDetector
from src.image_processing.roi_extraction import ROIExtractor
from src.measurement.pitch_calculation import PitchCalculator
from src.measurement.calibration import Calibrator
from src.classification.defect_classifier import DefectClassifier, DefectType
from src.utils.config import Config
from src.camera import CameraManager, CameraConfig, CameraSource, CameraProfile
from src.watcher import HotFolderWatcher


class AppController:
    def __init__(self, config_path=None):
        self.config = Config(config_path)
        self.current_image = None
        self.result_image = None
        self.detection_result = None
        self.classification_result = None
        self.pitches = []
        self.latest_report = ""
        self.current_source = ""
        self._image_scale = 1.0
        self._init_modules()
        self._init_camera()
        self._watcher = None
        self._watch_result_callback = None

    def _init_modules(self):
        cfg = self.config.raw
        self.preprocessor = Preprocessor(cfg.get("image_processing", {}))
        self.detector = PinDetector(cfg.get("pin_detection", {}))
        self.roi_extractor = ROIExtractor(cfg.get("roi_extraction", {}))
        self.calibrator = Calibrator(cfg.get("calibration", {}))
        cal_cfg = cfg.get("calibration", {})
        if cal_cfg.get("pixel_per_mm"):
            self.calibrator.load_scale_from_config(cal_cfg["pixel_per_mm"])
        pitch_cfg = cfg.get("measurement", {})
        self.pitch_calculator = PitchCalculator(pixel_per_mm=self.calibrator.scale)
        cls_cfg = dict(cfg.get("classification", {}))
        meas_cfg = cfg.get("measurement", {})
        if "nominal_pitch_mm" in meas_cfg:
            cls_cfg["nominal_pitch_mm"] = meas_cfg["nominal_pitch_mm"]
        self.classifier = DefectClassifier(cls_cfg)
        self._auto_detection = cfg.get("system", {}).get("auto_detection", True)

    def _init_camera(self):
        cam_cfg = self.config.raw.get("camera", {})
        src_map = {"usb": CameraSource.USB, "rtsp": CameraSource.RTSP, "file": CameraSource.FILE}
        source = src_map.get(cam_cfg.get("source", "usb"), CameraSource.USB)
        profiles_raw = cam_cfg.get("device_profiles", [])
        profiles = []
        for p in profiles_raw:
            profiles.append(CameraProfile(
                name=p.get("name", "unknown"), device_index=p.get("device_index", 0),
                resolution_width=p.get("resolution_width", 1920),
                resolution_height=p.get("resolution_height", 1080)))
        self.cam_config = CameraConfig(
            source=source, device_index=cam_cfg.get("device_index", 0),
            device_profiles=profiles, rtsp_url=cam_cfg.get("rtsp_url", ""),
            watch_folder=cam_cfg.get("watch_folder", ""),
            resolution_width=cam_cfg.get("resolution_width", 1920),
            resolution_height=cam_cfg.get("resolution_height", 1080),
            exposure=cam_cfg.get("exposure", -1), brightness=cam_cfg.get("brightness", -1),
            contrast=cam_cfg.get("contrast", -1), save_dir=cam_cfg.get("save_dir", ""),
            auto_run_detection=cam_cfg.get("auto_run_detection", True))
        self.camera = CameraManager(self.cam_config)

    # ========== 摄像头代理方法 ==========

    def camera_list_devices(self, max_index=10):
        return self.camera.list_devices(max_index)

    def camera_get_profiles(self):
        return self.camera.config.device_profiles

    def camera_switch_device(self, index):
        return self.camera.switch_device(index)

    def camera_open_live(self):
        return self.camera.open()

    def camera_capture(self):
        frame = self.camera.capture()
        if frame is not None:
            self._set_image(frame.image, frame.source_info)
            if self.config.raw.get("camera", {}).get("auto_run_detection", True):
                self.run_detection()
            return True
        return False

    def load_image(self, path):
        # Windows下 cv2.imread不支持中文路径，用 np.fromfile + imdecode 代替
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError("Cannot load: " + path)
        self._set_image(img, "File:" + os.path.basename(path))

    def _set_image(self, img, source_info=""):
        # 自动缩放到目标最大尺寸，使不同分辨率的引脚像素面积一致
        target_max = self.config.raw.get("system", {}).get("target_max_dim", 800)
        h, w = img.shape[:2]
        max_dim = max(w, h)
        if max_dim > target_max:
            scale = target_max / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # 更新标定比例：像素缩小了，pixel_per_mm 要等比放大
            if self.calibrator.scale is not None:
                ori_pxm = self.calibrator.scale
                self.calibrator.load_scale_from_config(ori_pxm / scale)
            self._image_scale = scale
        else:
            self._image_scale = 1.0
        self.current_image = img
        self.current_source = source_info
        self.result_image = None
        self.detection_result = None
        self.classification_result = None
        self.pitches = []
        self.latest_report = ""

    def _on_watch_image(self, img, fname):
        self._set_image(img, "Watch:" + fname)
        if self._auto_detection:
            self.run_detection()
        if self._watch_result_callback:
            self._watch_result_callback()

    def start_watch(self, folder: str, on_result=None):
        """启动文件夹监听"""
        self.stop_watch()
        self._watch_result_callback = None
        self._watcher = HotFolderWatcher(folder, self._on_watch_image)
        self._watch_result_callback = on_result
        self._watcher.start()

    def stop_watch(self):
        """停止文件夹监听"""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    def run_detection(self):
        """执行完整检测流水线"""
        if self.current_image is None:
            return None
        image = self.current_image

        # Compute row bands and validate them
        bands, std_smooth = self.preprocessor._compute_row_bands(image)
        filtered = self.preprocessor._filter_pins_from_binary(self.preprocessor.process_for_pins(image), self.config.raw.get("pin_detection", {}))
        valid_bands = self.detector._validate_bands(filtered, bands, std_smooth)
        # Recompute binary using only valid bands
        binary_for_pins = self.preprocessor.process_for_pins_with_bands(image, valid_bands)
        det_result = self.detector.detect(binary_for_pins, image)

        if not det_result.success or len(det_result.pins) < 2:
            self.detection_result = det_result
            self._build_report()
            return None

        # ROI提取
        roi_img, _, _ = self.roi_extractor.extract(image, image)
        if roi_img is not None:
            binary_for_roi = self.preprocessor.process_for_pins(roi_img)
            roi_det = self.detector.detect(binary_for_roi, roi_img)
            if roi_det.success:
                # Re-validate bands against ROI result to filter noise rows
                roi_bands, roi_std_smooth = self.preprocessor._compute_row_bands(roi_img)
                roi_filtered = self.preprocessor._filter_pins_from_binary(binary_for_roi, self.config.raw.get(b"pin_detection", {}))
                roi_valid_bands = self.detector._validate_bands(roi_filtered, roi_bands, roi_std_smooth)
                if len(roi_valid_bands) < len(roi_bands):
                    # Some bands were filtered out, re-detect with valid bands
                    binary_for_roi2 = self.preprocessor.process_for_pins_with_bands(roi_img, roi_valid_bands)
                    roi_det = self.detector.detect(binary_for_roi2, roi_img)
                det_result = roi_det

        # 恢复索引 -> 间距计算 -> 分类 -> 标注
        self._restore_pin_indices(det_result.pins)
        self.pitches = self._calculate_row_pitches(det_result.pins)
        cls_result = self.classifier.classify(self.pitches, det_result.pins)
        self._restore_pin_indices(det_result.pins)  # classify 内部会改索引，恢复之

        self.result_image = self.detector.draw_pins(self.current_image, det_result)
        self._annotate_result(self.result_image, det_result.pins, cls_result)
        self.detection_result = det_result
        self.classification_result = cls_result
        self._build_report()
        return cls_result

    def _annotate_result(self, image, pins, cls_result):
        for i, pitch in enumerate(self.pitches):
            p1_idx, p2_idx = pitch[0], pitch[1]
            p1 = next(p for p in pins if p.index == p1_idx)
            p2 = next(p for p in pins if p.index == p2_idx)
            cx = int((p1.center[0] + p2.center[0]) / 2)
            cy = int((p1.center[1] + p2.center[1]) / 2)

            detail = cls_result.pin_details[i] if i < len(cls_result.pin_details) else None
            if detail and detail[2] != DefectType.OK:
                color = (0, 0, 255)
                label = str(detail[3])
            else:
                color = (0, 255, 0)
                label = str(pitch[3])
            cv2.line(image, (int(p1.center[0]), int(p1.center[1])),
                     (int(p2.center[0]), int(p2.center[1])), color, 1)
            cv2.putText(image, label, (cx - 20, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _restore_pin_indices(self, pins):
        """恢复原始引脚索引，排序 + 重编号"""
        pins.sort(key=lambda p: (p.center[1], p.center[0]))
        for i, p in enumerate(pins):
            p.index = i + 1

    def _calculate_row_pitches(self, pins):
        """多排行内间距计算。所有引脚 y 差 ≤ 5 时视为单行。"""
        ys = [p.center[1] for p in pins]
        # 若所有引脚在水平线上，不拆分
        if max(ys) - min(ys) <= 5:
            rows = [sorted(pins, key=lambda p: p.center[0])]
        else:
            mean_y = sum(ys) / len(ys)
            rows = [[], []]
            for p in pins:
                rows[0 if p.center[1] <= mean_y else 1].append(p)
            for r in rows:
                r.sort(key=lambda p: p.center[0])
        from src.image_processing.pin_detection import DetectionResult
        all_pitches = []
        for row_pins in rows:
            if len(row_pins) >= 2:
                row_result = DetectionResult()
                row_result.pins = row_pins
                row_result.success = True
                all_pitches.extend(self.pitch_calculator.calculate_pitches(row_result))
        return all_pitches

    # ========== 自动标定 ==========

    def auto_calibrate(self, nominal_pitch_mm: float) -> dict:
        """
        自动标定：用当前检测结果中的平均像素间距计算 pixel_per_mm。
        """
        if not self.pitches or len(self.pitches) < 2:
            return {"success": False, "message": "请先拍照/加载图片并执行检测"}

        pixel_distances = [p[2] for p in self.pitches]
        # 剔除离群间距：偏离中位数超过 30% 的视为异常（弯曲/误检），保留剩余计算均值
        if len(pixel_distances) >= 3:
            arr = np.array(pixel_distances)
            med = np.median(arr)
            filtered = [x for x in pixel_distances if abs(x - med) / med <= 0.30]
            if filtered:
                pixel_distances = filtered
        avg_px = float(np.mean(pixel_distances))

        if avg_px <= 0:
            return {"success": False, "message": "像素间距无效"}
        if nominal_pitch_mm <= 0:
            return {"success": False, "message": "标称间距必须大于 0"}

        effective_ppm = nominal_pitch_mm / avg_px
        raw_ppm = effective_ppm * self._image_scale

        self.calibrator.load_scale_from_config(effective_ppm)
        self.pitch_calculator.pixel_per_mm = effective_ppm

        if "calibration" not in self.config.raw:
            self.config.raw["calibration"] = {}
        self.config.raw["calibration"]["pixel_per_mm"] = round(effective_ppm, 6)
        self.config.save()

        if self.current_image is not None:
            self.run_detection()

        return {
            "success": True,
            "pixel_per_mm": round(effective_ppm, 6),
            "avg_pixel_distance": round(avg_px, 2),
            "nominal_pitch_mm": nominal_pitch_mm,
            "pitch_count": len(pixel_distances),
            "message": (
                f"标定完成: pixel_per_mm = {round(effective_ppm, 6)} "
                f"(标称间距{nominal_pitch_mm}mm / 平均像素{round(avg_px, 2)}px)"
            )
        }

    def _build_report(self):
        lines = ["Source: " + self.current_source]
        # 引脚中心坐标
        if self.detection_result and self.detection_result.pins:
            lines.append("Pins:")
            for p in self.detection_result.pins:
                lines.append(f"  Pin{p.index}: center=({p.center[0]:.0f},{p.center[1]:.0f})")
        r = self.classification_result
        if r:
            lines.append("Verdict: " + r.overall_verdict)
            lines.append("Info: " + r.message)
            if r.pitch_stats:
                lines.append("Pitch stats:")
                for k, v in r.pitch_stats.items():
                    lines.append("  " + str(k) + ": " + str(v))
            if r.angle_stats:
                lines.append("Angle stats:")
                for k, v in r.angle_stats.items():
                    lines.append("  " + str(k) + ": " + str(v))
            if r.row_info:
                lines.append("Rows: " + str(r.row_info["rows"]))
            if r.missing_pins:
                lines.append("Missing: " + str(r.missing_pins))
            if r.bent_pins:
                lines.append("Bent pins:")
                for b in r.bent_pins:
                    lines.append("  Pin" + str(b[0]) + " angle=" + str(round(b[1], 1)))
            if r.pin_details:
                lines.append("Per-pitch:")
                for d in r.pin_details:
                    lines.append("  Pin" + str(d[0]) + "-Pin" + str(d[1]) + ": " + d[2] + " (" + str(d[3]) + "mm dev=" + str(round(d[4], 4)) + "mm)")
            lines.append("Defect counts: " + str(r.defect_counts))
        else:
            lines.append("No detection")
        self.latest_report = "\n".join(lines)

    def save_report_html(self, path):
        r = self.classification_result
        if not r:
            with open(path, "w", encoding="utf-8") as f:
                f.write("<html><body><h2>No detection</h2></body></html>")
            return
        d = r.to_dict()
        color_map = {"OK": "green", "MISSING_PIN": "red", "BENT_PIN": "orange", "PITCH_TOO_WIDE": "red", "PITCH_TOO_NARROW": "red"}
        vc = color_map.get(d["verdict"], "gray")
        details = "".join(
            '<tr><td>' + str(pd["p1"]) + "-" + str(pd["p2"]) + '</td><td>' + str(pd["mm"]) + '</td><td>' + str(pd["deviation"]) + '</td><td style=color:' + ('red' if pd["defect"]!="OK" else "green") + '>' + pd["defect"] + '</td></tr>'
            for pd in d["pin_details"])
        html = (
            '<html><meta charset=utf-8><title>Report</title>'
            '<style>body{font-family:sans-serif;margin:20px}table{border-collapse:collapse;width:100%}'
            'th,td{border:1px solid #ccc;padding:6px;text-align:center}th{background:#f0f0f0}</style>'
            '<body><h1 style=color:' + vc + '>Connector Pin Report</h1>'
            '<p><b>Source:</b> ' + self.current_source + '</p>'
            '<p><b>Verdict:</b> <span style=color:' + vc + ';font-size:1.2em>' + d["verdict"] + '</span></p>'
            '<p><b>Message:</b> ' + d["message"] + '</p>'
            '<table><tr><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>'
            '<tr><td>' + str(d["stats"]["mean"]) + '</td><td>' + str(d["stats"]["std"]) + '</td><td>'
            + str(d["stats"]["min"]) + '</td><td>' + str(d["stats"]["max"]) + '</td><td>'
            + str(d["stats"]["count"]) + '</td></tr></table>'
            '<h3>Per-pitch</h3><table><tr><th>Pair</th><th>mm</th><th>Dev</th><th>Result</th></tr>'
            + details + '</table>'
            '<h3>Defect Counts</h3><p>' + str(d["defect_counts"]) + '</p></body></html>')
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def batch_process(self, folder):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
        report = []
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                self.load_image(path)
                cls_result = self.run_detection()
                status = cls_result.overall_verdict if cls_result else "FAIL"
                report.append((fname, status))
            except Exception as e:
                report.append((fname, "ERR: " + str(e)))
        with open(os.path.join(folder, "batch_report.txt"), "w", encoding="utf-8") as f:
            for n, s in report:
                f.write(n + "\t" + s + "\n")
        rows_str = "".join(
            '<tr><td>' + n + '</td><td style=color:' + ('red' if s!="OK" else "green") + '>' + s + '</td></tr>'
            for n, s in report)
        with open(os.path.join(folder, "batch_report.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><meta charset=utf-8><body><h2>Batch Report</h2>'
                '<table><tr><th>File</th><th>Result</th></tr>'
                + rows_str + '</table></body></html>')
        return report

    def save_report(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.latest_report or "No detection\n")
