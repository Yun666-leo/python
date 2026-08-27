"""缺陷判定模块 - 支持缺引脚、引脚弯曲、多排引脚检测"""
import numpy as np


class DefectType:
    OK                 = "合格"
    PITCH_TOO_WIDE     = "间距偏大"
    PITCH_TOO_NARROW   = "间距偏小"
    MISSING_PIN        = "缺引脚"
    BENT_PIN           = "引脚弯曲"
    POSITION_OFFSET    = "位置偏移"
    UNKNOWN            = "未知异常"


class ClassificationResult:
    def __init__(self):
        self.overall_verdict   = DefectType.OK
        self.pin_details       = []
        self.pitch_stats       = {}
        self.message           = ""
        self.missing_pins      = []
        self.bent_pins         = []
        self.row_info          = None
        self.angle_stats       = {}
        self.defect_counts     = {"合格": 0, "间距偏大": 0, "间距偏小": 0, "缺引脚": 0, "引脚弯曲": 0}
        self.pin_angle_details = []

    def to_dict(self) -> dict:
        return {
            "verdict": self.overall_verdict,
            "message": self.message,
            "stats": self.pitch_stats,
            "angle_stats": self.angle_stats,
            "defect_counts": self.defect_counts,
            "pin_details": [
                {"p1": d[0], "p2": d[1], "defect": d[2], "mm": d[3], "deviation": d[4]}
                for d in self.pin_details
            ],
            "missing_pins": self.missing_pins,
            "bent_pins": [{"index": b[0], "angle": round(b[1], 2), "avg_angle": round(b[2], 2)} for b in self.bent_pins],
            "row_info": self.row_info,
        }


class DefectClassifier:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.tolerance            = self.config.get("pitch_tolerance_mm", 0.05)
        self.nominal_pitch        = self.config.get("nominal_pitch_mm", None)
        self.missing_pin_ratio    = self.config.get("missing_pin_ratio", 1.6)
        self.bent_angle_threshold = self.config.get("bent_angle_threshold", 8.0)

    def classify(self, pitches: list, pins: list = None):
        result = ClassificationResult()

        # multi-row detection runs even with empty pitches
        if pins and len(pins) >= 4:
            rows = self._cluster_rows(pins)
            if len(rows) > 1:
                ri = {"rows": len(rows)}
                for rid, pl in rows.items():
                    ri["row" + str(rid) + "_count"] = len(pl)
                    ri["row" + str(rid) + "_pins"] = [p.index for p in pl]
                result.row_info = ri

        if not pitches:
            result.overall_verdict = DefectType.UNKNOWN
            result.message = "未检测到有效引脚间距"
            return result

        pitch_values = [p[3] for p in pitches]
        result.pitch_stats = {
            "mean": round(float(np.mean(pitch_values)), 4),
            "std": round(float(np.std(pitch_values)), 4),
            "min": round(float(np.min(pitch_values)), 4),
            "max": round(float(np.max(pitch_values)), 4),
            "count": len(pitch_values)
        }

        # bent pin detection — 角度归一化+基于中位数的异常检测
        has_bent = False
        if pins and len(pins) >= 2:
            raw_angles = [p.angle for p in pins]
            # 归一化所有角度到 [0, 90]，解决 -90/90 环绕问题
            # 归一化后：0=水平, 90=竖直
            norm_angles = []
            for a in raw_angles:
                n = a % 180
                if n > 90:
                    n = 180 - n
                norm_angles.append(n)
            # 以中位数为主方向（稳健，不受少数异常影响）
            median_angle = float(np.median(norm_angles))
            result.angle_stats = {
                "mean": round(float(np.mean(raw_angles)), 2),
                "std": round(float(np.std(raw_angles)), 2),
                "min": round(float(np.min(raw_angles)), 2),
                "max": round(float(np.max(raw_angles)), 2),
                "norm_median": round(median_angle, 2),
            }
            for i, p in enumerate(pins):
                dev_from_major = abs(norm_angles[i] - median_angle)
                result.pin_angle_details.append(
                    (p.index, round(raw_angles[i], 2), round(dev_from_major, 2),
                     dev_from_major > self.bent_angle_threshold)
                )
                if dev_from_major > self.bent_angle_threshold:
                    result.bent_pins.append((p.index, float(raw_angles[i]), median_angle))
                    has_bent = True

        # missing pin detection
        has_missing = False
        missing_idx = self._detect_missing_pins(pitches, result.pitch_stats)
        if missing_idx:
            result.missing_pins = missing_idx
            has_missing = True

        # pitch evaluation
        if self.nominal_pitch is not None:
            self._evaluate_pitches(pitches, result)
        else:
            result.message = "未设置标称间距，仅输出测量值"

        # overall verdict priority: missing > bent > pitch > OK
        defects_found = [d[2] for d in result.pin_details if d[2] != DefectType.OK]
        if has_missing:
            result.overall_verdict = DefectType.MISSING_PIN
            result.message = "检测到 " + str(len(result.missing_pins)) + " 处缺引脚"
        elif has_bent:
            result.overall_verdict = DefectType.BENT_PIN
            result.message = "检测到 " + str(len(result.bent_pins)) + " 个弯曲引脚"
        elif defects_found:
            wide = sum(1 for d in defects_found if d == DefectType.PITCH_TOO_WIDE)
            narrow = sum(1 for d in defects_found if d == DefectType.PITCH_TOO_NARROW)
            parts = []
            if wide: parts.append(str(wide) + "处偏大")
            if narrow: parts.append(str(narrow) + "处偏小")
            max_dev = max((d[4] for d in result.pin_details if d[2] != DefectType.OK), default=0)
            if wide >= narrow:
                result.overall_verdict = DefectType.PITCH_TOO_WIDE
            else:
                result.overall_verdict = DefectType.PITCH_TOO_NARROW
            result.message = "间距超差（" + "、".join(parts) + "），最大偏差 " + format(max_dev, ".4f") + "mm"
        else:
            if not result.message:
                result.message = "所有引脚间距在公差范围内"

        # defect counts
        for d in result.pin_details:
            result.defect_counts[d[2]] = result.defect_counts.get(d[2], 0) + 1
        result.defect_counts["缺引脚"] = len(result.missing_pins)
        result.defect_counts["引脚弯曲"] = len(result.bent_pins)
        return result

    def _detect_missing_pins(self, pitches, stats=None):
        # 基于行内中位数的局部异常检测
        if not pitches:
            return []
        # 只有 2 个间距（如 3 引脚场景）也检查
        if len(pitches) == 2:
            p0, p1 = pitches[0][3], pitches[1][3]
            if min(p0, p1) > 0 and max(p0, p1) / min(p0, p1) >= self.missing_pin_ratio:
                return [0 if p0 > p1 else 1]
            return []

        # 检测行边界：从 pitch 的 p1/p2 索引推测换行位置
        row_boundaries = [0]
        for i in range(1, len(pitches)):
            prev_p2 = pitches[i - 1][1]
            curr_p1 = pitches[i][0]
            if curr_p1 - prev_p2 > 1 or curr_p1 < prev_p2:
                row_boundaries.append(i)
        row_boundaries.append(len(pitches))

        missing = []
        for ri in range(len(row_boundaries) - 1):
            start = row_boundaries[ri]
            end = row_boundaries[ri + 1]
            row_pitches = pitches[start:end]
            if len(row_pitches) < 3:
                continue

            row_mm = [p[3] for p in row_pitches]
            median_pitch = float(np.median(row_mm))
            if median_pitch <= 0:
                continue

            threshold = median_pitch * self.missing_pin_ratio
            # 超过30%间距都显著偏大(>中位数x1.3) => 整体标定问题
            above_count = sum(1 for v in row_mm if v > median_pitch * 1.4)
            is_calibration_error = above_count > len(row_mm) * 0.5

            for j, p in enumerate(row_pitches):
                if p[3] > threshold and not is_calibration_error:
                    missing.append(start + j)

        return missing
        row_boundaries.append(len(pitches))
        missing = []
        for ri in range(len(row_boundaries) - 1):
            start = row_boundaries[ri]
            end = row_boundaries[ri + 1]
            row_pitches = pitches[start:end]
            if len(row_pitches) < 3:
                continue
            row_mm = [p[3] for p in row_pitches]
            median_pitch = float(np.median(row_mm))
            if median_pitch <= 0:
                continue
            threshold = median_pitch * self.missing_pin_ratio
            above_count = sum(1 for v in row_mm if v > median_pitch * 1.3)
            is_calibration_error = above_count > len(row_mm) * 0.3
            for j, p in enumerate(row_pitches):
                if p[3] > threshold and not is_calibration_error:
                    missing.append(start + j)
        return missing
    def _cluster_rows(self, pins):
        y_coords = np.array([p.center[1] for p in pins])
        if len(pins) < 4:
            return {0: pins}
        mean_y = float(np.mean(y_coords))
        rows = {0: [], 1: []}
        for p in pins:
            rows[0 if p.center[1] <= mean_y else 1].append(p)
        for rid in rows:
            rows[rid].sort(key=lambda p: p.center[0])
            for j, p in enumerate(rows[rid]):
                p.index = j + 1
        return rows

    def _evaluate_pitches(self, pitches, result):
        for p in pitches:
            p1, p2, dist_px, dist_mm = p
            deviation = abs(dist_mm - self.nominal_pitch)
            if deviation > self.tolerance:
                dt = DefectType.PITCH_TOO_WIDE if dist_mm > self.nominal_pitch else DefectType.PITCH_TOO_NARROW
                result.pin_details.append((p1, p2, dt, dist_mm, deviation))
            else:
                result.pin_details.append((p1, p2, DefectType.OK, dist_mm, deviation))
