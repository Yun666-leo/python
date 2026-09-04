"""缺陷判定模块 - 支持缺引脚、引脚弯曲、多排引脚检测"""

# 开发思路：接收引脚间距和引脚检测结果，统一完成质量判定。
# 整体步骤：
# 1. 创建结果对象，并整理多排引脚信息。
# 2. 统计间距数据，检测缺引脚和弯曲引脚。
# 3. 对照标称间距和公差，判断每段间距是否偏大或偏小。
# 4. 按缺引脚、弯曲引脚、间距异常的优先级生成总体结论。
# 5. 汇总缺陷数量，并通过结果对象或 to_dict() 输出检测结果。
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
        self.pin_details       = []  # 每项：(引脚1索引, 引脚2索引, 缺陷类型, 间距mm, 偏差mm)
        self.pitch_stats       = {}  # {"mean": 2.0, "std": 0.01, "min": 1.99, "max": 2.01, "count": 2}
        self.message           = ""
        self.missing_pins      = []  # 存放疑似缺失引脚的索引
        self.bent_pins         = []  # 每项：(引脚索引, 实际角度, 主方向角度)
        self.row_info          = {}
        self.angle_stats       = {}  # {"mean": 1.2, "std": 2.5, "min": -3.0, "max": 12.5, "dominant": 0.0}
        self.defect_counts     = {"合格": 0, "间距偏大": 0, "间距偏小": 0, "缺引脚": 0, "引脚弯曲": 0}  

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
            "bent_pins": [{"index": b[0], "angle": round(b[1], 2), "dominant_angle": round(b[2], 2)} for b in self.bent_pins],
            "row_info": self.row_info,
        }


class DefectClassifier:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.tolerance            = self.config.get("pitch_tolerance_mm", 0.05)
        self.nominal_pitch        = self.config.get("nominal_pitch_mm", None)
        self.missing_pin_ratio    = self.config.get("missing_pin_ratio", 1.6)
        self.bent_angle_threshold = self.config.get("bent_angle_threshold", 8.0)

    # 对输入的引脚间距和引脚点进行总判定，并返回统一结果对象
    def classify(self, pitches: list, pins: list = None):#pitches=[ (引脚1索引, 引脚2索引, 像素距离, 毫米距离)]
        result = ClassificationResult()
        self._attach_row_info(result, pins)# 统计多排引脚的分组情况，附加到结果中

        if not pitches:
            result.overall_verdict = DefectType.UNKNOWN
            result.message = "未检测到有效引脚间距"
            return result

        result.pitch_stats = self._build_pitch_stats(pitches)# 汇总所有引脚间距的均值、标准差和范围等统计信息
        has_bent = self._detect_bent_pins(result, pins)# 检测哪些引脚角度明显偏离主流方向，判定为弯曲
        result.missing_pins = self._detect_missing_pins(pitches)
        has_missing = bool(result.missing_pins)

        if self.nominal_pitch is not None:
            self._evaluate_pitches(pitches, result)
        elif not result.message:
            result.message = "未设置标称间距，仅输出测量值"

        result.overall_verdict, result.message = self._build_verdict(result, has_missing, has_bent)
        self._update_defect_counts(result)
        return result

    # 统计多排引脚的分组情况，附加到结果中
    def _attach_row_info(self, result, pins):
        if pins and len(pins) >= 4:
            rows = self._cluster_rows(pins)
            if len(rows) > 1:
                ri = {"rows": len(rows)}
                for rid, pl in rows.items():
                    ri["row" + str(rid) + "_count"] = len(pl)
                    ri["row" + str(rid) + "_pins"] = [p.index for p in pl]
                result.row_info = ri

    # 汇总所有引脚间距的均值、标准差和范围等统计信息
    def _build_pitch_stats(self, pitches):
        pitch_values = [p[3] for p in pitches]
        return {
            "mean": round(float(np.mean(pitch_values)), 4),
            "std": round(float(np.std(pitch_values)), 4),
            "min": round(float(np.min(pitch_values)), 4),
            "max": round(float(np.max(pitch_values)), 4),
            "count": len(pitch_values)
        }

    # 将角度归一化到 0~90 度范围，避免正负方向导致误判
    def _normalize_angle(self, angle):
        normalized = angle % 180
        return normalized if normalized <= 90 else 180 - normalized

    # 检测哪些引脚角度明显偏离主流方向，判定为弯曲
    def _detect_bent_pins(self, result, pins):
        if not pins or len(pins) < 2:
            return False

        raw_angles = [p.angle for p in pins]
        norm_angles = [self._normalize_angle(a) for a in raw_angles]
        median_angle = float(np.median(norm_angles))
        result.angle_stats = {
            "mean": round(float(np.mean(raw_angles)), 2),
            "std": round(float(np.std(raw_angles)), 2),
            "min": round(float(np.min(raw_angles)), 2),
            "max": round(float(np.max(raw_angles)), 2),
            "dominant": round(median_angle, 2),
        }

        has_bent = False
        for i, p in enumerate(pins):
            dev = abs(norm_angles[i] - median_angle)
            if dev > self.bent_angle_threshold:
                result.bent_pins.append((p.index, float(raw_angles[i]), median_angle))
                has_bent = True
        return has_bent

    # 根据缺失、弯曲和间距偏差的优先级，生成最终结论和描述
    def _build_verdict(self, result, has_missing, has_bent):
        defects_found = [d[2] for d in result.pin_details if d[2] != DefectType.OK]

        if has_missing:
            return DefectType.MISSING_PIN, "检测到 " + str(len(result.missing_pins)) + " 处缺引脚"
        if has_bent:
            return DefectType.BENT_PIN, "检测到 " + str(len(result.bent_pins)) + " 个弯曲引脚"
        if defects_found:
            wide = sum(1 for d in defects_found if d == DefectType.PITCH_TOO_WIDE)
            narrow = sum(1 for d in defects_found if d == DefectType.PITCH_TOO_NARROW)
            parts = []
            if wide:
                parts.append(str(wide) + "处偏大")
            if narrow:
                parts.append(str(narrow) + "处偏小")
            max_dev = max((d[4] for d in result.pin_details if d[2] != DefectType.OK), default=0)
            verdict = DefectType.PITCH_TOO_WIDE if wide >= narrow else DefectType.PITCH_TOO_NARROW
            message = "间距超差（" + "、".join(parts) + "），最大偏差 " + format(max_dev, ".4f") + "mm"
            return verdict, message
        if result.message:
            return result.overall_verdict, result.message
        return DefectType.OK, "所有引脚间距在公差范围内"

    # 将各类缺陷的详细结果汇总成计数表，便于输出和统计
    def _update_defect_counts(self, result):
        for d in result.pin_details:
            result.defect_counts[d[2]] = result.defect_counts.get(d[2], 0) + 1
        result.defect_counts["缺引脚"] = len(result.missing_pins)
        result.defect_counts["引脚弯曲"] = len(result.bent_pins)

    # 检测是否存在引脚缺失，按间距异常倍数与行内中位数判断
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
            above_count = sum(1 for v in row_mm if v > median_pitch * 1.4)
            is_calibration_error = above_count > len(row_mm) * 0.5

            for j, p in enumerate(row_pitches):
                if p[3] > threshold and not is_calibration_error:
                    missing.append(start + j)

        return missing

    # 按 Y 坐标间隙把引脚分成多行，适配双排或多排器件
    def _cluster_rows(self, pins, gap_threshold: int = 80):
        """按 Y 坐标间隙把引脚分成多行（每一行一组）。"""
        if not pins:
            return {}

        if len(pins) < 4:
            return {0: pins}

        sorted_pins = sorted(pins, key=lambda p: p.center[1])
        rows = []

        for pin in sorted_pins:
            if rows and abs(pin.center[1] - rows[-1][0].center[1]) <= gap_threshold:
                rows[-1].append(pin)
            else:
                rows.append([pin])

        for row in rows:
            row.sort(key=lambda p: p.center[0])
            for j, p in enumerate(row):
                p.index = j + 1

        return {rid: row for rid, row in enumerate(rows)}

    # 按标称间距评估每个引脚间距是否偏大或偏小
    def _evaluate_pitches(self, pitches, result):
        for p in pitches:
            p1, p2, dist_px, dist_mm = p
            deviation = abs(dist_mm - self.nominal_pitch)
            if deviation > self.tolerance:
                dt = DefectType.PITCH_TOO_WIDE if dist_mm > self.nominal_pitch else DefectType.PITCH_TOO_NARROW
                result.pin_details.append((p1, p2, dt, dist_mm, deviation))
            else:
                result.pin_details.append((p1, p2, DefectType.OK, dist_mm, deviation))

