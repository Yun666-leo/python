"""引脚检测模块"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Pin:
    """单引脚检测结果"""
    index: int = 0
    center: tuple = (0, 0)
    bbox: tuple = (0, 0, 0, 0)
    area: float = 0.0
    angle: float = 0.0
    width: float = 0.0
    height: float = 0.0
    contour: Optional[np.ndarray] = None


@dataclass
class DetectionResult:
    """引脚检测结果容器"""
    pins: list = field(default_factory=list)
    success: bool = False
    error_msg: str = ""
    raw_image_shape: tuple = None


class PinDetector:
    """引脚检测器"""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def detect(self, binary: np.ndarray, original: np.ndarray = None) -> DetectionResult:
        """执行引脚检测"""
        result = DetectionResult()
        result.raw_image_shape = binary.shape
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pins = self._filter_pins(contours)
            # 按行分组后，对每行做网格对齐（合并断裂的引脚）
            pins = self._grid_align(pins)
            pins = self._align_y(pins)
            pins = self._normalize_bbox_height(pins, target_height=150)
            pins.sort(key=lambda p: p.center[0])
            for i, pin in enumerate(pins):
                pin.index = i + 1
            result.pins = pins
            result.success = True
        except Exception as e:
            result.error_msg = str(e)
        return result

    @staticmethod
    def _validate_bands(pins: list, bands: list, std_smooth: np.ndarray = None) -> list:
        """Filter out row bands that are likely noise.

        Two criteria:
        1. Area ratio: median pin area in a band must be >= 75% of the main band's median.
        2. Std ratio (optional): peak std-dev in a band must be >= 85% of the max peak.
           Pass std_smooth array from _find_pin_rows to enable this.
        """
        if len(bands) < 2 or not pins:
            return bands
        # Compute median area per band
        band_medians = []
        for s, e in bands:
            areas = [p.area for p in pins if s <= p.center[1] <= e]
            band_medians.append((s, e, float(np.median(areas)) if areas else 0.0))
        # Find the band with highest median area (main pin row)
        main_band = max(band_medians, key=lambda b: b[2])
        if main_band[2] <= 0:
            return bands
        # Area filter: keep bands with median area >= 75% of main
        area_valid = [(s, e) for s, e, med in band_medians if med >= main_band[2] * 0.75]
        if not area_valid:
            return [main_band[:2]]
        # Std filter (optional): if std_smooth provided, also check peak std ratio
        if std_smooth is not None:
            band_peaks = []
            for s, e in bands:
                peak = float(np.max(std_smooth[s:e])) if e > s else 0.0
                band_peaks.append((s, e, peak))
            max_peak = max(p[2] for p in band_peaks)
            if max_peak > 0:
                valid = [(s, e) for s, e, peak in band_peaks
                         if (s, e) in set(a[:2] for a in area_valid) and peak >= max_peak * 0.85]
                if not valid:
                    return area_valid
                return valid
        return area_valid

    def _grid_align(self, pins: list) -> list:
        """对每行引脚做网格对齐。

        同一行中，完整引脚和断裂碎片交替出现（上下半断裂）。
        通过计算中位间距，将碎片归并到最近的网格位置。
        """
        # 先做垂直合并：同列上下半断裂的引脚合并为一个
        if len(pins) >= 4:
            pins = self._merge_vertical(pins)

        if len(pins) < 4:
            return pins

        # 按 y 分两行（水平合并）
        ys = [p.center[1] for p in pins]
        mean_y = sum(ys) / len(ys)
        rows = [[], []]
        for p in pins:
            rows[0 if p.center[1] <= mean_y else 1].append(p)

        merged = []
        for row in rows:
            row.sort(key=lambda p: p.center[0])
            if len(row) < 2:
                merged.extend(row)
                continue

            # 计算中位间距
            xs = np.array([p.center[0] for p in row])
            diffs = np.diff(xs)
            median_pitch = np.median(diffs)
            min_allowed = median_pitch * 0.4

            # 合并过近的引脚
            result_row = [row[0]]
            for i in range(1, len(row)):
                dx = row[i].center[0] - result_row[-1].center[0]
                if dx < min_allowed:
                    # 过近：保留面积大的
                    if row[i].area > result_row[-1].area:
                        result_row[-1] = row[i]
                else:
                    result_row.append(row[i])

            merged.extend(result_row)

        return merged

    
    def _merge_vertical(self, pins: list) -> list:
        """对每行引脚做垂直合并：上下半断裂的引脚合成一个完整引脚。

        按 Y 坐标间隙（>80px）将引脚分组，每组独立合并后再汇总。
        """
        if len(pins) < 3:
            return pins
        sorted_pins = sorted(pins, key=lambda p: p.center[1])
        # 按大间隙拆分成独立的行组
        groups = []
        start = 0
        for i in range(1, len(sorted_pins)):
            if sorted_pins[i].center[1] - sorted_pins[i-1].center[1] > 80:
                groups.append(sorted_pins[start:i])
                start = i
        groups.append(sorted_pins[start:])
        # 对每组做垂直合并
        all_merged = []
        for group in groups:
            all_merged.extend(self._merge_group(group))
        return all_merged

    def _merge_group(self, group: list) -> list:
        """在单行引脚组内做垂直合并：同列上下半断裂的引脚合并为一个。"""
        n = len(group)
        matched = set()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(group[i].center[0] - group[j].center[0])
                dy = abs(group[i].center[1] - group[j].center[1])
                if dx < 20 and dy < 50:
                    pairs.append((dx, i, j))
        pairs.sort(key=lambda x: x[0])
        merge_map = {}
        for _, i, j in pairs:
            idi, idj = id(group[i]), id(group[j])
            if idi in matched or idj in matched:
                continue
            if group[i].center[1] <= group[j].center[1]:
                merge_map[idj] = idi
                group[i].area += group[j].area
            else:
                merge_map[idi] = idj
                group[j].area += group[i].area
            matched.add(idi)
            matched.add(idj)
        from src.image_processing.pin_detection import Pin
        loser_ids = set(merge_map.keys())
        winner_ids = set(merge_map.values())
        result = []
        added_winner_ids = set()
        for p in group:
            pid = id(p)
            if pid in loser_ids:
                winner_id = merge_map[pid]
                if winner_id not in added_winner_ids:
                    q = next((c for c in group if id(c) == winner_id), None)
                    if q is not None:
                        if p.contour is not None and q.contour is not None:
                            combined_cnt = np.vstack((p.contour, q.contour))
                        elif p.contour is not None:
                            combined_cnt = p.contour
                        else:
                            combined_cnt = q.contour
                        x, y, bw, bh = cv2.boundingRect(combined_cnt)
                        merged_pin = Pin(
                            center=(q.center[0], min(p.center[1], q.center[1])),
                            area=q.area,
                            bbox=(x, y, bw, bh),
                            angle=q.angle,
                            width=bw, height=bh,
                            contour=combined_cnt,
                        )
                        result.append(merged_pin)
                        added_winner_ids.add(winner_id)
        for p in group:
            pid = id(p)
            if pid not in loser_ids and pid not in winner_ids:
                result.append(p)
        return result
    @staticmethod
    def _align_y(pins):
        """将每行引脚的中心 y 统一为中位数。按 y 间隙分组后每组内对齐。"""
        if len(pins) < 3:
            return pins
        # 按 y 间隙拆分成行组
        sorted_pins = sorted(pins, key=lambda p: p.center[1])
        groups = []
        start = 0
        for i in range(1, len(sorted_pins)):
            if sorted_pins[i].center[1] - sorted_pins[i-1].center[1] > 80:
                groups.append(sorted_pins[start:i])
                start = i
        groups.append(sorted_pins[start:])
        # 每组内对齐 y
        result = []
        for group in groups:
            if len(group) < 2:
                result.extend(group)
                continue
            median_y = float(np.median([p.center[1] for p in group]))
            for p in group:
                object.__setattr__(p, 'center', (p.center[0], median_y))
            result.extend(group)
        return result
    @staticmethod
    def _normalize_bbox_height(pins, target_height=150):
        """统一所有引脚 bbox 高度为 target_height，保持中心 y 不变"""
        if not pins:
            return pins
        for p in pins:
            cx, cy = p.center
            x, y, bw, bh = p.bbox
            new_y = int(cy - target_height / 2 + 20)
            new_cy = cy + 20
            object.__setattr__(p, "bbox", (x, new_y, bw, target_height))
            object.__setattr__(p, "center", (p.center[0], new_cy))
        return pins

    def _filter_pins(self, contours: list) -> list:
        """筛选引脚轮廓"""
        min_area = self.config.get("min_pin_area", 20)
        max_area = self.config.get("max_pin_area", 5000)
        aspect_min = self.config.get("aspect_ratio_min", 1.0)
        aspect_max = self.config.get("aspect_ratio_max", 10.0)
        solidity_min = self.config.get("solidity_min", 0.6)
        pins = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            if w < h:
                w, h = h, w
                angle += 90
            aspect = w / h if h > 0 else 1
            if aspect < aspect_min or aspect > aspect_max:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < solidity_min:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pin = Pin(center=(round(cx, 1), round(cy, 1)), bbox=(x, y, bw, bh), area=area, angle=angle, width=w, height=h, contour=cnt)
            pins.append(pin)
        return pins

    @staticmethod
    def draw_pins(image: np.ndarray, result: DetectionResult, color: tuple = (0, 255, 0)) -> np.ndarray:
        """在图像上绘制检测结果"""
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pin in result.pins:
            x, y, w, h = pin.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.circle(img, (int(pin.center[0]), int(pin.center[1])), 2, (0, 0, 255), -1)
            cv2.putText(img, str(pin.index), (int(pin.center[0]) - 5, max(y - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return img
