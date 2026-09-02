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

    # 执行完整的引脚检测流程，并返回检测结果对象
    def detect(self, binary: np.ndarray, original: np.ndarray = None) -> DetectionResult:
        """执行引脚检测"""
        result = DetectionResult()
        result.raw_image_shape = binary.shape
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pins = self._filter_pins(contours)
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

    # 过滤无效候选带区，保留真正包含足够引脚的行带
    @staticmethod
    def _validate_bands(pins: list, bands: list, std_smooth: np.ndarray = None, min_pins: int = 3) -> list:
        if len(bands) < 2 or not pins:
            return bands
        band_stats = []
        for s, e in bands:
            row_pins = [p for p in pins if s <= p.center[1] <= e]
            areas = [p.area for p in row_pins]
            band_stats.append((s, e, len(row_pins), float(np.median(areas)) if areas else 0.0))
        main_band = max(band_stats, key=lambda b: b[3])
        if main_band[3] <= 0:
            return bands
        valid = [(s, e) for s, e, n, med in band_stats
                 if n >= min_pins and med >= main_band[3] * 0.75]
        if not valid:
            return [main_band[:2]]
        if std_smooth is not None:
            band_peaks = []
            for s, e in bands:
                peak = float(np.max(std_smooth[s:e])) if e > s else 0.0
                band_peaks.append((s, e, peak))
            max_peak = max(p[2] for p in band_peaks)
            if max_peak > 0:
                valid = [(s, e) for s, e, peak in band_peaks
                         if (s, e) in set(a[:2] for a in valid) and peak >= max_peak * 0.85]
                if not valid:
                    return [(s, e) for s, e, n, med in band_stats
                            if n >= min_pins and med >= main_band[3] * 0.75]
                return valid
        return valid

    # 按 Y 方向间隙把引脚拆成多行，每行视为一组
    @staticmethod
    def _split_by_y_gap(pins: list, gap_threshold: int = 80) -> list:
        """按 y 坐标间隙将引脚分成多组（每行一组）。"""
        if len(pins) < 2:
            return [pins]
        sorted_pins = sorted(pins, key=lambda p: p.center[1])
        groups = []
        for pin in sorted_pins:
            if groups and abs(pin.center[1] -groups[-1][0].center[1] ) <= gap_threshold:
                groups[-1].append(pin)
            else:
                groups.append([pin]) 
        for group in groups:
            group.sort(key=lambda p: p.center[0])

        return groups

    # 对同一行中的引脚做网格对齐和去重，减少碎片和错位
    def _grid_align(self, pins: list) -> list:
        if len(pins) >= 4:
            pins = self._merge_vertical(pins)
        if len(pins) < 4:
            return pins
        rows=self._split_by_y_gap(pins)
        merged = []
        for row in rows:
            if len(row) < 2:
                merged.extend(row)
                continue
            xs = np.array([p.center[0] for p in row])
            diffs = np.diff(xs)
            median_pitch = float(np.median(diffs)) if diffs.size else 0.0
            min_allowed = max(median_pitch * 0.4, 1.0)

            result_row = []
            cluster = [row[0]]
            for pin in row[1:]:
                dx = pin.center[0] - cluster[-1].center[0]
                if dx < min_allowed:
                    cluster.append(pin)
                else:
                    best = max(cluster, key=lambda p: p.area)
                    result_row.append(best)
                    cluster = [pin]

            if cluster:
                best = max(cluster, key=lambda p: p.area)
                result_row.append(best)

            merged.extend(result_row)
        return merged

    # 合并同一列中被分裂的近邻碎片，避免一个引脚被检测成多个
    def _merge_vertical(self, pins: list) -> list:
        if len(pins) < 3:
            return pins
        groups = self._split_by_y_gap(pins)
        all_merged = []
        for group in groups:
            all_merged.extend(self._merge_group(group))
        return all_merged

    # 将同一根引脚的相邻碎片合并成一个代表对象
    def _merge_group(self, group: list) -> list:
        """把同一根引脚被分裂成的近邻碎片合并成一个代表。"""
        if len(group) < 2:
            return group

        def merge_pins(pin_a: Pin, pin_b: Pin) -> Pin:
            ax, ay, aw, ah = pin_a.bbox
            bx, by, bw, bh = pin_b.bbox
            x1 = min(ax, bx)
            y1 = min(ay, by)
            x2 = max(ax + aw, bx + bw)
            y2 = max(ay + ah, by + bh)
            merged = Pin(
                center=((x1 + x2) / 2, (y1 + y2) / 2),
                bbox=(x1, y1, x2 - x1, y2 - y1),
                area=pin_a.area + pin_b.area,
                angle=pin_a.angle,
                width=x2 - x1,
                height=y2 - y1,
                contour=None,
            )
            return merged

        merged_pins = []
        used = set()

        for i, pin in enumerate(group):
            if i in used:
                continue

            current = pin
            for j in range(i + 1, len(group)):
                if j in used:
                    continue

                other = group[j]
                dx = abs(current.center[0] - other.center[0])
                dy = abs(current.center[1] - other.center[1])
                avg_h = (current.height + other.height) / 2
                gap = dy - avg_h

                if dx < 20 and gap < 50:
                    current = merge_pins(current, other)
                    used.add(j)

            merged_pins.append(current)

        return merged_pins


    # 调整同一排中各引脚的 Y 位置到同一水平线，统一基线
    @staticmethod
    def _align_y(pins: list) -> list:
        if len(pins) < 3:
            return pins
        groups = PinDetector._split_by_y_gap(pins)
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

    # 将检测框高度统一到目标值，方便后续进行统一比较和绘制
    @staticmethod
    def _normalize_bbox_height(pins: list, target_height: int = 150) -> list:
        if not pins:
            return pins
        for p in pins:
            _, cy = p.center
            x, _, bw, _ = p.bbox
            new_y = int(cy - target_height / 2)
            object.__setattr__(p, "bbox", (x, new_y, bw, target_height))
        return pins

    # 根据面积、长宽比、轮廓 solidity 等条件过滤出真正的引脚候选
    def _filter_pins(self, contours: list) -> list:
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

    # 在原图上绘制所有检测到的引脚框和编号，便于人工检查
    @staticmethod
    def draw_pins(image: np.ndarray, result: DetectionResult, color: tuple = (0, 255, 0)) -> np.ndarray:
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pin in result.pins:
            x, y, w, h = pin.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.circle(img, (int(pin.center[0]), int(pin.center[1])), 2, (0, 0, 255), -1)
            cv2.putText(img, str(pin.index), (int(pin.center[0]) - 5, max(y - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return img
