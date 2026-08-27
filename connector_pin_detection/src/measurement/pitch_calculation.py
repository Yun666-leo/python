"""间距测量模块"""
import numpy as np
from ..image_processing.pin_detection import DetectionResult, Pin


class PitchCalculator:
    """引脚间距计算器"""

    def __init__(self, pixel_per_mm: float = None):
        self.pixel_per_mm = pixel_per_mm

    def calculate_pitches(self, result: DetectionResult) -> list:
        """计算相邻引脚间距，返回[(pin_i, pin_j, distance_px, distance_mm)]"""
        pitches = []
        pins = result.pins
        for i in range(len(pins) - 1):
            p1 = pins[i]
            p2 = pins[i + 1]
            dist_px = np.sqrt((p2.center[0] - p1.center[0])**2 + (p2.center[1] - p1.center[1])**2)
            dist_mm = self._to_mm(dist_px)
            pitches.append((p1.index, p2.index, round(dist_px, 2), round(dist_mm, 4)))
        return pitches

    def calculate_all_distances(self, result: DetectionResult) -> np.ndarray:
        """计算所有引脚之间的两两距离矩阵"""
        n = len(result.pins)
        if n == 0:
            return np.array([])
        centers = np.array([p.center for p in result.pins])
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d
        return dist_matrix

    def _to_mm(self, pixels: float) -> float:
        if self.pixel_per_mm is None:
            return pixels
        return round(pixels * self.pixel_per_mm, 4)
