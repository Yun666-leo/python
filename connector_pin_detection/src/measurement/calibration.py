"""标定模块"""
import cv2
import numpy as np


class Calibrator:
    """相机标定与像素-物理尺寸转换"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._pixel_per_mm = None
        self._scale_loaded = False

    def load_scale_from_image(self, image: np.ndarray, known_length_mm: float, num_pixels: float):
        """从已知长度标定像素当量"""
        if num_pixels > 0:
            self._pixel_per_mm = known_length_mm / num_pixels
            self._scale_loaded = True
        return self._pixel_per_mm

    def load_scale_from_config(self, pixel_per_mm: float):
        """从配置文件加载标定值"""
        self._pixel_per_mm = pixel_per_mm
        self._scale_loaded = True

    def auto_calibrate_from_avg_pitch(self, avg_pixel_pitch: float, known_pitch_mm: float) -> float:
        """从相邻引脚的平均像素间距自动标定 pixel_per_mm"""
        if avg_pixel_pitch > 0:
            self._pixel_per_mm = known_pitch_mm / avg_pixel_pitch
            self._scale_loaded = True
        return self._pixel_per_mm

    def pixels_to_mm(self, pixels: float) -> float:
        """像素值转换为毫米"""
        if self._pixel_per_mm is None:
            return pixels
        return pixels * self._pixel_per_mm

    @property
    def scale(self):
        return self._pixel_per_mm

    @property
    def is_calibrated(self) -> bool:
        return self._scale_loaded
