"""预处理模块 - 图像去噪、增强、二值化"""
import cv2
import numpy as np


class Preprocessor:
    """图像预处理器"""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def process(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(image)
        denoised = self._denoise(gray)
        enhanced = self._enhance_contrast(denoised)
        binary = self._binarize(enhanced, invert=True)
        cleaned = self._morphological_clean(binary)
        return cleaned

    def process_for_pins(self, image: np.ndarray) -> np.ndarray:
        """引脚检测专用预处理管线：垂直 std 剖面定位行 → crop 顶部 90px → 各行纯 Otsu
        → 垂直闭合(3,9) 连接同列断裂 → OPEN(3,3)×2 → CLOSE(3,3)×1。"""
        gray = self._to_grayscale(image)
        # 自适应：暗图（mean < 60）用 CLAHE 增强对比度以改善引脚检测
        if gray.mean() < 60:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        strength = self.config.get("denoise_strength", 3)
        if strength % 2 == 0:
            strength += 1
        denoised = cv2.medianBlur(gray, strength)

        row_bands = self._find_pin_rows(denoised)
        if len(row_bands) == 0:
            return self._fallback(denoised)

        k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = np.zeros_like(denoised)
        for y1, y2 in row_bands:
            # Crop 到顶部 90px，避免引脚与外壳连接
            crop_end = min(y1 + 70, y2)
            row_band = denoised[y1:crop_end, :]
            _, binary = cv2.threshold(row_band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            closed_vert = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k9, iterations=1)
            opened = cv2.morphologyEx(closed_vert, cv2.MORPH_OPEN, k3, iterations=2)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k3, iterations=1)
            # 膨胀以统一暗光/亮光下的引脚大小
            dilated = cv2.dilate(closed, k3, iterations=2)
            result[y1:crop_end, :] = dilated
        return result

    def _compute_row_bands(self, image: np.ndarray) -> tuple:
        """Compute row bands and std_smooth profile from image.

        Returns (bands, std_smooth) where std_smooth is the smoothed std-dev
        profile used by _validate_bands for peak-std filtering.
        """
        gray = self._to_grayscale(image)
        if gray.mean() < 60:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        strength = self.config.get("denoise_strength", 3)
        if strength % 2 == 0:
            strength += 1
        denoised = cv2.medianBlur(gray, strength)
        bands = self._find_pin_rows(denoised)
        # Compute std_smooth for band validation
        h = denoised.shape[0]
        half_w = 2
        std_profile = np.zeros(h)
        for y in range(half_w, h - half_w):
            band = denoised[y - half_w:y + half_w + 1, :]
            std_profile[y] = np.std(band)
        blur_ksize = min(15, max(3, h // 20 * 2 + 1))
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        std_smooth = cv2.GaussianBlur(std_profile.reshape(-1, 1), (1, blur_ksize), 5).ravel()
        return bands, std_smooth

    @staticmethod
    def _filter_pins_from_binary(binary, config):
        """Filter pin contours from a binary image using PinDetector config."""
        from src.image_processing.pin_detection import PinDetector
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        det = PinDetector(config)
        return det._filter_pins(contours)

    def process_for_pins_with_bands(self, image: np.ndarray, bands: list) -> np.ndarray:
        """Process image for pin detection, using only the specified bands."""
        gray = self._to_grayscale(image)
        if gray.mean() < 60:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        strength = self.config.get("denoise_strength", 3)
        if strength % 2 == 0:
            strength += 1
        denoised = cv2.medianBlur(gray, strength)

        if len(bands) == 0:
            return self._fallback(denoised)

        k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = np.zeros_like(denoised)
        for y1, y2 in bands:
            crop_end = min(y1 + 70, y2)
            row_band = denoised[y1:crop_end, :]
            _, binary = cv2.threshold(row_band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            closed_vert = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k9, iterations=1)
            opened = cv2.morphologyEx(closed_vert, cv2.MORPH_OPEN, k3, iterations=2)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k3, iterations=1)
            dilated = cv2.dilate(closed, k3, iterations=2)
            result[y1:crop_end, :] = dilated
        return result
    def _fallback(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned

    def _find_pin_rows(self, image: np.ndarray, max_rows: int = 2) -> list:
        h, w = image.shape
        if h < 30:
            return []
        half_w = 2
        std_profile = np.zeros(h)
        for y in range(half_w, h - half_w):
            band = image[y - half_w : y + half_w + 1, :]
            std_profile[y] = np.std(band)
        blur_ksize = min(15, max(3, h // 20 * 2 + 1))
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        std_smooth = cv2.GaussianBlur(std_profile.reshape(-1, 1), (1, blur_ksize), 5).ravel()
        thr = np.mean(std_smooth) + 0.5 * np.std(std_smooth)
        above = std_smooth > thr
        regions = []
        i = 0
        while i < h:
            if above[i]:
                start = i
                while i < h and above[i]:
                    i += 1
                end = i
                # Only keep regions tall enough to be a real pin row
                # (noise/artifacts like housing are typically <20px tall)
                if end - start >= 20:
                    regions.append((start, end, std_smooth[start + np.argmax(std_smooth[start:end])]))
            else:
                i += 1
        if not regions:
            return []
        regions.sort(key=lambda r: r[2], reverse=True)
        regions = regions[:max_rows]
        regions.sort(key=lambda r: r[0])
        margin = 12
        return [(max(0, s - margin), min(h, e + margin)) for s, e, _ in regions]

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        strength = self.config.get("denoise_strength", 3)
        if strength % 2 == 0:
            strength += 1
        median = cv2.medianBlur(image, strength)
        gaussian = cv2.GaussianBlur(median, (strength, strength), 0)
        return gaussian

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clip_limit = self.config.get("clahe_clip_limit", 2.0)
        tile_size = self.config.get("clahe_tile_size", 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)

    def _binarize(self, image: np.ndarray, invert: bool = True) -> np.ndarray:
        method = self.config.get("binarize_method", "adaptive")
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if method == "adaptive":
            block_size = self.config.get("adaptive_block_size", 11)
            if block_size % 2 == 0:
                block_size += 1
            c_val = self.config.get("adaptive_c", 2)
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, c_val)
        else:
            _, binary = cv2.threshold(image, 0, 255, thresh_type + cv2.THRESH_OTSU)
            return binary

    def _morphological_clean(self, binary: np.ndarray) -> np.ndarray:
        kernel_size = self.config.get("morph_kernel_size", 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed

    def extract_pin_region(self, image: np.ndarray) -> list:
        binary = self.process(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.config.get("min_pin_area", 20)
        max_area = self.config.get("max_pin_area", 5000)
        return [c for c in contours if min_area < cv2.contourArea(c) < max_area]
