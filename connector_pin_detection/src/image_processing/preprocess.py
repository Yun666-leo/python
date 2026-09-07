"""预处理模块 - 图像去噪、增强、二值化"""
import cv2
import numpy as np


class Preprocessor:
    """图像预处理器"""

    # 初始化图像预处理配置。
    def __init__(self, config: dict = None):
        self.config = config or {}

    # 执行通用的图像预处理流程。
    def process(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(image)
        denoised = self._denoise(gray)
        enhanced = self._enhance_contrast(denoised)
        binary = self._binarize(enhanced, invert=True)
        cleaned = self._morphological_clean(binary)
        return cleaned

    # 执行引脚检测专用的图像预处理流程。
    def process_for_pins(self, image: np.ndarray, bands: list = None) -> tuple:
        """先生成干净二值图，再按逐行白像素数量定位并细化引脚行带。"""
        binary = self._prepare_pin_binary(image)
        if bands is None:
            bands, row_white = self._find_pin_rows(binary)
        else:
            row_white = None
        binary = self.process_for_pins_with_bands(binary, bands)
        return binary, bands, row_white

    # 根据配置过滤二值图像中的引脚轮廓。
    @staticmethod
    def _filter_pins_from_binary(binary, config):
        """使用 PinDetector 配置从二值图像中过滤引脚轮廓。"""
        from src.image_processing.pin_detection import PinDetector
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        det = PinDetector(config)
        return det._filter_pins(contours)

    # 使用指定的行带处理图像并提取引脚区域。
    def process_for_pins_with_bands(self, image: np.ndarray, bands: list) -> np.ndarray:
        """只保留指定行带，并通过局部开闭运算和连通域面积去除噪点。"""
        binary = self._as_binary(image)
        if len(bands) == 0:
            return self._clean_binary(binary)

        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        result = np.zeros_like(binary)
        min_area = max(1, int(self.config.get("pin_component_min_area", 20)))
        for y1, y2 in bands:
            start = max(0, int(y1))
            end = min(binary.shape[0], int(y2) - 70)
            min_process_height = int(self.config.get("pin_min_process_height", 20))
            if end - start < min_process_height:
                # 扣除 70 像素后过短时，保留完整短 band，避免只剩边缘碎片。
                end = min(binary.shape[0], int(y2))
            if end <= start:
                continue
            row_band = binary[start:end, :]
            closed = cv2.morphologyEx(row_band, cv2.MORPH_CLOSE, k_vertical, iterations=1)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k3, iterations=1)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
            cleaned = np.zeros_like(opened)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == label] = 255
            result[start:end, :] = cleaned
        return result

    # 按示例备份中的流程生成引脚检测专用二值图。
    def _prepare_pin_binary(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(image)
        if gray.mean() < 60:
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
            gray = clahe.apply(gray)
            gray = cv2.medianBlur(gray, 5)
            kernel = np.ones((3, 3), dtype=np.uint8)
            gray = cv2.dilate(gray, kernel, iterations=1)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = np.ones((2, 2), dtype=np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _as_binary(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8 and len(np.unique(image)) <= 2:
            return np.where(image > 0, 255, 0).astype(np.uint8)
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _clean_binary(binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 根据每行白像素数量定位引脚所在的行带。
    def _find_pin_rows(self, image: np.ndarray, max_rows: int = 2) -> list:
        binary = self._as_binary(image)
        h, w = binary.shape
        if h < 30 or w == 0:
            return [], np.zeros(h, dtype=np.int32)

        row_white = np.count_nonzero(binary == 255, axis=1)
        ratio = float(self.config.get("pin_row_white_ratio", 0.05))
        minimum = int(self.config.get("pin_row_white_min", 20))
        threshold = max(minimum, int(round(w * ratio)))
        above = row_white >= threshold
        regions = []
        i = 0
        while i < h:
            if above[i]:
                start = i
                while i < h and above[i]:
                    i += 1
                end = i
               
                min_height = int(self.config.get("pin_row_min_height", 8))
                if end - start >= min_height:
                    regions.append((start, end, int(row_white[start:end].max())))
            else:
                i += 1
        if not regions:
            return []
        regions.sort(key=lambda r: (r[2], r[1] - r[0]), reverse=True)
        regions = regions[:max_rows]
        regions.sort(key=lambda r: r[0])
        margin = int(self.config.get("pin_row_margin", 12))
        bands = [(max(0, s - margin), min(h, e + margin)) for s, e, _ in regions]
        merged = []
        for start, end in bands:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        bands = merged[:max_rows]
        return bands, row_white
    # 将彩色图像转换为灰度图像。
    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
   
    # 使用中值滤波去除图像噪声。
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        strength = self.config.get("denoise_strength", 3)
        if strength % 2 == 0:
            strength += 1
        median = cv2.medianBlur(image, strength)
        return median

    # 使用 CLAHE 方法增强图像局部对比度。
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clip_limit = self.config.get("clahe_clip_limit", 2.0)
        tile_size = self.config.get("clahe_tile_size", 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)

    # 按配置将灰度图像转换为二值图像。
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

            # 通过形态学开闭运算清理二值图像。
    def _morphological_clean(self, binary: np.ndarray) -> np.ndarray:
        kernel_size = self.config.get("morph_kernel_size", 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed

    # 从预处理结果中提取符合面积范围的引脚轮廓。
    def extract_pin_region(self, image: np.ndarray) -> list:
        binary = self.process(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.config.get("min_pin_area", 20)
        max_area = self.config.get("max_pin_area", 5000)
        return [c for c in contours if min_area < cv2.contourArea(c) < max_area]
   
