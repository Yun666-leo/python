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
    def process_for_pins(self, image: np.ndarray) -> np.ndarray:
        """引脚检测专用预处理管线：垂直 std 剖面定位行 → crop 顶部 90px → 各行纯 Otsu
        → 垂直闭合(3,9) 连接同列断裂 → OPEN(3,3)×2 → CLOSE(3,3)×1。"""
        denoised =self.image_handle(image)
        row_bands, _ = self._find_pin_rows(denoised)
        result=self.process_for_pins_with_bands(image,row_bands)
        return result

    # 计算图像中的引脚行带及其标准差曲线。
    def _compute_row_bands(self, image: np.ndarray) -> tuple:
        """从图像计算行带和平滑标准差（std_smooth）曲线。
返回值 (bands, std_smooth)，其中 std_smooth 是用于 _validate_bands 的峰值标准差过滤的平滑标准差曲线。
        """
        denoised =self.image_handle(image)
        bands, std_smooth = self._find_pin_rows(denoised)
        return bands, std_smooth

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
        """处理图像以检测引脚，只使用指定的行带。"""
        denoised =self.image_handle(image)
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

    # 在未检测到行带时使用备用二值化流程。
    def _fallback(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned

    # 根据垂直标准差曲线定位引脚所在的行带。
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
        return [(max(0, s - margin), min(h, e + margin)) for s, e, _ in regions],std_smooth
    
    # 对图像进行灰度化、暗图增强和中值去噪。
    def image_handle(self,image: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(image)
        # 自适应：暗图（mean < 60）用 CLAHE 增强对比度以改善引脚检测
        if gray.mean() < 60:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        strength = self.config.get("denoise_strength", 3)
        if strength % 2 == 0:
            strength += 1
        denoised = cv2.medianBlur(gray, strength)
        return denoised

    # 将彩色图像转换为灰度图像。
    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
   
    # 使用中值滤波和高斯滤波去除图像噪声。
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        strength = self.config.get("denoise_strength", 3)
        if strength % 2 == 0:
            strength += 1
        median = cv2.medianBlur(image, strength)
        gaussian = cv2.GaussianBlur(median, (strength, strength), 0)
        return gaussian

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
   
