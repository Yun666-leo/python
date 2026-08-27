"""ROI提取模块"""
import cv2
import numpy as np


class ROIExtractor:
    """ROI提取器"""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def extract(self, binary: np.ndarray, original: np.ndarray) -> tuple:
        """提取连接器主体ROI，返回(roi, mask, rect)"""
        if len(binary.shape) == 3:
            gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary

        # Otsu 二值化，自动选择正确极性
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 如果白像素占比 > 50%，说明 Otsu 选了背景（亮主体时=背景），反转
        white_ratio = np.sum(thresh > 0) / thresh.size
        if white_ratio < 0.5:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        pad = self.config.get("roi_padding", 20)
        H, W = binary.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        roi = original[y1:y2, x1:x2]
        mask = thresh[y1:y2, x1:x2]
        return roi, mask, (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def correct_orientation(contour: np.ndarray) -> float:
        """计算连接器主方向角"""
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if angle < -45:
            angle += 90
        return angle
