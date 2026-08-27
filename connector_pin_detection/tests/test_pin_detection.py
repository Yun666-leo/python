"""引脚检测单元测试"""
import sys
import os
import numpy as np
import cv2

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.image_processing.preprocess import Preprocessor
from src.image_processing.pin_detection import PinDetector, Pin
from src.measurement.pitch_calculation import PitchCalculator
from src.measurement.calibration import Calibrator
from src.classification.defect_classifier import DefectClassifier, DefectType


def create_test_image(width=640, height=480, num_pins=8, spacing=40):
    """生成模拟引脚图像用于测试：暗色主体 + 亮色金属引脚"""
    img = np.ones((height, width), dtype=np.uint8) * 50
    cx, cy = width // 2, height // 2
    pins = []
    for i in range(num_pins):
        x = cx + (i - num_pins // 2) * spacing
        y = cy
        cv2.rectangle(img, (x - 4, y - 10), (x + 4, y + 10), 240, -1)
        pins.append((x, y))
    return img, pins


def test_preprocess():
    """测试预处理模块"""
    pre = Preprocessor({"denoise_strength": 3, "clahe_clip_limit": 2.0})
    img, _ = create_test_image()
    result = pre.process(img)
    assert result is not None
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    print(f"[PASS] test_preprocess: shape={result.shape}")


def test_pin_detection():
    """测试引脚检测"""
    detector = PinDetector({"min_pin_area": 20, "solidity_min": 0.5})
    img, expected = create_test_image(num_pins=6, spacing=50)

    pre = Preprocessor()
    binary = pre.process(img)
    result = detector.detect(binary, img)

    assert result.success, f"Detection failed: {result.error_msg}"
    assert len(result.pins) >= 4, f"Expected >=4 pins, got {len(result.pins)}"
    print(f"[PASS] test_pin_detection: {len(result.pins)} pins detected")


def test_pitch_calculation():
    """测试间距计算"""
    from src.image_processing.pin_detection import DetectionResult
    calc = PitchCalculator(pixel_per_mm=0.05)
    result = DetectionResult()
    result.pins = [
        Pin(index=1, center=(100, 200), bbox=(96, 190, 8, 20), area=80, angle=0, width=8, height=20),
        Pin(index=2, center=(150, 200), bbox=(146, 190, 8, 20), area=80, angle=0, width=8, height=20),
        Pin(index=3, center=(200, 200), bbox=(196, 190, 8, 20), area=80, angle=0, width=8, height=20),
    ]
    result.success = True

    pitches = calc.calculate_pitches(result)
    assert len(pitches) == 2
    p1 = pitches[0]
    assert abs(p1[2] - 50.0) < 0.1, f"Expected 50px pitch, got {p1[2]}"
    print(f"[PASS] test_pitch_calculation: pitches={pitches}")


def test_calibration():
    """测试标定"""
    cal = Calibrator()
    cal.load_scale_from_config(0.05)
    assert cal.is_calibrated
    mm = cal.pixels_to_mm(100)
    assert abs(mm - 5.0) < 0.001
    print(f"[PASS] test_calibration: 100px = {mm}mm")


def test_classifier():
    """测试缺陷判定"""
    config = {"nominal_pitch_mm": 2.0, "pitch_tolerance_mm": 0.05}
    cls = DefectClassifier(config)

    # 合格
    pitches = [(1, 2, 50, 2.01), (2, 3, 50, 1.99)]
    result = cls.classify(pitches)
    assert result.overall_verdict == DefectType.OK
    print(f"[PASS] test_classifier OK: {result.overall_verdict}")

    # 超差
    pitches2 = [(1, 2, 50, 2.20), (2, 3, 50, 1.80)]
    result2 = cls.classify(pitches2)
    assert result2.overall_verdict != DefectType.OK
    print(f"[PASS] test_classifier defect: {result2.overall_verdict}")


def test_full_pipeline():
    """测试完整流水线"""
    from src.app_controller import AppController
    import tempfile

    # 生成测试图像并保存
    img, _ = create_test_image(num_pins=6, spacing=45)
    tmp_path = os.path.join(tempfile.gettempdir(), "_test_connector_pin.png")
    cv2.imwrite(tmp_path, img)

    # 配置标称间距
    config_data = {
        "image_processing": {"denoise_strength": 3},
        "pin_detection": {"min_pin_area": 10, "solidity_min": 0.5},
        "measurement": {"nominal_pitch_mm": 2.5},
        "classification": {"pitch_tolerance_mm": 0.5}
    }
    import yaml
    cfg_path = os.path.join(tempfile.gettempdir(), "_test_config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    app = AppController(cfg_path)
    app.load_image(tmp_path)
    cls_result = app.run_detection()
    assert cls_result is not None, "Pipeline returned None"
    print(f"[PASS] test_full_pipeline: {cls_result.overall_verdict}")

    # 清理
    os.remove(tmp_path)
    os.remove(cfg_path)


if __name__ == "__main__":
    test_preprocess()
    test_pin_detection()
    test_pitch_calculation()
    test_calibration()
    test_classifier()
    test_full_pipeline()
    print("\n所有测试通过!")
