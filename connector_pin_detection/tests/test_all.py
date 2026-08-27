# encoding: utf-8
"""Extended tests"""
import sys, os, numpy as np, cv2
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path: sys.path.insert(0, _root)
import tempfile, yaml
from src.image_processing.preprocess import Preprocessor
from src.image_processing.pin_detection import PinDetector, Pin
from src.measurement.pitch_calculation import PitchCalculator
from src.measurement.calibration import Calibrator
from src.camera import CameraManager, CameraConfig, CameraSource
from src.classification.defect_classifier import DefectClassifier, DefectType

def img(n=8, s=40):
    im = np.ones((480,640), dtype=np.uint8)*50
    cx,cy = 320,240
    for i in range(n):
        x = cx + (i-n//2)*s
        cv2.rectangle(im,(int(x)-4,int(cy)-10),(int(x)+4,int(cy)+10),240,-1)
    return im

def test_preprocess():
    Preprocessor().process(img()); print("[PASS] preprocess")

def test_pin_detection():
    r = PinDetector({"min_pin_area":20,"solidity_min":0.5}).detect(Preprocessor().process(img(6,50)),None)
    assert r.success and len(r.pins)>=4; print("[PASS] pin_detection")

def test_missing_pin():
    cls = DefectClassifier({"nominal_pitch_mm":2.5,"pitch_tolerance_mm":0.1,"missing_pin_ratio":1.6})
    r = cls.classify([(1,2,50,2.5),(2,3,50,5.0),(3,4,50,2.5)])
    assert r.overall_verdict == DefectType.MISSING_PIN
    assert len(r.missing_pins)==1; print("[PASS] missing_pin")

def test_bent_pin():
    cls = DefectClassifier({"nominal_pitch_mm":2.5,"bent_angle_threshold":8.0})
    pins = [Pin(1,(100,200),(96,190,8,20),80,1,8,20),
            Pin(2,(150,200),(146,190,8,20),80,2,8,20),
            Pin(3,(200,200),(196,190,8,20),80,20,8,20)]
    r = cls.classify([(1,2,50,2.5),(2,3,50,2.5)], pins)
    assert r.overall_verdict == DefectType.BENT_PIN
    assert len(r.bent_pins)==1 and r.bent_pins[0][0]==3; print("[PASS] bent_pin")

def test_classifier():
    cls = DefectClassifier({"nominal_pitch_mm":2.0,"pitch_tolerance_mm":0.05})
    assert cls.classify([(1,2,50,2.01),(2,3,50,1.99)]).overall_verdict == DefectType.OK
    assert cls.classify([(1,2,50,2.20)]).overall_verdict != DefectType.OK; print("[PASS] classifier")

def test_pitch():
    from src.image_processing.pin_detection import DetectionResult
    dr = DetectionResult(); dr.success=True
    dr.pins = [Pin(1,(100,200),(),80,0,8,20),Pin(2,(150,200),(),80,0,8,20),Pin(3,(200,200),(),80,0,8,20)]
    p = PitchCalculator(0.05).calculate_pitches(dr); assert abs(p[0][2]-50)<0.1; print("[PASS] pitch")

def test_cal():
    c = Calibrator(); c.load_scale_from_config(0.05); assert c.is_calibrated; print("[PASS] cal")

def test_pipeline():
    from src.app_controller import AppController
    tp = os.path.join(tempfile.gettempdir(),"_t.png"); cv2.imwrite(tp,img(6,45))
    cp = os.path.join(tempfile.gettempdir(),"_c.yaml")
    with open(cp,"w",encoding="utf-8") as f: yaml.dump({"measurement":{"nominal_pitch_mm":2.5},"classification":{"pitch_tolerance_mm":0.5}},f)
    app = AppController(cp); app.load_image(tp); r = app.run_detection()
    assert r is not None; os.remove(tp); os.remove(cp); print("[PASS] pipeline")

def test_html():
    from src.app_controller import AppController
    tp = os.path.join(tempfile.gettempdir(),"_th.png"); cv2.imwrite(tp,img())
    cp = os.path.join(tempfile.gettempdir(),"_ch.yaml")
    with open(cp,"w",encoding="utf-8") as f: yaml.dump({"measurement":{"nominal_pitch_mm":2.5},"classification":{"pitch_tolerance_mm":0.5}},f)
    app = AppController(cp); app.load_image(tp); app.run_detection()
    hp = os.path.join(tempfile.gettempdir(),"_rh.html"); app.save_report_html(hp)
    with open(hp,encoding="utf-8") as f: assert "html" in f.read()
    os.remove(tp); os.remove(cp); os.remove(hp); print("[PASS] html")

def test_dual_row():
    cls = DefectClassifier({})
    pins = [Pin(i+1,(x,y),(),80,0,8,20) for i,(x,y) in enumerate([(100,100),(150,100),(200,100),(100,200),(150,200),(200,200)])]
    r = cls.classify([], pins)
    assert r.row_info is not None and r.row_info["rows"]==2; print("[PASS] dual_row")

def test_to_dict():
    d = DefectClassifier({"nominal_pitch_mm":2.5}).classify([(1,2,50,2.5)]).to_dict()
    assert "verdict" in d and "stats" in d; print("[PASS] to_dict")

def test_camera_config():
    CameraManager(CameraConfig(source=CameraSource.USB,device_index=0))
    print("[PASS] camera_config")

tests = [test_preprocess,test_pin_detection,test_pitch,test_cal,test_classifier,test_missing_pin,test_bent_pin,test_pipeline,test_html,test_dual_row,test_to_dict,test_camera_config]
for t in tests: t()
print(f"All {len(tests)} tests passed!")
