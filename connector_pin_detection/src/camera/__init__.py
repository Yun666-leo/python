"""摄像机管理模块"""
import cv2
import numpy as np
import os
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
# ---------- MVS (HIKROBOT) 工业相机支持 ----------
try:
    from ctypes import cdll, c_char_p, c_int, c_void_p, byref, create_string_buffer, c_uint32, POINTER, Structure
    import ctypes.util
    _HAVE_MVS = False
    MVS_SDK = None
except Exception:
    _HAVE_MVS = False
    MVS_SDK = None



class CameraSource(Enum):
    USB = "usb"
    RTSP = "rtsp"
    FILE = "file"


@dataclass
class CameraProfile:
    """摄像头预设备"""
    name: str = "默认摄像头"
    device_index: int = 0
    resolution_width: int = 1920
    resolution_height: int = 1080


@dataclass
class CameraConfig:
    source: CameraSource = CameraSource.USB
    device_index: int = 0
    device_profiles: list = field(default_factory=lambda: [
        CameraProfile(name="默认USB摄像头", device_index=0),
        CameraProfile(name="USB摄像头2", device_index=1),
    ])
    rtsp_url: str = ""
    watch_folder: str = ""
    resolution_width: int = 1920
    resolution_height: int = 1080
    fps: int = 30
    exposure: float = -1
    brightness: float = -1
    contrast: float = -1
    auto_capture_interval: float = 0
    save_dir: str = ""
    auto_run_detection: bool = True
    trigger_key: str = "space"


class CameraFrame:
    def __init__(self, image: np.ndarray, timestamp: float, source_info: str):
        self.image = image
        self.timestamp = timestamp
        self.source_info = source_info


class CameraManager:
    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        self._cap = None
        self._is_running = False
        self._thread = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._callback = None
        self._watch_thread = None
        self._last_error = ""
        self._current_device_index = config.device_index if config else 0

    # ---------- 多摄像头支持 ----------

    def list_devices(self, max_index=10) -> list:
        """扫描可用摄像头（USB + MVS 工业相机）"""
        available = []
        # 扫描 USB 摄像头（OpenCV）
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2] if frame is not None else (0, 0)
                    available.append((i, w, h))
                cap.release()
        # 扫描 MVS 工业相机（若 SDK 可用）
        try:
            mvs_cams = self._scan_mvs_devices()
            # MVS 设备用负索引区分（-100 起）
            for j, cam_info in enumerate(mvs_cams):
                available.append((-100 - j, 0, 0))  # 0x0 表示 MVS 相机
        except Exception:
            pass
        return available

    def _scan_mvs_devices(self) -> list:
        """使用 MVS SDK 扫描海康/华睿工业相机，返回 [(serial, ip, model), ...]"""
        if not _HAVE_MVS or MVS_SDK is None:
            return []
        try:
            from ctypes import byref, c_uint32, create_string_buffer
            dev_list = (c_void_p * 32)()
            n_dev = c_uint32(0)
            # MVS SDK 枚举设备
            ret = MVS_SDK.MV_CC_EnumDevices(1, dev_list, byref(n_dev))
            if ret != 0 or n_dev.value == 0:
                return []
            result = []
            for i in range(n_dev.value):
                st_dev = c_void_p()
                ret = MVS_SDK.MV_CC_HandleDeviceSelected(dev_list[i], byref(st_dev))
                # 获取序列号和 IP
                buf = create_string_buffer(256)
                MVS_SDK.MV_CC_GetDeviceInfo(st_dev, 1, buf, 256)  # 1=serial
                serial = buf.value.decode('utf-8', errors='replace') if buf.value else f'mvs_{i}'
                MVS_SDK.MV_CC_GetDeviceInfo(st_dev, 2, buf, 256)  # 2=IP
                ip = buf.value.decode('utf-8', errors='replace') if buf.value else ''
                result.append((serial, ip, ''))
            return result
        except Exception:
            return []

    def switch_device(self, index: int) -> bool:
        """切换到指定摄像头索引"""
        was_opened = self.is_opened
        if was_opened:
            self.close()
        self._current_device_index = index
        self.config.device_index = index
        if was_opened:
            return self.open()
        return True

    @property
    def current_device_index(self) -> int:
        return self._current_device_index

    # ---------- 打开/关闭 ----------

    def open(self) -> bool:
        self.close()
        try:
            if self._current_device_index < 0:
                # MVS 工业相机（通过 MVS SDK 连接）
                return self._open_mvs()
            if self.config.source == CameraSource.USB:
                self._cap = cv2.VideoCapture(self._current_device_index)
            elif self.config.source == CameraSource.RTSP:
                self._cap = cv2.VideoCapture(self.config.rtsp_url)
            else:
                raise ValueError("FILE 模式不需要 open()")

            if not self._cap or not self._cap.isOpened():
                self._last_error = "无法打开摄像头"
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            if self.config.exposure > 0:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
            if self.config.brightness > 0:
                self._cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
            if self.config.contrast > 0:
                self._cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def _open_mvs(self) -> bool:
        """使用 MVS SDK 连接工业相机（需安装 MVS SDK）"""
        if not _HAVE_MVS or MVS_SDK is None:
            self._last_error = "MVS SDK 未安装，无法打开工业相机。请先安装 MVS SDK。"
            return False
        try:
            mvs_index = abs(self._current_device_index) - 100
            from ctypes import byref, c_uint32, c_char_p, c_void_p
            # 枚举设备选择第 mvs_index 个
            dev_list = (c_void_p * 32)()
            n_dev = c_uint32(0)
            ret = MVS_SDK.MV_CC_EnumDevices(1, dev_list, byref(n_dev))
            if ret != 0 or n_dev.value <= mvs_index:
                self._last_error = f"未找到 MVS 相机 (index={mvs_index}/{n_dev.value})"
                return False
            self._cap = c_void_p()
            ret = MVS_SDK.MV_CC_CreateHandle(byref(self._cap), dev_list[mvs_index])
            if ret != 0:
                self._last_error = f"MVS 创建句柄失败: {ret}"
                return False
            ret = MVS_SDK.MV_CC_OpenDevice(self._cap)
            if ret != 0:
                self._last_error = f"MVS 打开设备失败: {ret}"
                MVS_SDK.MV_CC_DestroyHandle(self._cap)
                self._cap = None
                return False
            # MVS 打开成功，_cap 存为句柄（非 None 即可）
            # 通过 self._cap is not None 判断已打开
            return True
        except Exception as e:
            self._last_error = f"MVS 打开异常: {str(e)}"
            return False

    def close(self):
        if hasattr(self, '_cap') and self._cap is not None:
            try:
                if self._current_device_index < 0:
                    # MVS 关闭
                    MVS_SDK.MV_CC_CloseDevice(self._cap)
                    MVS_SDK.MV_CC_DestroyHandle(self._cap)
                else:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_running = False

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ---------- 拍照 ----------

    def capture(self) -> Optional[CameraFrame]:
        """手动拍照：读取一帧并保存到磁盘"""
        if self.config.source == CameraSource.FILE:
            return self._latest_file_frame()
        if not self.is_opened:
            if not self.open():
                return None
        ret, frame = self._cap.read()
        if ret and frame is not None:
            ts = time.time()
            src = f"Camera[{self._current_device_index}]"
            cf = CameraFrame(frame, ts, src)
            self._save_frame(cf)
            return cf
        return None

    def read_frame(self) -> Optional[CameraFrame]:
        """实时预览：读取一帧，不保存到磁盘"""
        if self.config.source == CameraSource.FILE:
            return self._latest_file_frame()
        if not self.is_opened:
            if not self.open():
                return None
        try:
            if self._current_device_index < 0:
                # MVS 相机读帧
                from ctypes import byref, c_uint32, create_string_buffer
                frame_info = c_void_p()
                ret = MVS_SDK.MV_CC_GetImageBuffer(self._cap, byref(frame_info), 1000)
                if ret == 0 and frame_info:
                    # 获取图像数据
                    p_data = c_void_p()
                    n_size = c_uint32(0)
                    MVS_SDK.MV_CC_GetImageData(frame_info, byref(p_data))
                    MVS_SDK.MV_CC_GetImageSize(frame_info, byref(n_size))
                    # 转换为 numpy 并转为 BGR
                    buf = (c_ubyte * n_size.value).from_address(p_data.value)
                    np_data = np.frombuffer(buf, dtype=np.uint8).copy()
                    # 假设为 Bayer8，需要去马赛克
                    img = cv2.cvtColor(np_data.reshape(1024, 1280), cv2.COLOR_BAYER_BG2BGR)
                    MVS_SDK.MV_CC_FreeImageBuffer(self._cap, frame_info)
                    ts = time.time()
                    return CameraFrame(img, ts, f"MVS_Camera[{self._current_device_index}]")
                return None
            ret, frame = self._cap.read()
            if ret and frame is not None:
                ts = time.time()
                src = f"Camera[{self._current_device_index}]"
                return CameraFrame(frame, ts, src)
        except Exception:
            return None

    def _save_frame(self, frame: CameraFrame):
        if not self.config.save_dir:
            return
        os.makedirs(self.config.save_dir, exist_ok=True)
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(frame.timestamp))
        ms = int((frame.timestamp - int(frame.timestamp)) * 1000)
        fname = f"capture_{ts_str}_{ms:03d}.png"
        path = os.path.join(self.config.save_dir, fname)
        cv2.imwrite(path, frame.image)
        frame.source_info += f" -> {path}"

    # ---------- 文件夹监听 ----------

    def watch_folder(self, callback=None):
        folder = self.config.watch_folder
        if not folder or not os.path.isdir(folder):
            self._last_error = f"监听目录不存在: {folder}"
            return False
        self._callback = callback
        self._is_running = True
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        return True

    def _watch_loop(self):
        seen = set()
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        folder = self.config.watch_folder
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in exts:
                seen.add(f)
        while self._is_running:
            try:
                for f in os.listdir(folder):
                    if f in seen:
                        continue
                    ext = os.path.splitext(f)[1].lower()
                    if ext in exts:
                        path = os.path.join(folder, f)
                        img = cv2.imread(path)
                        if img is not None:
                            seen.add(f)
                            ts = os.path.getmtime(path)
                            cf = CameraFrame(img, ts, f"WATCH:{path}")
                            if self._callback:
                                self._callback(cf)
            except Exception:
                pass
            time.sleep(1)

    def _latest_file_frame(self) -> Optional[CameraFrame]:
        folder = self.config.watch_folder
        if not folder or not os.path.isdir(folder):
            return None
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in exts]
        if not files:
            return None
        latest = max(files, key=os.path.getmtime)
        img = cv2.imread(latest)
        if img is not None:
            return CameraFrame(img, os.path.getmtime(latest), f"FILE:{latest}")
        return None

    @property
    def last_error(self) -> str:
        return self._last_error
