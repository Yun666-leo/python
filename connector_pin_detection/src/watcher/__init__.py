"""文件夹监听 + 拍照触发检测模块"""
import os
import time
import threading
import cv2
import numpy as np


class HotFolderWatcher:
    """热文件夹监听器：自动检测新图片并触发回调"""

    def __init__(self, folder: str, callback, interval: float = 0.5):
        self.folder = folder
        self.callback = callback
        self.interval = interval
        self._seen = set()
        self._running = False
        self._thread = None
        self._exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def start(self):
        self._running = True
        # 初始扫描
        os.makedirs(self.folder, exist_ok=True)
        for f in os.listdir(self.folder):
            if os.path.splitext(f)[1].lower() in self._exts:
                self._seen.add(f)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def _loop(self):
        while self._running:
            try:
                for fname in os.listdir(self.folder):
                    if fname in self._seen:
                        continue
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in self._exts:
                        continue
                    path = os.path.join(self.folder, fname)
                    img = cv2.imread(path)
                    if img is not None:
                        self._seen.add(fname)
                        if self.callback:
                            self.callback(img, fname)
            except Exception:
                pass
            time.sleep(self.interval)
