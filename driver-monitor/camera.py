"""Frame source: Pi Camera (CSI, via rpicam-vid), USB/laptop webcam, or a video file."""
import subprocess
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, source="0", width=480, picamera=False):
        self.proc = None
        self.cap = None
        if picamera:
            # rpicam-vid works with any Python version (picamera2 only imports on the system Python)
            self.w = max(32, (width // 32) * 32)      # YUV420 output is clean when width is a multiple of 32
            self.h = (self.w * 3 // 4) // 2 * 2       # even height
            self.nbytes = self.w * self.h * 3 // 2
            self.proc = subprocess.Popen(
                ["rpicam-vid", "-t", "0", "--nopreview", "--flush",
                 "--codec", "yuv420", "--width", str(self.w), "--height", str(self.h),
                 "--framerate", "15", "-o", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            time.sleep(2.0)   # let the camera start and auto-exposure settle
            if self.proc.poll() is not None:
                raise SystemExit("rpicam-vid failed to start. Check: rpicam-hello --list-cameras "
                                 "(ribbon cable, camera enabled, no other program using the camera)")
        else:
            src = int(source) if str(source).isdigit() else source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open video source {source!r}")

    def read(self):
        if self.proc is not None:
            buf = b""
            while len(buf) < self.nbytes:
                chunk = self.proc.stdout.read(self.nbytes - len(buf))
                if not chunk:
                    return False, None
                buf += chunk
            yuv = np.frombuffer(buf, np.uint8).reshape(self.h * 3 // 2, self.w)
            return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return self.cap.read()

    def release(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.cap is not None:
            self.cap.release()
