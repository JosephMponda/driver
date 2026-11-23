"""Geometry helpers: eye/mouth aspect ratios and head pose from MediaPipe Face Mesh landmarks."""
import math
import numpy as np
import cv2

# MediaPipe Face Mesh landmark indices
R_EYE = [33, 160, 158, 133, 153, 144]    # subject's right eye: corner, top, top, corner, bottom, bottom
L_EYE = [362, 385, 387, 263, 373, 380]   # subject's left eye
MOUTH = [78, 308, 13, 14, 81, 178, 311, 402]  # corners, inner-lip centre, inner-lip sides
POSE_IDX = [1, 152, 33, 263, 61, 291]    # nose tip, chin, eye outer corners, mouth corners

# Generic 3D face model (mm, camera-like axes: x right, y down, z away from camera)
MODEL_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, 75.0, 25.0),      # chin
    (-45.0, -35.0, 40.0),   # right eye outer corner (image left)
    (45.0, -35.0, 40.0),    # left eye outer corner (image right)
    (-25.0, 40.0, 25.0),    # mouth right corner
    (25.0, 40.0, 25.0),     # mouth left corner
], dtype=np.float64)


def _d(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * _d(p1, p4) + 1e-6)


def mouth_aspect_ratio(pts):
    c1, c2, up, lo, u1, l1, u2, l2 = (pts[i] for i in MOUTH)
    return (_d(u1, l1) + _d(up, lo) + _d(u2, l2)) / (3.0 * _d(c1, c2) + 1e-6)


def head_pose(pts, w, h):
    """Return (yaw, pitch, roll) in degrees, relative to a frontal face. None if solve fails."""
    img = np.array([pts[i] for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(R[0, 0], R[1, 0])
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def wrap(a):
    """Wrap an angle difference into [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0
