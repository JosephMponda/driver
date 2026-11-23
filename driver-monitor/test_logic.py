"""Offline sanity tests - no camera needed:  python test_logic.py"""
import math
import numpy as np
import cv2
import features as F
from state import DriverState, DANGER, OK


def project(yaw_deg, pitch_deg, w=640, h=480):
    ry, rx = math.radians(yaw_deg), math.radians(pitch_deg)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    pts3d = (Ry @ Rx @ F.MODEL_3D.T).T + np.array([0, 0, 600.0])
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]])
    img = (cam @ pts3d.T).T
    img = img[:, :2] / img[:, 2:3]
    pts = {}
    for idx, p in zip(F.POSE_IDX, img):
        pts[idx] = p
    return pts, w, h


def test_head_pose():
    base = F.head_pose(*project(0, 0))
    right = F.head_pose(*project(30, 0))
    left = F.head_pose(*project(-30, 0))
    down = F.head_pose(*project(0, 25))
    print("frontal", [round(x, 1) for x in base])
    print("yaw +30", [round(x, 1) for x in right], "| yaw -30", [round(x, 1) for x in left])
    print("pitch +25", [round(x, 1) for x in down])
    assert abs(base[0]) < 3 and abs(base[1]) < 3
    assert abs(abs(F.wrap(right[0] - base[0])) - 30) < 4
    assert abs(abs(F.wrap(left[0] - base[0])) - 30) < 4
    assert abs(abs(F.wrap(down[1] - base[1])) - 25) < 4


def run(frames):
    s, t = DriverState(), 1000.0
    out = None
    for dur, kw in frames:
        end = t + dur
        while t < end:
            out = s.update(t, **kw)
            t += 1 / 15
    return out


OPEN = dict(face=True, ear=0.30, mar=0.2, pose=(0, 0, 0), alcohol=False)


def test_state():
    assert run([(4, OPEN), (5, OPEN)])[0] == OK
    lvl, msgs = run([(4, OPEN), (3, {**OPEN, "ear": 0.05})])
    assert lvl == DANGER and any("EYES CLOSED" in m for m in msgs), msgs
    lvl, msgs = run([(4, OPEN), (3, {**OPEN, "pose": (45, 0, 0)})])
    assert lvl == DANGER and any("DISTRACTED" in m for m in msgs), msgs
    lvl, msgs = run([(4, OPEN), (3, {**OPEN, "face": False, "ear": None, "mar": None, "pose": None})])
    assert any("DISTRACTED" in m for m in msgs), msgs
    lvl, msgs = run([(4, OPEN), (2, {**OPEN, "mar": 0.9})])
    assert any("YAWNING" in m for m in msgs), msgs
    lvl, msgs = run([(4, OPEN), (1, {**OPEN, "alcohol": True})])
    assert any("ALCOHOL" in m for m in msgs), msgs


if __name__ == "__main__":
    test_head_pose(); test_state(); print("all tests passed")
