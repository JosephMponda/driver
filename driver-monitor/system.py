#!/usr/bin/env python3
"""Driver monitoring prototype: drowsiness, distraction and (sensor-based) intoxication.

PC test :  python monitor.py
Video   :  python monitor.py --source clip.mp4
Pi      :  python monitor.py --profile pi --picamera --width 320 --sensor dfr --buzzer 17
           add --headless over SSH / without a monitor, --rotate 180 if the image is upside down
           add --log run.csv to record every frame (yaw/pitch are relative to your calibrated
           straight-ahead position; ppm is the alcohol sensor reading)
Keys    :  q quit | c recalibrate | a toggle simulated alcohol (simulated sensor only)
"""
import argparse
import csv
import signal
import time

import cv2
import numpy as np
import mediapipe as mp

import features as F
from alerts import Alerter
from sensors import SimulatedAlcoholSensor
from camera import Camera
from state import DriverState, OK, WARNING, DANGER, make_config, PROFILES

COLORS = {OK: (0, 200, 0), WARNING: (0, 200, 255), DANGER: (0, 0, 255)}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="camera index or video file")
    p.add_argument("--picamera", action="store_true", help="use the Raspberry Pi CSI camera (via rpicam-vid)")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="rotate the image (e.g. 180 if the CSI camera is mounted upside down)")
    p.add_argument("--profile", choices=list(PROFILES), default="pc", help="threshold set in state.py")
    p.add_argument("--alcohol-ppm", type=float, default=0.3,
                   help="DFRobot sensor: rise above clean-air baseline (ppm) that counts as alcohol")
    p.add_argument("--width", type=int, default=480, help="processing width (use 320 on a Pi 3B)")
    p.add_argument("--sensor", choices=["sim", "dfr"], default="sim",
                   help="sim = simulated (PC), dfr = DFRobot SEN0376 alcohol sensor on the Pi UART")
    p.add_argument("--buzzer", type=int, default=None, help="GPIO pin of an active buzzer")
    p.add_argument("--headless", action="store_true", help="no window (SSH / no monitor)")
    p.add_argument("--log", default=None, help="write a CSV log of every frame")
    return p.parse_args()


def _stop(*_):
    """SIGTERM (systemctl stop / shutdown) -> same clean exit path as Ctrl+C."""
    raise KeyboardInterrupt


def main():
    a = parse()
    signal.signal(signal.SIGTERM, _stop)
    cap = Camera(a.source, a.width, a.picamera)
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    if a.sensor == "dfr":
        from sensors import DFRobotAlcoholSensor
        sensor = DFRobotAlcoholSensor(delta_ppm=a.alcohol_ppm)
    else:
        sensor = SimulatedAlcoholSensor()
    print(f"[init] alcohol sensor: {sensor.name} (calibrating in clean air...)")
    sensor.calibrate()

    mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    state, alerter = DriverState(make_config(a.profile)), Alerter(a.buzzer)
    print(f"[init] threshold profile: {a.profile}")

    logf = writer = None
    if a.log:
        logf = open(a.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["t", "face", "ear", "mar", "yaw", "pitch", "perclos", "level", "ppm", "alerts"])

    fps, t_prev, last_print = 0.0, time.time(), ""
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if a.rotate:
                frame = cv2.rotate(frame, rot[a.rotate])
            h0, w0 = frame.shape[:2]
            frame = cv2.resize(frame, (a.width, int(h0 * a.width / w0)))
            h, w = frame.shape[:2]

            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = bool(res.multi_face_landmarks)
            ear = mar = pose = None
            if face:
                lm = res.multi_face_landmarks[0].landmark
                pts = {i: (lm[i].x * w, lm[i].y * h) for i in range(len(lm))}
                pts = {k: np.array(v) for k, v in pts.items()}
                ear = (F.eye_aspect_ratio(pts, F.R_EYE) + F.eye_aspect_ratio(pts, F.L_EYE)) / 2
                mar = F.mouth_aspect_ratio(pts)
                pose = F.head_pose(pts, w, h)

            now = time.time()
            ppm, alcohol = sensor.read()
            level, msgs = state.update(now, face, ear, mar, pose, alcohol)
            alerter.update(level)

            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            if writer:
                # yaw/pitch logged relative to the calibrated straight-ahead position
                rel = ((pose[0] - (state.yaw0 or 0), pose[1] - (state.pitch0 or 0))
                       if pose else ("", ""))
                writer.writerow([f"{now:.3f}", int(face), ear, mar, rel[0], rel[1],
                                 f"{state.perclos:.3f}", level, f"{ppm:.3f}", "|".join(msgs)])

            text = " / ".join(msgs) if msgs else "OK"
            if a.headless:
                if text != last_print:
                    print(f"[{time.strftime('%H:%M:%S')}] {text}")
                    last_print = text
                continue

            if face:
                for i in F.R_EYE + F.L_EYE + F.MOUTH:
                    cv2.circle(frame, tuple(int(v) for v in pts[i]), 1, (255, 255, 0), -1)
            col = COLORS[level]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 4 if level else 1)
            cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            info = (f"EAR {ear:.2f} (thr {state.ear_thr:.2f}) MAR {mar:.2f} PERCLOS {state.perclos:.0%}"
                    if face else "NO FACE")
            cv2.putText(frame, info, (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            alc = f"{ppm:.2f} ppm" if a.sensor == "dfr" else "sim"
            sub = f"FPS {fps:.1f}  alcohol: {alc}"
            if pose:
                sub += f"  yaw {pose[0]-state.yaw0:+.0f} pitch {pose[1]-state.pitch0:+.0f}"
            cv2.putText(frame, sub, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Driver Monitor", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                state.reset()
            if k == ord("a") and isinstance(sensor, SimulatedAlcoholSensor):
                sensor.toggle()
    except KeyboardInterrupt:
        pass
    finally:
        alerter.close()
        cap.release()
        if logf:
            logf.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:   # no display (headless / service)
            pass


if __name__ == "__main__":
    main()
