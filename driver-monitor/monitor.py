#!/usr/bin/env python3
"""Driver monitoring prototype: drowsiness, distraction and (sensor-based) intoxication.

PC test :  python monitor.py --ultrasonic sim
Video   :  python monitor.py --source clip.mp4
Pi      :  python monitor.py --profile pi --picamera --width 320 --sensor dfr --ultrasonic hcsr04 --buzzer 17
           add --headless over SSH / without a monitor, --rotate 180 if the image is upside down
           add --log run.csv to record every frame (yaw/pitch are relative to your calibrated
           straight-ahead position; ppm is the alcohol sensor reading; dist_cm is the ultrasonic reading)
Black box: add --blackbox DIR [--button 27]  -> continuous crash-proof log in DIR (same columns as --log)
           + an incident report of the last 30 s, saved when you press the button, press 'i', or run:
           sudo systemctl kill --kill-whom=main --signal=USR1 driver-monitor
           Read a log later (e.g. on a PC):  python3 report.py DIR/live_XXXX.csv --plot incident.png
Crash    : add --imu mpu6050 [--crash-g 4]  -> an impact above the threshold saves the incident report
           automatically (needs --blackbox). PC testing: --imu sim and press 'x'.
           The monitor keeps running and recording after the impact. --post-crash N (default 30) saves a
           second, full report N seconds later covering the 30 s BEFORE + N s AFTER the impact
           (incident_<stamp>_full.txt). --post-crash 0 turns the follow-up off.
Keys    :  q quit | c recalibrate | i save incident report (needs --blackbox)
           a toggle simulated alcohol (simulated sensor only)
           x simulate a crash (simulated IMU only)
           u toggle simulated "too close" (simulated ultrasonic only)
"""
import argparse
import csv
import os
import signal
import time

import cv2
import numpy as np
import mediapipe as mp

import features as F
from alerts import Alerter
from blackbox import BlackBox
from sensors import SimulatedAlcoholSensor, SimulatedUltrasonicSensor
from camera import Camera
from imu import MPU6050, SimulatedImu
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
    p.add_argument("--ultrasonic", choices=["off", "sim", "hcsr04"], default="off",
                   help="off = not used, sim = simulated (PC), hcsr04 = HC-SR04 on the Pi GPIO")
    p.add_argument("--near-cm", type=float, default=30.0,
                   help="ultrasonic: closer than this (cm) counts as too close")
    p.add_argument("--imu", choices=["off", "sim", "mpu6050"], default="off",
                   help="crash sensor: off, sim = simulated (PC, press x), mpu6050 = MPU-6050 on the Pi I2C bus")
    p.add_argument("--crash-g", type=float, default=4.0,
                   help="impact threshold in g (peak acceleration magnitude) that counts as a crash")
    p.add_argument("--imu-addr", type=lambda s: int(s, 0), default=0x68,
                   help="MPU-6050 I2C address (0x68, or 0x69 if AD0 is high)")
    p.add_argument("--post-crash", type=float, default=30.0,
                   help="seconds to keep recording after a crash before saving the full before+after report "
                        "(0 = off)")
    p.add_argument("--buzzer", type=int, default=None, help="GPIO pin of an active buzzer")
    p.add_argument("--headless", action="store_true", help="no window (SSH / no monitor)")
    p.add_argument("--log", default=None, help="write a CSV log of every frame")
    p.add_argument("--blackbox", default=None, metavar="DIR",
                   help="continuous crash-proof log + incident reports (last 30 s) saved in DIR")
    p.add_argument("--button", type=int, default=None,
                   help="GPIO pin of an incident push-button wired to GND (needs --blackbox)")
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

    if a.ultrasonic == "hcsr04":
        from sensors import UltrasonicSensor
        usensor = UltrasonicSensor(near_cm=a.near_cm)
    elif a.ultrasonic == "sim":
        usensor = SimulatedUltrasonicSensor()
    else:
        usensor = None
    if usensor:
        print(f"[init] ultrasonic sensor: {usensor.name} (too close < {a.near_cm:.0f} cm)")

    if a.imu == "mpu6050":
        imu = MPU6050(crash_g=a.crash_g, addr=a.imu_addr)
    elif a.imu == "sim":
        imu = SimulatedImu()
    else:
        imu = None
    if imu:
        print(f"[init] crash sensor: {imu.name} (crash = impact of {a.crash_g:.1f} g or more)")
        if not a.blackbox:
            print("[init] note: add --blackbox DIR so a detected crash also saves the incident report")
        elif a.post_crash > 0:
            print(f"[init] after a crash the monitor keeps recording for {a.post_crash:.0f} s, "
                  f"then saves a full before+after report")

    mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    state, alerter = DriverState(make_config(a.profile)), Alerter(a.buzzer)
    print(f"[init] threshold profile: {a.profile}")

    logf = writer = None
    if a.log:
        # one new file per run (a reboot after a crash must never overwrite the previous log)
        base, ext = os.path.splitext(a.log)
        log_path = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.csv'}"
        logf = open(log_path, "w", newline="")
        print(f"[init] logging to {log_path}")
        writer = csv.writer(logf)
        writer.writerow(["t", "face", "ear", "mar", "yaw", "pitch", "perclos", "level", "ppm", "dist_cm", "g", "alerts"])

    bb = button = None
    if a.blackbox:
        bb = BlackBox(a.blackbox, post_s=a.post_crash)
        print(f"[blackbox] logging to {bb.live_path}; incident reports go in {a.blackbox}")
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, lambda *_: bb.request("signal USR1"))
        if a.button is not None:
            try:
                from gpiozero import Button
                button = Button(a.button, pull_up=True)
                button.when_pressed = lambda: bb.request("button")
            except Exception as e:
                print(f"[blackbox] button unavailable: {e}")

    fps, t_prev, last_print, last_sync = 0.0, time.time(), "", 0.0
    crash_until, imu_fault = 0.0, False
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

            # ultrasonic: (distance_cm, too_close); too_close only after it holds for hold_s
            dist, too_close = usensor.read() if usensor else (None, False)
            if too_close:
                msgs.append("TOO CLOSE")
                level = max(level, WARNING)

            # crash sensor: peak g since the last frame. An impact saves the incident report at once and
            # (follow_up) keeps recording for --post-crash seconds, then saves a full before+after report.
            g, crash = imu.read() if imu else (None, None)
            if crash is not None:
                crash_until = now + 10.0
                print(f"[crash] impact detected: {crash:.1f} g")
                if bb:
                    bb.request(f"CRASH: {crash:.1f} g impact", follow_up=True)
            if now < crash_until:
                msgs.append("CRASH DETECTED")
                level = DANGER
            if imu and imu.fault != imu_fault:
                imu_fault = imu.fault
                print("[imu] no data from the crash sensor, check the wiring!" if imu_fault
                      else "[imu] crash sensor data is back")
            alerter.update(level)

            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            # yaw/pitch logged relative to the calibrated straight-ahead position
            rel = ((pose[0] - (state.yaw0 or 0), pose[1] - (state.pitch0 or 0))
                   if pose else ("", ""))

            left = None
            if bb:
                bb.add(now, face, ear, mar, rel[0], rel[1], state.perclos, level, ppm, dist, g, msgs)
                saved = bb.service(now)
                if saved:
                    print(f"[blackbox] incident saved: {saved}")
                    if bb.event is not None:
                        print(f"[blackbox] still recording: full before+after report in {a.post_crash:.0f} s")
                saved = bb.service_followup(now)
                if saved:
                    print(f"[blackbox] full crash report saved: {saved}")
                left = bb.follow_left(now)

            if writer:
                writer.writerow([f"{now:.3f}", int(face), ear, mar, rel[0], rel[1],
                                 f"{state.perclos:.3f}", level, f"{ppm:.3f}",
                                 "" if dist is None else f"{dist:.1f}",
                                 "" if g is None else f"{g:.2f}", "|".join(msgs)])
                # push the row to the SD card now: after a crash the power can vanish at any moment
                logf.flush()
                if now - last_sync >= 1.0:
                    os.fsync(logf.fileno())
                    last_sync = now

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
            if left is not None:
                cv2.putText(frame, f"RECORDING AFTER CRASH: {left:.0f}s", (8, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            info = (f"EAR {ear:.2f} (thr {state.ear_thr:.2f}) MAR {mar:.2f} PERCLOS {state.perclos:.0%}"
                    if face else "NO FACE")
            cv2.putText(frame, info, (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            alc = f"{ppm:.2f} ppm" if a.sensor == "dfr" else "sim"
            sub = f"FPS {fps:.1f}  alcohol: {alc}"
            if dist is not None:
                sub += f"  dist {dist:.0f}cm"
            if g is not None:
                sub += f"  {g:.1f}g"
            if pose:
                sub += f"  yaw {pose[0]-state.yaw0:+.0f} pitch {pose[1]-state.pitch0:+.0f}"
            cv2.putText(frame, sub, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Driver Monitor", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                state.reset()
            if k == ord("i") and bb:
                bb.request("key i")
            if k == ord("a") and isinstance(sensor, SimulatedAlcoholSensor):
                sensor.toggle()
            if k == ord("u") and isinstance(usensor, SimulatedUltrasonicSensor):
                usensor.toggle()
            if k == ord("x") and isinstance(imu, SimulatedImu):
                imu.trigger()
    except KeyboardInterrupt:
        pass
    finally:
        alerter.close()
        cap.release()
        if usensor and hasattr(usensor, "close"):
            usensor.close()
        if imu:
            imu.close()
        if logf:
            logf.close()
        if bb:
            bb.close()      # also saves the full crash report so far if you quit mid-recording
        if button:
            button.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:   # no display (headless / service)
            pass


if __name__ == "__main__":
    main()
