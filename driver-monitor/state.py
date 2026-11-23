"""Turns per-frame measurements into driver-state decisions (time-based, so FPS-independent)."""
import time
from collections import deque
from statistics import median

OK, WARNING, DANGER = 0, 1, 2


class Config:
    """Every tunable threshold lives here. Defaults are the PC values (laptop webcam).

    PROFILES below hold per-device overrides, so the same repo works on both machines:
        python monitor.py                 -> profile "pc"
        python monitor.py --profile pi    -> profile "pi"

    HOW TO TUNE: run the program, deliberately do each behaviour (close eyes, yawn, look away),
    watch the numbers on screen (EAR / MAR / yaw / pitch) and nudge the matching value below.
    "Too jumpy" = false alerts -> make the threshold stricter. "Too slow" = misses it -> loosen.
    """

    # --- start-up calibration (both devices) ---------------------------------------------
    calib_seconds = 3.0     # look straight ahead, eyes open, while this runs. Press 'c' to redo it.

    # --- EYES / DROWSINESS ----------------------------------------------------------------
    # Eyes count as closed when EAR < ear_ratio x YOUR open-eye EAR (measured at calibration).
    #   False "eyes closed" while awake (squinting, glare, noisy low-res landmarks) -> LOWER (0.65)
    #   Real closed eyes not detected                                              -> RAISE (0.80)
    #   PC: 0.72 is a good start.   Pi: 320 px frames make landmarks noisier -> profile uses 0.70.
    ear_ratio = 0.80
    ear_floor = 0.14        # safety net: the threshold never goes below this EAR
    # Continuous eye closure that triggers DANGER. A normal blink is ~0.1-0.4 s.
    #   Demo on PC: 1.0 s so the panel sees it quickly.  Pi at ~5-10 FPS: keep >= 1.0 s.
    microsleep_s = 1.5
    # PERCLOS = share of the window with eyes closed. Literature uses 60 s; 30 s is better for demos.
    #   Shorten (15 s) if you want it to react during a short demo.
    perclos_window_s = 30.0
    perclos_limit = 0.15    # 0.15 = more sensitive, 0.35 = less sensitive

    # --- YAWNING / FATIGUE ------------------------------------------------------------------
    # MAR = mouth opening / mouth width. Talking ~0.2-0.4, big yawn ~0.7+.
    #   Check the MAR number on screen while you yawn and while you talk, then put this between them.
    #   The camera angle changes MAR: a camera BELOW the face (dashboard) exaggerates it -> raise it.
    mar_open = 0.60
    yawn_s = 0.6            # mouth must stay open this long to count as a yawn (talking is shorter)
    yawns_for_fatigue = 1.5   # yawns in the window below -> FATIGUE warning. Demo: use 2.
    yawn_window_s = 150.0

    # --- HEAD POSE / DISTRACTION -------------------------------------------------------------
    # Degrees away from the straight-ahead pose captured at calibration (so the camera's own
    # angle is cancelled out; just look at the road while calibrating).
    #   PC: a laptop camera is in front of you -> 25 / 20 works.
    #   Pi: a dashboard camera sits off your line of sight, so ordinary road scanning (mirrors,
    #       side windows) looks like bigger head turns -> profile uses 30 / 25. Widen further if
    #       it complains while you are looking at the "road".
    yaw_limit = 25.0        # left/right
    pitch_limit = 20.0      # up/down (phone use, nodding off)
    distract_s = 5.0        # looking away or face missing this long -> DANGER. Demo: 1.5 s.


# Per-device overrides. Edit these (not Config) once you know what each device needs.
PROFILES = {
    "pc": {},
    "pi": {"ear_ratio": 0.70, "yaw_limit": 30.0, "pitch_limit": 25.0},
}


def make_config(profile="pc"):
    cfg = Config()
    for k, v in PROFILES[profile].items():
        setattr(cfg, k, v)
    return cfg


class DriverState:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.reset()

    def reset(self):
        self.t_start = None
        self.calibrated = False
        self.cal = {"ear": [], "yaw": [], "pitch": []}
        self.ear_thr = 0.2
        self.yaw0 = self.pitch0 = 0.0
        self.closed_since = self.yawn_since = self.away_since = None
        self.perclos_buf = deque()   # (t, closed)
        self.yawn_times = deque()
        self.yawn_counted = False
        self.alerts = []
        self.level = OK
        self.perclos = 0.0

    def update(self, t, face, ear=None, mar=None, pose=None, alcohol=False):
        c = self.cfg
        if self.t_start is None:
            self.t_start = t

        # ---- calibration -------------------------------------------------
        if not self.calibrated:
            if face and pose is not None:
                self.cal["ear"].append(ear)
                self.cal["yaw"].append(pose[0])
                self.cal["pitch"].append(pose[1])
            if t - self.t_start >= c.calib_seconds:
                if len(self.cal["ear"]) >= 5:
                    self.ear_thr = max(c.ear_floor, c.ear_ratio * median(self.cal["ear"]))
                    self.yaw0 = median(self.cal["yaw"])
                    self.pitch0 = median(self.cal["pitch"])
                    self.calibrated = True
                else:   # no face seen: try again
                    self.t_start = t
                    self.cal = {"ear": [], "yaw": [], "pitch": []}
            self.alerts = ["CALIBRATING - look straight ahead"]
            self.level = OK
            return self.level, self.alerts

        alerts, level = [], OK

        # ---- drowsiness: eyes closed (microsleep + PERCLOS) --------------
        closed = bool(face and ear is not None and ear < self.ear_thr)
        self.perclos_buf.append((t, closed))
        while self.perclos_buf and t - self.perclos_buf[0][0] > c.perclos_window_s:
            self.perclos_buf.popleft()
        if len(self.perclos_buf) > 1:
            span = self.perclos_buf[-1][0] - self.perclos_buf[0][0]
            tot = sum(self.perclos_buf[i + 1][0] - self.perclos_buf[i][0]
                      for i in range(len(self.perclos_buf) - 1) if self.perclos_buf[i][1])
            self.perclos = tot / span if span > 0 else 0.0

        if closed:
            self.closed_since = self.closed_since or t
            if t - self.closed_since >= c.microsleep_s:
                alerts.append("DROWSY: EYES CLOSED"); level = max(level, DANGER)
        else:
            self.closed_since = None
        if self.perclos > c.perclos_limit and (t - self.t_start) > c.calib_seconds + 10:
            alerts.append(f"DROWSY: PERCLOS {self.perclos:.0%}"); level = max(level, WARNING)

        # ---- fatigue: yawning -------------------------------------------
        if face and mar is not None and mar > c.mar_open:
            self.yawn_since = self.yawn_since or t
            if t - self.yawn_since >= c.yawn_s:
                if not self.yawn_counted:
                    self.yawn_times.append(t); self.yawn_counted = True
                alerts.append("YAWNING"); level = max(level, WARNING)
        else:
            self.yawn_since = None; self.yawn_counted = False
        while self.yawn_times and t - self.yawn_times[0] > c.yawn_window_s:
            self.yawn_times.popleft()
        if len(self.yawn_times) >= c.yawns_for_fatigue:
            alerts.append(f"FATIGUE: {len(self.yawn_times)} yawns"); level = max(level, WARNING)

        # ---- inattention: looking away or face lost ----------------------
        away = False
        if not face or pose is None:
            away = True
        else:
            dyaw = abs(((pose[0] - self.yaw0) + 180) % 360 - 180)
            dpitch = abs(((pose[1] - self.pitch0) + 180) % 360 - 180)
            away = dyaw > c.yaw_limit or dpitch > c.pitch_limit
        if away:
            self.away_since = self.away_since or t
            if t - self.away_since >= c.distract_s:
                alerts.append("DISTRACTED: EYES OFF ROAD"); level = max(level, DANGER)
        else:
            self.away_since = None

        # ---- intoxication (breath sensor) -------------------------------
        if alcohol:
            alerts.append("ALCOHOL DETECTED"); level = DANGER

        self.alerts, self.level = alerts, level
        return level, alerts
