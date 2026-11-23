"""Black box for the driver monitor.

Keeps the last 30 s of driver data in memory, writes a continuous log to disk (so the data survives a
power cut in a crash) and builds an INCIDENT REPORT that says what the driver did wrong in the 30 s
before an event. It uses only what the monitor already measures: face present, EAR, MAR, head pose,
PERCLOS, alert level, alcohol ppm, ultrasonic distance, crash-sensor g and the alert messages from state.py.

Crash events (request(..., follow_up=True)) save TWO reports:
  incident_<stamp>.txt/.csv        at once: the 30 s before the impact (safe even if the power dies right away)
  incident_<stamp>_full.txt/.csv   post_s seconds later: the 30 s before + post_s seconds AFTER the impact
The live log keeps recording the whole time, so nothing stops at the moment of the crash.
"""
import csv
import glob
import os
import time
from collections import deque

COLS = ["t", "face", "ear", "mar", "yaw", "pitch", "perclos", "level", "ppm", "dist_cm", "g", "alerts"]
ALERTS = COLS.index("alerts")
LEVELS = {0: "OK", 1: "WARNING", 2: "DANGER"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(x, nd=3):
    return "" if x is None or x == "" else f"{x:.{nd}f}"


def _vals(rows, col):
    return [v for v in (_f(r[col]) for r in rows) if v is not None]


def _episodes(rows, t_end):
    """Each distinct alert message becomes an episode: (start, end, message, still_active_at_end)."""
    active, episodes = {}, []
    for r in rows:
        t = _f(r[0])
        msgs = {m for m in str(r[ALERTS] or "").split("|") if m}
        for m in msgs:
            active.setdefault(m, t)
        for m in [m for m in active if m not in msgs]:
            episodes.append((active.pop(m), t, m, False))
    for m, st in active.items():
        episodes.append((st, t_end, m, True))
    episodes.sort()
    return episodes


def analyze(rows, trigger="manual", window_s=30.0, t_event=None):
    """rows: time-ordered lists in COLS order (numbers or strings).

    t_event None  -> report on the last window_s seconds of rows (T0 = the end of the data).
    t_event given -> rows already hold the whole event (before AND after); T0 = t_event,
                     T- = before the impact, T+ = after the impact.
    """
    rows = [r for r in rows if len(r) >= len(COLS) and _f(r[0]) is not None]
    if not rows:
        return "No data in the log."
    t_end = _f(rows[-1][0])
    if t_event is None:
        rows = [r for r in rows if t_end - _f(r[0]) <= window_s]
        t_ref = t_end
    else:
        t_ref = t_event
    t_start = _f(rows[0][0])
    span = max(t_end - t_start, 1e-6)

    def T(t):
        d = t - t_ref
        if abs(d) < 0.05:
            return "T0"
        return f"T+{d:.1f}s" if d > 0 else f"T-{-d:.1f}s"

    episodes = _episodes(rows, t_end)

    face_lost = 0.0
    for a, b in zip(rows, rows[1:]):
        if str(a[1]) == "0":
            face_lost += _f(b[0]) - _f(a[0])

    yaws = [abs(v) for v in _vals(rows, 4)]
    pitches = [abs(v) for v in _vals(rows, 5)]
    ears, mars = _vals(rows, 2), _vals(rows, 3)
    perclos, levels, ppms, dists = _vals(rows, 6), _vals(rows, 7), _vals(rows, 8), _vals(rows, 9)
    gs = _vals(rows, 10)

    if t_event is None:
        L = [f"Trigger: {trigger}     Window: last {span:.0f} s of data ({len(rows)} samples)", "",
             "WHAT THE DRIVER DID WRONG (time before the event)"]
        ongoing_label = "T0 (still happening)"
    else:
        L = [f"Trigger: {trigger}     Window: {t_event - t_start:.0f} s before + "
             f"{max(t_end - t_event, 0.0):.0f} s after the impact ({len(rows)} samples)", "",
             "WHAT HAPPENED (T0 = impact, T- = before, T+ = after)"]
        ongoing_label = "end of recording"

    if episodes:
        for s, e, m, ongoing in episodes:
            end = ongoing_label if ongoing else T(e)
            L.append(f"  {T(s):>8} -> {end:<21} {e - s:5.1f} s   {m}")
    else:
        L.append("  Nothing was flagged by the monitor in this window.")
    L += ["", "SUMMARY",
          f"  Face not visible (looking away / out of frame): {face_lost:.1f} s ({100 * face_lost / span:.0f}% of window)"]
    if yaws and pitches:
        L.append(f"  Largest head turn from straight ahead: {max(yaws):.0f} deg yaw, {max(pitches):.0f} deg pitch")
    if ears:
        L.append(f"  Lowest eye openness (EAR): {min(ears):.2f}" +
                 (f"    peak PERCLOS: {max(perclos):.0%}" if perclos else ""))
    if mars:
        L.append(f"  Mouth opening peak (MAR): {max(mars):.2f}")
    if ppms:
        L.append(f"  Alcohol sensor peak: {max(ppms):.2f} ppm")
    if dists:
        L.append(f"  Closest object ahead (ultrasonic): {min(dists):.0f} cm")
    if gs:
        L.append(f"  Peak acceleration (crash sensor): {max(gs):.1f} g")
    if levels:
        L.append(f"  Worst alert level: {LEVELS.get(int(max(levels)), int(max(levels)))}")
    L.append(("  State at T0: " if t_event is None else "  State at the end of the recording: ") +
             (str(rows[-1][ALERTS]).replace("|", " + ") or "OK"))

    if t_event is not None:
        post = [r for r in rows if _f(r[0]) > t_event + 0.05]
        L += ["", f"AFTER THE IMPACT (T0 to T+{max(t_end - t_event, 0.0):.0f}s, {len(post)} samples)"]
        if post:
            vis = sum(1 for r in post if str(r[1]) == "1")
            L.append(f"  Face visible: {100 * vis / len(post):.0f}% of the time")
            pg, pd, pl = _vals(post, 10), _vals(post, 9), _vals(post, 7)
            if pg:
                L.append(f"  Peak acceleration after the impact: {max(pg):.1f} g")
            if pd:
                L.append(f"  Closest object ahead: {min(pd):.0f} cm")
            if pl:
                L.append(f"  Worst alert level: {LEVELS.get(int(max(pl)), int(max(pl)))}")
        else:
            L.append("  No data was recorded after the impact (power lost or recording stopped).")
    return "\n".join(L)


class BlackBox:
    def __init__(self, folder, window_s=30.0, max_mb=300, post_s=30.0):
        os.makedirs(folder, exist_ok=True)
        self.folder, self.window_s, self.post_s = folder, window_s, post_s
        self.ring = deque()
        self.pending = None
        self.pending_follow = False
        self.event = None                      # crash being recorded: {"t0", "end", "reason", "stamp", "rows"}
        self._last_trigger = -1e9
        self._last_sync = 0.0
        self._prune(max_mb)
        self.live_path = os.path.join(folder, time.strftime("live_%Y%m%d_%H%M%S.csv"))
        self.f = open(self.live_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(COLS)

    def _prune(self, max_mb):
        """Keep the live logs under max_mb in total (oldest deleted first). Incident files are never deleted."""
        files = sorted(glob.glob(os.path.join(self.folder, "live_*.csv")))
        total = sum(os.path.getsize(p) for p in files)
        while files and total > max_mb * 1e6:
            p = files.pop(0)
            total -= os.path.getsize(p)
            os.remove(p)

    def add(self, t, face, ear, mar, yaw, pitch, perclos, level, ppm, dist, g, msgs):
        row = [f"{t:.3f}", int(face), _fmt(ear, 4), _fmt(mar, 4), _fmt(yaw, 1), _fmt(pitch, 1),
               f"{perclos:.3f}", int(level), _fmt(ppm, 3), _fmt(dist, 1), _fmt(g, 2), "|".join(msgs)]
        self.ring.append((t, row))
        while self.ring and t - self.ring[0][0] > self.window_s:
            self.ring.popleft()
        if self.event is not None:             # a crash is being recorded: keep everything after the impact
            self.event["rows"].append(row)
        self.w.writerow(row)
        self.f.flush()
        if t - self._last_sync >= 1.0:         # force it onto the SD card about once a second
            os.fsync(self.f.fileno())
            self._last_sync = t

    def request(self, reason="manual", follow_up=False):
        """Safe to call from a button callback or a signal handler: only remembers the request.

        follow_up=True (used for crashes) also keeps recording for post_s seconds afterwards and then
        saves a second, full report covering before + after the impact."""
        if self.pending is not None and self.pending_follow and not follow_up:
            return                             # a crash request is already waiting: don't downgrade it
        self.pending_follow = follow_up
        self.pending = reason

    def follow_left(self, now):
        """Seconds of post-crash recording left, or None when no crash is being recorded."""
        return None if self.event is None else max(0.0, self.event["end"] - now)

    def service(self, now):
        """Call once per frame. Saves an incident if one was requested. Returns the report path or None."""
        if self.pending is None:
            return None
        reason, follow = self.pending, self.pending_follow
        self.pending, self.pending_follow = None, False
        if follow and self.event is not None:
            return None                        # already recording this crash; the full report will include it
        if not follow and now - self._last_trigger < 5.0:   # ignore double presses
            return None
        self._last_trigger = now
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.save_incident(reason, stamp)
        if follow and self.post_s > 0:
            self.event = {"t0": now, "end": now + self.post_s, "reason": reason, "stamp": stamp,
                          "rows": [r for _, r in self.ring]}     # the 30 s before the impact (ring includes this frame)
        return path

    def service_followup(self, now):
        """Call once per frame. When the post-crash recording time is over, saves the full report."""
        if self.event is not None and now >= self.event["end"]:
            return self._finish_event()
        return None

    def _finish_event(self):
        ev, self.event = self.event, None
        base = os.path.join(self.folder, f"incident_{ev['stamp']}_full")
        return self._write_report(base, "FULL CRASH REPORT (before + after impact)",
                                  ev["rows"], ev["reason"], ev["t0"])

    def _write_report(self, base, title, rows, reason, t_event=None):
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        text = (f"{title}   " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n" +
                analyze(rows, reason, self.window_s, t_event) + "\n" +
                f"\nData: {os.path.basename(base)}.csv   Full log: {os.path.basename(self.live_path)}\n")
        with open(base + ".txt", "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        print(text)
        return base + ".txt"

    def save_incident(self, reason="manual", stamp=None):
        rows = [r for _, r in self.ring]
        base = os.path.join(self.folder, "incident_" + (stamp or time.strftime("%Y%m%d_%H%M%S")))
        return self._write_report(base, "INCIDENT REPORT", rows, reason)

    def close(self):
        if self.event is not None:             # stopped (Ctrl+C / shutdown) mid-recording: save what we have
            try:
                self._finish_event()
            except Exception:
                self.event = None
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()
        except Exception:
            pass
