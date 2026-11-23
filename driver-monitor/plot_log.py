#!/usr/bin/env python3
"""Plot a driver-monitor log as graphs (for accident investigation).

Run on any Linux PC with the SD card plugged in:
    python3 plot_log.py                          # newest .csv in the current folder
    python3 plot_log.py run_20261112_101500.csv  # a specific log
    python3 plot_log.py /media/USER/rootfs/home/edr2026/driver-monitor   # newest log in a folder
    python3 plot_log.py LOG.csv --last 120       # only the final 120 seconds
    python3 plot_log.py LOG.csv --out report.png # save an image (or .pdf) instead of opening a window

Time axis: seconds before the end of the recording (0 = the last frame that was logged), because the
Pi has no battery-backed clock and its wall-clock time may be wrong after a power loss.
Shading: yellow = WARNING, red = DANGER, grey = no face detected.

Install once (Linux Mint):  sudo apt install python3-matplotlib python3-numpy
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np


def newest_csv(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
        if not files:
            sys.exit(f"No .csv logs found in {path}")
        return max(files, key=os.path.getmtime)
    return path


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("The log file is empty.")

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(name)))
            except (TypeError, ValueError):   # missing column, blank value or a half-written last row
                out.append(np.nan)
        return np.array(out)

    alerts = [(r.get("alerts") or "") for r in rows]
    return {k: col(k) for k in ("t", "face", "ear", "mar", "yaw", "pitch",
                                "perclos", "level", "ppm", "dist_cm", "g")}, alerts


def spans(t, mask):
    """Yield (start, end) times of consecutive True runs in mask."""
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = t[i]
        if not m and start is not None:
            yield start, t[i]
            start = None
    if start is not None:
        yield start, t[-1]


def event_table(t, alerts):
    """Print each alert type with when it first appeared (relative to the end) and how long it was active."""
    tokens = {}
    for i, a in enumerate(alerts):
        for tok in filter(None, (x.strip() for x in a.split("|"))):
            d = tokens.setdefault(tok, {"first": t[i], "n": 0})
            d["n"] += 1
    if not tokens:
        print("No alerts were raised in the plotted period.")
        return
    dt = np.nanmedian(np.diff(t)) if len(t) > 1 else 0.0
    print(f"\n{'ALERT':<28}{'FIRST SEEN (s before end)':>28}{'ACTIVE (s)':>14}")
    for tok, d in sorted(tokens.items(), key=lambda kv: kv[1]["first"]):
        print(f"{tok:<28}{d['first']:>28.1f}{d['n'] * dt:>14.1f}")


def main():
    p = argparse.ArgumentParser(description="Plot a driver-monitor CSV log.")
    p.add_argument("log", nargs="?", default=".", help="CSV file, or a folder (newest .csv is used)")
    p.add_argument("--last", type=float, default=None, help="only plot the final N seconds")
    p.add_argument("--out", default=None, help="save to this image/PDF instead of opening a window")
    a = p.parse_args()

    import matplotlib
    if a.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = newest_csv(a.log)
    d, alerts = load(path)

    t = d["t"] - d["t"][-1]                      # 0 = last logged frame
    keep = np.ones(len(t), bool) if a.last is None else t >= -a.last
    t, alerts = t[keep], [x for x, k in zip(alerts, keep) if k]
    d = {k: v[keep] for k, v in d.items()}
    if len(t) < 2:
        sys.exit("Not enough rows to plot.")

    level = np.nan_to_num(d["level"])            # assumes OK=0, WARNING=1, DANGER=2 (see state.py)
    no_face = d["face"] == 0

    # (title, [(label, values)], y-label); panels with no data (e.g. no ultrasonic/IMU) are skipped
    panels = [
        ("Eye aspect ratio (low = eyes closing)", [("EAR", d["ear"])], "EAR"),
        ("Mouth aspect ratio (high = yawning)", [("MAR", d["mar"])], "MAR"),
        ("PERCLOS (fraction of time eyes closed)", [("PERCLOS", d["perclos"])], "fraction"),
        ("Head direction (relative to straight ahead)", [("yaw", d["yaw"]), ("pitch", d["pitch"])], "degrees"),
        ("Breath alcohol sensor", [("ppm", d["ppm"])], "ppm"),
        ("Ultrasonic distance", [("distance", d["dist_cm"])], "cm"),
        ("Acceleration / crash sensor", [("g", d["g"])], "g"),
        ("Alert level (0 OK, 1 WARNING, 2 DANGER)", [("level", level)], "level"),
    ]
    panels = [pn for pn in panels if any(np.isfinite(v).any() for _, v in pn[1])]

    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(13, 2.0 * len(panels) + 1.2))
    axes = np.atleast_1d(axes)
    for ax, (title, series, ylab) in zip(axes, panels):
        for label, v in series:
            if label == "level":
                ax.step(t, v, where="post", color="black", lw=1.2)
                ax.set_yticks([0, 1, 2])
            else:
                ax.plot(t, v, lw=1.2, label=label)
        if len(series) > 1:
            ax.legend(loc="upper left", fontsize=8)
        for s, e in spans(t, no_face):
            ax.axvspan(s, e, color="grey", alpha=0.25, lw=0)
        for s, e in spans(t, level == 1):
            ax.axvspan(s, e, color="gold", alpha=0.35, lw=0)
        for s, e in spans(t, level >= 2):
            ax.axvspan(s, e, color="red", alpha=0.35, lw=0)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("seconds before end of recording (0 = last frame logged)")
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    fig.suptitle(f"{os.path.basename(path)}  |  {t[-1] - t[0]:.0f} s plotted  |  file last written {mtime} "
                 f"(Pi clock may be inaccurate)", fontsize=10)
    fig.tight_layout()

    event_table(t, alerts)
    if a.out:
        fig.savefig(a.out, dpi=150)
        print(f"Saved {a.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
