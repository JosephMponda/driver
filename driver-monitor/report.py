#!/usr/bin/env python3
"""Build an incident report (what the driver did wrong) from a monitor / black-box CSV.

  python3 report.py blackbox/live_20261106_143000.csv
      -> text report for the LAST 30 s of the file (no crash in it, e.g. a button-triggered log).
  python3 report.py LOG.csv --plot incident.png   -> also save a graph of that window
  python3 report.py LOG.csv --seconds 60          -> 60 s before the event instead of 30
  python3 report.py LOG.csv --end 1793969540.0    -> the window ending at that timestamp (column t)
If the log contains a CRASH DETECTED alert, the report is centred on the impact: --seconds before it
(default 30) AND --after seconds after it (default 30), so you can see what happened once it hit.
  python3 report.py LOG.csv --after 60            -> 60 s of aftermath
  python3 report.py LOG.csv --from-end            -> ignore the crash and use the end of the file instead

Needs only Python 3. The graph also needs matplotlib:  sudo apt install python3-matplotlib
(Works on any PC, e.g. with the Pi's SD card plugged in: the logs are plain CSV files.)
"""
import argparse
import csv

from blackbox import ALERTS, COLS, _episodes, _f, analyze


def plot(rows, before_s, after_s, out, title, t_ref=None):
    """Graph rows from before_s seconds before t_ref to after_s seconds after it (t_ref default: last row)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        raise SystemExit("Graphs need matplotlib:  sudo apt install python3-matplotlib   (or pip install matplotlib)")

    t_ref = _f(rows[-1][0]) if t_ref is None else t_ref
    rows = [r for r in rows if -before_s <= _f(r[0]) - t_ref <= after_s]
    if not rows:
        raise SystemExit("No data to plot.")
    t_last = _f(rows[-1][0])
    x = [_f(r[0]) - t_ref for r in rows]                      # seconds relative to the event (0 = event)

    def col(i):
        return [v if v is not None else float("nan") for v in (_f(r[i]) for r in rows)]

    panels = [("Eye openness\n(EAR)", [("EAR", 2)]),
              ("PERCLOS\n(eyes-closed share)", [("PERCLOS", 6)]),
              ("Head angle\n(deg from straight)", [("yaw", 4), ("pitch", 5)]),
              ("Alcohol\n(ppm)", [("ppm", 8)])]
    if any(_f(r[9]) is not None for r in rows):
        panels.append(("Distance ahead\n(cm)", [("dist", 9)]))
    if any(_f(r[10]) is not None for r in rows):
        panels.append(("Acceleration\n(peak g)", [("g", 10)]))
    fig, axs = plt.subplots(len(panels), 1, sharex=True, figsize=(10, 9))
    for ax, (ylabel, lines) in zip(axs, panels):
        for name, c in lines:
            ax.plot(x, col(c), label=name, linewidth=1.3)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        ax.axvline(0, color="black", linewidth=1)
        if len(lines) > 1:
            ax.legend(loc="upper left", fontsize=8)

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = []
    for i, (s, e, m, _) in enumerate(_episodes(rows, t_last)):
        color = palette[(i + 3) % len(palette)]
        for ax in axs:
            ax.axvspan(s - t_ref, e - t_ref, color=color, alpha=0.18)
        if m not in [h.get_label() for h in handles]:
            handles.append(Patch(color=color, alpha=0.4, label=m))
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                   ncol=min(len(handles), 3), fontsize=9)
    axs[-1].set_xlabel("seconds relative to the event (0 = event, negative = before, positive = after)")
    fig.suptitle(title, y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=130)
    print(f"Graph saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("--seconds", type=float, default=30.0, help="seconds BEFORE the event (default 30)")
    ap.add_argument("--after", type=float, default=30.0,
                    help="seconds AFTER a detected crash to include (default 30)")
    ap.add_argument("--end", type=float, default=None, help="window ends at this timestamp (default: end of file)")
    ap.add_argument("--plot", default=None, metavar="FILE.png", help="also save a graph of the window")
    ap.add_argument("--from-end", action="store_true",
                    help="use the last seconds of the file even if a crash was detected earlier in it")
    a = ap.parse_args()

    with open(a.csv_file, newline="") as f:
        lines = f.read().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()                                         # half-written last line after a power cut
    rows = [[(d.get(c) or "") for c in COLS] for d in csv.DictReader(lines)]
    rows = [r for r in rows if _f(r[0]) is not None]

    t_event = None
    if a.end is None and not a.from_end:                     # a recorded crash marks the event time
        for r in rows:
            if "CRASH DETECTED" in str(r[ALERTS]):
                t_event = _f(r[0])
                print(f"Crash detected in this log: showing {a.seconds:.0f} s before and "
                      f"{a.after:.0f} s after the impact.\n")
                break

    trigger = f"report from {a.csv_file}"
    if t_event is not None:
        rows = [r for r in rows if t_event - a.seconds <= _f(r[0]) <= t_event + a.after]
        print(analyze(rows, trigger=trigger, window_s=a.seconds, t_event=t_event))
        title = (f"{a.seconds:.0f} s before to {a.after:.0f} s after the impact "
                 f"(0 = impact) - {a.csv_file}")
    else:
        if a.end is not None:
            rows = [r for r in rows if _f(r[0]) <= a.end]
        print(analyze(rows, trigger=trigger, window_s=a.seconds))
        title = f"Last {a.seconds:.0f} s before the end of {a.csv_file}"

    if a.plot:
        if not rows:
            raise SystemExit("No data to plot.")
        if t_event is not None:
            plot(rows, a.seconds, a.after, a.plot, title, t_ref=t_event)
        else:
            plot(rows, a.seconds, 0.0, a.plot, title)


if __name__ == "__main__":
    main()
