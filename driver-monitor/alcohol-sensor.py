#!/usr/bin/env python3
"""Standalone test for the DFRobot SEN0376 alcohol sensor (UART, auto-upload mode).

Run on the Pi:   python test_dfr.py            (default port /dev/serial0)
Other port:      python test_dfr.py --port /dev/ttyAMA0
Log to CSV:      python test_dfr.py --log dfr.csv
Stop:            Ctrl+C  (prints a summary)

What to do:
  1. Start in clean air and watch the ppm settle (it drifts for the first minutes after power-up).
  2. Briefly wave a sanitizer-dampened cotton ball a few cm from the sensor (do not touch it).
  3. ppm should climb within seconds and fall back slowly (recovery can take up to about 60 s).
Needs: pip install pyserial
"""
import argparse
import csv
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing. Run: pip install pyserial")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/serial0")
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument("--log", default=None, help="write time,ppm to this CSV file")
    p.add_argument("--baseline-s", type=float, default=8.0, help="seconds of clean air to average at start")
    p.add_argument("--delta", type=float, default=0.3, help="ppm above baseline shown as ALCOHOL")
    return p.parse_args()


def decode(buf):
    """Pull complete valid frames out of buf. Returns (list_of_ppm, bad_frame_count)."""
    out, bad = [], 0
    while len(buf) >= 9:
        i = buf.find(b"\xff")
        if i < 0:
            buf.clear()
            break
        if i > 0:
            del buf[:i]
        if len(buf) < 9:
            break
        fr = bytes(buf[:9])
        chk = (~sum(fr[1:8]) + 1) & 0xFF
        if chk == fr[8] and fr[1] == 0x17:
            out.append((fr[4] * 256 + fr[5]) / 1000.0)    # ppb -> ppm
            del buf[:9]
        else:
            bad += 1
            del buf[:1]
    return out, bad


def main():
    a = parse()
    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.2)
    except Exception as e:
        sys.exit(f"Cannot open {a.port}: {e}\n"
                 "Check: serial hardware enabled in raspi-config, rebooted, user in 'dialout' group.")
    print(f"Opened {a.port} at {a.baud} baud. Waiting for sensor frames (1 per second)...")

    logf = writer = None
    if a.log:
        logf = open(a.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["t", "ppm"])

    buf = bytearray()
    raw_seen = bytearray()          # first raw bytes, for diagnosis
    ppms, bad_total = [], 0
    baseline, base_vals = None, []
    t0 = time.time()
    last_frame = None
    warned = False
    peak = 0.0

    try:
        while True:
            data = ser.read(64)
            if data:
                if len(raw_seen) < 36:
                    raw_seen += data[:36 - len(raw_seen)]
                buf += data
            vals, bad = decode(buf)
            bad_total += bad
            now = time.time()

            for ppm in vals:
                last_frame = now
                ppms.append(ppm)
                peak = max(peak, ppm)
                if writer:
                    writer.writerow([f"{now:.3f}", f"{ppm:.3f}"])
                    logf.flush()
                if baseline is None:
                    base_vals.append(ppm)
                    print(f"[{now - t0:5.1f}s] baseline sampling: {ppm:.3f} ppm")
                    if now - t0 >= a.baseline_s:
                        baseline = sum(base_vals) / len(base_vals)
                        print(f"\n>>> clean-air baseline = {baseline:.3f} ppm. Now test with alcohol vapour.\n")
                else:
                    rise = ppm - baseline
                    bar = "#" * min(40, int(rise / 0.05)) if rise > 0 else ""
                    flag = "  <-- ALCOHOL" if rise > a.delta else ""
                    print(f"[{now - t0:5.1f}s] {ppm:6.3f} ppm  (rise {rise:+.3f}) {bar}{flag}")

            # No valid frame for a while: print a diagnosis once
            if not warned and (last_frame or t0) + 5 < now:
                warned = True
                if not raw_seen:
                    print("\n!! No bytes at all after 5 s. Check: UART switch on the sensor, sensor D/T -> Pi pin 10 (RXD),\n"
                          "   power (3.3 V pin 1) and GND, serial hardware enabled + reboot, correct port.\n")
                else:
                    print(f"\n!! Bytes arrive but no valid frame. First raw bytes: {raw_seen.hex(' ')}\n"
                          "   Valid data starts with 'ff 17 04'. Garbage usually means wrong baud/serial setup; "
                          "try --port /dev/ttyAMA0 or /dev/ttyS0.\n")
            if last_frame and now - last_frame < 5:
                warned = False
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        if logf:
            logf.close()
        print("\n--- summary ---")
        print(f"valid frames: {len(ppms)}   bad/garbled frames: {bad_total}")
        if ppms:
            print(f"min {min(ppms):.3f}  max {max(ppms):.3f}  last {ppms[-1]:.3f} ppm")
            if baseline is not None:
                print(f"baseline {baseline:.3f} ppm, peak rise {peak - baseline:+.3f} ppm "
                      f"(use something below this peak for --alcohol-ppm)")
        else:
            print("No valid data received: the sensor is not working yet. See the hints above.")


if __name__ == "__main__":
    main()
