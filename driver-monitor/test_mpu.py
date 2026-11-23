#!/usr/bin/env python3
"""Standalone test for the MPU-6050 crash sensor (uses the same imu.py class as monitor.py).

Run on the Pi:   python test_mpu.py                 (default impact threshold 4 g)
                 python test_mpu.py --crash-g 3     (try another threshold)
Stop:            Ctrl+C  (prints a summary)

What to do:
  1. Keep the board still for the first 3 s: it should read about 1.0 g (gravity), in any orientation.
  2. Tap the table / the board's mount firmly, then shake it. Watch the peak g in each line.
  3. Pick --crash-g above the values you see from normal handling and below a real hit.
Do NOT drop or smash the Pi: a firm tap or shake is enough to test it.
"""
import argparse
import time

from imu import MPU6050


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash-g", type=float, default=4.0, help="impact threshold in g (default 4)")
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x68, help="I2C address (0x68, or 0x69 if AD0 is high)")
    a = ap.parse_args()

    imu = MPU6050(crash_g=a.crash_g, bus=a.bus, addr=a.addr)
    print(f"MPU-6050 found. Impact threshold {a.crash_g:.1f} g. Keep it still for 3 s, then tap / shake it.\n")
    t0, rest, top, hits = time.time(), [], 0.0, 0
    try:
        while True:
            time.sleep(0.1)
            g, crash = imu.read()
            top = max(top, g)
            if time.time() - t0 < 3.0:
                rest.append(g)
            bar = "#" * min(60, int(g * 4))
            flag = ""
            if crash is not None:
                hits += 1
                flag = f"   <-- CRASH detected ({crash:.1f} g)"
            print(f"{g:6.2f} g  {bar}{flag}")
            if imu.fault:
                print("!! no data from the sensor: check the wiring")
    except KeyboardInterrupt:
        pass
    finally:
        imu.close()
        print("\n--- summary ---")
        if rest:
            print(f"at rest (first 3 s): about {sum(rest) / len(rest):.2f} g (should be close to 1.0)")
        print(f"highest peak: {top:.1f} g    crash events: {hits}    read errors: {imu.errors}")


if __name__ == "__main__":
    main()
