"""Crash (impact) sensor: MPU-6050 accelerometer over I2C, plus a simulated one for the PC.

A crash impact lasts only tens of milliseconds and the main loop runs at ~15 FPS, so the MPU-6050 is read
in a background thread (~150-200 Hz). read() returns the PEAK acceleration since the last call.

Impact = acceleration magnitude >= crash_g for at least `confirm_s` (10 ms), so one-sample glitches are ignored. The magnitude includes gravity
(about 1 g at rest), so the board can be mounted in any orientation, but it must be fixed rigidly to the
vehicle body (not loose in a drawer or on a flexible bracket).

Wiring (GY-521 / MPU-6050 board to the Pi):  VCC -> 3.3 V (pin 17)   GND -> GND (pin 9)
                                             SDA -> GPIO2 (pin 3)    SCL -> GPIO3 (pin 5)   AD0 -> GND (address 0x68)
Enable I2C:  sudo raspi-config -> Interface Options -> I2C -> Enable, reboot.   Needs: pip install smbus2
"""
import math
import threading
import time

REG_PWR_MGMT_1, REG_CONFIG, REG_ACCEL_CONFIG = 0x6B, 0x1A, 0x1C
REG_ACCEL_XOUT_H, REG_WHO_AM_I = 0x3B, 0x75


def _s16(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


class MPU6050:
    name = "MPU-6050"
    LSB_PER_G = 2048.0            # +-16 g range

    def __init__(self, crash_g=4.0, bus=1, addr=0x68, confirm_s=0.010, cooldown_s=10.0):
        try:
            from smbus2 import SMBus
        except ImportError:
            raise SystemExit("The MPU-6050 needs smbus2:  pip install smbus2")
        self.crash_g, self.confirm_s, self.cooldown_s, self.addr = crash_g, confirm_s, cooldown_s, addr
        try:
            self.bus = SMBus(bus)
            self.bus.write_byte_data(addr, REG_PWR_MGMT_1, 0x00)    # wake up (it starts in sleep mode)
            time.sleep(0.1)
            self.bus.write_byte_data(addr, REG_CONFIG, 0x01)         # low-pass filter about 184 Hz
            self.bus.write_byte_data(addr, REG_ACCEL_CONFIG, 0x18)   # accelerometer range +-16 g
            who = self.bus.read_byte_data(addr, REG_WHO_AM_I)
        except OSError as e:
            raise SystemExit(f"MPU-6050 not found on I2C bus {bus} at 0x{addr:02x} ({e}).\n"
                             "Check the wiring (SDA/SCL), that I2C is enabled in raspi-config, "
                             "and run: i2cdetect -y 1")
        if who != 0x68:
            print(f"[imu] warning: WHO_AM_I = 0x{who:02x} (expected 0x68). Probably a clone board; continuing.")

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._now, self._peak, self._event = 1.0, 1.0, None
        self._last_ok = time.time()
        self.errors = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        over_since, cool_until = None, 0.0
        while not self._stop.is_set():
            try:
                d = self.bus.read_i2c_block_data(self.addr, REG_ACCEL_XOUT_H, 6)
            except OSError:                       # loose wire under vibration etc.
                with self._lock:
                    self.errors += 1
                time.sleep(0.01)
                continue
            now = time.time()
            ax, ay, az = (_s16(d[i], d[i + 1]) / self.LSB_PER_G for i in (0, 2, 4))
            g = math.sqrt(ax * ax + ay * ay + az * az)
            with self._lock:
                self._last_ok, self._now = now, g
                self._peak = max(self._peak, g)
                if g >= self.crash_g:
                    over_since = over_since or now
                    if now - over_since >= self.confirm_s and now >= cool_until:
                        self._event = self._peak
                        cool_until = now + self.cooldown_s     # one event per crash, not one per bounce
                else:
                    over_since = None
            time.sleep(0.002)

    def read(self):
        """Return (peak_g_since_last_call, crash_peak_g_or_None). The crash value is returned once."""
        with self._lock:
            g, ev = self._peak, self._event
            self._peak, self._event = self._now, None
        return g, ev

    @property
    def fault(self):
        """True if no sample has arrived for 1 s (sensor unplugged / wire shaken loose)."""
        return time.time() - self._last_ok > 1.0

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.bus.close()
        except Exception:
            pass


class SimulatedImu:
    """PC testing: press 'x' in the video window to simulate a crash."""
    name = "simulated IMU"
    fault = False

    def __init__(self):
        self._crash = False

    def trigger(self):
        self._crash = True

    def read(self):
        if self._crash:
            self._crash = False
            return 9.0, 9.0
        return 1.0, None

    def close(self):
        pass
