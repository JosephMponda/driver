"""Alcohol (intoxication) channel.

A camera cannot reliably tell whether someone is drunk, so this prototype uses a breath-alcohol
sensor as the intoxication input:
  - DFRobotAlcoholSensor : DFRobot Gravity SEN0376 electrochemical sensor over UART (Raspberry Pi)
  - SimulatedAlcoholSensor : no hardware; on a PC press 'a' in the video window to toggle it.
"""
import time


class SimulatedAlcoholSensor:
    name = "simulated"

    def __init__(self):
        self.present = False

    def toggle(self):
        self.present = not self.present

    def calibrate(self):
        pass

    def read(self):
        """Return (level, alcohol_detected)."""
        return (0.9 if self.present else 0.05), self.present


class DFRobotAlcoholSensor:
    """DFRobot SEN0376 electrochemical sensor over UART (auto-upload mode, 1 frame/s).

    Wiring (sensor switch set to UART): + -> 3.3 V (pin 1), - -> GND (pin 6),
    D/T (sensor TX) -> Pi RXD GPIO15 (pin 10), C/R (sensor RX) -> Pi TXD GPIO14 (pin 8).
    Enable the port: sudo raspi-config -> Interface Options -> Serial Port
    (login shell: NO, serial hardware: YES), then reboot. Needs: pip install pyserial.

    Range is 0-5 ppm. First power-up needs a 24-48 h preheat; later ones about 3 minutes.
    Start the program in clean air: the clean-air baseline is recorded at start-up.

    delta_ppm = rise above the baseline that counts as "alcohol".
      Triggers with no alcohol around -> RAISE it
      Sanitizer cotton ball is ignored -> LOWER it
    """
    name = "DFRobot SEN0376"

    def __init__(self, port="/dev/serial0", delta_ppm=0.3):
        import serial   # imported lazily so the PC never needs pyserial
        self.ser = serial.Serial(port, 9600, timeout=0.1)
        self.delta = delta_ppm
        self.baseline = 0.0
        self.ppm = 0.0
        self.frames = 0
        self._buf = bytearray()

    def _poll(self):
        """Read whatever bytes are waiting and decode any complete 9-byte frames (non-blocking)."""
        n = self.ser.in_waiting
        if n:
            self._buf += self.ser.read(n)
        while len(self._buf) >= 9:
            i = self._buf.find(b"\xff")
            if i < 0:
                self._buf.clear()
                break
            if i > 0:
                del self._buf[:i]
            if len(self._buf) < 9:
                break
            fr = bytes(self._buf[:9])
            chk = (~sum(fr[1:8]) + 1) & 0xFF
            if chk == fr[8] and fr[1] == 0x17:             # valid ethanol frame
                self.ppm = (fr[4] * 256 + fr[5]) / 1000.0  # ppb -> ppm
                self.frames += 1
                del self._buf[:9]
            else:
                del self._buf[:1]                            # resync

    def calibrate(self, seconds=6):
        vals, last, t0 = [], 0, time.time()
        while time.time() - t0 < seconds:
            self._poll()
            if self.frames != last:
                last = self.frames
                vals.append(self.ppm)
            time.sleep(0.1)
        if not vals:
            raise SystemExit("DFRobot sensor: no data. Check wiring, UART switch and serial settings.")
        self.baseline = sum(vals) / len(vals)
        print(f"[init] clean-air baseline: {self.baseline:.2f} ppm")

    def read(self):
        """Return (ppm, alcohol_detected)."""
        self._poll()
        return self.ppm, (self.ppm - self.baseline) > self.delta


class SimulatedUltrasonicSensor:
    """No hardware; on a PC press 'u' in the video window to toggle 'too close'."""
    name = "simulated"

    def __init__(self):
        self.close = False

    def toggle(self):
        self.close = not self.close

    def read(self):
        """Return (distance_cm, too_close)."""
        return (15.0 if self.close else 60.0), self.close


class UltrasonicSensor:
    """HC-SR04 ultrasonic distance sensor on Pi GPIO (via gpiozero).

    Wiring: VCC -> 5 V (pin 2), GND -> GND (pin 14), TRIG -> GPIO23 (pin 16),
    ECHO -> GPIO24 (pin 18) THROUGH A VOLTAGE DIVIDER (ECHO outputs 5 V, the Pi pins are 3.3 V):
    ECHO -> 1 kOhm -> GPIO24, and GPIO24 -> 2 kOhm -> GND. (HC-SR04P / 3.3 V versions need no divider.)
    Needs: pip install gpiozero lgpio

    near_cm = closer than this counts as "too close".
    hold_s  = it must stay that close this long before it is reported (ignores flicker).
    """
    name = "HC-SR04"

    def __init__(self, trigger=23, echo=24, near_cm=30.0, max_cm=200.0, hold_s=1.0):
        from gpiozero import DistanceSensor   # imported lazily so the PC never needs gpiozero
        self.s = DistanceSensor(echo=echo, trigger=trigger,
                                max_distance=max_cm / 100.0, queue_len=5)
        self.near = near_cm
        self.hold = hold_s
        self._since = None

    def read(self):
        """Return (distance_cm, too_close). Non-blocking."""
        d = self.s.distance * 100.0
        if d < self.near:
            if self._since is None:
                self._since = time.time()
        else:
            self._since = None
        return d, (self._since is not None and time.time() - self._since >= self.hold)

    def close(self):
        self.s.close()
