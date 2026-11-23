"""Audible alerts: GPIO buzzer on the Pi, a beep on Windows, silent elsewhere."""
import threading
import time


class Alerter:
    def __init__(self, gpio_pin=None):
        self.buzzer = None
        self.buzzing = False
        self._last_beep = 0.0
        if gpio_pin is not None:
            try:
                from gpiozero import Buzzer
                self.buzzer = Buzzer(gpio_pin)
            except Exception as e:  # not on a Pi, or gpiozero missing
                print(f"[alerts] buzzer unavailable: {e}")

    def update(self, level):
        """level: 0 = ok, 1 = warning, 2 = danger."""
        if self.buzzer is not None:
            if level >= 1 and not self.buzzing:
                self.buzzer.beep(on_time=0.15 if level == 2 else 0.4, off_time=0.15, background=True)
                self.buzzing = True
            elif level == 0 and self.buzzing:
                self.buzzer.off()
                self.buzzing = False
            return
        # PC fallback: beep on Windows, rate limited
        now = time.time()
        if level >= 1 and now - self._last_beep > (0.6 if level == 2 else 1.5):
            self._last_beep = now
            try:
                import winsound
                threading.Thread(target=winsound.Beep, args=(1200 if level == 2 else 800, 250),
                                 daemon=True).start()
            except ImportError:
                pass

    def close(self):
        if self.buzzer is not None:
            self.buzzer.off()
