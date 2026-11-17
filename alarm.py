"""Optional GPIO LED/active-buzzer outputs, with a hardware-free simulation."""

import math
import threading


class AlarmOutputs:
    def __init__(self, mode="simulate", led_pin=17, buzzer_pin=27, seconds=3.0):
        if mode not in ("simulate", "gpio"):
            raise ValueError("Alarm mode must be simulate or gpio")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Alarm duration must be positive and finite")
        if led_pin == buzzer_pin or not all(0 <= pin <= 27 for pin in (led_pin, buzzer_pin)):
            raise ValueError("Use distinct BCM GPIO pins between 0 and 27")
        self.mode = mode
        self.seconds = seconds
        self._devices = []
        self._lock = threading.Lock()
        self._timer = None
        self._generation = 0
        self._active = False
        self._closed = False
        self._error = None
        if mode == "gpio":
            try:
                from gpiozero import LED, Buzzer
                self._devices.append(LED(led_pin, initial_value=False))
                self._devices.append(Buzzer(buzzer_pin, initial_value=False))
            except Exception as error:
                for device in self._devices:
                    device.close()
                raise RuntimeError("Cannot initialize GPIO alarm. Install gpiozero and a GPIO backend "
                                   "on the Pi, or use --alarm simulate without hardware.") from error
        print(f"Alarm ready ({mode}): LED BCM {led_pin}, active buzzer BCM {buzzer_pin}; "
              f"both OFF, hold {seconds:g}s", flush=True)

    def _set_state(self, active):
        # Attempt both outputs even if one fails; the caller will stop monitoring.
        try:
            if self._devices:
                self._devices[0].value = active
        finally:
            if len(self._devices) > 1:
                self._devices[1].value = active
        if active != self._active:
            self._active = active
            state = "ON" if active else "OFF"
            print(f"ALARM ({self.mode}): light {state}, buzzer {state}", flush=True)

    def trigger(self):
        """Retrigger the hold interval without blocking microphone inference."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Alarm is closed")
            self._generation += 1
            if self._timer is not None:
                self._timer.cancel()
            self._set_state(True)
            self._timer = threading.Timer(self.seconds, self._expire, args=(self._generation,))
            self._timer.daemon = True
            self._timer.start()

    def _expire(self, generation):
        with self._lock:
            if not self._closed and generation == self._generation:
                try:
                    self._set_state(False)
                except Exception as error:
                    self._error = error

    def check(self):
        """Surface timer-thread output errors to the live monitoring loop."""
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Alarm output failed while turning off") from self._error

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
            try:
                self._set_state(False)
            finally:
                try:
                    if self._devices:
                        self._devices[0].close()
                finally:
                    if len(self._devices) > 1:
                        self._devices[1].close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
