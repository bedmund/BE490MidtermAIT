"""Opt-in Gmail SMTP alerts; no third-party dependencies or audio uploads."""

from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
import math
import queue
import runpy
import smtplib
import ssl
import threading
import time


def load_email_config():
    path = Path(__file__).with_name("email_config.py")
    if not path.exists():
        raise ValueError("Copy email_config.example.py to email_config.py, then edit and uncomment its fields.")
    config = runpy.run_path(str(path))
    keys = ("GMAIL_SENDER", "EMAIL_RECIPIENT", "GMAIL_APP_PASSWORD")
    values = [config.get(key, "") for key in keys]
    if any(not isinstance(value, str) or not value.strip() or '\n' in value or '\r' in value
           for value in values):
        raise ValueError("Set GMAIL_SENDER, EMAIL_RECIPIENT, and GMAIL_APP_PASSWORD in email_config.py.")
    sender, recipient, password = [value.strip() for value in values]
    if ('@' not in sender or '@' not in recipient or sender.startswith('your-')
            or recipient.startswith('your-') or password.startswith('your-')):
        raise ValueError("Replace the dummy email fields in email_config.py with your own configuration.")
    return sender, recipient, password.replace(" ", "")


class EmailAlerts:
    def __init__(self, cooldown=60.0):
        if not math.isfinite(cooldown) or cooldown <= 0:
            raise ValueError("Email cooldown must be positive and finite")
        self.sender, self.recipient, self.password = load_email_config()
        self.cooldown = cooldown
        self._last_queued = float('-inf')
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="glass-email", daemon=True)
        self._thread.start()

    def notify(self, probability, start_seconds, end_seconds):
        """Called from live inference; never waits for a network request."""
        if self._stop.is_set():
            return False
        now = time.monotonic()
        if now - self._last_queued < self.cooldown:
            return False
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = self.recipient
        message["Subject"] = "Possible glass breaking detected"
        message.set_content(
            f"Possible glass breaking detected.\n"
            f"Detection time (UTC): {datetime.now(timezone.utc).isoformat()}\n"
            f"Live audio interval: {start_seconds:.1f}-{end_seconds:.1f} seconds\n"
            f"Glass score: {probability:.3f} (not a calibrated confidence)\n"
            "This is a prototype detection and may be a false alarm. No audio is attached.\n")
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            print("Email queue full; alert skipped.", flush=True)
            return False
        self._last_queued = now
        return True

    def _send(self, message):
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10,
                              context=ssl.create_default_context()) as smtp:
            smtp.login(self.sender, self.password)
            smtp.send_message(message)

    def _worker(self):
        while not self._stop.is_set():
            try:
                message = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stop.is_set():
                return
            try:
                self._send(message)
                print("Alert email accepted by Gmail (inbox delivery is not confirmed).", flush=True)
            except Exception as error:
                # Do not print SMTP responses or config values containing account details.
                print(f"Email failed ({type(error).__name__}); live detection continues. "
                      "Check credentials/network. No automatic retry.", flush=True)

    def close(self):
        # Pending messages are discarded. An in-flight SMTP request may finish.
        self._stop.set()
        self._thread.join(timeout=0.5)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
