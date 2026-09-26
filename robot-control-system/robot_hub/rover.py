import json
import threading
import time
import urllib.parse
import urllib.request


def mix_speeds(linear, turn, maximum):
    linear = max(-1.0, min(1.0, float(linear)))
    turn = max(-1.0, min(1.0, float(turn)))
    left = max(-1.0, min(1.0, linear + turn))
    right = max(-1.0, min(1.0, linear - turn))
    return round(left * maximum), round(right * maximum)


class RoverDriver:
    """Duration-bounded rover commands with a local refresh watchdog."""

    def __init__(self, base_url, max_speed, safety, opener=None):
        self.base_url = base_url.rstrip("/")
        self.max_speed = max(0, min(int(max_speed), 1800))
        self.safety = safety
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._done = threading.Event()
        self._desired = (0, 0)
        self._deadline = 0.0
        self._last_sent = None
        self._connected = False
        self._error = "Not contacted"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _url(self, left, right):
        command = json.dumps({"T": 1, "L": left, "R": right}, separators=(",", ":"))
        return f"{self.base_url}/js?{urllib.parse.urlencode({'json': command})}"

    def command(self, linear, turn, duration_ms):
        duration_ms = int(duration_ms)
        if not 50 <= duration_ms <= 500:
            raise ValueError("Rover duration must be between 50 and 500 ms")
        speeds = mix_speeds(linear, turn, self.max_speed)
        with self._lock:
            self._desired = speeds
            self._deadline = time.monotonic() + duration_ms / 1000
        self._wake.set()
        return speeds

    def stop(self):
        with self._lock:
            self._desired = (0, 0)
            self._deadline = 0
        self._wake.set()

    def _send(self, command):
        with self.opener.open(self._url(*command), timeout=.35) as response:
            response.read(64)

    def _run(self):
        while not self._done.is_set():
            with self._lock:
                expired = time.monotonic() >= self._deadline
                command = (0, 0) if expired else self._desired
            if command != self._last_sent or command != (0, 0):
                try:
                    self._send(command)
                    with self._lock:
                        self._connected = True
                        self._error = ""
                    self._last_sent = command
                except Exception as exc:
                    with self._lock:
                        self._connected = False
                        self._error = str(exc)
                    self._last_sent = None
                    # Failure of a zero-speed startup probe is a status issue,
                    # not a new motion hazard.  Active-command failures still
                    # latch STOP immediately.
                    if command != (0, 0):
                        self.safety.latch("Rover connection failed", "rover")
                    with self._lock:
                        self._desired = (0, 0)
                        self._deadline = 0
            self._wake.wait(.1)
            self._wake.clear()

    def state(self):
        with self._lock:
            return {
                "connected": self._connected,
                "error": self._error,
                "last_sent": self._last_sent,
                "url": self.base_url,
                "max_speed": self.max_speed,
            }

    def close(self):
        self.stop()
        deadline = time.monotonic() + .8
        while self._last_sent != (0, 0) and time.monotonic() < deadline:
            time.sleep(.03)
        self._done.set()
        self._wake.set()
        self._thread.join(timeout=1)
