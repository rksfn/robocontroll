import threading
import time
from collections import deque


class SafetyController:
    """Human-resettable stop latch and bounded event log."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latched = True
        self._reason = "Startup stop"
        self._events = deque(maxlen=100)
        self.record("system", "stop", self._reason)

    def record(self, source, event, detail=""):
        with self._lock:
            self._events.appendleft({
                "time": round(time.time(), 3),
                "source": str(source),
                "event": str(event),
                "detail": str(detail)[:300],
            })

    def latch(self, reason, source="system"):
        with self._lock:
            self._latched = True
            self._reason = str(reason)[:200]
            self._events.appendleft({
                "time": round(time.time(), 3), "source": str(source),
                "event": "stop", "detail": self._reason,
            })

    def enable_human(self):
        with self._lock:
            self._latched = False
            self._reason = ""
            self._events.appendleft({
                "time": round(time.time(), 3), "source": "human",
                "event": "enabled", "detail": "Controls enabled from local dashboard",
            })

    def require_enabled(self):
        with self._lock:
            if self._latched:
                raise RuntimeError("STOP is latched: " + self._reason)

    def state(self):
        with self._lock:
            return {
                "latched": self._latched,
                "reason": self._reason,
                "events": list(self._events)[:20],
            }
