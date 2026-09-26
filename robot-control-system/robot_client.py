"""Tiny Python client for the Robot Control Hub HTTP API.

Use this from any script, notebook or AI agent:

    from robot_client import Robot
    bot = Robot()                      # http://127.0.0.1:8765
    bot.pick(pixel=(312, 188))         # pick the object at that camera pixel
    bot.place(pixel=(480, 240))
    bot.home()

Every motion call blocks until the task finishes and raises RobotError if
it fails (out of reach, STOP pressed, not calibrated, ...).  A human must
press "Enable controls" on the dashboard first; the API cannot do that.
"""

import json
import time
import urllib.error
import urllib.request


class RobotError(RuntimeError):
    pass


class Robot:
    def __init__(self, url="http://127.0.0.1:8765", source="agent", timeout=60):
        self.url = url.rstrip("/")
        self.source = source
        self.timeout = timeout

    # ------------------------------------------------------------ transport
    def _req(self, path, data=None, raw=False):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(self.url + path, data=body,
                                     method="GET" if data is None else "POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                payload = r.read()
        except urllib.error.HTTPError as exc:
            try:
                msg = json.loads(exc.read()).get("error", str(exc))
            except Exception:
                msg = str(exc)
            raise RobotError(msg) from None
        except urllib.error.URLError as exc:
            raise RobotError(f"Robot hub not reachable at {self.url}: {exc.reason}") from None
        return payload if raw else json.loads(payload)

    # ------------------------------------------------------------ reading
    def state(self):
        return self._req("/api/state")

    def frame(self):
        """Current overhead camera frame as JPEG bytes."""
        return self._req("/api/frame.jpg", raw=True)

    def image_size(self):
        return self.state()["camera"]["size"]

    def pixel_to_arm(self, u, v):
        return self._req(f"/api/pixel?u={u}&v={v}")["xy"]

    # ------------------------------------------------------------ safety
    def stop(self, reason="Agent stop"):
        return self._req("/api/stop", {"reason": reason})

    # ------------------------------------------------------------ tasks
    def task(self, wait=True, **request):
        request.setdefault("source", self.source)
        status = self._req("/api/task", request)
        if not wait:
            return status
        task_id, end = status["id"], time.monotonic() + self.timeout
        while time.monotonic() < end:
            st = self._req("/api/task")
            if st["id"] != task_id or st["state"] != "running":
                if st["state"] == "failed":
                    raise RobotError(st["message"])
                return st
            time.sleep(0.1)
        self.cancel()
        raise RobotError("Task timed out")

    def cancel(self):
        return self._req("/api/task/cancel", {"source": self.source})

    def pick(self, pixel=None, xy=None, **kw):
        return self.task(task="pick", **_target(pixel, xy), **kw)

    def place(self, pixel=None, xy=None, **kw):
        return self.task(task="place", **_target(pixel, xy), **kw)

    def move_above(self, pixel=None, xy=None, height_mm=None, **kw):
        extra = {} if height_mm is None else {"height_mm": height_mm}
        return self.task(task="move_above", **_target(pixel, xy), **extra, **kw)

    def move(self, **pose):
        """Absolute arm pose: x, y, z (mm), pitch, roll (deg). Omitted = unchanged."""
        return self.task(task="move", **pose)

    def gripper(self, deg):
        return self.task(task="gripper", deg=deg)

    def home(self):
        return self.task(task="home")

    # ------------------------------------------------------------ rover
    def drive(self, linear, turn=0.0, duration_ms=300):
        return self._req("/api/action", {"source": self.source, "action": "drive",
                                         "linear": linear, "turn": turn,
                                         "duration_ms": duration_ms})


def _target(pixel, xy):
    if (pixel is None) == (xy is None):
        raise ValueError("Give exactly one of pixel=(u, v) or xy=(x_mm, y_mm)")
    return {"pixel": list(pixel)} if pixel is not None else {"xy": list(xy)}
