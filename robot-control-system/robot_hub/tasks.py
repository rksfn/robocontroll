"""High-level arm tasks (pick / place / move above) for humans and AI agents.

A task is a short sequence of waypoints executed by the real-time arm
engine.  Each waypoint is checked for reachability *before* the task starts,
so a bad request fails immediately with a clear message instead of moving
half-way.  STOP cancels a running task.

Task JSON (POST /api/task):
  {"task": "move_above", "pixel": [u, v], "height_mm": 90}
  {"task": "pick",  "pixel": [u, v]}              # or "xy": [x, y]
  {"task": "place", "pixel": [u, v]}              # or "xy": [x, y]
  {"task": "move",  "x": .., "y": .., "z": .., "pitch": ..}
  {"task": "gripper", "deg": 0..90}
  {"task": "home"}
Optional: "image_size": [w, h] if pixel coordinates refer to another
resolution than the calibration image; "speed": 0.1..1.
"""

import itertools
import math
import threading
import time

TASK_DEFAULTS = {
    "home_pose": [235, 0, 234, 0, 0, 30],
    "hover_height_mm": 90,          # above the table while travelling
    "grasp_height_mm": 15,          # gripper tip height above table to close
    "release_height_mm": 35,
    "gripper_open_deg": 60,
    "gripper_closed_deg": 0,
    "grasp_pitches_deg": [90, 75, 60, 45],   # preferred first: straight down
    "task_speed": 0.6,
}


class TaskError(Exception):
    pass


class TaskRunner:
    _ids = itertools.count(1)

    def __init__(self, arm, safety, calibration, config=None):
        self.arm = arm
        self.safety = safety
        self.cal = calibration
        self.cfg = dict(TASK_DEFAULTS)
        self.cfg.update({k: v for k, v in (config or {}).items() if k in TASK_DEFAULTS})
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread = None
        self._status = {"id": None, "task": None, "state": "idle", "step": "", "message": ""}
        self.holding = False

    # ----------------------------------------------------------- public
    def status(self):
        with self._lock:
            return dict(self._status, holding=self.holding)

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def cancel(self, reason="Cancelled"):
        if self.busy():
            self._cancel.set()
            self._set(message=reason)

    def submit(self, request, source="human"):
        if not isinstance(request, dict):
            raise ValueError("Task must be a JSON object")
        if self.busy():
            raise ValueError("Another task is running - wait, or send STOP")
        name = request.get("task")
        plan = self.plan(request)          # validates everything up front
        task_id = next(self._ids)
        self._cancel.clear()
        with self._lock:
            self._status = {"id": task_id, "task": name, "state": "running",
                            "step": "", "message": "", "source": source,
                            "steps": [p[0] for p in plan]}
        self.safety.record(source, "task", f"#{task_id} {name} " + _summary(request))
        speed = float(request.get("speed", self.cfg["task_speed"]))
        self._thread = threading.Thread(target=self._run, args=(plan, speed), daemon=True)
        self._thread.start()
        return self.status()

    # ----------------------------------------------------------- planning
    def _xy(self, req):
        if "xy" in req:
            x, y = (float(v) for v in req["xy"])
        elif "pixel" in req:
            u, v = (float(p) for p in req["pixel"])
            x, y = self.cal.pixel_to_arm(u, v, req.get("image_size"))
        else:
            raise ValueError("Give 'pixel': [u, v] (camera) or 'xy': [x, y] (arm mm)")
        if not all(math.isfinite(t) for t in (x, y)):
            raise ValueError("Target must be finite")
        return x, y

    def _table(self):
        if self.cal.ready:
            return self.cal.table_z
        raise ValueError("Table height unknown - calibrate the camera first")

    def _pick_pitch(self, x, y, heights):
        for pitch in self.cfg["grasp_pitches_deg"]:
            if all(self.arm.reachable([x, y, z, pitch, 0, 0]) for z in heights):
                return pitch
        pose = [x, y, heights[-1], self.cfg["grasp_pitches_deg"][0], 0, 0]
        raise ValueError(f"Object at ({x:.0f}, {y:.0f}) mm is out of the arm's pick range. "
                         + self.arm.explain_unreachable(pose))

    def plan(self, req):
        """Return [(step name, pose dict, extra wait)] - raises on bad requests."""
        name = req.get("task")
        c = self.cfg
        cur = self.arm.state().get("target")
        if cur is None:
            raise RuntimeError("Arm is not connected")
        grip = cur[5]
        if name == "home":
            h = c["home_pose"]
            return [("home", dict(x=h[0], y=h[1], z=h[2], pitch=h[3], roll=h[4]), 0)]
        if name == "gripper":
            deg = float(req["deg"])
            if not 0 <= deg <= 90:
                raise ValueError("Gripper angle must be 0-90")
            return [("gripper", dict(grip=deg), 0.4)]
        if name == "move":
            pose = {k: float(req[k]) for k in ("x", "y", "z", "pitch", "roll") if k in req}
            full = [pose.get("x", cur[0]), pose.get("y", cur[1]), pose.get("z", cur[2]),
                    pose.get("pitch", cur[3]), pose.get("roll", cur[4]), grip]
            if not self.arm.reachable(full):
                raise ValueError(self.arm.explain_unreachable(full))
            return [("move", pose, 0)]
        table = self._table()
        x, y = self._xy(req)
        hover = table + float(req.get("height_mm", c["hover_height_mm"]))
        if name == "move_above":
            pitch = self._pick_pitch(x, y, [hover])
            return [("move above", dict(x=x, y=y, z=hover, pitch=pitch, roll=0), 0)]
        if name == "pick":
            low = table + float(req.get("grasp_height_mm", c["grasp_height_mm"]))
            pitch = self._pick_pitch(x, y, [hover, low])
            open_deg = float(req.get("open_deg", c["gripper_open_deg"]))
            close_deg = float(req.get("close_deg", c["gripper_closed_deg"]))
            return [("open gripper", dict(grip=open_deg), 0.3),
                    ("move above object", dict(x=x, y=y, z=hover, pitch=pitch, roll=0), 0),
                    ("descend", dict(z=low), 0.1),
                    ("close gripper", dict(grip=close_deg), 0.6),
                    ("lift", dict(z=hover), 0)]
        if name == "place":
            low = table + float(req.get("release_height_mm", c["release_height_mm"]))
            pitch = self._pick_pitch(x, y, [hover, low])
            open_deg = float(req.get("open_deg", c["gripper_open_deg"]))
            return [("move above place", dict(x=x, y=y, z=hover, pitch=pitch, roll=0), 0),
                    ("lower", dict(z=low), 0.1),
                    ("release", dict(grip=open_deg), 0.5),
                    ("lift", dict(z=hover), 0)]
        raise ValueError("Unknown task. Use move_above, pick, place, move, gripper or home")

    # ----------------------------------------------------------- execution
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _check(self):
        if self._cancel.is_set():
            raise TaskError(self._status.get("message") or "Cancelled")
        if self.safety.state()["latched"]:
            raise TaskError("STOP is latched")
        if not self.arm.state()["connected"]:
            raise TaskError("Arm disconnected")

    def _go(self, pose, speed, timeout=15.0):
        notice_before = (self.arm.state().get("notice") or {}).get("time")
        goal = self.arm.move_to(speed=speed, **pose)
        end = time.monotonic() + timeout
        while True:
            self._check()
            st = self.arm.state()
            notice = st.get("notice") or {}
            if notice.get("time") and notice.get("time") != notice_before:
                raise TaskError(notice["message"])
            if st["goal"] is None:
                measured = st["pose"] or st["target"]
                err = math.dist(measured[:3], goal[:3])
                if err < 12 or time.monotonic() > end:
                    return
            if time.monotonic() > end:
                raise TaskError("Timed out waiting for the arm")
            time.sleep(0.03)

    def _run(self, plan, speed):
        try:
            for step, pose, wait in plan:
                self._set(step=step)
                try:
                    self._go(pose, speed)
                except TaskError as exc:
                    if step.startswith("move above") and "reach" in str(exc):
                        # Path cut through an unreachable region: go via home.
                        self._set(step=step + " (via home)")
                        h = self.cfg["home_pose"]
                        self._go(dict(x=h[0], y=h[1], z=h[2], pitch=h[3], roll=h[4]), speed)
                        self._go(pose, speed)
                    else:
                        raise
                t_end = time.monotonic() + wait
                while time.monotonic() < t_end:
                    self._check()
                    time.sleep(0.03)
            name = self._status.get("task")
            if name == "pick":
                self.holding = True
            elif name == "place":
                self.holding = False
            self._set(state="done", step="", message="Done")
        except TaskError as exc:
            self.arm.hold()
            self._set(state="failed", message=str(exc))
            self.safety.record("task", "failed", str(exc))
        except Exception as exc:  # unexpected: stop the arm, report
            self.arm.hold()
            self._set(state="failed", message=f"{type(exc).__name__}: {exc}")
            self.safety.record("task", "failed", str(exc))


def _summary(req):
    keys = ("pixel", "xy", "x", "y", "z", "deg")
    return " ".join(f"{k}={req[k]}" for k in keys if k in req)
