"""High-level arm tasks (pick / place / move above) for humans and AI agents.

A task is a short sequence of waypoints executed by the real-time arm
engine.  Each waypoint is checked for reachability *before* the task starts,
so a bad request fails immediately with a clear message instead of moving
half-way.  STOP cancels a running task.

Task JSON (POST /api/task):
  {"task": "move_above", "pixel": [u, v], "height_mm": 90}
  {"task": "pick",  "pixel": [u, v]}              # or "xy": [x, y]
  {"task": "place", "pixel": [u, v]}              # or "xy": [x, y]
  {"task": "pick",  "xyz": [x, y, z]}             # 3D point, e.g. from the OAK-D
  {"task": "move",  "x": .., "y": .., "z": .., "pitch": ..}
  {"task": "gripper", "deg": 0..90}
  {"task": "home"}   {"task": "park"}   {"task": "look"}  (camera viewpoint)
Optional: "image_size": [w, h] if pixel coordinates refer to another
resolution than the calibration image; "speed": 0.1..1.
"""

import itertools
import math
import threading
import time

TASK_DEFAULTS = {
    "home_pose": [235, 0, 234, 0, 0, 30],
    "park_pose": [170, 0, 80, 30, 0, 10],       # compact, low, safe to power off
    "hover_height_mm": 90,          # above the table while travelling
    "grasp_height_mm": 15,          # gripper tip height above table to close
    "release_height_mm": 35,
    "gripper_open_deg": 60,
    "gripper_closed_deg": 0,
    "grasp_pitches_deg": [90, 75, 60, 45],   # preferred first: straight down
    "task_speed": 1.0,
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
        self.observer = None
        self.container_pose = None      # [x, y, z, pitch, roll]: where the gripper lets go over the rover's box            # observer(task, step) is called before each step (camera snapshots)

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
            self._status = {"id": task_id, "task": name, "state": "running", "slow": [],
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
        if name in ("home", "park"):
            h = c[name + "_pose"]
            return [(name, dict(x=h[0], y=h[1], z=h[2], pitch=h[3], roll=h[4], grip=h[5]), 0)]
        if name == "look":
            lp = self.cal.look_pose
            if not lp:
                raise ValueError("No look pose set yet (Calibrate panel: 'Set look pose here')")
            return [("go to look pose", dict(x=lp[0], y=lp[1], z=lp[2], pitch=lp[3], roll=lp[4]), 0.4)]
        if name == "container":
            cp = self.container_pose
            if not cp:
                raise ValueError("No container drop taught yet ('📦 Save container drop here')")
            x, y, z, p, r = (float(v) for v in cp[:5])
            if not self.arm.reachable([x, y, z, p, r, grip]):
                raise ValueError("Container drop pose is not reachable: "
                                 + self.arm.explain_unreachable([x, y, z, p, r, grip]))
            # highest approach point straight above the drop that the arm can reach with the same wrist angle
            # (close to the base the arm can't go much higher with the gripper pointing down)
            up = next((h for h in (float(req.get("clearance_mm", 50)), 35, 20, 10)
                       if self.arm.reachable([x, y, z + h, p, r, grip])), 0.0)
            clear = z + up
            steps = []
            lift_z = max(clear, cur[2] + 40)
            if cur[2] < lift_z and self.arm.reachable([cur[0], cur[1], lift_z, cur[3], cur[4], grip]):
                steps.append(("lift", dict(z=lift_z), 0))       # straight up first: don't drag the item
            steps += [("move above the container", dict(x=x, y=y, z=clear, pitch=p, roll=r), 0)]
            if up:
                steps.append(("down to the drop point", dict(z=z), 0))
            steps += [("release into the container", dict(grip=c["gripper_open_deg"]), 0.25)]
            if up:
                steps.append(("back up", dict(z=clear), 0))
            return steps
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
        if "xyz" in req:            # 3D point from the OAK-D: grab at that height
            x, y, zo = (float(v) for v in req["xyz"])
            table = zo - (c["grasp_height_mm"] if name == "pick" else
                          c["release_height_mm"] if name == "place" else 0)
        else:
            table = self._table()
            x, y = self._xy(req)
        hover = table + float(req.get("height_mm", c["hover_height_mm"]))
        if name == "move_above":
            pitch = self._pick_pitch(x, y, [hover])
            return [("move above", dict(x=x, y=y, z=hover, pitch=pitch, roll=float(req.get("roll", 0))), 0)]
        if name == "pick":
            low = table + float(req.get("grasp_height_mm", c["grasp_height_mm"]))
            pitch = self._pick_pitch(x, y, [hover, low])
            open_deg = float(req.get("open_deg", c["gripper_open_deg"]))
            close_deg = float(req.get("close_deg", c["gripper_closed_deg"]))
            steps = [] if grip >= open_deg - 5 else [("open gripper", dict(grip=open_deg), 0.1)]
            return steps + [
                    ("move above object", dict(x=x, y=y, z=hover, pitch=pitch, roll=float(req.get("roll", 0))), 0),
                    ("descend", dict(z=low), 0.05),     # stops on the object if it is taller
                    ("close gripper", dict(grip=close_deg), 0.25)] + (
                    [] if req.get("no_lift") else [("lift", dict(z=hover), 0)])   # no_lift: next task lifts
        if name == "place":
            low = table + float(req.get("release_height_mm", c["release_height_mm"]))
            pitch = self._pick_pitch(x, y, [hover, low])
            open_deg = float(req.get("open_deg", c["gripper_open_deg"]))
            steps = [("move above place", dict(x=x, y=y, z=hover, pitch=pitch, roll=0), 0),
                     ("lower", dict(z=low), 0.05),
                     ("release", dict(grip=open_deg), 0.2)]
            if not req.get("no_lift"):     # the next move lifts anyway (e.g. back to the look pose)
                steps.append(("lift", dict(z=hover), 0))
            return steps
        raise ValueError("Unknown task. Use move_above, pick, place, move, gripper, home or park")

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

    def _go(self, pose, speed, timeout=15.0, contact=False):
        """Move and wait. contact=True (going down onto an object): if the arm is blocked
        on the way down, that IS the object - stop pushing there and carry on."""
        notice_before = (self.arm.state().get("notice") or {}).get("time")
        if set(pose) <= {"grip"}:
            # jaws only: wait until they have really closed on the object (they stop moving) or reached
            # the angle - never lift while still closing. Stopping on the object is not an error.
            self.arm.move_to(speed=speed, **pose)
            want = float(pose["grip"])
            t0 = time.monotonic()
            last_g, last_t = None, t0
            while time.monotonic() - t0 < 2.0:
                self._check()
                m = self.arm.state().get("pose")
                g = m[5] if m else None
                now = time.monotonic()
                if g is not None:
                    if last_g is None or abs(g - last_g) > 0.7:
                        last_g, last_t = g, now
                    if abs(g - want) < 3 and now - t0 > 0.1:
                        break                                   # fully open / fully shut
                    if now - t0 > 0.35 and now - last_t > 0.25:
                        break                                   # jaws stopped: squeezing the object
                time.sleep(0.02)
            return
        goal = self.arm.move_to(speed=speed, **pose)
        end = time.monotonic() + (6.0 if contact else timeout)
        still_z, still_t = None, time.monotonic()
        last_m, last_t = None, time.monotonic()
        woke = 0.0
        while True:
            self._check()
            st = self.arm.state()
            m = st.get("pose")
            if contact and m:
                if still_z is None or m[2] < still_z - 1.5:
                    still_z, still_t = m[2], time.monotonic()
                if m[2] <= goal[2] + 5:
                    self.bottom_z = m[2]
                    return                                     # really down at the grab height
                blocked = m[2] > goal[2] + 5 and time.monotonic() - still_t > 0.5
                if blocked or (time.monotonic() > end and m[2] > goal[2] + 5):
                    self.arm.move_to(z=m[2] + 3, speed=speed)   # stop pressing on it
                    self.touched_z = self.bottom_z = m[2]
                    time.sleep(0.15)
                    return
            notice = st.get("notice") or {}
            if notice.get("time") and notice.get("time") != notice_before:
                if contact and m:          # going down and the arm can't go further: that's the bottom
                    self.bottom_z = m[2]
                    self.arm.move_to(z=m[2] + 3, speed=speed)
                    self.touched_z = m[2]
                    return
                raise TaskError(notice["message"])
            if st["goal"] is None:
                measured = st["pose"] or st["target"]
                err = math.dist(measured[:3], goal[:3])
                # arrived - or settled a little short: real arms sag 1-2.5 cm under their own weight
                if m is not None and (last_m is None or math.dist(m[:3], last_m[:3]) > 1.0):
                    last_m, last_t = m, time.monotonic()
                settled = time.monotonic() - last_t > 0.15
                if contact:
                    pass                       # going down: only the checks above end this (really low / blocked)
                elif err < 8 or (err < 35 and settled) or time.monotonic() > end:
                    return
            elif m is not None and (last_m is None or math.dist(m[:3], last_m[:3]) > 1.0):
                last_m, last_t = m, time.monotonic()
            elif (not contact and m is not None and time.monotonic() - last_t > 0.4
                  and math.dist(m[:3], goal[:3]) > 15 and time.monotonic() - woke > 0.8):
                # should be moving but isn't: servos switched torque off (overload) - switch it back on
                # straight away instead of waiting seconds for the "not responding" alarm
                woke = time.monotonic()
                try:
                    self.arm.wake()
                except Exception:
                    pass
            if time.monotonic() > end:
                raise TaskError("Timed out waiting for the arm")
            time.sleep(0.02)

    def _run(self, plan, speed):
        self.touched_z = None
        self.bottom_z = None
        try:
            slow = []
            for step, pose, wait in plan:
                t_step = time.monotonic()
                self._set(step=step)
                if self.observer:
                    try:
                        self.observer(self._status.get("task"), step)
                    except Exception:
                        pass
                if step == "lift" and self.touched_z is not None and "z" in pose:
                    pose = dict(pose, z=max(pose["z"], self.touched_z + 80))   # lift clear of a tall object
                try:
                    try:
                        self._go(pose, speed, contact=step in ("descend", "lower"))
                    except TaskError as exc:
                        if not any(k in str(exc) for k in ("not responding", "position feedback",
                                                           "could not follow", "Timed out")):
                            raise
                        self._set(step=step + " (servo hiccup - waking, retrying)")
                        try:
                            self.arm.wake()
                        except Exception:
                            pass
                        time.sleep(0.3)
                        self._go(pose, speed, contact=step in ("descend", "lower"))
                except TaskError as exc:
                    if step.startswith("move above") and ("reach" in str(exc) or "Timed out" in str(exc)):
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
                if time.monotonic() - t_step > 2.5:
                    slow.append(f"{step} {time.monotonic() - t_step:.1f}s")
                    self._set(slow=list(slow))
            name = self._status.get("task")
            if name == "pick":
                self.holding = True
            elif name in ("place", "container"):
                self.holding = False
            elif name == "gripper" and plan and plan[0][1].get("grip", 0) > 20:
                self.holding = False           # Open button: nothing is held any more
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
