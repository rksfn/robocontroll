"""Qwen autopilot: closed-loop control of the arm from the GRIPPER camera.

No camera calibration.
  0. SCAN: the arm goes to a high "look" pose and turns the base through a
     fan of angles, taking a photo at each. Qwen checks all photos at once
     (parallel requests) and the arm turns to the one where the target is.
  1. LOOP (about 1-2 s per step): fresh photo -> Qwen says where the target
     is, where the jaws are, and the phase (align / approach / grasp / lift /
     release / done) -> one small, reach-checked move -> repeat.

Which way the image moves when the arm moves is learned on the fly (a 2x2
image Jacobian from two small probe moves, refined after every step), so it
works however the camera is mounted.  STOP / Space / Stop ends it at once.
"""

import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ai import extract_json

STATES = ("search", "align", "approach", "grasp", "lift", "release", "done", "fail")
STEP_MM = 130           # biggest XY move per step
STEP_Z_MM = 70          # biggest Z move per step
PROBE_MM = (25, 12)     # probe sizes: try the big one, then the small one
ALIGN_PX = 0.07         # "lined up" when the error is below 7 % of the image width
GAIN = 1.0              # correct the whole measured error in one move
SETTLE_S = 0.12         # pause after the arm has arrived, before the photo
IMG_WIDTH = 336         # photos are shrunk to this width for Qwen (faster)
SPEED = 1.0             # arm speed for autopilot moves
CONTACT_MM = 20         # "touching": the arm stopped this far above where it was sent
                        # (real arms sag/settle a few mm short under their own weight)
SCAN_ANGLES = (0, -30, 30, -60, 60, -90, 90, -120, 120)
SCAN_POSES = ((210, 100, 45), (230, 50, 60), (180, 150, 30), (250, 0, 60), (200, 40, 75))


class PilotStopped(Exception):
    pass


class QwenPilot:
    def __init__(self, controller, log_path=None):
        self.c = controller
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.J = None                    # [[du/dt, du/dr], [dv/dt, dv/dr]] px per mm
        self._jaws = None
        self._reset_status()

    # ------------------------------------------------------------ public
    def _reset_status(self, goal=""):
        self._status = {"running": False, "goal": goal, "step": 0, "state": "idle",
                        "say": "", "target": None, "gripper": None, "log": [], "result": "",
                        "scene_target": None, "zone": None, "zone_poly": None, "held": None}

    def status(self):
        with self._lock:
            st = dict(self._status)
            st["log"] = list(st["log"][-25:])
            return st

    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, goal, max_steps=40, scan=True, strategy="claw"):
        goal = str(goal or "").strip()[:300]
        if not goal:
            raise ValueError("Type what the robot should do, e.g. 'pick up the cup'")
        if self.running():
            raise ValueError("Autopilot is already running")
        if self.c.tasks.busy():
            raise ValueError("A task is running - wait or cancel it first")
        self.c.safety.require_enabled()
        if not self.c.arm.state().get("connected"):
            raise RuntimeError("Arm is not connected")
        self._stop.clear()
        with self._lock:
            self._reset_status(goal)
            self._status.update(running=True, state="starting")
        self.J, self._jaws = None, None
        cal = self.c.calibration
        plan_ok = cal.ready and bool(cal.look_pose)     # camera is on the gripper: needs a look pose
        container = bool(self.c.config.get("container_pose"))
        if strategy == "auto" and plan_ok and container:
            strategy = "collect"                         # everything goes into the rover's container
        if strategy == "auto":
            if plan_ok and re.search(r"\b(sort|tidy|zone|yellow|triangle|drop[- ]?off|coin|pile|everything|paper|segregate|separate)\b", goal, re.I):
                strategy = "sort"
            else:
                strategy = "plan" if plan_ok else "claw"
        if strategy == "plan" and not plan_ok:
            raise ValueError("Plan & pick needs the gripper-camera calibration first: "
                             "Calibrate → Set look pose here → 5-6 touch points")
        if strategy == "sort" and not plan_ok:
            raise ValueError("Sorting needs the gripper-camera calibration first (Calibrate → look pose → points)")
        if strategy == "collect" and not plan_ok:
            raise ValueError("Collecting needs the gripper-camera calibration first (Calibrate → look pose → points)")
        if strategy == "collect" and not container:
            raise ValueError("Teach the container drop first: jog the gripper over the rover's container, "
                             "then press '📦 Save container drop here'")
        run = {"plan": self._run_plan, "claw": self._run_claw, "sort": self._run_sort,
               "collect": self._run_collect}.get(strategy, self._run)
        self._thread = threading.Thread(target=run, args=(goal, int(max_steps), bool(scan)),
                                        daemon=True)
        self._thread.start()
        self.c.safety.record("qwen", "autopilot", f"start: {goal}")
        return self.status()

    def stop(self, reason="Stopped"):
        if self.running():
            self._stop.set()
            self._log("stop", reason)

    # ------------------------------------------------------------ helpers
    def _log(self, state, text, **extra):
        try:
            p = self.c.arm.state().get("pose")
            extra.setdefault("pose", p and [round(v) for v in p])
        except Exception:
            pass
        entry = {"t": round(time.time(), 1), "state": state, "text": str(text)[:240]}
        entry.update(extra)
        with self._lock:
            self._status["log"].append(entry)
            self._status["log"] = self._status["log"][-60:]
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(time.strftime("%H:%M:%S ") + json.dumps(entry) + "\n")
            except OSError:
                pass

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _check(self):
        if self._stop.is_set():
            raise PilotStopped("Stopped")
        if self.c.safety.state()["latched"]:
            raise PilotStopped("STOP pressed")
        if not self.c.arm.state().get("connected"):
            raise PilotStopped("Arm disconnected")

    def _pose(self):
        st = self.c.arm.state()
        p = st.get("pose") or st.get("target")
        if p is None:
            raise PilotStopped("Arm position unknown")
        return list(p)

    def _wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check()
            time.sleep(0.02)

    def _go(self, timeout=3.5, speed=SPEED, **pose):
        """Move (clamped to the reachable workspace) and wait until the arm has
        REALLY arrived (measured position), so the next photo is sharp and true."""
        self._check()
        st0 = self.c.arm.state()
        before_cmd = st0.get("goal") or st0.get("target")
        before_meas = st0.get("pose")
        goal = self.c.arm.move_to(clamp=True, speed=speed, **pose)
        if set(pose) == {"grip"}:              # jaws only: they may stop on an object
            self._wait(0.5)
            return goal
        end = time.monotonic() + timeout
        m = None
        going_down = before_cmd is not None and goal[2] < before_cmd[2] - 5
        low_z, low_t = None, time.monotonic()
        prev, still_since = None, time.monotonic()
        while time.monotonic() < end:
            self._check()
            st = self.c.arm.state()
            m = st.get("pose")
            if m and (prev is None or math.dist(m[:3], prev[:3]) > 1.0):
                prev, still_since = m, time.monotonic()
            if st["goal"] is None and m:
                off = math.dist(m[:3], goal[:3])
                # arrived, or settled a little short (real arms sag under their own weight)
                if off < 8 or (off < CONTACT_MM and time.monotonic() - still_since > 0.2):
                    return goal
            if going_down and m:
                if low_z is None or m[2] < low_z - 1.5:
                    low_z, low_t = m[2], time.monotonic()
                elif st["goal"] is None and m[2] > goal[2] + CONTACT_MM and time.monotonic() - low_t > 0.5:
                    break                    # stopped well short of the target: touching something
            time.sleep(0.02)
        # did not arrive: find out why
        if m and before_cmd and going_down and m[2] > goal[2] + CONTACT_MM:
            # blocked on the way down: touching the object or the table. Stop pushing NOW.
            self._contact = True
            self.c.arm.move_to(x=m[0], y=m[1], z=m[2] + 4, clamp=True, speed=speed)
            self._log("contact", f"touched something at z {m[2]:.0f} mm - stopped pushing")
            self._frozen = 0
            return goal
        if m and before_meas and before_cmd:
            asked = math.dist(goal[:3], before_cmd[:3])
            moved = math.dist(m[:3], before_meas[:3])
            if asked > 30 and moved < 3 and time.monotonic() - getattr(self, "_woke_at", 0) > 3:
                self._frozen = getattr(self, "_frozen", 0) + 1
                if self._frozen >= 2:
                    raise PilotStopped("The arm is not moving at all - its servos are probably unpowered or "
                                       "in overload protection. Check the arm's power supply/switch, press "
                                       "'Wake arm', then try again.")
                self._log("arm", "arm did not move - re-enabling servo torque and retrying")
                self.c.arm.wake()
                self._woke_at = 0
                self._wait(0.3)
                return self._go(timeout=timeout, speed=speed, **pose)
            self._frozen = 0
            if goal[2] < before_cmd[2] - 5 and m[2] > goal[2] + CONTACT_MM:
                # blocked on the way down: touching the object or the table. Stop pushing.
                self._contact = True
                self.c.arm.move_to(x=m[0], y=m[1], z=m[2] + 4, clamp=True, speed=speed)
                self._log("contact", f"touched something at z {m[2]:.0f} mm - stopped pushing")
                return goal
        self._frozen = 0
        return goal

    def _frame(self):
        """Camera frame taken after the arm stopped (short settle, no big pause)."""
        self._wait(SETTLE_S)
        t0 = time.time()
        while time.time() - t0 < 0.5:        # make sure the frame is newer than the stop
            self._check()
            age = self.c.camera.state().get("frame_age_seconds")
            if age is not None and age <= time.time() - t0 + SETTLE_S - 0.05:
                break
            time.sleep(0.02)
        return self.c._frame()

    @staticmethod
    def _axes(pose):
        """Tool frame: r = straight out from the base, t = to the right of that."""
        yaw = math.atan2(pose[1], pose[0]) if math.hypot(pose[0], pose[1]) > 1 else 0.0
        return (math.cos(yaw), math.sin(yaw)), (math.sin(yaw), -math.cos(yaw))

    def _reachable_pitch(self, x, y, z, want, pose):
        for cand in sorted({want, 90, 75, 60, 45, 30, 15, 0}, key=lambda a: abs(a - want)):
            if self.c.arm.reachable([x, y, z, cand, pose[4], pose[5]]):
                return cand
        return None

    def _target(self):
        st = self.c.arm.state()
        p = st.get("goal") or st.get("target") or st.get("pose")
        if p is None:
            raise PilotStopped("Arm position unknown")
        return list(p)

    def _move_tool(self, dt=0.0, dr=0.0, dz=0.0, dpitch=0.0, grip=None):
        if getattr(self, "_claw_mode", False) and not dz and not dpitch and grip is None:
            d = self._claw_slide(dt, dr)
            return (d[0], d[1], 0.0)
        pose = self._target()
        r, t = self._axes(pose)
        dt = max(-STEP_MM, min(STEP_MM, dt))
        dr = max(-STEP_MM, min(STEP_MM, dr))
        dz = max(-STEP_Z_MM, min(STEP_Z_MM, dz))
        x = pose[0] + dt * t[0] + dr * r[0]
        y = pose[1] + dt * t[1] + dr * r[1]
        z = pose[2] + dz
        pitch = self._reachable_pitch(x, y, z, max(-10, min(90, pose[3] + dpitch)), pose)
        if pitch is None and dz > 0:          # can't go up here: stay at this height
            z = pose[2]
            pitch = self._reachable_pitch(x, y, z, pose[3], pose)
        kw = dict(x=x, y=y, z=z, pitch=pose[3] if pitch is None else pitch)
        if grip is not None:
            kw["grip"] = grip
        try:
            after = self._go(**kw)
        except ValueError as exc:
            self._log("limit", f"move skipped: {exc}")
            after = pose
        mx, my = after[0] - pose[0], after[1] - pose[1]
        return (mx * t[0] + my * t[1], mx * r[0] + my * r[1], after[2] - pose[2])

    # ------------------------------------------------------------ Qwen
    def _parse(self, raw, seen, scale, size):
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("Qwen reply was not an object")
        out = {"state": str(data.get("state", "search")).lower().strip(),
               "say": str(data.get("why", data.get("say", "")))[:120], "size": size}
        if out["state"] not in STATES:
            out["state"] = "search"
        for key, name in (("target", "target"), ("jaws", "jaws")):
            try:
                box, centre = self.c.ai.to_camera(data.get(key) or data.get(key + "_2d"), seen, scale, size)
                out[name] = {"bbox": box, "px": list(centre)}
            except (ValueError, TypeError):
                out[name] = None
        if out["target"] is None and out["state"] in ("align", "approach"):
            out["state"] = "search"
        return out

    def _prompt(self, goal, pose, history):
        grip = pose[5]
        held = self.c.tasks.holding
        return f"""Camera ON A ROBOT GRIPPER (the view moves with the gripper).
Goal: {goal}
Jaws {"OPEN" if grip > 20 else "CLOSED"}. {"HOLDING an object." if held else "Not holding anything."} Height {pose[2]:.0f} mm.
Recent: {"; ".join(history[-3:]) or "none"}
{{COORDS}}
Answer with compact JSON only, no extra text:
{{"target":[x1,y1,x2,y2] or null,"jaws":[x,y] or null,"state":"...","why":"max 8 words"}}
target = tight box around the object of the goal (when placing: where to put it); null if not visible.
jaws = point between the gripper fingers if visible, else null.
state = search (target not visible) | align (visible, not lined up with the jaws) | approach (lined up, go closer) | grasp (object between OPEN jaws, close now) | lift (holding, raise it) | release (at the place spot, open) | done (goal achieved) | fail."""

    def _ask(self, goal, history):
        jpeg, size = self._frame()
        pose = self._pose()
        raw, seen, scale = self.c.ai.ask(self._prompt(goal, pose, history), jpeg, size,
                                         max_tokens=90, max_width=IMG_WIDTH)
        return self._parse(raw, seen, scale, size)

    # ------------------------------------------------------------ scan
    def _scan(self, goal):
        """Turn the base through a fan of angles, photo at each, Qwen checks all at once."""
        pose = self._pose()
        spot = None
        for r0, z0, p0 in SCAN_POSES:
            if all(self.c.arm.reachable([r0 * math.cos(math.radians(a)), r0 * math.sin(math.radians(a)),
                                         z0, p0, 0, pose[5]]) for a in (0, 60, -60)):
                spot = (r0, z0, p0)
                break
        if spot is None:
            self._log("scan", "no reachable scan pose - skipping the scan")
            return False
        r0, z0, p0 = spot
        here = math.degrees(math.atan2(pose[1], pose[0]))
        # one sweep, starting from the end nearest to where the arm is now
        angles = sorted(SCAN_ANGLES)
        if abs(angles[-1] - here) < abs(angles[0] - here):
            angles.reverse()
        shots = []
        self._set(state="scanning")
        for a in angles:
            x, y = r0 * math.cos(math.radians(a)), r0 * math.sin(math.radians(a))
            if not self.c.arm.reachable([x, y, z0, p0, 0, pose[5]]):
                continue
            self._go(x=x, y=y, z=z0, pitch=p0, roll=0)
            jpeg, size = self._frame()
            shots.append((a, jpeg, size))
        self._log("scan", f"took {len(shots)} photos from {angles[0]} to {angles[-1]} deg, asking Qwen…")
        if len(shots) >= 2 and self._same_photo(shots[0][1], shots[-1][1]):
            raise PilotStopped("All scan photos look the same - the arm did not turn. Check the arm's power, "
                               "press 'Wake arm', then try again.")
        sheet = self._contact_sheet(shots)
        if sheet is not None:
            best = self._pick_tile(goal, sheet, shots)
            if best is None:
                self._log("scan", "Qwen: target not in any photo")
                return False
            a = best
            self._go(x=r0 * math.cos(math.radians(a)), y=r0 * math.sin(math.radians(a)), z=z0, pitch=p0, roll=0)
            self._log("scan", f"Qwen picked the photo at {a} deg - turning there")
            return True
        prompt = (f"Robot gripper camera photo. Goal: {goal}. Is the object this goal is about visible? "
                  "{COORDS} Answer compact JSON only: "
                  '{"target":[x1,y1,x2,y2] or null,"why":"max 6 words"}')

        def look(shot):
            a, jpeg, size = shot
            try:
                raw, seen, scale = self.c.ai.ask(prompt, jpeg, size, max_tokens=60, max_width=IMG_WIDTH)
                box, centre = self.c.ai.to_camera(extract_json(raw).get("target"), seen, scale, size)
                return a, box, centre, size
            except Exception:
                return a, None, None, size

        with ThreadPoolExecutor(max_workers=len(shots) or 1) as pool:
            found = list(pool.map(look, shots))
        self._check()
        best, best_score = None, -9
        for a, box, centre, size in found:
            if not box:
                continue
            w, h = size
            area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            off = math.hypot(centre[0] / w - 0.5, centre[1] / h - 0.5)
            score = min(area, 0.25) - 0.6 * off
            if score > best_score:
                best, best_score = (a, box, centre), score
        seen_at = [a for a, box, _, _ in found if box]
        if best is None:
            self._log("scan", "target not seen in any photo")
            return False
        a = best[0]
        self._go(x=r0 * math.cos(math.radians(a)), y=r0 * math.sin(math.radians(a)), z=z0, pitch=p0, roll=0)
        self._log("scan", f"target seen at {seen_at} deg - best view at {a} deg, turning there")
        return True

    @staticmethod
    def _same_photo(a, b):
        try:
            import cv2
            import numpy as np
            x = cv2.imdecode(np.frombuffer(a, np.uint8), cv2.IMREAD_GRAYSCALE)
            y = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
            if x is None or y is None or x.shape != y.shape:
                return False
            x = cv2.resize(x, (80, 60)).astype("float32")
            y = cv2.resize(y, (80, 60)).astype("float32")
            return float(np.mean(np.abs(x - y))) < 4.0
        except Exception:
            return False

    @staticmethod
    def _contact_sheet(shots):
        """All scan photos in one numbered grid image (Qwen compares them in one go)."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        tiles = []
        for i, (_, jpeg, _) in enumerate(shots, 1):
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.resize(img, (320, 240))
            cv2.rectangle(img, (0, 0), (46, 38), (0, 0, 0), -1)
            cv2.putText(img, str(i), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            cv2.rectangle(img, (0, 0), (319, 239), (255, 255, 255), 2)
            tiles.append(img)
        cols = 3
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        grid = np.vstack(rows)
        ok, enc = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return (enc.tobytes(), (grid.shape[1], grid.shape[0])) if ok else None

    def _pick_tile(self, goal, sheet, shots):
        jpeg, size = sheet
        n = len(shots)
        prompt = (f"This grid shows {n} numbered photos (1-{n}) taken by a robot gripper camera while "
                  f"turning. Goal: {goal}. Which ONE photo shows the object of the goal most clearly? "
                  f"If it is in none of them, answer 0. Answer compact JSON only: "
                  '{"photo": number, "why": "max 6 words"}')
        raw, _, _ = self.c.ai.ask(prompt, jpeg, size, max_tokens=40, max_width=672)
        try:
            k = int(extract_json(raw).get("photo", 0))
        except (ValueError, TypeError, AttributeError):
            return None
        if not 1 <= k <= n:
            return None
        return shots[k - 1][0]

    # ------------------------------------------------------------ OAK-D: find it in 3D
    def _oak_find(self, goal):
        """(x, y, z) of the target in arm coordinates from the OAK-D, or None."""
        sc = self.c.scene
        if sc is None or not self.c.scene_cal.ready or not sc.state().get("depth"):
            return None
        depth, K = sc.depth_frame()
        jpeg, size = sc.frame(), sc.size()
        if jpeg is None or depth is None:
            return None
        depth = depth.copy()
        prompt = (f"Photo from the camera on the front of a small robot vehicle that has a robot arm. "
                  f"Goal: {goal}. Box the object this goal is about. "
                  "{COORDS} Answer compact JSON only: "
                  '{"target":[x1,y1,x2,y2] or null}')
        self._set(state="looking with OAK-D")
        raw, seen, scale = self.c.ai.ask(prompt, jpeg, size, max_tokens=40, max_width=IMG_WIDTH)
        try:
            box, centre = self.c.ai.to_camera(extract_json(raw).get("target"), seen, scale, size)
        except (ValueError, TypeError, AttributeError):
            self._log("oak", "OAK-D: target not visible")
            return None
        self._set(scene_target={"bbox": box, "px": list(centre)})
        cam = sc.point_3d(centre[0], centre[1], box=box, depth=depth, K=K)
        if cam is None:
            self._log("oak", "OAK-D sees it but has no depth there")
            return None
        x, y, z = self.c.scene_cal.cam_to_arm(cam)
        self._log("oak", f"OAK-D: target at arm x {x:.0f}, y {y:.0f}, z {z:.0f} mm ({cam[2] / 10:.0f} cm away)")
        return (x, y, z)

    def _oak_locate(self, goal):
        """Qwen finds the target in the OAK-D photo, depth gives its 3D position,
        the arm goes straight above it. False if the OAK-D can't help."""
        sc = self.c.scene
        if sc is None or not self.c.scene_cal.ready or not sc.state().get("depth"):
            return False
        depth, K = sc.depth_frame()
        jpeg, size = sc.frame(), sc.size()
        if jpeg is None or depth is None:
            return False
        depth = depth.copy()
        prompt = (f"Photo from the camera on the front of a small robot vehicle that has a robot arm. "
                  f"Goal: {goal}. Box the object this goal is about (when placing: where to put it). "
                  "{COORDS} Answer compact JSON only: "
                  '{"target":[x1,y1,x2,y2] or null,"why":"max 6 words"}')
        self._set(state="looking with OAK-D")
        raw, seen, scale = self.c.ai.ask(prompt, jpeg, size, max_tokens=60, max_width=IMG_WIDTH)
        try:
            box, centre = self.c.ai.to_camera(extract_json(raw).get("target"), seen, scale, size)
        except (ValueError, TypeError, AttributeError):
            self._log("oak", "OAK-D: target not visible")
            return False
        self._set(scene_target={"bbox": box, "px": list(centre)})
        cam = sc.point_3d(centre[0], centre[1], box=box, depth=depth, K=K)
        if cam is None:
            self._log("oak", "OAK-D sees it but has no depth there (too close / shiny?)")
            return False
        x, y, z = self.c.scene_cal.cam_to_arm(cam)
        self.oak_xyz = (x, y, z)
        pose = self._pose()
        for hover in (z + 90, z + 60, z + 120, z + 40):
            for pitch in (90, 75, 60, 45, 30):
                if self.c.arm.reachable([x, y, hover, pitch, 0, pose[5]]):
                    self._log("oak", f"OAK-D: target at arm x {x:.0f}, y {y:.0f}, z {z:.0f} mm "
                                     f"({cam[2] / 10:.0f} cm away) → moving above it")
                    self._go(x=x, y=y, z=hover, pitch=pitch, roll=0, timeout=6)
                    return True
        self._log("oak", f"OAK-D: target at x {x:.0f}, y {y:.0f}, z {z:.0f} mm is out of the arm's reach")
        return False

    # ------------------------------------------------------------ image Jacobian
    def _probe(self, goal, history, first):
        base = first["target"]["px"]
        cols = []
        for axis in ("t", "r"):
            ok = False
            h = self._target()[2] - float(self.c.arm.cfg.get("arm_z_min_mm", -120))
            sizes = PROBE_MM if not getattr(self, "_claw_mode", False) else \
                (max(8.0, min(25.0, 0.12 * h)), max(5.0, min(12.0, 0.06 * h)))
            for size_mm in sizes:
                d = self._move_tool(dt=size_mm if axis == "t" else 0, dr=size_mm if axis == "r" else 0)
                moved = d[0] if axis == "t" else d[1]
                if getattr(self, "_claw_mode", False):
                    base_box = first["target"]["bbox"]
                    area = lambda b: max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                    for _try in range(3):                     # skip boxes that are clearly something else
                        tgt, sz = self._box_down(goal)
                        if tgt is None:
                            break
                        jump = math.hypot(tgt["px"][0] - base[0], tgt["px"][1] - base[1]) / sz[0]
                        ratio = area(tgt["bbox"]) / area(base_box)
                        if 0.4 < ratio < 2.5 and jump < 0.35:
                            break
                        self._log("probe", "that box looks like a different thing - looking again")
                        tgt = None
                    seen = {"target": tgt}
                else:
                    seen = self._ask(goal, history + [f"small test move {axis}"])
                self._show(seen)
                self._move_tool(dt=-d[0], dr=-d[1])
                if seen.get("target") and abs(moved) >= 4:
                    p = seen["target"]["px"]
                    cols.append(((p[0] - base[0]) / moved, (p[1] - base[1]) / moved))
                    ok = True
                    break
            if not ok:
                self._log("probe", f"test move {axis}: target lost - using a default guess")
                return False
        self.J = [[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]]
        det = self.J[0][0] * self.J[1][1] - self.J[0][1] * self.J[1][0]
        self._log("probe", "learned image motion %.2f %.2f / %.2f %.2f px/mm" % (
            self.J[0][0], self.J[0][1], self.J[1][0], self.J[1][1]))
        if abs(det) < 0.01:
            self.J = None
            self._log("probe", "image barely moved - using a default guess")
            return False
        return True

    @staticmethod
    def _default_J(size):
        """Guess: scene moves opposite to the camera, ~2 px/mm at 640 px wide."""
        k = 2.0 * size[0] / 640.0
        return [[-k, 0.0], [0.0, k]]

    def _solve(self, du, dv, size):
        """Wanted target shift in the image (px) -> tool-frame move (dt, dr) in mm."""
        J = self.J
        if J is None:   # guess: scene moves opposite to the camera, ~2 px/mm at 640 px
            k = size[0] / 320.0
            return -du / (2.0 * k), dv / (2.0 * k)
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        if abs(det) < 1e-3:
            self.J = J = self._default_J(size)
            det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        return ((J[1][1] * du - J[0][1] * dv) / det, (-J[1][0] * du + J[0][0] * dv) / det)

    def _update_J(self, d, dp):
        if self.J is None or (d[0] ** 2 + d[1] ** 2) < 25:
            return
        J = self.J
        pred = (J[0][0] * d[0] + J[0][1] * d[1], J[1][0] * d[0] + J[1][1] * d[1])
        n = d[0] ** 2 + d[1] ** 2
        for i in range(2):
            e = (dp[i] - pred[i]) * 0.8
            J[i][0] += e * d[0] / n
            J[i][1] += e * d[1] / n

    def _show(self, seen):
        if seen.get("jaws"):
            self._jaws = seen["jaws"]["px"]
        self._set(target=seen.get("target"),
                  gripper={"px": self._jaws, "bbox": None} if self._jaws else None)

    # ------------------------------------------------------------ claw-game pickup
    CLAW_PITCHES = (90, 85, 80, 75, 70, 65, 60, 55)   # straight down preferred
    CENTER_TOL = 0.05                            # centred when within 5 % of the image width
    DESCEND_STEP = 70

    def _claw_pose(self, x, y, z_pref, z_min=-400):
        """Reachable (z, pitch) near z_pref with the gripper pointing (almost) straight down."""
        roll = self._target()[4]
        z = z_pref
        while z >= z_min:
            for pitch in self.CLAW_PITCHES:
                if self.c.arm.reachable([x, y, z, pitch, roll, 0]):
                    return z, pitch
            z -= 15
        return None

    def _pick_claw_pitch(self, x, y, hover, bottom):
        """Steepest tilt that works both at the hover height and at the bottom here."""
        roll = self._target()[4]
        for pitch in self.CLAW_PITCHES:
            for h in (hover, hover - 30, hover - 60, hover - 90):
                if h <= bottom + 30:
                    break
                if all(self.c.arm.reachable([x, y, z, pitch, roll, 0]) for z in (h, (h + bottom) / 2, bottom + 5)):
                    return pitch, h
        return None, hover

    def _claw_slide(self, dt, dr):
        """Sideways move at the same height with the FIXED claw tilt (shrinks the step at the edge)."""
        p = self._target()
        r, t = self._axes(p)
        dt = max(-STEP_MM, min(STEP_MM, dt))
        dr = max(-STEP_MM, min(STEP_MM, dr))
        def done(goal):
            mx, my = goal[0] - p[0], goal[1] - p[1]
            return (mx * t[0] + my * t[1], mx * r[0] + my * r[1])
        x = p[0] + dt * t[0] + dr * r[0]
        y = p[1] + dt * t[1] + dr * r[1]
        ok = lambda z, pitch: self.c.arm.reachable([x, y, z, pitch, p[4], p[5]])
        if ok(p[2], self._pitch):
            return done(self._go(x=x, y=y, pitch=self._pitch))
        # 1) same tilt, a bit lower: reaching further out works better closer to the table
        floor = float(self.c.arm.cfg.get("arm_z_min_mm", -120))
        z = p[2] - 30
        while z > floor + 60:
            if ok(z, self._pitch):
                self._log("limit", f"lowering to z {z:.0f} mm to reach further with the same tilt")
                return done(self._go(x=x, y=y, z=z, pitch=self._pitch))
            z -= 30
        # 2) tilt a little less steeply (the view changes, so re-learn it)
        for pitch in self.CLAW_PITCHES:
            if pitch < self._pitch and ok(p[2], pitch) and ok(floor + 40, pitch):
                self._log("limit", f"tilting to {pitch}° to reach further")
                self._pitch = pitch          # keep the learned view; each photo keeps refining it
                return done(self._go(x=x, y=y, pitch=pitch))
        # 3) part of the step at least
        for f in (0.5, 0.25):
            x2 = p[0] + f * (dt * t[0] + dr * r[0])
            y2 = p[1] + f * (dt * t[1] + dr * r[1])
            if self.c.arm.reachable([x2, y2, p[2], self._pitch, p[4], p[5]]):
                return done(self._go(x=x2, y=y2, pitch=self._pitch))
        self._log("limit", "edge of the arm's reach here - drive the rover closer")
        return (0.0, 0.0)

    def _claw_move(self, x, y, z, **kw):
        found = self._claw_pose(x, y, z, z - 200)
        if found is None:
            raise PilotStopped(f"Can't point straight down at x {x:.0f}, y {y:.0f} - the object is out of "
                               "the arm's top-down reach. Drive the rover closer.")
        z2, pitch = found
        self._go(x=x, y=y, z=z2, pitch=pitch, **kw)
        return z2, pitch

    def _box_down(self, goal):
        """Qwen: box around the target in the downward-looking gripper camera."""
        jpeg, size = self._frame()
        prompt = (f"Camera on a robot gripper looking DOWN. Goal: {goal}. "
                  "Box the object to pick up (if several fit, the one nearest the image centre). "
                  "Only box it if you can CLEARLY see that object - if you are not sure, answer null. "
                  "{COORDS} Answer compact JSON only: "
                  '{"target":[x1,y1,x2,y2] or null}')
        raw, seen, scale = self.c.ai.ask(prompt, jpeg, size, max_tokens=40, max_width=IMG_WIDTH)
        try:
            box, centre = self.c.ai.to_camera(extract_json(raw).get("target"), seen, scale, size)
        except (ValueError, TypeError, AttributeError):
            return None, size
        return {"bbox": box, "px": list(centre)}, size

    def _world_of(self, tgt, size, p):
        """Table position (arm x, y) of a detection, from the current pose and the learned view."""
        cp = self._aim_point(tgt, size)
        dt, dr = self._solve(cp[0] - tgt["px"][0], cp[1] - tgt["px"][1], size)
        r_ax, t_ax = self._axes(p)
        return (p[0] + dt * t_ax[0] + dr * r_ax[0], p[1] + dt * t_ax[1] + dr * r_ax[1])

    def _median_est(self):
        e = self._ests
        return (sorted(a[0] for a in e)[len(e) // 2], sorted(a[1] for a in e)[len(e) // 2])

    def _aim_point(self, tgt, size):
        """Where the object should sit in the picture: the middle while it looks small
        (far away), moving to the claw point as it fills the view (close)."""
        cp = self._claw_point(size)
        if tgt is None:
            return cp
        b = tgt["bbox"]
        frac = max(b[2] - b[0], (b[3] - b[1]) * size[0] / size[1]) / size[0]
        k = max(0.0, min(1.0, (frac - 0.15) / max(0.05, self.HUGE - 0.15)))
        return [size[0] / 2 + (cp[0] - size[0] / 2) * k, size[1] / 2 + (cp[1] - size[1] / 2) * k]

    def _claw_point(self, size):
        f = self.c.config.get("claw_point") or [0.5, 0.5]
        return [f[0] * size[0], f[1] * size[1]]

    def _holding_check(self, goal):
        jpeg, size = self._frame()
        prompt = (f"Camera on a robot gripper that just tried to pick something up. Goal: {goal}. "
                  "Is the object held between the gripper's fingers now? Answer compact JSON only: "
                  '{"holding": true or false}')
        try:
            raw, _, _ = self.c.ai.ask(prompt, jpeg, size, max_tokens=20, max_width=IMG_WIDTH)
            return bool(extract_json(raw).get("holding"))
        except Exception:
            return None

    def _center(self, goal, rounds, history):
        """Slide sideways (no height change) until the target is under the jaws.
        One Qwen call per step; each photo also teaches how the view moves."""
        lost, pending = 0, None
        for _ in range(rounds):
            self._check()
            tgt, size = self._box_down(goal)
            cp = self._claw_point(size)
            self._set(target=tgt, gripper={"px": cp, "bbox": None})
            if pending and tgt is not None:
                (d, before) = pending
                self._update_J(d, (tgt["px"][0] - before[0], tgt["px"][1] - before[1]))
            pending = None
            if tgt is None:
                lost += 1
                if lost >= 2:
                    return False
                p = self._target()
                if self.c.arm.reachable([p[0], p[1], p[2] + 30, self._pitch, p[4], p[5]]):
                    self._go(z=p[2] + 30, pitch=self._pitch)
                self._log("center", "target not in view → a bit higher for a wider view")
                continue
            lost = 0
            du, dv = cp[0] - tgt["px"][0], cp[1] - tgt["px"][1]
            err = math.hypot(du, dv) / size[0]
            if err < self.CENTER_TOL:
                self._log("center", f"centred (off by {err * 100:.1f}%)")
                return True
            if not self._probed:
                self._probed = True
                if not self._probe(goal, history, {"target": tgt}):
                    self.J = self._default_J(size)
                continue
            dt, dr = self._solve(du * GAIN, dv * GAIN, size)
            d = self._claw_slide(dt, dr)
            pending = (d, tgt["px"])
            self._log("center", f"off by {err * 100:.0f}% → slide right {d[0]:+.0f}, out {d[1]:+.0f} mm")
        return False

    def _recover(self):
        """After a failed move: servo torque back on, short pause."""
        try:
            self.c.arm.wake()
        except Exception:
            pass
        try:
            self.c.tasks.cancel("retry")
        except Exception:
            pass
        self._wait(0.8)

    def _identify_held(self, fallback=""):
        """Photo right after the lift: ask Qwen (in the background) what is in the jaws."""
        try:
            jpeg, size = self.c._frame()
        except Exception:
            return
        self._set(held={"label": fallback or "?", "checking": True})

        def ask():
            try:
                raw, _, _ = self.c.ai.ask(
                    "Photo from a camera on a robot gripper that has just lifted something. The gripper "
                    "jaws are at the bottom-centre of the image. What object is held in the jaws? "
                    "Answer with 1-4 words (e.g. 'white tissue'), or exactly 'nothing'.",
                    jpeg, size, max_tokens=12, max_width=448)
                ans = re.sub(r"[^\w\s'-]", "", raw).strip()[:40] or "?"
                self._set(held={"label": ans, "checking": False, "t": time.time()})
                self._log("held", f"in the gripper: {ans}")
            except Exception:
                self._set(held={"label": fallback or "?", "checking": False})
        threading.Thread(target=ask, daemon=True).start()

    # ------------------------------------------------------------ grab feedback (gripper camera)
    def _run_task(self, body):
        """Submit one arm task and wait for it. -> (ok, message)"""
        end = time.time() + 5
        while self.c.tasks.busy() and time.time() < end:
            self._wait(0.05)
        try:
            task = self.c.tasks.submit(body, "qwen")
        except ValueError as exc:
            return False, str(exc)
        while True:
            self._wait(0.05)
            cur = self.c.tasks.status()
            if cur.get("id") != task["id"] or cur["state"] != "running":
                break
        if cur.get("slow"):
            self._log("slow", f"{body.get('task')}: slow step(s): {', '.join(cur['slow'])}")
        return cur["state"] != "failed", cur.get("message", "")

    @staticmethod
    def _warp(tpl, A):
        """Turn the look-pose template the way the camera turned (A: 2x2 image map), keep the middle."""
        import cv2
        import numpy as np
        h, w = tpl.shape[:2]
        c = np.array([w / 2.0, h / 2.0])
        A = np.array(A, float)
        det = abs(np.linalg.det(A)) ** 0.5 or 1.0
        A = A / det                                        # rotation/mirror only; scale is searched
        M = np.hstack([A, (c - A @ c).reshape(2, 1)])
        out = cv2.warpAffine(tpl, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = int(0.15 * min(w, h))
        return out[m:h - m, m:w - m] if min(w, h) - 2 * m >= 8 else out

    def _hover_find(self, jpeg, size, tpl, near=None, scale=None, A=None, radius=30):
        """The item (template cut from the look-pose photo) seen from the hover pose, where it
        looks 1.3-3x bigger and turned with the base. -> (score, [u, v] camera px, scale) or None."""
        from . import tracker as live
        gray = live.decode(jpeg)
        if gray is None or tpl is None:
            return None
        if A is not None:
            tpl = self._warp(tpl, A)
        f = size[0] / gray.shape[1]
        if near is None:
            sc, u, v, s = live.match(gray, tpl, scales=(1.3, 1.45, 1.6, 1.8, 2.0, 2.25, 2.5, 2.8, 3.1))
            if sc < 0.45:
                return None
        else:
            u, v, s = near[0] / f, near[1] / f, scale
            sc = -1.0
        sc2, u2, v2, s2 = live.match(gray, tpl, (u, v), 18 if near is None else radius / f,
                                     scales=(s * 0.9, s, s * 1.1))
        if sc2 > sc:
            sc, u, v, s = sc2, u2, v2, s2
        return sc, [u * f, v * f], s

    def _look_J(self, o):
        """mm per camera px at the look pose, around the item (from the calibration)."""
        cal = self.c.calibration
        u, v = o["pixel"]
        a0, a1, a2 = cal.pixel_to_arm(u, v), cal.pixel_to_arm(u + 5, v), cal.pixel_to_arm(u, v + 5)
        return [[(a1[0] - a0[0]) / 5, (a2[0] - a0[0]) / 5], [(a1[1] - a0[1]) / 5, (a2[1] - a0[1]) / 5]]

    def _turn(self, xy):
        lp = self.c.calibration.look_pose or [1, 0]
        return math.atan2(xy[1], xy[0]) - math.atan2(lp[1], lp[0])

    def _hover_view_map(self, o, xy):
        """2x2 map: look-pose image offsets -> how they appear at the hover pose (turned with the base)."""
        import numpy as np
        J = np.array(self._look_J(o), float)
        d = -self._turn(xy)
        R = np.array([[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]])
        return (np.linalg.inv(J) @ R @ J).tolist()

    def _hover_mm_per_px(self, o, xy, s):
        """2x2 matrix: camera-px offset seen at the hover pose -> arm mm (from the look-pose
        calibration, zoomed by the template scale and turned by the base-angle change)."""
        J = self._look_J(o)
        d = self._turn(xy)
        c, sn = math.cos(d), math.sin(d)
        R = [[c, -sn], [sn, c]]
        return [[sum(R[i][k] * J[k][j] for k in range(2)) / s for j in range(2)] for i in range(2)]

    def _grip_roll(self, o, xy):
        """Wrist roll (deg) so the jaws close across the item's SHORT side (oval fruit, long boxes).
        Uses the look-pose photo; 0 for round items. Config: jaw_axis_deg (jaw closing direction at
        roll 0, relative to the arm's reach direction: 90 = sideways), roll_sign (+1 / -1). Checked on the real arm: at roll 0 the jaws close ALONG the reach direction (0)."""
        cfg = self.c.config
        if not cfg.get("align_roll", True) or not o.get("orient"):
            return 0.0
        ang_img, ratio = o["orient"]
        if ratio < float(cfg.get("roll_min_ratio", 1.3)):
            return 0.0                                          # round enough: any angle works
        cal = self.c.calibration
        u, v = o["pixel"]
        a0 = cal.pixel_to_arm(u, v)
        a1 = cal.pixel_to_arm(u + 10 * math.cos(math.radians(ang_img)), v + 10 * math.sin(math.radians(ang_img)))
        long_axis = math.degrees(math.atan2(a1[1] - a0[1], a1[0] - a0[0]))    # in the arm's x/y frame
        heading = math.degrees(math.atan2(xy[1], xy[0]))
        want_close = long_axis + 90.0                                         # squeeze across the short side
        roll = float(cfg.get("roll_sign", 1)) * (want_close - heading - float(cfg.get("jaw_axis_deg", 0)))
        roll = (roll + 90.0) % 180.0 - 90.0                                   # jaws are symmetric: ±90
        return round(roll, 1)

    def _feedback_pick(self, o, tpl, xy, verify=True):
        """Pick with the gripper camera watching:
          1. hover above the item and find it in the close-up (template from the look pose)
          2. live align: if the item is off the spot where successful grabs saw it, nudge the arm
             (checked: the error must shrink, else the nudge is undone)
          3. grab, lift, and look again from the same hover pose: item STILL lying there
             -> empty grab, reported at once (no pointless trip to the paper)
        -> ("ok" | "empty" | "fail", message, xy used)"""
        self._pending_hover = None
        label = o.get("label", "item")
        cfg = self.c.config
        ok, msg = self._run_task({"task": "move_above", "xy": list(xy)})
        if not ok:
            return "fail", msg, xy
        size = self.c.image_size()
        det = None
        pitch = None
        try:
            table = self.c.calibration.table_z
            h = self.c.tasks.cfg
            pitch = self.c.tasks._pick_pitch(xy[0], xy[1], [table + h["hover_height_mm"],
                                                           table + h["grasp_height_mm"]])
        except Exception:
            pass
        A = None
        if tpl is not None:
            try:
                A = self._hover_view_map(o, xy)
            except Exception:
                A = None
            jpeg, size = self._frame()
            det = self._hover_find(jpeg, size, tpl, A=A)
            if det and det[0] < 0.55:
                det = None
        ref = cfg.get("hover_ref") if self._hover_valid() else None
        if det and ref and pitch == 90 and cfg.get("live_align", True):
            err = [det[1][0] - ref[0], det[1][1] - ref[1]]
            e0 = math.hypot(*err)
            if e0 > 0.03 * size[0]:
                try:
                    M = self._hover_mm_per_px(o, xy, det[2])
                    corr = [M[0][0] * err[0] + M[0][1] * err[1], M[1][0] * err[0] + M[1][1] * err[1]]
                except Exception:
                    corr = None
                if corr and 2 < math.hypot(*corr) <= float(cfg.get("live_align_max_mm", 30)):
                    new = [xy[0] + corr[0], xy[1] + corr[1]]
                    ok, _ = self._run_task({"task": "move_above", "xy": new})
                    if ok:
                        jpeg, size = self._frame()
                        det2 = self._hover_find(jpeg, size, tpl, ref, det[2], A=A, radius=max(60, 0.8 * e0))
                        e1 = math.hypot(det2[1][0] - ref[0], det2[1][1] - ref[1]) if det2 and det2[0] > 0.5 else None
                        if e1 is not None and e1 < 0.7 * e0:
                            self._log("align", f"live align: {label} was {e0:.0f}px off → nudged "
                                               f"{corr[0]:+.0f},{corr[1]:+.0f} mm → {e1:.0f}px off ✓")
                            xy, det = new, det2
                        else:
                            self._log("align", f"live align didn't help ({e0:.0f}px → "
                                               f"{'lost' if e1 is None else f'{e1:.0f}px'}) - using the first aim")
        roll = 0.0
        try:
            roll = self._grip_roll(o, xy)
        except Exception:
            roll = 0.0
        if abs(roll) >= 8:
            self._log("grab", f"{label} is oval (x{o['orient'][1]:.1f}) - turning the gripper {roll:+.0f}° "
                              "to grab it across its short side")
            det = None                  # the camera turns with the wrist: close-up checks no longer line up
        else:
            roll = 0.0
        ok, msg = self._run_task({"task": "pick", "xy": list(xy), "roll": roll, "no_lift": not verify})
        if not ok:
            return "fail", msg, xy
        if not verify:                   # collect mode: jaws shut -> straight to the container, no checks
            return "ok", "", xy
        bz, tz = getattr(self.c.tasks, "bottom_z", None), self.c.calibration.table_z
        if bz is not None and tz is not None:
            self._log("grab", f"jaws closed at {bz - tz:.0f} mm above the table"
                              + (" (stopped on something)" if self.c.tasks.touched_z is not None else ""))
        # --- did we get it?
        grip = (self.c.arm.state().get("pose") or [0] * 6)[5]
        if grip > float(cfg.get("held_grip_deg", 6)):
            verdict = "ok"                 # jaws stopped on something
        elif verify and det and grip < float(cfg.get("empty_grip_deg", 2.0)):
            jpeg, size = self._frame()
            again = self._hover_find(jpeg, size, tpl, det[1], det[2], A=A, radius=12)
            if again and again[0] >= float(cfg.get("empty_grab_score", 0.72)):
                self._log("check", f"EMPTY GRAB: {label} is still lying there (match {again[0]:.2f})")
                return "empty", "empty grab", xy
            verdict = "ok"
        else:
            verdict = "ok"                 # can't see - trust it; the look-pose check will tell
        self._pending_hover = [round(det[1][0], 1), round(det[1][1], 1)] if det and pitch == 90 else None
        if verdict == "ok":
            self._identify_held(label)
        return verdict, "", xy

    def _hover_valid(self):
        """Learned hover spot only counts for the camera set-up it was learned with."""
        lp = self.c.calibration.look_pose or []
        hl = self.c.config.get("hover_look") or []
        return len(lp) >= 3 and len(hl) >= 3 and all(abs(a - b) < 3 for a, b in zip(lp[:5], hl[:5]))

    def _learn_hover(self):
        """The grab was confirmed from the look pose: remember where the hover close-up saw it."""
        p, self._pending_hover = getattr(self, "_pending_hover", None), None
        if not p:
            return False
        cfg = self.c.config
        if not self._hover_valid():            # new calibration / camera moved: start learning again
            cfg["hover_hist"], cfg["hover_ref"] = [], None
            cfg["hover_look"] = list(self.c.calibration.look_pose or [])
            self.c._save_config(hover_look=cfg["hover_look"])
        hist = (cfg.get("hover_hist") or [])[-6:] + [p]
        cfg["hover_hist"] = hist
        if len(hist) >= 2:
            us, vs = sorted(q[0] for q in hist), sorted(q[1] for q in hist)
            cfg["hover_ref"] = [us[len(hist) // 2], vs[len(hist) // 2]]
        self.c._save_config(hover_hist=hist, hover_ref=cfg.get("hover_ref"))
        if len(hist) == 2:
            self._log("align", "learned where a good grab sees the item - live align is now on")
            return True
        return False

    # ------------------------------------------------------------ sort into the yellow-tape zone
    @staticmethod
    def find_zone(jpeg):
        """Yellow tape drop-off zone in the image -> [x1, y1, x2, y2] or None (colour, no AI)."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 90, 90), (36, 255, 255))           # yellow
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None
        h, w = mask.shape
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.002 * w * h or bw < 0.08 * w and bh < 0.08 * h:
                continue                                                 # specks, small yellow things
            # tape outline: big box, thin line (low fill) - prefer the biggest outline
            score = bw * bh
            if best is None or score > best[0]:
                best = (score, [float(x), float(y), float(x + bw), float(y + bh)])
        return best[1] if best else None

    @staticmethod
    def find_triangle(jpeg):
        """Big BLACK tape triangle (outline or filled) -> {"bbox", "poly", "center"} or None."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        thr = min(90, int(np.percentile(v, 50) * 0.45) + 10)          # dark relative to the scene
        mask = ((v < thr) & (hsv[:, :, 1] < 120)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            hull = cv2.convexHull(c)
            area = cv2.contourArea(hull)
            if area < 0.04 * w * h:                                     # must be BIG
                continue
            tri = cv2.approxPolyDP(hull, 0.06 * cv2.arcLength(hull, True), True)
            if len(tri) != 3:
                continue
            # a triangle fills ~all of its own hull; long cables/phones don't give 3 clean corners
            if best is None or area > best[0]:
                best = (area, tri.reshape(3, 2).astype(float))
        if best is None:
            return None
        poly = best[1]
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        x1, y1 = poly.min(axis=0)
        x2, y2 = poly.max(axis=0)
        return {"bbox": [float(x1), float(y1), float(x2), float(y2)],
                "poly": [[float(a), float(b)] for a, b in poly], "center": [float(cx), float(cy)]}

    def _spot_xy(self, name):
        """Clicked drop spot for this name ('coin', 'paper', ...) -> [x, y] arm mm or None."""
        spots = dict(self.c.config.get("drop_spots") or {})
        if not spots and self.c.config.get("drop_xy"):
            spots = {"coin": self.c.config["drop_xy"]}
        n = str(name or "").lower().strip()
        for k, v in spots.items():
            k2 = k.lower().strip()
            if v and (k2 == n or k2 in n or n in k2 or (set(k2.split()) & set(n.split()))):
                return v
        if len(spots) == 1 and not name:
            return list(spots.values())[0]
        return None

    @staticmethod
    def _stems(text):
        out = set()
        for w in re.findall(r"[a-z]+", str(text).lower()):
            out.add(w)
            if w.endswith("es") and len(w) > 4:
                out.add(w[:-2])
            if w.endswith("s") and len(w) > 3:
                out.add(w[:-1])
        return {w for w in out if len(w) >= 3 and w not in ("the", "and", "all", "some", "every", "lying", "on")}

    def _parse_rules(self, goal):
        """'put the prunes on the coin and the tissues on the paper' -> [{what, dest}, ...]"""
        text = re.sub(r"[-_]+", " ", str(goal)).lower()
        rules = []
        for part in re.split(r"\s*(?:,|;|\band\b|\bthen\b|\balso\b)\s*", text):
            m = re.search(r"(?:put|place|move|sort|drop|take|bring|stack|pile)?\s*(?:all\s+)?(?:of\s+)?"
                          r"(?:the\s+)?([a-z][a-z ]*?)\s+(?:on|onto|in|into|to|at|next to)\s+"
                          r"(?:the\s+|a\s+)?([a-z][a-z ]*?)\s*$", part.strip())
            if not m:
                continue
            what, dest = m.group(1).strip(), m.group(2).strip()
            if re.fullmatch(r"(everything|every ?thing|items?|things|objects|stuff|it|them|rest|the rest|all)( else)?|anything else|all the rest", what):
                what = None
            rules.append({"what": what, "dest": dest})
        return rules

    def _rule_for(self, label, rules):
        stems = self._stems(label)
        for r in rules:
            if r["what"] and self._stems(r["what"]) & stems:
                return r
        return next((r for r in rules if not r["what"]), None)

    def _locate_zone(self, jpeg, size, name=None):
        """Drop zone by name: a clicked drop spot, black triangle by shape, yellow by colour, else ask Qwen."""
        dxy = self._spot_xy(name) or getattr(self, "_zone_cache", {}).get(str(name))
        if dxy and self.c.calibration.ready:
            cal = self.c.calibration
            c = cal.arm_to_pixel(*dxy)
            rad = 60 if re.search(r"paper|sheet|mat|box|bin|tray", str(name or ""), re.I) else 35
            r = max(12.0, math.dist(c, cal.arm_to_pixel(dxy[0] + rad, dxy[1])))     # rad mm around it
            poly = [[c[0] + r * math.cos(k * math.pi / 4), c[1] + r * math.sin(k * math.pi / 4)] for k in range(8)]
            return {"bbox": [c[0] - r, c[1] - r, c[0] + r, c[1] + r], "poly": poly, "center": list(c),
                    "xy": list(dxy)}
        kind = str(name or self.c.config.get("drop_zone", "black triangle")).lower()
        if "triangle" in kind:
            z = self.find_triangle(jpeg)
            if z:
                return z
        if "yellow" in kind:
            b = self.find_zone(jpeg)
            if b:
                return {"bbox": b, "poly": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
                        "center": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]}
        found = []
        for q in (f"the {kind} (the drop-off spot on the floor/table)",
                  {"coin": "a small round flat metal coin lying on the table (silver or gold coloured, "
                           "maybe partly covered)",
                   "paper": "a flat sheet of paper lying on the table"}.get(kind.split()[-1] if kind else "", "")):
            if not q:
                continue
            try:                                                         # backup: Qwen finds it
                found = self.c.ai.locate(jpeg, size, q).get("objects", [])
            except Exception:
                found = []
            if found:
                break
        if found:
            b = found[0]["bbox"]
            z = {"bbox": b, "poly": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
                 "center": list(found[0]["pixel"])}
            try:                                   # remember it (arm mm): no more Qwen calls for it this run
                xy = [round(v, 1) for v in self.c.calibration.pixel_to_arm(*z["center"])]
                if not isinstance(getattr(self, "_zone_cache", None), dict):
                    self._zone_cache = {}
                self._zone_cache[str(name)] = xy
                self._log("sort", f"found the {kind} at arm {xy[0]:.0f}, {xy[1]:.0f} mm (remembered for this run - "
                                  f"click it with 📍 Drop spot to save it for good)")
                return self._locate_zone(jpeg, size, name)
            except Exception:
                return z
        return None

    @staticmethod
    def _in_poly(poly, u, v, grow=1.08):
        """Point inside the (slightly enlarged) polygon?"""
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        pts = [[cx + (p[0] - cx) * grow, cy + (p[1] - cy) * grow] for p in poly]
        inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > v) != (yj > v) and u < (xj - xi) * (v - yi) / ((yj - yi) or 1e-9) + xi:
                inside = not inside
            j = i
        return inside

    def _zone_spot(self, zone, k):
        """k-th drop spot: the centre dot first, then around it, pulled 35 % towards each corner."""
        cx, cy = zone["center"]
        poly = zone["poly"]
        if k == 0:
            return [cx, cy]
        p = poly[(k - 1) % len(poly)]
        f = 0.35 if k <= len(poly) else 0.2
        return [cx + (p[0] - cx) * f, cy + (p[1] - cy) * f]

    def _run_sort(self, goal, max_rounds, scan):
        """Look -> find the yellow zone -> Qwen lists loose items ONCE -> pick nearest -> place in zone
        -> back at the look pose the live tracker checks (no AI, ~5 ms) that the item really left its
        spot and that the other items are still where Qwen saw them -> next item.
        Qwen is only asked again when the list is used up (to find anything missed / confirm done)."""
        from . import tracker as live
        result, moved, rounds = "Stopped", 0, 10 ** 6       # runs until done or STOP
        failures = []                     # spots where a pick failed: skip after two tries
        cache = []                        # [(object, template)] from the last Qwen look, not yet moved
        last = None                       # (object, template, xy) of the item just moved
        blind = None                      # xy of a just-moved item we could not track (plain texture)
        empties = []                      # xy of empty grabs (retried once live align has learned)
        unreach = []                      # xy the arm physically cannot reach (move those closer)
        empty_looks = 0                   # Qwen must see an empty table twice before "done"
        qwen_calls, reused = 0, 0
        self._claw_mode = False
        # "put the prunes on the coin and the tissues on the paper" -> one rule per item type
        self._zone_cache = {}
        rules = self._parse_rules(goal)
        if not rules:
            rules = [{"what": None, "dest": str(self.c.config.get("drop_zone", "coin"))}]
        dests = list(dict.fromkeys(r["dest"] for r in rules))
        whats = [r["what"] for r in rules if r["what"]]
        if whats and all(r["what"] for r in rules):
            query = ("every " + " and every ".join(whats) + " lying on the surface. Label each one with "
                     "exactly one of: " + ", ".join(f"'{w}'" for w in whats))
        else:
            query = ("every loose object lying on the surface that could be picked up (trash, cups, paper, "
                     "packaging, small items)")
        query += (" - NOT the " + ", NOT the ".join(dests) + " (those are the drop-off spots), NOT tape, "
                  "NOT cables or wires, NOT the robot")
        self._log("sort", "plan: " + "; ".join(f"{r['what'] or 'everything'} → {r['dest']}" for r in rules))
        still = float(self.c.config.get("still_there_score", 0.68))
        try:
            for rnd in range(1, rounds + 1):
                try:
                    self._check()
                    self._set(step=rnd, state="looking")
                    self.c.go_look()
                    jpeg, size = self._frame()
                    zones, missing = {}, []
                    with ThreadPoolExecutor(max_workers=max(1, len(dests))) as ex:
                        found_z = list(ex.map(lambda d: self._locate_zone(jpeg, size, d), dests))
                    for d, z in zip(dests, found_z):
                        if z is None:
                            missing.append(d)
                        else:
                            zones[d] = z
                    if missing:
                        msg = (f"WAITING: can't see the {' / '.join(missing)}. Click 📍 Drop spot above the camera, "
                               f"type '{missing[0]}' in the little box, then click it in the image "
                               "(works while this keeps running)")
                        self._set(state=f"click the {missing[0]}", say=msg)
                        if rnd % 3 == 1:
                            self._log("sort", msg)
                        self._wait(2.0)
                        continue
                    zone = zones[dests[0]]
                    self._set(zone=zone["bbox"], zone_poly=zone["poly"],
                              zones=[{"name": d, "poly": z["poly"]} for d, z in zones.items()])
                    t0 = time.time()
                    # 1. feedback: did the last item really leave its spot?
                    if last is not None:
                        o, tpl, xy = last
                        last = None
                        sc, px, bb = (live.find_near(jpeg, size, tpl, o["bbox"], radius_boxes=0.35)
                                      if tpl is not None else (-1, None, None))
                        if px and sc >= still and not self._in_poly(zones.get(o.get("dest"), zone)["poly"], *px):
                            moved -= 1
                            failures.append(xy)
                            self._log("check", f"{o.get('label', 'item')} is STILL THERE (match {sc:.2f}) - "
                                               "the grab missed, trying it again")
                            cache.insert(0, (dict(o, pixel=px, bbox=bb), tpl))
                            self._pending_hover = None
                        elif tpl is not None:
                            self._log("check", f"{o.get('label', 'item')} is gone from its spot (match {sc:.2f}) ✓")
                            if self._learn_hover() and empties:
                                failures = [f for f in failures if f not in empties]
                                empties = []
                                self._log("align", "giving the items that were missed before another try")
                    # 2. the other items: still where Qwen saw them? (they may have been nudged)
                    keep = []
                    for o, tpl in cache:
                        if tpl is None:
                            keep.append((o, tpl))                 # plain object: trust Qwen's box
                            continue
                        sc, px, bb = live.find_near(jpeg, size, tpl, o["bbox"])
                        if px and sc >= 0.55:
                            keep.append((dict(o, pixel=px, bbox=bb), tpl))
                    cache = keep
                    # 3. only ask Qwen when the list is used up
                    found = []
                    if not cache:
                        found = self.c.ai.locate(jpeg, size, query).get("objects", [])
                        qwen_calls += 1
                        self.c.live.seed(found, jpeg, size, "qwen")
                        for o in found:
                            try:
                                o["orient"] = live.orientation(jpeg, size, o["bbox"])
                            except Exception:
                                o["orient"] = None
                        cache = [(o, live.template_from(jpeg, size, o["bbox"])) for o in found]
                        if blind is not None:           # last item had no texture to track: did Qwen see it again?
                            for o in found:
                                try:
                                    if math.dist(self.c.calibration.pixel_to_arm(*o["pixel"]), blind) < 30:
                                        moved -= 1
                                        failures.append(blind)
                                        self._log("check", f"{o.get('label', 'item')} is still there (Qwen sees it "
                                                           "again) - the grab missed")
                                        break
                                except Exception:
                                    pass
                            blind = None
                        src = f"Qwen {time.time() - t0:.1f}s"
                    else:
                        reused += 1
                        self.c.live.seed([o for o, _ in cache], jpeg, size, "tracker")
                        src = f"tracker {1000 * (time.time() - t0):.0f}ms, no Qwen call"
                    todo, occupied = [], []
                    for o, tpl in cache:
                        u, v = o["pixel"]
                        b = o["bbox"]
                        r = self._rule_for(o.get("label", ""), rules)
                        if r is None:
                            continue                                          # not something to sort
                        o["dest"] = r["dest"]
                        if self._in_poly(zones[r["dest"]]["poly"], u, v):
                            occupied.append((u, v))
                            continue                                          # already where it belongs
                        if any(self._in_poly(z["poly"], u, v, 0.9) for d, z in zones.items() if d != r["dest"]) \
                                and not r["what"]:
                            continue                                          # catch-all: leave other piles alone
                        if (b[2] - b[0]) * (b[3] - b[1]) > 0.5 * size[0] * size[1]:
                            continue                                          # "the whole table" etc.
                        try:
                            xy = self.c.calibration.pixel_to_arm(u, v)
                        except Exception:
                            continue
                        if any(math.dist(f, xy) < 30 for f in unreach):
                            continue                                          # physically out of reach
                        n_fail = sum(1 for f in failures if math.dist(f, xy) < 40)
                        todo.append((n_fail * 1000 + math.hypot(*xy), o, xy, tpl))   # least-tried, nearest first
                    todo.sort(key=lambda t: t[0])                             # nearest first
                    cache = [(t[1], t[3]) for t in todo]
                    self._log("sort", f"round {rnd} ({src}): {len(todo)} item(s) to sort"
                                      + (": " + ", ".join(f"{t[1].get('label', '?')}→{t[1].get('dest')}" for t in todo[:5])
                                         if todo else ""))
                    if not todo:
                        if not src.startswith("Qwen"):
                            cache = []                                        # make Qwen look before deciding
                            continue
                        far = []
                        for o in found:
                            try:
                                if any(math.dist(self.c.calibration.pixel_to_arm(*o["pixel"]), f) < 30 for f in unreach):
                                    far.append(o)
                            except Exception:
                                pass
                        if far:
                            if rnd % 5 == 1:
                                self._log("sort", "only out-of-reach items left - push them closer to the arm, "
                                                  "I keep watching")
                            self._wait(3.0)
                            if rnd % 10 == 0:
                                unreach = []                                  # maybe it was moved: try again
                            continue
                        empty_looks += 1
                        if empty_looks < 2:
                            self._log("sort", "table looks clear - double-checking")
                            continue
                        plan_txt = "; ".join(f"{r['what'] or 'all'} → {r['dest']}" for r in rules)
                        result = (f"Done: sorted {moved} item(s) ({plan_txt})"
                                  f" - {qwen_calls} Qwen look(s), {reused} tracker-only round(s)")
                        break
                    empty_looks = 0
                    _, o, xy, tpl = todo[0]
                    cache = cache[1:]
                    self._set(target={"bbox": o["bbox"], "px": o["pixel"]})
                    zone = zones.get(o.get("dest"), zone)
                    if self.c.config.get("drop_spread", False):     # optional: spread items around the centre
                        gap = 0.12 * max(zone["bbox"][2] - zone["bbox"][0], zone["bbox"][3] - zone["bbox"][1])
                        spot = next((sp for sp in (self._zone_spot(zone, k) for k in range(7))
                                     if all(math.hypot(sp[0] - a, sp[1] - b) > gap for a, b in occupied)),
                                    self._zone_spot(zone, moved))
                    else:
                        spot = list(zone["center"])                 # always the centre of the paper
                    label = o.get("label", "item")
                    self._set(state=f"pick {label}")
                    n_fail = sum(1 for f in failures if math.dist(f, xy) < 40)
                    if n_fail and not (self.c.config.get("hover_ref") and self._hover_valid()):
                        # missed before and live align can't help yet: search around the spot
                        offs = ((0, 0), (12, 0), (-12, 0), (0, 12), (0, -12), (20, 12), (-20, -12), (12, -20), (-12, 20))
                        dx, dy = offs[n_fail % len(offs)]
                        if dx or dy:
                            self._log("sort", f"retry {n_fail + 1} on {o.get('label', 'item')}: aiming {dx:+d},{dy:+d} mm off")
                        xy = (xy[0] + dx, xy[1] + dy)
                    verdict, msg, xy = self._feedback_pick(o, tpl, xy)
                    ok = verdict == "ok"
                    if verdict == "fail":
                        self._log("sort", f"pick {label} failed: {msg[:120]} - will retry")
                        if "reach" in msg:
                            unreach.append(xy)                # out of reach: needs moving closer
                        else:
                            self._recover()
                    elif verdict == "empty":
                        cache.insert(0, (o, tpl))              # still there: retry it next round (no Qwen)
                        empties.append(xy)
                    if ok:
                        self._set(state=f"place {label} on the {o.get('dest')}")
                        where = {"xy": zone["xy"]} if zone.get("xy") else {"pixel": spot}
                        ok, msg = self._run_task({"task": "place", **where, "no_lift": True,
                                                  "release_height_mm": float(self.c.config.get("sort_release_height_mm", 50))})
                        if not ok:
                            self._log("sort", f"place failed: {msg[:120]}")
                    self._set(held=None)
                    if ok:
                        moved += 1
                        last = (o, tpl, xy)                          # checked from the look pose next round
                        blind = xy if tpl is None else None
                        self._log("sort", f"moved {o.get('label', 'item')} onto the {o.get('dest')} ({moved} so far)")
                    elif verdict == "empty":                     # a real miss (not an arm error)
                        failures.append(xy)
                        if sum(1 for f in failures if math.dist(f, xy) < 40) % 3 == 0:
                            cache = []                           # keeps missing: fresh look with Qwen
                            self._log("sort", f"{label} missed 3x - asking Qwen to look again, then retrying")
                    if not ok and self.c.tasks.holding:
                        self._run_task({"task": "gripper", "deg": self.c.tasks.cfg["gripper_open_deg"]})
                except PilotStopped:
                    raise
                except Exception as exc:                      # never give up: log, wake the arm, go on
                    self._log("retry", f"{type(exc).__name__}: {str(exc)[:120]} - carrying on")
                    self._recover()
            else:
                result = f"Stopped after {rounds} rounds ({moved} sorted)"
        except PilotStopped as exc:
            result = str(exc)
        except Exception as exc:
            result = f"Error: {type(exc).__name__}: {exc}"
            self._log("error", result)
            try:
                self.c.tasks.cancel("autopilot error")
                self.c.arm.hold()
            except Exception:
                pass
        finally:
            if self._stop.is_set() or self.c.safety.state()["latched"]:
                self.c.tasks.cancel("Stopped")
            self._set(running=False, state="finished", result=result)
            self._log("end", result)
            self.c.safety.record("qwen", "autopilot", result)

    # ------------------------------------------------------------ collect into the rover's container
    def _collect_query(self, goal):
        """'pick up all the prunes and tissues' -> (query for Qwen, [names] or None = anything)."""
        text = re.sub(r"[-_]+", " ", str(goal)).lower().strip()
        m = re.search(r"(?:pick\s*up|collect|grab|gather|clean\s*up|tidy\s*up|tidy|clear|put|move|take|get|sort)\s+"
                      r"(?:out\s+|up\s+)?(?:all\s+)?(?:of\s+)?(?:the\s+)?(.+?)"
                      r"(?:\s+((?:in|inside|into|on|onto|to|from|off|out of)\b.*))?$", text)
        what = (m.group(1) if m else text).strip()
        where = (m.group(2) or "") if m else ""
        where = "" if re.search(r"\b(container|rover)\b", where) else where   # that's the destination
        names = [w.strip() for w in re.split(r"\s*(?:,|\band\b|&|\bplus\b)\s*", what) if w.strip()]
        generic = r"(everything|every ?thing|items?|things|objects|stuff|all|trash|rubbish|table|it|them)"
        names = [n for n in names if not re.fullmatch(generic, n)]
        if not names:
            return ("every loose object lying on the surface that could be picked up (trash, food, paper, "
                    "packaging, small items) - NOT tape, NOT cables or wires, NOT the robot"), None
        return ("every " + " and every ".join(names) + (f" ({where})" if where else " lying on the surface")
                + ". Label each one with exactly one "
                "of: " + ", ".join(f"'{n}'" for n in names) + " - NOT tape, NOT cables or wires, NOT the robot"), names

    def _run_collect(self, goal, max_rounds, scan):
        """Look -> Qwen lists the items ONCE -> pick nearest -> drop it into the container on the rover
        (fixed, taught arm pose) -> back at the look pose the tracker checks the item really left its
        spot -> next. Runs until the table is clear (Qwen sees nothing twice) or STOP."""
        from . import tracker as live
        moved, failures, cache, unreach = 0, [], [], []
        last, blind, empty_looks, qwen_calls, reused = None, None, 0, 0, 0
        result = "Stopped"
        self._claw_mode = False
        query, names = self._collect_query(goal)
        still = float(self.c.config.get("still_there_score", 0.68))
        self._log("collect", "collecting " + (", ".join(names) if names else "every loose item")
                  + " into the rover container")
        try:
            rnd = 0
            while True:
                rnd += 1
                try:
                    self._check()
                    self._set(step=rnd, state="looking")
                    self.c.go_look()
                    jpeg, size = self._frame()
                    t0 = time.time()
                    # 1. did the last item really leave its spot?
                    if last is not None:
                        o, tpl, xy = last
                        last = None
                        sc, px, bb = (live.find_near(jpeg, size, tpl, o["bbox"], radius_boxes=0.35)
                                      if tpl is not None else (-1, None, None))
                        if px and sc >= still:
                            moved -= 1
                            failures.append(xy)
                            self._log("check", f"{o.get('label', 'item')} is STILL THERE (match {sc:.2f}) - "
                                               "the grab missed, trying again")
                            cache.insert(0, (dict(o, pixel=px, bbox=bb), tpl))
                            self._pending_hover = None
                        elif tpl is not None:
                            self._log("check", f"{o.get('label', 'item')} is gone from the table ✓")
                            self._learn_hover()
                    # 2. the other items still where Qwen saw them?
                    keep = []
                    for o, tpl in cache:
                        if tpl is None:
                            keep.append((o, tpl))
                            continue
                        sc, px, bb = live.find_near(jpeg, size, tpl, o["bbox"], radius_boxes=0.6)
                        if px and sc >= 0.55:
                            keep.append((dict(o, pixel=px, bbox=bb), tpl))
                    cache = keep
                    # 3. Qwen only when the list is used up
                    found = []
                    if not cache:
                        found = self.c.ai.locate(jpeg, size, query).get("objects", [])
                        qwen_calls += 1
                        self.c.live.seed(found, jpeg, size, "qwen")
                        for o in found:
                            try:
                                o["orient"] = live.orientation(jpeg, size, o["bbox"])
                            except Exception:
                                o["orient"] = None
                        cache = [(o, live.template_from(jpeg, size, o["bbox"])) for o in found]
                        if blind is not None:
                            for o in found:
                                try:
                                    if math.dist(self.c.calibration.pixel_to_arm(*o["pixel"]), blind) < 30:
                                        moved -= 1
                                        failures.append(blind)
                                        self._log("check", f"{o.get('label', 'item')} is still there - retrying")
                                        break
                                except Exception:
                                    pass
                            blind = None
                        src = f"Qwen {time.time() - t0:.1f}s"
                    else:
                        reused += 1
                        self.c.live.seed([o for o, _ in cache], jpeg, size, "tracker")
                        src = f"tracker {1000 * (time.time() - t0):.0f}ms, no Qwen call"
                    todo = []
                    for o, tpl in cache:
                        b = o["bbox"]
                        if names and not (self._stems(o.get("label", "")) & set().union(*(self._stems(n) for n in names))):
                            continue                                  # not one of the things asked for
                        if (b[2] - b[0]) * (b[3] - b[1]) > 0.5 * size[0] * size[1]:
                            continue
                        try:
                            xy = self.c.calibration.pixel_to_arm(*o["pixel"])
                        except Exception:
                            continue
                        if any(math.dist(f, xy) < 30 for f in unreach):
                            continue
                        n_fail = sum(1 for f in failures if math.dist(f, xy) < 40)
                        todo.append((n_fail * 1000 + math.hypot(*xy), o, xy, tpl))
                    todo.sort(key=lambda t: t[0])
                    cache = [(t[1], t[3]) for t in todo]
                    self._log("collect", f"round {rnd} ({src}): {len(todo)} item(s) left"
                                         + (": " + ", ".join(t[1].get("label", "?") for t in todo[:6]) if todo else ""))
                    if not todo:
                        if not src.startswith("Qwen"):
                            cache = []
                            continue
                        if unreach and found:
                            if rnd % 5 == 1:
                                self._log("collect", "only out-of-reach items left - push them closer, I keep watching")
                            self._wait(3.0)
                            if rnd % 10 == 0:
                                unreach = []
                            continue
                        empty_looks += 1
                        if empty_looks < 2:
                            self._log("collect", "table looks clear - double-checking")
                            continue
                        result = (f"Done: {moved} item(s) in the rover container - {qwen_calls} Qwen look(s), "
                                  f"{reused} tracker-only round(s)")
                        break
                    empty_looks = 0
                    _, o, xy, tpl = todo[0]
                    cache = cache[1:]
                    label = o.get("label", "item")
                    self._set(target={"bbox": o["bbox"], "px": o["pixel"]}, state=f"pick {label}")
                    n_fail = sum(1 for f in failures if math.dist(f, xy) < 40)
                    if n_fail and not (self.c.config.get("hover_ref") and self._hover_valid()):
                        offs = ((0, 0), (12, 0), (-12, 0), (0, 12), (0, -12), (20, 12), (-20, -12), (12, -20), (-12, 20))
                        dx, dy = offs[n_fail % len(offs)]
                        if dx or dy:
                            self._log("collect", f"retry {n_fail + 1} on {label}: aiming {dx:+d},{dy:+d} mm off")
                        xy = (xy[0] + dx, xy[1] + dy)
                    verdict, msg, xy = self._feedback_pick(o, tpl, xy, verify=False)
                    ok = verdict == "ok"
                    if verdict == "fail":
                        self._log("collect", f"pick {label} failed: {msg[:120]} - will retry")
                        if "reach" in msg:
                            unreach.append(xy)
                        else:
                            self._recover()
                    elif verdict == "empty":
                        cache.insert(0, (o, tpl))
                        failures.append(xy)
                        if sum(1 for f in failures if math.dist(f, xy) < 40) % 3 == 0:
                            cache = []
                            self._log("collect", f"{label} missed 3x - asking Qwen to look again")
                    if verdict != "fail" or "reach" not in msg:
                        # ALWAYS take whatever is in the jaws to the container and let go there -
                        # dark plums/prunes are hard to see in the gripper, so don't trust a "nothing"
                        self._set(state=f"drop {label} in the container")
                        ok2, msg2 = self._run_task({"task": "container"})
                        if not ok2:
                            self._log("collect", f"container drop failed: {msg2[:120]} - retrying it")
                            self._recover()
                            ok2, msg2 = self._run_task({"task": "container"})
                        if not ok2:                              # last try: via the home pose
                            h = self.c.tasks.cfg["home_pose"]                # (keeps the jaws shut)
                            self._run_task({"task": "move", "x": h[0], "y": h[1], "z": h[2], "pitch": h[3]})
                            ok2, msg2 = self._run_task({"task": "container"})
                            if not ok2:
                                self._log("collect", f"could NOT reach the container: {msg2[:140]} - re-teach "
                                                     "'📦 Save container drop here' and press '📦 Test drop'")
                        ok = ok and ok2
                    self._set(held=None)
                    if ok:
                        moved += 1
                        last = (o, tpl, xy)
                        blind = xy if tpl is None else None
                        self._log("collect", f"{label} dropped in the container ({moved} so far)")
                    if not ok and self.c.tasks.holding:
                        self._run_task({"task": "gripper", "deg": self.c.tasks.cfg["gripper_open_deg"]})
                except PilotStopped:
                    raise
                except Exception as exc:
                    self._log("retry", f"{type(exc).__name__}: {str(exc)[:120]} - carrying on")
                    self._recover()
        except PilotStopped as exc:
            result = str(exc)
        finally:
            if self._stop.is_set() or self.c.safety.state()["latched"]:
                self.c.tasks.cancel("Stopped")
            self._set(running=False, state="finished", result=result)
            self._log("end", result)
            self.c.safety.record("qwen", "autopilot", result)

    # ------------------------------------------------------------ plan & pick (point-and-shoot)
    def _run_plan(self, goal, max_rounds, scan):
        """Look from the fixed look pose -> Qwen reasons once (what, where, where to put it)
        -> calibrated pixel->arm mapping -> one smooth pick/place each -> look again."""
        result, picks, rounds = "Stopped", 0, 40
        self._claw_mode = False
        pose = self.c.arm.state().get("pose") or [0] * 6
        if pose[5] > 20:
            self.c.tasks.holding = False                      # jaws are open: not holding anything
        try:
            clean = re.sub(r"[-_]+", " ", goal).strip()
            # a plain "pick up X" needs no planner: one quick "where is X" question (~2 s)
            simple = re.match(r"^(please\s+)?(pick\s*up|grab|take|get|collect|lift)\s+(the\s+|a\s+|an\s+)?(.+)$",
                              clean, re.I)
            if simple and re.search(r"\b(put|place|drop|into|bin|sort|all|then|and)\b", simple.group(4), re.I):
                simple = None                                  # multi-step goal: use the planner
            for rnd in range(1, rounds + 1):
                self._check()
                self._set(step=rnd, state="looking")
                t0 = time.time()
                if simple:
                    if picks > 0:                              # picked it up in THIS run: done
                        result = f"Done ({picks} picked up) - holding it"
                        break
                    found = self.c.ai_locate({"what": simple.group(4)}).get("objects", [])
                    res = {"observation": (f"found {found[0].get('label', simple.group(4))}" if found
                                           else f"no {simple.group(4)} visible"),
                           "steps": [{"action": "pick", "label": found[0].get("label", simple.group(4)),
                                      "bbox": found[0]["bbox"], "pixel": found[0]["pixel"],
                                      "xy": found[0].get("xy")}] if found else []}
                else:
                    res = self.c.ai_plan({"goal": goal})       # goes to the look pose first
                steps = [st for st in res.get("steps", []) if st.get("action") != "done"]
                obs = res.get("observation", "")
                self._set(say=obs)
                self._log("plan", f"round {rnd} ({time.time() - t0:.1f}s): {obs} → "
                                  + (", ".join(f"{st['action']} {st.get('label', '')}".strip() for st in steps)
                                     or "nothing to do"))
                if not steps and picks == 0 and not self.c.tasks.holding:
                    # Qwen saw the scene but planned nothing: ask it directly for the object's box
                    what = re.sub(r"^(please\s+)?(pick\s*up|grab|take|get|collect)\s+(the\s+)?", "",
                                  re.sub(r"[-_]+", " ", goal).strip(), flags=re.I) or goal
                    try:
                        found = self.c.ai_locate({"what": what}).get("objects", [])
                    except Exception as exc:
                        found = []
                        self._log("plan", f"direct look-up failed: {exc}")
                    if found:
                        o = found[0]
                        steps = [{"action": "pick", "label": o.get("label", what), "bbox": o["bbox"],
                                  "pixel": o["pixel"], "xy": o.get("xy")}]
                        self._log("plan", f"Qwen planned nothing, but directly found '{o.get('label', what)}' "
                                          "- picking it")
                if not steps:
                    result = f"Done ({picks} picked up)" + (f": {obs}" if obs else "")
                    break
                failed = False
                for st in steps:
                    self._check()
                    if st.get("bbox"):
                        self._set(target={"bbox": st["bbox"], "px": st.get("pixel")})
                    body = {"task": "home"} if st["action"] == "home" else {"task": st["action"], "pixel": st["pixel"]}
                    self._set(state=f"{st['action']} {st.get('label', '')}".strip())
                    try:
                        task = self.c.tasks.submit(body, "qwen")
                    except ValueError as exc:
                        self._log("plan", f"can't {st['action']} {st.get('label', '')}: {exc}")
                        failed = True
                        break
                    while True:
                        self._wait(0.1)
                        cur = self.c.tasks.status()
                        if cur.get("id") != task["id"] or cur["state"] != "running":
                            break
                    if cur["state"] == "failed":
                        self._log("plan", f"{st['action']} failed: {cur.get('message', '')} - looking again")
                        failed = True
                        break
                    if st["action"] == "pick":
                        picks += 1
                    self._log("plan", f"{st['action']} {st.get('label', '')} done"
                                      + (f" at arm {st['xy']}" if st.get("xy") else ""))
                if failed and rnd == rounds:
                    result = f"Stopped: last step failed ({picks} picked up)"
            else:
                result = f"Stopped after {rounds} rounds ({picks} picked up)"
        except PilotStopped as exc:
            result = str(exc)
        except Exception as exc:
            result = f"Error: {type(exc).__name__}: {exc}"
            self._log("error", result)
            try:
                self.c.tasks.cancel("autopilot error")
                self.c.arm.hold()
            except Exception:
                pass
        finally:
            if self._stop.is_set() or self.c.safety.state()["latched"]:
                self.c.tasks.cancel("Stopped")
            self._set(running=False, state="finished", result=result)
            self._log("end", result)
            self.c.safety.record("qwen", "autopilot", result)

    # ---- remembered view (so later runs skip the test moves)
    def _load_view(self, h):
        v = self.c.config.get("pilot_view")
        if not self.c.config.get("pilot_remember_view", False):
            return None          # off: re-learning each run (2 quick test moves) proved more reliable
        if not v or abs(v.get("pitch", 0) - self._pitch) > 12 or h <= 10:
            return None
        return [[a / h for a in row] for row in v["Jh"]]

    def _save_view(self, h):
        if self.J is None or h <= 10:
            return
        Jh = [[round(a * h, 2) for a in row] for row in self.J]
        self.c.config["pilot_view"] = {"Jh": Jh, "pitch": self._pitch}
        self.c._save_config(pilot_view=self.c.config["pilot_view"])

    def _claw_step(self, dt, dr, dz):
        """Combined sideways + down move at the fixed tilt; falls back to sideways only."""
        p = self._target()
        r, t = self._axes(p)
        dt = max(-STEP_MM, min(STEP_MM, dt))
        dr = max(-STEP_MM, min(STEP_MM, dr))
        x, y = p[0] + dt * t[0] + dr * r[0], p[1] + dt * t[1] + dr * r[1]
        floor = float(self.c.arm.cfg.get("arm_z_min_mm", -120))
        z = max(floor, p[2] + dz)
        if dz and self.c.arm.reachable([x, y, z, self._pitch, p[4], p[5]]):
            goal = self._go(x=x, y=y, z=z, pitch=self._pitch, speed=SPEED if dz < -25 else 0.6)
            mx, my = goal[0] - p[0], goal[1] - p[1]
            return (mx * t[0] + my * t[1], mx * r[0] + my * r[1], goal[2] - p[2])
        d = self._claw_slide(dt, dr)
        return (d[0], d[1], self._target()[2] - p[2])

    def _keep_in_view(self, tgt, size, dt, dr, dz, floor):
        """Shrink (dt, dr, dz) until the target is predicted to stay in the picture.
        Uses the learned image motion for sideways moves and a pinhole zoom model
        (things grow as the camera gets closer) for the drop."""
        if self.J is None:
            return dt, dr, dz, 1.0
        W, H = size
        x1, y1, x2, y2 = tgt["bbox"]
        z = self._target()[2]
        ref = self.oak_xyz[2] if self.oak_xyz else floor
        h = max(30.0, z - ref + 40.0)                 # camera height above the object (approx.)
        cx, cy = W / 2, H / 2
        big = (x2 - x1) > 0.6 * W or (y2 - y1) > 0.6 * H
        m = 0.03 * W

        def fits(fx, fz):
            a, b = dt * fx, dr * fx
            du = self.J[0][0] * a + self.J[0][1] * b
            dv = self.J[1][0] * a + self.J[1][1] * b
            s = h / max(20.0, h + dz * fz)            # zoom factor from dropping (dz < 0)
            bx = [cx + (x1 + du - cx) * s, cx + (x2 + du - cx) * s]
            by = [cy + (y1 + dv - cy) * s, cy + (y2 + dv - cy) * s]
            if big:                                   # fills the view: just keep its middle in the picture
                mx, my = (bx[0] + bx[1]) / 2, (by[0] + by[1]) / 2
                return 0.06 * W < mx < 0.94 * W and 0.06 * H < my < 0.97 * H
            return bx[0] > m and bx[1] < W - m and by[0] > m and by[1] < H - m

        for fx, fz in ((1, 1), (1, 0.6), (1, 0.3), (0.7, 0.3), (1, 0), (0.6, 0), (0.35, 0), (0.2, 0)):
            if fits(fx, fz):
                return dt * fx, dr * fx, dz * fz, min(fx, 1.0 if dz == 0 else fz)
        return dt * 0.2, dr * 0.2, 0.0, 0.0

    BIG = 0.28      # object this wide (fraction of the view): slow down
    HUGE = 0.5      # this wide: it is right under the jaws - final ease-in

    def _run_claw(self, goal, max_steps, scan):
        cfg = self.c.tasks.cfg
        open_deg, closed_deg = cfg["gripper_open_deg"], cfg["gripper_closed_deg"]
        history, result = [], "Stopped"
        self._probed, self.oak_xyz, self._contact, self._frozen = False, None, False, 0
        self._claw_mode = True
        self._scale, self._last_seen, self._stuck = 1.0, None, 0
        self._prev_box, self._last_err = None, None
        self._ests, self._jumps, self._obj_xy, self._rejects = [], 0, None, 0
        try:
            floor = float(self.c.arm.cfg.get("arm_z_min_mm", -120))
            extra = float(self.c.config.get("claw_extra_depth_mm", 0))
            # 1) find it: OAK-D gives x, y, height directly
            xyz = self._oak_find(goal)
            self.oak_xyz = xyz
            p = self._target()
            gx, gy = (xyz[0], xyz[1]) if xyz else (p[0], p[1])
            hover = (xyz[2] + 110) if xyz else max(p[2], 0.0)
            mem = getattr(self.c, "last_object", None)
            if not xyz and mem and time.time() - mem["t"] < 1800 and mem["goal"] == goal:
                gx, gy = mem["xy"]
                hover = max(p[2], 0.0)
                self._ests = [tuple(mem["xy"])] * 2
                self._obj_xy = tuple(mem["xy"])
                self._log("memory", f"remembered this object at x {gx:.0f}, y {gy:.0f} - going straight there")
            self._pitch, hover = self._pick_claw_pitch(gx, gy, hover, floor)
            if self._pitch is None:
                raise PilotStopped("The object is out of the arm's reach from above - drive the rover closer")
            # 2) straight above it, jaws opening on the way
            self._set(state="moving above")
            self._go(x=gx, y=gy, z=hover, pitch=self._pitch, roll=0, grip=open_deg, timeout=6)
            self._log("claw", ("above the OAK-D position" if xyz else "looking down from here")
                      + f" (z {hover:.0f} mm, tilt {self._pitch}°), jaws open")
            self.J = self._load_view(hover - floor + 40)
            self._probed = self.J is not None
            # 3) dive: centre and drop in the same move, fast while small, slow when big
            lost, pending, attempt = 0, None, 1
            for step in range(1, max_steps + 1):
                self._check()
                self._set(step=step)
                tgt, size = self._box_down(goal)
                p = self._target()
                h = p[2] - floor
                # ---- is this really OUR object? (size and position must agree with what we know)
                why_not = None
                if tgt is not None:
                    prev = self._prev_box
                    area = lambda b: max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                    if prev is not None and area(tgt["bbox"]) < 0.35 * area(prev):
                        why_not = "much smaller box than before"
                    elif self.J is not None and len(self._ests) >= 2:
                        wx, wy = self._world_of(tgt, size, p)
                        mx, my = self._median_est()
                        if math.hypot(wx - mx, wy - my) > 80:
                            why_not = f"{math.hypot(wx - mx, wy - my) / 10:.0f} cm away from where the object is"
                if why_not:
                    self._rejects += 1
                    if self._rejects < 4:
                        self._log("dive", f"ignoring that detection ({why_not}) - taking another look")
                        continue
                    self._log("dive", "several detections agree on a new spot - trusting them")
                    self._ests.clear()
                    self._prev_box = None
                self._rejects = 0
                cp = self._aim_point(tgt, size)
                self._set(target=tgt, gripper={"px": cp, "bbox": None})
                if pending and tgt is not None:
                    self._update_J(pending[0], (tgt["px"][0] - pending[1][0], tgt["px"][1] - pending[1][1]))
                pending = None
                if tgt is None:
                    lost += 1
                    seen = self._last_seen
                    if lost == 1 and seen:
                        ox, oy = self._obj_xy if self._obj_xy else (seen[0], seen[1])
                        z_back = max(seen[2], p[2]) + 20
                        if not self.c.arm.reachable([ox, oy, z_back, self._pitch, p[4], p[5]]):
                            ox, oy, z_back = seen[0], seen[1], seen[2]
                        self._go(x=ox, y=oy, z=z_back, pitch=self._pitch)
                        self._scale = max(0.25, self._scale * 0.5)
                        self._log("dive", "lost sight - back above the object's remembered spot, smaller steps")
                        continue
                    if lost <= 2:
                        for up in (50, 30, 15):
                            if self.c.arm.reachable([p[0], p[1], p[2] + up, self._pitch, p[4], p[5]]):
                                self._go(z=p[2] + up, pitch=self._pitch)
                                self._log("dive", f"still not visible - up {up / 10:.0f} cm for a wider view")
                                break
                        continue
                    if lost == 3 and scan and self._scan(goal):
                        continue
                    result = "Lost the object"
                    break
                lost = 0
                self._last_seen = list(p)
                self._prev_box = tgt["bbox"]
                if not self._probed:
                    self._probed = True
                    if not self._probe(goal, history, {"target": tgt}):
                        self.J = self._default_J(size)
                    continue
                bw = max(tgt["bbox"][2] - tgt["bbox"][0], (tgt["bbox"][3] - tgt["bbox"][1]) * size[0] / size[1])
                frac = bw / size[0]
                du, dv = cp[0] - tgt["px"][0], cp[1] - tgt["px"][1]
                err = math.hypot(du, dv) / size[0]
                if self._last_err is not None and err > self._last_err * 1.3 + 0.08:
                    # a trusted detection says the last move made it WORSE: directions are wrong
                    self._log("dive", f"that move went the wrong way (off {self._last_err * 100:.0f}% → "
                                      f"{err * 100:.0f}%) - re-learning the directions with small steps")
                    self._scale = max(0.3, self._scale * 0.5)
                    self.J, self._last_err = None, None
                    self._ests.clear()
                    if not self._probe(goal, history, {"target": tgt}):
                        self.J = self._default_J(size)
                    continue
                self._last_err = err
                # ---- object permanence: its table position, averaged over the good sightings
                self._ests.append(self._world_of(tgt, size, p))
                del self._ests[:-6]
                mx, my = self._median_est()
                self._obj_xy = (mx, my)
                r_ax, t_ax = self._axes(p)
                dx, dy = mx - p[0], my - p[1]
                dt, dr = dx * t_ax[0] + dy * t_ax[1], dx * r_ax[0] + dy * r_ax[1]
                # sideways steps no bigger than what the camera can see at this height
                cam_h = max(40.0, h + 40.0)
                cap = min(70.0, 0.45 * cam_h)
                n = math.hypot(dt, dr)
                if n > cap:
                    dt, dr = dt * cap / n, dr * cap / n
                if frac >= self.HUGE and err < 0.08:
                    break                                   # right under the jaws: ease in
                if frac >= self.BIG and err < 0.06 and self._stuck >= 1:
                    break                                   # centred and close, can't drop safely more: ease in
                if frac < self.BIG:
                    phase = "fast"
                    dz = -max(40.0, 0.5 * h) if err < 0.15 else (-20.0 if err < 0.3 else 0.0)
                else:
                    phase = "slow"
                    dz = -30.0 if err < 0.08 else 0.0
                    dt, dr = dt * 0.8, dr * 0.8
                want_dz = dz
                dt, dr, dz, kept = self._keep_in_view(tgt, size, dt * self._scale, dr * self._scale,
                                                      dz * self._scale, floor)
                self._stuck = (self._stuck + 1) if (want_dz < 0 and dz == 0) else 0
                if kept < 1.0:
                    phase += " (held back to keep it in view)"
                d = self._claw_step(dt, dr, dz)
                pending = ((d[0], d[1]), tgt["px"])
                self._log(phase, f"object {frac * 100:.0f}% of view, off {err * 100:.0f}% → "
                                 f"right {d[0]:+.0f}, out {d[1]:+.0f}, down {-d[2]:.0f} mm")
                if self._contact:
                    break
            else:
                result = f"Stopped after {max_steps} steps"
                return
            if lost:
                return
            view_h = max(20.0, self._target()[2] - floor + 40)
            # 4) ease in: slow straight down until it touches, then close
            self._set(state="easing in")
            p = self._target()
            z = p[2]
            while not self._contact and z > floor + 1:
                nz = max(floor, z - 30)
                if not self.c.arm.reachable([p[0], p[1], nz, self._pitch, p[4], p[5]]):
                    break
                z = nz
                self._go(z=z, pitch=self._pitch, speed=0.45)
            if extra and not self._contact:
                self._go(z=z - extra, pitch=self._pitch, speed=0.3)
            q = self._target()
            self._log("claw", ("touched" if self._contact else "at the bottom") + f" at z {q[2]:.0f} mm - closing")
            self._set(state="grabbing")
            self._go(grip=closed_deg, timeout=2.5)
            self._wait(0.35)
            self._go(z=self._target()[2] + 110, pitch=self._pitch, speed=SPEED)
            if self._obj_xy:
                self.c.last_object = {"xy": list(self._obj_xy), "t": time.time(), "goal": goal}
            held = self._holding_check(goal)
            if held is not False:
                self.c.tasks.holding = True
                self.c.last_object = None          # it is in the gripper now
                result = "Done: picked it up" + ("" if held else " (could not confirm by camera)")
                if held:
                    self._save_view(view_h)
            else:
                result = "Missed - the jaws closed on nothing. Try again"
                self._go(grip=open_deg)
            self._log("done" if held is not False else "claw", result)
        except PilotStopped as exc:
            result = str(exc)
        except Exception as exc:
            result = f"Error: {type(exc).__name__}: {exc}"
            self._log("error", result)
            try:
                self.c.arm.hold()
            except Exception:
                pass
        finally:
            self._set(running=False, state="finished", result=result)
            self._log("end", result)
            self.c.safety.record("qwen", "autopilot", result)

    def _run_claw_old(self, goal, max_steps, scan):
        cfg = self.c.tasks.cfg
        open_deg, closed_deg = cfg["gripper_open_deg"], cfg["gripper_closed_deg"]
        history, result = [], "Stopped"
        self._probed, self.oak_xyz, self._contact, self._frozen = False, None, False, 0
        self._claw_mode = True
        try:
            self.c.arm.wake()
            self._wait(0.3)
            floor = float(self.c.arm.cfg.get("arm_z_min_mm", -120))
            extra = float(self.c.config.get("claw_extra_depth_mm", 0))
            # 1) where is it?  OAK-D gives x, y and height directly
            xyz = self._oak_find(goal)
            self.oak_xyz = xyz
            if xyz:
                gx, gy, hover, bottom0 = xyz[0], xyz[1], xyz[2] + 130, xyz[2] - 15
            else:
                p = self._target()
                gx, gy, hover, bottom0 = p[0], p[1], max(p[2], 20.0), floor
            self._pitch, hover = self._pick_claw_pitch(gx, gy, hover, bottom0)
            if self._pitch is None:
                raise PilotStopped("The object is out of the arm's reach from above - drive the rover closer")
            self._set(state="moving above")
            self._go(x=gx, y=gy, z=hover, pitch=self._pitch, roll=0, timeout=6)
            self._log("claw", ("above the OAK-D position" if xyz else "no OAK-D fix - looking down here")
                      + f" at z {hover:.0f} mm, tilt {self._pitch}° (kept fixed)")
            self._go(grip=open_deg)
            for attempt in (1, 2):
                # 2) centre over it (sideways only)
                self._set(state="centring")
                if not self._center(goal, 8, history):
                    if scan and attempt == 1 and not xyz and self._scan(goal):
                        p = self._target()
                        self._claw_move(p[0], p[1], p[2])
                        if not self._center(goal, 8, history):
                            result = "Could not centre over the target"
                            break
                    else:
                        result = "Could not see / centre over the target"
                        break
                # 3) straight down: halfway, re-centre, then down to the object
                p = self._target()
                # go down until the jaws touch the table / object (contact stops the push);
                # OAK-D height only shortens the trip, it never stops the arm early
                bottom = floor
                if p[2] - bottom > 90:
                    self._set(state="lowering")
                    mid = ((xyz[2] + 60) if xyz else (p[2] + bottom) / 2 + 20)
                    if self.c.arm.reachable([p[0], p[1], mid, self._pitch, p[4], p[5]]) and mid < p[2]:
                        self._go(z=mid, pitch=self._pitch)
                    self._center(goal, 3, history)
                self._set(state="lowering to grab")
                self._contact = False
                p = self._target()
                z = p[2]
                while z > bottom + 1 and not self._contact:
                    nz = max(bottom, z - self.DESCEND_STEP)
                    if not self.c.arm.reachable([p[0], p[1], nz, self._pitch, p[4], p[5]]):
                        self._log("claw", f"lowest reachable point here is z {z:.0f} mm")
                        break
                    z = nz
                    self._go(z=z, pitch=self._pitch, speed=0.6)
                if extra and not self._contact:
                    self._go(z=z - extra, pitch=self._pitch, speed=0.4)
                q = self._target()
                self._log("claw", ("touched down" if self._contact else "reached the bottom")
                          + f" at z {q[2]:.0f} mm - closing")
                # 4) grab and lift straight up
                self._set(state="grabbing")
                self._go(grip=closed_deg, timeout=2.5)
                self._wait(0.4)
                p = self._target()
                self._go(z=p[2] + 110, speed=0.7)
                # 5) did it work?
                held = self._holding_check(goal)
                if held is not False:
                    self.c.tasks.holding = True
                    result = "Done: picked it up" + ("" if held else " (could not confirm by camera)")
                    self._log("done", result)
                    break
                self._log("claw", f"missed (attempt {attempt}) - opening and trying again")
                self._go(grip=open_deg)
            else:
                result = "Missed twice - try moving the object or the rover a little"
        except PilotStopped as exc:
            result = str(exc)
        except Exception as exc:
            result = f"Error: {type(exc).__name__}: {exc}"
            self._log("error", result)
            try:
                self.c.arm.hold()
            except Exception:
                pass
        finally:
            self._set(running=False, state="finished", result=result)
            self._log("end", result)
            self.c.safety.record("qwen", "autopilot", result)

    # ------------------------------------------------------------ main loop
    def _run(self, goal, max_steps, scan):
        cfg = self.c.tasks.cfg
        open_deg, closed_deg = cfg["gripper_open_deg"], cfg["gripper_closed_deg"]
        history, last = [], None
        lost, rescans = 0, 0
        self._claw_mode = False
        self._probed, self.oak_xyz = False, None
        self._contact, self._frozen = False, 0
        result = "Stopped"
        try:
            self.c.arm.wake()                       # servo torque on, in case it tripped earlier
            self._wait(0.3)
            if self._oak_locate(goal):
                history.append("moved above the target using the OAK-D")
            elif scan and not self._scan(goal):
                result = "Could not find the target in the scan - move it into view and try again"
                return
            for step in range(1, max_steps + 1):
                self._set(step=step)
                seen = self._ask(goal, history)
                state = seen["state"]
                self._set(state=state, say=seen["say"])
                self._show(seen)
                size = seen["size"]
                if last and seen.get("target"):
                    p0, p1 = last[1], seen["target"]["px"]
                    self._update_J(last[0], (p1[0] - p0[0], p1[1] - p0[1]))
                last = None
                pose = self._target()
                note = seen["say"]

                if state == "done":
                    result = "Done: " + note
                    self._log("done", note)
                    break
                if state == "fail":
                    result = "Qwen gave up: " + note
                    self._log("fail", note)
                    break

                if state == "search":
                    lost += 1
                    if lost == 1 and not self.c.tasks.holding:
                        d = self._move_tool(dz=30)            # back off a little for a wider view
                        act = f"lost it - back off up {d[2]:+.0f} mm"
                    elif rescans < 2:
                        rescans += 1
                        self.J, self._probed = None, False
                        act = "lost it - looking again (OAK-D, then scan)"
                        self._log("search", f"{note} → {act}")
                        if not self._oak_locate(goal) and not self._scan(goal):
                            result = "Target lost and not found again"
                            break
                        lost = 0
                        continue
                    else:
                        result = "Target lost"
                        self._log("search", note)
                        break
                    self._log("search", f"{note} → {act}")
                    history.append(act)
                    continue
                lost = 0

                if state == "grasp":
                    if pose[5] < 20:
                        self._go(grip=open_deg)
                        self._log("grasp", f"{note} → open jaws first")
                        history.append("opened jaws")
                        continue
                    self._go(grip=closed_deg, timeout=2.5)
                    self._wait(0.4)
                    self.c.tasks.holding = True
                    self._move_tool(dz=35)
                    self._log("grasp", f"{note} → closed jaws, lifted")
                    history.append("closed jaws, lifted")
                    continue
                if state == "lift":
                    self._move_tool(dz=35)
                    self._log("lift", f"{note} → lifted")
                    history.append("lifted")
                    continue
                if state == "release":
                    self._go(grip=open_deg)
                    self._wait(0.3)
                    self.c.tasks.holding = False
                    self._move_tool(dz=35)
                    self._log("release", f"{note} → opened jaws, moved up")
                    history.append("released")
                    continue

                # align / approach
                tp = seen["target"]["px"]
                gp = self._jaws or [size[0] / 2, size[1] / 2]
                if not self._probed:
                    self._probed = True
                    if self._probe(goal, history, seen):
                        history.append("learned how the view moves")
                        continue
                    self.J = self._default_J(size)     # learn it while moving instead
                du, dv = gp[0] - tp[0], gp[1] - tp[1]
                err = math.hypot(du, dv) / size[0]
                dt, dr = self._solve(du * GAIN, dv * GAIN, size)
                dz = 0.0
                if (state == "approach" or err < ALIGN_PX) and err < 2.5 * ALIGN_PX:
                    if not self.c.tasks.holding and pose[5] < 20:
                        self._go(grip=open_deg)
                    # go down faster while far above; slow near the object
                    above = pose[2] - self.oak_xyz[2] if self.oak_xyz else 100.0
                    dz = -max(20.0, min(STEP_Z_MM, 0.5 * above)) if above > 30 else -20.0
                self._contact = False
                d = self._move_tool(dt=dt, dr=dr, dz=dz)
                last = ((d[0], d[1]), tp)
                if self._contact and dz < 0 and not self.c.tasks.holding:
                    self._go(grip=closed_deg, timeout=2.5)
                    self._wait(0.4)
                    self.c.tasks.holding = True
                    self._move_tool(dz=40)
                    self._log("grasp", "touched down on the target → closed jaws, lifted")
                    history.append("touched down, closed jaws, lifted")
                    continue
                act = f"{state}: right {d[0]:+.0f} out {d[1]:+.0f} up {d[2]:+.0f} mm (off by {err*100:.0f}%)"
                self._log(state, f"{note} → {act}")
                history.append(act)
            else:
                result = f"Stopped after {max_steps} steps"
        except PilotStopped as exc:
            result = str(exc)
        except Exception as exc:
            result = f"Error: {type(exc).__name__}: {exc}"
            self._log("error", result)
            try:
                self.c.arm.hold()
            except Exception:
                pass
        finally:
            self._set(running=False, state="finished", result=result)
            self._log("end", result)
            self.c.safety.record("qwen", "autopilot", result)
