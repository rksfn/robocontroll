import os
import threading
import time
from pathlib import Path

from .ai import VisionModel
from .arm import ArmDriver
from .camera import list_sources, make_camera
from .rover import RoverDriver
from .safety import SafetyController
from .tasks import TaskRunner
from .tunnel import Tunnel, default_command
from .pilot import QwenPilot
from .tracker import LiveVision
from .scene import SceneCalibration
from .vision import Calibration

ROOT = Path(__file__).resolve().parents[1]


class RobotController:
    ALLOWED_SOURCES = {"human", "qwen", "agent", "test"}

    def __init__(self, config, arm_factory=None, rover_opener=None, camera_factory=None):
        self.safety = SafetyController()
        self.rover = RoverDriver(config["rover_url"], config["rover_max_speed"],
                                 self.safety, opener=rover_opener,
                                 on_found=lambda url: self._save_config(rover_url=url))
        self.arm = ArmDriver(config["arm_port"], self.safety, factory=arm_factory,
                             config=config)
        self.config = config
        self._camera_factory = camera_factory
        self._camera_lock = threading.Lock()
        self.camera = make_camera(config, camera_factory)
        # second camera: OAK-D on the rover front, colour + depth
        self.scene = None
        self._ensure_scene()
        self.scene_cal = SceneCalibration(ROOT / "scene_calibration.json")
        self._autocal = {"running": False, "message": ""}
        cal_path = config.get("calibration_file") or str(ROOT / "calibration.json")
        self.calibration = Calibration(cal_path)
        self._sync_table()
        self.tasks = TaskRunner(self.arm, self.safety, self.calibration, config)
        self.tasks.container_pose = config.get("container_pose")
        self.ai = VisionModel(config.get("ai_base_url", "http://127.0.0.1:8000/v1"),
                              config.get("ai_model", ""), config.get("ai_api_key", ""),
                              config.get("ai_coords", "auto"))
        self._command_lock = threading.Lock()
        self._cal_pending = None
        self.pilot = QwenPilot(self, ROOT / "pilot.log" if camera_factory is None else None)
        self.live = LiveVision(self)       # frame-rate tracking of Qwen's boxes + yellow zone
        self.tunnel = None
        tcfg = config.get("ai_tunnel", "auto")
        if tcfg == "auto":
            base = str(config.get("ai_base_url", ""))
            tcfg = default_command(ROOT / "tunnel.sh") if (os.name == "nt" and ("127.0.0.1:8000" in base or "localhost:8000" in base)) else ""
        if tcfg and tcfg != "off" and camera_factory is None:
            self.tunnel = Tunnel(tcfg, ROOT / "tunnel.log", watch=ROOT / "tunnel.sh")
            self.tunnel.start()

    def emergency_stop(self, reason="Emergency stop", source="human"):
        self.safety.latch(reason, source)
        self.pilot.stop(reason)
        self.tasks.cancel(reason)
        self.rover.stop()
        self.arm.hold()
        return self.state()

    def arm_jog(self, data):
        """High-rate human velocity input. Not logged per call."""
        if not isinstance(data, dict):
            raise ValueError("Jog must be a JSON object")
        values = [data.get(k, 0) for k in ("vx", "vy", "vz", "vpitch", "vroll", "vgrip")]
        if any(values):
            self.safety.require_enabled()
        target = self.arm.jog(*values, frame=data.get("frame", "cartesian"),
                              scale=data.get("scale", 1.0))
        return {"ok": True, "target": target}

    # ------------------------------------------------------------ tasks
    def task(self, data, source="human"):
        if source not in self.ALLOWED_SOURCES:
            raise ValueError("Unknown command source")
        self.safety.require_enabled()
        if self.pilot.running():
            raise ValueError("Qwen autopilot is running - stop it first")
        data = dict(data)
        from_photo = bool(data.pop("from_photo", False))   # pixel came from a look-pose photo
        if ("pixel" in data and self.calibration.look_pose and not from_photo
                and not self.at_look_pose()):
            raise ValueError("The camera is on the gripper: go to the look pose first "
                             "(Look button), then click the image")
        return self.tasks.submit(data, source)

    # ------------------------------------------------------------ look pose (camera on gripper)
    LOOK_TOL_MM, LOOK_TOL_DEG = 40.0, 8.0      # the arm sags 1-2.5 cm short of commanded poses

    def at_look_pose(self):
        lp = self.calibration.look_pose
        pose = self.arm.state().get("pose")
        if not lp or pose is None:
            return False
        import math
        return (math.dist(pose[:3], lp[:3]) <= self.LOOK_TOL_MM
                and abs(pose[3] - lp[3]) <= self.LOOK_TOL_DEG
                and abs(pose[4] - lp[4]) <= self.LOOK_TOL_DEG)

    def set_look_pose(self, data):
        if data.get("clear"):
            status = self.calibration.set_look_pose(None)
            msg = "Look pose removed (fixed camera mode)"
        else:
            pose = self.arm.state().get("pose")
            if pose is None:
                raise RuntimeError("Arm position unknown - is the arm connected?")
            status = self.calibration.set_look_pose(pose)
            msg = f"Look pose set to {[round(v) for v in pose[:5]]}; calibration points cleared"
        self._cal_pending = None
        self._sync_table()
        self.safety.record("human", "calibration", msg)
        return status

    def go_look(self, timeout=20.0):
        """Move to the look pose (if needed) and wait for a fresh camera frame."""
        if not self.calibration.look_pose or self.at_look_pose():
            return
        if self.safety.state()["latched"]:
            raise RuntimeError("The arm has to move to the look pose to take the photo - press Enable first")
        st = self.tasks.submit({"task": "look"}, "qwen")
        end = time.time() + timeout
        while time.time() < end:
            cur = self.tasks.status()
            if cur.get("id") != st["id"] or cur["state"] != "running":
                if cur["state"] == "failed":
                    raise RuntimeError("Could not reach the look pose: " + cur.get("message", ""))
                break
            time.sleep(0.1)
        arrived = time.time()
        time.sleep(0.2)                      # let the image settle (motion blur)
        while time.time() < arrived + 3:
            age = self.camera.state().get("frame_age_seconds")
            if age is not None and age < time.time() - arrived - 0.4:
                break
            time.sleep(0.05)

    # ------------------------------------------------------------ calibration
    def image_size(self):
        size = getattr(self.camera, "size", None)
        return list(size()) if callable(size) else None

    def calibration_add(self, data):
        pixel = data.get("pixel")
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            raise ValueError("Send pixel: [u, v]")
        if self.calibration.look_pose:
            # camera on the gripper: remember the pixel now (seen from the look pose),
            # pair it with the arm position once the tip touches that spot.
            if not self.at_look_pose():
                raise ValueError("Go to the look pose first (Look button), then click the mark")
            size = data.get("image_size") or self.image_size()
            self._cal_pending = {"px": [float(pixel[0]), float(pixel[1])], "size": size}
            st = self.calibration.status()
            st["pending"] = self._cal_pending["px"]
            return st
        pose = self.arm.state().get("pose")
        if pose is None:
            raise RuntimeError("Arm position unknown - is the arm connected?")
        size = data.get("image_size") or self.image_size()
        if not size:
            raise RuntimeError("Camera image size unknown - is the camera running?")
        status = self.calibration.add_point(pixel, pose, size)
        self._sync_table()
        self.safety.record("human", "calibration", f"point {len(status['points'])}: pixel {pixel} -> arm {pose[:3]}")
        return status

    def calibration_touch(self, data=None):
        """Camera on the gripper: the tip now touches the clicked mark - save the pair."""
        pending = getattr(self, "_cal_pending", None)
        if not pending:
            raise ValueError("Click the mark in the image first (at the look pose)")
        pose = self.arm.state().get("pose")
        if pose is None:
            raise RuntimeError("Arm position unknown - is the arm connected?")
        if self.at_look_pose():
            raise ValueError("The arm is still at the look pose - lower the gripper tip onto the mark first")
        status = self.calibration.add_point(pending["px"], pose, pending["size"])
        self._cal_pending = None
        self._sync_table()
        self.safety.record("human", "calibration", f"point {len(status['points'])}: pixel {pending['px']} -> arm {[round(v) for v in pose[:3]]}")
        return status

    def calibration_undo(self):
        status = self.calibration.remove_last()
        self._sync_table()
        return status

    def calibration_clear(self):
        status = self.calibration.clear()
        self._sync_table()
        return status

    def _sync_table(self):
        # The measured table height is only used for pick/place heights. It limits
        # manual motion only if "table_floor": true in config.json (full range by default).
        use = self.config.get("table_floor", False) and self.calibration.ready
        self.arm.table_z = self.calibration.table_z if use else None

    def pixel_info(self, u, v, image_size=None):
        x, y = self.calibration.pixel_to_arm(u, v, image_size)
        return {"pixel": [u, v], "xy": [round(x, 1), round(y, 1)],
                "table_z": self.calibration.table_z}

    # ------------------------------------------------------------ vision model
    def _frame(self):
        jpeg, size = self.camera.frame(), self.image_size()
        if jpeg is None or not size:
            raise RuntimeError("No camera frame available")
        return jpeg, size

    def ai_locate(self, data):
        what = str(data.get("what", "")).strip()[:200]
        if not what:
            raise ValueError("Say what to look for")
        self.go_look()
        jpeg, size = self._frame()
        res = self._with_xy(self.ai.locate(jpeg, size, what))
        self.live.seed(res.get("objects", []), jpeg, size, "find")    # keep the boxes on the objects
        return res

    def ai_plan(self, data):
        goal = str(data.get("goal", "")).strip()[:300]
        if not goal:
            raise ValueError("Type a goal")
        self.go_look()
        jpeg, size = self._frame()
        res = self._with_xy(self.ai.plan(jpeg, size, goal, self.tasks.holding))
        self.live.seed([st for st in res.get("steps", []) if st.get("bbox")], jpeg, size, "plan")
        return res

    def _with_xy(self, result):
        """Add arm X/Y for every pixel so the user sees where the arm will go."""
        for item in result.get("objects", []) + result.get("steps", []):
            if "pixel" in item and self.calibration.ready:
                x, y = self.calibration.pixel_to_arm(*item["pixel"])
                item["xy"] = [round(x), round(y)]
        return result

    # ------------------------------------------------------------ camera source
    def camera_sources(self, scan=False):
        """Probing cameras opens each one (seconds on Windows, and it can disturb the running
        camera), so page loads get the last result; only the Scan button probes again."""
        current = str(self.config.get("camera_source", "auto"))
        if not scan:
            cached = getattr(self, "_sources_cache", None)
            if cached:
                return dict(cached, current=current)
            return {"current": current, "sources": [{"source": current, "label": current}]}
        srcs = list_sources(self.config.get("camera_source", "auto"))
        if self.scene is not None:
            srcs = [dict(x, label=x["label"] + " (now the rover camera)") if x["source"] == "oak" else x for x in srcs]
            if not any(x["source"] == "oak" for x in srcs):
                pass
        self._sources_cache = {"current": current, "sources": srcs}
        return self._sources_cache

    def set_camera_source(self, data):
        source = str(data.get("source", "")).strip()
        if not source:
            raise ValueError("Pick a camera")
        if source.startswith(("http://", "https://", "rtsp://")):
            source = "url:" + source
        with self._camera_lock:
            old = self.camera
            old.close()                    # frees the USB camera before reopening
            if source == "oak" and self.scene is not None:
                self.scene.close()         # the OAK-D can only be opened once
                self.scene = None
            self.config["camera_source"] = source
            self.camera = make_camera(self.config, self._camera_factory)
            self._ensure_scene()
        self._save_config(camera_source=source)
        self.safety.record("human", "camera", f"Camera source set to {source}")
        return {"current": source}

    # ------------------------------------------------------------ vision model setup
    def ai_configure(self, data):
        url = str(data.get("base_url", "")).strip().rstrip("/")
        if url:
            if not url.startswith(("http://", "https://")):
                url = "http://" + url
            if not url.endswith("/v1"):
                url += "/v1"
            self.ai.base_url = url
        if "api_key" in data:
            self.ai.api_key = str(data.get("api_key") or "").strip()
        self.ai.model = str(data.get("model", "") or "").strip()   # "" = ask the server
        self.ai._status_cache = None
        self._save_config(ai_base_url=self.ai.base_url, ai_api_key=self.ai.api_key,
                          ai_model=self.ai.model)
        self.safety.record("human", "ai", f"Model address set to {self.ai.base_url}")
        return self.ai.status()

    # ------------------------------------------------------------ OAK-D scene camera
    def _ensure_scene(self):
        """OAK-D runs as the rover camera whenever it is not the main camera."""
        if self._camera_factory is not None:
            return
        scene_src = str(self.config.get("scene_camera_source", "oak"))
        main_is_oak = type(self.camera).__name__ in ("OakCamera", "OakProcessCamera")
        want = scene_src not in ("", "off", "none") and not main_is_oak
        if want and self.scene is None:
            self.scene = make_camera(dict(self.config, camera_source=scene_src, camera_mode="depth",
                                          camera_width=self.config.get("scene_width", 640),
                                          camera_height=self.config.get("scene_height", 400)))
        elif not want and self.scene is not None:
            self.scene.close()
            self.scene = None
    def _scene_ok(self):
        if self.scene is None:
            raise RuntimeError("No OAK-D scene camera (scene_camera_source in config.json)")
        return self.scene

    def scene_point(self, u, v, box=None):
        """Pixel in the OAK-D image -> 3D point (camera) and arm coordinates."""
        cam = self._scene_ok().point_3d(float(u), float(v), box)
        out = {"pixel": [u, v], "cam": cam and [round(c) for c in cam], "arm": None,
               "distance_mm": cam and round(cam[2])}
        if cam and self.scene_cal.ready:
            out["arm"] = [round(c) for c in self.scene_cal.cam_to_arm(cam)]
        return out

    def scene_cal_click(self, data):
        """Manual link point: the gripper tip is at this pixel of the OAK-D image."""
        px = data.get("pixel")
        if not isinstance(px, (list, tuple)) or len(px) != 2:
            raise ValueError("Send pixel: [u, v]")
        cam = self._scene_ok().point_3d(float(px[0]), float(px[1]))
        if cam is None:
            raise ValueError("No depth at that spot - click right on the gripper tip")
        pose = self.arm.state().get("pose")
        if pose is None:
            raise RuntimeError("Arm position unknown")
        return self.scene_cal.add(cam, pose[:3], px)

    def scene_autocal_state(self):
        return dict(self._autocal)

    def scene_autocal(self, data=None):
        """Arm shows its gripper to the OAK-D at several spots; Qwen finds the tip in
        each photo; the depth gives its 3D position; fit the OAK-D -> arm transform."""
        self._scene_ok()
        self.safety.require_enabled()
        if self._autocal["running"] or self.pilot.running() or self.tasks.busy():
            raise ValueError("Something else is moving the arm - wait or stop it first")
        self._autocal = {"running": True, "message": "starting"}
        threading.Thread(target=self._autocal_run, args=(bool((data or {}).get("keep")),),
                         daemon=True).start()
        return self.scene_autocal_state()

    def _autocal_run(self, keep):
        import math
        from concurrent.futures import ThreadPoolExecutor
        from .ai import extract_json
        def say(msg):
            self._autocal["message"] = msg
        try:
            start = self.arm.state().get("pose")
            spots = []
            for x in (240, 320):
                for y in (-100, 0, 100):
                    for z in (-60, 10, 80):
                        for pitch in (45, 30, 60, 15, 75):
                            if self.arm.reachable([x, y, z, pitch, 0, 30]):
                                spots.append((x, y, z, pitch))
                                break
            shots = []
            for i, (x, y, z, pitch) in enumerate(spots, 1):
                if self.safety.state()["latched"]:
                    raise RuntimeError("STOP pressed")
                say(f"moving to spot {i}/{len(spots)}")
                self.arm.move_to(x=x, y=y, z=z, pitch=pitch, roll=0, speed=0.6)
                end = time.time() + 6
                while time.time() < end and self.arm.state()["goal"] is not None:
                    if self.safety.state()["latched"]:
                        raise RuntimeError("STOP pressed")
                    time.sleep(0.03)
                time.sleep(0.5)
                depth, K = self.scene.depth_frame()
                jpeg = self.scene.frame()
                pose = self.arm.state().get("pose")
                if jpeg is not None and depth is not None and pose:
                    shots.append((jpeg, depth.copy(), K, pose[:3]))
            if start:
                self.arm.move_to(x=start[0], y=start[1], z=start[2], pitch=start[3], roll=start[4],
                                 clamp=True, speed=0.6)
            say(f"Qwen is finding the gripper in {len(shots)} photos…")
            size = self.scene.size()
            prompt = ("Photo from a camera on a small robot vehicle. A robot arm reaches in front of it. "
                      "Find the TIP of the arm's gripper (the very end of its two fingers). {COORDS} "
                      'Answer compact JSON only: {"tip":[x,y] or null}')

            def find(shot):
                jpeg, depth, K, arm_xyz = shot
                try:
                    raw, seen, scale = self.ai.ask(prompt, jpeg, size, max_tokens=40)
                    _, (u, v) = self.ai.to_camera(extract_json(raw).get("tip"), seen, scale, size)
                    cam = self.scene.point_3d(u, v, depth=depth, K=K)
                    return (cam, arm_xyz, [u, v]) if cam else None
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=6) as pool:
                pairs = [p for p in pool.map(find, shots) if p]
            if not keep:
                self.scene_cal.clear()
            for cam, arm_xyz, px in pairs:
                st = self.scene_cal.add(cam, arm_xyz, px)
            st = self.scene_cal.status()
            if st["ready"]:
                say(f"Done: OAK-D linked using {len(st['used'])} of {st['points']} spots, "
                    f"error {st['error_mm']} mm")
            else:
                say(f"Not enough good spots ({len(pairs)} found) - make sure the OAK-D can see the gripper")
            self.safety.record("system", "calibration", self._autocal["message"])
        except Exception as exc:
            say(f"Stopped: {exc}")
        finally:
            self._autocal["running"] = False

    def tunnel_state(self):
        return self.tunnel.state() if self.tunnel else {"enabled": False}

    def tunnel_restart(self):
        if not self.tunnel:
            raise RuntimeError("No tunnel configured (ai_tunnel in config.json)")
        self.tunnel.restart()
        self.ai._status_cache = None
        return self.tunnel.state()

    def ai_ask(self, data):
        question = str(data.get("question", "")).strip()[:500] or \
            "Describe what you see in this image in 2-3 short sentences."
        jpeg, size = self._frame()
        return self.ai.describe(jpeg, size, question)

    def _save_config(self, **values):
        """Remember a setting in config.json for the next start (best effort)."""
        try:
            import json
            path = ROOT / "config.json"
            cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg.update(values)
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def set_rover_url(self, data):
        url = self.rover.set_url(data.get("url", ""))
        try:                                   # remember it for next start
            import json
            path = ROOT / "config.json"
            cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg["rover_url"] = url or "auto"
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        self.safety.record("human", "rover", f"Rover address set to {url or 'auto'}")
        return self.rover.state()

    def set_claw_point(self, data):
        """Where the jaws come down in the gripper-camera image (fraction of width/height)."""
        px, size = data.get("pixel"), self.image_size()
        if not isinstance(px, (list, tuple)) or len(px) != 2 or not size:
            raise ValueError("Send pixel: [u, v]")
        frac = [round(float(px[0]) / size[0], 4), round(float(px[1]) / size[1], 4)]
        self.config["claw_point"] = frac
        self._save_config(claw_point=frac)
        return {"claw_point": frac}

    def set_container_pose(self, data):
        """Teach the rover-container drop: the arm's CURRENT pose is where it lets go."""
        if data.get("clear"):
            self.config["container_pose"] = self.tasks.container_pose = None
            self._save_config(container_pose=None)
            return {"container_pose": None}
        if data.get("test"):
            return self.task({"task": "container"}, "human")
        pose = self.arm.state().get("pose")
        if pose is None:
            raise RuntimeError("Arm position unknown - is the arm connected?")
        cp = [round(float(v), 1) for v in pose[:5]]
        self.config["container_pose"] = self.tasks.container_pose = cp
        self._save_config(container_pose=cp)
        self.safety.record("human", "container", f"container drop pose set to {cp}")
        return {"container_pose": cp}

    def set_drop_spot(self, data):
        """Click the coin (at the look pose): items get dropped exactly there (arm mm, stable)."""
        name = str(data.get("name") or "coin").strip().lower()[:30] or "coin"
        spots = dict(self.config.get("drop_spots") or {})
        if not spots and self.config.get("drop_xy"):
            spots = {"coin": self.config["drop_xy"]}
        if data.get("clear"):
            spots.pop(name, None) if name != "all" else spots.clear()
            self.config.update(drop_spots=spots, drop_xy=spots.get("coin"))
            self._save_config(drop_spots=spots, drop_xy=spots.get("coin"))
            return {"drop_spots": spots}
        px = data.get("pixel")
        if not isinstance(px, (list, tuple)) or len(px) != 2:
            raise ValueError("Send pixel: [u, v]")
        if not self.calibration.ready:
            raise ValueError("Calibrate the camera first")
        if self.calibration.look_pose and not self.at_look_pose():
            raise ValueError("Go to the look pose first (Look button), then click the coin")
        x, y = self.calibration.pixel_to_arm(float(px[0]), float(px[1]))
        xy = [round(x, 1), round(y, 1)]
        pose = [x, y, (self.calibration.table_z or 0) + 60, 90, 0, 0]
        if not self.arm.reachable(pose) and not self.arm.reachable(pose[:3] + [60, 0, 0]):
            raise ValueError(f"The arm can't reach that spot ({xy[0]:.0f}, {xy[1]:.0f} mm) - put the coin closer")
        spots[name] = xy
        self.config.update(drop_spots=spots, drop_xy=spots.get("coin"), drop_zone=name)
        self._save_config(drop_spots=spots, drop_xy=spots.get("coin"), drop_zone=name)
        self.safety.record("human", "drop spot", f"drop spot '{name}' set to {xy}")
        return {"drop_xy": xy, "name": name, "drop_spots": spots}

    def arm_wake(self):
        """Servo torque on (after overload protection switched it off)."""
        self.arm.wake()
        self.safety.record("human", "arm", "servo torque re-enabled")
        return {"ok": True}

    def enable_human(self):
        self.rover.stop()
        self.arm.hold()
        self.safety.enable_human()
        return self.state()

    def action(self, data, source="human"):
        if source not in self.ALLOWED_SOURCES:
            raise ValueError("Unknown command source")
        if not isinstance(data, dict):
            raise ValueError("Action must be a JSON object")
        action = data.get("action")
        request_id = str(data.get("request_id", ""))[:80]
        with self._command_lock:
            if action == "stop":
                return self.emergency_stop("Stop action", source)
            self.safety.require_enabled()
            if action == "drive":
                speeds = self.rover.command(data.get("linear", 0), data.get("turn", 0),
                                             data.get("duration_ms", 250))
                result = {"speeds": speeds}
            elif action == "arm_nudge":
                pose = self.arm.nudge(data.get("dx_mm", 0), data.get("dy_mm", 0),
                                      data.get("dz_mm", 0), data.get("gripper_deg", 0))
                result = {"pose": pose}
            elif action == "arm_target":
                if source != "human":
                    raise ValueError("Only the local dashboard can set an absolute arm target")
                pose = self.arm.move_to(data.get("x_mm"), data.get("y_mm"),
                                        data.get("z_mm"), data.get("pitch_deg"),
                                        data.get("roll_deg"), data.get("gripper_deg"),
                                        clamp=bool(data.get("clamp", False)),
                                        speed=data.get("speed", 0.6))
                result = {"pose": pose}
            else:
                raise ValueError("Allowed actions are stop, drive, arm_nudge and arm_target")
            if not (action == "drive" and source == "human"):
                self.safety.record(source, action, request_id)
            return {"ok": True, "action": action, "result": result}

    def state(self):
        return {
            "time": round(time.time(), 3),
            "safety": self.safety.state(),
            "rover": self.rover.state(),
            "arm": self.arm.state(),
            "camera": dict(self.camera.state(), size=self.image_size()),
            "task": self.tasks.status(),
            "calibration": self._cal_state(),
            "pilot": self.pilot.status(),
            "live": self.live.state(),
            "container_pose": self.config.get("container_pose"),
            "claw_point": self.config.get("claw_point", [0.5, 0.5]),
            "scene": self._scene_state(),
        }

    def _drop_px(self):
        """All named drop spots in camera pixels (valid at the look pose)."""
        spots = dict(self.config.get("drop_spots") or {})
        if not spots and self.config.get("drop_xy"):
            spots = {"coin": self.config["drop_xy"]}
        if not spots or not self.calibration.ready:
            return []
        out = []
        for name, xy in spots.items():
            try:
                out.append({"name": name, "px": [round(v, 1) for v in self.calibration.arm_to_pixel(*xy)]})
            except Exception:
                pass
        return out

    def _scene_state(self):
        if self.scene is None:
            return {"enabled": False}
        st = dict(self.scene.state(), enabled=True, size=list(self.scene.size()))
        st["link"] = self.scene_cal.status()
        st["autocal"] = self.scene_autocal_state()
        pose = self.arm.state().get("pose")
        st["tip_px"] = None
        if pose and self.scene_cal.ready:          # where the gripper tip should appear
            try:
                c = self.scene_cal.arm_to_cam(pose[:3])
                depth, K = self.scene.depth_frame()
                if K and c[2] > 50:
                    st["tip_px"] = [round(K[0][0] * c[0] / c[2] + K[0][2], 1),
                                    round(K[1][1] * c[1] / c[2] + K[1][2], 1)]
            except Exception:
                pass
        return st

    def _cal_state(self):
        st = self.calibration.status()
        st["at_look"] = self.at_look_pose()
        pend = getattr(self, "_cal_pending", None)
        st["pending"] = pend["px"] if pend else None
        arm = self.arm.state()
        p = arm.get("pose") or arm.get("target")
        st["arm_pixel"] = None
        if p is not None and self.calibration.ready and (not self.calibration.look_pose or st["at_look"]):
            try:
                st["arm_pixel"] = [round(v, 1) for v in self.calibration.arm_to_pixel(p[0], p[1])]
            except ValueError:
                pass
        return st

    def close(self):
        self.live.close()
        self.emergency_stop("Service shutdown", "system")
        self.camera.close()
        if self.scene is not None:
            self.scene.close()
        if self.tunnel:
            self.tunnel.stop()
        self.arm.close()
        self.rover.close()
