import threading
import time
from pathlib import Path

from .arm import ArmDriver
from .camera import make_camera
from .rover import RoverDriver
from .safety import SafetyController
from .tasks import TaskRunner
from .vision import Calibration

ROOT = Path(__file__).resolve().parents[1]


class RobotController:
    ALLOWED_SOURCES = {"human", "qwen", "agent", "test"}

    def __init__(self, config, arm_factory=None, rover_opener=None, camera_factory=None):
        self.safety = SafetyController()
        self.rover = RoverDriver(config["rover_url"], config["rover_max_speed"],
                                 self.safety, opener=rover_opener)
        self.arm = ArmDriver(config["arm_port"], self.safety, factory=arm_factory,
                             config=config)
        self.camera = make_camera(config, camera_factory)
        cal_path = config.get("calibration_file") or str(ROOT / "calibration.json")
        self.calibration = Calibration(cal_path)
        self.arm.table_z = self.calibration.table_z if self.calibration.ready else None
        self.tasks = TaskRunner(self.arm, self.safety, self.calibration, config)
        self._command_lock = threading.Lock()

    def emergency_stop(self, reason="Emergency stop", source="human"):
        self.safety.latch(reason, source)
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
        return self.tasks.submit(data, source)

    # ------------------------------------------------------------ calibration
    def image_size(self):
        size = getattr(self.camera, "size", None)
        return list(size()) if callable(size) else None

    def calibration_add(self, data):
        pixel = data.get("pixel")
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            raise ValueError("Send pixel: [u, v]")
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

    def calibration_undo(self):
        status = self.calibration.remove_last()
        self._sync_table()
        return status

    def calibration_clear(self):
        status = self.calibration.clear()
        self._sync_table()
        return status

    def _sync_table(self):
        self.arm.table_z = self.calibration.table_z if self.calibration.ready else None

    def pixel_info(self, u, v, image_size=None):
        x, y = self.calibration.pixel_to_arm(u, v, image_size)
        return {"pixel": [u, v], "xy": [round(x, 1), round(y, 1)],
                "table_z": self.calibration.table_z}

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
        }

    def _cal_state(self):
        st = self.calibration.status()
        arm = self.arm.state()
        p = arm.get("pose") or arm.get("target")
        st["arm_pixel"] = None
        if p is not None and self.calibration.ready:
            try:
                st["arm_pixel"] = [round(v, 1) for v in self.calibration.arm_to_pixel(p[0], p[1])]
            except ValueError:
                pass
        return st

    def close(self):
        self.emergency_stop("Service shutdown", "system")
        self.camera.close()
        self.arm.close()
        self.rover.close()
