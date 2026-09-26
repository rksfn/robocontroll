import threading
import time

from .arm import ArmDriver
from .camera import OakCamera
from .rover import RoverDriver
from .safety import SafetyController


class RobotController:
    ALLOWED_SOURCES = {"human", "qwen", "agent", "test"}

    def __init__(self, config, arm_factory=None, rover_opener=None, camera_factory=None):
        self.safety = SafetyController()
        self.rover = RoverDriver(config["rover_url"], config["rover_max_speed"],
                                 self.safety, opener=rover_opener)
        self.arm = ArmDriver(config["arm_port"], self.safety, factory=arm_factory)
        camera_class = camera_factory or OakCamera
        self.camera = camera_class(config.get("camera_mode", "rgb"),
                                   config.get("camera_width", 640),
                                   config.get("camera_height", 400),
                                   config.get("camera_fps", 15))
        self._command_lock = threading.Lock()

    def emergency_stop(self, reason="Emergency stop", source="human"):
        self.safety.latch(reason, source)
        self.rover.stop()
        return self.state()

    def enable_human(self):
        self.rover.stop()
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
                pose = self.arm.move_xyz(data.get("x_mm"), data.get("y_mm"),
                                         data.get("z_mm"))
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
            "camera": self.camera.state(),
        }

    def close(self):
        self.emergency_stop("Service shutdown", "system")
        self.camera.close()
        self.arm.close()
        self.rover.close()
