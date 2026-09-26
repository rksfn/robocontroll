import math
import threading
import time


POSE_LIMITS = ((-500, 500), (-500, 500), (-600, 600),
               (-90, 90), (-180, 180), (0, 90))


def validate_pose(values):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError("No valid six-value arm pose")
    pose = [float(value) for value in values]
    for value, (low, high) in zip(pose, POSE_LIMITS):
        if not math.isfinite(value) or not low - 2 <= value <= high + 2:
            raise ValueError("Arm pose is outside the supported range")
    pose = [max(low, min(high, value))
            for value, (low, high) in zip(pose, POSE_LIMITS)]
    return pose


class ArmDriver:
    """Discrete Cartesian commands, serialized over the RoArm USB connection."""

    def __init__(self, port, safety, factory=None):
        self.port = port
        self.safety = safety
        self._lock = threading.Lock()
        self._device = None
        self._pose = None
        self._error = "Not connected"
        self._factory = factory
        self._last_connect_attempt = 0.0
        self.connect()

    def connect(self):
        with self._lock:
            if self._device is not None:
                return
            self._last_connect_attempt = time.monotonic()
            try:
                if self._factory:
                    self._device = self._factory()
                else:
                    from roarm_sdk.roarm import roarm
                    self._device = roarm(roarm_type="roarm_m3", port=self.port,
                                         baudrate=115200, timeout=.15)
                    self._device._serial_port.write_timeout = .3
                self._pose = validate_pose(self._device.pose_get())
                self._error = ""
            except Exception as exc:
                self._device = None
                self._error = str(exc)

    def refresh(self):
        if self._device is None:
            self.connect()
        with self._lock:
            if self._device is None:
                raise RuntimeError("Arm is not connected: " + self._error)
            self._pose = validate_pose(self._device.pose_get())
            return self._pose.copy()

    def nudge(self, dx=0, dy=0, dz=0, gripper=0):
        deltas = [float(dx), float(dy), float(dz), 0.0, 0.0, float(gripper)]
        if any(not math.isfinite(value) for value in deltas):
            raise ValueError("Arm movement must contain finite numbers")
        if any(abs(value) > 10 for value in deltas[:3]):
            raise ValueError("Each XYZ nudge is limited to 10 mm")
        if abs(deltas[5]) > 5:
            raise ValueError("Each gripper nudge is limited to 5 degrees")
        if not any(deltas):
            return self.refresh()
        with self._lock:
            if self._device is None:
                raise RuntimeError("Arm is not connected: " + self._error)
            try:
                current = validate_pose(self._device.pose_get())
                target = current.copy()
                for index, delta in enumerate(deltas):
                    low, high = POSE_LIMITS[index]
                    target[index] = max(low, min(high, current[index] + delta))
                # roarm-sdk converts degrees to radians in-place.  Give it a
                # copy so our cached/UI pose remains in millimetres/degrees.
                result = self._device.pose_ctrl(target.copy())
                if result == -1:
                    raise IOError("Arm rejected the coordinate command")
                self._pose = target
                self._error = ""
                return target.copy()
            except Exception as exc:
                self._error = str(exc)
                try:
                    self._device.disconnect()
                except Exception:
                    pass
                self._device = None
                self.safety.latch("Arm command failed", "arm")
                raise

    def move_xyz(self, x, y, z):
        """Move to one human-selected XYZ target while preserving wrist/grip."""
        requested = [float(x), float(y), float(z)]
        if any(not math.isfinite(value) for value in requested):
            raise ValueError("XYZ target must contain finite numbers")
        with self._lock:
            if self._device is None:
                raise RuntimeError("Arm is not connected: " + self._error)
            try:
                current = validate_pose(self._device.pose_get())
                # A single UI typo must not command a large jump.  Larger
                # moves can be made as several visible, deliberate targets.
                if any(abs(value - current[index]) > 75
                       for index, value in enumerate(requested)):
                    raise ValueError("XYZ target is limited to 75 mm from the current pose")
                target = current.copy()
                target[:3] = requested
                target = validate_pose(target)
                result = self._device.pose_ctrl(target.copy())
                if result == -1:
                    raise IOError("Arm rejected the XYZ target")
                self._pose = target
                self._error = ""
                return target.copy()
            except ValueError:
                raise
            except Exception as exc:
                self._error = str(exc)
                try:
                    self._device.disconnect()
                except Exception:
                    pass
                self._device = None
                self.safety.latch("Arm command failed", "arm")
                raise

    def state(self):
        if self._device is None and time.monotonic() - self._last_connect_attempt >= 2:
            self.connect()
        with self._lock:
            return {
                "connected": self._device is not None,
                "error": self._error,
                "port": self.port,
                "pose": self._pose,
            }

    def close(self):
        with self._lock:
            if self._device is not None:
                self._device.disconnect()
                self._device = None
