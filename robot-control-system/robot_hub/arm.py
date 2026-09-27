"""Real-time RoArm-M3 control over USB serial.

Design (why this is fast and does not stutter):

* One background loop streams the *commanded* pose to the arm at a fixed
  rate (default 40 Hz) using the firmware's direct Cartesian command
  (T:1041).  Motion is produced by integrating a velocity, not by reading
  the arm and adding a delta, so there is no request/response round trip
  in the control path and no drift from servo sag.
* A separate reader thread parses feedback frames (T:1051) that the loop
  requests at ~10 Hz.  Feedback is only used for display, start-up
  alignment, anti-windup and holding position on STOP.
* Human/gamepad input is a *velocity* ("jog") with a dead-man timeout, so
  releasing a key or losing the browser stops motion within ~0.3 s.
* Absolute targets ("goals") are approached at a capped speed, so a far
  target is a smooth, stoppable move instead of a jump.
* Serial glitches reconnect automatically; they do not latch a global STOP.
"""

import json
import math
import threading
import time

from . import kinematics


# Legacy limits kept for validate_pose() and agent input checks.
POSE_LIMITS = ((-500, 500), (-500, 500), (-600, 600),
               (-90, 90), (-180, 180), (0, 90))

DEFAULTS = {
    "arm_rate_hz": 30,
    "arm_max_speed_mm_s": 450,
    "arm_goal_speed_mm_s": 450,
    "arm_wrist_speed_deg_s": 140,
    "arm_grip_speed_deg_s": 200,
    # Extra user limits on top of the real kinematic reach check.
    "arm_reach_min_mm": 60,
    "arm_reach_max_mm": 540,
    "arm_z_min_mm": -120,
    "arm_z_max_mm": 500,
    "arm_lag_limit_mm": 90,
    "arm_accel_mm_s2": 1600,
    "arm_deadman_s": 0.3,
    "gripper_torque": 350,          # 1-1000 squeeze limit (firmware T:107); 0 = leave as is
}


def validate_pose(values):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError("No valid six-value arm pose")
    pose = [float(value) for value in values]
    for value, (low, high) in zip(pose, POSE_LIMITS):
        if not math.isfinite(value) or not low - 2 <= value <= high + 2:
            raise ValueError("Arm pose is outside the supported range")
    return [max(low, min(high, value))
            for value, (low, high) in zip(pose, POSE_LIMITS)]


def _clamp(value, low, high):
    return max(low, min(high, value))


def _finite(values):
    out = [float(v) for v in values]
    if any(not math.isfinite(v) for v in out):
        raise ValueError("Arm values must be finite numbers")
    return out


def encode_pose(pose):
    """User pose [x,y,z mm, pitch deg, roll deg, grip deg] -> T:1041 line."""
    x, y, z, pitch, roll, grip = pose
    cmd = {"T": 1041, "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
           "t": round(math.radians(pitch), 5), "r": round(math.radians(roll), 5),
           # Same "angular_direct" gripper convention as roarm-sdk.
           "g": round(math.pi - math.radians(grip), 5)}
    return (json.dumps(cmd, separators=(",", ":")) + "\n").encode()


def decode_feedback(line):
    """T:1051 feedback frame -> user pose, or None if the line is not one."""
    try:
        data = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("T") != 1051:
        return None
    try:
        pose = [float(data["x"]), float(data["y"]), float(data["z"]),
                math.degrees(float(data.get("tit", 0))),
                math.degrees(float(data.get("r", 0))),
                180 - math.degrees(float(data.get("g", math.pi)))]
    except (KeyError, TypeError, ValueError):
        return None
    return pose if all(math.isfinite(v) for v in pose) else None


def open_serial(port):
    import serial  # pyserial (installed with roarm-sdk)
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 115200
    ser.timeout = 0.05
    ser.write_timeout = 0.3
    ser.rts = False  # same as roarm-sdk: do not reset the ESP32 on open
    ser.open()
    return ser


def find_arm_port(skip=()):
    """Find the RoArm on any USB serial port: it answers {"T":105} with a T:1051 frame."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    for info in list_ports.comports():
        if info.device in skip:
            continue
        try:
            ser = open_serial(info.device)
        except Exception:
            continue                       # busy or not a serial device we can open
        try:
            buf, end = b"", time.monotonic() + 1.5
            while time.monotonic() < end:
                ser.write(b'{"T":105}\n')
                buf += ser.read(512)
                if b'"T":1051' in buf and b'"tit"' in buf:
                    return info.device
                time.sleep(0.1)
        except Exception:
            pass
        finally:
            try:
                ser.close()
            except Exception:
                pass
    return None


class ArmDriver:
    def __init__(self, port, safety, factory=None, config=None, autostart=True):
        self.port = port
        self.safety = safety
        self.cfg = dict(DEFAULTS)
        self.cfg.update({k: v for k, v in (config or {}).items() if k in DEFAULTS})
        self._auto_port = factory is None       # real hardware: may search for the arm
        self._factory = factory or (lambda: open_serial(self.port))
        self._search_next = str(port).lower() == "auto"
        self._lock = threading.RLock()
        self._ser = None
        self._error = "Not connected"
        self._measured = None
        self._measured_time = 0.0
        self._target = None          # commanded pose, streamed to the arm
        self._goal = None            # optional absolute destination
        self._jog = [0.0] * 6        # normalized -1..1: x y z pitch roll grip
        self._jog_frame = "cartesian"
        self._jog_scale = 1.0
        self._jog_time = 0.0
        self._last_sent = None
        self._last_feedback_request = 0.0
        self._last_connect_attempt = -10.0
        self._sent_count = 0
        self._vel = [0.0, 0.0, 0.0]
        self._goal_speed = 0.0
        self._goal_scale = 1.0
        self._stall_since = None
        self._blocked_since = None
        self._edge_notice = 0.0
        self._goal_mode = "line"
        self._goal_mode_for = None
        self._notice = None
        self.table_z = None
        self._done = threading.Event()
        self._reader = None
        self._thread = None
        if autostart:
            self._thread = threading.Thread(target=self._run, daemon=True, name="arm-loop")
            self._thread.start()

    # ------------------------------------------------------------ connection
    def _connect(self):
        self._last_connect_attempt = time.monotonic()
        if self._auto_port and self._search_next:
            found = find_arm_port()
            self._search_next = False
            if found:
                if found != self.port:
                    self.safety.record("arm", "found", f"Arm found on {found}")
                self.port = found
            elif str(self.port).lower() == "auto":
                self._error = "Arm not found on any USB port (plugged in and powered?)"
                self._search_next = True
                return False
        try:
            ser = self._factory()
        except Exception as exc:
            self._error = f"Cannot open {self.port}: {exc} - searching other USB ports"
            self._search_next = True            # e.g. plugged into a different USB port
            return False
        with self._lock:
            self._ser = ser
            self._measured = None
            self._target = None
            self._goal = None
            self._last_sent = None
            self._error = "Waiting for arm feedback"
        self._reader = threading.Thread(target=self._read_loop, args=(ser,),
                                        daemon=True, name="arm-reader")
        self._reader.start()
        return True

    def _drop(self, reason):
        with self._lock:
            ser, self._ser = self._ser, None
            self._error = reason
            self._jog = [0.0] * 6
            self._goal = None
            self._target = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        self.safety.record("arm", "disconnected", reason)

    def _write(self, data):
        ser = self._ser
        if ser is None:
            return False
        try:
            ser.write(data)
            return True
        except Exception as exc:
            self._drop(f"Serial write failed: {exc}")
            return False

    def _read_loop(self, ser):
        buf = b""
        while not self._done.is_set() and self._ser is ser:
            try:
                chunk = ser.read(256)
            except Exception as exc:
                if self._ser is ser:
                    self._drop(f"Serial read failed: {exc}")
                return
            if not chunk:
                continue
            buf += chunk
            if len(buf) > 8192:
                buf = buf[-2048:]
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                start = line.find(b"{")
                if start < 0:
                    continue
                pose = decode_feedback(line[start:].strip().decode("utf-8", "replace"))
                if pose is not None:
                    self._on_feedback(pose)

    @staticmethod
    def _plausible(pose):
        """Unpowered servos make the board report impossible angles (e.g. pitch -540, grip 180)."""
        return abs(pose[3]) <= 200 and -20 <= pose[5] <= 135 and all(abs(v) < 2000 for v in pose[:3])

    def _on_feedback(self, pose):
        with self._lock:
            now = time.monotonic()
            if not self._plausible(pose):
                # link is alive but the servos are not: stop everything, resync when they answer again
                self._measured_time = now
                self._target = self._goal = None
                self._jog = [0.0] * 6
                self._vel = [0.0, 0.0, 0.0]
                self._error = ("Arm servos not answering (impossible position readings) - "
                               "check the arm's power supply is on")
                return
            if self._measured is None or any(abs(a - b) > 0.05 for a, b in zip(pose, self._measured)):
                self._changed_time = now          # real servo readings always jitter a little
            self._measured = pose
            self._measured_time = now
            if self._target is None:
                # First feedback after connect: start from where the arm is,
                # so enabling control never causes a jump.
                self._target = self._clamp_pose(list(pose))
                self._last_sent = list(self._target)
                self._error = ""
                self._torque_pending = True

    # ------------------------------------------------------------ geometry
    def _clamp_xyz(self, x, y, z):
        c = self.cfg
        r = math.hypot(x, y)
        if r < 1e-6:
            x, y, r = c["arm_reach_min_mm"], 0.0, c["arm_reach_min_mm"]
        rc = _clamp(r, c["arm_reach_min_mm"], c["arm_reach_max_mm"])
        if rc != r:
            x, y = x * rc / r, y * rc / r
        return x, y, _clamp(z, c["arm_z_min_mm"], c["arm_z_max_mm"])

    def _clamp_pose(self, pose):
        x, y, z = self._clamp_xyz(*pose[:3])
        return [x, y, z,
                _clamp(pose[3], *POSE_LIMITS[3]),
                _clamp(pose[4], *POSE_LIMITS[4]),
                _clamp(pose[5], *POSE_LIMITS[5])]

    def _z_floor(self):
        floor = self.cfg["arm_z_min_mm"]
        if self.table_z is not None:
            # 10 mm below the measured table: room for uneven tables and for
            # touching the table during calibration, still far from a crash.
            floor = max(floor, self.table_z - 10)
        return floor

    def pose_slack(self, pose):
        """> = 0 when the pose is inside user limits AND the arm can reach it."""
        x, y, z = pose[:3]
        c = self.cfg
        r = math.hypot(x, y)
        user = min(r - c["arm_reach_min_mm"], c["arm_reach_max_mm"] - r,
                   z - self._z_floor(), c["arm_z_max_mm"] - z)
        if user < -0.5:
            return -10.0 + user / 1000
        return kinematics.slack(x, y, z, pose[3])

    def reachable(self, pose):
        return self.pose_slack(pose) >= 0

    def _limit_move(self, old, new):
        """Largest part of the move old->new that stays reachable."""
        s_new = self.pose_slack(new)
        if s_new >= 0.004:
            return new, False
        s_old = self.pose_slack(old)
        if s_old < 0:
            # Already outside (e.g. arm was switched on in an odd pose):
            # allow any move that gets closer to being valid.
            return (new if s_new > s_old else old), True
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            if self.pose_slack([o + (n - o) * mid for o, n in zip(old, new)]) >= 0.004:
                lo = mid
            else:
                hi = mid
        return [o + (n - o) * lo for o, n in zip(old, new)], True

    def in_workspace(self, x, y, z):
        c = self.cfg
        r = math.hypot(x, y)
        return (c["arm_reach_min_mm"] - 1 <= r <= c["arm_reach_max_mm"] + 1
                and c["arm_z_min_mm"] - 1 <= z <= c["arm_z_max_mm"] + 1)

    # ------------------------------------------------------------ control loop
    def _check_link(self, now):
        """Never move blind: if position feedback stops, freeze the motion."""
        with self._lock:
            if self._target is None or self._measured is None:
                return
            stale = now - self._measured_time
            if stale > 1.5 and (any(self._jog) or self._goal is not None):
                self._jog = [0.0] * 6
                self._goal = None
                self._vel = [0.0, 0.0, 0.0]
                self._goal_speed = 0.0
                self._notify("Arm stopped: lost position feedback (check USB cable/power)")
        if stale > 5.0:
            self._drop("No feedback for 5 s - reconnecting")
            return
        # Servo power missing: the board (USB-powered) keeps answering with the
        # last known position, frozen to the last decimal, while commands are ignored.
        with self._lock:
            frozen = now - getattr(self, "_changed_time", now)
            off = math.dist(self._target[:3], self._measured[:3]) if self._target and self._measured else 0
            # real arms settle 1-2.5 cm short (sag) and then read perfectly still, so only a big
            # gap between command and position counts as "no servo power"
            # count only the time the arm has been BOTH far from its command AND not moving:
            # an arm resting still at the look pose (identical readings) is fine until a move starts
            if off > 35:
                if getattr(self, "_off_since", None) is None:
                    self._off_since = now
            else:
                self._off_since = None
            stuck = now - max(getattr(self, "_changed_time", now), self._off_since or now)
            self._servos_silent = frozen > 3.0 and off > 35 and stuck > 3.0
        if self._servos_silent and now - getattr(self, "_silent_notice", 0) > 10:
            self._silent_notice = now
            self._notify("Arm servos are not responding (position frozen) - check that the arm's "
                         "power supply is plugged in and switched on")

    def _notify(self, message):
        self._notice = {"time": time.time(), "message": message}
        self.safety.record("arm", "warning", message)

    def _lag(self, xyz, now):
        m = self._measured
        if m is None or now - self._measured_time > 1.5:
            return None
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(xyz, m[:3])))

    def step(self, dt, now=None):
        """Advance the commanded pose by dt seconds. Returns the pose to send."""
        now = time.monotonic() if now is None else now
        c = self.cfg
        accel = c["arm_accel_mm_s2"]
        with self._lock:
            if self._target is None:
                return None
            latched = self.safety.state()["latched"]
            if latched or now - self._jog_time > c["arm_deadman_s"]:
                self._jog = [0.0] * 6
            if latched:
                self._goal = None
                self._vel = [0.0, 0.0, 0.0]
                self._goal_speed = 0.0
            target = list(self._target)
            old_xyz = target[:3]

            if any(self._jog):
                self._goal = None  # manual input always overrides a goal
            if self._goal is None:
                # Velocity jog with acceleration limit (smooth start/stop).
                s = self._jog_scale
                want = [v * c["arm_max_speed_mm_s"] * s for v in self._jog[:3]]
                if self._jog_frame == "polar":
                    h = math.atan2(target[1], target[0])
                    radial, side = want[0], want[1]
                    want[0] = radial * math.cos(h) - side * math.sin(h)
                    want[1] = radial * math.sin(h) + side * math.cos(h)
                dv = accel * dt
                def ramp(v, w):
                    # braking is twice as sharp as speeding up
                    lim = dv * 2 if abs(w) < abs(v) or v * w < 0 else dv
                    return v + _clamp(w - v, -lim, lim)
                self._vel = [ramp(v, w) for v, w in zip(self._vel, want)]
                for i in range(3):
                    target[i] += self._vel[i] * dt
                target[3] += self._jog[3] * c["arm_wrist_speed_deg_s"] * s * dt
                target[4] += self._jog[4] * c["arm_wrist_speed_deg_s"] * s * dt
                target[5] += self._jog[5] * c["arm_grip_speed_deg_s"] * dt
            else:
                # Goal: interpolate in JOINT space (like an industrial arm's
                # "MoveJ").  Every in-between pose is then a valid joint
                # configuration, and the base never wraps through +-180 deg.
                # Falls back to a cylindrical path if IK is unavailable.
                goal = self._goal
                self._vel = [0.0, 0.0, 0.0]
                vmax = c["arm_goal_speed_mm_s"] * self._goal_scale
                if self._goal_mode_for is not goal:
                    # Straight line if every point on it is reachable (precise,
                    # e.g. straight down onto an object); otherwise joint space.
                    self._goal_mode_for = goal
                    self._goal_mode = "line" if all(
                        self.pose_slack([a + (b - a) * k / 24 for a, b in zip(target, goal)]) >= 0.004
                        for k in range(1, 25)) else "joint"
                j0 = None
                if self._goal_mode == "line":
                    d4 = [b - a for a, b in zip(target[:4], goal[:4])]
                    length = max(math.dist(target[:3], goal[:3]), 150 * math.radians(abs(d4[3])))
                else:
                    try:
                        j0 = kinematics.ik(target[0], target[1], target[2], math.radians(target[3]))
                        j1 = kinematics.ik(goal[0], goal[1], goal[2], math.radians(goal[3]))
                    except (ValueError, ZeroDivisionError):
                        j0 = j1 = None
                if self._goal_mode == "line":
                    pass
                elif j0 is not None:
                    dj = [b - a for a, b in zip(j0, j1)]
                    cart = math.dist(target[:3], goal[:3])
                    # 1 rad of joint motion ~ 300 mm of tool motion
                    length = max(cart, 300 * max(abs(v) for v in dj[:3]),
                                 150 * abs(dj[3]))
                else:
                    r0, t0 = math.hypot(target[0], target[1]), math.atan2(target[1], target[0])
                    r1, t1 = math.hypot(goal[0], goal[1]), math.atan2(goal[1], goal[0])
                    dth = t1 - t0          # no wrap-around: the base cannot spin through 180
                    dr, dz = r1 - r0, goal[2] - target[2]
                    length = math.sqrt(dr * dr + (0.5 * (r0 + r1) * dth) ** 2 + dz * dz)
                self._goal_speed = min(self._goal_speed + accel * dt, vmax,
                                       math.sqrt(2 * accel * length) + 5)
                step = self._goal_speed * dt
                if length <= max(step, 0.05):
                    target[:4] = goal[:4]
                elif self._goal_mode == "line":
                    f = step / length
                    target[:4] = [a + f * d for a, d in zip(target[:4], d4)]
                elif j0 is not None:
                    f = step / length
                    x, y, z, t = kinematics.fk(*[a + f * d for a, d in zip(j0, dj)])
                    target[:4] = [x, y, z, math.degrees(t)]
                else:
                    f = step / length
                    r, th, z = r0 + f * dr, t0 + f * dth, target[2] + f * dz
                    target[:3] = [r * math.cos(th), r * math.sin(th), z]
                    target[3] += _clamp(goal[3] - target[3], -c["arm_wrist_speed_deg_s"] * dt,
                                        c["arm_wrist_speed_deg_s"] * dt)
                for i, rate in ((4, c["arm_wrist_speed_deg_s"]), (5, c["arm_grip_speed_deg_s"])):
                    target[i] += _clamp(goal[i] - target[i], -rate * dt, rate * dt)

            target = self._clamp_pose(target)
            target, limited = self._limit_move(self._target, target)
            if limited and self._goal is not None:
                if self._blocked_since is None:
                    self._blocked_since = now
                elif now - self._blocked_since > 0.4:
                    self._goal = None
                    self._goal_speed = 0.0
                    self._blocked_since = None
                    self._notify("Path to the target leaves the arm's reach - stopped at the edge")
            else:
                self._blocked_since = None
            if limited and any(self._jog) and now - self._edge_notice > 3:
                self._edge_notice = now
                self._notice = {"time": time.time(), "message": "Edge of reach - the arm cannot go further this way"}

            # Anti-windup: if the physical arm is far behind the command
            # (blocked, overloaded or unreachable pose) pause the command
            # instead of running away.  If it stays stuck, give up clearly.
            new_lag = self._lag(target[:3], now)
            # Feedback is up to ~0.1 s old, so allow for distance commanded since.
            speed_now = max(self._goal_speed, math.sqrt(sum(v * v for v in self._vel)))
            allowance = c["arm_lag_limit_mm"] + speed_now * (now - self._measured_time + 0.1)
            if new_lag is not None and new_lag > allowance \
                    and new_lag > (self._lag(old_xyz, now) or 0):
                target[:3] = old_xyz
                self._vel = [0.0, 0.0, 0.0]
                self._goal_speed = 0.0
                if self._stall_since is None:
                    self._stall_since = now
                    self._wake_pending = True          # overloaded servos switch torque off: re-enable NOW
                elif now - self._stall_since > 1.5 and (self._goal or any(self._jog)):
                    self._goal = None
                    self._jog = [0.0] * 6
                    self._stall_since = None
                    m = self._clamp_pose(list(self._measured))
                    if self.pose_slack(m) >= 0.004:
                        self._target = m
                    else:
                        self._target = list(self._last_sent or self._target)
                    self._notify("Arm could not follow (blocked, unpowered or overloaded) - "
                                 "stopped where it is; re-enabling servo torque")
                    self._stalls = getattr(self, "_stalls", 0) + 1
                    self._wake_pending = True
                    return list(self._target)
            else:
                self._stall_since = None
            self._target = target
            if self._goal is not None and all(abs(g - t) < 0.05 for g, t in zip(self._goal, target)):
                self._goal = None          # arrived (checked after all limits)
                self._goal_speed = 0.0
            return list(target)

    def _run(self):
        period = 1.0 / max(5, min(100, float(self.cfg["arm_rate_hz"])))
        last = time.monotonic()
        while not self._done.is_set():
            now = time.monotonic()
            dt, last = min(now - last, 0.1), now
            if self._ser is None:
                if now - self._last_connect_attempt >= 2.0:
                    self._connect()
            else:
                if now - self._last_feedback_request >= 0.1:
                    self._last_feedback_request = now
                    self._write(b'{"T":105}\n')
                if getattr(self, "_torque_pending", False):
                    self._torque_pending = False
                    tor = int(self.cfg["gripper_torque"])
                    if 0 < tor <= 1000:
                        # stored in the servo's EEPROM, so only once per connection
                        self._write(('{"T":107,"tor":%d}\n' % tor).encode())
                if getattr(self, "_wake_pending", False):
                    self._wake_pending = False
                    self._write(b'{"T":210,"cmd":1}\n')      # servo torque back on
                self._check_link(now)
                target = self.step(dt, now)
                if target is not None and target != self._last_sent \
                        and kinematics.slack(round(target[0], 2), round(target[1], 2), round(target[2], 2),
                                             math.degrees(round(math.radians(target[3]), 5))) < 0 \
                        and self._last_sent is not None and kinematics.slack(*self._last_sent[:4]) >= 0:
                    target = None      # never transmit a pose that rounding pushed out of reach
                if target is not None and target != self._last_sent:
                    if self._write(encode_pose(target)):
                        self._last_sent = target
                        self._sent_count += 1
                if (self._target is None and self._ser is not None
                        and now - self._last_connect_attempt > 4.0):
                    self._search_next = True
                    self._drop("No feedback from arm on " + str(self.port) + " - searching other USB ports")
            self._done.wait(max(0.0, period - (time.monotonic() - now)))

    # ------------------------------------------------------------ public API
    def _require_ready(self):
        if self._ser is None or self._target is None:
            raise RuntimeError("Arm is not ready: " + (self._error or "no feedback yet"))

    def jog(self, vx=0, vy=0, vz=0, vpitch=0, vroll=0, vgrip=0, frame="cartesian", scale=1.0):
        """Set a normalized (-1..1) velocity. Must be refreshed every <0.3 s."""
        values = [_clamp(v, -1.0, 1.0) for v in _finite([vx, vy, vz, vpitch, vroll, vgrip])]
        if frame not in ("cartesian", "polar"):
            raise ValueError("Jog frame must be cartesian or polar")
        scale = _clamp(_finite([scale])[0], 0.05, 1.0)
        with self._lock:
            if any(values):
                self._require_ready()
            self._jog = values
            self._jog_frame = frame
            self._jog_scale = scale
            self._jog_time = time.monotonic()
            return self._target and list(self._target)

    def move_to(self, x=None, y=None, z=None, pitch=None, roll=None, grip=None, clamp=False,
                speed=1.0):
        """Smoothly move to an absolute pose; omitted values are kept.

        clamp=True moves to the nearest reachable point instead of refusing
        (used for map clicks)."""
        with self._lock:
            self._require_ready()
            goal = list(self._goal or self._target)
            for i, v in enumerate((x, y, z, pitch, roll, grip)):
                if v is not None:
                    goal[i] = _finite([v])[0]
            goal[3:] = [_clamp(goal[i], *POSE_LIMITS[i]) for i in (3, 4, 5)]
            base_pose = self._goal or self._target
            same_place = all(abs(a - b) < 1e-6 for a, b in zip(goal[:4], base_pose[:4]))
            if not same_place and not self.reachable(goal):
                if not clamp:
                    raise ValueError(self.explain_unreachable(goal))
                projected = self._project_reachable(goal)
                if projected is not None:
                    goal = projected
                else:
                    start = list(self._target)
                    if not self.reachable(start):
                        raise ValueError(self.explain_unreachable(goal))
                    goal, _ = self._limit_move(start, goal)
            self._jog = [0.0] * 6
            self._goal = goal
            self._goal_scale = _clamp(_finite([speed])[0], 0.05, 1.0)
            return list(goal)

    def _project_reachable(self, goal):
        """Nearest reachable point in the SAME direction from the base
        (same height and pitch): what a user clicking far away means."""
        x, y, z, pitch = goal[:4]
        angle = math.atan2(y, x)
        lim = kinematics.BASE_LIMIT - math.radians(4)
        angle = max(-lim, min(lim, angle))
        r_want = math.hypot(x, y)
        best = None
        for a, b in kinematics.radial_intervals(z, pitch, step=2):
            for r in (a + 2, b - 2, min(max(r_want, a + 2), b - 2)):
                cand = [r * math.cos(angle), r * math.sin(angle), z] + list(goal[3:])
                if self.pose_slack(cand) >= 0.004 and (best is None or abs(r - r_want) < best[0]):
                    best = (abs(r - r_want), cand)
        return best and best[1]

    def explain_unreachable(self, pose):
        x, y, z, pitch = pose[:4]
        r = math.hypot(x, y)
        if z < self._z_floor():
            return f"Z {z:.0f} mm is below the safe floor ({self._z_floor():.0f} mm)"
        spans = kinematics.radial_intervals(z, pitch)
        spans = [[max(a, self.cfg["arm_reach_min_mm"]), min(b, self.cfg["arm_reach_max_mm"])]
                 for a, b in spans]
        spans = [s for s in spans if s[0] < s[1]]
        where = ", ".join(f"{a:.0f}-{b:.0f} mm" for a, b in spans) or "nowhere"
        return (f"Out of reach: reach {r:.0f} mm at Z {z:.0f} with pitch {pitch:.0f} deg. "
                f"At this height and pitch the arm can reach {where}.")

    def move_xyz(self, x, y, z):
        return self.move_to(x=x, y=y, z=z)

    def nudge(self, dx=0, dy=0, dz=0, gripper=0):
        deltas = _finite([dx, dy, dz, 0, 0, gripper])
        if any(abs(v) > 10 for v in deltas[:3]):
            raise ValueError("Each XYZ nudge is limited to 10 mm")
        if abs(deltas[5]) > 5:
            raise ValueError("Each gripper nudge is limited to 5 degrees")
        with self._lock:
            self._require_ready()
            base = self._goal or self._target
            goal = self._clamp_pose([b + d for b, d in zip(base, deltas)])
            if not self.reachable(goal):
                raise ValueError(self.explain_unreachable(goal))
            self._jog = [0.0] * 6
            self._goal = goal
            self._goal_scale = 0.4
            return list(goal)

    def wake(self):
        """Servo torque ON (RoArm JSON T:210). Servos switch torque off to protect
        themselves after an overload (e.g. pushing into the table) - this turns it back on."""
        self._write(b'{"T":210,"cmd":1}\n')
        self._stalls = 0
        return True

    def hold(self):
        """Stop all arm motion now and hold the current physical position."""
        with self._lock:
            self._jog = [0.0] * 6
            self._goal = None
            self._vel = [0.0, 0.0, 0.0]
            self._goal_speed = 0.0
            m = self._measured
            if self._target is not None and m is not None \
                    and time.monotonic() - self._measured_time < 0.5:
                held = self._clamp_pose(list(m))
                if self.pose_slack(held) >= 0.004 or self.pose_slack(self._target) < 0:
                    self._target = held

    def refresh(self):
        with self._lock:
            self._require_ready()
            return list(self._measured or self._target)

    def state(self):
        with self._lock:
            age = (time.monotonic() - self._measured_time) if self._measured else None
            return {
                "connected": self._ser is not None and self._target is not None,
                "error": self._error,
                "port": self.port,
                "pose": self._measured and [round(v, 1) for v in self._measured],
                "target": self._target and [round(v, 1) for v in self._target],
                "goal": self._goal and [round(v, 1) for v in self._goal],
                "moving": bool(any(self._jog) or self._goal or any(abs(v) > 1 for v in self._vel)),
                "notice": self._notice,
                "link_ok": age is not None and age < 1.2,
                "stalls": getattr(self, "_stalls", 0),
                "servos_silent": bool(getattr(self, "_servos_silent", False)),
                "feedback_age_s": None if age is None else round(age, 2),
                "workspace": {k: self.cfg[k] for k in ("arm_reach_min_mm", "arm_reach_max_mm",
                                                       "arm_z_min_mm", "arm_z_max_mm")},
                "max_speed_mm_s": self.cfg["arm_max_speed_mm_s"],
                "z_floor": self._z_floor(),
                "reach_here": self._reach_here(),
            }

    def _reach_here(self):
        """Reachable reach bands at the current height and pitch (cached)."""
        t = self._target
        if t is None:
            return None
        key = (round(t[2] / 5) * 5, round(t[3] / 2) * 2)
        if getattr(self, "_reach_key", None) != key:
            spans = kinematics.radial_intervals(key[0], key[1])
            c = self.cfg
            self._reach_cache = [[max(a, c["arm_reach_min_mm"]), min(b, c["arm_reach_max_mm"])]
                                 for a, b in spans if min(b, c["arm_reach_max_mm"]) > max(a, c["arm_reach_min_mm"])]
            self._reach_key = key
        return self._reach_cache

    def close(self):
        self.hold()
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        with self._lock:
            ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
