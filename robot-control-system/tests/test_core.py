import json
import threading
import time
import unittest

from robot_hub.arm import ArmDriver
from robot_hub.controller import RobotController
from robot_hub.rover import RoverDriver, mix_speeds
from robot_hub.safety import SafetyController


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, count=None): return b"OK"


class FakeOpener:
    def __init__(self):
        self.urls = []
        self.fail = False
    def open(self, url, timeout):
        self.urls.append(url)
        if self.fail:
            raise OSError("offline")
        return Response()


class FakeArm:
    """Simulates RoArm-M3 firmware on the serial line (T:105 / T:1041)."""
    def __init__(self, pose=(300, 0, 200, 0, 0, 0)):
        import math
        self.m = math
        self.pose = list(pose)
        self.targets = []
        self.out = b""
        self.closed = False
        self.lock = threading.Lock()
    def _feedback(self):
        m = self.m
        x, y, z, p, r, g = self.pose
        return (json.dumps({"T": 1051, "x": x, "y": y, "z": z, "tit": m.radians(p),
                            "b": 0, "s": 0, "e": 0, "t": 0, "r": m.radians(r),
                            "g": m.pi - m.radians(g)}) + "\r\n").encode()
    def write(self, data):
        for line in data.decode().strip().splitlines():
            cmd = json.loads(line)
            self.__dict__.setdefault("raw", []).append(cmd)
            if getattr(self, "mute", False) and cmd["T"] == 105:
                continue
            with self.lock:
                if cmd["T"] == 105:
                    self.out += self._feedback()
                elif cmd["T"] == 1041:
                    m = self.m
                    self.pose = [cmd["x"], cmd["y"], cmd["z"], m.degrees(cmd["t"]),
                                 m.degrees(cmd["r"]), 180 - m.degrees(cmd["g"])]
                    self.targets.append(list(self.pose))
    def read(self, n):
        with self.lock:
            data, self.out = self.out[:n], self.out[n:]
        if not data:
            time.sleep(.01)
        return data
    def close(self): self.closed = True


def wait_until(fn, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(.02)
    return False


class FakeCamera:
    def __init__(self, *args): pass
    def state(self): return {"connected": True, "error": "", "mode": "test",
                             "frame_age_seconds": 0, "detections": []}
    def frame(self): return b"jpeg"
    def size(self): return (640, 400)
    def close(self): pass


class CoreTests(unittest.TestCase):
    def test_speed_mixing_and_clamping(self):
        self.assertEqual(mix_speeds(1, 0), (1800, 1800))        # same as the rover's own page
        self.assertEqual(mix_speeds(0, 1), (1800, -1800))
        self.assertEqual(mix_speeds(9, -9), (0, 1800))
        self.assertEqual(mix_speeds(0.6, 0), (1080, 1080))

    def test_rover_duration_watchdog_sends_stop(self):
        safety, opener = SafetyController(), FakeOpener()
        rover = RoverDriver("http://robot", 700, safety, opener)
        safety.enable_human()
        rover.command(1, 0, 500)
        time.sleep(.2)
        rover.command(1, 0, 500)
        time.sleep(.35)
        rover.command(1, 0, 500)            # held ~0.8 s: ramps up to full speed
        time.sleep(1.0)
        state = rover.state()
        rover.close()
        decoded = [json.loads(__import__('urllib.parse').parse.parse_qs(
            __import__('urllib.parse').parse.urlparse(url).query)['json'][0]) for url in opener.urls]
        self.assertTrue(any(item["L"] == 700 for item in decoded))     # rover_max_speed 700
        ups = [item["L"] for item in decoded if item["L"] > 0]
        self.assertLess(ups[0], 300)            # starts gently (no jump to full speed)
        self.assertEqual((decoded[-1]["L"], decoded[-1]["R"]), (0, 0))
        self.assertTrue(state["connected"])

    def make_arm(self, fake=None, **cfg):
        safety, fake = SafetyController(), fake or FakeArm()
        safety.enable_human()
        arm = ArmDriver("COM7", safety, factory=lambda: fake, config=cfg)
        self.assertTrue(wait_until(lambda: arm.state()["connected"]))
        return arm, fake, safety

    def test_protocol_round_trip(self):
        from robot_hub.arm import encode_pose, decode_feedback
        line = encode_pose([300, -20, 150, 10, -30, 45]).decode()
        cmd = json.loads(line)
        self.assertEqual(cmd["T"], 1041)
        fake = FakeArm()
        fake.write(line.encode())
        pose = decode_feedback(fake._feedback().decode())
        for a, b in zip(pose, [300, -20, 150, 10, -30, 45]):
            self.assertAlmostEqual(a, b, places=1)
        self.assertIsNone(decode_feedback('{"T":1,"L":0}'))
        self.assertIsNone(decode_feedback("garbage"))

    def test_connect_starts_from_actual_pose_without_jump(self):
        arm, fake, _ = self.make_arm(FakeArm((250, 40, 120, 5, 0, 20)))
        time.sleep(.2)
        self.assertEqual(arm.state()["target"][:3], [250, 40, 120])
        self.assertEqual(fake.targets, [])  # nothing sent while idle
        arm.close()

    def test_jog_streams_smooth_velocity_and_deadman_stops(self):
        arm, fake, _ = self.make_arm(arm_max_speed_mm_s=200)
        for _ in range(10):          # hold "forward" for ~0.5 s
            arm.jog(vx=1)
            time.sleep(.05)
        moved = fake.pose[0] - 300
        self.assertTrue(60 < moved < 190, moved)       # ~200 mm/s, ramped
        self.assertGreater(len(fake.targets), 10)      # many small steps
        steps = [b[0] - a[0] for a, b in zip(fake.targets, fake.targets[1:])]
        self.assertLess(max(steps), 12)                # no big jumps
        time.sleep(.8)                                 # released: dead-man + braking
        x = fake.pose[0]
        time.sleep(.3)
        self.assertEqual(fake.pose[0], x)
        arm.close()

    def test_polar_jog_moves_along_arm_heading(self):
        arm, fake, _ = self.make_arm(FakeArm((0, 300, 100, 0, 0, 0)))
        for _ in range(6):
            arm.jog(vx=1, frame="polar")
            time.sleep(.05)
        self.assertAlmostEqual(fake.pose[0], 0, delta=1)
        self.assertGreater(fake.pose[1], 320)
        arm.close()

    def test_goal_moves_smoothly_and_arrives(self):
        arm, fake, _ = self.make_arm(arm_goal_speed_mm_s=400)
        goal = arm.move_xyz(380, 60, 150)
        self.assertEqual(goal[:3], [380, 60, 150])
        self.assertTrue(wait_until(lambda: fake.pose[:3] == [380, 60, 150], 2))
        self.assertGreater(len(fake.targets), 5)
        self.assertFalse(arm.state()["moving"])
        with self.assertRaises(ValueError):
            arm.move_xyz(900, 0, 100)
        arm.close()

    def test_workspace_clamp_and_stop_hold(self):
        arm, fake, safety = self.make_arm(FakeArm((440, 0, 200, 0, 0, 0)),
                                          arm_max_speed_mm_s=400)
        from robot_hub import kinematics
        for _ in range(8):
            arm.jog(vx=1)
            time.sleep(.05)
        self.assertTrue(kinematics.reachable(*fake.pose[:4]))
        arm.move_xyz(300, 0, 200)
        time.sleep(.1)
        safety.latch("test", "test")
        arm.hold()
        time.sleep(.15)
        x = fake.pose[0]
        time.sleep(.3)
        self.assertAlmostEqual(fake.pose[0], x, delta=2)
        with self.assertRaises(RuntimeError):
            safety.require_enabled()
        arm.close()

    def test_far_goal_arcs_around_base_and_click_clamps(self):
        arm, fake, _ = self.make_arm(FakeArm((0, 300, 150, 0, 0, 0)), arm_goal_speed_mm_s=600,
                                     arm_accel_mm_s2=3000)
        arm.move_to(x=0, y=-300)          # other side of the base
        self.assertTrue(wait_until(lambda: [round(v) for v in fake.pose[:2]] == [0, -300], 3))
        import math
        closest = min(math.hypot(p[0], p[1]) for p in fake.targets)
        self.assertGreater(closest, 280)  # went around, not through the base
        with self.assertRaises(ValueError):
            arm.move_to(x=2000, y=0)
        goal = arm.move_to(x=2000, y=0, clamp=True)
        self.assertTrue(arm.reachable(goal))
        self.assertGreater(goal[0], 250)
        import math
        goal = arm.move_to(x=-400, y=420, clamp=True)       # far, behind-left
        self.assertTrue(arm.reachable(goal))
        self.assertAlmostEqual(math.degrees(math.atan2(goal[1], goal[0])), 133.6, delta=1)
        arm.close()

    def test_never_sends_unreachable_pose(self):
        """Regression: out-of-reach targets made the real arm shoot away."""
        from robot_hub import kinematics
        arm, fake, _ = self.make_arm(FakeArm((300, 0, 150, 0, 0, 0)), arm_max_speed_mm_s=600)
        for v in ([1, 0, -1, 0], [-1, 0, -1, 0], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, -1, 1]):
            for _ in range(12):
                arm.jog(vx=v[0], vy=v[1], vz=v[2], vpitch=v[3])
                time.sleep(.03)
        bad = [p for p in fake.targets if not kinematics.reachable(*p[:4])]
        self.assertEqual(bad, [])
        self.assertGreater(len(fake.targets), 20)
        with self.assertRaises(ValueError) as ctx:
            arm.move_to(x=200, y=0, z=-120, pitch=0)   # table height, gripper level
        self.assertIn("Out of reach", str(ctx.exception))
        arm.close()

    def test_kinematics_match_firmware_fk(self):
        import math, random
        from robot_hub import kinematics as k
        random.seed(3)
        checked = 0
        for _ in range(3000):
            x, y, z, p = (random.uniform(-450, 450), random.uniform(-450, 450),
                          random.uniform(-200, 450), random.uniform(-90, 90))
            if k.reachable(x, y, z, p):
                f = k.fk(*k.ik(x, y, z, math.radians(p)))
                self.assertAlmostEqual(f[0], x, places=6)
                self.assertAlmostEqual(f[2], z, places=6)
                self.assertAlmostEqual(math.degrees(f[3]), p, places=6)
                checked += 1
        self.assertGreater(checked, 300)

    def test_far_goal_never_spins_base_through_180(self):
        """Base servo cannot wrap: 170 deg -> -170 deg must go round the front."""
        import math
        from robot_hub import kinematics
        r = 300
        a0, a1 = math.radians(170), math.radians(-170)
        arm, fake, _ = self.make_arm(FakeArm((r * math.cos(a0), r * math.sin(a0), 150, 0, 0, 0)),
                                     arm_goal_speed_mm_s=2000, arm_accel_mm_s2=20000)
        arm.move_to(x=r * math.cos(a1), y=r * math.sin(a1))
        self.assertTrue(wait_until(lambda: arm.state()["goal"] is None, 6))
        angles = [math.degrees(math.atan2(p[1], p[0])) for p in fake.targets]
        self.assertTrue(all(abs(a) <= 175.5 for a in angles), max(angles, key=abs))
        self.assertTrue(any(abs(a) < 10 for a in angles))          # went via the front
        steps = [abs(b - a) for a, b in zip(angles, angles[1:])]
        self.assertLess(max(steps), 20)                             # no sudden spin
        self.assertEqual([p for p in fake.targets if not kinematics.reachable(*p[:4])], [])
        arm.close()

    def test_side_view_goal_changes_reach_and_height_safely(self):
        from robot_hub import kinematics
        arm, fake, _ = self.make_arm(FakeArm((180, 0, 250, 0, 0, 0)),
                                     arm_goal_speed_mm_s=2000, arm_accel_mm_s2=20000)
        arm.move_to(x=440, y=0, z=-60, pitch=0)
        self.assertTrue(wait_until(lambda: arm.state()["goal"] is None, 6))
        self.assertAlmostEqual(fake.pose[0], 440, delta=0.5)
        self.assertEqual([p for p in fake.targets if not kinematics.reachable(*p[:4])], [])
        arm.close()

    def test_rover_wifi_glitch_does_not_stop_everything(self):
        safety, opener = SafetyController(), FakeOpener()
        rover = RoverDriver("http://robot", 700, safety, opener)
        safety.enable_human()
        opener.fail = True
        rover.command(1, 0, 300)
        time.sleep(.5)
        self.assertFalse(safety.state()["latched"])
        self.assertFalse(rover.state()["connected"])
        opener.fail = False
        time.sleep(.3)
        self.assertTrue(rover.state()["connected"])
        rover.close()

    def test_rover_is_found_automatically(self):
        class Page:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self, n=None): return self.body
        class Net:
            def __init__(self): self.rover = "10.9.8.77"
            def open(self, url, timeout):
                host = url.split("/")[2]
                if host != self.rover:
                    raise OSError("no route")
                return Page(b"<span>VOLTAGE</span><span>RSSI</span>" if url.endswith("/") else b"OK")
        net, safety = Net(), SafetyController()
        rover = RoverDriver("auto", 700, safety, opener=net, discover_prefixes=["10.9.8."])
        self.assertTrue(wait_until(lambda: rover.state()["connected"], 6))
        self.assertEqual(rover.state()["url"], "http://10.9.8.77")
        net.rover = "10.9.8.12"                     # rover gets a new address
        self.assertTrue(wait_until(lambda: rover.state()["url"] == "http://10.9.8.12", 20))
        self.assertTrue(wait_until(lambda: rover.state()["connected"], 3))
        rover.close()

    def test_blocked_arm_stops_with_notice(self):
        class StuckArm(FakeArm):
            def write(self, data):
                if b'"T":1041' in data:
                    return          # servos never move
                super().write(data)
        arm, fake, _ = self.make_arm(StuckArm(), arm_goal_speed_mm_s=400)
        arm.move_xyz(420, 0, 200)
        self.assertTrue(wait_until(lambda: arm.state()["notice"] is not None, 4))
        st = arm.state()
        self.assertFalse(st["moving"])
        self.assertLess(abs(st["target"][0] - 300), 1)
        arm.close()

    def _calibrated(self, tmpdir):
        """Synthetic overhead camera: 1 px = 1 mm, arm base at pixel (320, 380)."""
        import os
        from robot_hub.vision import Calibration
        cal = Calibration(os.path.join(tmpdir, "cal.json"))
        for u, v in ((120, 100), (520, 100), (520, 330), (120, 330), (320, 200)):
            x, y = 380 - v, 320 - u
            cal.add_point([u, v], [x, y, -110], [640, 400])
        return cal

    def test_calibration_maps_pixels_to_arm(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cal = self._calibrated(d)
            self.assertTrue(cal.ready)
            x, y = cal.pixel_to_arm(300, 150)
            self.assertAlmostEqual(x, 230, delta=.01)
            self.assertAlmostEqual(y, 20, delta=.01)
            u, v = cal.arm_to_pixel(230, 20)
            self.assertAlmostEqual(u, 300, delta=.01)
            x2, _ = cal.pixel_to_arm(600, 300, image_size=[1280, 800])   # scaled image
            self.assertAlmostEqual(x2, 230, delta=.01)
            self.assertEqual(cal.table_z, -110)
            from robot_hub.vision import Calibration
            import os
            self.assertTrue(Calibration(os.path.join(d, "cal.json")).ready)  # persisted

    def test_pick_and_place_by_pixel(self):
        import tempfile
        from robot_hub.tasks import TaskRunner
        from robot_hub import kinematics
        with tempfile.TemporaryDirectory() as d:
            cal = self._calibrated(d)
            arm, fake, safety = self.make_arm(FakeArm((235, 0, 234, 0, 0, 30)),
                                              arm_goal_speed_mm_s=900, arm_accel_mm_s2=6000)
            arm.table_z = cal.table_z
            runner = TaskRunner(arm, safety, cal)
            runner.submit({"task": "pick", "pixel": [300, 150], "speed": 1})
            self.assertTrue(wait_until(lambda: runner.status()["state"] != "running", 15))
            st = runner.status()
            self.assertEqual(st["state"], "done", st)
            self.assertTrue(runner.holding)
            low = min(fake.targets, key=lambda p: p[2])
            self.assertAlmostEqual(low[0], 230, delta=1)
            self.assertAlmostEqual(low[1], 20, delta=1)
            self.assertAlmostEqual(low[2], -110 + 15, delta=1)
            self.assertAlmostEqual(low[3], 90, delta=1)          # gripper pointing down
            self.assertAlmostEqual(fake.pose[5], 0, delta=1)      # closed
            self.assertEqual([p for p in fake.targets if not kinematics.reachable(*p[:4])], [])
            runner.submit({"task": "place", "xy": [200, -80], "speed": 1})
            self.assertTrue(wait_until(lambda: runner.status()["state"] != "running", 15))
            self.assertEqual(runner.status()["state"], "done")
            self.assertFalse(runner.holding)
            self.assertAlmostEqual(fake.pose[5], 60, delta=1)     # opened
            with self.assertRaises(ValueError) as ctx:            # too far: refused up front
                runner.submit({"task": "pick", "xy": [600, 0]})
            self.assertIn("out of the arm's pick range", str(ctx.exception))
            arm.close()

    def test_stop_cancels_task(self):
        import tempfile
        from robot_hub.tasks import TaskRunner
        with tempfile.TemporaryDirectory() as d:
            cal = self._calibrated(d)
            arm, fake, safety = self.make_arm(FakeArm((235, 0, 234, 0, 0, 30)),
                                              arm_goal_speed_mm_s=60)
            arm.table_z = cal.table_z
            runner = TaskRunner(arm, safety, cal)
            runner.submit({"task": "move_above", "xy": [150, 150]})
            time.sleep(.4)
            safety.latch("test", "test")
            runner.cancel("STOP")
            self.assertTrue(wait_until(lambda: runner.status()["state"] == "failed", 3))
            x = fake.pose[0]
            time.sleep(.3)
            self.assertAlmostEqual(fake.pose[0], x, delta=2)
            arm.close()

    def test_grip_force_sent_once_and_blind_motion_frozen(self):
        arm, fake, _ = self.make_arm(gripper_torque=300, arm_goal_speed_mm_s=40)
        time.sleep(.3)
        self.assertEqual([c for c in fake.raw if c["T"] == 107], [{"T": 107, "tor": 300}])
        arm.move_xyz(360, 80, 150)
        time.sleep(.05)
        fake.mute = True                    # feedback stops (USB glitch)
        time.sleep(2.0)
        st = arm.state()
        self.assertIsNone(st["goal"])
        self.assertIn("lost position feedback", st["notice"]["message"])
        x = fake.pose[0]
        time.sleep(.3)
        self.assertEqual(fake.pose[0], x)   # no blind motion
        fake.mute = False
        arm.close()

    def test_nonsense_servo_readings_block_motion(self):
        arm, fake, _ = self.make_arm()
        sent = len(fake.targets)
        fake.pose = [57, 0, -223, -540, 180, 180]      # what the arm reports with servos unpowered
        time.sleep(.4)
        st = arm.state()
        self.assertFalse(st["connected"])
        self.assertIn("servos not answering", st["error"])
        with self.assertRaises(RuntimeError):
            arm.jog(vx=1)
        with self.assertRaises(RuntimeError):
            arm.move_xyz(300, 0, 150)
        self.assertEqual(len(fake.targets), sent)       # nothing sent to the arm
        fake.pose = [300, 0, 200, 0, 0, 20]             # power restored
        self.assertTrue(wait_until(lambda: arm.state()["connected"], 2))
        self.assertEqual(arm.state()["target"][:3], [300, 0, 200])   # resync, no jump
        arm.close()

    def test_park_task(self):
        import tempfile
        from robot_hub.tasks import TaskRunner
        with tempfile.TemporaryDirectory() as d:
            cal = self._calibrated(d)
            arm, fake, safety = self.make_arm(arm_goal_speed_mm_s=900, arm_accel_mm_s2=6000)
            runner = TaskRunner(arm, safety, cal)
            runner.submit({"task": "park"})
            self.assertTrue(wait_until(lambda: runner.status()["state"] == "done", 8))
            self.assertEqual([round(v) for v in fake.pose[:4]], [170, 0, 80, 30])
            arm.close()

    def test_arm_nudge_preserves_wrist(self):
        arm, fake, _ = self.make_arm()
        pose = arm.nudge(5, -5, 2, 3)
        self.assertEqual([round(v, 3) for v in pose], [305, -5, 202, 0, 0, 3])
        self.assertTrue(wait_until(lambda: [round(v, 1) for v in fake.pose] == [305, -5, 202, 0, 0, 3]))
        with self.assertRaises(ValueError): arm.nudge(11, 0, 0, 0)
        arm.close()
        self.assertTrue(fake.closed)

    def test_arm_encoder_noise_is_clamped(self):
        from robot_hub.arm import validate_pose
        self.assertEqual(validate_pose([300, 0, 200, 0, 0, -.09])[-1], 0)

    def test_arm_reconnects_after_initial_failure(self):
        safety, fake, attempts = SafetyController(), FakeArm(), []
        def factory():
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("not ready")
            return fake
        arm = ArmDriver("COM7", safety, factory=factory)
        self.assertFalse(arm.state()["connected"])
        self.assertTrue(wait_until(lambda: arm.state()["connected"], 3.5))
        arm.close()

    def test_controller_starts_latched_and_bounds_agent(self):
        fake_arm, opener = FakeArm(), FakeOpener()
        config = {"rover_url": "http://robot", "rover_max_speed": 700,
                  "arm_port": "COM7", "camera_mode": "test",
                  "camera_width": 1, "camera_height": 1, "camera_fps": 1}
        controller = RobotController(config, arm_factory=lambda: fake_arm,
                                     rover_opener=opener, camera_factory=FakeCamera)
        with self.assertRaises(RuntimeError):
            controller.action({"action": "drive", "linear": 1, "duration_ms": 100}, "qwen")
        controller.enable_human()
        self.assertTrue(wait_until(lambda: controller.state()["arm"]["connected"]))
        result = controller.action({"action": "arm_nudge", "dx_mm": 5}, "qwen")
        self.assertEqual(result["result"]["pose"][0], 305)
        controller.arm_jog({"vz": 1})
        controller.emergency_stop("test")
        with self.assertRaises(RuntimeError):
            controller.arm_jog({"vz": 1})
        controller.enable_human()
        with self.assertRaises(ValueError):
            controller.action({"action": "arm_target", "x_mm": 310,
                               "y_mm": 0, "z_mm": 200}, "qwen")
        with self.assertRaises(ValueError):
            controller.action({"action": "arm_nudge", "dx_mm": 50}, "qwen")
        controller.close()

    def test_vision_model_coordinates_and_validation(self):
        from robot_hub.ai import VisionModel, extract_json
        import cv2, numpy as np
        jpeg = cv2.imencode(".jpg", np.zeros((400, 640, 3), np.uint8))[1].tobytes()
        m = VisionModel(model="Qwen/Qwen2.5-VL-32B-Instruct")
        self.assertEqual(m.mode(), "pixel")
        send, seen, scale = m.prepare(jpeg, (640, 400))
        self.assertEqual(seen, (644, 392))                       # multiples of 28
        img = cv2.imdecode(np.frombuffer(send, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (392, 644))
        box, px = m.to_camera([322, 196, 322, 196], seen, scale, (640, 400))
        self.assertEqual(px, (320.0, 200.0))                     # scaled back exactly
        self.assertEqual(VisionModel(model="qwen3-vl-8b").mode(), "norm1000")
        self.assertEqual(extract_json('```json\n[{"a":1}]\n```'), [{"a": 1}])

        replies = []
        m.ask = lambda prompt, jpeg, size, max_tokens=700: (replies.pop(0), (644, 392), (640 / 644, 400 / 392))
        replies.append('{"observation":"block, bowl","steps":['
                       '{"action":"pick","object":"red block","bbox_2d":[290,135,310,160]},'
                       '{"action":"place","target":"bowl","bbox_2d":[460,215,500,255]}]}')
        plan = m.plan(jpeg, (640, 400), "goal")
        self.assertEqual([s["action"] for s in plan["steps"]], ["pick", "place"])
        self.assertAlmostEqual(plan["steps"][0]["pixel"][0], 300 * 640 / 644, delta=0.1)
        replies.append('{"steps":[{"action":"place","bbox_2d":[1,1,2,2]}]}')
        with self.assertRaises(ValueError):                      # place while empty
            m.plan(jpeg, (640, 400), "goal")
        replies.append('{"steps":[{"action":"self_destruct"}]}')
        with self.assertRaises(ValueError):
            m.plan(jpeg, (640, 400), "goal")
        replies.append('[{"bbox_2d":[100,100,120,130],"label":"red block"},{"bbox_2d":[9999,1,9999,2]}]')
        found = m.locate(jpeg, (640, 400), "red block")["objects"]
        self.assertEqual(len(found), 1)                          # off-image box dropped

class CameraSourceTests(unittest.TestCase):
    def test_parse_source(self):
        from robot_hub.camera import parse_source
        self.assertEqual(parse_source("webcam:2"), ("webcam", 2))
        self.assertEqual(parse_source("oak"), ("oak", None))
        self.assertEqual(parse_source(""), ("auto", None))
        self.assertEqual(parse_source("http://10.0.0.5:8080/video"), ("url", "http://10.0.0.5:8080/video"))
        self.assertEqual(parse_source("url:rtsp://cam/1"), ("url", "rtsp://cam/1"))


if __name__ == "__main__":
    unittest.main()
