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
        self.assertEqual(mix_speeds(1, 0, 700), (700, 700))
        self.assertEqual(mix_speeds(0, 1, 700), (700, -700))
        self.assertEqual(mix_speeds(9, -9, 700), (0, 700))

    def test_rover_duration_watchdog_sends_stop(self):
        safety, opener = SafetyController(), FakeOpener()
        rover = RoverDriver("http://robot", 700, safety, opener)
        safety.enable_human()
        rover.command(1, 0, 100)
        time.sleep(.35)
        state = rover.state()
        rover.close()
        decoded = [json.loads(__import__('urllib.parse').parse.parse_qs(
            __import__('urllib.parse').parse.urlparse(url).query)['json'][0]) for url in opener.urls]
        self.assertTrue(any(item["L"] == 700 for item in decoded))
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

    def test_qwen_agent_plan_validation(self):
        import qwen_agent as qa
        self.assertEqual(qa.coord_mode("qwen2.5vl:7b", "auto"), "pixel")
        self.assertEqual(qa.coord_mode("qwen3-vl:8b", "auto"), "norm1000")
        self.assertEqual(qa.to_pixel([500, 500], (640, 400), "norm1000"), (320, 200))
        self.assertEqual(qa.to_pixel([100, 50, 140, 90], (640, 400), "pixel"), (120, 70))
        with self.assertRaises(ValueError):
            qa.to_pixel([700, 10], (640, 400), "pixel")

        class Bot:
            def state(self): return {"task": {"holding": False}}
        good = ('```json\n{"observation":"red block, bowl","steps":['
                '{"action":"pick","object":"red block","point":[300,150]},'
                '{"action":"place","target":"bowl","point":[480,240]}]}\n```')
        orig = qa.call_model
        try:
            qa.call_model = lambda *a, **k: good
            obs, steps, _ = qa.plan(Bot(), b"", (640, 400), "goal", "pixel", "m")
            self.assertEqual([s["action"] for s in steps], ["pick", "place"])
            self.assertEqual(steps[0]["pixel"], (300, 150))
            qa.call_model = lambda *a, **k: '{"steps":[{"action":"place","point":[1,1]}]}'
            with self.assertRaises(ValueError):          # place while empty
                qa.plan(Bot(), b"", (640, 400), "goal", "pixel", "m")
            qa.call_model = lambda *a, **k: '{"steps":[{"action":"self_destruct"}]}'
            with self.assertRaises(ValueError):
                qa.plan(Bot(), b"", (640, 400), "goal", "pixel", "m")
        finally:
            qa.call_model = orig


if __name__ == "__main__":
    unittest.main()
