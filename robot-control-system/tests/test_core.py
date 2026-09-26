import json
import threading
import time
import unittest

from robot_hub.arm import ArmDriver
from robot_hub.controller import RobotController
from robot_hub.rover import RoverDriver, mix_speeds
from robot_hub.safety import SafetyController
from qwen_agent import parse_action


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
    def __init__(self):
        self.pose = [300, 0, 200, 0, 0, 0]
        self.targets = []
        self.closed = False
    def pose_get(self): return self.pose.copy()
    def pose_ctrl(self, target):
        self.pose = list(target)
        self.targets.append(list(target))
        return b"sent"
    def disconnect(self): self.closed = True


class FakeCamera:
    def __init__(self, *args): pass
    def state(self): return {"connected": True, "error": "", "mode": "test",
                             "frame_age_seconds": 0, "detections": []}
    def frame(self): return b"jpeg"
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

    def test_arm_nudge_preserves_wrist(self):
        safety, fake = SafetyController(), FakeArm()
        arm = ArmDriver("COM7", safety, factory=lambda: fake)
        pose = arm.nudge(5, -5, 2, 3)
        self.assertEqual(pose, [305, -5, 202, 0, 0, 3])
        with self.assertRaises(ValueError): arm.nudge(11, 0, 0, 0)
        arm.close()
        self.assertTrue(fake.closed)

    def test_human_xyz_target_preserves_wrist_and_limits_jump(self):
        safety, fake = SafetyController(), FakeArm()
        arm = ArmDriver("COM7", safety, factory=lambda: fake)
        pose = arm.move_xyz(325, 20, 180)
        self.assertEqual(pose, [325, 20, 180, 0, 0, 0])
        with self.assertRaises(ValueError):
            arm.move_xyz(450, 20, 180)
        arm.close()

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
        arm._last_connect_attempt = 0
        self.assertTrue(arm.state()["connected"])
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
        result = controller.action({"action": "arm_nudge", "dx_mm": 5}, "qwen")
        self.assertEqual(result["result"]["pose"][0], 305)
        with self.assertRaises(ValueError):
            controller.action({"action": "arm_target", "x_mm": 310,
                               "y_mm": 0, "z_mm": 200}, "qwen")
        with self.assertRaises(ValueError):
            controller.action({"action": "arm_nudge", "dx_mm": 50}, "qwen")
        controller.close()

    def test_qwen_action_validation(self):
        action = parse_action('{"action":"drive","linear":0.2,"turn":0,"duration_ms":200}')
        self.assertEqual(action["action"], "drive")
        with self.assertRaises(ValueError):
            parse_action('{"action":"drive","linear":2,"turn":0,"duration_ms":200}')
        with self.assertRaises(ValueError):
            parse_action('{"action":"enable"}')
        with self.assertRaises(ValueError):
            parse_action('{"action":"arm_nudge","dx_mm":50}')


if __name__ == "__main__":
    unittest.main()
