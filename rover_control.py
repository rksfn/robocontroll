#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pygame>=2.6.1,<3"]
# ///

import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

ROVER_URL = os.getenv("ROVER_URL", "http://10.176.60.15").rstrip("/")
try:
    MAX_SPEED = max(0, min(int(os.getenv("ROVER_MAX_SPEED", "900")), 1800))
except ValueError:
    MAX_SPEED = 900
DEADZONE = 0.12


def motor_speeds(x, y):
    x = 0 if abs(x) < DEADZONE else x
    y = 0 if abs(y) < DEADZONE else y
    return (
        round(max(-1, min(1, y + x)) * MAX_SPEED),
        round(max(-1, min(1, y - x)) * MAX_SPEED),
    )


def command_url(left, right):
    command = json.dumps({"T": 1, "L": left, "R": right}, separators=(",", ":"))
    return f"{ROVER_URL}/js?{urllib.parse.urlencode({'json': command})}"


if "--self-test" in sys.argv:
    assert motor_speeds(0, 1) == (MAX_SPEED, MAX_SPEED)
    assert motor_speeds(1, 0) == (MAX_SPEED, -MAX_SPEED)
    assert motor_speeds(0.05, -0.05) == (0, 0)
    assert "%7B%22T%22%3A1" in command_url(0, 0)
    print("Checks passed")
    raise SystemExit

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:
    raise SystemExit("pygame is required: python -m pip install pygame")


class Rover:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.last_sent = None
        self.failures = 0
        self.closed = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def send(self, left, right):
        with self.condition:
            self.pending = (left, right)
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                command = self.pending
                self.pending = None
            try:
                with urllib.request.urlopen(command_url(*command), timeout=1) as response:
                    response.read()
                with self.condition:
                    self.last_sent = command
                    self.failures = 0
                    self.condition.notify_all()
            except Exception:
                with self.condition:
                    if self.pending is None:
                        self.pending = command
                    self.failures += 1
                    if self.failures == 5:
                        print("\nRover unavailable; still retrying", file=sys.stderr)
                time.sleep(0.1)

    def stop(self):
        deadline = time.monotonic() + 4
        with self.condition:
            self.pending = (0, 0)
            self.last_sent = None
            self.condition.notify()
            while self.last_sent != (0, 0) and time.monotonic() < deadline:
                self.condition.wait(deadline - time.monotonic())
            self.closed = True
            self.condition.notify()
        self.worker.join(timeout=1)


def controller():
    if pygame.joystick.get_count() == 0:
        return None
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick


def main():
    pygame.init()
    pygame.joystick.init()
    rover = Rover()
    gamepad = None
    last = None
    last_sent = 0

    print("Xbox controller: hold A and use the left stick. Release A to stop.")
    print(f"Speed: {MAX_SPEED}/1800 (set ROVER_MAX_SPEED to change); rover: {ROVER_URL}")
    try:
        while True:
            pygame.event.pump()
            if gamepad is None or not gamepad.get_init():
                gamepad = controller()
            if gamepad is None:
                command = (0, 0)
                print("\rNo controller detected - STOP                    ", end="", flush=True)
            else:
                try:
                    x, y = gamepad.get_axis(0), -gamepad.get_axis(1)
                    armed = gamepad.get_button(0) == 1
                    command = motor_speeds(x, y) if armed else (0, 0)
                    print(
                        f"\r{gamepad.get_name()}  A:{'HELD' if armed else 'UP'}  "
                        f"stick x:{x:+.2f} y:{y:+.2f} -> L:{command[0]:+5d} R:{command[1]:+5d}   ",
                        end="",
                        flush=True,
                    )
                except pygame.error:
                    gamepad = None
                    continue

            now = time.monotonic()
            if command != last or now - last_sent >= 1:
                rover.send(*command)
                last, last_sent = command, now
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rover.stop()
        pygame.quit()
        print("\nSTOP")


if __name__ == "__main__":
    main()
