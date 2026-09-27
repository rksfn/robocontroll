import argparse
import os
import json
import shutil
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .controller import RobotController


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "static" / "dashboard.html"


class RobotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows, address reuse lets a second hub silently share the port with
    # an old (possibly hidden) one; refuse instead so it is obvious.
    allow_reuse_address = os.name != "nt"


def load_config(path):
    path = Path(path)
    example = path.with_name("config.example.json")
    if not path.exists() and example.exists():
        # first start on a new machine: personal settings live in config.json (not in git)
        shutil.copyfile(example, path)
        print(f"Created {path.name} from {example.name} - edit it for this computer (arm COM port, rover IP, camera)")
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"rover_url", "rover_max_speed", "arm_port", "server_host", "server_port"}
    missing = required.difference(config)
    if missing:
        raise ValueError("Missing configuration: " + ", ".join(sorted(missing)))
    return config


def make_handler(controller):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RobotHub/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if "/api/arm/jog" in self.requestline or "/api/state" in self.requestline:
                return
            print("HTTP", self.address_string(), fmt % args)

        def _headers(self, status, content_type, length=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()

        def json_response(self, status, data):
            body = json.dumps(data, separators=(",", ":")).encode()
            self._headers(status, "application/json", len(body))
            self.wfile.write(body)

        def body_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            if length < 0 or length > 8192:
                raise ValueError("JSON request size is invalid")
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            try:
                if url.path == "/api/task":
                    self.json_response(200, controller.tasks.status())
                    return
                if url.path == "/api/calibration":
                    self.json_response(200, controller.calibration.status())
                    return
                if url.path == "/api/camera/sources":
                    self.json_response(200, controller.camera_sources(
                        scan="scan=1" in (url.query or "")))
                    return
                if url.path == "/api/ai/status":
                    self.json_response(200, dict(controller.ai.status(), tunnel=controller.tunnel_state()))
                    return
                if url.path == "/api/arm/reach":
                    from .kinematics import reach_grid
                    pitch = round(float(query.get("pitch", ["0"])[0]) / 5) * 5
                    self.json_response(200, reach_grid(pitch))
                    return
                if url.path == "/api/pixel":
                    u, v = float(query["u"][0]), float(query["v"][0])
                    self.json_response(200, controller.pixel_info(u, v))
                    return
                if url.path.startswith("/camera.mjpg"):
                    self._mjpeg(query.get("cam", [""])[0])
                    return
                if url.path == "/api/scene/point":
                    box = query.get("box", [None])[0]
                    box = [float(v) for v in box.split(",")] if box else None
                    self.json_response(200, controller.scene_point(query["u"][0], query["v"][0], box))
                    return
                if self.path == "/":
                    body = DASHBOARD.read_bytes()
                    self._headers(200, "text/html; charset=utf-8", len(body))
                    self.wfile.write(body)
                elif self.path == "/api/state":
                    self.json_response(200, controller.state())
                elif url.path == "/api/frame.jpg":
                    cam = controller.scene if query.get("cam", [""])[0] == "scene" else controller.camera
                    frame = cam.frame() if cam is not None else None
                    if frame is None:
                        self.json_response(503, {"error": "Camera frame not available"})
                    else:
                        self._headers(200, "image/jpeg", len(frame))
                        self.wfile.write(frame)
                else:
                    self.json_response(404, {"error": "Not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.json_response(500, {"error": str(exc)})

        def _mjpeg(self, which=""):
            self._headers(200, "multipart/x-mixed-replace; boundary=frame")
            last, last_time = None, 0.0
            while True:
                cam = controller.scene if which == "scene" else controller.camera
                frame = cam.frame() if cam is not None else None
                # resend unchanged frames too: browsers draw a frame when the
                # next one starts, so a static image would never appear.
                if frame is not None and (frame is not last or time.time() - last_time > .5):
                    last_time = time.time()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    last = frame
                time.sleep(.05)

        def do_POST(self):
            try:
                data = self.body_json()
                if self.path == "/api/action":
                    source = str(data.pop("source", "human"))
                    self.json_response(200, controller.action(data, source))
                elif self.path == "/api/task":
                    source = str(data.pop("source", "human"))
                    self.json_response(200, controller.task(data, source))
                elif self.path == "/api/rover/url":
                    self.json_response(200, controller.set_rover_url(data))
                elif self.path == "/api/camera/source":
                    self.json_response(200, controller.set_camera_source(data))
                elif self.path == "/api/ai/config":
                    self.json_response(200, controller.ai_configure(data))
                elif self.path == "/api/ai/tunnel/restart":
                    self.json_response(200, controller.tunnel_restart())
                elif self.path == "/api/ai/pilot/start":
                    self.json_response(200, controller.pilot.start(data.get("goal"), data.get("max_steps", 40), data.get("scan", True), data.get("strategy", "claw")))
                elif self.path == "/api/ai/pilot/stop":
                    controller.pilot.stop("Stopped by user")
                    self.json_response(200, controller.pilot.status())
                elif self.path == "/api/live":
                    q = data.get("query")
                    if data.get("clear"):
                        controller.live.clear()
                    self.json_response(200, controller.live.set_auto(q) if q is not None else controller.live.state())
                elif self.path == "/api/ai/ask":
                    self.json_response(200, controller.ai_ask(data))
                elif self.path == "/api/ai/locate":
                    self.json_response(200, controller.ai_locate(data))
                elif self.path == "/api/ai/plan":
                    self.json_response(200, controller.ai_plan(data))
                elif self.path == "/api/task/cancel":
                    controller.tasks.cancel("Cancelled by " + str(data.get("source", "human")))
                    self.json_response(200, controller.tasks.status())
                elif self.path == "/api/calibration/point":
                    self.json_response(200, controller.calibration_add(data))
                elif self.path == "/api/calibration/touch":
                    self.json_response(200, controller.calibration_touch(data))
                elif self.path == "/api/calibration/look":
                    self.json_response(200, controller.set_look_pose(data))
                elif self.path == "/api/scene/cal/point":
                    self.json_response(200, controller.scene_cal_click(data))
                elif self.path == "/api/scene/cal/auto":
                    self.json_response(200, controller.scene_autocal(data))
                elif self.path == "/api/scene/cal/undo":
                    self.json_response(200, controller.scene_cal.undo())
                elif self.path == "/api/scene/cal/clear":
                    self.json_response(200, controller.scene_cal.clear())
                elif self.path == "/api/calibration/undo":
                    self.json_response(200, controller.calibration_undo())
                elif self.path == "/api/calibration/clear":
                    self.json_response(200, controller.calibration_clear())
                elif self.path == "/api/arm/jog":
                    self.json_response(200, controller.arm_jog(data))
                elif self.path == "/api/stop":
                    self.json_response(200, controller.emergency_stop(
                        str(data.get("reason", "Dashboard stop")), "human"))
                elif self.path == "/api/pilot/container":
                    self.json_response(200, controller.set_container_pose(data))
                elif self.path == "/api/pilot/drop_spot":
                    self.json_response(200, controller.set_drop_spot(data))
                elif self.path == "/api/pilot/claw_point":
                    self.json_response(200, controller.set_claw_point(data))
                elif self.path == "/api/arm/wake":
                    self.json_response(200, controller.arm_wake())
                elif self.path == "/api/enable":
                    self.json_response(200, controller.enable_human())
                else:
                    self.json_response(404, {"error": "Not found"})
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                msg = f"Missing field {exc}" if isinstance(exc, KeyError) else str(exc)
                self.json_response(400, {"error": msg})
            except OSError as exc:            # e.g. model server unreachable
                self.json_response(502, {"error": f"Model server not reachable: {exc}"})
            except Exception as exc:
                controller.emergency_stop("API fault", "system")
                self.json_response(500, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local rover, arm and OAK-D control hub")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--no-oak", action="store_true", help="start without the OAK-D scene camera")
    parser.add_argument("--lan", action="store_true",
                        help="also serve the dashboard to other devices on this network (phone/laptop)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.no_oak:
        config["scene_camera_source"] = "off"
    if args.lan:
        config["server_host"] = "0.0.0.0"
    controller = RobotController(config)
    try:
        server = RobotHTTPServer((config["server_host"], int(config["server_port"])),
                                 make_handler(controller))
    except OSError:
        controller.close()
        raise SystemExit(
            f'Port {config["server_port"]} is already in use - another Robot Hub is still running '
            "(maybe hidden).\nClose it with:  Get-Process python | Stop-Process -Force   then start again.")
    if config["server_host"] == "0.0.0.0":
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                lan_ip = s.getsockname()[0]
        except OSError:
            lan_ip = "<this-laptop-ip>"
        print(f'Robot dashboard: http://127.0.0.1:{config["server_port"]}/   '
              f'(other devices on this Wi-Fi: http://{lan_ip}:{config["server_port"]}/)')
    else:
        print(f'Robot dashboard: http://{config["server_host"]}:{config["server_port"]}/')
    print("Service starts with STOP latched. Use the local dashboard to enable controls.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        controller.close()


if __name__ == "__main__":
    main()
