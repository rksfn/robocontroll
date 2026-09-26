import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .controller import RobotController


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "static" / "dashboard.html"


class RobotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def load_config(path):
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

        def log_message(self, fmt, *args):
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
            if length < 1 or length > 8192:
                raise ValueError("JSON request size is invalid")
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            try:
                if self.path == "/":
                    body = DASHBOARD.read_bytes()
                    self._headers(200, "text/html; charset=utf-8", len(body))
                    self.wfile.write(body)
                elif self.path == "/api/state":
                    self.json_response(200, controller.state())
                elif self.path == "/api/frame.jpg":
                    frame = controller.camera.frame()
                    if frame is None:
                        self.json_response(503, {"error": "Camera frame not available"})
                    else:
                        self._headers(200, "image/jpeg", len(frame))
                        self.wfile.write(frame)
                elif self.path == "/camera.mjpg":
                    self._mjpeg()
                else:
                    self.json_response(404, {"error": "Not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.json_response(500, {"error": str(exc)})

        def _mjpeg(self):
            self._headers(200, "multipart/x-mixed-replace; boundary=frame")
            last = None
            while True:
                frame = controller.camera.frame()
                if frame is not None and frame is not last:
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
                elif self.path == "/api/stop":
                    self.json_response(200, controller.emergency_stop(
                        str(data.get("reason", "Dashboard stop")), "human"))
                elif self.path == "/api/enable":
                    self.json_response(200, controller.enable_human())
                else:
                    self.json_response(404, {"error": "Not found"})
            except (ValueError, RuntimeError) as exc:
                self.json_response(400, {"error": str(exc)})
            except Exception as exc:
                controller.emergency_stop("API fault", "system")
                self.json_response(500, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local rover, arm and OAK-D control hub")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    controller = RobotController(config)
    server = RobotHTTPServer((config["server_host"], int(config["server_port"])),
                             make_handler(controller))
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
