import os
import threading
import time

# Load the heavy libraries once, here in the main thread. With two cameras
# starting at the same time, first-time imports from two threads can deadlock
# ("_DeadlockError ... numpy._core._multiarray_umath").
try:
    import numpy  # noqa: F401
    import cv2  # noqa: F401
except ImportError:
    pass
try:
    import depthai  # noqa: F401
except ImportError:
    pass



# OAK-D load levels: if the device keeps dropping out (usually USB power),
# step down instead of failing.  0 = full depth, 1 = light depth, 2 = colour only.
OAK_LEVELS = ("full depth", "light depth", "colour only")

class OakCamera:
    """OAK-D RGB or optional spatial-YOLO stream with cached JPEG frames."""

    def __init__(self, mode="rgb", width=640, height=400, fps=15, usb2=True, level=0):
        self.mode = mode
        self.usb2 = bool(usb2)       # USB 2 is far more reliable on long/loose rover cables
        self.level = max(0, min(2, int(level)))
        self._fails = 0
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._jpeg = None
        self._timestamp = 0.0
        self._detections = []
        self._error = "Starting"
        self._depth = None           # uint16 mm, aligned to the colour image
        self._depth_time = 0.0
        self._K = None               # colour camera intrinsics at (width, height)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ depth
    def has_depth(self):
        return self.mode == "depth"

    def depth_frame(self):
        with self._lock:
            return self._depth, self._K

    def point_3d(self, u, v, box=None, depth=None, K=None):
        """3D point (mm, camera frame: x right, y down, z forward) at pixel (u, v).
        With box=[x1,y1,x2,y2] the depth is taken from the middle of the box,
        using the NEAR part (the object, not the table behind it)."""
        import numpy as np
        if depth is None:
            depth, K = self.depth_frame()
        if depth is None or K is None:
            return None
        h, w = depth.shape[:2]
        if box is not None:
            x1, y1, x2, y2 = box
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
            x1, x2 = cx - bw * 0.3, cx + bw * 0.3
            y1, y2 = cy - bh * 0.3, cy + bh * 0.3
        else:
            x1, x2, y1, y2 = u - 5, u + 5, v - 5, v + 5
        x1, x2 = int(max(0, min(w - 1, x1))), int(max(1, min(w, x2 + 1)))
        y1, y2 = int(max(0, min(h - 1, y1))), int(max(1, min(h, y2 + 1)))
        patch = depth[y1:y2, x1:x2].astype("float32")
        valid = patch[(patch > 80) & (patch < 6000)]
        if valid.size < 4:
            return None
        z = float(np.percentile(valid, 25))       # near part of the patch
        fx, fy, cx0, cy0 = K[0][0], K[1][1], K[0][2], K[1][2]
        return [(u - cx0) * z / fx, (v - cy0) * z / fy, z]

    def _run_depth(self, dai):
        """Colour + stereo depth aligned to the colour camera."""
        import cv2
        device = dai.Device(dai.UsbSpeed.HIGH) if self.usb2 else None
        try:
            self._run_depth_on(dai, cv2, device)
        finally:
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass

    def _run_depth_on(self, dai, cv2, device):
        with (dai.Pipeline(device) if device is not None else dai.Pipeline()) as pipeline:
            light = self.level >= 1
            fps = min(self.fps, 8) if light else self.fps
            camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A,
                                                            sensorFps=fps)
            rgb_q = camera.requestOutput((self.width, self.height),
                                         dai.ImgFrame.Type.NV12 if light else dai.ImgFrame.Type.BGR888i,
                                         fps=fps).createOutputQueue(maxSize=2, blocking=False)
            stereo = pipeline.create(dai.node.StereoDepth).build(
                True, dai.node.StereoDepth.PresetMode.DEFAULT, (640, 400), fps)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setLeftRightCheck(not light)
            stereo.setSubpixel(not light)
            depth_q = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
            pipeline.start()
            try:
                calib = pipeline.getDefaultDevice().readCalibration()
                K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, self.width, self.height)
            except Exception as exc:
                K = None
                with self._lock:
                    self._error = f"No OAK intrinsics: {exc}"
            with self._lock:
                self._K = K
            while pipeline.isRunning() and not self._done.is_set():
                got = False
                d = depth_q.tryGet()
                if d is not None:
                    arr = d.getFrame()
                    if arr.shape[1] != self.width or arr.shape[0] != self.height:
                        arr = cv2.resize(arr, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    with self._lock:
                        self._depth, self._depth_time = arr, time.time()
                    got = True
                m = rgb_q.tryGet()
                if m is not None:
                    self._save(m.getCvFrame(), [])
                    got = True
                if not got:
                    self._done.wait(.01)

    def _save(self, frame, detections):
        import cv2
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._lock:
            self._jpeg = encoded.tobytes()
            self._timestamp = time.time()
            self._detections = detections
            self._error = ""

    def _run_rgb(self, dai):
        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build(sensorFps=self.fps)
            queue = camera.requestOutput((self.width, self.height)).createOutputQueue()
            pipeline.start()
            while pipeline.isRunning() and not self._done.is_set():
                message = queue.tryGet()
                if message is None:
                    self._done.wait(.01)
                    continue
                self._save(message.getCvFrame(), [])

    def _run_spatial(self, dai):
        import cv2
        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build(sensorFps=self.fps)
            depth = pipeline.create(dai.node.Depth).build(
                dai.node.Depth.Algorithm.AUTO, self.fps, (self.width, self.height))
            network = pipeline.create(dai.node.SpatialDetectionNetwork).build(
                camera, depth, dai.NNModelDescription("yolov6-nano"))
            network.setDepthLowerThreshold(100)
            network.setDepthUpperThreshold(5000)
            rgb_queue = network.passthrough.createOutputQueue()
            detection_queue = network.out.createOutputQueue()
            labels = network.getClasses()
            pipeline.start()
            last_detections = []
            while pipeline.isRunning() and not self._done.is_set():
                detection_message = detection_queue.tryGet()
                if detection_message is not None:
                    last_detections = []
                    for item in detection_message.detections:
                        label = getattr(item, "labelName", None)
                        if not label:
                            label = labels[item.label] if item.label < len(labels) else str(item.label)
                        last_detections.append({
                            "label": label,
                            "confidence": round(float(item.confidence), 3),
                            "bbox": [item.xmin, item.ymin, item.xmax, item.ymax],
                            "xyz_mm": [int(item.spatialCoordinates.x),
                                       int(item.spatialCoordinates.y),
                                       int(item.spatialCoordinates.z)],
                        })
                rgb = rgb_queue.tryGet()
                if rgb is None:
                    self._done.wait(.01)
                    continue
                frame = rgb.getCvFrame()
                height, width = frame.shape[:2]
                for item in last_detections:
                    x1, y1, x2, y2 = item["bbox"]
                    p1, p2 = (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))
                    cv2.rectangle(frame, p1, p2, (0, 220, 255), 2)
                    text = f'{item["label"]} {item["xyz_mm"][2]}mm'
                    cv2.putText(frame, text, (p1[0], max(18, p1[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 220, 255), 1)
                self._save(frame, last_detections)

    def _run(self):
        try:
            import depthai as dai
        except ImportError:
            with self._lock:
                self._error = "depthai is not installed (pip install -r requirements.txt)"
            return
        while not self._done.is_set():
            started = time.time()
            try:
                with self._lock:
                    self._error = "Connecting"
                if not dai.Device.getAllAvailableDevices():
                    raise RuntimeError("No OAK-D found on USB")
                started = time.time()
                if self.mode == "spatial":
                    self._run_spatial(dai)
                elif self.mode == "depth" and self.level < 2:
                    self._run_depth(dai)
                else:
                    self._run_rgb(dai)
                if self._done.is_set():
                    break
                raise RuntimeError("OAK-D stream stopped")
            except Exception as exc:
                msg = str(exc)
                if "X_LINK" in msg or "Communication exception" in msg or "stream stopped" in msg:
                    msg = "OAK-D USB connection dropped - check the cable/port/power (reconnecting…)"
                dropped = "USB connection dropped" in msg
                if self.mode == "depth" and dropped and time.time() - started < 60:
                    self._fails += 1
                    if self._fails >= 2 and self.level < 2:
                        self.level += 1
                        self._fails = 0
                        msg += f" - switching to {OAK_LEVELS[self.level]} mode"
                with self._lock:
                    self._error = msg
            self._done.wait(2)

    def frame(self):
        with self._lock:
            return self._jpeg

    def size(self):
        return (self.width, self.height)

    def state(self):
        with self._lock:
            age = None if not self._timestamp else round(time.time() - self._timestamp, 2)
            dage = None if not self._depth_time else round(time.time() - self._depth_time, 2)
            return {
                "connected": self._jpeg is not None and age is not None and age < 2,
                "error": self._error,
                "mode": "OAK-D " + (OAK_LEVELS[self.level] if self.mode == "depth" else self.mode),
                "depth": self.mode == "depth" and self.level < 2 and dage is not None and dage < 2,
                "frame_age_seconds": age,
                "detections": list(self._detections),
            }

    def close(self):
        self._done.set()
        self._thread.join(timeout=2)


class WebcamCamera:
    """Any USB/UVC camera through OpenCV (index 0, 1, ...) or a network stream URL
    (e.g. a phone "IP Webcam" app: http://PHONE_IP:8080/video)."""

    def __init__(self, index=0, width=640, height=480, fps=15):
        self.index = index
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.mode = f"webcam {index}" if isinstance(index, int) else "network camera"
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._jpeg = None
        self._timestamp = 0.0
        self._size = (self.width, self.height)
        self._error = "Starting"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            import cv2
        except ImportError:
            with self._lock:
                self._error = "opencv-python is not installed"
            return
        while not self._done.is_set():
            cap = open_capture(cv2, self.index)
            if isinstance(self.index, int):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # USB cams start faster
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                              # always the newest frame
            last_enc = 0.0
            if not cap.isOpened():
                with self._lock:
                    self._error = (f"Cannot open webcam {self.index} (in use by another app, or not plugged in)"
                                   if isinstance(self.index, int) else f"Cannot open stream {self.index}")
                cap.release()
                self._done.wait(2)
                continue
            while not self._done.is_set():
                # read EVERY frame the camera makes (so no old frames pile up in the driver = no lag),
                # but only encode at the configured rate
                if not cap.grab():
                    with self._lock:
                        self._error = "Camera stopped sending frames"
                    break
                if isinstance(self.index, int) and time.time() - last_enc < 1.0 / max(1, self.fps) - 0.005:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                last_enc = time.time()
                ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = enc.tobytes()
                        self._timestamp = time.time()
                        self._size = (frame.shape[1], frame.shape[0])
                        self._error = ""
                if not isinstance(self.index, int):
                    self._done.wait(0.005)
            cap.release()
            self._done.wait(1)

    def frame(self):
        with self._lock:
            return self._jpeg

    def size(self):
        with self._lock:
            return self._size

    def state(self):
        with self._lock:
            age = None if not self._timestamp else round(time.time() - self._timestamp, 2)
            return {"connected": self._jpeg is not None and age is not None and age < 2,
                    "error": self._error, "mode": self.mode,
                    "frame_age_seconds": age, "detections": []}

    def close(self):
        self._done.set()
        self._thread.join(timeout=2)


def open_capture(cv2, index):
    """Windows: DirectShow opens USB cameras faster and more reliably than MSMF."""
    if isinstance(index, int) and os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def oak_devices():
    """Names of OAK devices on USB/network ([] if depthai is missing)."""
    try:
        import depthai as dai
        found = dai.Device.getAllAvailableDevices()
    except Exception:
        return []
    out = []
    for info in found:
        name = getattr(info, "name", "") or ""
        dev_id = getattr(info, "deviceId", "") or ""
        if callable(dev_id):
            dev_id = dev_id()
        out.append(str(dev_id or name or "OAK"))
    return out


def webcam_indexes(max_index=4, skip=()):
    """Indexes 0..max_index-1 that open and deliver a frame."""
    try:
        import cv2
    except ImportError:
        return []
    found = []
    for index in range(max_index):
        if index in skip:
            continue
        cap = open_capture(cv2, index)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(index)
        finally:
            cap.release()
    return found


def parse_source(source):
    """"oak" | "auto" | "webcam:N" | "url:http://..." -> (kind, arg)."""
    source = str(source or "auto").strip()
    if source.startswith("url:"):
        return "url", source[4:].strip()
    if source.startswith(("http://", "https://", "rtsp://")):
        return "url", source
    if source.startswith("webcam"):
        return "webcam", int(source.split(":", 1)[1]) if ":" in source else 0
    if source == "oak":
        return "oak", None
    return "auto", None


def list_sources(current=None):
    """Cameras the hub can use right now, for the dashboard's camera picker.
    The camera that is currently open cannot be probed again, so it is added as-is."""
    kind, arg = parse_source(current)
    sources = []
    oaks = oak_devices()
    if oaks or kind == "oak":
        sources.append({"source": "oak", "label": "OAK-D" + (f" ({oaks[0]})" if oaks else " (in use by hub)" if kind == "oak" else "")})
    skip = {arg} if kind == "webcam" else set()
    indexes = sorted(set(webcam_indexes(skip=skip)) | skip)
    for index in indexes:
        label = f"Webcam {index}" + (" (laptop camera?)" if index == 0 else "")
        sources.append({"source": f"webcam:{index}", "label": label})
    if kind == "url":
        sources.append({"source": "url:" + arg, "label": "Network camera"})
    return sources


class OakProcessCamera:
    """OakCamera running in a child process (see oak_worker.py). Same interface."""

    def __init__(self, mode="rgb", width=640, height=400, fps=15, usb2=True):
        self.mode, self.width, self.height, self.fps, self.usb2 = mode, int(width), int(height), int(fps), usb2
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._jpeg = None
        self._timestamp = 0.0
        self._depth = None
        self._depth_time = 0.0
        self._K = None
        self._state = {"connected": False, "error": "Starting OAK-D", "mode": "OAK-D " + mode,
                       "depth": False, "frame_age_seconds": None, "detections": []}
        self._proc = None
        self._conn = None
        self.crashes = 0
        self._thread = threading.Thread(target=self._supervise, daemon=True)
        self._thread.start()

    def _supervise(self):
        import multiprocessing as mp
        from . import oak_worker
        ctx = mp.get_context("spawn")
        while not self._done.is_set():
            parent, child = ctx.Pipe()
            level = min(2, self.crashes // 2)
            proc = ctx.Process(target=oak_worker.run, daemon=True,
                               args=(child, self.mode, self.width, self.height, self.fps, self.usb2, level))
            proc.start()
            self._proc, self._conn = proc, parent
            try:
                while not self._done.is_set():
                    if parent.poll(1.0):
                        self._take(parent.recv())
                    elif not proc.is_alive():
                        break
            except (EOFError, OSError):
                pass
            if self._done.is_set():
                break
            proc.join(timeout=1)
            self.crashes += 1
            with self._lock:
                self._state = dict(self._state, connected=False, depth=False,
                                   error=f"OAK-D crashed (x{self.crashes}) - restarting it; "
                                         "the rest of the hub keeps running. Check its USB cable/power.")
            self._done.wait(min(20, 3 * self.crashes))

    def _take(self, msg):
        import numpy as np
        import zlib
        with self._lock:
            if "jpeg" in msg:
                self._jpeg, self._timestamp = msg["jpeg"], time.time()
            if "depth" in msg:
                d = np.frombuffer(zlib.decompress(msg["depth"]), np.uint16).reshape(msg["depth_shape"])
                self._depth, self._depth_time, self._K = d, time.time(), msg.get("K")
            if "state" in msg:
                self._state = msg["state"]

    def frame(self):
        with self._lock:
            return self._jpeg

    def size(self):
        return (self.width, self.height)

    def has_depth(self):
        return self.mode == "depth"

    def depth_frame(self):
        with self._lock:
            return self._depth, self._K

    def point_3d(self, u, v, box=None, depth=None, K=None):
        return OakCamera.point_3d(self, u, v, box, depth, K)

    def state(self):
        with self._lock:
            st = dict(self._state)
            age = None if not self._timestamp else round(time.time() - self._timestamp, 2)
            st["frame_age_seconds"] = age
            st["connected"] = bool(st.get("connected")) and age is not None and age < 2
            st["crashes"] = self.crashes
            return st

    def close(self):
        self._done.set()
        try:
            if self._conn is not None:
                self._conn.send("stop")
        except Exception:
            pass
        if self._proc is not None:
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.terminate()
        self._thread.join(timeout=2)


def make_camera(config, factory=None):
    """camera_source: "auto" (OAK-D, else first webcam), "oak", "webcam:N",
    or "url:http://..." for a network MJPEG/RTSP stream."""
    width = config.get("camera_width", 640)
    height = config.get("camera_height", 400)
    fps = config.get("camera_fps", 15)
    if factory is not None:
        return factory(config.get("camera_mode", "rgb"), width, height, fps)
    kind, arg = parse_source(config.get("camera_source", "auto"))
    if kind == "auto":
        if oak_devices():
            kind = "oak"
        else:
            cams = webcam_indexes()
            kind, arg = ("webcam", cams[0]) if cams else ("oak", None)
    if kind == "webcam":
        return WebcamCamera(arg, width, height, fps)
    if kind == "url":
        return WebcamCamera(arg, width, height, fps)
    cls = OakCamera if config.get("oak_in_process") else OakProcessCamera
    return cls(config.get("camera_mode", "rgb"), width, height, fps,
               usb2=config.get("oak_usb2", True))
