import threading
import time


class OakCamera:
    """OAK-D RGB or optional spatial-YOLO stream with cached JPEG frames."""

    def __init__(self, mode="rgb", width=640, height=400, fps=15):
        self.mode = mode
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._jpeg = None
        self._timestamp = 0.0
        self._detections = []
        self._error = "Starting"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

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
        import depthai as dai
        while not self._done.is_set():
            try:
                with self._lock:
                    self._error = "Connecting"
                if not dai.Device.getAllAvailableDevices():
                    raise RuntimeError("No OAK-D found on USB")
                if self.mode == "spatial":
                    self._run_spatial(dai)
                else:
                    self._run_rgb(dai)
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
            self._done.wait(2)

    def frame(self):
        with self._lock:
            return self._jpeg

    def state(self):
        with self._lock:
            age = None if not self._timestamp else round(time.time() - self._timestamp, 2)
            return {
                "connected": self._jpeg is not None and age is not None and age < 2,
                "error": self._error,
                "mode": self.mode,
                "frame_age_seconds": age,
                "detections": list(self._detections),
            }

    def close(self):
        self._done.set()
        self._thread.join(timeout=2)
