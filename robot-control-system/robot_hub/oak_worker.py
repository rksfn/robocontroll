"""Runs the OAK-D in its OWN process.

A USB drop or firmware crash of the OAK-D can crash the depthai library itself
(native code), which would kill the whole hub.  Isolated here, only this
worker dies; the hub keeps running and restarts it.
"""

import os
import time
import zlib


def run(conn, mode, width, height, fps, usb2, level=0):
    os.environ.setdefault("DEPTHAI_LEVEL", "critical")          # USB drop-outs are handled; don't spam the console
    os.environ.setdefault("XLINK_LEVEL", "critical")
    os.environ.setdefault("DEPTHAI_DISABLE_CRASHDUMP_COLLECTION", "1")
    os.environ.setdefault("DEPTHAI_CRASHDUMP", "0")
    if mode == "fake":                                   # for tests without hardware
        cam = _Fake(width, height)
    else:
        from .camera import OakCamera
        cam = OakCamera(mode, width, height, fps, usb2, level)
    last_jpeg, last_depth_t, last_state = None, 0.0, 0.0
    try:
        while True:
            if conn.poll():  # (Ctrl+C in the hub window also reaches this process; exit quietly)
                if conn.recv() == "stop":
                    break
            now = time.time()
            jpeg = cam.frame()
            depth, K = cam.depth_frame()
            dt = getattr(cam, "_depth_time", 0.0)
            msg = {}
            if jpeg is not None and jpeg is not last_jpeg:
                msg["jpeg"], last_jpeg = jpeg, jpeg
            if depth is not None and dt != last_depth_t:
                last_depth_t = dt
                msg["depth"] = zlib.compress(depth.tobytes(), 1)
                msg["depth_shape"] = list(depth.shape)
                msg["K"] = K
            if msg or now - last_state > 0.5:
                last_state = now
                msg["state"] = cam.state()
                conn.send(msg)
            time.sleep(1.0 / 30)
    except (EOFError, BrokenPipeError, OSError, KeyboardInterrupt):
        pass
    finally:
        try:
            cam.close()
        except Exception:
            pass


class _Fake:
    def __init__(self, w, h):
        import numpy as np
        import cv2
        self.w, self.h, self.np, self.cv2 = w, h, np, cv2
        self._depth_time = 0.0

    def frame(self):
        img = self.np.full((self.h, self.w, 3), 90, self.np.uint8)
        self.cv2.putText(img, time.strftime("%H:%M:%S"), (20, 60), 0, 1.5, (0, 255, 255), 2)
        return self.cv2.imencode(".jpg", img)[1].tobytes()

    def depth_frame(self):
        self._depth_time = time.time()
        return self.np.full((self.h, self.w), 800, self.np.uint16), [[400, 0, self.w / 2], [0, 400, self.h / 2], [0, 0, 1]]

    def state(self):
        return {"connected": True, "error": "", "mode": "OAK-D fake", "depth": True,
                "frame_age_seconds": 0.0, "detections": []}

    def close(self):
        pass
