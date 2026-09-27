"""Live vision: fast "eyes" that run between the (slow, 2-10 s) Qwen calls.

Qwen decides WHAT the objects are and roughly WHERE.  This module keeps those
boxes glued to the objects at camera frame rate (template matching, pure
OpenCV, ~1-2 ms per object), finds the yellow drop-off paper by colour on
every frame, and - with the dashboard's "Live" button - asks Qwen again in the
background so the list stays fresh.  That gives YOLO-style live boxes without
installing anything.

The autopilot uses the same tools to check that a pick REALLY worked (the
object is gone from its spot) and to reuse Qwen's list instead of asking
again after every item.
"""

import itertools
import threading
import time

SHRINK = 2            # work on half-size frames (fast; IMREAD_REDUCED_*_2)
MATCH_OK = 0.60       # template score that counts as "found it"
REFIND_OK = 0.72      # stricter when searching the whole frame for a lost object
FLAT_STD = 6.0        # templates with less texture than this can't be tracked (plain paper)


def _cv():
    import cv2
    import numpy as np
    return cv2, np


def decode(jpeg, colour=False):
    """JPEG -> half-size grey (or colour) image, or None."""
    if not jpeg:
        return None
    cv2, np = _cv()
    flag = cv2.IMREAD_REDUCED_COLOR_2 if colour else cv2.IMREAD_REDUCED_GRAYSCALE_2
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), flag)


def _factor(img, size):
    return float(size[0]) / img.shape[1] if size else float(SHRINK)


def make_template(gray, bbox, f):
    """Central part of the box (less background) as a small grey template. None if unusable."""
    x1, y1, x2, y2 = (v / f for v in bbox)
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 + 0.12 * w, x2 - 0.12 * w
    y1, y2 = y1 + 0.12 * h, y2 - 0.12 * h
    H, W = gray.shape[:2]
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(W, int(round(x2))), min(H, int(round(y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    tpl = gray[y1:y2, x1:x2].copy()
    if tpl.shape[0] > 0.8 * H or tpl.shape[1] > 0.8 * W:
        return None                                   # "the whole table": nothing to track
    return tpl


def is_flat(tpl):
    return tpl is None or float(tpl.std()) < FLAT_STD


def match(gray, tpl, near=None, radius=None, scales=(1.0,)):
    """Best place for tpl in gray -> (score, cx, cy, scale) in half-size pixels.
    near/radius limit the search to a window around a point (much faster, fewer mix-ups)."""
    cv2, np = _cv()
    best = (-1.0, 0.0, 0.0, 1.0)
    H, W = gray.shape[:2]
    for s in scales:
        t = tpl if s == 1.0 else cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        th, tw = t.shape[:2]
        if th < 6 or tw < 6 or th >= H or tw >= W:
            continue
        if near is not None:
            r = radius
            x0 = int(max(0, near[0] - tw / 2 - r)); x1 = int(min(W, near[0] + tw / 2 + r))
            y0 = int(max(0, near[1] - th / 2 - r)); y1 = int(min(H, near[1] + th / 2 + r))
            win = gray[y0:y1, x0:x1]
        else:
            x0 = y0 = 0
            win = gray
        if win.shape[0] <= th or win.shape[1] <= tw:
            continue
        res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (float(score), x0 + loc[0] + tw / 2.0, y0 + loc[1] + th / 2.0, s)
    return best


def find_near(jpeg, size, tpl, bbox, radius_boxes=1.5):
    """Where is this object now, near where it was? -> (score, pixel, bbox) in camera px.
    Used to check a pick: a high score at the old spot means the object is STILL THERE."""
    gray = decode(jpeg)
    if gray is None or tpl is None:
        return -1.0, None, None
    f = _factor(gray, size)
    cx, cy = (bbox[0] + bbox[2]) / 2 / f, (bbox[1] + bbox[3]) / 2 / f
    r = radius_boxes * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / f + 10
    score, u, v, s = match(gray, tpl, (cx, cy), r, scales=(0.9, 1.0, 1.1))
    if score < 0:
        return score, None, None
    w, h = (bbox[2] - bbox[0]) * s, (bbox[3] - bbox[1]) * s
    u, v = u * f, v * f
    return score, [round(u, 1), round(v, 1)], [round(u - w / 2, 1), round(v - h / 2, 1),
                                               round(u + w / 2, 1), round(v + h / 2, 1)]


def template_from(jpeg, size, bbox):
    gray = decode(jpeg)
    if gray is None:
        return None
    tpl = make_template(gray, bbox, _factor(gray, size))
    return None if is_flat(tpl) else tpl


def orientation(jpeg, size, bbox):
    """Long axis of the object in the box -> (angle in image degrees, elongation ratio) or None.
    Segments the object from the table (Otsu inside the box), then uses its second moments."""
    cv2, np = _cv()
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE) if jpeg else None
    if img is None:
        return None
    H, W = img.shape[:2]
    fx, fy = W / float(size[0]), H / float(size[1])
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, (x1 - 0.15 * bw) * fx)), int(min(W, (x2 + 0.15 * bw) * fx))
    y1, y2 = int(max(0, (y1 - 0.15 * bh) * fy)), int(min(H, (y2 + 0.15 * bh) * fy))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    roi = cv2.GaussianBlur(img[y1:y2, x1:x2], (5, 5), 0)
    _, m = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    if border.mean() > 127:                       # the table came out white: object is the dark part
        m = 255 - m
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    cy, cx = m.shape[0] / 2.0, m.shape[1] / 2.0
    best = min(range(1, n), key=lambda i: (abs(cents[i][0] - cx) + abs(cents[i][1] - cy)) / (stats[i][4] ** 0.5 + 1))
    if stats[best][4] < 30:
        return None
    mo = cv2.moments((lab == best).astype(np.uint8))
    mu20, mu02, mu11 = mo["mu20"], mo["mu02"], mo["mu11"]
    ang = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    common = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
    ratio = float(np.sqrt(l1 / l2)) if l2 > 1e-6 else 9.9
    # angle measured in the resized image: undo the axis scaling
    ang = float(np.degrees(np.arctan2(np.sin(np.radians(ang)) / fy, np.cos(np.radians(ang)) / fx)))
    return ang, ratio


def yellow_zone(img_bgr, f):
    """Largest yellow area (the drop-off paper) -> polygon in camera px, or None."""
    cv2, np = _cv()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 90, 90), (36, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.002 * w * h or (bw < 0.08 * w and bh < 0.08 * h):
            continue
        if best is None or bw * bh > best[0]:
            best = (bw * bh, [float(x * f), float(y * f), float((x + bw) * f), float((y + bh) * f)])
    if not best:
        return None
    x1, y1, x2, y2 = best[1]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class LiveVision:
    """Background thread: tracks the last Qwen detections on every new camera frame."""
    _ids = itertools.count(1)

    def __init__(self, controller, hz=15.0):
        self.c = controller
        self.period = 1.0 / hz
        self._lock = threading.Lock()
        self._tracks = []
        self._zone = None
        self._fps = 0.0
        self._last_jpeg = None
        self._auto = ""                   # Qwen query for the continuous "Live" mode
        self._auto_err = ""
        self._qwen_s = None
        self._seeded = 0.0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-vision")
        self._thread.start()
        self._auto_thread = None

    # ------------------------------------------------------------ public
    def close(self):
        self._done.set()
        self._auto = ""

    def seed(self, objects, jpeg, size, source="qwen"):
        """New Qwen detections (from the photo `jpeg`) -> tracks. The tracker then finds each
        object again in the NEWEST frame, so the model's delay doesn't leave boxes behind."""
        try:
            gray = decode(jpeg)
        except Exception:
            return
        if gray is None:
            return
        f = _factor(gray, size)
        with self._lock:
            old = list(self._tracks)
        new = []
        for o in objects or []:
            b = o.get("bbox")
            if not b:
                continue
            tpl = make_template(gray, b, f)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            prev = min(old, key=lambda t: abs(t["px"][0] - cx) + abs(t["px"][1] - cy), default=None)
            same = prev is not None and abs(prev["px"][0] - cx) + abs(prev["px"][1] - cy) < \
                0.6 * max(b[2] - b[0], b[3] - b[1])
            new.append({"id": prev["id"] if same else next(self._ids), "label": o.get("label", "?"),
                        "bbox": list(b), "px": [cx, cy], "tpl": None if is_flat(tpl) else tpl,
                        "wh": [b[2] - b[0], b[3] - b[1]], "score": 1.0, "seen": time.time(),
                        "lost": False, "refind_t": 0.0, "source": source})
        with self._lock:
            self._tracks = new
            self._seeded = time.time()

    def clear(self):
        with self._lock:
            self._tracks = []

    def tracks(self):
        with self._lock:
            return [{k: v for k, v in t.items() if k != "tpl"} for t in self._tracks]

    def zone(self):
        with self._lock:
            return self._zone

    def set_auto(self, query):
        """Continuous detection: ask Qwen again and again (as fast as it answers). '' = off."""
        self._auto = str(query or "").strip()[:200]
        self._auto_err = ""
        if self._auto and (self._auto_thread is None or not self._auto_thread.is_alive()):
            self._auto_thread = threading.Thread(target=self._auto_loop, daemon=True, name="live-qwen")
            self._auto_thread.start()
        return self.state()

    def state(self):
        with self._lock:
            return {"fps": round(self._fps, 1), "auto": self._auto, "error": self._auto_err,
                    "qwen_s": self._qwen_s,
                    "age_s": round(time.time() - self._seeded, 1) if self._seeded else None,
                    "zone": self._zone,
                    "tracks": [{"id": t["id"], "label": t["label"], "bbox": [round(v, 1) for v in t["bbox"]],
                                "score": round(t["score"], 2), "lost": t["lost"],
                                "static": t["tpl"] is None} for t in self._tracks]}

    # ------------------------------------------------------------ loops
    def _run(self):
        try:
            cv2, _ = _cv()
        except ImportError:
            return
        n, t_fps = 0, time.time()
        while not self._done.is_set():
            t0 = time.time()
            try:
                self._step(cv2)
                n += 1
            except Exception:
                pass
            if time.time() - t_fps >= 1.0:
                with self._lock:
                    self._fps = n / (time.time() - t_fps)
                n, t_fps = 0, time.time()
            self._done.wait(max(0.005, self.period - (time.time() - t0)))

    def _step(self, cv2):
        cam = self.c.camera
        jpeg = cam.frame()
        if jpeg is None or jpeg is self._last_jpeg:
            return
        self._last_jpeg = jpeg
        size = self.c.image_size()
        img = decode(jpeg, colour=True)
        if img is None:
            return
        f = _factor(img, size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        zone = None
        if "yellow" in str(self.c.config.get("drop_zone", "yellow paper")).lower():
            zone = yellow_zone(img, f)
        with self._lock:
            tracks = list(self._tracks)
        now = time.time()
        for t in tracks:
            if t["tpl"] is None:
                continue                                   # plain object: keep Qwen's box
            th, tw = t["tpl"].shape[:2]
            near = (t["px"][0] / f, t["px"][1] / f)
            score, u, v, s = match(gray, t["tpl"], near, max(tw, th) * 0.8 + 12)
            if score < MATCH_OK:
                score, u, v, s = match(gray, t["tpl"], near, max(tw, th) * 0.8 + 12, (0.85, 1.15))
            if score < MATCH_OK and now - t["refind_t"] > 0.4:     # lost: look everywhere (rate-limited)
                t["refind_t"] = now
                sc2, u2, v2, s2 = match(gray, t["tpl"], scales=(0.8, 1.0, 1.25))
                if sc2 >= REFIND_OK:
                    score, u, v, s = sc2, u2, v2, s2
            if score >= MATCH_OK:
                u, v = u * f, v * f
                w, h = t["wh"][0] * s, t["wh"][1] * s
                t.update(px=[u, v], bbox=[u - w / 2, v - h / 2, u + w / 2, v + h / 2], score=score,
                         seen=now, lost=False, wh=[w, h])
                if s != 1.0 and 10 <= min(th, tw) * s and max(th, tw) * s < 0.7 * min(gray.shape[:2]):
                    # object got bigger/smaller (arm moved up/down): rescale the template
                    t["tpl"] = cv2.resize(t["tpl"], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            else:
                t.update(score=max(0.0, score), lost=now - t["seen"] > 0.3)
        with self._lock:
            self._zone = zone

    def _auto_loop(self):
        while self._auto and not self._done.is_set():
            try:
                if self.c.pilot.running():               # the autopilot asks Qwen itself
                    time.sleep(0.5)
                    continue
                jpeg, size = self.c._frame()
                t0 = time.time()
                found = self.c.ai.locate(jpeg, size, self._auto).get("objects", [])
                if not self._auto:
                    break
                self.seed(found, jpeg, size, "live")
                with self._lock:
                    self._qwen_s = round(time.time() - t0, 1)
                    self._auto_err = ""
            except Exception as exc:
                with self._lock:
                    self._auto_err = f"{type(exc).__name__}: {exc}"[:160]
                time.sleep(2.0)
            time.sleep(0.2)
