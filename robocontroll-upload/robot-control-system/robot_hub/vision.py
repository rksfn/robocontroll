"""Overhead camera <-> arm calibration.

The camera looks down at the table.  Points on the table plane are related
to arm X/Y by a homography, fitted from >= 4 matched points:

    1. Jog the arm so the gripper tip touches (or just clears) a spot on
       the table.  2. Click that exact spot in the camera image.

The average Z of the recorded points is the table height, which then
becomes the arm's safe floor.  Objects are assumed to sit on the table;
tall objects appear slightly shifted in an overhead view (parallax), so
grasp heights are given relative to the table.
"""

import json
import math
import threading
from pathlib import Path


def _solve(a, b):
    """Solve the linear system a x = b (Gaussian elimination, no numpy)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("Calibration points are degenerate (all in a line?)")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _norm_matrix(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    d = sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / len(pts) or 1.0
    s = math.sqrt(2) / d
    return [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def fit_homography(src, dst):
    """Least-squares homography mapping src (u,v) -> dst (x,y) (normalized DLT)."""
    if len(src) < 4:
        raise ValueError("Need at least 4 calibration points")
    ts, td = _norm_matrix(src), _norm_matrix(dst)
    ns = [(ts[0][0] * u + ts[0][2], ts[1][1] * v + ts[1][2]) for u, v in src]
    nd = [(td[0][0] * x + td[0][2], td[1][1] * y + td[1][2]) for x, y in dst]
    rows, rhs = [], []
    for (u, v), (x, y) in zip(ns, nd):
        rows.append([u, v, 1, 0, 0, 0, -u * x, -v * x]); rhs.append(x)
        rows.append([0, 0, 0, u, v, 1, -u * y, -v * y]); rhs.append(y)
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(8)] for i in range(8)]
    atb = [sum(r[i] * b for r, b in zip(rows, rhs)) for i in range(8)]
    h = _solve(ata, atb)
    hn = [h[0:3], h[3:6], h[6:8] + [1.0]]
    full = _mul(invert(td), _mul(hn, ts))
    k = full[2][2]
    return [[v / k for v in row] for row in full]


def apply(h, u, v):
    w = h[2][0] * u + h[2][1] * v + h[2][2]
    if abs(w) < 1e-12:
        raise ValueError("Point is at the camera horizon")
    return ((h[0][0] * u + h[0][1] * v + h[0][2]) / w,
            (h[1][0] * u + h[1][1] * v + h[1][2]) / w)


def invert(h):
    a, b, c = h[0]
    d, e, f = h[1]
    g, i_, k = h[2]
    det = a * (e * k - f * i_) - b * (d * k - f * g) + c * (d * i_ - e * g)
    if abs(det) < 1e-12:
        raise ValueError("Calibration is singular")
    inv = [[(e * k - f * i_), -(b * k - c * i_), (b * f - c * e)],
           [-(d * k - f * g), (a * k - c * g), -(a * f - c * d)],
           [(d * i_ - e * g), -(a * i_ - b * g), (a * e - b * d)]]
    return [[v / det for v in row] for row in inv]


class Calibration:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.points = []            # {"px": [u, v], "arm": [x, y, z]}
        self.h = None               # pixel -> arm
        self.h_inv = None
        self.table_z = None
        self.image_size = None
        self.error_mm = None
        self.load()

    # -------------------------------------------------------------- storage
    def load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.points = data.get("points", [])
            self.image_size = data.get("image_size")
            if len(self.points) >= 4:
                self._fit()
        except Exception:
            self.points, self.h = [], None

    def save(self):
        self.path.write_text(json.dumps({
            "points": self.points, "image_size": self.image_size,
            "table_z": self.table_z, "error_mm": self.error_mm,
            "note": "pixel (u,v) in the camera image <-> arm (x,y,z) mm; written by the hub",
        }, indent=2), encoding="utf-8")

    # -------------------------------------------------------------- editing
    def add_point(self, px, arm_pose, image_size):
        u, v = (float(px[0]), float(px[1]))
        x, y, z = (float(a) for a in arm_pose[:3])
        if not all(math.isfinite(t) for t in (u, v, x, y, z)):
            raise ValueError("Calibration point must be finite numbers")
        with self._lock:
            if self.image_size and list(image_size) != list(self.image_size):
                raise ValueError("Camera resolution changed - clear calibration first")
            self.image_size = list(image_size)
            self.points.append({"px": [round(u, 1), round(v, 1)],
                                "arm": [round(x, 1), round(y, 1), round(z, 1)]})
            if len(self.points) >= 4:
                self._fit()
            self.save()
            return self.status()

    def remove_last(self):
        with self._lock:
            if self.points:
                self.points.pop()
            self.h = self.h_inv = None
            self.table_z = self.error_mm = None
            if len(self.points) >= 4:
                self._fit()
            self.save()
            return self.status()

    def clear(self):
        with self._lock:
            self.points, self.h, self.h_inv = [], None, None
            self.table_z = self.error_mm = self.image_size = None
            self.save()
            return self.status()

    def _fit(self):
        src = [p["px"] for p in self.points]
        dst = [p["arm"][:2] for p in self.points]
        h = fit_homography(src, dst)
        errs = []
        for (u, v), (x, y) in zip(src, dst):
            px, py = apply(h, u, v)
            errs.append(math.hypot(px - x, py - y))
        self.h, self.h_inv = h, invert(h)
        self.table_z = sum(p["arm"][2] for p in self.points) / len(self.points)
        self.error_mm = round(max(errs), 1)

    # -------------------------------------------------------------- use
    @property
    def ready(self):
        return self.h is not None

    def pixel_to_arm(self, u, v, image_size=None):
        if not self.ready:
            raise RuntimeError("Camera is not calibrated yet (need 4+ points)")
        u, v = self._scale_in(u, v, image_size)
        return apply(self.h, u, v)

    def arm_to_pixel(self, x, y, image_size=None):
        if not self.ready:
            return None
        u, v = apply(self.h_inv, x, y)
        return self._scale_out(u, v, image_size)

    def _scale_in(self, u, v, image_size):
        if image_size and self.image_size and list(image_size) != list(self.image_size):
            u = u * self.image_size[0] / image_size[0]
            v = v * self.image_size[1] / image_size[1]
        return float(u), float(v)

    def _scale_out(self, u, v, image_size):
        if image_size and self.image_size and list(image_size) != list(self.image_size):
            u = u * image_size[0] / self.image_size[0]
            v = v * image_size[1] / self.image_size[1]
        return u, v

    def status(self):
        return {"ready": self.ready, "points": list(self.points),
                "table_z": None if self.table_z is None else round(self.table_z, 1),
                "error_mm": self.error_mm, "image_size": self.image_size,
                "h_inv": self.h_inv}
