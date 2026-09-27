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


def fit_affine(src, dst):
    """Least-squares affine map src (u,v) -> dst (x,y) as a 3x3 matrix (last row 0 0 1).
    A camera looking straight down at a flat table is almost exactly affine, and unlike a
    homography it cannot "blow up" when the points are few or bunched together."""
    if len(src) < 3:
        raise ValueError("Need at least 3 calibration points")
    rows = [[u, v, 1.0] for u, v in src]
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    ax = _solve(ata, [sum(r[i] * d[0] for r, d in zip(rows, dst)) for i in range(3)])
    ay = _solve(ata, [sum(r[i] * d[1] for r, d in zip(rows, dst)) for i in range(3)])
    return [ax, ay, [0.0, 0.0, 1.0]]


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
        self.inliers = []
        self.point_errors = []
        self.look_pose = None       # camera on the gripper: pose the photos are taken from
        self.load()

    # -------------------------------------------------------------- storage
    def load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.points = data.get("points", [])
            self.image_size = data.get("image_size")
            self.look_pose = data.get("look_pose")
            if len(self.points) >= 3:
                self._fit()
        except Exception:
            self.points, self.h = [], None

    def save(self):
        self.path.write_text(json.dumps({
            "points": self.points, "image_size": self.image_size,
            "table_z": self.table_z, "error_mm": self.error_mm,
            "look_pose": self.look_pose,
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
            if len(self.points) >= 3:
                self._fit()
            self.save()
            return self.status()

    def remove_last(self):
        with self._lock:
            if self.points:
                self.points.pop()
            self.h = self.h_inv = None
            self.table_z = self.error_mm = None
            if len(self.points) >= 3:
                self._fit()
            self.save()
            return self.status()

    def set_look_pose(self, pose):
        """New viewpoint for a gripper-mounted camera: old points no longer apply."""
        with self._lock:
            self.look_pose = [round(float(v), 1) for v in pose[:5]] if pose else None
            self.points, self.h, self.h_inv = [], None, None
            self.table_z = self.error_mm = self.image_size = None
            self.inliers, self.point_errors = [], []
            self.save()
            return self.status()

    def clear(self):
        with self._lock:
            self.points, self.h, self.h_inv = [], None, None
            self.table_z = self.error_mm = self.image_size = None
            self.save()
            return self.status()

    INLIER_MM = 15.0

    def _fit(self):
        """Fit pixel -> arm on the largest set of points that agree with each
        other (within INLIER_MM), so a few badly clicked points are ignored."""
        src = [p["px"] for p in self.points]
        dst = [p["arm"][:2] for p in self.points]
        keep = list(range(len(src)))
        if len(src) >= 6:
            keep = self._consistent_set(src, dst) or keep
        # affine unless there are plenty of good points AND a homography is clearly better
        h = fit_affine([src[i] for i in keep], [dst[i] for i in keep])
        self.model = "affine"
        if len(keep) >= 8:
            try:
                hh = fit_homography([src[i] for i in keep], [dst[i] for i in keep])
                ea = max(math.dist(apply(h, *src[i]), dst[i]) for i in keep)
                eh = max(math.dist(apply(hh, *src[i]), dst[i]) for i in keep)
                if eh < 0.7 * ea and abs(hh[2][0]) * 640 + abs(hh[2][1]) * 480 < 0.5:
                    h, self.model = hh, "homography"
            except (ValueError, ZeroDivisionError):
                pass
        errs = []
        for (u, v), (x, y) in zip(src, dst):
            px, py = apply(h, u, v)
            errs.append(math.hypot(px - x, py - y))
        self.h, self.h_inv = h, invert(h)
        self.inliers = keep
        self.point_errors = [round(e, 1) for e in errs]
        self.table_z = sum(self.points[i]["arm"][2] for i in keep) / len(keep)
        self.error_mm = round(max(errs[i] for i in keep), 1)

    def _consistent_set(self, src, dst):
        import itertools
        import random
        n = len(src)
        combos = itertools.combinations(range(n), 3)
        if n > 20:
            rng = random.Random(0)
            combos = (tuple(rng.sample(range(n), 3)) for _ in range(3000))
        best = []
        for combo in combos:
            try:
                h = fit_affine([src[i] for i in combo], [dst[i] for i in combo])
            except (ValueError, ZeroDivisionError):
                continue
            inl = []
            for i in range(n):
                try:
                    x, y = apply(h, *src[i])
                except ValueError:
                    continue
                if math.hypot(x - dst[i][0], y - dst[i][1]) < self.INLIER_MM:
                    inl.append(i)
            if len(inl) > len(best):
                best = inl
        return best if len(best) >= 3 else None

    # -------------------------------------------------------------- use
    @property
    def ready(self):
        return self.h is not None

    def pixel_to_arm(self, u, v, image_size=None):
        if not self.ready:
            raise RuntimeError("Camera is not calibrated yet (need 3+ points)")
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
                "look_pose": self.look_pose,
                "used": list(self.inliers) if self.ready else [],
                "model": getattr(self, "model", ""),
                "point_errors": list(self.point_errors) if self.ready else [],
                "h_inv": self.h_inv}
