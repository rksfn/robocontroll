"""OAK-D (fixed on the rover) <-> arm coordinates.

The OAK-D measures 3D points in its own frame (x right, y down, z forward, mm).
Because the OAK-D and the arm are both bolted to the rover, one rigid
transform  arm = R * cam + t  links them.  It is fitted from pairs of
(gripper tip seen by the OAK-D in 3D, gripper tip position reported by the arm),
collected automatically (Qwen finds the gripper in each photo) or by clicking.
Badly measured pairs are ignored (RANSAC).
"""

import itertools
import json
import math
import threading
from pathlib import Path

INLIER_MM = 25.0


def kabsch(cam, arm):
    """Least-squares rotation R and translation t with arm ~= R @ cam + t."""
    import numpy as np
    P, Q = np.asarray(cam, float), np.asarray(arm, float)
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    return R, t


class SceneCalibration:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.points = []          # {"cam": [x,y,z], "arm": [x,y,z], "px": [u,v]}
        self.R = self.t = None
        self.used, self.errors, self.error_mm = [], [], None
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.points = json.loads(self.path.read_text(encoding="utf-8")).get("points", [])
                self._fit()
            except Exception:
                self.points = []

    def save(self):
        self.path.write_text(json.dumps({"points": self.points, "error_mm": self.error_mm,
                                         "note": "OAK-D 3D point (mm) <-> arm gripper tip (mm)"},
                                        indent=2), encoding="utf-8")

    @property
    def ready(self):
        return self.R is not None

    def add(self, cam, arm, px=None):
        with self._lock:
            self.points.append({"cam": [round(float(v), 1) for v in cam[:3]],
                                "arm": [round(float(v), 1) for v in arm[:3]],
                                "px": px and [round(float(v), 1) for v in px]})
            self._fit()
            self.save()
            return self.status()

    def undo(self):
        with self._lock:
            if self.points:
                self.points.pop()
            self._fit()
            self.save()
            return self.status()

    def clear(self):
        with self._lock:
            self.points = []
            self._fit()
            self.save()
            return self.status()

    def _fit(self):
        import numpy as np
        self.R = self.t = None
        self.used, self.errors, self.error_mm = [], [], None
        n = len(self.points)
        if n < 4:
            return
        cam = np.array([p["cam"] for p in self.points], float)
        arm = np.array([p["arm"] for p in self.points], float)
        best = list(range(n))
        if n >= 5:
            best = []
            combos = itertools.combinations(range(n), 3)
            for combo in itertools.islice(combos, 2000):
                a = cam[list(combo)]
                if np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0])) < 500:   # nearly collinear
                    continue
                R, t = kabsch(a, arm[list(combo)])
                err = np.linalg.norm(cam @ R.T + t - arm, axis=1)
                inl = [i for i in range(n) if err[i] < INLIER_MM]
                if len(inl) > len(best):
                    best = inl
            if len(best) < 4:
                return
        R, t = kabsch(cam[best], arm[best])
        err = np.linalg.norm(cam @ R.T + t - arm, axis=1)
        self.R, self.t = R, t
        self.used = best
        self.errors = [round(float(e), 1) for e in err]
        self.error_mm = round(float(max(err[i] for i in best)), 1)

    def cam_to_arm(self, p):
        if not self.ready:
            raise RuntimeError("OAK-D is not linked to the arm yet (Auto-calibrate first)")
        import numpy as np
        return [float(v) for v in self.R @ np.asarray(p, float) + self.t]

    def arm_to_cam(self, p):
        if not self.ready:
            return None
        import numpy as np
        return [float(v) for v in self.R.T @ (np.asarray(p, float) - self.t)]

    def status(self):
        return {"ready": self.ready, "points": len(self.points), "used": list(self.used),
                "error_mm": self.error_mm, "errors": list(self.errors),
                "pixels": [p.get("px") for p in self.points]}
