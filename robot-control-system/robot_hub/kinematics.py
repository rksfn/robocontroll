"""RoArm-M3 kinematics, ported from Waveshare's firmware (RoArm-M3_module.h).

Why this exists: the firmware's direct Cartesian command (T:1041) runs IK
and writes the result to the servos *without checking that IK succeeded*.
An unreachable target produces NaN joint angles, which become garbage servo
positions and the arm "shoots away".  Every pose is therefore checked here
before it is sent.

Frame (same as firmware feedback):
  x forward, y left, z up, millimetres.  z = 0 is the SHOULDER joint height
  (about 126 mm above the bottom of the base), so the table is usually at
  z of roughly -120 to -130.
  pitch t (radians): 0 = gripper pointing straight out, +pi/2 = pointing down.
"""

import math

# Geometry from RoArm-M3_config.h
L2A, L2B = 236.82, 30.00          # shoulder -> elbow
L3A, L3B = 144.49, 0.0            # elbow -> wrist
L4A, L4B = 171.67, 13.69          # wrist -> gripper tip ("edge" gripper)
L2 = math.hypot(L2A, L2B)
T2 = math.atan2(L2B, L2A)
L3 = math.hypot(L3A, L3B)
T3 = math.atan2(L3B, L3A)
LE = math.hypot(L4A, L4B)
TE = math.atan2(L4B, L4A)
SHOULDER_HEIGHT = 126.06          # base bottom -> shoulder joint

# Joint ranges the firmware can actually command (it silently clamps
# outside these, which also makes the arm jump).
SHOULDER_RANGE = (-math.pi / 2, math.pi / 2)
# Upper end: the real elbow folds to ~178 deg (measured on this arm), more than the
# 2960-count figure (170 deg) suggested - using 170 blocked poses the arm actually reaches.
ELBOW_RANGE = ((512 - 1024) * 2 * math.pi / 4096, math.radians(180))
WRIST_RANGE = (-math.pi / 2, math.pi / 2)
# The base servo cannot wrap: +180 and -180 deg are opposite ends of its
# travel.  Keep a dead zone directly behind the arm so no move crosses it.
BASE_LIMIT = math.radians(178)
MARGIN = math.radians(1)          # stay a little inside every limit
REACH_MARGIN_MM = 2.0             # avoid the fully-stretched / fully-folded singularity


def _linkage(a_in, b_in):
    """simpleLinkageIkRad(l2, l3, a, b) -> (shoulder, elbow, delta)."""
    l2c = a_in * a_in + b_in * b_in
    lc = math.sqrt(l2c)
    if lc < 1e-6:
        raise ValueError("target at shoulder")
    if lc > L2 + L3 - REACH_MARGIN_MM or lc < abs(L2 - L3) + REACH_MARGIN_MM:
        raise ValueError("out of reach")
    lam = math.atan2(b_in, a_in)
    psi = math.acos((L2 * L2 + l2c - L3 * L3) / (2 * L2 * lc)) + T2
    alpha = math.pi / 2 - lam - psi
    omega = math.acos((L3 * L3 + l2c - L2 * L2) / (2 * lc * L3))
    beta = psi + omega - T3
    delta = math.pi / 2 - alpha - beta
    return alpha, beta, delta


def ik(x, y, z, pitch_rad):
    """Joint angles (base, shoulder, elbow, wrist) or raise ValueError."""
    rot = TE + (pitch_rad - math.pi)
    dx, dy = -LE * math.cos(rot), -LE * math.sin(rot)
    dist = math.hypot(x, y)
    if dist - dx <= 1e-6:
        raise ValueError("wrist would pass through the base axis")
    ratio = (dist - dx) / dist
    bx, by = x * ratio, y * ratio
    base = math.atan2(by, bx)
    shoulder, elbow, delta = _linkage(math.hypot(bx, by), z + dy)
    wrist = delta + pitch_rad
    return base, shoulder, elbow, wrist


def fk(base, shoulder, elbow, wrist):
    """Firmware RoArmM3_computePosbyJointRad -> (x, y, z, pitch_rad)."""
    r = (L2 * math.cos(math.pi / 2 - (shoulder + T2))
         + L3 * math.cos(math.pi / 2 - (elbow + shoulder + T3))
         + LE * math.cos(math.pi / 2 - (elbow + shoulder + wrist + TE)))
    z = (L2 * math.sin(math.pi / 2 - (shoulder + T2))
         + L3 * math.sin(math.pi / 2 - (elbow + shoulder + T3))
         + LE * math.sin(math.pi / 2 - (elbow + shoulder + wrist + TE)))
    return r * math.cos(base), r * math.sin(base), z, elbow + shoulder + wrist - math.pi / 2


def _outside(x, y, z, pitch_deg):
    """Graded 'how far out of reach' (always < 0) for poses IK cannot solve, so a
    move that brings an out-of-reach arm back towards its workspace is recognised
    as an improvement instead of every nearby pose scoring the same."""
    rot = TE + (math.radians(pitch_deg) - math.pi)
    dx, dy = -LE * math.cos(rot), -LE * math.sin(rot)
    dist = math.hypot(x, y)
    wr = dist - dx
    if wr <= 1e-6:
        return -5.0 + max(-3.0, wr / 1000.0)
    lc = math.hypot(wr, z + dy)
    inner = abs(L2 - L3) + REACH_MARGIN_MM
    outer = L2 + L3 - REACH_MARGIN_MM
    miss = inner - lc if lc < inner else lc - outer if lc > outer else 0.0
    return -1.0 - min(miss, 400.0) / 100.0


def slack(x, y, z, pitch_deg):
    """How far inside the joint limits a pose is (radians); < 0 = not allowed."""
    try:
        b, s, e, w = ik(x, y, z, math.radians(pitch_deg))
    except (ValueError, ZeroDivisionError):
        try:
            return _outside(x, y, z, pitch_deg)
        except (ValueError, ZeroDivisionError):
            return -10.0
    values = [BASE_LIMIT - abs(b),
              s - SHOULDER_RANGE[0], SHOULDER_RANGE[1] - s,
              e - ELBOW_RANGE[0], ELBOW_RANGE[1] - e,
              w - WRIST_RANGE[0], WRIST_RANGE[1] - w]
    if any(math.isnan(v) for v in values):
        return -10.0
    return min(values) - MARGIN


def reachable(x, y, z, pitch_deg):
    return slack(x, y, z, pitch_deg) >= 0


def radial_intervals(z, pitch_deg, r_max=600, step=4):
    """Reachable reach ranges [[r0, r1], ...] at height z for a pitch."""
    out, start = [], None
    r = step
    while r <= r_max:
        ok = reachable(r, 0.0, z, pitch_deg)
        if ok and start is None:
            start = r
        if not ok and start is not None:
            out.append([start, r - step])
            start = None
        r += step
    if start is not None:
        out.append([start, r_max])
    return out


def reach_grid(pitch_deg, r_range=(0, 560), z_range=(-260, 520), cell=10):
    """Side-view reachability mask as rows of '0'/'1' (top row = highest z)."""
    rows = []
    z = z_range[1]
    while z >= z_range[0]:
        rows.append("".join("1" if reachable(r + cell / 2, 0.0, z, pitch_deg) else "0"
                            for r in range(r_range[0], r_range[1], cell)))
        z -= cell
    return {"pitch": pitch_deg, "r0": r_range[0], "z_top": z_range[1],
            "cell": cell, "rows": rows}
