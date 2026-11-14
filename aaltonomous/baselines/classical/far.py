"""Far delivery (config grasp.far): boxes whose drop slots are out of the top-down reach.

Why: the published plan puts the boxes' near edge 0.40 m in front of the waist joint; the venue (measured 2026-11-14)
has it at ~0.50 m (+- 3 cm), the organisers' recordings at 0.47-0.62 m. The top-down release (grasp.slots: palm level,
tilt <= 10 deg) gets the grasp point to x ~0.515-0.535 m, so from a near edge of ~0.46 m on a box has few or no slots
and the object would stay on the counter (0 points).

The far release (static): only for a box the box detection found (Box.src 'detected', conf >= far.min_conf, a
certain name) that has NO reachable drop slot for this arm (Planner.slots empty). The hand may be oriented freely
(approach axis up to far.max_tilt_deg from straight down: at the farthest point the palm faces forward-down-inward,
the fingers point forward-outward-slightly-up, the object sits ahead of the wrist, as the teleoperators held it), the
release height is chosen for reach (best near shoulder height, ~0.28 m above the counter) and the joints may use their
IK bounds. Reach envelope (envelope / max_reach): per box-frame lateral position v and height z the farthest grasp
point along the box's depth axis u (SLSQP over the 7 joints: max u . p subject to the lateral position, the height,
the tilt cone and the joint bounds). The release point is far.static_backoff short of it (never deeper than the far
wall allows); the arm settles there (far.settle_s), the hand opens (`release` posture, far.hand_vel). The opening
hand pushes the object: it lands far.release_shift = [least, most] ahead of the point straight below the grasp point
and up to far.release_side sideways (measured in the simulator with the real finger gains: sim/far_lab.py).
Accepted only if
  * the least landing centre >= the near wall's inner face + object half size (far.object_size / 2: venue objects
    ~47 mm) + far.near_clear at the nominal box pose. pos_err is NOT added on this side: an error that makes it
    short leaves the object on the rim or on the counter in front (0 points, retried; the walls are cardboard:
    never count on bouncing off them),
  * the farthest <= the far inner face - half size - land_clear - far_clear - pos_err (venue: the boxes stand at the
    table's far edge: long = off the table = -1),
  * sideways within the inner width - half size - side_clear - release_side - pos_err (neighbouring boxes 1-3 cm
    away = a wrong box): the releases stay near the box's middle line, away from the side walls,
and the wrist / palm stay far.wrist_clear above every rim while over a box (+ far.bulk_margin): the real wrists carry
a camera bracket and cable, the arms a loose T-shirt. Among the accepted (lateral, height) points: sure before
marginal (far.max_short), the farthest from the objects already delivered into that box (planner.drop_used;
far.min_spacing), then landings the head camera can
see (visible: not hidden behind a wall, inside the image; the next look confirms them), then the largest near margin
(up to far.margin_enough), then the lowest height. Otherwise the plan fails as before (the object stays).
Planner.can_deliver / box_reachable (reachable: the envelope check without IK, cached per box pose) let the policy
skip a box no arm can deliver into before any per-object planning.

Release in forward motion was designed, built and dropped (2026-11-14): at the reach limit the arm is nearly
stretched, and with every joint at <= 0.85 rad/s (filter 1.0) the fastest forward grasp-point motion with no sideways
component is 0.04-0.16 m/s 0-5 cm short of the envelope (LP over the joint Jacobian, forward_speed; ~0.3-0.4 m/s only
8-10 cm back). Over a 0.23 s fall that buys 1-4 cm, minus the release-time uncertainty (fingers at kp 1.5: +-50 ms x
the speed) and the 0-5 cm it had to start short: <= ~1 cm net, never enough to be worth the risk.

Everything depends on the configured grasp (grasp.point_in_palm, hand.closed / release): nothing here knows today's
numbers. Envelopes are cached per (box pose, arm), recomputed when the box moved more than grasp.slots.box_tol.

    python -m aaltonomous.baselines.classical.far map [along ...]   # reach map: near edge x box x arm x grasp point
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import numpy as np

from .kinematics import DOWN, ArmKinematics, _skew

G = 9.81
DEFAULTS = {
    "enabled": False,
    "min_conf": 0.5,            # box detection confidence needed
    "max_front": 0.60,          # m: near wall midpoint farther than this along the box's depth axis: not attempted
    "object_size": 0.05,        # m: object extent used for every margin (venue objects ~48 mm, drop 2: 50-55)
    "near_clear": -0.01,        # m: least landing centre >= near inner face + half size + this (nominal pose, no
                                # pos_err): short = on the rim / counter in front = 0 points, retried (sim: every
                                # landing centre >= 22 mm past the inner face was in, 14 / 14)
    "land_clear": 0.012,        # m: ... and this far inside at the far wall (+ pos_err there)
    "side_clear": 0.0,          # m: ... and this at the side walls (+ pos_err + release_side)
    "far_clear": 0.04,          # m: extra at the far wall (venue: boxes at the table's far edge, long = -1)
    "pos_err": 0.008,           # m: box pose (detection) + arm tracking error, at the far wall / the sides
    # where the object lands vs straight below the grasp point at the release: the opening hand pushes it forward
    # and sideways (sim far_lab 2026-11-14, m23 cage grasp, real finger gains, 3 shapes, both arms: near edge 0.50,
    # 14 releases: forward +19 ... +46 mm, sideways up to 50 mm, 14 / 14 in; 0.47, 7: +13 ... +32 mm, 7 / 7 in;
    # released 2-3.5 cm before the inner wall face (0.53): 7 / 16 in, the rest back on the counter, none wrong / off)
    "release_shift": [0.012, 0.06],   # m forward: least / most
    "release_side": 0.05,       # m sideways, either way
    "margin_enough": 0.015,     # m: near margins above this count the same (then the lowest height wins)
    "max_short": 0.035,         # m: a release whose least landing is short of the near rule by up to this is still
                                # attempted (MARGINAL: after the sure ones; a miss lands on the counter in front = 0,
                                # picked again once) as long as the far wall and the sides hold. Sim, near edge 0.53,
                                # short by 22-38 mm: 7 / 16 in (centre box 5 / 8, side boxes 2 / 8), the rest on the
                                # counter in front, none in a wrong box / off the table
    "wrist_clear": 0.05,        # m: wrist / palm above the rims while over a box (+ bulk_margin around it): the real
    "bulk_margin": 0.04,        # wrists carry a camera bracket + cable, the arms a loose T-shirt (not in the URDF)
    "max_tilt_deg": 60.0,       # approach axis vs straight down at the release
    "heights": [0.20, 0.25, 0.29],   # m above the counter: release heights tried (reach is best at shoulder height)
    "laterals": 3,              # release positions across the box (box frame; few: release_side keeps them central)
    "static_backoff": 0.005,    # m: release this far short of the envelope
    "joint_margin": None,       # rad (scalar / {joint key: rad}) for the far poses; null = the IK bounds
    "hand_vel": 2.5,            # rad/s: release ramp (filter 3.0)
    "min_spacing": 0.06,        # m between release points in one box (a stacked object sticks out above the rim)
    "settle_s": 0.3,            # s at the release pose before the hand opens (the arm settles: robot 90 % in 0.2 s)
}


def far_cfg(cfg):
    out = dict(DEFAULTS)
    out.update((cfg.get("grasp") or {}).get("far") or {})
    return out


def far_bounds(kin, margin=None):
    """Joint bounds (lo, hi) (14,): the URDF limits minus `margin` (rad, scalar or {joint key: rad, 'default': rad}
    as ik.joint_margin); None = the kinematics' own IK bounds."""
    if margin is None:
        return kin.lo.copy(), kin.hi.copy()
    sp = kin.spec
    if not isinstance(margin, dict):
        m = np.full(len(sp.arm_names), float(margin))
    else:
        m = np.full(len(sp.arm_names), float(margin.get("default", 0.08)))
        for key, v in margin.items():
            for i, n in enumerate(sp.arm_names):
                if n.endswith(f"_{key}_joint"):
                    m[i] = float(v)
    return sp.arm_lower + m, sp.arm_upper - m


def _pose_terms(kin, side, arm, off):
    """(grasp point (3,), its Jacobian (3, 7), palm rotation (3, 3), angular Jacobian (3, 7)) at arm (14,)."""
    pin, sl, fid = kin.pin, ArmKinematics.sl(side), kin.palm[side]
    q = kin._fk(arm)
    M = kin.data.oMf[fid]
    r = M.rotation @ off
    J6 = pin.computeFrameJacobian(kin.model, kin.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, kin.arm_iv[sl]]
    return M.translation + r, J6[:3] - _skew(r) @ J6[3:], M.rotation.copy(), J6[3:]


def max_reach(kin, side, u, v0, z, max_tilt, seed_q, lo, hi, c0=(0.0, 0.0), seeds=(), offset=None, restarts=0):
    """Farthest grasp point along unit direction u (2,) (pelvis xy) on the line {c0 + s u + v0 n} (n = u turned +90
    deg) at height z, approach axis within max_tilt (rad) of straight down, joints in [lo, hi] (7,). offset: grasp
    point in the palm frame (None: the kinematics' own). Starts: seed_q, seeds, `restarts` random ones.
    Returns (s, q (14,), tilt rad) or None."""
    from scipy.optimize import minimize
    sl = ArmKinematics.sl(side)
    u = np.asarray(u, float) / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    c0 = np.asarray(c0, float)
    off = kin.offset[side] if offset is None else np.asarray(offset, float)
    base = np.asarray(seed_q, float).copy()
    cache = {}

    def ev(x):
        k = x.tobytes()
        if k not in cache:
            arm = base.copy()
            arm[sl] = x
            p, Jp, R, Jw = _pose_terms(kin, side, arm, off)
            a = R @ kin.axis[side]
            cache.clear()
            cache[k] = (p, Jp, a, -_skew(a) @ Jw)
        return cache[k]

    cons = [{"type": "eq", "fun": lambda x: np.r_[(ev(x)[0][:2] - c0) @ n - v0, ev(x)[0][2] - z],
             "jac": lambda x: np.vstack([n @ ev(x)[1][:2], ev(x)[1][2]])},
            {"type": "ineq", "fun": lambda x: np.array([ev(x)[2] @ DOWN - np.cos(max_tilt)]),
             "jac": lambda x: (DOWN @ ev(x)[3])[None]}]
    rng = np.random.default_rng(0)
    starts = [base[sl]] + [np.asarray(s, float)[sl] if len(s) == 14 else np.asarray(s, float) for s in seeds]
    starts += [rng.uniform(lo, hi) for _ in range(int(restarts))]
    best = None
    for x0 in starts:
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        r = minimize(lambda x: -(ev(x)[0][:2] - c0) @ u, x0, jac=lambda x: -(u @ ev(x)[1][:2]), method="SLSQP",
                     bounds=list(zip(lo, hi)), constraints=cons, options={"maxiter": 150, "ftol": 1e-9})
        x = np.clip(r.x, lo, hi)
        p, _, a, _ = ev(x)
        if (abs((p[:2] - c0) @ n - v0) < 1e-3 and abs(p[2] - z) < 1e-3 and a @ DOWN >= np.cos(max_tilt) - 1e-3):
            s = float((p[:2] - c0) @ u)
            if best is None or s > best[0]:
                q = base.copy()
                q[sl] = x
                best = (s, q, float(np.arccos(np.clip(a @ DOWN, -1, 1))))
    return best


def solve_pose(kin, side, p, R, seed_q, lo, hi, w_rot=0.3, pos_tol=0.0015, rot_tol_deg=2.0):
    """IK for a full pose: grasp point at p (3,), palm rotation R (3, 3), joints in [lo, hi] (7,), from seed_q (14,)
    (a small pull towards it keeps neighbouring waypoints on one branch). Returns (q (14,), ok)."""
    from scipy.optimize import least_squares
    sl, pin = ArmKinematics.sl(side), kin.pin
    base = np.asarray(seed_q, float).copy()
    off, p, R = kin.offset[side], np.asarray(p, float), np.asarray(R, float)
    x0 = np.clip(base[sl], lo + 1e-6, hi - 1e-6)
    w_reg = 1e-3

    def fun(x):
        arm = base.copy()
        arm[sl] = x
        q, Jp, Rq, Jw = _pose_terms(kin, side, arm, off)
        e = pin.log3(Rq @ R.T)
        return np.r_[q - p, w_rot * e, w_reg * (x - x0)], np.vstack([Jp, w_rot * Jw, w_reg * np.eye(7)])

    cache = {}

    def f(x):
        k = x.tobytes()
        if k not in cache:
            cache.clear()
            cache[k] = fun(x)
        return cache[k]
    r = least_squares(lambda x: f(x)[0], x0, jac=lambda x: f(x)[1], bounds=(lo, hi), method="trf", max_nfev=60,
                      xtol=1e-10, ftol=1e-12, gtol=1e-12)
    out = base.copy()
    out[sl] = r.x
    res = f(r.x)[0]
    ok = np.linalg.norm(res[:3]) <= pos_tol and np.degrees(np.linalg.norm(res[3:6]) / w_rot) <= rot_tol_deg
    return out, bool(ok)


def flight(p0, v0, z_land):
    """Point-mass ballistics: released at p0 (3,) with velocity v0 (3,), the centre reaches z_land. Returns (landing
    xyz (3,), time s); p0 below z_land: (p0 with z_land, 0)."""
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    dz = p0[2] - z_land
    if dz <= 0:
        return np.r_[p0[:2], z_land], 0.0
    T = (v0[2] + np.sqrt(v0[2] ** 2 + 2 * G * dz)) / G
    return np.r_[p0[:2] + v0[:2] * T, z_land], float(T)


def forward_speed(kin, side, q, u, cap=0.85, up_deg=20.0):
    """Fastest grasp-point speed (m/s) along pelvis-xy direction u at arm pose q (14,) with every joint of `side` at
    <= cap rad/s, no sideways component, upwards at most up_deg (LP over the position Jacobian; the palm may turn).
    The release-in-motion study (module doc)."""
    from scipy.optimize import linprog
    _, Jp, _, _ = _pose_terms(kin, side, q, kin.offset[side])
    u = np.asarray(u, float) / np.linalg.norm(u)
    ju, jn, jz = np.r_[u, 0] @ Jp, np.r_[-u[1], u[0], 0] @ Jp, Jp[2]
    r = linprog(-ju, A_ub=np.vstack([-jz, jz - np.tan(np.radians(up_deg)) * ju]), b_ub=[0.0, 0.0], A_eq=jn[None],
                b_eq=[0.0], bounds=[(-cap, cap)] * 7, method="highs")
    return float(-r.fun) if r.success else 0.0


@dataclass
class Release:
    """One far release: q (14,) = the release pose (the carry ends there); the side's arm ticks (n, 7) and hand
    ticks (n, 7) from there (settle, open, hold open); prediction in `info`."""
    q: np.ndarray
    arm: np.ndarray
    hand: np.ndarray
    land: np.ndarray                     # predicted landing centre (pelvis xyz)
    info: dict = field(default_factory=dict)


class FarDelivery:
    """grasp.far for one Planner (its kinematics, boxes, drop_used)."""

    def __init__(self, planner):
        self.pl = planner
        self.c = far_cfg(planner.cfg)
        self._env = {}           # (bin, side) -> (Box, envelope rows)
        self.camera = None       # (K (3, 3), T_pelvis_optical (4, 4), width, height) of the head camera (the policy sets
                                 # it): landings it can see are preferred (the look after the delivery confirms them)
        self.last = None         # info of the last release() call (choice or why not) for logs and labs

    # ------------------------------------------------------------------ when
    def on(self):
        return bool(self.c.get("enabled"))

    def applies(self, name, side):
        """(True, '') if this box / arm gets a far release instead of the stock one, else (False, why)."""
        pl = self.pl
        if not self.on() or name not in pl.boxes:
            return False, "off"
        bx = pl.boxes[name]
        if bx.src != "detected" or not bx.ok:
            return False, "box not detected"
        if not pl.slots_on():
            return False, "drop slots off"
        if bx.conf < float(self.c["min_conf"]):
            return False, f"box confidence {bx.conf:.2f} < far.min_conf"
        if len(pl.slots(name, side)):
            return False, "a drop slot is reachable"
        return True, ""

    # ------------------------------------------------------------------ geometry
    def _r(self):
        return 0.5 * float(self.c["object_size"])

    def _limits(self, bx):
        """Box-frame window for the landing centre: (u_near, u_far, v_half); region() adds pos_err at the far wall
        and the sides (module doc)."""
        r, lc = self._r(), float(self.c["land_clear"])
        hu, hv = bx.inner
        return (-hu + r + float(self.c["near_clear"]), hu - r - lc - float(self.c["far_clear"]),
                hv - r - float(self.c["side_clear"]))

    def _frame(self, bx):
        c, s = np.cos(bx.yaw), np.sin(bx.yaw)
        return np.array([c, s]), np.array([-s, c])

    def _point(self, bx, s, v, z):
        u, n = self._frame(bx)
        return np.r_[np.asarray(bx.c, float) + s * u + v * n, self.pl.scene.counter_z + z]

    def bulk_low(self, side, arm):
        """Lowest height (m) above its rim of the wrist / palm (wrist roll, wrist yaw joints, palm origin) over any
        box footprint + far.bulk_margin, along arm ticks (T, 14); inf if never over a box."""
        kin, zc = self.pl.kin, self.pl.scene.counter_z
        js = [kin.model.getJointId(f"{side}_{j}_joint") for j in ("wrist_roll", "wrist_yaw")]
        m, out = float(self.c["bulk_margin"]), np.inf
        for a in np.asarray(arm, float).reshape(-1, 14):
            kin._fk(a)
            P = np.array([kin.data.oMi[j].translation for j in js] + [kin.data.oMf[kin.palm[side]].translation])
            for bx in self.pl.boxes.all():
                ov = bx.inside(P[:, :2], m)
                if ov.any():
                    out = min(out, float((P[ov, 2] - (zc + bx.rim)).min()))
        return out

    def bulk_why(self, side, arm):
        """None if bulk_low(side, arm) >= far.wrist_clear, else the reason."""
        low, need = self.bulk_low(side, arm), float(self.c["wrist_clear"])
        return None if low >= need else f"wrist {100 * low:.1f} cm above a rim (far.wrist_clear {100 * need:.0f})"

    def envelope(self, name, side):
        """Reach envelope of box `name` for arm `side`: rows {v, z, s, q, tilt, side} (box-frame lateral v, height z
        above the counter, farthest u = s along the depth axis, the joints there). Cached per box pose."""
        pl, key = self.pl, (name, side)
        bx = pl.boxes[name]
        if key in self._env and not pl._moved(self._env[key][0], bx):
            return self._env[key][1]
        c, kin, zc = self.c, pl.kin, pl.scene.counter_z
        lo, hi = far_bounds(kin, c.get("joint_margin"))
        sl = ArmKinematics.sl(side)
        u, _ = self._frame(bx)
        vh = self._limits(bx)[2] - float(c["pos_err"]) - self.spread()[2]
        rows = []
        if vh >= 0:
            n = max(1, int(c["laterals"]))
            vs = np.linspace(-vh, vh, n) if n > 1 else np.zeros(1)
            sgn = 1.0 if side == "left" else -1.0
            for z in sorted(float(h) for h in c["heights"]):
                if z < bx.rim + self._r() + float(c["wrist_clear"]):
                    continue
                prev = None
                for v in sorted(vs, key=lambda v: -sgn * v):      # outward first: neighbours seed each other
                    r = max_reach(kin, side, u, v, zc + z, np.radians(float(c["max_tilt_deg"])), pl.home, lo[sl],
                                  hi[sl], c0=bx.c, seeds=[prev] if prev is not None else [],
                                  restarts=0 if prev is not None else 2)
                    if r is None:
                        continue
                    prev = r[1]
                    rows.append({"v": float(v), "z": z, "s": r[0], "q": r[1], "tilt": r[2], "side": side})
        self._env[key] = (bx, rows)
        return rows

    # ------------------------------------------------------------------ prediction
    def predict(self, bx, p0, vel=(0.0, 0.0, 0.0)):
        """Landing of the object centre released at p0 (3,) with velocity vel (3,): (pelvis xyz, box frame (u, v),
        flight time s)."""
        p, T = flight(p0, vel, self.pl.scene.counter_z + bx.floor + self._r())
        return p, bx.local(p[:2]), T

    def spread(self):
        """(forward shift min, max, sideways +-) of the landing vs the grasp point at the release (far.release_*)."""
        lo, hi = (float(x) for x in self.c["release_shift"])
        return lo, hi, float(self.c["release_side"])

    def fits(self, bx, uv):
        """(ok, why) of a release above uv (box frame: the grasp point's straight-down landing) against the window
        (module doc): the object lands far.release_shift ahead of it and up to far.release_side sideways."""
        u_near, u_far, v_half = self._limits(bx)
        e = float(self.c["pos_err"])
        lo, hi, side = self.spread()
        if uv[0] + lo < u_near:
            return False, f"lands from u {uv[0] + lo:+.3f} (needs {u_near:+.3f})"
        if uv[0] + hi + e > u_far:
            return False, f"lands up to u {uv[0] + hi:+.3f} (+ pos_err beyond the far limit {u_far:+.3f})"
        if abs(uv[1]) + side + e > v_half:
            return False, f"lands sideways up to v {abs(uv[1]) + side:.3f} (+ pos_err beyond {v_half:.3f})"
        return True, ""

    def visible(self, bx, uv, step=0.005, margin_px=20):
        """True if the head camera (self.camera) sees the top of an object landed at box-frame uv: inside the image
        (margin_px) and the line of sight above every box wall it crosses (walls: outer minus inner footprint, up to
        the rim). None without a camera model."""
        if self.camera is None:
            return None
        K, T, w, h = self.camera
        zc = self.pl.scene.counter_z
        p = np.r_[bx.pelvis(np.asarray(uv, float)), zc + bx.floor + 2 * self._r()]
        x = np.linalg.inv(T) @ np.r_[p, 1.0]
        if x[2] <= 0:
            return False
        px = K @ (x[:3] / x[2])
        if not (margin_px <= px[0] <= w - margin_px and margin_px <= px[1] <= h - margin_px):
            return False
        o = T[:3, 3]
        n = max(2, int(np.linalg.norm(p - o) / step))
        S = o + (p - o) * np.linspace(0.0, 1.0, n)[:-1, None]
        for b in self.pl.boxes.all():
            wall = b.inside(S[:, :2]) & ~(np.abs(b.local(S[:, :2])) <= b.inner).all(-1)
            if (wall & (S[:, 2] < zc + b.rim)).any():
                return False
        return True

    def grade(self, bx, uv):
        """('sure' | 'marginal' | None, near margin m (< 0: short), why) of a release above uv: fits(); marginal =
        fits except the near rule, short by <= far.max_short (the far wall and the sides must hold: a miss there could
        be off the table / in a wrong box)."""
        ok, why = self.fits(bx, uv)
        margin = uv[0] + self.spread()[0] - self._limits(bx)[0]
        if ok:
            return "sure", margin, ""
        if margin < 0 and -margin <= float(self.c["max_short"]) + 1e-9 and self.fits(bx, (uv[0] - margin, uv[1]))[0]:
            return "marginal", margin, why
        return None, margin, why

    def s_release(self, bx, row):
        """Release depth (box u) for an envelope row: as deep as the arm reaches (far.static_backoff short of it),
        never deeper than the far wall allows."""
        u_far, e, hi = self._limits(bx)[1], float(self.c["pos_err"]), self.spread()[1]
        return min(row["s"] - float(self.c["static_backoff"]), u_far - hi - e)

    # ------------------------------------------------------------------ planning
    def reachable(self, name, side):
        """(ok, how): does some envelope point of box `name` / arm `side` fit (the cheap part of release(): no IK,
        no wrist check; the envelope is cached per box pose). For gating a look before any per-object planning."""
        bx = self.pl.boxes[name]
        u_dir, _ = self._frame(bx)
        if float(np.asarray(bx.front, float) @ u_dir) > float(self.c["max_front"]):
            return False, "beyond far.max_front"
        rows = self.envelope(name, side)
        best, marginal = -np.inf, False
        for row in rows:
            g, m, _ = self.grade(bx, (self.s_release(bx, row), row["v"]))
            if g == "sure":
                return True, "far"
            marginal |= g == "marginal"
            best = max(best, m)
        if marginal:
            return True, f"far marginal: short by {1000 * -best:.0f} mm"
        return False, ("far: no release height reachable" if not rows
                       else f"far: short by {1000 * -best:.0f} mm")

    def release(self, name, side, hand_cls, hand_rel, extra=(), alt=0):
        """The far release into box `name` with arm `side`: (Release, '') or (None, reason). Lateral position
        sure before marginal (grade), then farthest from planner.drop_used[name] + extra, then landings the camera
        sees, then the largest near margin, then the lowest height. alt: start at the alt-th candidate of that order
        (a retry after a short miss: another release). hand_cls / hand_rel: (14,) closed and release hand vectors."""
        pl, c = self.pl, self.c
        bx = pl.boxes[name]
        u_dir, _ = self._frame(bx)
        front = float(np.asarray(bx.front, float) @ u_dir)
        if front > float(c["max_front"]):
            self.last = {"why": "max_front"}
            return None, f"far: near wall at {front:.3f} m > far.max_front"
        rows = self.envelope(name, side)
        if not rows:
            self.last = {"why": "no envelope"}
            return None, "far: nothing reachable at any release height"
        u_near, u_far, _ = self._limits(bx)
        used = [bx.local(np.asarray(p, float)[:2]) for p in list(pl.drop_used.get(name, ())) + list(extra)]
        sp, enough = float(c["min_spacing"]), float(c["margin_enough"])
        cands, best_miss = [], None
        for i, row in enumerate(rows):
            s_rel = self.s_release(bx, row)
            g, m, why = self.grade(bx, (s_rel, row["v"]))
            if g is None:
                if best_miss is None or m > best_miss[0]:
                    best_miss = (m, why)
                continue
            lo, hi, _ = self.spread()
            land = (s_rel + 0.5 * (lo + hi), row["v"])
            d = min((np.hypot(row["v"] - w[1], land[0] - w[0]) for w in used), default=np.inf)
            seen = all(self.visible(bx, (u, row["v"])) is not False for u in (s_rel + lo, s_rel + hi))
            cands.append((g != "sure", round(max(0.0, sp - d), 3), not seen, -min(m, enough), row["z"], i, s_rel))
        if not cands:
            self.last = {"why": best_miss[1], "short_by": -best_miss[0]}
            return None, f"far: {best_miss[1]} (best of {len(rows)} release points)"
        why = ""
        cands = sorted(cands)
        k = int(alt) % len(cands)
        for cand in cands[k:] + cands[:k]:
            out = self._build(name, side, bx, rows[cand[5]], cand[6], hand_cls, hand_rel)
            if isinstance(out, Release):
                self.last = out.info
                return out, ""
            why = out
        self.last = {"why": why}
        return None, f"far: {why}"

    def _build(self, name, side, bx, row, s_rel, hand_cls, hand_rel):
        """Ticks of one release point: Release, or a reason."""
        pl, c, dt = self.pl, self.c, self.pl.dt
        sl = ArmKinematics.sl(side)
        lo, hi = far_bounds(pl.kin, c.get("joint_margin"))
        p = self._point(bx, s_rel, row["v"], row["z"])
        pl.kin._fk(row["q"])
        R = pl.kin.data.oMf[pl.kin.palm[side]].rotation.copy()
        q, ok = solve_pose(pl.kin, side, p, R, row["q"], lo[sl], hi[sl])
        if not ok:
            return f"no IK at u {s_rel:+.3f}"
        why = self.bulk_why(side, q)
        if why is not None:
            return why
        h_c, h_r = np.asarray(hand_cls, float)[sl], np.asarray(hand_rel, float)[sl]
        n_open = max(1, int(np.ceil(np.abs(h_r - h_c).max() / (float(c["hand_vel"]) * dt) - 1e-9)))
        n_set = int(round(float(c["settle_s"]) / dt))
        hold = int(round(float(pl.cfg["motion"]["open_hold_s"]) / dt))
        ramp_h = h_c + (h_r - h_c) * (np.arange(1, n_open + 1) / n_open)[:, None]
        A = np.repeat(q[sl][None], n_set + n_open + hold, 0)
        H = np.vstack([np.repeat(h_c[None], n_set, 0), ramp_h, np.repeat(h_r[None], hold, 0)])
        land, uv, T = self.predict(bx, p)
        u_near, u_far, v_half = self._limits(bx)
        e = float(c["pos_err"])
        lo, hi, sd = self.spread()
        land = np.r_[bx.pelvis(np.array([uv[0] + 0.5 * (lo + hi), uv[1]])), land[2]]   # expected (bookkeeping)
        info = {"box": name, "side": side, "z": row["z"], "v_lat": row["v"], "s_env": row["s"], "s_rel": s_rel,
                "tilt_deg": float(np.degrees(row["tilt"])), "release": p.tolist(), "land_uv": uv.tolist(),
                "t_flight": T, "k_open": n_set, "ticks": len(A), "margin": float(uv[0] + lo - u_near),
                "margin_far": float(u_far - uv[0] - hi - e), "margin_side": float(v_half - abs(uv[1]) - sd - e),
                "marginal": bool(uv[0] + lo - u_near < 0), "under_uv": uv.tolist(), "visible": self.visible(bx, (uv[0] + 0.5 * (lo + hi), uv[1]))}
        return Release(q, A, H, land, info)


# ---------------------------------------------------------------------- reports
def reach_map(fronts=(0.43, 0.46, 0.47, 0.50, 0.53, 0.56), alongs=None, profile="none", walls=(0.068, 0.10),
              gap=0.02, yaw_deg=0.0, log=print):
    """Far reach per near edge x wall x box x arm x grasp point along the fingers: drop slots (stock), the far
    release's margin (m: landing centre beyond what the near wall needs; < 0 = short by that, not attempted), its
    height / lateral position, and the forward hand speed available there (forward_speed: what a release in motion
    could use). Grasp point: the config's, then its palm x (along the fingers) replaced by each of `alongs`."""
    from aaltonomous.scene.scene import Scene
    from .bins import Box, Boxes
    from .config import hand_postures, home_arm, load_config
    from .mock_world import box_row
    from .planner import Planner
    base = load_config(profile=profile)
    p0 = np.asarray(base["grasp"]["point_in_palm"], float)
    alongs = [float(p0[0])] + [a for a in (alongs or ()) if abs(a - p0[0]) > 1e-6]
    sc = Scene.load()
    out = []
    for along in alongs:
        cfg = load_config(profile=profile, overrides={"grasp.point_in_palm": [along, float(p0[1]), float(p0[2])],
                                                      "grasp.far.enabled": True})
        pl = Planner(cfg, sc, hand_postures(cfg), home_arm(cfg))
        for wall in walls:
            for front in fronts:
                rows = box_row(front, yaw_deg, wall, gap, size=0.21, wall_t=0.004)
                pl.set_boxes(Boxes({b["name"]: Box(b["name"], tuple(b["c"]), b["yaw"], b["size"] / 2, b["size"] / 2,
                                                   b["wall_h"], b["wall_t"], b["floor_t"], 1.0, True, "detected")
                                    for b in rows}, source="detected"))
                for name, side in (("left", "left"), ("center", "left"), ("center", "right"), ("right", "right")):
                    pl.drop_used = {}
                    rel, why = pl.far.release(name, side, pl.post["closed"], pl.post["release"])
                    rec = {"along": along, "wall": wall, "front": front, "box": name, "side": side,
                           "slots": len(pl.slots(name, side))}
                    if rel is not None:
                        i = rel.info
                        n_ok = sum(pl.far.fits(pl.boxes[name], (min(r["s"] - pl.far.c["static_backoff"], 0.0),
                                                                 r["v"]))[0] for r in pl.far.envelope(name, side))
                        rec.update(margin=round(i["margin"], 4), margin_far=round(i["margin_far"], 4),
                                   z=i["z"], v_lat=round(i["v_lat"], 3), points_ok=n_ok,
                                   fwd_speed=round(forward_speed(pl.kin, side, rel.q,
                                                                 pl.far._frame(pl.boxes[name])[0]), 3))
                    else:
                        rec.update(margin=-round(float((pl.far.last or {}).get("short_by") or np.nan), 4), why=why)
                    out.append(rec)
                    log(json.dumps(rec))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "map"
    if what == "map":
        reach_map(alongs=[float(a) for a in sys.argv[2:]] or [0.085, 0.115, 0.13])
