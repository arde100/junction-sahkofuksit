"""Pick-and-place planning for one detected object: Cartesian waypoints -> IK -> 30 Hz joint trajectories.

Sequence (one arm; the other holds its current target):
  home -> pregrasp at carry height (joint space, hand to preshape) -> descend (Cartesian) -> [settle: dwell at the grasp
  pose, grasp.correct] -> close -> lift -> carry
  above the bin (Cartesian, else joint space direct / via a raised midpoint; motion.joint_carry_first: joint space
  first) -> lower to drop height -> release (hand.release) -> retreat up -> home (hand back to open)
motion.shortcut_tol: a Cartesian segment becomes one joint-space move if the grasp point stays that close to its line.
Each side is tried with every grasp.finger_hints direction (fingers forward can touch the centre bin's wall), after a
cached reach test of the arm vs the bin (grasp.reach_check). plan(variant=k) changes that order for a retry: the hint
list is rotated, and for the centre bin (both arms reach it) odd variants try the other arm first.
abort(): after a verified miss / slip, the rest of one arm's queue -> decelerate along it, joint-space move home.
correction_dq() / correct() (grasp.correct, called by the policy in act() at the end of `settle`): the measured
tracking error of the grasp point -> a joint shift of that arm's queued ticks (blended in before the close, out after
the lift starts), checked like a plan on every shifted tick (bounds, speed, counter / rims, neighbours, other arm, 0
clips) or refused.
Cross-body (neither arm can do object -> bin): a RELOCATION plan picks with one arm and puts the object down on
the counter at a spot from which the bin-side arm has a full plan; the policy's next look picks it up from there.
Every planned tick is checked before it is used:
  * per-joint speed <= motion.arm_max_vel (trapezoidal time scaling on the inf-norm path length),
  * hand points (palm, grasp point, finger links) over the counter footprint stay above the counter
    (counter_clearance in transit; fingertip_clearance during the grasp itself),
  * hand points over a bin footprint (+ motion.bin_margin) stay above its rim, except while releasing,
  * a dry run through a fresh SafetyFilter (same config as the env) produces 0 clipped ticks.
Paired cycle (config two_arm): plan_pair() plans a second object with the other arm from the same start pose and
merge_plans() runs both from one (T, 14) queue, the second arm delayed by the first lag (0, lag_step_s, ... up to
lag_cap_frac of the first plan) at which the two arms' sphere model (kinematics.ARM_SPHERES, + a sphere for each held
object) stays two_arm.min_clearance apart and the merged dry run has 0 clips. Nothing else checks arm vs arm.
Parked arm (two_arm.avoid_parked): keep_clear() checks a single-arm plan against the other arm holding its pose (same
model, clearance and held-object mask); if it comes too close, evade() moves that arm to a two_arm.clear_poses pose and
back within the same merged queue (the plan delayed if needed).
Neighbours (config neighbours, plan(neighbours=[other detections])): every other object is a vertical cylinder on the
counter (shape circumradius, unknown shape = the largest; object height); every tick of a plan (and of an abort's
move home, an evade's idle-arm moves) keeps the arm's sphere model (kinematics.ARM_SPHERES, no held object) >=
neighbours.margin from each (or, if the start pose is already closer, no closer than there). A candidate that fails
it is rejected (Plan.blocked, reason names the object); least_bad=True accepts >= neighbours.floor instead and keeps
the candidate with the most clearance (pinch axis perpendicular to the nearest neighbour tried first).
Grasp candidates per arm (_candidates): face-aligned finger directions (grasp.face_align: cube face normals /
triangle at triangle_phi_deg from an apex, from Detection.shape / yaw, gated by shape_conf), then grasp.finger_hints.
Drop slots (grasp.slots): per (bin, arm) a few release points inside the bin (rows along the near wall, reach-tested,
spaced as far apart as the footprint allows); drop_slot() takes the one farthest from drop_used[bin] (the policy's
delivered objects / the oracle's objects in the bin). Off: grasp.drop_offsets as before.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from aaltonomous.interface.safety import JointSpec, SafetyConfig, SafetyFilter
from aaltonomous.interface.tasks import FIXED_MAPPING

from .instruction import object_bin
from .kinematics import ARM_MIRROR, ArmKinematics, cylinder_distance

PAIR_CHECK_STEP = 1  # ticks between arm-arm clearance checks of a merged pair (2 was not a guarantee: the distance can
                     # change by up to ~14 mm per tick away from its minimum; 1 costs ~5 ms more per lag tried)


@dataclass
class Plan:
    ok: bool
    side: str | None = None
    color: str = ""
    reason: str = ""
    target_bin: str | None = None
    waypoints: dict = field(default_factory=dict)      # name -> grasp point (pelvis)
    ik: dict = field(default_factory=dict)             # name -> (ok, pos_err, tilt_deg)
    arm: np.ndarray | None = None                      # (T, 14)
    hand: np.ndarray | None = None                     # (T, 14)
    phases: list = field(default_factory=list)         # (name, start tick, end tick)
    safety_clips: int | None = None
    variant: int = 0                                   # Planner.plan(variant=...) (retry with another grasp)
    hint: int | None = None                            # index into grasp.finger_hints of the finger direction used
    parts: list = field(default_factory=list)          # merged pair (merge_plans): [first, second] single-arm plans
    lags: dict = field(default_factory=dict)           # merged pair: side -> start delay (ticks) of its part
    clearance: tuple | None = None                     # merged pair: (min arm-arm model distance m, tick)
    evade: dict = field(default_factory=dict)          # Planner.evade: merged queue [plan, idle arm's move aside]
                                                       # and that idle part: side, pose, lag, back, ...
    blocked: bool = False                              # failed: some candidate failed only on a neighbour
    nb_clear: float | None = None                      # min sphere-model distance (m) to the neighbours (checked)
    nb_slack: float | None = None                      # ... minus what was required (>= 0 when accepted)
    nb: np.ndarray | None = None                       # neighbour cylinders it was checked against (x, y, r, top z)
    least_bad: bool = False                            # accepted by the least-bad rule (neighbours.floor)
    face: float | None = None                          # face-aligned grasp: finger direction (deg, pelvis)

    @property
    def duration(self):
        return 0.0 if self.arm is None else len(self.arm) / 30.0

    def summary(self):
        if self.parts and self.evade:
            p, e = self.parts[0], self.evade
            c = "" if self.clearance is None else f" clearance {1000 * self.clearance[0]:.0f}mm"
            return (f"EVADE {p.color} {p.side} arm -> {p.target_bin} (+{e['lag'] / 30:.1f} s) while the {e['side']} "
                    f"arm moves aside (clear pose {e['pose']}, back at {e['back'] / 30:.1f} s){c}  "
                    f"{self.duration:.1f}s (single {p.duration:.1f}s)  safety_clips={self.safety_clips}")
        if self.parts:
            parts = " + ".join(f"{p.color} {p.side} arm -> {p.target_bin} (+{self.lags[p.side] / 30:.1f} s)"
                               for p in self.parts)
            c = "" if self.clearance is None else f" clearance {1000 * self.clearance[0]:.0f}mm"
            return f"PAIR {parts}{c}  {self.duration:.1f}s  safety_clips={self.safety_clips}"
        if not self.ok:
            return f"{self.color:<6} SKIP ({self.reason})"
        wp = " ".join(f"{k}=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})" for k, v in self.waypoints.items()
                      if k in ("grasp", "drop"))
        tilt = max(t for _, _, t in self.ik.values())
        err = max(e for _, e, _ in self.ik.values()) * 1000
        nb = "" if self.nb_clear is None else f"  neighbours >= {1000 * self.nb_clear:.0f}mm"
        nb += " (LEAST-BAD)" if self.least_bad else ""
        face = "" if self.face is None else f" face {self.face:.0f}deg"
        return (f"{self.color:<6} {self.side:<5} arm -> {self.target_bin} bin  {wp}  IK ok max_err={err:.1f}mm "
                f"max_tilt={tilt:.0f}deg{face}  {self.duration:.1f}s{nb}  safety_clips={self.safety_clips}")


def time_scale(Qs, vmax, amax, dt):
    """Piecewise-linear joint path through Qs (n, d) -> samples (T, d) at dt, EXCLUDING Qs[0], with a trapezoidal
    profile on the inf-norm arc length: no joint ever moves faster than vmax."""
    Qs = np.asarray(Qs, float)
    seg = np.abs(np.diff(Qs, axis=0)).max(axis=1)
    S = float(seg.sum())
    if S < 1e-9:
        return np.empty((0, Qs.shape[1]))
    ta = vmax / amax
    if amax * ta * ta > S:                       # triangle profile
        ta = np.sqrt(S / amax)
        vpk = amax * ta
        T = 2 * ta
    else:
        vpk = vmax
        T = 2 * ta + (S - amax * ta * ta) / vmax
    n = max(1, int(np.ceil(T / dt)))
    t = np.arange(1, n + 1) * (T / n)
    s = np.where(t < ta, 0.5 * amax * t ** 2,
                 np.where(t <= T - ta, 0.5 * amax * ta ** 2 + vpk * (t - ta), S - 0.5 * amax * (T - t) ** 2))
    s = np.clip(s, 0, S)
    cum = np.r_[0.0, np.cumsum(seg)]
    out = np.empty((n, Qs.shape[1]))
    for k, sk in enumerate(s):
        i = min(int(np.searchsorted(cum, sk, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] < 1e-12 else (sk - cum[i]) / seg[i]
        out[k] = Qs[i] + f * (Qs[i + 1] - Qs[i])
    return out


def _slack(p):
    return np.inf if p.nb_slack is None else p.nb_slack


def tri_shift(phi, r=0.030):
    """Midpoint (m, along f) of the triangle's extent in direction f, f at angle phi from the apex direction; vertices
    at r from the centroid (apex 30 mm, faces 15 mm). phi=0 -> +7.5 mm (towards the apex), phi=60 deg -> -7.5 mm."""
    v = np.radians([0.0, 120.0, 240.0])
    return 0.5 * (r * np.cos(phi - v).max() - r * np.cos(phi + np.pi - v).max())


def ramp(a, b, vmax, dt):
    """Linear move a -> b (excluding a) with every component at most vmax."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(np.ceil(np.abs(b - a).max() / (vmax * dt) + 1e-9))
    if n == 0:
        return np.empty((0, len(a)))
    return a + (b - a) * (np.arange(1, n + 1) / n)[:, None]


def correct_ticks(cc, dt=1.0 / 30.0):
    """grasp.correct -> (ticks of the settle phase before the measurement = settle_s + obs_lag, blend-in ticks).
    The settle phase is their sum: the policy measures after the first count and blends the shift in over the second,
    which ends where the close starts."""
    return (int(np.ceil(float(cc.get("settle_s", 0.2)) / dt - 1e-9)) + int(cc.get("obs_lag", 2)),
            max(1, int(round(float(cc.get("ramp_s", 0.13)) / dt))))


def merge_plans(first: Plan, second: Plan, lag=0, arm_q=None, hand_q=None):
    """One queue from two single-arm plans (different arms) made from the same start pose arm_q / hand_q (14,):
    each arm's 7 arm + 7 hand columns from its own plan, `second` delayed by `lag` ticks (its arm holds the start
    pose meanwhile), the shorter part padded with its last tick. Default start pose: the other plan's columns of that
    arm (a single-arm plan keeps them constant). phases: ('<side>:<name>', start, end), shifted by each arm's lag;
    parts / lags keep the part plans (records, bins, variants, verification windows)."""
    if first.side == second.side or {first.side, second.side} != {"left", "right"}:
        raise ValueError(f"merge_plans needs one left and one right plan ({first.side}, {second.side})")
    lags = {first.side: 0, second.side: int(lag)}
    T = max(len(p.arm) + lags[p.side] for p in (first, second))
    arm, hand, phases = np.empty((T, 14)), np.empty((T, 14)), []
    for p, o in ((first, second), (second, first)):
        sl, L = ArmKinematics.sl(p.side), lags[p.side]
        a0 = (o.arm[0] if arm_q is None else np.asarray(arm_q, float))[sl]
        h0 = (o.hand[0] if hand_q is None else np.asarray(hand_q, float))[sl]
        k = np.minimum(np.arange(T) - L, len(p.arm) - 1)
        arm[:, sl] = np.where((k < 0)[:, None], a0, p.arm[np.maximum(k, 0), sl])
        hand[:, sl] = np.where((k < 0)[:, None], h0, p.hand[np.maximum(k, 0), sl])
        phases += [(f"{p.side}:{n}", s + L, e + L) for n, s, e in p.phases]
    return Plan(True, side="both", color=f"{first.color}+{second.color}", arm=arm, hand=hand,
                phases=sorted(phases, key=lambda x: x[1]), variant=first.variant, hint=first.hint,
                parts=[first, second], lags=lags)


def held_masks(plan: Plan, T=None):
    """(left (T,), right (T,)) bool: the hand may hold an object, from the start of its `close` phase to the end of its
    `open` phase (merged pair: per part, shifted by its lag; single plan: its own side). T default: len(plan.arm)."""
    T = len(plan.arm) if T is None else int(T)
    out = {"left": np.zeros(T, bool), "right": np.zeros(T, bool)}
    for p in plan.parts or [plan]:
        ph, L = {n: (s, e) for n, s, e in p.phases}, plan.lags.get(p.side, 0)
        if p.side in out and "close" in ph and "open" in ph:
            out[p.side][ph["close"][0] + L: ph["open"][1] + L] = True
    return out["left"], out["right"]


def checked_masks(held, step=PAIR_CHECK_STEP):
    """held masks for a clearance check every `step` ticks: a tick also counts the objects of the next step-1 ticks,
    so every unchecked tick's spheres are a subset of those of the checked tick before it (an object sphere that
    appears at an odd tick is not skipped)."""
    out = []
    for h in held:
        h = np.asarray(h, bool)
        o = h.copy()
        for i in range(1, int(step)):
            o[:-i] |= h[i:]
        out.append(o)
    return out


def phase_at(plan: Plan, side, k):
    """Name of `side`'s phase at tick k of a (merged) plan ('wait' before it starts, 'done' after it ends)."""
    p = next((q for q in plan.parts if q.side == side), None) if plan.parts else plan
    if p is None or p.side != side:
        return "idle"
    k = int(k) - plan.lags.get(side, 0)
    if k < 0:
        return "wait"
    return next((n for n, s, e in p.phases if s <= k < e), "done")


class Planner:
    def __init__(self, cfg, scene, postures, home, kin: ArmKinematics | None = None, safety: SafetyConfig | None = None,
                 spec: JointSpec | None = None, mapping=None, dt=1.0 / 30.0):
        self.cfg, self.scene, self.post, self.home = cfg, scene, postures, np.asarray(home, float)
        self.spec = spec or JointSpec()
        self.kin = kin or ArmKinematics(cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=dt)
        self.dt = dt
        self.height = float(cfg["perception"].get("object_height") or scene.object_height())
        self.mapping = mapping or dict(FIXED_MAPPING)  # plan() without bin_name only (the policies pass the bin)
        self.drop_offsets = {}  # bin -> (dx, dy) pelvis offset of the release point (set per object by the policy)
        self.drop_used = {}     # grasp.slots: bin -> [pelvis xy] of objects delivered there (set by the policy)
        self._drop_avoid = []   # plan_side(drop_avoid=...): more points to keep away from (the first plan of a pair)
        self._slots = {}        # (bin, side) -> slot xy (k, 2) in preference order
        self._nb, self._nb_tags = None, []  # neighbour cylinders (x, y, r, top z) of the current plan call + labels
        self._least = False     # least-bad pass: neighbours.floor instead of neighbours.margin, most clearance wins
        self._reach = {}        # (side, bin) -> reach test result (grasp.reach_check)
        self._dry = None        # SafetyFilter reused by dry_run_safety (building one loads the URDF)
        self._clear = None      # two_arm.clear_poses as (14,) vectors (clear_poses())
        self._asides = {}       # evade: (side, start, hand, clear pose) -> checked moves aside / back

    # ------------------------------------------------------------------ geometry helpers
    def bin_xy(self, name):
        return self.scene.mat_to_pelvis(np.r_[self.scene.cfg["bins"]["centers"][name], 0.0])[:2]

    def drop_xy(self, name, offset=True):
        """Release point (pelvis xy) in bin `name`: the point of the inner footprint nearest the pelvis, `grasp.drop_inset`
        inside the inner walls (the published box row puts the box centres past the arm's reach). None -> centre.
        offset: + drop_offsets[name], so the next object does not land on top of the previous one."""
        inset = self.cfg["grasp"].get("drop_inset")
        c = self.bin_xy(name)
        if inset is None:
            out = c
        else:
            lo, hi = self.scene.bin_inner_box(name)
            a, b = self.scene.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2], self.scene.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]
            lo2 = np.minimum(np.minimum(a, b) + inset, c)
            hi2 = np.maximum(np.maximum(a, b) - inset, c)
            out = np.clip(np.zeros(2), lo2, hi2)
        if offset and name in self.drop_offsets:
            out = out + np.asarray(self.drop_offsets[name], float)
        return out

    # ------------------------------------------------------------------ drop slots (grasp.slots)
    def slots_on(self):
        return bool((self.cfg["grasp"].get("slots") or {}).get("enabled"))

    def prepare(self):
        """Compute (cache) the drop slots of every bin for the arm(s) that deliver there (centre bin: both; side bin:
        its own side), so the first look does not pay for the reach tests."""
        if self.slots_on():
            for b in self.scene.bin_names:
                y = self.bin_xy(b)[1]
                for side in (("left", "right") if abs(y) <= 0.1 else ("left" if y > 0 else "right",)):
                    self.slots(b, side)

    def slots(self, name, side):
        """Drop slots of bin `name` for arm `side`: pelvis xy (k, 2), in preference order (see _make_slots)."""
        key = (name, side)
        if key not in self._slots:
            # reach-tested with the default finger direction: computed lazily inside a face-aligned candidate (its
            # direction forced with hint_weight) the IK above the bin tilts and NO slot passes, cached for good
            saved = (self.kin.hint, self.kin.w_hint)
            g = self.cfg["grasp"]
            self.kin.hint = np.asarray(g.get("finger_hint", [1.0, 0.0, 0.0]), float)
            self.kin.w_hint = float(g.get("finger_hint_weight", 0.0))
            try:
                self._slots[key] = self._make_slots(name, side)
            finally:
                self.kin.hint, self.kin.w_hint = saved
        return self._slots[key]

    def _slot_reach(self, side, xy):
        """Can `side` hold its grasp point top-down (tilt <= slots.max_tilt_deg) over xy at carry and at drop
        height (IK from home, no random restarts: the plan's own IK starts from a nearby pose)?"""
        sc, g = self.scene, self.cfg["grasp"]
        wall = float(sc.cfg["bins"]["wall_height"])
        zc, tilt = sc.counter_z, np.radians(float(g["slots"].get("max_tilt_deg", 10.0)))
        q1, ok1, _, t1 = self.kin.solve(side, np.r_[xy, zc + max(g["carry_height"], wall + self.height + 0.03)],
                                        self.home, restarts=False)
        if not ok1 or t1 > tilt:
            return False
        _, ok2, _, t2 = self.kin.solve(side, np.r_[xy, zc + wall + self.height / 2 + g["drop_above_rim"]], q1,
                                       restarts=False)
        return bool(ok2 and t2 <= tilt)

    def _make_slots(self, name, side):
        """Rows parallel to the bin's near wall, inside its inner footprint inset by slots.wall_clear. Row 0 (nearest
        the pelvis): from the reachable extent of that line (the ends bisected towards the stock drop point until
        _slot_reach holds), 1 + floor(extent / min_spacing) points spread evenly over it. Row r > 0: the midpoints of
        row r-1 (staggered), min_spacing from them diagonally, kept where reachable. Order: greedy farthest-first,
        starting at the row-0 end nearest the stock drop point (centre bin: the end on the arm's own side), capped at
        slots.count. Empty if nothing is reachable (drop_slot then uses the stock drop point)."""
        s = self.cfg["grasp"]["slots"]
        lo, hi = self.scene.bin_inner_box(name)
        a, b = self.scene.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2], self.scene.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]
        wc, sp = float(s.get("wall_clear", 0.04)), float(s.get("min_spacing", 0.06))
        (x0, y0), (x1, y1) = np.minimum(a, b) + wc, np.maximum(a, b) - wc
        if x0 > x1 or y0 > y1:
            return np.empty((0, 2))
        nom = self.drop_xy(name, offset=False)
        yn = float(np.clip(nom[1], y0, y1))
        if not self._slot_reach(side, (x0, yn)):
            return np.empty((0, 2))
        ends = []
        for ye in (y0, y1):        # reachable extent of row 0: bisect each end towards the nominal point
            good, bad = yn, ye
            if self._slot_reach(side, (x0, ye)):
                good = ye
            else:
                for _ in range(4):
                    m = 0.5 * (good + bad)
                    good, bad = (m, bad) if self._slot_reach(side, (x0, m)) else (good, m)
            ends.append(good)
        k = 1 + int(np.floor((ends[1] - ends[0]) / sp + 1e-9))
        rows = [[(x0, y) for y in (np.linspace(ends[0], ends[1], k) if k > 1 else [yn])]]
        for _ in range(1, int(s.get("rows", 2))):
            prev = rows[-1]
            ys = [0.5 * (p[1] + q[1]) for p, q in zip(prev, prev[1:])] or [prev[0][1]]
            half = 0.5 * (prev[1][1] - prev[0][1]) if len(prev) > 1 else 0.0
            x = prev[0][0] + np.sqrt(max(sp * sp - half * half, 0.0))
            row = [(x, y) for y in ys if x <= x1 and self._slot_reach(side, (x, y))]
            if not row:
                break
            rows.append(row)
        pts = [np.array(p, float) for r in rows for p in r]
        sgn = 1.0 if side == "left" else -1.0
        e = [rows[0][0], rows[0][-1]]
        seed = min(range(2), key=lambda i: (round(np.linalg.norm(np.subtract(e[i], nom)), 3), -sgn * e[i][1]))
        order = [next(i for i, p in enumerate(pts) if np.allclose(p, e[seed]))]
        while len(order) < min(len(pts), int(s.get("count", 5))):
            rest = [i for i in range(len(pts)) if i not in order]
            order.append(max(rest, key=lambda i: min(np.linalg.norm(pts[i] - pts[j]) for j in order)))
        return np.array([pts[i] for i in order])

    def drop_slot(self, name, side, extra=()):
        """The slot of bin `name` / arm `side` farthest from the points in drop_used[name] (+ `extra`, + the
        plan_side drop_avoid points): max of the min distance; ties (1 mm) -> preference order. None: no slots."""
        S = self.slots(name, side)
        if not len(S):
            return None
        used = [np.asarray(u, float)[:2] for u in list(self.drop_used.get(name, ())) + list(extra) + self._drop_avoid]
        if not used:
            return S[0].copy()
        U = np.array(used)
        d = np.linalg.norm(S[:, None] - U[None], axis=-1).min(1)
        return S[int(np.flatnonzero(d >= d.max() - 1e-3)[0])].copy()

    def _drop_point(self, name, side):
        """Release point (pelvis xy) of a pick -> bin plan: a drop slot (grasp.slots) or drop_xy (drop_offsets)."""
        if self.slots_on() and name in self.scene.bin_names:
            p = self.drop_slot(name, side)
            if p is not None:
                return p
        return self.drop_xy(name)

    # ------------------------------------------------------------------ neighbours (config neighbours)
    def neighbour_cylinders(self, neighbours):
        """Other objects -> (rows (m, 4): x, y, radius, top z; labels). neighbours: Detections (shape, top) or xy.
        Radius = the shape's circumradius (scene.object_dims), unknown shape = the largest. None if the check is
        off (neighbours.enabled)."""
        nc = self.cfg.get("neighbours") or {}
        if not nc.get("enabled"):
            return None, []
        sc, rows, tags = self.scene, [], []
        rmax = max(sc.object_dims(s)["circumradius"] for s in ("cube", "cylinder", "triangle"))
        for o in neighbours:
            top = np.asarray(getattr(o, "top", o), float)
            shape = getattr(o, "shape", None)
            r = sc.object_dims(shape)["circumradius"] if shape in ("cube", "cylinder", "triangle") else rmax
            rows.append([top[0], top[1], r, sc.counter_z + sc.object_height(shape)])
            tags.append(f"{getattr(o, 'color', 'object')}({top[0]:+.3f},{top[1]:+.3f})")
        return np.array(rows, float).reshape(-1, 4), tags

    def _scoped(self, neighbours, least_bad, fn, *a, **kw):
        """fn(*a, **kw) with the neighbour cylinders of `neighbours` (None: keep the current ones) and the
        least-bad switch (None: keep) set for its duration."""
        saved = (self._nb, self._nb_tags, self._least)
        if neighbours is not None:
            self._nb, self._nb_tags = self.neighbour_cylinders(neighbours)
        if least_bad is not None:
            self._least = bool(least_bad)
        try:
            return fn(*a, **kw)
        finally:
            self._nb, self._nb_tags, self._least = saved

    def neighbour_clearance(self, side, arm_q, hand_q, arm, hand, cyl=None, kin=None):
        """Sphere-model distance (m) of `side`'s arm to each neighbour cylinder at the start pose arm_q / hand_q and
        at every tick of (arm, hand): (d0 (m,), D (T, m)). cyl default: the current plan call's neighbours."""
        cyl = self._nb if cyl is None else cyl
        kin = kin or self.kin
        c, r = kin.hand_spheres(side, np.vstack([np.asarray(arm_q, float)[None], arm]),
                                np.vstack([np.asarray(hand_q, float)[None], hand]))
        D = cylinder_distance(c, r, cyl, self.scene.counter_z)
        return D[0], D[1:]

    def _nb_need(self, d0, least=None):
        """Required distance per neighbour: neighbours.margin (least-bad: .floor), or 2 mm less than at the start pose
        if that is already closer (an arm parked above an object must be able to leave)."""
        nc = self.cfg.get("neighbours") or {}
        least = self._least if least is None else least
        need = float(nc.get("floor", 0.0) if least else nc.get("margin", 0.01))
        return np.minimum(need, d0 - 0.002)

    def _good_enough(self, p):
        """Least-bad pass: stop searching at a plan whose model distance to every neighbour is >= neighbours.
        least_bad_good (default 0: no contact in the conservative model); planning every candidate of both arms
        took up to 12 s per look."""
        good = float((self.cfg.get("neighbours") or {}).get("least_bad_good", 0.0))
        return p.nb_clear is None or p.nb_clear >= good

    def _check_neighbours(self, side, arm_q, hand_q, plan):
        """Neighbour check of a finished single-arm plan (every tick): sets plan.nb / nb_clear / nb_slack; on a
        violation plan.blocked and plan.reason (which object, phase, distance). Returns True if clear."""
        if self._nb is None or not len(self._nb):
            return True
        d0, D = self.neighbour_clearance(side, arm_q, hand_q, plan.arm, plan.hand)
        need = self._nb_need(d0)
        S = D - need
        plan.nb, plan.nb_clear, plan.nb_slack = self._nb.copy(), float(D.min()), float(S.min())
        if plan.nb_slack >= 0:
            return True
        k, j = np.unravel_index(int(np.argmin(S)), S.shape)
        ph = next((n for n, s, e in plan.phases if s <= k < e), "?")
        plan.blocked = True
        plan.reason = (f"hand {1000 * D[k, j]:.0f} mm from {self._nb_tags[j]} during {ph} "
                       f"(need {1000 * need[j]:.0f})")
        return False

    def _over_counter(self, p):
        pm = self.scene.pelvis_to_mat(p)
        c = self.scene.cfg["counter"]
        mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
        lo, hi = mat / 2 - size / 2, mat / 2 + size / 2
        return np.all((pm[..., :2] >= lo) & (pm[..., :2] <= hi), axis=-1)

    def bin_clearance(self, side, arm_q, hand_q, margin=None):
        """Lowest height (m) above the bin rims of the hand points over a bin's outer footprint (+margin, default
        motion.bin_margin), inf if none. Bins are static boxes in the sim and on the table: only the drop itself may go
        below a rim."""
        return self._bin_clear(self.kin.hand_points(side, arm_q, hand_q), margin)

    def _bin_clear(self, pts, margin=None):
        if margin is None:
            margin = self.cfg["motion"].get("bin_margin")
            margin = 0.01 if margin is None else float(margin)
        pm = self.scene.pelvis_to_mat(pts)
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + margin
        rim = self.scene.counter_z + float(b["wall_height"])
        out = np.inf
        for c in b["centers"].values():
            over = np.all(np.abs(pm[:, :2] - np.asarray(c, float)) <= half, axis=1)
            if over.any():
                out = min(out, float((pts[over, 2] - rim).min()))
        return out

    def clearance(self, side, arm_q, hand_q, fingers_only=False):
        """Lowest height (m) above the counter surface of the hand points that are over the counter (inf if none)."""
        pts = self.kin.hand_points(side, arm_q, hand_q)
        return self._counter_clear(pts[2:] if fingers_only else pts)

    def _counter_clear(self, pts):
        over = self._over_counter(pts)
        if not over.any():
            return np.inf
        return float((pts[over, 2] - self.scene.counter_z).min())

    def _clear_many(self, P, rim=None):
        """_counter_clear and _bin_clear (motion.bin_margin) of hand points P (T, n, 3) per tick, vectorised:
        (counter (T,), bin rims (T,); inf where no point is over it, or where rim (T,) is False)."""
        pm = self.scene.pelvis_to_mat(P.reshape(-1, 3)).reshape(P.shape)
        c = self.scene.cfg["counter"]
        mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
        over = np.all((pm[..., :2] >= mat / 2 - size / 2) & (pm[..., :2] <= mat / 2 + size / 2), axis=-1)
        cc = np.where(over, P[..., 2] - self.scene.counter_z, np.inf).min(-1)
        margin = self.cfg["motion"].get("bin_margin")
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + (0.01 if margin is None else float(margin))
        rim_z = self.scene.counter_z + float(b["wall_height"])
        bb = np.full(len(P), np.inf)
        for ctr in b["centers"].values():
            ov = np.all(np.abs(pm[..., :2] - np.asarray(ctr, float)) <= half, axis=-1)
            bb = np.minimum(bb, np.where(ov, P[..., 2] - rim_z, np.inf).min(-1))
        return cc, (bb if rim is None else np.where(rim, bb, np.inf))

    def tick_rules(self, phases, T, place=False):
        """Per-tick clearance rules of a single-arm plan of T ticks: (counter need (T,) m, bin rims checked (T,)).
        The grasp phases (descend, settle, close, lift; a counter place also carry, open, retreat) keep
        grasp.fingertip_clearance, the close (and a place's open) 3 mm (the thumb swings through its lowest point:
        touching the mat lightly is acceptable), the rest motion.counter_clearance; the bin rims every tick except
        while releasing (open)."""
        g, mo = self.cfg["grasp"], self.cfg["motion"]
        near = ("descend", "settle", "close", "lift") + (("carry", "open", "retreat") if place else ())
        need, rim = np.full(int(T), float(mo["counter_clearance"])), np.ones(int(T), bool)
        for name, s, e in phases:
            if name in near:
                need[s:e] = g["fingertip_clearance"]
            if name == "close" or (place and name == "open"):
                need[s:e] = 0.003
            if name == "open":
                rim[s:e] = False
        return need, rim

    # ------------------------------------------------------------------ IK helpers
    def _ik(self, side, p, seed, name, plan):
        q, ok, err, tilt = self.kin.solve(side, p, seed, top_down=True, extra_seeds=[self.home[ArmKinematics.sl(side)]])
        plan.ik[name] = (ok, err, float(np.degrees(tilt)))
        plan.waypoints[name] = np.asarray(p, float)
        return q, ok

    def _cartesian(self, side, p0, p1, q0, name, plan):
        """IK waypoints along the straight line p0 -> p1 (excluding p0), seeded continuously. A step whose joints
        jump more than `max_joint_step` is bisected (up to 3 times) before it counts as a failure. Then
        motion.shortcut_tol: only the last waypoint if the joint-space line keeps the grasp point near p0 -> p1."""
        n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / self.cfg["motion"]["cart_step"])))
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        qs, q, info = [], q0, [True, 0.0, 0.0]
        jmax = float(self.cfg["motion"].get("max_joint_step", 0.4))

        def seg(pa, pb, q, depth):
            qn, ok, err, tilt = self.kin.solve(side, pb, q, top_down=True, restarts=False)
            if ok and np.abs(qn - q).max() <= jmax:
                info[1], info[2] = max(info[1], err), max(info[2], float(np.degrees(tilt)))
                return [qn]
            if depth >= 3:
                plan.reason = plan.reason or (f"IK jump on {name}" if ok else "")
                return None
            pm = 0.5 * (pa + pb)
            a = seg(pa, pm, q, depth + 1)
            b = a and seg(pm, pb, a[-1], depth + 1)
            return a + b if b else None

        for k in range(1, n + 1):
            out = seg(p0 + (p1 - p0) * (k - 1) / n, p0 + (p1 - p0) * k / n, q, 0)
            if out is None:
                plan.ik[name] = (False, info[1], info[2])
                return None
            qs += out
            q = out[-1]
        plan.ik[name] = (True, info[1], info[2])
        plan.waypoints[name] = p1
        return self._shortcut(side, p0, p1, q0, qs)

    def _shortcut(self, side, p0, p1, q0, qs):
        """[qs[-1]] if the joint-space line q0 -> qs[-1] keeps the grasp point within motion.shortcut_tol of the line
        p0 -> p1, else qs. The per-waypoint IK can let wrist yaw wander (oracle face-aligned grasps: ~3x longer joint
        paths, i.e. ~3x slower moves at the same joint speed limit)."""
        tol = self.cfg["motion"].get("shortcut_tol")
        if not tol or len(qs) < 2:
            return qs
        q0, q1 = np.asarray(q0, float), qs[-1]
        d = p1 - p0
        L2 = float(d @ d)
        for f in np.linspace(0, 1, 12)[1:-1]:
            p = self.kin.grasp_pose(side, q0 + f * (q1 - q0))[0]
            t = np.clip((p - p0) @ d / L2, 0, 1) if L2 > 0 else 0.0
            if np.linalg.norm(p - (p0 + t * d)) > tol:
                return qs
        return [q1]

    def _carry(self, side, q0, p0, p1, z_drop, plan):
        """lift -> above_bin: Cartesian, else joint space (grasp point above z_drop). motion.joint_carry_first: joint
        space first (usually the shorter joint path), grasp point above the bin drop height (also for a relocation)."""
        if self.cfg["motion"].get("joint_carry_first"):
            wall = float(self.scene.cfg["bins"]["wall_height"])
            z_min = self.scene.counter_z + wall + self.height / 2 + self.cfg["grasp"]["drop_above_rim"]
            reason = plan.reason
            out = self._joint_carry(side, q0, p0, p1, z_min, "carry", plan)
            if out is not None:
                return out
            plan.reason = reason
        return (self._cartesian(side, p0, p1, q0, "carry", plan)
                or self._joint_carry(side, q0, p0, p1, z_drop, "carry", plan))

    def _joint_carry(self, side, q_from, p_from, p_to, z_min, name, plan):
        """Joint-space move to an IK solution at p_to (direct, else via a raised midpoint), accepted if the grasp
        point never dips below z_min on the way."""
        def low(qa, qb):
            return any(self.kin.grasp_pose(side, qa + f * (qb - qa))[0][2] < z_min for f in np.linspace(0, 1, 15))

        q_to, ok = self._ik(side, p_to, q_from, name, plan)
        if ok and not low(q_from, q_to):
            return [q_to]
        for dz in (0.06, 0.12):
            mid = 0.5 * (np.asarray(p_from, float) + p_to) + [0, 0, dz]
            q_mid, ok1 = self._ik(side, mid, q_from, name + "_mid", plan)
            if not ok1 or low(q_from, q_mid):
                continue
            q_to, ok2 = self._ik(side, p_to, q_mid, name, plan)
            if ok2 and not low(q_mid, q_to):
                plan.reason = ""
                return [q_mid, q_to]
        plan.ik[name] = (False, 0.0, 0.0)
        plan.reason = f"{name}: no joint-space path above the bin rim"
        return None

    # ------------------------------------------------------------------ main
    def plan(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True, variant=0, neighbours=None,
             least_bad=None):
        """Plan for Detection `det` starting from the commanded arm_q/hand_q (14,). Tries the preferred arm (the
        target bin's side; for the centre bin the object's side), then the other one. If neither arm can do the
        whole pick -> bin (cross-body: the arm that reaches the object cannot reach the bin), plans a RELOCATION
        instead (`grasp.relocate`): pick + place on the counter at a spot from which an arm can finish the job;
        the next look re-detects the object there. obstacles: (n, 2) pelvis xy of the other detected objects.
        variant: retry hook (0 = the order above). Centre bin: odd variants try the other arm first, the finger hint
        order is rotated by variant // 2; side bins: rotated by variant.
        neighbours: the other objects (Detections) every tick must keep clear of (config neighbours); a failure
        caused by one has .blocked (then no relocation: taking another object first frees this one). least_bad:
        neighbours.floor instead of .margin, both arms planned, the most clearance wins.
        Returns the first Plan that passes every check, else the last failure."""
        return self._scoped(neighbours, least_bad, self._plan, det, arm_q, hand_q, bin_name, obstacles, relocate,
                            variant)

    def _plan(self, det, arm_q, hand_q, bin_name, obstacles, relocate, variant):
        g = self.cfg["grasp"]
        # the policies pass the object's bin (ClassicalPolicy.bin_for); direct callers (tests, tools): the same rule
        bin_name = bin_name or object_bin(self.mapping, det.color, getattr(det, "shape", None))
        by = self.bin_xy(bin_name)[1] if bin_name is not None else 0.0
        centre = abs(by) <= 0.1
        ref = det.top[1] if centre else by
        side0 = "left" if ref > g["y_split"] else "right"
        last = Plan(False, color=det.color, reason=f"no bin for {det.color} {getattr(det, 'shape', None) or 'object'}")
        if bin_name is None:
            return last
        variant = int(variant)
        sides = (side0, "right" if side0 == "left" else "left")
        if centre and variant % 2:
            sides = sides[::-1]
        rot = variant // 2 if centre else variant
        reasons, good, blocked = [], [], False
        for side in sides:
            p = self._plan_side_hints(det, side, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float),
                                      variant=rot)
            p.variant = variant
            if p.ok:
                if g.get("arm", "best") != "best" and (not self._least or self._good_enough(p)):
                    return p
                good.append(p)
            else:
                reasons.append(f"{side}: {p.reason}")
                blocked |= p.blocked
                last = p
        if good:
            if self._least:  # most neighbour clearance (1 mm ties: arm order)
                return next(p for p in good if _slack(p) >= max(_slack(q) for q in good) - 1e-3)
            return min(good, key=lambda p: p.duration)
        last.reason = "; ".join(reasons)
        last.blocked = blocked
        if blocked:  # (a relocation would be blocked as well, and costs 10-30 s to find out)
            return last
        if relocate and g.get("relocate", {}).get("enabled", True):
            r = self.plan_relocation(det, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float), obstacles,
                                     variant=rot)
            r.variant = variant
            if r.ok:
                return r
            last.reason += f"; relocate: {r.reason}"
        return last

    # ------------------------------------------------------------------ paired two-arm cycle
    def plan_side(self, det, side, bin_name, arm_q, hand_q, variant=0, drop_offset=None, drop_avoid=None,
                  neighbours=None):
        """Pick `det` -> bin `bin_name` with the given arm only (no relocation). variant: finger hint rotation as in
        plan() (centre bin: variant // 2). drop_offset: (dx, dy) used instead of drop_offsets[bin_name] (the next drop
        point when both objects of a pair go to the same bin); drop_avoid: pelvis xy points the drop slot keeps away
        from as if delivered (grasp.slots: the first plan's drop point). neighbours: as plan()."""
        by = self.bin_xy(bin_name)[1]
        rot = int(variant) // 2 if abs(by) <= 0.1 else int(variant)
        saved, avoid = dict(self.drop_offsets), self._drop_avoid
        if drop_offset is not None:
            self.drop_offsets[bin_name] = drop_offset
        self._drop_avoid = avoid + [np.asarray(a, float)[:2] for a in (drop_avoid or ())]
        try:
            p = self._scoped(neighbours, None, self._plan_side_hints, det, side, bin_name, np.asarray(arm_q, float),
                             np.asarray(hand_q, float), variant=rot)
        finally:
            self.drop_offsets, self._drop_avoid = saved, avoid
        p.variant = int(variant)
        return p

    def pair(self, first, second, arm_q, hand_q, kin=None):
        """Merge two single-arm plans from arm_q / hand_q: `first` is never delayed, `second` starts at lag 0, then
        two_arm.lag_step_s, ... up to two_arm.lag_cap_frac * first.duration. Accepted at the first lag where the sphere
        model distance (held_masks: object spheres from close to open, checked_masks) is >= two_arm.min_clearance at
        every tick (PAIR_CHECK_STEP) and a safety dry run from arm_q / hand_q has 0 clips.
        Returns (merged Plan, None) or (None, reason: first tick below the clearance at lag 0 and at the last lag)."""
        tw, kin = self.cfg.get("two_arm") or {}, kin or self.kin
        need = float(tw.get("min_clearance", 0.02))
        step = max(1, int(round(float(tw.get("lag_step_s", 0.5)) / self.dt)))
        cap = int(float(tw.get("lag_cap_frac", 0.6)) * len(first.arm))
        fails, clipped = [], 0
        for lag in range(0, cap + 1, step):
            m = merge_plans(first, second, lag, arm_q, hand_q)
            held = checked_masks(held_masks(m))
            d, k = kin.min_arm_arm_distance(m.arm, m.hand, step=PAIR_CHECK_STEP, with_object=held, stop_below=need)
            if d < need:
                fails.append((lag, d, k, m, held))
                continue
            m.safety_clips = self.dry_run_safety(arm_q, hand_q, m.arm, m.hand)
            if m.safety_clips:
                clipped += 1
                continue
            m.clearance = (d, k)
            return m, None
        why = f"no lag <= {cap / 30:.1f} s with clearance >= {1000 * need:.0f} mm"
        if clipped:
            why += f" ({clipped} clear lags had safety clips)"
        for lag, d, k, m, held in fails[:1] + fails[-1:] if len(fails) > 1 else fails:
            why += f"; lag {lag / 30:.1f} s: {1000 * d:.0f} mm at {self.where(m, k, held, kin)}"
        return None, why

    @staticmethod
    def where(m, k, held=None, kin=None):
        """'tick k (left <phase>/<sphere>, right <phase>/<sphere>)' of a merged plan: what comes closest there."""
        s = f"tick {k} (left {phase_at(m, 'left', k)}, right {phase_at(m, 'right', k)})"
        if kin is not None:
            wo = (False, False) if held is None else (bool(held[0][k]), bool(held[1][k]))
            _, a, b = kin.arm_arm_closest(m.arm[k], m.hand[k], wo)
            s = s[:-1] + f"; {a} / {b})"
        return s

    def plan_pair(self, first, det, bin_name, arm_q, hand_q, variant=0, drop_offset=None, drop_avoid=None,
                  neighbours=None):
        """Second object `det` -> `bin_name` with the arm `first` does not use, from the same start pose, merged with
        `first` (pair()). neighbours: every other object incl. first's (it stands there when both arms start).
        Returns (merged Plan | None, second's single-arm Plan, reason)."""
        side = "right" if first.side == "left" else "left"
        p = self.plan_side(det, side, bin_name, arm_q, hand_q, variant=variant, drop_offset=drop_offset,
                           drop_avoid=drop_avoid, neighbours=neighbours)
        if not p.ok:
            return None, p, p.reason
        m, why = self.pair(first, p, arm_q, hand_q)
        return m, p, why

    # ------------------------------------------------------------------ one arm vs the parked other arm
    def clear_poses(self):
        """two_arm.clear_poses (LEFT arm 7-vectors, the right arm's mirrored: roll / yaw joints flip sign) as (14,)
        vectors [left, right]. Raises if one is outside the IK bounds (ik.limit_margin / ik.joint_margin)."""
        if self._clear is None:
            out = []
            for q in (self.cfg.get("two_arm") or {}).get("clear_poses") or []:
                q = np.asarray(q, float)
                if q.shape != (7,):
                    raise ValueError(f"two_arm.clear_poses: {q.tolist()} is not a 7-vector")
                v = np.r_[q, q * ARM_MIRROR]
                out_of = (v < self.kin.lo - 1e-9) | (v > self.kin.hi + 1e-9)
                bad = [self.spec.arm_names[i] for i in np.flatnonzero(out_of)]
                if bad:
                    raise ValueError(f"two_arm.clear_poses {q.tolist()} outside the IK joint bounds: {bad}")
                out.append(v)
            self._clear = out
        return self._clear

    def parked_clearance(self, plan, kin=None, stop_below=None):
        """(min model distance m, tick) of a single-arm plan vs the other arm, which holds its pose (the plan's own
        columns of it); the plan's hand carries an object sphere from close to open (held_masks), every tick."""
        kin = kin or self.kin
        return kin.min_arm_arm_distance(plan.arm, plan.hand, step=PAIR_CHECK_STEP, stop_below=stop_below,
                                        with_object=checked_masks(held_masks(plan)))

    def keep_clear(self, plan, arm_q, hand_q, kin=None):
        """two_arm.avoid_parked for a single-arm `plan` made from arm_q / hand_q. Returns (Plan | None, info):
        `plan` itself if it keeps two_arm.min_clearance from the parked other arm at every tick, else the evade()
        queue, else None (info['why']). info: 'parked' = (min distance, tick) of the plan alone, 'where' = what comes
        closest there."""
        kin = kin or self.kin
        need = float((self.cfg.get("two_arm") or {}).get("min_clearance", 0.02))
        osl = ArmKinematics.sl("right" if plan.side == "left" else "left")
        if np.abs(plan.arm[:, osl] - np.asarray(arm_q, float)[osl]).max() > 1e-9:
            return None, {"parked": (-np.inf, 0), "where": "", "why": "the other arm is not parked in this plan"}
        d, k = self.parked_clearance(plan, kin, stop_below=need)
        if d < need:  # (conflict: the whole plan, for the log)
            d, k = self.parked_clearance(plan, kin)
        info = {"parked": (d, k), "where": self.where(plan, k, held_masks(plan), kin)}
        if d >= need:
            return plan, info
        m, why = self.evade(plan, arm_q, hand_q, kin)
        info["why"] = why
        return m, info

    def evade(self, plan, arm_q, hand_q, kin=None):
        """The parked arm gets out of the way of a single-arm `plan` (made from arm_q / hand_q): it moves to a
        two_arm.clear_poses pose (tried in order; joint space at the motion limits, its hand as it is), waits and
        moves back to its start pose, merged with `plan` into one queue (merge_plans: the idle arm's part is a plan
        without a grasp). `plan` starts at the first lag 0, two_arm.lag_step_s, ... (<= lag_cap_frac of it) for which
        a return time keeps two_arm.min_clearance at every tick; the return starts as early as that allows, at the
        latest when `plan` has ended. Checked like any plan: the idle arm's moves keep motion.counter_clearance and
        clear the bin rims, the merged queue keeps the clearance at every tick (min_arm_arm_distance, as pair()) and
        has 0 clips in a safety dry run from arm_q / hand_q. Returns (merged Plan: parts [plan, idle part], evade
        info; or None, reason)."""
        tw, kin = self.cfg.get("two_arm") or {}, kin or self.kin
        need = float(tw.get("min_clearance", 0.02))
        step = max(1, int(round(float(tw.get("lag_step_s", 0.5)) / self.dt)))
        cap = int(float(tw.get("lag_cap_frac", 0.6)) * len(plan.arm))
        side = "right" if plan.side == "left" else "left"
        sl, n = ArmKinematics.sl(side), len(plan.arm)
        arm_q, hand_q = np.asarray(arm_q, float), np.asarray(hand_q, float)
        # plan state 0 = its start pose (held while delayed), state k + 1 = plan tick k
        ca, ra = kin.side_spheres(plan.side, np.vstack([arm_q, plan.arm]), np.vstack([hand_q, plan.hand]))
        wa = np.r_[False, held_masks(plan)[0 if plan.side == "left" else 1]]
        why = []
        for i, c in enumerate(self.clear_poses()):
            bad, a1, a2 = self._aside(side, arm_q, hand_q, c[sl], kin)
            if bad:
                why.append(f"pose {i}: {bad}")
                continue
            n1, n2 = len(a1), len(a2)
            Q = np.repeat(arm_q[None], n1 + n2, 0)
            Q[:n1, sl], Q[n1:, sl] = a1, a2
            cb, rb = kin.side_spheres(side, Q, np.repeat(hand_q[None], n1 + n2, 0))
            # ok[plan state, idle state]; idle states: aside ticks (the last = clear pose), back ticks (last = start)
            ok = kin.sphere_distance_matrix(ca, ra, cb, rb, wa=wa, below=need) >= need
            found, first = None, None
            for lag in range(0, cap + 1, step):
                Ta = lag + n                                  # merged tick at which `plan` has ended
                t = np.arange(Ta + n2)
                ai = np.where(t < lag, 0, 1 + np.minimum(t - lag, n - 1))
                f1 = np.flatnonzero(~ok[ai, np.minimum(t, n1 - 1)])  # idle arm aside / at the clear pose
                r_hi = min(Ta, int(f1[0]) if len(f1) else Ta)         # the return must start by then
                first = first or (int(f1[0]) if len(f1) else None, n1)
                if r_hi < n1:
                    continue
                R = np.arange(n1, r_hi + 1)
                back = ok[ai[R[:, None] + np.arange(n2)], n1 + np.arange(n2)].all(1)
                fe = ok[ai[:Ta], n1 + n2 - 1]                         # idle arm back at its start pose
                tail = np.r_[np.flip(np.cumprod(np.flip(fe))).astype(bool), True]
                good = np.flatnonzero(back & tail[np.minimum(R + n2, Ta)])
                if len(good):
                    found = (lag, int(R[good[0]]))
                    break
            if found is None:
                k1 = first[0]
                aside = k1 is not None and k1 < n1
                why.append(f"pose {i}: no lag <= {cap / 30:.1f} s with a return time"
                           + (f" (lag 0: conflict at tick {k1} while moving aside)" if aside else ""))
                continue
            lag, R = found
            e = Plan(True, side=side, reason="evade", phases=[("aside", 0, n1)] + ([("clear", n1, R)] if R > n1 else [])
                     + [("back", R, R + n2)], evade={"pose": i})
            e.arm, e.hand = np.repeat(arm_q[None], R + n2, 0), np.repeat(hand_q[None], R + n2, 0)
            e.arm[:n1, sl], e.arm[n1:R, sl], e.arm[R:, sl] = a1, c[sl], a2
            if plan.nb is not None and len(plan.nb):  # idle arm vs the objects (+ the one being picked, its start)
                g = plan.waypoints.get("grasp", np.full(3, np.inf))
                cyl = np.vstack([plan.nb, [g[0], g[1], plan.nb[:, 2].max(), plan.nb[:, 3].max()]])
                mv = np.r_[np.arange(n1), np.arange(R, R + n2)]              # moving ticks (aside, back)
                d0, D = self.neighbour_clearance(side, arm_q, hand_q, e.arm[mv], e.hand[mv], cyl, kin)
                if (D < self._nb_need(d0, least=False)).any():
                    why.append(f"pose {i}: idle arm {1000 * D.min():.0f} mm from an object")
                    continue
            m = merge_plans(e, plan, lag, arm_q, hand_q)
            m.parts, m.color, m.variant, m.hint = [plan, e], plan.color, plan.variant, plan.hint
            held = checked_masks(held_masks(m))
            d, k = kin.min_arm_arm_distance(m.arm, m.hand, step=PAIR_CHECK_STEP, with_object=held, stop_below=need)
            if d < need:  # (the tick matrix above is exact below `need`: should not happen)
                why.append(f"pose {i}: merged queue {1000 * d:.0f} mm at {self.where(m, k, held, kin)}")
                continue
            m.safety_clips = self.dry_run_safety(arm_q, hand_q, m.arm, m.hand)
            if m.safety_clips:
                why.append(f"pose {i}: {m.safety_clips} safety clips in dry run")
                continue
            m.clearance = (d, k)
            m.evade = {"side": side, "pose": i, "lag": lag, "back": R, "aside_ticks": n1, "back_ticks": n2,
                       "added": len(m.arm) - n}
            return m, None
        return None, "; ".join(why) or "no two_arm.clear_poses"

    def _aside(self, side, arm_q, hand_q, q7, kin):
        """(reason | None, move to q7, move back) of `side`'s arm from arm_q (joint space, motion limits, hand_q
        held): every tick's hand points keep motion.counter_clearance over the counter and clear the bin rims."""
        sl = ArmKinematics.sl(side)
        key = (side, arm_q[sl].tobytes(), hand_q[sl].tobytes(), np.asarray(q7, float).tobytes())
        if key not in self._asides:
            mo = self.cfg["motion"]
            a1 = time_scale([arm_q[sl], q7], mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
            a2 = time_scale([q7, arm_q[sl]], mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
            bad = None if len(a1) else "already there"
            for k, q in enumerate(np.vstack([a1, a2]) if len(a1) else ()):
                full = arm_q.copy()
                full[sl] = q
                pts = kin.hand_points(side, full, hand_q)
                c, b = self._counter_clear(pts), self._bin_clear(pts)
                if c < mo["counter_clearance"] - 2e-3 or b < 0.003:
                    what = "moving aside" if k < len(a1) else "moving back"
                    bad = (f"hand {100 * c:.1f} cm above counter while {what}" if c < mo["counter_clearance"] - 2e-3
                           else f"hand {100 * b:.1f} cm above a bin rim while {what}")
                    break
            self._asides[key] = (bad, a1, a2)
        return self._asides[key]

    def relocation_spots(self, det, bin_name, obstacles=()):
        """Candidate counter spots (pelvis xy) for a relocation, nearest the target bin first."""
        rc = self.cfg["grasp"].get("relocate", {})
        sc = self.scene
        L = sc.cfg["layout"]
        m = float(rc.get("region_margin", 0.02))
        xs = np.linspace(L["region_x"][0] - m, L["region_x"][1] + m, 9)
        ys = np.linspace(L["region_y"][0] - m, L["region_y"][1] + m, 4)
        pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in xs for y in ys]))[:, :2]
        clear = float(rc.get("clearance", L["min_spacing"]))
        obs = np.asarray(list(obstacles), float).reshape(-1, 2)
        keep = [p for p in pts if np.linalg.norm(p - det.top[:2]) > 0.06
                and (len(obs) == 0 or np.linalg.norm(obs - p, axis=1).min() > clear)]
        b = self.bin_xy(bin_name)
        keep.sort(key=lambda p: np.linalg.norm(p - b))
        return keep[: int(rc.get("max_spots", 6))]

    def plan_relocation(self, det, bin_name, arm_q, hand_q, obstacles=(), variant=0):
        """Cross-body fallback: pick `det` with an arm that reaches it and put it down at a counter spot from which
        some arm has a full plan to `bin_name` (checked from home). Returns a Plan with target_bin 'counter'.
        variant: finger hint rotation of the pick (as plan())."""
        from dataclasses import replace
        last = Plan(False, color=det.color, reason="no relocation spot")
        H = self.height
        for spot in self.relocation_spots(det, bin_name, obstacles):
            d2 = replace(det, top=np.r_[spot, self.scene.counter_z + H])
            by = self.bin_xy(bin_name)[1]
            fin_side = "left" if (by if abs(by) > 0.1 else spot[1]) > 0 else "right"
            fin = self._plan_side_hints(d2, fin_side, bin_name, self.home.copy(), self.post["open"].copy())
            if not fin.ok:
                last.reason = f"no {fin_side} plan from spot ({spot[0]:+.2f},{spot[1]:+.2f}): {fin.reason}"
                continue
            side = "right" if fin_side == "left" else "left"
            p = self._plan_side_hints(det, side, bin_name, arm_q, hand_q, place_xy=spot, variant=variant)
            if p.ok:
                p.reason = f"relocate for the {fin_side} arm"
                return p
            last = p
            last.reason = f"{side} cannot place at ({spot[0]:+.2f},{spot[1]:+.2f}): {p.reason}"
        return last

    def _reaches_bin(self, side, bin_name):
        """Cheap reach test (cached per planner): can this arm put its grasp point above the bin at carry height at
        all? A cross-body attempt otherwise spends 10-30 s in the joint-space carry search before it fails."""
        key = (side, bin_name)
        if key not in self._reach:
            g, sc = self.cfg["grasp"], self.scene
            z = sc.counter_z + max(g["carry_height"], float(sc.cfg["bins"]["wall_height"]) + self.height + 0.03)
            xy = self.drop_xy(bin_name, offset=False)
            self._reach[key] = bool(self.kin.solve(side, np.r_[xy, z], self.home)[1])
        return self._reach[key]

    def _reach_reject(self, det, side, bin_name, place_xy=None):
        """Failed Plan if grasp.reach_check is on and `side` cannot reach bin `bin_name` (not for a counter place)."""
        if (self.cfg["grasp"].get("reach_check") and place_xy is None and bin_name in self.scene.cfg["bins"]["centers"]
                and not self._reaches_bin(side, bin_name)):
            return Plan(False, side=side, color=det.color, reason=f"{side} arm cannot reach the {bin_name} bin")
        return None

    def face_dirs(self, det, side, variant=0):
        """grasp.face_align finger directions (rad, pelvis yaw) for `det`, in the order tried: cube = outward face
        normals (yaw + k * 90 deg), triangle = triangle_phi_deg from an apex (apexes at yaw + 90 + k * 120 deg, the
        Detection / oracle yaw convention); only those within max_dev_deg of the arm's natural finger direction
        (finger_ref_deg), nearest first, rotated by `variant`. [] if off for this shape, no shape / yaw, or
        det.shape_conf < min_conf (camera: a weak fit gets the stock hints)."""
        fa = self.cfg["grasp"].get("face_align") or {}
        shape, yaw = getattr(det, "shape", None), getattr(det, "yaw", None)
        if (not fa.get("enabled") or yaw is None or shape not in ("cube", "triangle") or not fa.get(shape, True)
                or float(getattr(det, "shape_conf", 0.0)) < float(fa.get("min_conf", 0.0))):
            return []
        off, n = (0.0, 4) if shape == "cube" else (np.pi / 2 + np.radians(float(fa.get("triangle_phi_deg", 105))), 3)
        ref = np.radians(float((fa.get("finger_ref_deg") or {}).get(side, -25.0 if side == "left" else 25.0)))
        cands = []
        for k in range(n):
            a = yaw + off + 2 * np.pi * k / n
            da = abs((a - ref + np.pi) % (2 * np.pi) - np.pi)
            if da <= np.radians(float(fa.get("max_dev_deg", 75.0))):
                cands.append((da, a))
        cands = [a for _, a in sorted(cands)]
        k = int(variant) % len(cands) if cands else 0
        return cands[k:] + cands[:k]

    def _candidates(self, det, side, variant=0):
        """Grasp candidates for one arm, in the order tried: [(finger direction (3,), IK hint weight, detection to
        plan (triangle: shifted to the pinch midpoint), tolerance deg (None: no check), hint index, face deg)].
        Order: face-aligned directions (face_dirs), grasp.finger_hints rotated by `variant` (a retry after a
        verified miss grasps another way; the right arm mirrors the hints), then (neighbours.extra_dirs, nearest
        neighbour within extra_within) the two pinch axes perpendicular to the nearest neighbour (natural direction
        first) and the fingers towards it, then grasp.extra_hints. Least-bad pass: those neighbour directions first.
        kind: 'face' | 'hint' | 'perpendicular' | 'towards'."""
        g, fa = self.cfg["grasp"], self.cfg["grasp"].get("face_align") or {}
        nc = self.cfg.get("neighbours") or {}
        w_face, tol = float(fa.get("hint_weight", 3.0)), float(fa.get("tol_deg", 12.0))
        out, near = [], []
        if self._nb is not None and len(self._nb):
            dist = np.linalg.norm(self._nb[:, :2] - det.top[:2], axis=1)
            o = self._nb[np.argmin(dist), :2] - det.top[:2]
            ref = np.radians(float((fa.get("finger_ref_deg") or {}).get(side, -25.0 if side == "left" else 25.0)))
            perp = [np.arctan2(o[1], o[0]) + s * np.pi / 2 for s in (1, -1)]
            perp.sort(key=lambda a: abs((a - ref + np.pi) % (2 * np.pi) - np.pi))
            near = [dict(f=np.array([np.cos(a), np.sin(a), 0.0]), w=w_face, det=det, tol=tol, hint=None,
                         deg=float(np.degrees(a)), kind="perpendicular") for a in perp]
            a = float(np.arctan2(o[1], o[0]))    # fingers towards it: the thumb (5-8 cm behind the grasp) away
            near.append(dict(f=np.array([np.cos(a), np.sin(a), 0.0]), w=w_face, det=det, tol=tol, hint=None,
                             deg=float(np.degrees(a)), kind="towards"))
            if not self._least and not (nc.get("extra_dirs") and dist.min() <= float(nc.get("extra_within", 0.12))):
                near = []
        if self._least:
            out, near = near, []
        r_tri = self.scene.object_dims("triangle")["circumradius"]
        for a in self.face_dirs(det, side, variant):
            f = np.array([np.cos(a), np.sin(a), 0.0])
            d2 = det
            if det.shape == "triangle":          # pinch midpoint along f is not the centroid (support function)
                d2 = copy.copy(det)
                d2.top = det.top + tri_shift(a - (det.yaw + np.pi / 2), r_tri) * f
            out.append(dict(f=f, w=w_face, det=d2, tol=tol, hint=None, deg=float(np.degrees(a)), kind="face"))
        hints = g.get("finger_hints") or [g.get("finger_hint", [1.0, 0.0, 0.0])]
        extra = g.get("extra_hints") or []     # last resort, never rotated: index len(finger_hints) + k
        for i in np.roll(np.arange(len(hints)), -(int(variant) % len(hints))):
            h = np.asarray(hints[i], float) * ([1.0, -1.0, 1.0] if side == "right" else 1.0)
            out.append(dict(f=h / np.linalg.norm(h), w=None, det=det, tol=None, hint=int(i), deg=None, kind="hint"))
        out += near                            # neighbours.extra_dirs: only reached when everything above failed
        for k, h in enumerate(extra):
            h = np.asarray(h, float) * ([1.0, -1.0, 1.0] if side == "right" else 1.0)
            out.append(dict(f=h / np.linalg.norm(h), w=None, det=det, tol=None, hint=len(hints) + k, deg=None,
                            kind="hint"))
        return out

    def _plan_side_hints(self, det, side, bin_name, arm_q, hand_q, variant=0, **kw):
        """_plan_side with each grasp candidate (_candidates) in turn: face-aligned directions (accepted only if the
        IK reaches the direction within face_align.tol_deg), then the finger hints (fingers forward can reach over
        the centre bin's wall when the object is close to it; then fingers inward/sideways). First plan that passes
        wins; least-bad pass: all are planned, the most neighbour clearance wins (1 mm ties: candidate order).
        Failure: the first one that a neighbour blocked (.blocked), else the first finger-hint failure."""
        rej = self._reach_reject(det, side, bin_name, kw.get("place_xy"))
        if rej is not None:
            return rej
        saved = (self.kin.hint, self.kin.w_hint)
        first = blocked = best = p = None
        try:
            for c in self._candidates(det, side, variant):
                self.kin.hint, self.kin.w_hint = c["f"], saved[1] if c["w"] is None else c["w"]
                p = self._plan_side(c["det"], side, bin_name, arm_q, hand_q, **kw)
                p.hint, p.face = c["hint"], c["deg"] if c["kind"] == "face" else None
                if p.ok and c["tol"] is not None:
                    err = self._finger_err(p, side, c["f"])
                    if err > c["tol"]:
                        p.ok, p.reason = False, f"finger direction {err:.0f} deg from {c['deg']:.0f} deg"
                    else:
                        p.reason = {"face": "face-aligned", "towards": "fingers towards the nearest neighbour",
                                    "perpendicular": "pinch axis perpendicular to the nearest neighbour"}[c["kind"]]
                        p.reason += f" ({c['deg']:.0f} deg, err {err:.0f})"
                hint = c["hint"]
                if p.ok:
                    if self._least:
                        p.least_bad = True
                    if not self._least or self._good_enough(p):
                        return p
                    if best is None or _slack(p) > _slack(best) + 1e-3:
                        best = p
                    continue
                blocked = blocked or (p if p.blocked else None)
                first = first or (p if hint is not None else None)
            if best is not None:
                best.least_bad = True
                return best
            out = blocked or first or p
            out.blocked = blocked is not None
            return out
        finally:
            self.kin.hint, self.kin.w_hint = saved

    def _finger_err(self, plan, side, f):
        """Angle (deg) between the palm x axis (finger direction) at the grasp and f, in the horizontal plane."""
        e = next(e for name, s, e in plan.phases if name == "descend")
        self.kin._fk(plan.arm[e - 1])
        x = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0]
        a = np.arctan2(x[1], x[0]) - np.arctan2(f[1], f[0])
        return float(np.degrees(abs((a + np.pi) % (2 * np.pi) - np.pi)))

    def _plan_side(self, det, side, bin_name, arm_q, hand_q, place_xy=None):
        cfg, g, mo = self.cfg, self.cfg["grasp"], self.cfg["motion"]
        sc, zc = self.scene, self.scene.counter_z
        plan = Plan(False, side=side, color=det.color, target_bin=bin_name if place_xy is None else "counter")
        sl = ArmKinematics.sl(side)
        H, post = self.height, self.post
        hand_pre = hand_q.copy(); hand_pre[sl] = post["preshape"][sl]
        hand_cls = hand_q.copy(); hand_cls[sl] = post["closed"][sl]
        hand_open = hand_q.copy(); hand_open[sl] = post["open"][sl]
        hand_rel = hand_q.copy(); hand_rel[sl] = post[g.get("release_posture", "preshape")][sl]

        # grasp height: object mid-height, raised until the preshaped fingers clear the counter
        xy = det.top[:2] + np.asarray((g.get("arm_xy_offset") or {}).get(side, [0.0, 0.0]), float)  # per-arm, TUNE
        z = zc + g["height_frac"] * H
        for _ in range(4):
            qg, ok = self._ik(side, np.r_[xy, z], arm_q, "grasp", plan)
            if not ok:
                plan.reason = f"IK grasp {side}"
                return plan
            c = self.clearance(side, qg, hand_pre, fingers_only=True)
            if c >= g["fingertip_clearance"] - 1e-3:
                break
            z += g["fingertip_clearance"] - c
        else:
            plan.reason = "fingers hit the counter"
            return plan
        if z > zc + H - 0.008:
            plan.reason = f"grasp point {100 * (z - zc):.1f} cm above counter: above the object top (point_in_palm?)"
            return plan
        if self._nb is not None and len(self._nb):  # cheap early reject: the grasp pose, preshaped / closing / closed
            hand_mid = 0.5 * (hand_pre + hand_cls)
            d0, D = self.neighbour_clearance(side, arm_q, hand_q, np.stack([qg, qg, qg]),
                                             np.stack([hand_pre, hand_mid, hand_cls]))
            need = self._nb_need(d0)
            if (D < need).any():
                k, j = np.unravel_index(int(np.argmin(D - need)), D.shape)
                plan.blocked = True
                plan.reason = (f"hand {1000 * D[k, j]:.0f} mm from {self._nb_tags[j]} at the grasp "
                               f"({('preshaped', 'closing', 'closed')[k]}; need {1000 * need[j]:.0f})")
                return plan

        wall = float(sc.cfg["bins"]["wall_height"])
        z_carry = zc + max(g["carry_height"], wall + H + 0.03)
        z_drop = zc + wall + H / 2 + g["drop_above_rim"]
        if place_xy is not None:  # relocation: put it down on the counter (release just above the grasp height)
            bxy, z_drop = np.asarray(place_xy, float), z + 0.005
        else:
            bxy = self._drop_point(bin_name, side)
        P = {"pregrasp": np.r_[xy, max(z + g["approach_height"], z_carry)], "grasp": np.r_[xy, z], "lift": np.r_[xy, z_carry],
             "above_bin": np.r_[bxy, z_carry], "drop": np.r_[bxy, z_drop], "retreat": np.r_[bxy, z_drop + g["retreat_up"]]}

        q_pre, ok = self._ik(side, P["pregrasp"], qg, "pregrasp", plan)  # same IK branch as the grasp
        if not ok:
            plan.reason = f"IK pregrasp {side}"
            return plan
        plan.waypoints["grasp"] = np.r_[xy, z]
        down = self._cartesian(side, P["pregrasp"], P["grasp"], q_pre, "descend", plan)
        up = down and self._cartesian(side, P["grasp"], P["lift"], down[-1], "lift", plan)
        # grasp.face_align.carry_weight: the finger direction a face / neighbour candidate forces (hint weight 3) only
        # matters at the object. Kept to the bin, the IK there must point the fingers the same way: a quarter of
        # the triangle poses lost their aligned grasp to "carry: no joint-space path" (2026-11-14). From the lift
        # on, the direction is a weak preference again (the hand turns on the way).
        w_grasp = self.kin.w_hint
        w_carry = (g.get("face_align") or {}).get("carry_weight")
        if w_carry is not None:
            self.kin.w_hint = min(w_grasp, float(w_carry))
        try:
            carry = up and self._carry(side, up[-1], P["lift"], P["above_bin"], z_drop, plan)
            lower = carry and self._cartesian(side, P["above_bin"], P["drop"], carry[-1], "lower", plan)
            retreat = lower and self._cartesian(side, P["drop"], P["retreat"], lower[-1], "retreat", plan)
        finally:
            self.kin.w_hint = w_grasp
        if not retreat:
            plan.reason = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
            return plan
        plan.waypoints["drop"] = P["drop"]

        vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
        home = arm_q.copy(); home[sl] = self.home[sl]
        arm_parts, hand_parts, phases = [], [], []
        cur_a, cur_h = arm_q.copy(), hand_q.copy()

        def move(name, Qs, hand_to=None):
            nonlocal cur_a, cur_h
            a = time_scale([cur_a] + list(Qs), vmax, amax, self.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            if hand_to is not None:  # hand moves during the arm move (ends no later than the arm)
                hr = ramp(cur_h, hand_to, hv, self.dt)
                if len(hr) > len(a):
                    a = np.vstack([a, np.repeat((a[-1] if len(a) else cur_a)[None], len(hr) - len(a), 0)])
                    h = np.repeat(cur_h[None], len(a), 0)
                h[:len(hr)] = hr
                h[len(hr):] = hand_to
                cur_h = np.asarray(hand_to, float).copy()
            add(name, a, h)

        def hand(name, to, hold_s):
            nonlocal cur_h
            h = ramp(cur_h, to, hv, self.dt)
            h = np.vstack([h, np.repeat(np.asarray(to, float)[None], int(round(hold_s / self.dt)), 0)])
            cur_h = np.asarray(to, float).copy()
            add(name, np.repeat(cur_a[None], len(h), 0), h)

        def add(name, a, h):
            nonlocal cur_a
            if len(a):
                start = sum(len(x) for x in arm_parts)
                arm_parts.append(a); hand_parts.append(h)
                phases.append((name, start, start + len(a)))
                cur_a = a[-1].copy()

        move("to_pregrasp", [q_pre], hand_to=hand_pre)
        move("descend", down)
        cc = g.get("correct") or {}
        if cc.get("enabled"):  # settle and correct: dwell at the grasp pose, hand preshaped (policy._correct)
            n = sum(correct_ticks(cc, self.dt))
            add("settle", np.repeat(cur_a[None], n, 0), np.repeat(cur_h[None], n, 0))
        hand("close", hand_cls, mo["close_hold_s"])
        move("lift", up)
        move("carry", carry + lower)
        hand("open", hand_rel, mo["open_hold_s"])  # widest posture: `open` alone left the cube wedged in the sim
        move("retreat", retreat)
        move("home", [home], hand_to=hand_open)
        plan.arm, plan.hand, plan.phases = np.vstack(arm_parts), np.vstack(hand_parts), phases

        # clearance over the whole trajectory (grasp phases use the fingertip margin)
        need, rim = self.tick_rules(phases, len(plan.arm), place_xy is not None)
        for k in range(0, len(plan.arm), 3):
            c = self.clearance(side, plan.arm[k], plan.hand[k])
            if c < need[k] - 2e-3:
                name = next(n for n, s, e in phases if s <= k < e)
                # the object itself sits between the fingers during descend/lift; only the counter matters here
                plan.reason = f"hand {100 * c:.1f} cm above counter during {name} (need {100 * need[k]:.1f})"
                return plan
        for k in range(0, len(plan.arm), 3):
            if not rim[k]:
                continue
            c = self.bin_clearance(side, plan.arm[k], plan.hand[k])
            if c < 0.003:
                name = next(n for n, s_, e in phases if s_ <= k < e)
                plan.reason = f"hand {100 * c:.1f} cm above a bin rim during {name}"
                return plan
        if not self._check_neighbours(side, arm_q, hand_q, plan):   # every tick vs the other objects
            return plan
        plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
        if plan.safety_clips:
            plan.reason = f"{plan.safety_clips} safety clips in dry run"
            return plan
        plan.ok = True
        return plan

    def dry_run_safety(self, arm_q, hand_q, arm, hand):
        if self._dry is None:
            self._dry = SafetyFilter(self.spec, self.safety)
        f = self._dry
        f.reset(arm_q, hand_q, 0.0)  # (reset clears every bit of episode state)
        for k in range(len(arm)):
            f.filter({"arm_q_target": arm[k], "hand_q_target": hand[k]}, (k + 1) * self.dt)
        return f.clipped_steps

    def abort(self, side, arm_q, hand_q, rest_arm, rest_hand, neighbours=None):
        """Abort `side`'s part of an executing queue (verified miss / slip). arm_q, hand_q (14,): the last commanded
        tick; rest_arm, rest_hand (R, 14): the ticks still queued after it. That arm decelerates along its queued
        path (time warp: the path speed ramps down to 0 at about motion.arm_max_acc; no stop in one tick), then moves
        in joint space to its home pose with its hand going to `open` (arm_max_vel / arm_max_acc / hand_max_vel).
        The other arm's columns keep their queued ticks. Checked like a plan, every tick: hand points above the
        counter (grasp.fingertip_clearance while still on the planned path, then motion.counter_clearance), above
        the bin rims, and 0 clips in a safety dry run from arm_q / hand_q. neighbours: the plan's neighbour
        cylinders (Plan.nb): the move home keeps neighbours.margin from them (the decelerated part is on the
        checked path).
        Returns (arm, hand, info dict) = the new queue, or (None, None, reason) if a check fails."""
        mo, g, dt = self.cfg["motion"], self.cfg["grasp"], self.dt
        vmax, amax = mo["arm_max_vel"], mo["arm_max_acc"]
        sl = ArmKinematics.sl(side)
        rest_arm, rest_hand = np.asarray(rest_arm, float), np.asarray(rest_hand, float)
        R = len(rest_arm)
        if R == 0:
            return None, None, "nothing queued"
        P = np.vstack([np.asarray(arm_q, float)[None], rest_arm])[:, sl]       # path ticks 0..R (0 = commanded now)
        Ph = np.vstack([np.asarray(hand_q, float)[None], rest_hand])[:, sl]
        v = np.abs(np.diff(P[: int(np.ceil(vmax / amax / dt)) + 2], axis=0)).max() / dt  # fastest planned speed ahead
        n = int(np.ceil(v / (amax * dt) - 1e-9))
        # plan ticks advanced per tick: 1 -> 0 over n ticks (the last one at rest: the turn towards home starts at 0)
        tau = np.minimum(np.cumsum(1.0 - np.arange(1, n + 1) / n), R) if n > 1 else np.empty(0)
        i = np.floor(tau).astype(int)
        f, j = (tau - i)[:, None], np.minimum(i + 1, R)
        dec_a, dec_h = P[i] + f * (P[j] - P[i]), Ph[i] + f * (Ph[j] - Ph[i])
        q0, h0 = (dec_a[-1], dec_h[-1]) if len(dec_a) else (P[0], Ph[0])
        a = time_scale([q0, self.home[sl]], vmax, amax, dt)
        h = ramp(h0, self.post["open"][sl], mo["hand_max_vel"], dt)
        m = max(len(a), len(h))
        a = np.vstack([a, np.repeat(self.home[sl][None], m - len(a), 0)])
        h = np.vstack([h, np.repeat(self.post["open"][sl][None], m - len(h), 0)])
        side_a, side_h = np.vstack([dec_a, a]), np.vstack([dec_h, h])
        if not len(side_a):
            return None, None, "already at home"
        # other arm: its queued ticks until they stop changing (today: constant -> the queue ends with this arm)
        oth = np.ones(rest_arm.shape[1], bool)
        oth[sl] = False
        moving = np.flatnonzero((np.abs(rest_arm[:, oth] - rest_arm[-1, oth]).max(1) > 0)
                                | (np.abs(rest_hand[:, oth] - rest_hand[-1, oth]).max(1) > 0))
        L = max(len(side_a), min(int(moving[-1]) + 2, R) if len(moving) else 0)
        arm, hand = rest_arm[np.minimum(np.arange(L), R - 1)], rest_hand[np.minimum(np.arange(L), R - 1)]
        ks = np.minimum(np.arange(L), len(side_a) - 1)
        arm[:, sl], hand[:, sl] = side_a[ks], side_h[ks]
        for k in range(len(side_a)):
            pts = self.kin.hand_points(side, arm[k], hand[k])
            need = g["fingertip_clearance"] if k < len(dec_a) else mo["counter_clearance"]
            c, b = self._counter_clear(pts), self._bin_clear(pts)
            if c < need - 2e-3 or b < 0.003:
                what = "decelerating" if k < len(dec_a) else "moving home"
                return None, None, (f"hand {100 * c:.1f} cm above counter while {what} (need {100 * need:.1f})"
                                    if c < need - 2e-3 else f"hand {100 * b:.1f} cm above a bin rim while {what}")
        if neighbours is not None and len(neighbours) and len(side_a) > len(dec_a):
            k0 = len(dec_a)
            d0, D = self.neighbour_clearance(side, arm[k0 - 1] if k0 else arm_q, hand[k0 - 1] if k0 else hand_q,
                                             arm[k0:len(side_a)], hand[k0:len(side_a)], neighbours)
            if (D < self._nb_need(d0, least=False)).any():
                return None, None, f"hand {1000 * D.min():.0f} mm from an object while moving home"
        clips = self.dry_run_safety(arm_q, hand_q, arm, hand)
        if clips:
            return None, None, f"{clips} safety clips in dry run"
        return arm, hand, {"decel_ticks": len(dec_a), "home_ticks": m, "ticks": L, "saved_ticks": R - L}

    # ------------------------------------------------------------------ settle and correct (grasp.correct)
    def correction_dq(self, side, arm_cmd, arm_meas):
        """Tracking error of `side`'s grasp point and the joint shift that removes it. e = FK(commanded) -
        FK(measured) (pelvis, m). dq (7,): damped least squares (2 Gauss-Newton steps, grasp.correct.damping) on the
        grasp point and the palm orientation (weight rot_weight m/rad) such that the measured joints + dq give the
        COMMANDED grasp point and palm orientation: added to the commanded ticks it puts the hand where the plan wants
        it while the joint error stays as it is (sim: the sag changes by < 1 mrad over such a shift). The sag also
        tilts the palm (~1.7 deg): restoring the commanded orientation needs ~3x less joint motion than keeping the
        measured one (sim replay: 15 vs 41 mrad max per grasp). The commanded pose + dq stays inside the IK joint
        bounds: a joint that would leave them is pinned at its bound and the others solved again (the arm's 7th
        degree of freedom and the orientation weight absorb it). Returns (e (3,), dq (7,), finger direction (3,)
        horizontal, of the commanded palm, predicted grasp-point residual m)."""
        cc, kin, sl = self.cfg["grasp"]["correct"], self.kin, ArmKinematics.sl(side)
        w, lam2 = float(cc.get("rot_weight", 0.3)), float(cc.get("damping", 0.01)) ** 2
        qc = np.asarray(arm_cmd, float)
        pc, _, _, Rc = kin.grasp_jacobian(side, qc)
        x = np.asarray(arm_meas, float).copy()
        q0, e = x[sl].copy(), pc - kin.grasp_pose(side, x)[0]
        lo, hi = kin.lo[sl] - qc[sl], kin.hi[sl] - qc[sl]
        dq, free = np.zeros(7), np.ones(7, bool)
        for _ in range(4):   # Gauss-Newton (2 steps converge); later steps only after a joint was pinned
            x[sl] = q0 + dq
            p, Jp, Jw, R = kin.grasp_jacobian(side, x)
            J = np.vstack([Jp, w * Jw])[:, free]
            r = np.r_[pc - p, w * (R @ kin.pin.log3(R.T @ Rc))]
            dq[free] += J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), r)
            out = (dq < lo) | (dq > hi)
            dq = np.clip(dq, lo, hi)
            free &= ~out
            if not out.any() and np.linalg.norm(r[:3]) < 1e-5:
                break
        x[sl] = q0 + dq
        f = Rc[:, 0] * [1.0, 1.0, 0.0]
        return e, dq, f / max(np.linalg.norm(f), 1e-9), float(np.linalg.norm(pc - kin.grasp_pose(side, x)[0]))

    def correct(self, side, arm_q, hand_q, rest_arm, rest_hand, dq, phases, place=False, nb=None, held=None,
                pair_need=None):
        """Shift `side`'s columns of an executing queue by w_k * dq (the rest untouched). arm_q, hand_q (14,): the last
        commanded tick; rest_arm, rest_hand (R, 14): the ticks queued after it; phases: that arm's [(name, start,
        end)] in rest ticks, from its `settle` on. Weights: half-cosine 0 -> 1 from rest tick 0 to the start of `close`,
        1 through the close and grasp.correct.hold_lift_s into the lift, then back to 0 over >= blend_out_s, more slowly
        where a joint would exceed motion.arm_max_vel (|planned step| + |weight step| * |dq| per joint); it must be 0
        before the release (`open`) ends. Refused unless, on every shifted tick: max |dq| <= max_dq; no joint faster
        than motion.arm_max_vel; inside the IK joint bounds (settle / close; lift ticks beyond them are clamped onto
        them); tick_rules(phases, place) (counter / bin rims; only ticks
        the shift makes worse than planned count); the neighbour cylinders nb (Plan.nb) >= neighbours.margin (or the
        planned distance, if less); the arm-arm sphere model vs the other arm's queued ticks >= pair_need (if given;
        held: (left, right) (R,) object masks); 0 clips in a safety dry run from arm_q / hand_q through the shifted
        ticks + 1. Returns (new rest_arm, info) or (None, reason)."""
        cc, mo, kin = self.cfg["grasp"]["correct"], self.cfg["motion"], self.kin
        sl, dt = ArmKinematics.sl(side), self.dt
        rest_arm, rest_hand, dq = np.asarray(rest_arm, float), np.asarray(rest_hand, float), np.asarray(dq, float)
        R, ph = len(rest_arm), {n: (s, e) for n, s, e in phases}
        if "close" not in ph or "lift" not in ph:
            return None, "no close / lift queued"
        c0, l0 = ph["close"][0], ph["lift"][0]
        if c0 < 1:
            return None, "the close has started"
        if np.abs(dq).max() > float(cc.get("max_dq", 0.05)):
            j = int(np.argmax(np.abs(dq)))
            return None, f"|dq| {np.abs(dq).max():.3f} rad on {self.spec.arm_names[sl][j]} > max_dq"
        vstep, ad = float(mo["arm_max_vel"]) * dt, np.abs(dq)
        step = np.abs(np.diff(np.vstack([np.asarray(arm_q, float)[None, sl], rest_arm[:, sl]]), axis=0))  # planned
        w = np.zeros(R)
        w[:c0] = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, c0 + 1) / c0)
        k = min(l0 + int(round(float(cc.get("hold_lift_s", 0.1)) / dt)), R)
        w[c0:k] = 1.0
        k1, n_min, wk = k, max(1, int(round(float(cc.get("blend_out_s", 0.2)) / dt))), 1.0
        end = min(ph["open"][1] if "open" in ph else R, R)
        while wk > 0 and k < end:
            room = np.maximum(vstep - step[k], 0.0)[ad > 1e-12] / ad[ad > 1e-12]
            target = 0.5 + 0.5 * np.cos(np.pi * min(1.0, (k - k1 + 1) / n_min))
            wk = max(0.0 if target < 1e-9 else target, wk - (room.min() if len(room) else np.inf))
            w[k], k = wk, k + 1
        if wk > 0:
            return None, "no room to blend out before the release (speed limit)"
        n = k
        if (step[:n] + np.abs(np.diff(np.r_[0.0, w[:n]]))[:, None] * ad > np.maximum(step[:n], vstep) + 1e-9).any():
            return None, f"blend-in faster than motion.arm_max_vel ({c0} ticks for {ad.max():.3f} rad)"
        A = rest_arm[:n].copy()
        A[:, sl] += w[:n, None] * dq
        bad = (A[:, sl] < kin.lo[sl] - 1e-9) | (A[:, sl] > kin.hi[sl] + 1e-9)
        if bad[:l0].any():   # (correction_dq keeps the grasp pose inside: a lift tick nearer a bound is clamped)
            return None, f"{self.spec.arm_names[sl][int(np.flatnonzero(bad.any(0))[0])]} leaves the IK joint bounds"
        A[:, sl] = np.clip(A[:, sl], kin.lo[sl], kin.hi[sl])   # (clamping never makes a step longer)
        need, rim = self.tick_rules(phases, R, place)
        name = {k_: n_ for n_, s, e in phases for k_ in range(max(s, 0), min(e, R))}
        c, b = self._clear_many(np.array([kin.hand_points(side, A[k], rest_hand[k]) for k in range(n)]), rim[:n])
        for k in np.flatnonzero((c < need[:n] - 2e-3) | (b < 0.003)):
            p0 = kin.hand_points(side, rest_arm[k], rest_hand[k])              # planned: was it already there?
            if c[k] < need[k] - 2e-3 and c[k] < self._counter_clear(p0) - 1e-4:
                return None, f"hand {100 * c[k]:.1f} cm above the counter during {name.get(k, '?')}"
            if b[k] < 0.003 and b[k] < self._bin_clear(p0) - 1e-4:
                return None, f"hand {100 * b[k]:.1f} cm above a bin rim during {name.get(k, '?')}"
        if nb is not None and len(nb):
            D = self.neighbour_clearance(side, arm_q, hand_q, A, rest_hand[:n], nb, kin)[1]
            m = float((self.cfg.get("neighbours") or {}).get("margin", 0.01))
            if (D < m).any():
                Dp = self.neighbour_clearance(side, arm_q, hand_q, rest_arm[:n], rest_hand[:n], nb, kin)[1]
                worse = (D < m) & (D < Dp - 1e-4)
                if worse.any():
                    k, j = np.unravel_index(int(np.argmin(np.where(worse, D, np.inf))), D.shape)
                    return None, (f"hand {1000 * D[k, j]:.0f} mm from a neighbour during {name.get(k, '?')} "
                                  f"(margin {1000 * m:.0f}, planned {1000 * Dp[k, j]:.0f})")
        if pair_need is not None:
            wo = None if held is None else checked_masks([np.asarray(h, bool)[:n] for h in held])
            d, k = kin.min_arm_arm_distance(A, rest_hand[:n], step=PAIR_CHECK_STEP, with_object=wo,
                                            stop_below=pair_need)
            if d < pair_need:
                return None, f"arm-arm clearance {1000 * d:.0f} mm < {1000 * pair_need:.0f} mm during {name.get(k, '?')}"
        m1 = min(n + 1, R)
        clips = self.dry_run_safety(arm_q, hand_q, np.vstack([A, rest_arm[n:m1]]), rest_hand[:m1])
        if clips:
            return None, f"{clips} safety clips in dry run"
        out = rest_arm.copy()
        out[:n] = A
        return out, {"ticks": n, "ramp_ticks": c0, "blend_out_ticks": n - k1, "weights": w[:n]}

    def go_to(self, arm_from, hand_from, arm_to, hand_to):
        """Joint-space move (both arms + hands) at the configured speeds, e.g. start pose -> home."""
        mo = self.cfg["motion"]
        a = time_scale([arm_from, arm_to], mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
        h = ramp(hand_from, hand_to, mo["hand_max_vel"], self.dt)
        n = max(len(a), len(h))
        a = np.vstack([a, np.repeat(np.asarray(arm_to, float)[None], n - len(a), 0)]) if len(a) < n else a
        h = np.vstack([h, np.repeat(np.asarray(hand_to, float)[None], n - len(h), 0)]) if len(h) < n else h
        return a, h
