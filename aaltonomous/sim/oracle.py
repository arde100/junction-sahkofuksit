"""Scripted oracle for the Aalto-* sim (spec WP3): privileged object poses from the scene service (:55580) instead
of perception, the classical baseline's IK + grasp planner (aaltonomous/baselines/classical, imported, not edited),
commanded through AaltoEnv(backend="sim") + the safety filter like any policy (30 Hz absolute joint targets).

    from aaltonomous.sim.oracle import OraclePolicy
    pol = OraclePolicy(SceneClient())               # threaded: holds while it plans, never trips the watchdog
    pol.reset({"instruction": obs["instruction"]})  # rung 2: mapping read by instruction.py (else privileged)
    act = pol.act(obs)

What it adds over the baseline (all privileged-state only, no camera):
  * detections = active objects resting on the counter (not in a bin), exact centre + shape,
  * per-object target bin via scene.resolve_bin (colour, shape or (colour, shape) keys),
  * relay: an object one arm can grasp but cannot carry to its bin (cross-body) is put down at a free relay
    point near the centre line that the other arm can reach; it is sorted on the next look.
  * faster, slightly looser motion (ORACLE_OVERRIDES), still pre-checked against the safety filter.
  * paired two-arm cycles as the classical policy (config two_arm, from the profile / $AALTO_SET; relays never pair);
    the prefetch then plans from both arms at home without either object of the pair (used only if every object
    then on the counter is one it planned around: a missed object was not a neighbour of that plan).
  * face-aligned grasps (classical grasp.face_align) and drop slots (grasp.slots: the slot farthest from the objects
    really lying in the bin) from privileged shape / yaw / bin contents; neighbour-aware plans as the classical
    policy (blocked objects wait, least-bad rule last).
  * settle and correct (classical grasp.correct, profile sim's values, pinned): the arm's measured sag at the grasp
    is removed before the close (not privileged: measured joints only).

Run episodes / scores with aaltonomous/sim/evaluate.py (--policy oracle). Rung 2: --task sort_language.
"""
from __future__ import annotations

import copy
import itertools
import os as _os
import threading
import time

import numpy as np

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.instruction import parse_instruction as read_instruction
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.baselines.classical.planner import Planner
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.interface import tasks
from aaltonomous.interface.safety import JointSpec
from aaltonomous.sim.aalto_task.layout import ARM_HOME
from aaltonomous.scene.scene import Scene

_SIM = load_config(profile="sim")   # the classical sim profile: source of the grasp the oracle pins below

# Planner overrides (classical config.yaml keys). Defaults leave ~25 % of layout objects unplannable in offline
# tests (3 cm transit clearance, 0.4 rad IK step); these give ~75 % direct + relay for most of the rest.
ORACLE_OVERRIDES = {
    "motion.counter_clearance": 0.02,
    "motion.max_joint_step": 0.6,
    "ik.max_tilt_deg": 35,
    "motion.arm_max_vel": 0.97,    # safety filter allows 1.0
    "motion.arm_max_acc": 8.0,
    "motion.hand_max_vel": 2.5,    # filter allows 3.0
    "motion.close_hold_s": 0.25,
    "motion.settle_s": 0.3,
    "ik.random_restarts": 4,
    # Grasp = the sim profile's thumb-opposed pinch (config.yaml profiles.sim: hand.*, grasp.point_in_palm; pinned
    # here so AALTO_PROFILE=none does not change the oracle), with the IK margins of the sim's soft joint limits.
    # (Until 2026-11-14: grasp point [0.099, -0.084, 0.015] between thumb and index with the base postures.)
    "grasp.point_in_palm": _SIM["grasp"]["point_in_palm"],
    **{f"hand.{k}": v for k, v in _SIM["hand"].items()},
    "ik.joint_margin": _SIM["ik"]["joint_margin"],
    "ik.joint_range": _SIM["ik"].get("joint_range"),   # (none in sim: the base's real-arm range must not reach here)
    "right_hand_slots": _SIM["right_hand_slots"],
    "motion.open_hold_s": 0.4,
    "grasp.arm": "by_y",          # first valid arm (preferred side first) instead of planning both: ~2x faster looks
    # Sim home = the task's arm home pose (layout.ARM_HOME, spec "fixed home pose"). The classical home_arm_q
    # (wrist pitch 1.0, arms out to the sides) collides hand-vs-body in the sim: wrist pitch stalls at ~0.6 and
    # wrist yaw is deflected by ~0.7 rad, which ruins every grasp. The oracle needs no camera view, so no conflict.
    "home_arm_q": [ARM_HOME[n] for n in JointSpec().arm_names],
    # Planner features of the classical config (profile sim has the same values; set here so AALTO_PROFILE=none
    # does not change the oracle): carry joint-space first, segments shortcut to one joint move within 1 cm of the
    # line, no bin margin (the far object row is only ~3 cm from the centre bin wall: a forward-pointing hand was
    # rejected for a few mm and needed a 20 s relay), cached arm-vs-bin reach test.
    "motion.joint_carry_first": True,
    "motion.shortcut_tol": 0.01,
    "motion.bin_margin": 0.0,
    "grasp.reach_check": True,
    # Drop points inside a bin (pelvis xy offsets from the drop point), cycled by how many objects are already in it
    # (privileged count, _update_drop_offsets), so the next object does not land on top of the previous one (a
    # stacked object pokes above the rim = not "in"). Used only with grasp.slots off.
    "grasp.drop_offsets": [[0.0, 0.0], [0.03, 0.03], [-0.03, -0.03], [0.03, -0.03], [-0.03, 0.03]],
    # Drop slots: the one farthest from the objects really lying in the bin (_update_drop_offsets).
    "grasp.slots.enabled": True,
    # Face-aligned grasps (Planner.face_dirs) from the privileged shape + yaw (shape_conf 1.0). $AALTO_TRI_PHI: the
    # triangle's finger direction vs an apex (deg), as before; grasp_lab --tri-phi sets the same key.
    "grasp.face_align.enabled": True,
    "grasp.face_align.triangle_phi_deg": float(_os.environ.get("AALTO_TRI_PHI",
                                                               _SIM["grasp"]["face_align"]["triangle_phi_deg"])),
    "grasp.face_align.cube": _SIM["grasp"]["face_align"]["cube"],
    "grasp.face_align.carry_weight": _SIM["grasp"]["face_align"]["carry_weight"],
    # Neighbour-aware plans (as profile sim).
    **{f"neighbours.{k}": _SIM["neighbours"][k] for k in ("enabled", "margin", "floor", "order_free")},
    # Settle and correct (profile sim's values): the measured sag of the arm at the grasp is removed before the close.
    **{f"grasp.correct.{k}": v for k, v in _SIM["grasp"]["correct"].items()},
    # Sag feed-forward (profile sim's values: the fitted sim gains as the prior, learned online from measured joints).
    **{f"sag.{k}": v for k, v in _SIM["sag"].items()},
}
# Relay candidates (pelvis xy, m), tried in order; must be >= RELAY_CLEAR from every other object centre.
RELAY_POINTS = [(0.30, 0.0), (0.30, 0.05), (0.30, -0.05), (0.27, 0.0), (0.34, 0.0), (0.27, 0.06), (0.27, -0.06)]
RELAY_CLEAR = 0.075
MAX_RELAYS_PER_OBJECT = 1
MAX_RELAY_POINTS_TRIED = 3        # per object and look (each try = 2 full plans, ~1-3 s)
RELAY_BUDGET_S = 10.0             # per look


# The face-aligned grasp (cube face normals / triangle at triangle_phi_deg from an apex, from shape + yaw) lives in the
# classical Planner now (grasp.face_align, Planner.face_dirs); the oracle only feeds it privileged shape / yaw. The
# alias keeps old imports working.
_OraclePlanner = Planner


# ---------------------------------------------------------------------- rung-2 instruction parsing
def _parse_table():
    """Every training-template instruction (tasks.TEMPLATES x colour/bin permutations x bin phrasing) -> mapping
    (the reference the classical reader is tested against: test_instruction.py)."""
    table = {}
    for tpl in tasks.TEMPLATES:
        for colors in itertools.permutations(tasks.COLORS):
            for bins in itertools.permutations(tasks.BINS):
                w = {f"c{i}": c for i, c in enumerate(colors)}
                w.update({f"C{i}": c.capitalize() for i, c in enumerate(colors)})
                w.update({f"b{i}": b for i, b in enumerate(bins)})
                table[tpl.format(**w)] = dict(zip(colors, bins))
    return table


def parse_instruction(text):
    """Mapping read by the classical reader (instruction.py: every training-template sentence and the fixed one
    exactly, paraphrases too) if it understood the text with no error-level problem, else None (then the oracle
    falls back to the privileged mapping)."""
    r = read_instruction(text or tasks.FIXED_INSTRUCTION)
    return dict(r.mapping) if r.ok else None


# ---------------------------------------------------------------------- privileged "perception"
class PrivilegedPerception:
    """Drop-in for classical Perception.detect(): Detections from the scene service, not the camera."""

    def __init__(self, scene: Scene, client):
        self.scene, self.client = scene, client

    def detect(self, rgb=None, depth=None, debug=False):
        dets = []
        for o in self.client.state()["objects"]:
            if not o["active"]:
                continue
            p = np.asarray(o["pos"], float)
            m = self.scene.pelvis_to_mat(p)
            if not self.scene.on_counter(m) or self.scene.bin_of(m) is not None:
                continue
            if p[2] > self.scene.counter_z + self.scene.object_height(o["shape"]) * 1.5:
                continue                                   # in a hand / in the air: not a pick candidate
            top = p + np.array([0.0, 0.0, self.scene.object_height(o["shape"]) / 2])
            q = o.get("quat_xyzw") or [0, 0, 0, 1]
            yaw = float(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
            d = Detection(o["color"], top, 1000, 1000, "oracle", yaw=yaw)
            d.shape, d.name, d.shape_conf = o["shape"], o["name"], 1.0
            dets.append(d)
        return (dets, []) if debug else dets

    def masks(self, rgb):
        return {}


def _relay_scene(scene: Scene, cfg):
    """Scene copy whose 'bins' are the relay points at counter level, so the stock planner places (not drops) there:
    wall_height 0 -> release height = counter + object/2 + drop_above_rim."""
    cfg2 = copy.deepcopy(scene.cfg)
    b = cfg2["bins"]
    b["wall_height"] = 0.0
    b["centers"] = {}
    for i, (x, y) in enumerate(RELAY_POINTS):
        m = scene.pelvis_to_mat([x, y, 0.0])
        b["centers"][f"relay{i}"] = [float(m[0]), float(m[1])]
    return Scene(cfg2)


class OraclePolicy(ClassicalPolicy):
    def __init__(self, client, scene=None, overrides=None, privileged_mapping=None, threaded=True, log=print,
                 prefetch=True, **kw):
        ov = dict(ORACLE_OVERRIDES)
        ov.update(overrides or {})
        self.client = client
        self.prefetch = prefetch
        self._pre = None
        self._plan_lock = threading.Lock()     # pinocchio data + planner state are not thread-safe
        self.privileged_mapping = privileged_mapping     # callable -> mapping (e.g. lambda: env.mapping)
        super().__init__(scene=scene, overrides=ov, threaded=threaded, log=log, **kw)
        self.perception = PrivilegedPerception(self.scene, client)
        slots = self.planner._slots                    # (same config / scene / home: the prepared drop slots)
        self.planner = Planner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec,
                               mapping=self.mapping)
        self.planner.drop_offsets, self.planner._slots = {}, slots
        rcfg = copy.deepcopy(self.cfg)
        rcfg["grasp"]["drop_above_rim"] = 0.012
        rcfg["grasp"]["slots"]["enabled"] = False      # relay "bins" are counter points: no slots
        self.relay = Planner(rcfg, _relay_scene(self.scene, self.cfg), self.post, self.home, self.kin, self.safety,
                             self.spec)
        self.attach_ff(self.planner, self.relay)       # (sag feed-forward: commanded ticks in their dry runs)
        self.relays = {}

    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = dict(info or {})
        if getattr(self, "_pre", None) is not None:   # a prefetch from the previous episode may still run: drop it
            self._pre["cancel"] = True                  # (don't wait here: the env watchdog is already running;
                                                        # the next look waits for the plan lock in its thread)
        super().reset(info)   # mapping: _mapping below (mapping_source given / parsed / fixed / privileged)
        self.relays = {}
        self._pre = None

    def _mapping(self, info):
        """As ClassicalPolicy (explicit mapping, else the instruction read by instruction.py, no instruction: the fixed
        mapping), except that a text the reader did not fully understand (an error-level problem or nothing
        understood) takes the privileged mapping if there is one: the oracle is a reference, not a submission."""
        m, src, r, note = super()._mapping(info)
        if r is not None and not r.ok and self.privileged_mapping is not None:
            pm = self.privileged_mapping()
            if pm is not None:
                return dict(pm), "privileged", r, f"{note} -> privileged mapping {pm}"
        return m, src, r, note

    # ------------------------------------------------------------------ look + plan (privileged)
    def _start(self, plan, t):
        """Execute `plan` (single or pair) and, meanwhile, plan the next object(s) from the pose this plan ends in
        (its last tick; both arms at home after a pair), without the object(s) of this plan."""
        parts = plan.parts or [plan]
        self._exec_bins = [str(p.target_bin) for p in parts          # (an evade's idle part has no bin)
                           if p.target_bin and not str(p.target_bin).startswith("relay")]
        self._exec_drops = [(str(p.target_bin), p.waypoints["drop"][:2].copy()) for p in parts
                            if p.target_bin in self.scene.bin_names and "drop" in p.waypoints]
        out = super()._start(plan, t)
        if self.prefetch and plan.arm is not None and len(plan.arm):
            job = {"arm": plan.arm[-1].copy(), "hand": plan.hand[-1].copy(), "t": t + plan.duration,
                   "exclude": {getattr(p, "obj", None) for p in parts} - {None}, "result": None, "dets": None,
                   "ff": self._ff_version()}
            job["thread"] = threading.Thread(target=self._prefetch_job, args=(job,), daemon=True)
            self._pre = job
            job["thread"].start()
        return out

    def _prefetch_job(self, job):
        try:
            with self._plan_lock:
                if job.get("cancel"):
                    return
                job["result"] = self._plan_next(job["arm"], job["hand"], exclude=job["exclude"], commit=False,
                                                t=job["t"])
                job["dets"] = {d.name: d.top[:2].copy() for d in self.last_detections}   # the scene it planned for
        except Exception as e:  # fall back to a normal look
            job["result"] = ("error", repr(e))

    def _look(self, obs):
        with self._plan_lock:
            return self._look_locked(float(obs.get("t", 0.0)))

    def _look_locked(self, t=0.0):
        job, self._pre = getattr(self, "_pre", None), None
        if job is not None:
            res = job["result"]          # the job ran (or was skipped) before we got the lock
            if (res and res[0] == "plan" and np.allclose(job["arm"], self.arm_cmd, atol=1e-6)
                    and np.allclose(job["hand"], self.hand_cmd, atol=1e-6)):
                p = res[1]
                now = {d.name: d for d in self.perception.detect()}
                pre = job.get("dets") or {}
                if (all(now.get(q.obj) is not None and np.linalg.norm(now[q.obj].top[:2] - q.obj_xy) < 0.01
                        for q in p.parts or [p] if not q.evade)        # object(s) still where planned, and every
                        and all(n in pre and np.linalg.norm(d.top[:2] - pre[n]) < 0.01   # object on the counter is
                                for n, d in now.items())               # one it planned around (a missed one is not)
                        and self._ff_still_ok(job, p)):
                    self._commit(p)
                    return ("plan", p, len(now))
        return self._plan_next(self.arm_cmd, self.hand_cmd, t=t)

    def _ff_version(self):
        return None if self.ff is None else self.ff.version

    def _ff_still_ok(self, job, p):
        """Sag feed-forward: a prefetched plan was dry-run at the gain applied then; the settle before this look may
        have blended in a new one: dry-run it again at the current gain (0 clips) or plan afresh."""
        if job.get("ff") == self._ff_version():
            return True
        return self.planner.dry_run_safety(self.arm_cmd, self.hand_cmd, p.arm, p.hand) == 0

    def _commit(self, p):
        if p.relay is not None:
            self.relays[p.obj] = self.relays.get(p.obj, 0) + 1

    def _update_drop_offsets(self, exclude=None):
        """Privileged bin contents -> drop_offsets (count per bin) and drop_used (where they lie: grasp.slots)."""
        n = {b: 0 for b in self.scene.bin_names}
        used = {b: [] for b in self.scene.bin_names}
        for o in self.client.state()["objects"]:
            if o["active"]:
                p = np.asarray(o["pos"], float)
                b = self.scene.bin_of(self.scene.pelvis_to_mat(p))
                if b is not None:
                    n[b] += 1
                    used[b].append(p[:2].copy())
        if exclude:                                       # planning ahead: the object(s) being carried now
            for b in getattr(self, "_exec_bins", []):
                if b in n:
                    n[b] += 1
            for b, xy in getattr(self, "_exec_drops", []):
                used.setdefault(b, []).append(xy)
        offs = self.cfg["grasp"]["drop_offsets"]
        self.planner.drop_offsets = {b: offs[k % len(offs)] for b, k in n.items()}
        self.planner.drop_used = used

    def _plan_next(self, arm, hand, exclude=None, commit=True, t=0.0):
        """exclude: object name(s) not to plan (being carried now). t: when the plan would start (time budget of a
        pair; a single plan's budget is checked by _after_look as before)."""
        exclude = {exclude} if isinstance(exclude, str) else set(exclude or ())
        self._update_drop_offsets(exclude)
        dets = [d for d in self.perception.detect() if d.name not in exclude]
        self.last_detections = dets
        cands = []
        for d, rec in zip(dets, self._track(dets, commit=commit)):   # per-object records, keyed by name here
            if not self._eligible(rec, d.top[:2]):
                continue
            d.bin = self.bin_for(d)          # privileged shape: scene.resolve_bin; no bin -> not a candidate
            if d.bin is not None:
                d.rec = rec
                cands.append(d)
        cands.sort(key=lambda d: (d.rec["tries"] > 0, self._crowded(d, dets),
                                  np.linalg.norm(d.top[:2] - self.planner.bin_xy(d.bin))))
        unplanned, blocked = [], []
        for d in cands:
            others = [o for o in dets if o is not d]
            if self._same_scene(d.rec.get("blocked"), others):   # blocked in this very scene (see ClassicalPolicy)
                blocked.append(d)
                continue
            p = self.planner.plan(d, arm, hand, bin_name=d.bin, relocate=False,   # the oracle relays instead
                                  variant=self._variant(d.rec), neighbours=others)
            if p.ok:
                d.rec["blocked"] = None
                m = self._take(p, d, cands, unplanned + blocked, arm, hand, t, commit)
                if m is not None:
                    return ("plan", m, len(dets))
                continue
            if p.blocked:                            # a neighbour in the way: others (and relays) first
                if commit:
                    d.rec["blocked"] = np.array([o.top[:2] for o in others]).reshape(-1, 2)
                blocked.append(d)
                continue
            unplanned.append(d)
        t0 = time.monotonic()
        for d in unplanned:
            if time.monotonic() - t0 > RELAY_BUDGET_S:
                break
            p = self._plan_relay(d, dets, arm, hand, t)
            if p is not None:
                if commit:
                    self._commit(p)
                return ("plan", p, len(dets))
            if commit:
                d.rec["failed_at"] = d.top[:2].copy()
                self.log(f"[oracle] skip {d.name}: no direct or relay plan")
        for d in blocked:                            # nothing else to do: the least-bad rule
            others = [o for o in dets if o is not d]
            if not (self.cfg.get("neighbours") or {}).get("least_bad", True) or self._same_scene(
                    d.rec.get("lb_failed"), others):
                continue
            p = self.planner.plan(d, arm, hand, bin_name=d.bin, relocate=False, variant=self._variant(d.rec),
                                  neighbours=others, least_bad=True)
            if not p.ok:                             # stays blocked (not failed): may be clear once another has gone
                if commit:
                    d.rec["lb_failed"] = np.array([o.top[:2] for o in others]).reshape(-1, 2)
                    self.log(f"[oracle] {d.name}: blocked, no least-bad plan ({p.reason})")
                continue
            self._note(t, f"least-bad: {d.name} {p.side} arm, {1000 * p.nb_clear:.0f} mm from the nearest object; "
                          f"{p.reason}")
            m = self._take(p, d, cands, unplanned + blocked, arm, hand, t, commit)
            if m is not None:
                return ("plan", m, len(dets))
        return ("none", None, len(dets))

    def _take(self, p, d, cands, skip, arm, hand, t, commit):
        """Direct plan p of candidate d -> what to execute: a pair with a later candidate (not one in `skip`), else
        p / its evade (_guard); None if the parked arm is in the way without an evade (failed like a plan)."""
        p.obj, p.obj_xy, p.relay, p.rec = d.name, d.top[:2].copy(), None, d.rec
        rest = [(c, c.rec, c.bin) for c in cands if c is not d and not any(c is u for u in skip)]
        m = self._pair(p, rest, arm, hand, t)                                                 # two_arm
        if m is p:                                   # avoid_parked: clear of the parked arm, else an evade
            m = self._guard(p, arm, hand, t)
            if m is None:                            # (skipped like a failed plan; no relay for it)
                if commit:
                    d.rec["failed_at"] = d.top[:2].copy()
                return None
        else:
            q = m.parts[1]
            det2 = next(c for c in cands if c.rec is q.rec)
            q.obj, q.obj_xy, q.relay = det2.name, det2.top[:2].copy(), None
            m.obj, m.obj_xy, m.relay, m.rec = None, None, None, None   # (per part; pairs never relay)
        return m

    def _plan_relay(self, d, dets, arm, hand, t=0.0):
        """Relay plan for `d` (first leg; avoid_parked: clear of the parked arm or with an evade, else the next
        relay point), or None."""
        if self.relays.get(d.name, 0) >= MAX_RELAYS_PER_OBJECT:
            return None
        nbs = [o for o in dets if o.name != d.name]
        others = [o.top[:2] for o in nbs]
        tried = 0
        for i, xy in enumerate(RELAY_POINTS):
            if tried >= MAX_RELAY_POINTS_TRIED:
                break
            xy = np.asarray(xy, float)
            if np.linalg.norm(xy - d.top[:2]) < 0.04 or any(np.linalg.norm(xy - o) < RELAY_CLEAR for o in others):
                continue
            # second leg first (cheap reject): from the relay point to the bin, starting at home
            fake = Detection(d.color, np.r_[xy, d.top[2]], 1000, 1000, "relay")
            tried += 1
            p2 = self.planner.plan(fake, self.home, self.post["open"], bin_name=d.bin, relocate=False, neighbours=nbs)
            if not p2.ok:
                continue
            p1 = self.relay.plan(d, arm, hand, bin_name=f"relay{i}", relocate=False, variant=self._variant(d.rec),
                                 neighbours=nbs)
            if p1.ok:
                p1.target_bin = f"relay -> {d.bin}"
                p1.obj, p1.obj_xy, p1.relay, p1.rec = d.name, d.top[:2].copy(), i, d.rec
                g = self._guard(p1, arm, hand, t)                # (the second leg is guarded when it is planned)
                if g is None:
                    continue
                self.log(f"[oracle] relay {d.name} via ({xy[0]:.2f},{xy[1]:+.2f}): {p1.side} then {p2.side}"
                         + (" (evade)" if g is not p1 else ""))
                return g
        return None
