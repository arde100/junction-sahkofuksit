"""Grasp lab: fast single-object grasp trials in the Aalto sim, one grasp VARIANT per run, many cases (shape x yaw x
position x repeat). PRIVILEGED analysis tool: object poses come from the scene service. Nothing in a policy may
import this module.

Per case:
  1. place ONE object through the scene service (mat coords, as grasp_test.py), wait until it rests, read its exact
     pose (top centre, shape, yaw: oracle.PrivilegedPerception) + an optional constant xy error (--xy-error),
  2. plan with the oracle planner's counter-place path (Planner._plan_side(place_xy=...), the relocation path):
     pregrasp -> descend -> close -> lift to carry height -> HOLD (--hold s) -> lower at the same spot -> release ->
     retreat. The plan's home move is cut off and kept as a checked "tail". Every planner check still runs (counter
     + bin clearance, dry_run_safety); the executed ticks (+ hold, + tail) are dry-run again (0 clips required),
  3. execute through AaltoEnv(backend="sim") + its SafetyFilter at 30 Hz, sampling the object pose (--sample-hz)
     and the commanded / measured hand_q; derive lifted / held / released / displaced / sig / self_lift (see _measure).
     Settle and correct (grasp.correct, --correct on | off | dwell): at the end of the settle phase the measured
     grasp-point error and the shift (row "corr": err_mm, along / across / up, dq, applied or why not), as the
     policy does it; every case: err_close_* = planned - measured grasp point CLOSE_ERR_TICKS into the close (the
     fingers moving, not yet on the object): the A/B number.
     Sag feed-forward (config sag, --ff on | off; default the config's sag.enabled, sim profile: on): every tick is
     sent as intended + the feed-forward offset at the sim prior (fixed here: no online learning, so cases compare);
     plans, the correction and every error stay in intended space. err_pre_corr_* (with --correct on / dwell) =
     planned - measured grasp point at the end of the settle, before any shift: what the feed-forward alone achieves.
The next case is planned in a thread while this one runs (start = this case's retreat pose, object pose predicted
from its layout entry; planned again if the placed object is elsewhere), so the arm goes DIRECTLY to the next
pregrasp. Via the previous plan's home tail only if the direct move does not plan, the arm changes, or the hand is
too low to place the next object under it. A case whose plan fails is recorded with the reason and never touches the
scene (no placement, motion or settling; only its planning, which normally overlaps the previous case's motion).

    python -m aaltonomous.sim.grasp_lab --shapes cube,triangle --yaws -45,0,30 --out /data/.../base.jsonl
    python -m aaltonomous.sim.grasp_lab --set '{"hand.closed": [0, -0.1, 1.2, -1.0, -0.6, -1.0, -0.6]}' --label c2
    python -m aaltonomous.sim.grasp_lab --profile base ... # the classical config's base (real robot) values instead of
                                                           # ORACLE_OVERRIDES (sim: e.g. AALTO_VARIANT=real_like)
    python -m aaltonomous.sim.grasp_lab --mock ...        # offline mock world: control flow only, no physics
    python -m aaltonomous.sim.grasp_lab --plan-only ...   # which cases plan (no sim, no env)
(env `isaac`, PYTHONPATH=$REPO; 2nd sim: AALTO_* env vars as in evaluate.py / grasp_test.py)

Output (--out): JSON Lines, appended + flushed per case: one {"type": "meta"} line per run (args, variant, config
excerpt), then one {"type": "case"} row per case (_row0 + _measure fields, "trace" = sampled object poses).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import signal
import sys
import threading
import time
import traceback

import numpy as np

from aaltonomous.sim import evaluate as ev

HZ = 30.0
LIFT_MIN = 0.08          # lifted / held: object centre this far above its rest height
HELD_GAP = 0.04          # held: object centre this close to the measured grasp point
RELEASE_TOL = 0.015      # released: final centre within this of its rest height ...
CARRY_TOL = 0.02         # ... and it rose less than this with the retreating hand (not wedged in the fingers)
TIP_DEG = 20.0           # final tilt beyond this = tipped over
SIG_DELAY_S = 1.0        # sig: |measured - commanded hand_q| this long after `closed` is first fully commanded
POSE_TOL = (0.002, np.radians(2.0))    # placed vs predicted pose beyond this: the prefetched plan is redone
SYM = {"cube": np.pi / 2, "triangle": 2 * np.pi / 3}   # yaw symmetry
PLACE_BIN = "center"     # any bin name: the counter-place path ignores it (no reach test with place_xy)
LOW_HAND = 0.06          # recovery: a grasp point lower than this above the counter is raised before homing
CLOSE_ERR_TICKS = 3      # err_close_*: planned - measured grasp point this many ticks into the close


class CaseTimeout(RuntimeError):
    pass


def _wrap(a, period=2 * np.pi):
    return (a + period / 2) % period - period / 2


def _yaw_tilt(q):
    yaw = float(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
    return yaw, float(np.degrees(2 * np.arcsin(min(1.0, float(np.hypot(q[0], q[1]))))))


def _r(x, n=1):
    return None if x is None else round(float(x), n)


def _floats(s):
    return [float(v) for v in s.split(",")] if s else None


# ---------------------------------------------------------------------- scene service: retries + mock stand-in
class _Retry:
    """SceneClient wrapper: TimeoutError (SceneClient recreates its REQ socket) -> retried up to `tries` times."""

    def __init__(self, sc, tries=3, log=print):
        self.sc, self.tries, self.log = sc, tries, log

    def __getattr__(self, name):
        f = getattr(self.sc, name)
        if not callable(f):
            return f

        def call(*a, **kw):
            for k in range(self.tries):
                try:
                    return f(*a, **kw)
                except TimeoutError:
                    if k == self.tries - 1:
                        raise
                    self.log(f"[grasp_lab] scene service timeout on {name}, retry {k + 1}")
        return call


class MockScene:
    """SceneClient stand-in over RenderMockBackend.objects (reset / state / info / ping / close). Speed = finite
    difference between state() calls on the env clock."""

    def __init__(self, env):
        self.env, self.b = env, env.backend
        self._last = {}

    def reset(self, layout):
        self.b.reset_scene([dict(o) for o in layout], None)
        self._last = {}
        return {"ok": True, "active": [n for n, _ in self._objs()], "layout": layout}

    def _objs(self):
        return [(f"obj_{o['color']}_{o['shape']}_{i}", o) for i, o in enumerate(self.b.objects)]

    def state(self):
        from aaltonomous.sim.aalto_task.layout import mat_yaw_to_pelvis, yaw_quat_xyzw
        t = getattr(self.env, "_vt", 0.0)
        out = []
        for n, o in self._objs():
            p = np.asarray(o["pos"], float)
            t0, p0 = self._last.get(n, (t, p))
            v = float(np.linalg.norm(p - p0) / (t - t0)) if t > t0 else 0.0
            self._last[n] = (t, p.copy())
            out.append({"name": n, "color": o["color"], "shape": o["shape"], "active": True, "pos": p.tolist(),
                        "quat_xyzw": yaw_quat_xyzw(mat_yaw_to_pelvis(o["yaw"])).tolist(), "speed": v})
        return {"ok": True, "objects": out}

    def info(self):
        return {"ok": True, "dr": None, "mock": True}

    def ping(self):
        return {"ok": True}

    def close(self):
        pass


def make_mock(cfg, seed=0, sag=False):
    """Offline AaltoEnv(backend="mock") with mock_world's kinematic grasping (images blank: no render per tick)
    + a MockScene. Same construction as mock_world.make_mock_env, with this lab's config. sag: the mock arms track
    with the sim's gravity sag (mock_world.SIM_SAG_GAIN)."""
    from aaltonomous.baselines.classical.config import hand_postures
    from aaltonomous.baselines.classical.kinematics import ArmKinematics
    from aaltonomous.baselines.classical.mock_world import RenderMockBackend
    from aaltonomous.interface.backends import blank_images
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.interface.env import AaltoEnv

    class _Backend(RenderMockBackend):
        _blank = blank_images()

        def images(self):
            return dict(self._blank)

    env = AaltoEnv(backend="mock", task="sort_fixed", seed=seed, realtime=False, time_limit=1e9)
    env.backend = _Backend(env.scene, Calibration.load(), ArmKinematics(cfg), hand_postures(cfg), seed=seed,
                           sag_gain=bool(sag))
    return env, MockScene(env)


# ---------------------------------------------------------------------- the lab
class Lab:
    def __init__(self, a, cfg, env=None, sc=None, log=print):
        from aaltonomous.baselines.classical.config import hand_postures, home_arm
        from aaltonomous.baselines.classical.kinematics import ArmKinematics
        from aaltonomous.baselines.classical.planner import Planner
        from aaltonomous.interface.safety import JointSpec, SafetyConfig
        from aaltonomous.scene.scene import Scene
        from aaltonomous.sim import oracle
        self.a, self.cfg, self.env, self.sc, self.log = a, cfg, env, sc, log
        self.oracle = oracle
        self.scene = env.scene if env is not None else Scene.load()
        self.spec = spec = JointSpec()
        self.post = hand_postures(cfg, spec)
        self.home_q = home_arm(cfg, spec)
        safety = env.filter.cfg if env is not None else SafetyConfig(dt=1.0 / HZ)
        self.planner = Planner(cfg, self.scene, self.post, self.home_q, ArmKinematics(cfg, spec), safety, spec)
        self.mkin = ArmKinematics(cfg, spec) if env is not None else None   # main-thread FK (pinocchio data per thread)
        # settle and correct (grasp.correct, --correct): computed in the main thread with its own planner (the
        # planning thread owns self.planner); "dwell" = the settle phase, the error measured, the shift NOT applied
        cc = cfg["grasp"].get("correct") or {}
        self.correct = (getattr(a, "correct", None) or "on") if cc.get("enabled") else "off"
        self.cplanner = (Planner(cfg, self.scene, self.post, self.home_q, self.mkin, safety, spec)
                         if env is not None and self.correct != "off" else None)
        # sag feed-forward: planners dry-run the commanded ticks; _step sends them (main-thread kinematics: mkin)
        from aaltonomous.baselines.classical.sag import SagFF
        self.ff = SagFF(cfg) if (cfg.get("sag") or {}).get("enabled") else None
        if self.ff is not None:
            self.ff.learn = False          # the prior, fixed: cases stay comparable
            self.planner.ff = self.ff
            if self.cplanner is not None:
                self.cplanner.ff = self.ff
            self.fkin = self.mkin or ArmKinematics(cfg, spec)
        self.perc = oracle.PrivilegedPerception(self.scene, sc) if sc is not None else None
        self.dt = 1.0 / HZ
        self.H = self.scene.object_height()
        self.zc = self.scene.counter_z
        self.rest_speed = float(self.scene.cfg["success"]["at_rest_speed"])
        self.xy_err = np.r_[np.asarray(_floats(a.xy_error) or [0, 0], float) / 1000, 0.0]
        self.place_off = np.asarray(_floats(a.place_offset) or [0, 0], float) / 1000
        self.home_hand = np.clip(np.zeros(len(spec.hand_names)), spec.hand_lower + 0.05, spec.hand_upper - 0.05)
        self.last = (self.home_q.copy(), self.home_hand.copy())   # last targets sent = the planner's start
        self.prev = None          # {"side", "tail"}: executed body that ends at self.last (direct moves allowed)
        self.job, self.epoch = None, 0
        self.deadline, self._tbase, self._t_case = None, 0.0, 0.0
        self.trace, self.clips, self.clip_reasons = [], 0, set()
        self._name = self._main_xy = None     # the case object (by name) among the placed ones; its placed xy
        self._nb0 = {}                         # --neighbour: the other objects' poses after placement

    # ------------------------------------------------------------------ clock + stepping
    def _env_t(self):
        try:
            return self.env._now()
        except AttributeError:    # mock before its first reset
            return 0.0

    def now(self):
        """Sim seconds since start; never resets (env._now() restarts at every env.reset, e.g. in home_arms)."""
        if self.env is None:
            return 0.0
        c = self.env.clock
        return c.now() if c is not None else self._tbase + self._env_t()

    def _sent(self, arm, hand):
        """What goes to the env for intended `arm` (sag feed-forward: + its offset)."""
        return np.asarray(arm, float) if self.ff is None else self.ff.commanded(self.fkin, arm, hand)

    def _from_filter(self):
        """(intended arm, hand) of the filter's last command (sag feed-forward: the pose that command yields)."""
        a, h = self.env.filter.arm_cmd.copy(), self.env.filter.hand_cmd.copy()
        return (a if self.ff is None else self.ff.intended(self.fkin, a, h)), h

    def _step(self, arm, hand):
        _, info = self.env.step({"arm_q_target": self._sent(arm, hand), "hand_q_target": hand})
        self.last = (np.asarray(arm, float).copy(), np.asarray(hand, float).copy())   # intended
        if info["clipped"]:
            self.clips += 1
            self.clip_reasons.update(info["clip_reasons"])
        if info["safety_mode"] in ("safe_stop", "aborted") or info["episode_over"]:
            raise RuntimeError(f"safety filter {info['safety_mode']} (end {info['end_reason']})")
        if self.deadline is not None and self.now() > self.deadline:
            raise CaseTimeout(f"case time limit {self.a.time_limit:.0f} s (sim)")

    def _hold(self, n=1):
        for _ in range(n):   # explicit target = the last one (as deploy.runner while waiting): the watchdog stays quiet
            self._step(*self.last)

    def home(self):
        if self.env.clock is None:
            self._tbase += self._env_t()          # home_arms -> env.reset restarts env._now()
        err = ev.home_arms(self.env, self.sc)
        self.last = self._from_filter()
        self.prev, self.epoch = None, self.epoch + 1
        return err

    def _rearm(self):
        """Fresh filter episode from the measured pose (after a safe stop / abort the filter only holds)."""
        if self.env.clock is None:
            self._tbase += self._env_t()
        b = self.env.backend
        orig = b.reset_scene
        b.reset_scene = lambda layout, instruction: None
        try:
            self.env.reset()
        finally:
            b.reset_scene = orig
        self.last = self._from_filter()

    def _recover(self):
        """After an error: wait for the planner thread, raise a low hand straight up (checked), home_arms."""
        self.deadline, self.epoch = None, self.epoch + 1
        self._wait_job(hold=False)
        done = []
        try:
            self._rearm()
            done += [f"raised {s}" for s in self._raise_hands()]
        except Exception as e:
            self.log(f"[grasp_lab] raise before homing failed ({e!r}); homing directly")
        last = None
        for _ in range(2):
            try:
                self.home()
                return ", ".join(done + ["homed"])
            except Exception as e:
                last = e
                self.log(f"[grasp_lab] homing failed: {e!r}")
        raise RuntimeError(f"recovery failed: {last!r}")

    def _raise_hands(self):
        from aaltonomous.baselines.classical.planner import Plan, time_scale
        P, mo, out = self.planner, self.cfg["motion"], []
        for side in ("left", "right"):
            arm, hand = self.last
            p0 = P.kin.grasp_pose(side, arm)[0]
            if p0[2] > self.zc + LOW_HAND:
                continue
            qs = P._cartesian(side, p0, p0 + [0, 0, LOW_HAND], arm, "raise", Plan(False))
            if not qs:
                continue
            A = time_scale([arm] + qs, 0.3, mo["arm_max_acc"], self.dt)
            Hh = np.repeat(hand[None], len(A), 0)
            if len(A) and not P.dry_run_safety(arm, hand, A, Hh):
                for a_, h_ in zip(A, Hh):
                    self._step(a_, h_)
                out.append(side)
        return out

    # ------------------------------------------------------------------ scene
    @staticmethod
    def _pose(o):
        yaw, tilt = _yaw_tilt(o.get("quat_xyzw") or [0, 0, 0, 1])
        return {"p": np.asarray(o["pos"], float), "yaw": yaw, "tilt": tilt, "speed": float(o["speed"])}

    def _obj(self, phase=None, side=None, others=None):
        """The case's object (the one placed nearest the case position, then followed by name: --neighbour places
        more); others: dict filled with the other objects' poses by name."""
        objs = [o for o in self.sc.state()["objects"] if o["active"]]
        if not objs:
            return None
        o = next((o for o in objs if o.get("name") == self._name), None) if self._name else None
        if o is None:
            o = objs[0] if self._main_xy is None else min(
                objs, key=lambda o: np.linalg.norm(np.asarray(o["pos"][:2], float) - self._main_xy))
            self._name = o.get("name")
        if others is not None:
            others.update({q.get("name"): self._pose(q) for q in objs if q is not o})
        s = {"t": self.now(), **self._pose(o)}
        tilt = s["tilt"]
        if phase is not None:
            sig = self._sig(side) if side else None
            self.trace.append([_r(s["t"] - self._t_case, 2), phase] + [_r(1000 * v) for v in s["p"]]
                              + [_r(tilt), _r(sig, 3)])
        return s

    def _settle(self, min_s, max_s, phase, every=3):
        """Hold until the object is at rest (2 samples in a row below success.at_rest_speed), >= min_s, <= max_s."""
        t0, calm, o = self.now(), 0, None
        while True:
            self._hold(every)
            o = self._obj(phase)
            calm = calm + 1 if o is None or o["speed"] < self.rest_speed else 0
            if (self.now() - t0 >= min_s and calm >= 2) or self.now() - t0 >= max_s:
                return o

    def _detect(self, case):
        """The case object's detection (+ the --xy-error); the other detected objects (--neighbour) go to
        case["_nb_det"] (planned around if neighbours.enabled)."""
        dets = self.perc.detect()
        if not dets:
            raise RuntimeError("placed object not detected on the counter")
        pred = self.scene.mat_to_pelvis([case["x"], case["y"], self.H])[:2]
        d = min(dets, key=lambda q: np.linalg.norm(q.top[:2] - pred))
        case["_nb_det"] = [q for q in dets if q is not d]
        d.top = d.top + self._err(case)
        return d

    def _nb_pred(self, case):
        """Neighbour Detections expected from the case's layout (--neighbour), for planning before placement."""
        out = []
        for n in case.get("nb") or []:
            d = self._det_pred({**n, "rep": 0, "xy_error": (0.0, 0.0)})
            out.append(d)
        return out

    def _err(self, case):
        e = case.get("xy_error")
        return self.xy_err if e is None else np.r_[np.asarray(e, float) / 1000, 0.0]

    def _det_pred(self, case):
        """Detection expected from the layout entry (what the scene service will report once it rests)."""
        from aaltonomous.baselines.classical.perception import Detection
        from aaltonomous.sim.aalto_task.layout import mat_yaw_to_pelvis
        top = self.scene.mat_to_pelvis([case["x"], case["y"], self.H]) + self._err(case)
        yaw = float(_wrap(mat_yaw_to_pelvis(np.radians(case["yaw"]))))
        d = Detection(case["color"], top, 1000, 1000, "layout", yaw=yaw)
        d.shape, d.name, d.shape_conf = case["shape"], None, 1.0   # (as PrivilegedPerception: face_align.min_conf)
        return d

    def _moved(self, det, pred):
        if np.linalg.norm(det.top[:2] - pred.top[:2]) > POSE_TOL[0]:
            return True
        return det.shape in SYM and abs(_wrap(det.yaw - pred.yaw, SYM[det.shape])) > POSE_TOL[1]

    # ------------------------------------------------------------------ planning (thread)
    def _sides(self, det):
        if self.a.arm in ("left", "right"):
            return [self.a.arm]
        s0 = "left" if det.top[1] > self.cfg["grasp"]["y_split"] else "right"
        return [s0, "right" if s0 == "left" else "left"]

    def _plan_side(self, det, side, arm, hand, nbs=None):
        P = self.planner
        if nbs:   # other objects on the counter: the planner's neighbour rule (only with neighbours.enabled)
            return P._scoped(nbs, None, self._plan_side, det, side, arm, hand)
        off = np.asarray((self.cfg["grasp"].get("arm_xy_offset") or {}).get(side, [0.0, 0.0]), float)
        kw = {"place_xy": det.top[:2] + off + self.place_off}     # put it down where it was grasped
        if self.a.finger_yaw is not None:
            return self._plan_forced(det, side, arm, hand, **kw)
        # face-aligned candidates first if grasp.face_align is on (--face-align / --no-face-align), then the hints
        return P._plan_side_hints(det, side, PLACE_BIN, arm, hand, **kw)

    def _plan_forced(self, det, side, arm, hand, **kw):
        """--finger-yaw: fingers along this pelvis-frame yaw (strong IK hint as the face-aligned grasp; triangle
        grasp point shifted to the pinch midpoint like there)."""
        from aaltonomous.baselines.classical.planner import tri_shift
        P, fa = self.planner, self.cfg["grasp"]["face_align"]
        a = np.radians(self.a.finger_yaw)
        f = np.array([np.cos(a), np.sin(a), 0.0])
        d2 = copy.copy(det)
        if det.shape == "triangle" and det.yaw is not None:
            d2.top = det.top + tri_shift(a - (det.yaw + np.pi / 2), self.scene.object_dims("triangle")["circumradius"]) * f
        saved = (P.kin.hint, P.kin.w_hint)
        try:
            P.kin.hint, P.kin.w_hint = f, float(fa["hint_weight"])
            p = P._plan_side(d2, side, PLACE_BIN, arm, hand, **kw)
        finally:
            P.kin.hint, P.kin.w_hint = saved
        if p.ok:
            err = P._finger_err(p, side, f)
            p.ok = err <= float(fa["tol_deg"])
            p.reason = (f"forced finger yaw {self.a.finger_yaw:.0f} deg "
                        f"({'err' if p.ok else 'NOT reached, err'} {err:.0f})")
        return p

    def _body(self, p):
        """Plan -> executed body: home move cut off (kept as `tail`), --hold s inserted after the lift."""
        ph = {n: (s, e) for n, s, e in p.phases}
        hs = ph["home"][0] if "home" in ph else len(p.arm)
        cut = ph.get("lift", ph["close"])[1]
        nh = int(round(self.a.hold / self.dt))

        def ins(X):
            return np.vstack([X[:cut], np.repeat(X[cut - 1][None], nh, 0), X[cut:hs]])
        phases = []
        for n, s, e in p.phases:
            if s >= hs:
                continue
            sh = nh if s >= cut else 0
            phases.append(({"carry": "place"}.get(n, n), s + sh, e + sh))
            if e == cut:
                phases.append(("hold", cut, cut + nh))
        return {"A": ins(p.arm), "H": ins(p.hand), "phases": phases, "tail": (p.arm[hs:], p.hand[hs:])}

    def _transit_clear(self, side, A, Hh, phases):
        """Direct move: hand points over the counter stay >= object height + 1 cm while going to the pregrasp (the
        planner only keeps them off the counter; the next object may stand anywhere under the path)."""
        s, e = next(((s, e) for n, s, e in phases if n == "to_pregrasp"), (0, 0))
        ks = list(range(s, e, 3)) + ([e - 1] if e > s else [])
        return min((self.planner.clearance(side, A[k], Hh[k]) for k in ks), default=np.inf)

    def _plan_job(self, case, det, start, prev):
        """Arms in preference order; per arm: direct from `start` (same arm as the previous body, its hand clear of
        the next object), else from the end of the previous plan's home tail (executed first); without a previous
        body (at home / after recovery) from `start`."""
        t0 = time.monotonic()
        res = {"case": case, "det": det, "start": start, "ok": False, "tries": [], "reason": ""}
        try:
            need = self.H + 0.01
            opts = []
            if prev is None:
                sent = self.planner.commanded(start[0], start[1])   # (home_arms parks the COMMAND at home)
                how = ("home" if np.allclose(start[0], self.home_q, atol=1e-6) or np.allclose(sent, self.home_q,
                                                                                             atol=1e-6) else "here")
                opts = [(how, start, None, s) for s in self._sides(det)]
            else:
                clr = min(self.planner.clearance(s, *start) for s in ("left", "right"))
                if clr < need:
                    res["tries"].append(f"direct: hand {100 * clr:.1f} cm above the counter (< {100 * need:.1f})")
                ta, th = prev["tail"]
                for s in self._sides(det):
                    if s == prev["side"] and clr >= need:
                        opts.append(("direct", start, None, s))
                    if len(ta):
                        opts.append(("via_home", (ta[-1], th[-1]), prev["tail"], s))
            nbs = case.get("_nb_det")
            nbs = self._nb_pred(case) if nbs is None else nbs
            for how, (arm, hand), prefix, side in opts:
                p = self._plan_side(det, side, arm.copy(), hand.copy(), nbs)
                if not p.ok:
                    res["tries"].append(f"{how}/{side}: {p.reason}")
                    continue
                b = self._body(p)
                if how == "direct":
                    c = self._transit_clear(side, b["A"], b["H"], b["phases"])
                    if c < need:
                        res["tries"].append(f"direct/{side}: transit {100 * c:.1f} cm above the counter")
                        continue
                A, Hh = b["A"], b["H"]
                if prefix is not None:
                    A, Hh = np.vstack([prefix[0], A]), np.vstack([prefix[1], Hh])
                clips = self.planner.dry_run_safety(start[0], start[1], A, Hh)
                if clips:
                    res["tries"].append(f"{how}/{side}: {clips} clips in the dry run of the executed ticks")
                    continue
                res.update(ok=True, plan=p, how=how, prefix=prefix, info=self._plan_info(p, det), **b)
                res["info"]["exec_plan_s"] = _r(len(A) * self.dt, 2)
                res["info"]["phase_s"] = {n: _r((e - s) * self.dt, 2) for n, s, e in b["phases"]}
                if prefix is not None:
                    res["info"]["phase_s"]["home_tail"] = _r(len(prefix[0]) * self.dt, 2)
                break
            if not res["ok"]:
                res["reason"] = "; ".join(res["tries"]) or "no option"
        except Exception as e:
            res.update(ok=False, reason=f"planner error: {e!r}", tb=traceback.format_exc(limit=6))
        res["wall_s"] = time.monotonic() - t0
        return res

    def _plan_info(self, p, det):
        kin = self.planner.kin
        e = next(e for n, s, e in p.phases if n == "descend")
        kin._fk(p.arm[e - 1])
        x = kin.data.oMf[kin.palm[p.side]].rotation[:, 0]
        fy = float(np.arctan2(x[1], x[0]))
        rel = None     # cube: finger direction vs nearest face normal (0-45); triangle: vs nearest apex (0-60)
        if det.yaw is not None and det.shape == "cube":
            rel = abs(_wrap(fy - det.yaw, np.pi / 2))
        elif det.yaw is not None and det.shape == "triangle":
            rel = abs(_wrap(fy - det.yaw - np.pi / 2, 2 * np.pi / 3))
        g, d = p.waypoints["grasp"], p.waypoints["drop"]
        ik = [v for v in p.ik.values() if v[0]]
        return {"side": p.side, "plan_note": p.reason, "finger_yaw_deg": _r(np.degrees(fy)),
                "finger_rel_deg": _r(None if rel is None else np.degrees(rel)),
                "obj_yaw_deg": _r(None if det.yaw is None else np.degrees(det.yaw)),
                "tilt_grasp_deg": _r(p.ik["grasp"][2]), "tilt_max_deg": _r(max(t for _, _, t in ik)),
                "ik_err_max_mm": _r(1000 * max(e for _, e, _ in ik)), "grasp_z_mm": _r(1000 * (g[2] - self.zc)),
                "grasp_xy": [_r(v, 4) for v in g[:2]], "place_shift_mm": _r(1000 * np.linalg.norm(d[:2] - g[:2])),
                "plan_s": _r(p.duration, 2)}

    def _prefetch(self, case, start, prev, det=None):
        if case is None:
            self.job = None
            return
        det = det if det is not None else self._det_pred(case)
        job = {"case": case, "epoch": self.epoch}
        job["thread"] = threading.Thread(target=lambda: job.update(self._plan_job(case, det, start, prev)), daemon=True)
        self.job = job
        job["thread"].start()

    def _wait_job(self, hold=True):
        j, self.job = self.job, None
        if j is None:
            return None
        while j["thread"].is_alive():
            if hold and self.env is not None and self.env.realtime:
                self._hold(1)             # sim: keep commanding while the planner runs
            else:
                j["thread"].join()        # mock (virtual clock) / plan-only: planning takes no sim time
        return j

    def _take_job(self, case, det=None):
        j = self._wait_job()
        if (det is not None or j is None or j["case"] is not case or j["epoch"] != self.epoch
                or not (np.allclose(j["start"][0], self.last[0], atol=1e-6)
                        and np.allclose(j["start"][1], self.last[1], atol=1e-6))):
            self._prefetch(case, self.last, self.prev, det)
            j = self._wait_job()
        return j

    # ------------------------------------------------------------------ one case
    def _row0(self, i, c):
        a = self.a
        return {"type": "case", "label": a.label, "i": i, "shape": c["shape"], "color": c["color"], "yaw": c["yaw"],
                "x": round(c["x"], 3), "y": round(c["y"], 3), "rep": c["rep"], "arm_req": a.arm,
                "profile": getattr(a, "profile", None) or "oracle",
                "face_align": bool(self.cfg["grasp"]["face_align"]["enabled"]), "tri_phi": self.cfg["grasp"]["face_align"]["triangle_phi_deg"],
                "finger_yaw_req": a.finger_yaw, "ff": self.ff is not None, "correct": self.correct,
                "xy_error_mm": [_r(1000 * v) for v in self._err(c)[:2]], "hold_s": a.hold, "outcome": None,
                "plan_ok": None, "plan_reason": "", "error": None}

    def plan_only_case(self, i, case):
        row = self._row0(i, case)
        j = self._plan_job(case, self._det_pred(case), self.last, self.prev)
        row.update(plan_ok=j["ok"], plan_reason=j["reason"], plan_wall_s=_r(j["wall_s"], 2), plan_tries=j["tries"],
                   outcome=None if j["ok"] else "plan")
        if j["ok"]:
            row.update(how=j["how"], **j["info"])
            self.last, self.prev = (j["A"][-1], j["H"][-1]), {"side": j["plan"].side, "tail": j["tail"]}
        return row

    def run_case(self, i, case, nxt):
        row = self._row0(i, case)
        self._t_case = t_case = self.now()
        w0 = time.monotonic()
        self.deadline = t_case + self.a.time_limit
        self.trace, self.clips, self.clip_reasons = [], 0, set()
        try:
            j = self._take_job(case)
            row["t_plan_wait"] = _r(self.now() - t_case, 2)
            row["plan_wall_s"] = _r(j.get("wall_s"), 2)
            row["plan_tries"] = j.get("tries", [])      # options that failed before the one used
            if not j["ok"]:
                row.update(plan_ok=False, plan_reason=j["reason"], outcome="plan")
                self._prefetch(nxt, self.last, self.prev)
                return row
            row.update(plan_ok=True, how=j["how"])
            if j["prefix"] is not None:    # previous plan's checked home tail first, then place the next object
                for a_, h_ in zip(*j["prefix"]):
                    self._step(a_, h_)
                self.prev = None
            t1 = self.now()
            self._name, self._main_xy = None, self.scene.mat_to_pelvis([case["x"], case["y"], 0.0])[:2]
            self.sc.reset([{"color": c["color"], "shape": c["shape"], "x": c["x"], "y": c["y"],
                            "yaw": float(np.radians(c["yaw"]))} for c in [case] + list(case.get("nb") or [])])
            o0 = self._settle(0.2, 2.0, "settle")
            self._nb0 = {}
            if case.get("nb"):
                self._obj(others=self._nb0)
            det = self._detect(case)
            if self._moved(det, j["det"]):     # e.g. DR pose jitter: plan again for the measured pose
                row["replanned"] = True
                j = self._take_job(case, det)
                if not j["ok"]:
                    row.update(plan_ok=False, plan_reason=j["reason"], outcome="plan")
                    self._prefetch(nxt, self.last, self.prev)
                    return row
                row.update(how=j["how"], plan_tries=j.get("tries", []))
                if j["prefix"] is not None:
                    for a_, h_ in zip(*j["prefix"]):
                        self._step(a_, h_)
                    self.prev = None
            row.update(j["info"])
            row["start"] = [_r(1000 * v) for v in o0["p"]]
            row["t_settle"] = _r(self.now() - t1, 2)
            self._prefetch(nxt, (j["A"][-1].copy(), j["H"][-1].copy()), {"side": j["plan"].side, "tail": j["tail"]})
            self._measure(j, row, o0)
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=6), outcome="error")
            self.log(f"[grasp_lab] case {i} error: {e!r}")
            row["recovery"] = self._recover()
        finally:
            self.deadline = None
            row.update(clips=self.clips, clip_reasons=sorted(self.clip_reasons), sim_s=_r(self.now() - t_case, 2),
                       wall_s=_r(time.monotonic() - w0, 1), trace=self.trace)
        if self.env.filter.clipped_steps and row["outcome"] != "error":    # clean filter (clip counter) again
            row["recovery"] = self._recover()
        return row

    def _sig(self, side, kind="abs7"):
        """verify.grasp_signal of `side`'s hand now (abs7 = max |measured - commanded|; finger_shortfall = the closing
        shortfall of the 4 finger joints, the robot's signal)."""
        from aaltonomous.baselines.classical.verify import grasp_signal
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        return grasp_signal(self.env.backend.joints()[2][sl], self.env.filter.hand_cmd[sl], kind)

    def _gp(self, side, ref=None):
        """(measured, commanded) grasp point (FK of measured / commanded arm_q; ref: this arm_q as the command, e.g.
        the planned tick when the sent one is corrected)."""
        return (self.mkin.grasp_pose(side, self.env.backend.joints()[0])[0],
                self.mkin.grasp_pose(side, self.last[0] if ref is None else ref)[0])   # (intended: what was meant)

    def _correct(self, j, A, k, row):
        """Settle and correct at body tick k (the blend-in starts here): the measured error (cmd - meas, mm, pelvis
        + along / across the fingers), the shift (Planner.correct on the rest of the body, 0 clips etc.) unless
        --correct dwell. Returns the body to send."""
        side, P, cc = j["plan"].side, self.cplanner, self.cfg["grasp"]["correct"]
        t0 = time.perf_counter()
        arm_q, dq_m, _ = self.env.backend.joints()
        last_a, last_h = self.last[0].copy(), self.last[1].copy()     # intended (sag feed-forward: not what was sent)
        from aaltonomous.baselines.classical.policy import ClassicalPolicy
        e, dq, f, res = P.correction_dq(side, last_a, arm_q)
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        speed, err = float(np.abs(dq_m[sl]).max()), float(np.linalg.norm(e))
        why = ("dwell only (--correct dwell)" if self.correct == "dwell"
               else ClassicalPolicy._correct_guard(cc, err, res, speed))
        out = A
        if why is None:   # the lab's body names: hold / place = carry at the counter (relocation rules)
            ren = {"place": "carry", "hold": "carry"}
            phases = [(ren.get(n, n), s - k, e_ - k) for n, s, e_ in j["phases"] if e_ > k]
            new, info = P.correct(side, last_a, last_h, A[k:], j["H"][k:], dq, phases, place=True)
            if new is None:
                why = info
            else:
                out = np.vstack([A[:k], new])
        r = lambda v: [_r(1000 * x) for x in v]
        row["corr"] = {"err_mm": r(e), "err_norm_mm": _r(1000 * err), "along_across_up_mm": r([e @ f, e @ [-f[1], f[0], 0],
                       e[2]]), "dq_max_mrad": _r(1000 * np.abs(dq).max()), "res_mm": _r(1000 * res, 2),
                       "applied": why is None, "why": why,
                       "speed": _r(speed, 3), "ms": _r(1e3 * (time.perf_counter() - t0), 1)}
        # before any shift: what the sag feed-forward (--ff) alone leaves at the end of the settle (planned - measured)
        row.update(err_pre_corr_mm=r(e), err_pre_corr_norm_mm=_r(1000 * err),
                   err_pre_corr_along_across_up_mm=row["corr"]["along_across_up_mm"])
        return out

    def _measure(self, j, row, o0):
        A, Hh, side = j["A"], j["H"], j["plan"].side
        A0 = A                 # as planned: the reference of every error below (A = what is sent, maybe corrected)
        ph = {n: (s, e) for n, s, e in j["phases"]}
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        cs, ce = ph["close"]
        kc = next((k for k in range(cs, ce) if np.allclose(Hh[k][sl], self.post["closed"][sl], atol=1e-9)), ce - 1)
        from aaltonomous.baselines.classical.planner import correct_ticks
        k_corr = (cs - correct_ticks(self.cfg["grasp"]["correct"], self.dt)[1]
                  if self.cplanner is not None and "settle" in ph else None)
        marks = {}
        k_err = min(cs + CLOSE_ERR_TICKS, kc)      # the fingers moving, not yet on the object; blend-in caught up
        for tag, k in (("descend_end", ph["descend"][1] - 1), ("pre_close", k_err), ("close_end", ce - 1),
                       ("lift_end", ph["lift"][1] - 1), ("hold_end", ph["hold"][1] - 1),
                       ("sig_1s", kc + int(round(SIG_DELAY_S * HZ)))):
            marks.setdefault(min(k, len(A) - 1), []).append(tag)
        names = np.empty(len(A), object)
        for n, s, e in j["phases"]:
            names[s:e] = n
        every = max(1, int(round(HZ / self.a.sample_hz)))
        got = {}
        t_exec = self.now()
        for k in range(len(A)):
            if k == k_corr:
                A = self._correct(j, A, k, row)
            self._step(A[k], Hh[k])
            tags = marks.get(k, [])
            for t in tags:
                if t in ("descend_end", "pre_close", "close_end"):
                    got[t] = self._gp(side, A0[k])
                    if t == "pre_close":
                        self._close_q = self.env.backend.joints()[0].copy()
                elif t == "sig_1s":     # + signed per-joint residual (LEFT order for the left hand, DDS order right)
                    row["sig_1s"] = _r(self._sig(side), 4)
                    row["sig_fs_1s"] = _r(self._sig(side, "finger_shortfall"), 4)
                    row["sig_1s_joints"] = [_r(v, 4) for v in self.env.backend.joints()[2][sl]
                                            - self.env.filter.hand_cmd[sl]]
                elif t == "hold_end":
                    row["sig_hold"] = _r(self._sig(side), 4)
                    row["sig_fs_hold"] = _r(self._sig(side, "finger_shortfall"), 4)
                    got["gp_hold"] = self._gp(side)[0]
            if tags or k % every == 0 or k == len(A) - 1:
                o = self._obj(names[k], side)
                for t in tags:
                    got[t + "_obj"] = o
                if k == len(A) - 1:
                    got["retreat_obj"] = o
        row["t_exec"] = _r(self.now() - t_exec, 2)
        # the arm's joints (deg, LEFT-arm convention: the right arm x kinematics.ARM_MIRROR), planned per phase
        # (min, max) and measured at the close: what the plan asks of the wrist (W1, real-arm joint ranges)
        from aaltonomous.baselines.classical.kinematics import ARM_MIRROR
        mir = 1.0 if side == "left" else ARM_MIRROR
        row["joints_deg"] = {n: [[_r(v, 0) for v in np.degrees(A0[s:e, sl] * mir).min(0)],
                                 [_r(v, 0) for v in np.degrees(A0[s:e, sl] * mir).max(0)]]
                             for n, s, e in j["phases"] if e > s}
        if "pre_close" in got:
            row["joints_close_meas_deg"] = [_r(v, 0) for v in np.degrees(self._close_q[sl] * mir)]
        self.prev = {"side": side, "tail": j["tail"]}
        t_post = self.now()
        of = self._settle(0.2, 1.5, "post")
        row["t_post"] = _r(self.now() - t_post, 2)
        if self._nb0:   # --neighbour: how far the other objects were pushed / tipped by this grasp
            nb1 = {}
            self._obj(others=nb1)
            d = [(float(np.linalg.norm(nb1[n]["p"][:2] - p0["p"][:2])), nb1[n]["tilt"]) for n, p0 in self._nb0.items()
                 if n in nb1]
            row["nb_disp_mm"] = _r(1000 * max(x for x, _ in d)) if d else None
            row["nb_tilt_deg"] = _r(max(t for _, t in d)) if d else None
            row["nb_planned"] = bool((self.cfg.get("neighbours") or {}).get("enabled"))

        rest = o0["p"]
        def rise(o):
            return None if o is None else float(o["p"][2] - rest[2])
        m, c = got.get("descend_end", (None, None))
        if m is not None:
            row["pos_err_mm"] = _r(1000 * np.linalg.norm((m - c)[:2]))
            row["self_lift0_mm"] = _r(1000 * (m - c)[2])
        m, c = got.get("pre_close", (None, None))
        if m is not None:   # planned - measured grasp point as the fingers start to close (the A/B number)
            self.mkin._fk(A0[k_err])
            f = self.mkin.data.oMf[self.mkin.palm[side]].rotation[:, 0] * [1, 1, 0]
            f, e = f / np.linalg.norm(f), c - m
            row["err_close_mm"] = [_r(1000 * v) for v in e]
            row["err_close_norm_mm"] = _r(1000 * np.linalg.norm(e))
            row["err_close_along_across_up_mm"] = [_r(1000 * v) for v in (e @ f, e @ [-f[1], f[0], 0], e[2])]
        m, c = got.get("close_end", (None, None))
        if m is not None:
            row["self_lift_mm"] = _r(1000 * (m - c)[2])
        rl, rh = rise(got.get("lift_end_obj")), rise(got.get("hold_end_obj"))
        oh = got.get("hold_end_obj")
        gap = None if oh is None or "gp_hold" not in got else float(np.linalg.norm(oh["p"] - got["gp_hold"]))
        rr, rf = rise(got.get("retreat_obj")), rise(of)
        ups = [r[4] for r in self.trace if r[1] in ("lift", "hold") and r[4] is not None]
        mm = lambda v: None if v is None else _r(1000 * v)
        row.update(rise_lift_mm=mm(rl), rise_hold_mm=mm(rh), hold_gap_mm=mm(gap), carried_mm=mm(rr),
                   max_rise_mm=_r(max(ups) - 1000 * rest[2]) if ups else None)
        row["lifted"] = rl is not None and rl >= LIFT_MIN
        row["held"] = rh is not None and rh >= LIFT_MIN and gap is not None and gap < HELD_GAP
        if of is None:
            row.update(released=False, off_counter=True)
        else:
            on = self.scene.on_counter(self.scene.pelvis_to_mat(of["p"]))
            row["released"] = bool(on and abs(rf) < RELEASE_TOL and of["speed"] < self.rest_speed
                                   and (rr is None or rr < CARRY_TOL))
            g, d = j["plan"].waypoints["grasp"], j["plan"].waypoints["drop"]
            row.update(final=[_r(1000 * v) for v in of["p"]], off_counter=not on, tilt_deg=_r(of["tilt"]),
                       tipped=of["tilt"] > TIP_DEG, displaced_mm=_r(1000 * np.linalg.norm(of["p"][:2] - rest[:2])),
                       displaced_vs_cmd_mm=_r(1000 * np.linalg.norm(of["p"][:2] - rest[:2] - (d[:2] - g[:2]))))
        row["outcome"] = ("ok" if row["lifted"] and row["held"] and row["released"] else
                          "lift" if not row["lifted"] else "hold" if not row["held"] else "release")

    # ------------------------------------------------------------------ run
    def run(self, cases, out=None, meta=None):
        f = open(out, "a") if out else None
        rows = self.rows = []      # kept on the lab: main() prints the table of an interrupted run too

        def write(r):
            if f:
                f.write(json.dumps(r, default=ev._js) + "\n")
                f.flush()
        try:
            write({"type": "meta", **(meta or {})})
            if self.a.plan_only:
                for i, c in enumerate(cases):
                    rows.append(self.plan_only_case(i, c))
                    write(rows[-1])
                    self.log(_line(rows[-1]))
                return rows
            self.home()
            self._prefetch(cases[0] if cases else None, self.last, None)
            for i, c in enumerate(cases):
                rows.append(self.run_case(i, c, cases[i + 1] if i + 1 < len(cases) else None))
                write(rows[-1])
                self.log(_line(rows[-1]))
        finally:
            if self.env is not None and not self.a.plan_only:
                self._wait_job(hold=False)
                try:                       # low hand up, park the objects + arms home (home_arms), filter clean
                    self._rearm()
                    self._raise_hands()
                    self.home()
                except Exception as e:
                    self.log(f"[grasp_lab] final homing failed: {e!r}")
            if f:
                f.close()
        return rows


# ---------------------------------------------------------------------- cases, table, CLI
def make_cases(a, scene):
    """Position-major (consecutive cases at one spot: same arm, short direct moves); colours cycle."""
    L = scene.cfg["layout"]
    xs = _floats(a.xs) or [float(v) for v in np.linspace(*L["region_x"], 4)]
    ys = _floats(a.ys) or [float(np.mean(L["region_y"]))]
    colors = itertools.cycle(a.colors.split(","))
    nbs = [_floats(s) for s in (getattr(a, "neighbour", None) or "").split(";") if s.strip()]
    out = []
    for x, y in itertools.product(xs, ys):
        for shape in a.shapes.split(","):
            for yaw in _floats(a.yaws):
                for r in range(a.repeats):
                    c = {"shape": shape, "color": next(colors), "yaw": yaw, "x": x, "y": y, "rep": r}
                    if nbs:
                        c["nb"] = [{"shape": a.neighbour_shape, "color": next(colors), "yaw": 0.0,
                                    "x": x + dx / 1000, "y": y + dy / 1000} for dx, dy in nbs]
                    out.append(c)
    return out


def _line(r):
    head = f"{r['i']:3d} {r['shape']:8s} yaw={r['yaw']:+4.0f} x={r['x']:.2f} y={r['y']:.2f}"
    if r.get("error"):
        return f"{head} ERROR {r['error']}"
    if not r.get("plan_ok"):
        return f"{head} PLAN FAIL {r['plan_reason'][:150]}"
    if "lifted" not in r:   # plan only
        return f"{head} {r['side']:5s} {r['how']:8s} planned {r['exec_plan_s']} s  finger {r['finger_yaw_deg']} deg"
    yn = lambda k: "Y" if r.get(k) else "n"
    c = r.get("corr")
    corr = "" if c is None else f"corr {c['err_norm_mm']} mm {'applied' if c['applied'] else 'no: ' + str(c['why'])[:40]}  "
    return (f"{head} {r['side']:5s} {r['how']:8s} lift {yn('lifted')} held {yn('held')} rel {yn('released')}  "
            f"sig {r.get('sig_1s')}/{r.get('sig_hold')}  {corr}err@close {r.get('err_close_norm_mm')} mm  "
            f"self_lift {r.get('self_lift_mm')} mm  disp {r.get('displaced_mm')} mm  tilt {r.get('tilt_deg')}  "
            f"{r['sim_s']} s sim  [{r['outcome']}]")


def print_table(rows, out=print):
    cs = [r for r in rows if r.get("type") == "case"]
    if not cs:
        out("no cases")
        return
    groups = {}
    for r in cs:
        groups.setdefault(r["shape"], []).append(r)
        groups.setdefault(f"{r['shape']}/{r.get('side') or '-'}", []).append(r)
    executed = any("lifted" in r for r in cs)
    out(f"\n  {'group':16s} {'n':>3s} {'plan':>4s} " + (f"{'lift':>4s} {'held':>4s} {'rel':>4s} {'ok':>4s} "
        f"{'sig_1s':>7s} {'sim_s':>6s} {'err@close':>9s} {'corr':>4s} {'pre_corr':>8s}" if executed
        else f"{'exec_plan_s':>11s}"))
    for k in sorted(groups, key=lambda k: (k.split("/")[0], "/" in k, k)):
        g = groups[k]
        n = lambda f: sum(bool(r.get(f)) for r in g)
        if not executed:
            d = [r["exec_plan_s"] for r in g if r.get("exec_plan_s") is not None]
            out(f"  {k:16s} {len(g):3d} {n('plan_ok'):4d} {np.mean(d) if d else float('nan'):11.1f}")
            continue
        sig = [r["sig_1s"] for r in g if r.get("lifted") and r.get("sig_1s") is not None]
        sim = [r["sim_s"] for r in g if r.get("sim_s") is not None]
        ec = [r["err_close_norm_mm"] for r in g if r.get("err_close_norm_mm") is not None]
        pc = [r["err_pre_corr_norm_mm"] for r in g if r.get("err_pre_corr_norm_mm") is not None]
        out(f"  {k:16s} {len(g):3d} {n('plan_ok'):4d} {n('lifted'):4d} {n('held'):4d} {n('released'):4d} "
            f"{sum(r.get('outcome') == 'ok' for r in g):4d} {np.mean(sig) if sig else float('nan'):7.3f} "
            f"{np.mean(sim) if sim else float('nan'):6.1f} {np.mean(ec) if ec else float('nan'):9.1f} "
            f"{sum(bool((r.get('corr') or {}).get('applied')) for r in g):4d} "
            f"{np.mean(pc) if pc else float('nan'):8.1f}")
    bad = [r for r in cs if r.get("outcome") not in ("ok", None)]
    if bad:
        out("  failed cases:")
        for r in bad:
            why = r.get("error") or (r["plan_reason"][:120] if r["outcome"] == "plan" else
                                     f"rise_lift {r.get('rise_lift_mm')} rise_hold {r.get('rise_hold_mm')} carried "
                                     f"{r.get('carried_mm')} disp {r.get('displaced_mm')} tilt {r.get('tilt_deg')}")
            out(f"    {r['i']:3d} {r['shape']:8s} yaw={r['yaw']:+4.0f} x={r['x']:.2f} y={r['y']:.2f} "
                f"{r.get('side') or '-':5s} {r['outcome']:7s} {why}")


def build_cfg(a):
    """--profile oracle (default): ORACLE_OVERRIDES + --face-align / --tri-phi (grasp.face_align.enabled /
    .triangle_phi_deg; the oracle's default phi: $AALTO_TRI_PHI or the sim profile's) + --set (dotted keys, applied
    last) through load_config. --profile base: the classical config's BASE values (the robot profile, `none`) instead
    of ORACLE_OVERRIDES; --profile <name>: that config profile (profiles.<name>); the flags then override the
    profile's values only where given (--face-align / --no-face-align, --tri-phi, --correct, --ff), --set last."""
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.sim import oracle
    user = json.loads(a.set) if a.set else {}
    for k, v in user.items():
        if k.startswith("hand.") and k != "hand.right" and len(v) != 7:
            raise ValueError(f"{k}: a hand posture has 7 values (LEFT order), got {len(v)}")
    prof = getattr(a, "profile", None) or "oracle"
    if prof != "oracle":
        ov = {}
        if a.face_align is not None:
            ov["grasp.face_align.enabled"] = bool(a.face_align)
        if getattr(a, "correct", None):
            ov["grasp.correct.enabled"] = a.correct != "off"
        if getattr(a, "ff", None):
            ov["sag.enabled"] = a.ff == "on"
        if a.tri_phi is not None:
            ov["grasp.face_align.triangle_phi_deg"] = float(a.tri_phi)
        ov.update(user)
        return load_config(profile="none" if prof == "base" else prof, overrides=ov), user
    ov = dict(oracle.ORACLE_OVERRIDES)
    ov["grasp.face_align.enabled"] = True if a.face_align is None else bool(a.face_align)
    if getattr(a, "correct", None):
        ov["grasp.correct.enabled"] = a.correct != "off"
    if getattr(a, "ff", None):
        ov["sag.enabled"] = a.ff == "on"
    if a.tri_phi is not None:
        ov["grasp.face_align.triangle_phi_deg"] = float(a.tri_phi)
    ov.update(user)
    return load_config(overrides=ov), user


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="cube,cylinder,triangle")
    ap.add_argument("--colors", default="red,yellow,blue", help="cycled over the cases")
    ap.add_argument("--yaws", default="-45,-20,0,20,45", help="deg, mat frame (layout yaw)")
    ap.add_argument("--xs", default=None, help="mat x (m); default 4 points across layout.region_x")
    ap.add_argument("--ys", default=None, help="mat y (m); default the middle of layout.region_y")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--arm", default="auto", choices=["auto", "left", "right"],
                    help="auto: object side first, then the other")
    ap.add_argument("--profile", default="oracle",
                    help="configuration: oracle (default) = ORACLE_OVERRIDES (the sim profile's grasp, pinned); base = "
                         "the classical config's base values (the real robot's profile, `none`); any other name = "
                         "that config profile")
    ap.add_argument("--set", default=None, help="JSON {dotted.key: value} over the --profile configuration "
                                                "(hand.* = LEFT order)")
    ap.add_argument("--face-align", action=argparse.BooleanOptionalAction, default=None,
                    help="cube/triangle: oracle face-aligned finger direction (else stock finger hints only); "
                         "default: on with --profile oracle, the profile's grasp.face_align.enabled otherwise")
    ap.add_argument("--tri-phi", type=float, default=None,
                    help="deg, finger direction vs triangle apex (oracle default: AALTO_TRI_PHI or the sim profile's triangle_phi_deg)")
    ap.add_argument("--finger-yaw", type=float, default=None, help="deg, force the finger direction (pelvis frame)")
    ap.add_argument("--correct", default=None, choices=["on", "off", "dwell"],
                    help="settle and correct (grasp.correct): on = shift applied, off = no settle phase (plan as "
                         "before), dwell = settle phase + measurement, no shift; default: the config's grasp.correct")
    ap.add_argument("--ff", default=None, choices=["on", "off"],
                    help="sag feed-forward (config sag, the sim prior fixed): every tick sent as intended + the "
                         "gravity-sag offset; default: the config's sag.enabled (oracle / sim profile: on)")
    ap.add_argument("--neighbour", default=None, metavar="DX,DY[;DX,DY]",
                    help="mm, MAT frame (x = to the robot's left, y = away from it): place another object (yaw 0) "
                         "this far from every case object; rows get nb_disp_mm / nb_tilt_deg (how far the grasp "
                         "pushed / tipped it). Planned around only with neighbours.enabled (--set)")
    ap.add_argument("--neighbour-shape", default="cube")
    ap.add_argument("--xy-error", default="0,0", help="DX,DY mm added to the detected top centre (pelvis frame)")
    ap.add_argument("--place-offset", default="0,0", help="DX,DY mm: put down here relative to the grasp point")
    ap.add_argument("--hold", type=float, default=1.5, help="s at carry height")
    ap.add_argument("--sample-hz", type=float, default=5.0, help="object pose samples per sim second")
    ap.add_argument("--time-limit", type=float, default=40.0, help="sim s per case (safety net)")
    ap.add_argument("--randomize", type=int, default=None, metavar="SEED",
                    help="sim started with AALTO_DR: apply that config's parameter set for this seed once before the "
                         "sweep (e.g. a yaml that fixes objects.friction_scale / mass_scale); recorded in the meta line")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="JSON Lines file (appended)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mock", action="store_true", help="offline mock world instead of the sim")
    ap.add_argument("--mock-sag", action="store_true", help="mock arms sag like the sim's (mock_world.SIM_SAG_GAIN)")
    ap.add_argument("--plan-only", action="store_true", help="plan every case (chained), no execution, no sim")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    cfg, user = build_cfg(a)
    env = sc = None
    if a.mock:
        env, sc = make_mock(cfg, sag=a.mock_sag)
    elif not a.plan_only:
        ev.apply_instance_env()
        from aaltonomous.interface.env import AaltoEnv
        from aaltonomous.sim.scene_client import SceneClient
        sc = SceneClient(a.host, timeout_ms=2000)
        env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=1e9, realtime=True, img_server_ip=a.host,
                       head_rgbd=False, cameras=False, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))   # planner thread holds the GIL: give ticks a chance
    lab = None
    try:
        lab = Lab(a, cfg, env, _Retry(sc) if sc is not None else None)
        cases = make_cases(a, lab.scene)
        info = None
        if sc is not None:
            try:
                info = lab.sc.info()
            except Exception as e:
                info = {"error": repr(e)}
        dr = None
        if sc is not None and a.randomize is not None and not a.mock:
            dr = lab.sc.randomize(a.randomize).get("params")
            objs = (dr or {}).get("objects") or {}
            print(f"[grasp_lab] randomize seed {a.randomize}: " + (f"friction x{sorted({round(o['friction_scale'], 2) for o in objs.values()})} "
                  f"mass x{sorted({round(o['mass_scale'], 2) for o in objs.values()})}" if objs else "sim runs without AALTO_DR: nothing applied"))
        meta = {"label": a.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:] if argv is None
                else list(argv), "set": user, "args": vars(a),
                "tri_phi": cfg["grasp"]["face_align"]["triangle_phi_deg"], "sim": info, "dr_params": dr,
                "n_cases": len(cases), "profile": a.profile,
                "cfg": {k: cfg[k] for k in ("grasp", "hand", "motion", "ik", "sag", "verify", "home_arm_q")}}
        lab.run(cases, a.out, meta)
    finally:
        if env is not None:
            env.close()
        if sc is not None:
            sc.close()
        rows = getattr(lab, "rows", [])
        print_table(rows)
    return rows


def _sigterm(*_):
    raise KeyboardInterrupt   # queue kill -> the same clean exit as Ctrl-C (park, home, close)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    code = 0
    try:
        main()
    except KeyboardInterrupt:
        code = 130
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)     # DDS / zmq threads may not stop on their own
