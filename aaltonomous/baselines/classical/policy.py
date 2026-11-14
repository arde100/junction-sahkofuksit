"""ClassicalPolicy: obs -> action wrapper around perception + planner (rungs 1-2: sort_fixed, sort_language).

    pol = ClassicalPolicy()                   # config.yaml, scene.yaml, calibration.yaml
    pol.reset({"instruction": obs["instruction"]})   # mapping read from the text (instruction.py); logged
    while True:
        action = pol.act(obs)                 # {"arm_q_target": (k,14), "hand_q_target": (k,14), "done": bool}
        obs, info = env.step(action)

Behaviour: go home -> settle -> look (head RGB-D) -> plan one object -> execute -> back home -> look again ...
until nothing sortable is left (then `done`). While it is planning (background thread when `threaded=True`) or
waiting for an operator confirm, act() returns 1-tick HOLD actions (the last target), so the env's watchdog
never fires. Targets are absolute, planned at <= motion.arm_max_vel and pre-checked against a SafetyFilter.

Target bins (reset / bin_for): an explicit info["mapping"] wins, else the instruction is parsed (instruction.py:
colour / shape / (colour, shape) / "default" keys, scene.resolve_bin rules); no instruction -> the fixed mapping
(task.fixed_mapping). The mapping's colour words are matched to the world's colour classes (colors.py: "red" on the
robot = its orange objects, "purple" in the sim = its blue ones) and perception classifies into those classes; after
every look the blobs' own hues calibrate them (colors.Calibrator via _calibrate: "orange" and "yellow" are the lower
and the higher cluster whatever the prior hues say; logged as `colour calibration: ...`). An
object gets a bin only if every colour / shape it may have (shape None, colour "unknown" = any) leads to the same bin;
otherwise, or if its key was not understood, it stays on the counter (0 points, not a -1 guess) and is looked at again
in the next looks. Nothing understood -> nothing is sorted (the reset line in the log says so).

Rung 3 (insert.py select_mode at reset: task id insert_shapes, or a lid / hole instruction): no object gets a bin (the
lid lies on the left bin: a drop there is -1); the first look logs which holes an arm could reach (none in this
scene: every object stays on the counter, 0 points).

Objects are tracked per object across looks (_track: same colour near the last known / attempted position), with
a cap of policy.max_attempts tries each; untried objects go first. Grasp verification (config `verify`, sim
profile): while a plan executes, GraspMonitor (verify.py) reads the measured finger joints once per act(); on a
verified miss or slip the rest of that arm's queue becomes decelerate + joint-space move home (Planner.abort, own
kinematics instance: act() runs in the control thread), and the retry uses the next grasp variant.

Paired cycle (config two_arm, sim profile): after a look finds its plan as above, a second object is planned for the
other arm from the same start pose and both run from one queue (Planner.plan_pair: the second arm delayed until the
arms' sphere model keeps two_arm.min_clearance); verification and abort stay per arm, and an abort's new queue must
also keep that clearance from the other arm. No pair possible -> the single plan, as before.

Parked arm (config two_arm.avoid_parked, sim profile): a plan executed with one arm (no pair) is checked against the
other arm waiting at its pose with the same model and clearance (Planner.keep_clear). Too close -> that arm moves aside
to a two_arm.clear_poses pose and back within the same queue (Planner.evade), else the object is skipped as a failed
plan. Aborts of such plans are checked against the other arm's queued motion as for a pair.

Settle and correct (grasp.correct): every pick dwells at the grasp pose (planner phase `settle`). At its measurement
tick (the chunk before it is cut short so that an act() lands there) the measured arm joints give the grasp point's
tracking error (FK(commanded) - FK(measured): the PD arm sags without gravity compensation); Planner.correct shifts
that arm's queued ticks to cancel it (blended in before the close, out after the lift starts; the other arm's
columns untouched) if every guard passes, else the queue stays as planned. Logged as `correct <side>: ...`.

Drop slots (grasp.slots): a delivered object (grasp verified held until the release; verification off: release
executed) records its release point per bin (drops_at); the next drop into that bin takes the slot farthest from them
(the second plan of a same-bin pair also from the first's). Failed / aborted / slipped attempts use no slot.

Boxes (perception.boxes, bins.py): with mode on every look's boxes (detected, fused over the episode; reset per
episode) go to every planner (_use_boxes); the arms' sphere model at the measured pose is the detection's self-mask
(_self_mask). An object whose box is not found or not certainly named is left on the counter for now (not failed);
when only such objects remain, perception.boxes.relook more looks before `done`. Logged in the look line ('boxes').

Neighbours (config neighbours): each candidate is planned with every other detection as an obstacle. Blocked (no
clear grasp candidate) -> the next candidate; the object is not marked failed but remembers the scene it was blocked
in (planned again once another object moved / left). No other object plannable -> least-bad pass over the blocked
ones (Planner.plan(least_bad=True)). Candidate order: untried first, then (neighbours.order_free) objects with free
space around them, then policy.order.

Operator confirm (real robot): ClassicalPolicy(require_confirm=True) asks before the first move to home and before
each object; the runner calls confirm() / skip() / stop() while `pol.waiting` is True (`pol.pending_text` says what).

Sag feed-forward (config sag, sag.py): arm_cmd, the queues, every plan, abort, correction and check are INTENDED joint
space (where the arm should really be). The feed-forward is added only where targets leave the policy (_pop / _hold:
out_cmd = the last commanded tick) and the estimator compares measurements with those commands, so "tracking error"
(correction, verification) means measured - intended. The first act() takes the measured pose (or the probe's last
command) as the command in force and starts from the intended pose that command yields (no step). The applied gain is
frozen while a plan is planned or executed; the estimate it learns (still samples: the detector's probe via
feed_probe(), the settle + look holds at home, the settle before every grasp) is blended in at the start of the next
home settle (sag.blend_vel; that settle is longer by the blend). Logged once per episode and on material changes.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from .colors import Calibrator, calibrate_look, match_colors
from .config import fixed_mapping, hand_postures, home_arm, load_config
from .insert import look_note as insert_look_note
from .insert import select_mode as select_insert_mode
from .instruction import mapping_colors, object_bin, parse_instruction
from .kinematics import ArmKinematics
from .perception import Perception, world_classes
from .planner import PAIR_CHECK_STEP, Planner, checked_masks, correct_ticks, held_masks
from .sag import SagFF
from .still import StillGate
from .verify import EMPTY, HELD, GraspMonitor, grasp_window

HZ = 30.0
MAX_ABORT_TRIES = 3  # per grasp: an abort rejected by a check (e.g. still lifting, too low) is tried again later


def _mapping_text(m):
    return ", ".join(f"{' '.join(k) if isinstance(k, tuple) else k}->{v}" for k, v in m.items())


def _lower_thread_priority(nice=10):
    """Linux: raise the nice value of the calling thread only (setpriority on its native thread id)."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except (AttributeError, OSError):
        pass


class ClassicalPolicy:
    def __init__(self, config=None, scene=None, calibration=None, safety: SafetyConfig | None = None,
                 require_confirm=False, threaded=False, time_limit=120.0, log=print, overrides=None):
        self.cfg = config if isinstance(config, dict) else load_config(config, overrides)
        self.scene = scene if isinstance(scene, Scene) else Scene.load(scene)
        self.cal = calibration if isinstance(calibration, Calibration) else Calibration.load(calibration)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self.home = home_arm(self.cfg, self.spec)
        self.kin = ArmKinematics(self.cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=1.0 / HZ)
        self.perception = Perception(self.cfg, self.scene, self.cal)
        # stillness gate before a look (config still; still.py): the head camera moves with the body
        self.gate = StillGate(self.cfg["still"], self.cal) if (self.cfg.get("still") or {}).get("enabled") else None
        self._gate_ok = True
        self.world = world_classes(self.cfg["perception"], self.scene)   # colour classes + reject hues (measured)
        self.planner = Planner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
        L = self.scene.cfg["layout"]   # farthest object start (pelvis x = mat y): beyond it, objects get a reach test
        self._band_x = float(self.scene.mat_to_pelvis([0.0, L["region_y"][1], 0.0])[0])
        hc = self.cal.cams.get("head")   # grasp.far prefers landings the head camera sees (the next look confirms them)
        if hc is not None and hc.T_pelvis_optical is not None:
            self.planner.far.camera = (hc.K, self.cal._T("head"), hc.width, hc.height)
        vc = self.cfg.get("verify") or {}
        self.verify_on = bool(vc.get("enabled"))
        # The abort is computed inside act() (control thread) while a planning thread may be using self.planner /
        # self.kin (pinocchio data is not thread-safe): own kinematics + planner, dry-run filter built here, not then.
        self.abort_planner = None
        if self.verify_on and vc.get("abort", True):
            self.abort_planner = Planner(self.cfg, self.scene, self.post, self.home, ArmKinematics(self.cfg, self.spec),
                                         self.safety, self.spec)
            self.abort_planner.dry_run_safety(self.home, self.post["open"], self.home[None], self.post["open"][None])
        # settle and correct (grasp.correct) also runs in act(): the abort's instances, or its own built the same way
        self.correct_on = bool((self.cfg["grasp"].get("correct") or {}).get("enabled"))
        self.act_planner = self.abort_planner
        sag_on = bool((self.cfg.get("sag") or {}).get("enabled"))   # (its gravity in act(): this thread's instance)
        if (self.correct_on or sag_on) and self.act_planner is None:
            self.act_planner = Planner(self.cfg, self.scene, self.post, self.home, ArmKinematics(self.cfg, self.spec),
                                       self.safety, self.spec)
            self.act_planner.dry_run_safety(self.home, self.post["open"], self.home[None], self.post["open"][None])
        # sag feed-forward: one shared state; gravity in act() with an act()-thread kinematics instance
        self.ff = SagFF(self.cfg) if sag_on else None
        self.act_kin = None
        if self.ff is not None:
            self.attach_ff(self.planner, self.abort_planner, self.act_planner)
            self.act_kin = self.act_planner.kin
            sp, m = self.spec, float(self.safety.arm_pos_margin)   # the filter's (silent) position band: _ff_net
            m = np.minimum(m, 0.25 * (sp.arm_upper - sp.arm_lower))
            self._out_lim = (sp.arm_lower + m, sp.arm_upper - m)
            self.ff_net = 0     # commanded ticks the output net had to change (planned ones never need it)
        if self._avoid_parked():
            self.planner.clear_poses()  # bad clear poses fail here, not on the first conflict
        self.planner.prepare()          # drop slots (reach tests) now, not in the first look
        self.require_confirm = require_confirm
        self.threaded = threaded
        if threaded:  # pure-Python planning holds the GIL: hand it back to act() sooner (default 5 ms)
            sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
        self.time_limit = time_limit
        self.log = log or (lambda *a: None)
        if self.ff is not None:   # IK bounds + max_offset inside the filter's position band? (else the dry run refuses)
            room = np.minimum(self.kin.lo - self._out_lim[0], self._out_lim[1] - self.kin.hi) - self.ff.max_off
            if (room < -1e-9).any():
                self.log(f"[classical] sag feed-forward: the IK joint bounds leave less than sag.max_offset to the "
                         f"filter's position band on {[self.spec.arm_names[i] for i in np.flatnonzero(room < -1e-9)]}"
                         f" (plans that need it there fail their dry run: Planner.dry_run_safety)")
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = info or {}
        self.insert = select_insert_mode(info, self.cfg)   # rung 3 (insert.py): no bins; None = rungs 1-2
        self.mapping, self.mapping_source, self.instruction_parse, note = (
            self._mapping(info) if self.insert is None else ({}, f"insert:{self.insert.source}", None,
                                                            self.insert.note()))
        self.calib = self._calibrator(self.mapping, self.instruction_parse)  # calibrated from the looks (or None)
        self.colors = self.calib.ec if self.calib else self._colors(self.mapping, self.instruction_parse)
        if hasattr(self.perception, "set_classes"):                         # (the oracle's perception has none)
            self.perception.set_classes(self.colors.refs, self.colors.reject, self.colors.alias)
        self.planner.mapping = self.mapping
        self.time_limit = info.get("time_limit", self.time_limit)
        self.state = "init"
        self.queue_a = self.queue_h = None
        self.q_t = 0                # ticks popped from the current queue (plan tick index for the grasp monitor)
        self.arm_cmd = self.hand_cmd = None   # last popped tick, INTENDED (sag feed-forward: not what was sent)
        self.out_cmd = None         # last arm target sent (= arm_cmd + the feed-forward offset)
        self.const = np.zeros(2, int)   # per arm: ticks its sent target has been constant (still samples)
        self._t, self._resettles, self._ff_seen = 0.0, 0, None
        self.objs = []              # per-object records (_track): tries, verified misses / slips, deliveries, variants
        self.execs = {}             # side -> grasp being verified in the executing plan (_watch)
        self.exec_held = None       # executing pair: [left, right] per-tick "may hold an object" masks (abort check)
        self.exec_masks = None      # executing plan (any): the same masks (correction's arm-arm check)
        self.corrs = {}             # side -> pending / done settle-and-correct of the executing plan (_correct)
        self.corrections = []       # this episode's corrections: side, error, dq, applied / why, ms
        self.exec_parked = False    # executing a single plan with avoid_parked: aborts are checked vs the parked arm
        self.drops = {}             # bin -> drops started this episode (cycles grasp.drop_offsets)
        self.drops_at = {}          # bin -> release points (pelvis xy) of the objects delivered there (grasp.slots)
        self.far_watch = []         # grasp.far releases to check at the next look: (side, bin, expected landing xy,
                                    # colour, object record, carried out to the end)
        self.exec_drops = []        # executing plan: (side, bin, release xy) per pick -> bin part
        self.exec_nb = {}           # executing plan: side -> neighbour cylinders its plan was checked against (abort)
        self.pending = None         # Plan waiting for the operator
        self.pending_text = ""      # what the operator is asked to confirm
        self.waiting = False
        self.events = []
        self.plans = []
        self._job = None
        self._decision = None
        self.last_detections = []
        self._box_looks = 0                           # extra looks for boxes not found (perception.boxes.relook)
        self.look_s = []                              # wall s of each look (perception + planning)
        if hasattr(self.perception, "reset_boxes"):   # boxes are found anew every episode (they may have been moved)
            self.perception.reset_boxes()
            self._use_boxes()
        self._note(0.0, note)
        self._note(0.0, self.colors.summary())
        if self.ff is not None:     # (the estimate is kept across episodes: same arms, same process)
            self.ff.stretch = None
            if info.get("sag_probe") is not None:
                self.feed_probe(info["sag_probe"])

    def feed_probe(self, r):
        """The world detector's probe (autoprofile.WorldProbe result: r["sag"] = held arm command, measured arm /
        hand samples, arm_dq) -> one still sample for the sag estimate (both arms, if still), used from the first
        act() on. No-op without feed-forward or samples."""
        s = (r or {}).get("sag") if isinstance(r, dict) else None
        if self.ff is None or not s or not len(s.get("arm_q", ())):
            return None
        a, h = np.asarray(s["arm_q"], float).reshape(-1, 14), np.asarray(s["hand_q"], float).reshape(-1, 14)
        dq = np.asarray(s.get("arm_dq", np.zeros_like(a)), float).reshape(-1, 14)
        sp = float(self.ff.sc.get("still_speed", 0.05))
        ok = np.repeat([np.abs(dq[:, :7]).max() <= sp, np.abs(dq[:, 7:]).max() <= sp], 7)
        kin = self.act_kin
        self.ff.stretch = None
        for ai, hi in zip(a, h):
            self.ff.observe(kin, ai, np.asarray(s["arm_cmd"], float), hi, ok, "probe", 0.0)
        res = self.ff.flush(kin) if ok.any() else None
        self._ff_report(0.0, res)
        return res

    def _mapping(self, info):
        """This episode's target bins -> (mapping, source, Parse or None, log line). An explicit info["mapping"] wins
        (evaluator mock, tests); else info["instruction"] is parsed (instruction.py) and only what was understood is
        used (conflicting / negated keys dropped); no or empty instruction -> the fixed rung-1 mapping. Never the fixed
        mapping for an instruction that was not understood: under another mapping each guess scores -1, an object
        left on the counter 0. One exception, task sort_fixed on the robot profile: see below."""
        if info.get("mapping") is not None:
            m = dict(info["mapping"])
            return m, "given", None, f"mapping given: {_mapping_text(m) or 'empty: sorting nothing'}"
        text = (info.get("instruction") or "").strip()
        if not text:
            fm = fixed_mapping(self.cfg)
            return fm, "fixed", None, f"no instruction: fixed mapping {_mapping_text(fm)}"
        r = parse_instruction(text)
        # sort_fixed in a world whose fixed mapping is known (robot profile: task.fixed_mapping, the organisers'
        # statement): the text is followed only if it names this world's colours. A sentence in other colour words
        # (the simulator's "red ... left, yellow ... center, blue ... right", if the runner still sends it) is stale
        # wording of the same task, not another mapping: read literally, red would merge with orange and go left.
        fm = (self.cfg.get("task") or {}).get("fixed_mapping")
        if info.get("task") == "sort_fixed" and fm and (
                not r.mapping or any(c not in self.world[0] for c in mapping_colors(r.mapping))):
            fm = fixed_mapping(self.cfg)
            return fm, "fixed", None, (f"sort_fixed with instruction {text!r}: not in this world's colour words "
                                       f"-> fixed mapping {_mapping_text(fm)}")
        what = "" if r.mapping else " -> sorting nothing"
        return dict(r.mapping), "parsed", r, f"instruction {text!r}: {r.summary(self._colors(r.mapping).names)}{what}"

    def _colors(self, mapping, parse=None):
        """The episode's colour classes (colors.py): the mapping's colours matched to the world's classes
        (perception.color refs / reject; rules perception.color.match)."""
        return match_colors(mapping_colors(mapping), *self.world, (self.cfg["perception"].get("color") or {}).get(
            "match"), getattr(parse, "words", None))

    def _calibrator(self, mapping, parse=None):
        """colors.Calibrator for this episode (perception.color.calib), None if off."""
        c = self.cfg["perception"].get("color") or {}
        if not (c.get("calib") or {}).get("enabled"):
            return None
        return Calibrator(mapping_colors(mapping), *self.world, c.get("match"), c["calib"],
                          getattr(parse, "words", None))

    def _calibrate(self, dets, t):
        """After a look (planning thread): the blobs' hues calibrate the episode's colour classes (colors.Calibrator);
        if they changed, perception takes them and this look's detections are re-voted; detections of a cluster that
        two named colours could be get both names (a bin only where they agree). One log line when anything changed."""
        if self.calib is None or not hasattr(self.perception, "relabel"):
            return
        changed, line = calibrate_look(self.calib, self.perception, dets)
        if changed:
            self.colors = self.calib.ec
        if line:
            self._note(t, line)

    def bin_for(self, d):
        """Target bin of detection d: instruction.object_bin (scene.resolve_bin over every colour / shape the object
        may have: shape None or colour "unknown" count as any). None = no bin for now (its key was not understood, or
        the mapping needs the shape / colour and it is not recognised): it stays on the counter and is looked at
        again in later looks (neighbours gone, it may be recognised then). Never a guess."""
        return object_bin(self.mapping, d.color, getattr(d, "shape", None), self.colors.names)

    def confirm(self):
        self._decision = "go"

    def skip(self):
        self._decision = "skip"

    def stop(self):
        self._decision = "stop"

    def act(self, obs):
        if self.arm_cmd is None:  # first call: start from the measured pose
            # Clamp into the URDF limits: real Dex3 fingers at rest read slightly past 0 (2026-11-11), and holding
            # that measured pose would be clipped every tick -> safety abort before the first move.
            sp = self.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
            self.out_cmd = self.arm_cmd.copy()
            if self.ff is not None:   # that pose = the command in force: start from the pose it yields (no step)
                self.ff.set_applied(self.ff.target())
                self.arm_cmd = self.ff.intended(self.act_kin, self.out_cmd, self.hand_cmd)
                self.ff.logged = self.ff.applied.copy()
                ff, e = self.ff, self.ff.est
                self._note(float(obs.get("t", 0.0)), f"sag feed-forward: gain/prior {ff.text()} (estimate from "
                           f"{len(e.samples)} samples, {e.refused} refused; trust min {e.trusted().min():.2f}); "
                           f"offset max {1000 * np.abs(self.out_cmd - self.arm_cmd).max():.1f} mrad at the start pose")
        t = self._t = float(obs.get("t", 0.0))
        if self.ff is not None and self.ff.learn:
            self._ff_observe(obs, t)
        if self.queue_a is not None and len(self.queue_a):
            if self.corrs:
                self._correct(obs, t)
            if self.execs:
                self._watch(obs, t)
            return self._pop()
        if self.state == "init":
            P = self.planner if self.ff is None else self.act_planner   # (feed-forward timing: pinocchio in act())
            a, h = P.go_to(self.arm_cmd, self.hand_cmd, self.home, self.post["open"])
            self._home_move = (a, h)
            dq = np.abs(self.home - self.arm_cmd).max()
            self.pending_text = f"move both arms to home_arm_q (max joint change {dq:.2f} rad, {len(a) / HZ:.1f} s)"
            if self.require_confirm:
                self.state, self.waiting, self.pending = "await_home", True, None
                self._note(t, f"waiting for confirm: {self.pending_text}")
                return self._hold()
            self.state = "home"
        if self.state == "await_home":
            d, self._decision = self._decision, None
            if d in ("skip", "stop"):
                self.waiting, self.state = False, "done"
                self._note(t, "operator stop before the first move")
                return self._hold(done=True)
            if d != "go":
                return self._hold()
            self.waiting, self.state = False, "home"
        if self.state == "home":
            a, h = self._home_move
            self._enqueue(a, h)
            self.state = "settle"
            self._note(t, f"to home ({len(a) / HZ:.1f} s)")
            return self._pop() if len(a) else self._hold()
        if self.state == "settle":
            n = int(round(self.cfg["motion"]["settle_s"] * HZ))
            if self.ff is not None:   # the only place the applied gain changes: blended, then the settle as before
                n += self._ff_blend(t)
            self._enqueue(np.repeat(self.arm_cmd[None], n, 0), np.repeat(self.hand_cmd[None], n, 0))
            self.state = "look"
            if self.gate is not None:   # the look waits for two head images that agree (after the settle)
                self.gate.reset()
                self._gate_ok, self._gate_sph = False, None
            return self._pop()
        if self.state == "look":
            if not self._gate_ok and self._job is None:   # (before the feed-forward's check: that ends a still stretch)
                if self._gate_sph is None:  # the arms' silhouettes are not background (no look job runs: its kin)
                    self._self_mask(obs)
                    self._gate_sph = getattr(self.perception, "arm_spheres", None) or []
                r = self.gate.step(obs, t, self._gate_sph, self.threaded)
                if r is None:
                    return self._hold()
                self._gate_ok = True
                self._note(t, StillGate.text(r))
            if self.ff is not None and self._job is None and self._ff_resettle(t):
                self.state = "settle"      # the estimate moved materially: blend it in before planning with it
                return self.act(obs)
            if self.threaded:
                if self._job is None:
                    self._job = {"obs": obs, "result": None}
                    threading.Thread(target=self._look_job, args=(self._job,), daemon=True).start()
                if self._job["result"] is None:
                    return self._hold()
                res, self._job = self._job["result"], None
            else:
                res = self._look(obs)
            return self._after_look(res, t)
        if self.state == "await":
            d, self._decision = self._decision, None
            if d == "go":
                self.waiting = False
                if t + self.pending.duration > self.time_limit:  # confirmed too late: the not-enough-time path
                    return self._after_look(("plan", self.pending, 0), t)
                return self._start(self.pending, t)
            if d == "skip":
                self.waiting = False
                for p in self.pending.parts or [self.pending]:
                    rec = getattr(p, "rec", None)
                    if rec is not None:  # skipped until it moves (as a plan failure)
                        rec["failed_at"] = rec["xy"].copy()
                self._note(t, f"operator skipped {self.pending.color}")
                self.state = "look"
                return self._hold()
            if d == "stop":
                self.waiting = False
                self.state = "done"
                self._note(t, "operator stop")
            return self._hold(done=self.state == "done")
        if self.state == "executing":  # queue drained (plan or abort) -> back home: look again after settling
            self._finish(t)
            self.state = "settle"
            return self.act(obs)
        return self._hold(done=True)

    # ------------------------------------------------------------------ look + plan
    def _look_job(self, job):
        _lower_thread_priority()  # planning must not starve act() (the control loop's thread)
        try:
            job["result"] = self._look(job["obs"])
        except Exception as e:  # never kill the control loop
            job["result"] = ("error", repr(e))

    def _look(self, obs):
        w0 = time.perf_counter()
        try:
            return self._look_plan(obs)
        finally:   # wall time of the look + planning (perception, every plan tried), one log line per look (not an
            self.look_s.append(time.perf_counter() - w0)   # event: events stay deterministic)
            if self.log is not None:
                self.log(f"[classical t={float(obs.get('t', 0.0)):6.2f}] plan time {self.look_s[-1]:.2f} s "
                         f"(look + planning)")

    def _look_plan(self, obs):
        t = float(obs.get("t", 0.0))
        if self.insert is not None and self.insert.understood and not self.plans:   # rung 3: reach / decision, once
            n = insert_look_note(self.planner, self.cfg)
            if n:
                self._note(t, n)
        self._self_mask(obs)
        dets = self.perception.detect(obs["head_rgb"], obs["head_depth"])
        self._use_boxes()
        self._calibrate(dets, t)
        self.last_detections = dets
        shorts = self._far_check(dets, t)
        recs = self._track(dets)
        self._far_short(shorts, recs)
        bins = [self.bin_for(d) for d in dets]
        self._note(t, f"look: {len(dets)} detections" + "".join(  # colour:shape/hue(x,y)=object id/tries->bin
            f" {d.color}{self._shape_note(d)}{self._hue_note(d)}({d.top[0]:+.3f},{d.top[1]:+.3f})"
            f"=obj{r['id']}/{r['tries']}->{b or '-'}"
            for d, r, b in zip(dets, recs, bins)) + self._camera_note())
        # no bin (key not understood / shape or colour it needs not recognised) or its box not found (yet): left for
        # now, not failed
        # a box no arm can deliver into (grasp.far on: no drop slot, no far release in reach; cached per box pose):
        # its objects are left now, before any per-object planning, one line per box
        far = {b: self.planner.box_reachable(b) for b in set(bins) if b is not None and self.planner.box_ok(b)}
        for b, (ok, why) in sorted(far.items()):
            if not ok:
                self._note(t, f"{b} box out of reach ({why}): its {bins.count(b)} object(s) stay")
        cands = [(d, r) for d, r, b in zip(dets, recs, bins) if b is not None and self.planner.box_ok(b)
                 and far[b][0] and self._eligible(r, d.top[:2])]
        if self.cfg["policy"]["order"] == "nearest_bin":
            def key(d):
                return np.linalg.norm(d.top[:2] - self.planner.bin_xy(self.bin_for(d)))
        else:
            def key(d):
                return -d.n_px
        # untried first; grasp.far: objects for a box only a MARGINAL far release reaches after the others (a miss there
        # costs no points but time)
        grade = {b: self.planner.box_grade(b) for b in far}
        cands.sort(key=lambda dr: (dr[1]["tries"] > 0, grade.get(self.bin_for(dr[0])) == "marginal",
                                   self._crowded(dr[0], dets), key(dr[0])))
        self._set_drops()
        blocked = []
        budget = self.cfg["policy"].get("look_budget_s")   # wall s of planning per look (null: none)
        t0 = time.perf_counter()
        for i, (d, rec) in enumerate(cands):
            if budget is not None and i and time.perf_counter() - t0 > float(budget):
                self._note(t, f"look budget: {i} of {len(cands)} objects planned in {time.perf_counter() - t0:.1f} s, "
                              f"none executable: looking again")
                return ("budget", None, len(dets))
            others = [o for o in dets if o is not d]
            if self._same_scene(rec.get("blocked"), others):   # blocked in this very scene: planning again is moot
                blocked.append((d, rec, others))
                continue
            short = rec.get("far_short", 0)
            if (self.planner.far.on() and (short or d.top[0] > self._band_x + 0.04)
                    and not self.planner.grasp_in_reach(d)):   # (robot profile: grasp.far on)
                # beyond the layout band (e.g. a short far release, just in front of its box): often out of the
                # top-down grasp, and the full search (every finger direction, both arms, relocation) took 20-40 s
                # per look there (sim): one IK per arm decides, the object is left
                rec["failed_at"] = d.top[:2].copy()
                self._note(t, f"obj{rec['id']} {d.color} at ({d.top[0]:+.3f},{d.top[1]:+.3f}): out of grasp reach, left")
                continue
            self.planner.far_alt = short   # (a retry after a short far release: another one)
            try:
                p = self.planner.plan(d, self.arm_cmd, self.hand_cmd, bin_name=self.bin_for(d),
                                      obstacles=[o.top[:2] for o in others], variant=self._variant(rec),
                                      neighbours=others, relocate=not short)
            finally:
                self.planner.far_alt = 0
            self.plans.append(p)
            if p.ok:
                rec["blocked"] = None
                m = self._accept(p, d, rec, cands, t)
                if m is not None:
                    return ("plan", m, len(dets))
                continue
            movable = [c for c, _ in cands if c is not d]
            if p.blocked and self.planner.far.on() and not any(   # grasp.far (robot profile): no neighbour within
                    np.linalg.norm(o.top[:2] - d.top[:2]) < 0.12 and any(o is c for c in movable) for o in others):
                # 12 cm will be moved (no bin, box out of reach, out of tries): waiting for one is a deadlock, so the
                # least-bad rule now (as _least_bad does when nothing else plans)
                lb = self._least_bad([(d, rec, others)], cands, t)
                if lb is not None:
                    return ("plan", lb, len(dets))
            if p.blocked:   # a neighbour is in the way of every candidate: take another object first
                rec["blocked"] = np.array([o.top[:2] for o in others]).reshape(-1, 2)
                self._note(t, f"obj{rec['id']} {d.color}: blocked ({p.reason}); other objects first")
                blocked.append((d, rec, others))
                continue
            rec["failed_at"] = d.top[:2].copy()
            self.log(f"[classical] skip {d}: {p.reason}")
        if budget is not None and blocked and time.perf_counter() - t0 > float(budget):
            self._note(t, f"look budget: {time.perf_counter() - t0:.1f} s, least-bad pass left for the next look")
            return ("budget", None, len(dets))
        m = self._least_bad(blocked, cands, t)
        return ("none", None, len(dets)) if m is None else ("plan", m, len(dets))

    def _far_check(self, dets, t):
        """The look after a far release (grasp.far) that was executed to its end: an object of its colour on the
        counter within 12 cm of the expected landing = a short miss: {detection index: the released object's record}
        (_far_short: one more try with another release, then it is left); else it is taken as in the box, which the
        camera cannot confirm (objects inside a box footprint are masked out): logged only."""
        out = {}
        for side, b, xy, col, rec, done in self.far_watch:
            if not done:
                continue
            near = [i for i, d in enumerate(dets) if d.color == col and np.linalg.norm(d.top[:2] - xy) < 0.12
                    and i not in out]
            if near:
                out[near[0]] = rec
                d = dets[near[0]]
                self._note(t, f"far release into {b}: {col} object back on the counter at ({d.top[0]:+.3f},"
                              f"{d.top[1]:+.3f}) (short)")
            else:
                self._note(t, f"far release into {b}: not on the counter near it (in the box: unconfirmed)")
        self.far_watch = []
        return out

    def _far_short(self, short, recs):
        """Short far releases seen again (_far_check): the record now tracking the object inherits the tries and
        counts the short (far_short); the next plan of it uses another far release (Planner.far_alt); after two shorts
        it is left (_eligible)."""
        for i, old in short.items():
            r = recs[i]
            n = (old.get("far_short", 0) if old is not None else 0) + 1
            if old is not None and r is not old:
                r["tries"] = max(r["tries"], old["tries"])
            r["far_short"] = max(r.get("far_short", 0), n)

    @staticmethod
    def _shape_note(d):
        """':tri.85' = shape and its confidence (what decides a face-aligned grasp); ':?' = no shape accepted;
        'tri.85L+12' = lying on a side, ridge / axis at +12 deg."""
        sh = getattr(d, "shape", None)
        ly = f"L{np.degrees(d.ridge_yaw):+.0f}" if getattr(d, "lying", False) and d.ridge_yaw is not None else ""
        return ":?" if sh is None else f":{sh[:3]}{('%.2f' % float(getattr(d, 'shape_conf', 0.0)))[1:]}{ly}"

    @staticmethod
    def _hue_note(d):
        """'/H11.6' = the blob's own mean hue (what perception.color.refs are tuned from); '' without one."""
        h = getattr(d, "hue", None)
        return "" if h is None else f"/H{h:.1f}"

    def _camera_note(self):
        """How the last frame was read (for the log): detection path, self-levelling and bin registration
        corrections of the head pose, or why one was not applied. '' for a perception without them (oracle)."""
        pc = self.perception
        lv, rg = getattr(pc, "level", None), getattr(pc, "reg", None)
        if not isinstance(lv, dict) or not isinstance(rg, dict):
            return ""
        a = (f"level {lv.get('pitch_deg', 0):+.2f}/{lv.get('roll_deg', 0):+.2f} deg {lv.get('dz_mm', 0):+.1f} mm"
             if lv.get("used") else f"level off ({lv.get('why')})")
        b = (f"reg {rg.get('dx_mm', 0):+.1f},{rg.get('dy_mm', 0):+.1f} mm {rg.get('dyaw_deg', 0):+.2f} deg"
             if rg.get("used") else f"reg off ({rg.get('why')})")
        return f"  [{getattr(pc, 'path', '?')}; {a}; {b}{self._boxes_note()}]"

    def _boxes_note(self):
        """'; boxes L (x,y) yaw rim side conf | C ... | R - (why) [look: n found ...; ms]' (perception.boxes): the boxes
        in use (mode on; 'log' = found, not used), '-' = not delivered there."""
        bd, pc = getattr(self.perception, "box_det", None), self.perception
        if not isinstance(bd, dict) or bd.get("why") == "off" or bd.get("boxes") is None:
            return ""
        res = bd.get("res") or {}
        n = len(res.get("boxes", ()))
        look = f"look: {n} found" + ("" if res.get("named") or not n else ", names uncertain")
        look += f", {bd['why']}" if bd.get("why") else ""
        return (f"; boxes{'' if pc.boxes_on else ' (log)'} {bd['boxes'].text()} [{look}; {bd.get('ms', 0):.0f} ms]"
                + "".join(f"; {e}" for e in bd.get("events", ())))

    def _self_mask(self, obs):
        """The arms' sphere model (forearm + hand, measured pose) for the box detection's self-mask."""
        if not hasattr(self.perception, "arm_spheres"):
            return
        self.perception.arm_spheres = None
        for a, h in ((obs.get("arm_q"), obs.get("hand_q")), (self.arm_cmd, self.hand_cmd)):
            a, h = np.asarray(a if a is not None else np.nan, float), np.asarray(h if h is not None else np.nan, float)
            if a.shape == (14,) and h.shape == (14,) and np.isfinite(a).all() and np.isfinite(h).all():
                break
        else:
            return
        out = []
        for side in ("left", "right"):
            c, r = self.planner.kin.hand_spheres(side, a, h)
            out.append((c[0], r))
        self.perception.arm_spheres = out

    def _use_boxes(self):
        """perception.boxes.mode on: the boxes in use (detected, fused over the episode) -> every planner (drop slots,
        heights, rim clearances, abort checks). Release points already recorded stay where they are (pelvis frame:
        a re-estimated box does not move the objects in it)."""
        if not getattr(self.perception, "boxes_on", False):
            return
        for p in (self.planner, self.abort_planner, self.act_planner):
            if p is not None:
                p.set_boxes(self.perception.boxes)

    def _accept(self, p, d, rec, cands, t):
        """A plan found for (d, rec) at time t -> what to execute: the pair with a second object (_pair), else the
        plan itself or its evade (_guard); None if it does not fit the time budget (rec no_time) or the parked arm
        is in the way without an evade (rec failed_at, until it moves)."""
        if t + p.duration > self.time_limit:  # a shorter plan of another object may still fit
            rec["no_time"] = True
            self._note(t, f"obj{rec['id']} {d.color}: plan {p.duration:.1f} s > {self.time_limit - t:.1f} s left")
            return None
        p.rec = rec
        rest = [(d2, r2, self.bin_for(d2)) for d2, r2 in cands if r2 is not rec]
        m = self._pair(p, rest, self.arm_cmd, self.hand_cmd, t)
        if m is p:
            m = self._guard(p, self.arm_cmd, self.hand_cmd, t)
            if m is None:   # too close to the parked arm, no evade: as a failed plan (until it moves)
                rec["failed_at"] = d.top[:2].copy()
                return None
            if t + m.duration > self.time_limit:
                rec["no_time"] = True
                self._note(t, f"obj{rec['id']} {d.color}: evade {m.duration:.1f} s > {self.time_limit - t:.1f} s left")
                return None
        return m

    def _least_bad(self, blocked, cands, t):
        """No other object could be planned: the blocked ones in candidate order with the least-bad rule
        (neighbours.least_bad; Planner.plan(least_bad=True): >= neighbours.floor, the most clearance, pinch axis
        perpendicular to the nearest neighbour first). Returns what to execute (as _accept) or None; an object with
        no least-bad plan either stays blocked (not failed: once another object has gone, it may be clear)."""
        if not blocked or not (self.cfg.get("neighbours") or {}).get("least_bad", True):
            return None
        for d, rec, others in blocked:
            if self._same_scene(rec.get("lb_failed"), others):   # failed in this very scene already
                continue
            p = self.planner.plan(d, self.arm_cmd, self.hand_cmd, bin_name=self.bin_for(d),
                                  obstacles=[o.top[:2] for o in others], variant=self._variant(rec), neighbours=others,
                                  least_bad=True)
            self.plans.append(p)
            if not p.ok:
                rec["lb_failed"] = np.array([o.top[:2] for o in others]).reshape(-1, 2)
                self._note(t, f"obj{rec['id']} {d.color}: no least-bad plan either ({p.reason})")
                continue
            self._note(t, f"least-bad: obj{rec['id']} {d.color} {p.side} arm, {1000 * p.nb_clear:.0f} mm from the "
                          f"nearest object (floor {1000 * float(self.cfg['neighbours']['floor']):.0f}); {p.reason}")
            m = self._accept(p, d, rec, cands, t)
            if m is not None:
                return m
        return None

    def _crowded(self, d, dets):
        """How many other detections are closer than neighbours.order_free (0 = free): crowded objects are taken after
        the free ones, the outer objects of a touching cluster (fewer neighbours) before its inner ones, so the
        cluster opens up."""
        r = float((self.cfg.get("neighbours") or {}).get("order_free") or 0.0)
        return int(sum(np.linalg.norm(o.top[:2] - d.top[:2]) < r for o in dets if o is not d)) if r > 0 else 0

    @staticmethod
    def _same_scene(xy, others, tol=0.01):
        """True if the detections `others` stand where the points xy (n, 2) were (each within tol, same count)."""
        if xy is None or len(xy) != len(others):
            return False
        free = list(range(len(xy)))
        for o in others:
            j = next((j for j in free if np.linalg.norm(xy[j] - o.top[:2]) < tol), None)
            if j is None:
                return False
            free.remove(j)
        return True

    def _set_drops(self):
        """Planner drop state for this look: grasp.drop_offsets (k-th object started into a bin: offset k) and the
        release points of the objects delivered so far (grasp.slots)."""
        offs = self.cfg["grasp"].get("drop_offsets") or [[0.0, 0.0]]
        self.planner.drop_offsets = {b: offs[self.drops.get(b, 0) % len(offs)] for b in self.scene.bin_names}
        self.planner.drop_used = {b: [np.array(p) for p in self.drops_at.get(b, [])] for b in self.scene.bin_names}

    def _after_look(self, res, t):
        kind = res[0]
        if kind == "budget":   # policy.look_budget_s ran out without a plan: look (and plan the rest) again
            self.state = "look"
            return self._hold()
        if kind == "error":
            self._note(t, f"look/plan error {res[1]}: stopping")
            self.state = "done"
            return self._hold(done=True)
        if kind == "none":
            nb = sum(self.bin_for(d) is None for d in self.last_detections)
            nx = sum(self.bin_for(d) is not None and not self.planner.box_ok(self.bin_for(d))
                     for d in self.last_detections)
            nr = sum(self.bin_for(d) is not None and self.planner.box_ok(self.bin_for(d))
                     and not self.planner.box_reachable(self.bin_for(d))[0] for d in self.last_detections)
            self._note(t, f"nothing left to sort ({res[2]} detections" + (f"; {nb} without a bin, left on the counter"
                                                                          if nb else "")
                       + (f"; {nx} whose box was not found, left on the counter" if nx else "")
                       + (f"; {nr} whose box is out of reach, left on the counter" if nr else "") + ")")
            relook = int((self.cfg["perception"].get("boxes") or {}).get("relook", 0))
            if nx and self._box_looks < relook:     # a box not found yet: look again (noise, a passing occlusion)
                self._box_looks += 1
                self._note(t, f"looking again for the boxes ({self._box_looks}/{relook})")
                self.state = "settle"
                return self._hold()
            self.state = "done"
            return self._hold(done=True)
        plan = res[1]
        if plan.parts and not plan.evade and t + plan.duration > self.time_limit:  # pair no longer fits: its first
            if not getattr(plan, "solo_ok", True):                                  # plan alone may (if clear)
                self._note(t, f"not enough time for the pair ({plan.duration:.1f} s); {plan.parts[0].color} alone "
                              f"comes too close to the parked arm: looking again")
                self.state = "look"
                return self._hold()
            self._note(t, f"not enough time for the pair ({plan.duration:.1f} s): {plan.parts[0].color} alone")
            plan = plan.parts[0]
        if t + plan.duration > self.time_limit:  # (planning took time) never start a plan that cannot finish
            rec = getattr(plan, "rec", None)
            self._note(t, f"not enough time for {plan.color} ({plan.duration:.0f} s)")
            if rec is None:
                self.state = "done"
                return self._hold(done=True)
            self._register(rec)
            rec["no_time"] = True
            self.state = "look"  # another candidate may still fit
            return self._hold()
        if self.require_confirm:
            self.pending, self.waiting, self.state = plan, True, "await"
            self.pending_text = plan.summary()
            self._note(t, f"waiting for confirm: {plan.summary()}")
            return self._hold()
        return self._start(plan, t)

    def _start(self, plan, t):
        """Execute a single plan or a merged pair (plan.parts): per part the object's tries / variant, the drop count
        and a grasp-verification entry for its arm (window shifted by its lag). An evade's idle part: nothing (no
        object, no grasp)."""
        if self.ff is not None:     # the look's still stretch ends here
            self._ff_report(t, self.ff.flush(self.act_kin))
        self.execs, whos, self.exec_drops, self.exec_nb, self.corrs = {}, [], [], {}, {}
        for p in plan.parts or [plan]:
            if p.evade:
                continue
            ph = {n: (s, e) for n, s, e in p.phases}
            if self.correct_on and p.side in ("left", "right") and "settle" in ph and "close" in ph:
                lag = plan.lags.get(p.side, 0)   # measured where the blend-in must start (_pop ends a chunk there)
                ramp = correct_ticks(self.cfg["grasp"]["correct"], 1.0 / HZ)[1]
                self.corrs[p.side] = {"k_meas": ph["close"][0] + lag - ramp, "done": False,
                                      "pose": p.arm[ph["settle"][0], ArmKinematics.sl(p.side)].copy(),
                                      "phases": [(n, s + lag, e + lag) for n, s, e in p.phases if s >= ph["settle"][0]],
                                      "place": p.target_bin == "counter", "nb": p.nb}
            rec = getattr(p, "rec", None)
            if rec is not None:
                self._register(rec)
                rec["tries"] += 1
                rec["tried_xy"] = rec["xy"].copy()
                rec["variants"].append(p.variant)
            if p.target_bin in self.scene.bin_names:  # not a relocation ("counter")
                self.drops[p.target_bin] = self.drops.get(p.target_bin, 0) + 1
                if "drop" in p.waypoints:
                    self.exec_drops.append((p.side, p.target_bin, np.asarray(p.waypoints["drop"][:2], float).copy()))
                    if getattr(p, "far", None) is not None:
                        self.far_watch.append((p.side, p.target_bin, np.asarray(p.waypoints["drop"][:2], float).copy(),
                                               p.color, rec, False))
            if p.nb is not None and p.side in ("left", "right"):
                self.exec_nb[p.side] = p.nb
            w = grasp_window(p.phases, p.hand, ArmKinematics.sl(p.side)) if self.verify_on and p.side else None
            if w is not None:
                vc, lag = self.cfg["verify"], plan.lags.get(p.side, 0)
                self.execs[p.side] = {"rec": rec, "k_closed": w[0] + lag, "k_release": w[1] + lag, "target": w[2],
                                      "mon": GraspMonitor(vc["after_s"], vc["empty_below"], vc["confirm"],
                                                          vc.get("signal", "abs7")),
                                      "verdict": "unknown", "outcome": None, "tries": 0, "done": False,
                                      "aborted": False}
            whos.append(("" if rec is None else f"obj{rec['id']} try {rec['tries']} ")
                        + f"variant {p.variant} hint {p.hint}")
        self._enqueue(plan.arm, plan.hand)
        self.exec_masks = list(held_masks(plan))
        self.exec_held = [m.copy() for m in self.exec_masks] if plan.parts else None
        self.exec_parked = self._avoid_parked() and not plan.parts
        self.state = "executing"
        kind = "evade " if plan.evade else "pair " if plan.parts else ""
        self._note(t, f"execute {kind}{' + '.join(whos)}: {plan.summary()}")
        return self._pop()

    # ------------------------------------------------------------------ paired two-arm cycle
    def _pair(self, first, cands, arm, hand, t):
        """two_arm: add a second object for the other arm to `first` (a pick-to-bin plan, not a relocation / relay).
        cands: [(det, rec, bin)] in candidate order. The first eligible one that plans with the other arm alone (from
        the same start pose; same bin: the next drop offset), merges (Planner.pair: lag, arm-arm clearance, 0 clips)
        and still ends before time_limit is taken; at most two_arm.max_candidates are planned (a cross-body arm/bin
        is skipped first, free). Returns the merged Plan (parts [first, second]) or `first`."""
        tw = self.cfg.get("two_arm") or {}
        if (not tw.get("enabled") or first.parts or first.side not in ("left", "right")
                or first.target_bin not in self.scene.bin_names):
            return first
        side = "right" if first.side == "left" else "left"
        reach = self.cfg["grasp"].get("reach_check")
        n, t0 = 0, time.perf_counter()
        for d, rec, b in cands:
            if n >= int(tw.get("max_candidates", 3)):
                break
            if (b not in self.scene.bin_names or rec is getattr(first, "rec", None)
                    or not self._eligible(rec, d.top[:2])):
                continue
            by = self.planner.bin_xy(b)[1]
            if (self.planner._reach_reject(d, side, b) is not None if reach  # cached IK reach test
                    else abs(by) > 0.1 and (by > 0) != (side == "left")):     # without it: 10-30 s to fail
                continue
            others = [o for o in self.last_detections if o is not d]     # incl. first's object (still there at start)
            if self._same_scene(rec.get("blocked"), others):              # blocked in this scene already (free)
                continue
            n += 1
            same = b == first.target_bin
            off = self._next_drop_offset(b) if same else None
            avoid = [first.waypoints["drop"][:2]] if same and "drop" in first.waypoints else None
            m, p2, why = self.planner.plan_pair(first, d, b, arm, hand, variant=self._variant(rec), drop_offset=off,
                                                drop_avoid=avoid, neighbours=others)
            self.plans.append(p2)
            if m is not None and t + m.duration > self.time_limit:
                m, why = None, f"pair {m.duration:.1f} s > {self.time_limit - t:.1f} s left"
            if m is None:
                self._note(t, f"pair: {self._who(rec, d)} ({side} arm) rejected: {why}")
                continue
            p2.rec = rec
            dmin, k = m.clearance
            self._note(t, f"pair: {self._who(getattr(first, 'rec', None), first)} {first.side} + {self._who(rec, d)} "
                          f"{side}, lag {m.lags[side] / HZ:.1f} s, {m.duration:.1f} s (singles {first.duration:.1f} + "
                          f"{p2.duration:.1f}), min clearance {1000 * dmin:.0f} mm at "
                          f"{self.planner.where(m, k, held_masks(m), self.planner.kin)} "
                          f"({time.perf_counter() - t0:.1f} s planning)")
            if self._avoid_parked():  # _after_look falls back to `first` alone if the pair no longer fits: clear?
                need = float(tw.get("min_clearance", 0.02))
                m.solo_ok = self.planner.parked_clearance(first, stop_below=need)[0] >= need
            return m
        return first

    def _avoid_parked(self):
        return bool((self.cfg.get("two_arm") or {}).get("avoid_parked"))

    def _guard(self, plan, arm, hand, t):
        """two_arm.avoid_parked: `plan` (to be executed with one arm, no pair) vs the other arm parked at its pose
        (Planner.keep_clear). Returns `plan` if clear (or the feature is off), the merged evade queue (the other arm
        moves aside and back; it carries the plan's rec / oracle fields) or None: no evade works (logged)."""
        if not self._avoid_parked() or plan.parts or plan.side not in ("left", "right"):
            return plan
        t0 = time.perf_counter()
        m, info = self.planner.keep_clear(plan, arm, hand)
        if m is plan:
            return plan
        rec, d0 = getattr(plan, "rec", None), info["parked"][0]
        idle = "right" if plan.side == "left" else "left"
        what = (f"{self._who(rec, plan)} {plan.side} arm -> {plan.target_bin}: {1000 * d0:.0f} mm from the parked "
                f"{idle} arm at {info['where']}")
        ms = 1e3 * (time.perf_counter() - t0)
        if m is None:
            self._note(t, f"evade: {what}; skipped, no evade ({info['why']}; {ms:.0f} ms)")
            return None
        e, (d, k) = m.evade, m.clearance
        self._note(t, f"evade: {what}; {idle} arm to clear pose {e['pose']} and back at {e['back'] / HZ:.1f} s, "
                      f"{plan.side} lag {e['lag'] / HZ:.1f} s, {m.duration:.1f} s (single {plan.duration:.1f} s, "
                      f"+{e['added'] / HZ:.1f} s), min clearance {1000 * d:.0f} mm at "
                      f"{self.planner.where(m, k, held_masks(m), self.planner.kin)} ({ms:.0f} ms planning)")
        for a in ("rec", "obj", "obj_xy", "relay"):  # per-object fields of the plan (oracle: obj, obj_xy, relay)
            if hasattr(plan, a):
                setattr(m, a, getattr(plan, a))
        m.parts[1].rec = None                        # the idle arm's part: no object
        return m

    @staticmethod
    def _who(rec, d):
        return f"{d.color}" if rec is None or rec.get("id") is None else f"obj{rec['id']} {d.color}"

    def _next_drop_offset(self, b):
        """The drop offset after the one this look uses for bin b (both objects of a pair into the same bin)."""
        offs = self.cfg["grasp"].get("drop_offsets") or [[0.0, 0.0]]
        cur = self.planner.drop_offsets.get(b, offs[0])
        i = next((i for i, o in enumerate(offs) if np.allclose(o, cur)), 0)
        return offs[(i + 1) % len(offs)]

    def _pair_clear(self, side, arm, hand):
        """An abort queue (arm, hand) of `side` while a pair / an evade executes vs the other arm's still-planned
        motion (its held-object mask for these ticks; the aborting hand is verified empty), or (single plan with
        avoid_parked: exec_held None) vs the parked other arm: None if the model distance stays >=
        two_arm.min_clearance, else the reason."""
        need = float((self.cfg.get("two_arm") or {}).get("min_clearance", 0.02))
        if self.exec_held is None:
            held = [np.zeros(len(arm), bool), np.zeros(len(arm), bool)]
        else:
            idx = np.minimum(self.q_t + np.arange(len(arm)), len(self.exec_held[0]) - 1)  # queue tick -> plan tick
            held = [m[idx] for m in self.exec_held]
        held[0 if side == "left" else 1] = np.zeros(len(arm), bool)
        d, k = self.abort_planner.kin.min_arm_arm_distance(arm, hand, step=PAIR_CHECK_STEP, stop_below=need,
                                                           with_object=checked_masks(held))
        return None if d >= need else f"arm-arm clearance {1000 * d:.0f} mm < {1000 * need:.0f} mm in {k} ticks"

    # ------------------------------------------------------------------ per-object bookkeeping
    def _track(self, dets, commit=True):
        """Detection -> per-object record (list aligned with dets). Same object = same `name` if the detector gives
        one (oracle), else same colour within policy.same_object_radius of its last known or last attempted position
        (a failed grasp pushes the object a few cm); nearest pairs first, at most one detection per record.
        commit=False (planning ahead): new objects get records that are not stored, known ones keep their xy."""
        r = float(self.cfg["policy"]["same_object_radius"])
        pairs = []
        for i, d in enumerate(dets):
            name = getattr(d, "name", None)
            for o in list(self.objs):
                if name is not None and o["name"] is not None:
                    if name == o["name"]:
                        pairs.append((-1.0, i, o))
                elif o["color"] == d.color and not o["delivered"]:  # delivered = in its bin: this is another one
                    dist = min(np.linalg.norm(d.top[:2] - p) for p in (o["xy"], o["tried_xy"]) if p is not None)
                    if dist < r:
                        pairs.append((dist, i, o))
        out, used = [None] * len(dets), set()
        for _, i, o in sorted(pairs, key=lambda p: p[0]):
            if out[i] is None and id(o) not in used:
                out[i] = o
                used.add(id(o))
        for i, d in enumerate(dets):
            if out[i] is None:
                out[i] = {"id": None, "color": d.color, "name": getattr(d, "name", None), "xy": d.top[:2].copy(),
                          "tried_xy": None, "tries": 0, "misses": 0, "slips": 0, "delivered": 0, "variants": [],
                          "failed_at": None, "no_time": False, "blocked": None, "lb_failed": None}
                if commit:
                    self._register(out[i])
            elif commit:
                out[i]["xy"] = d.top[:2].copy()
        return out

    def _register(self, rec):
        if not any(o is rec for o in self.objs):
            rec["id"] = len(self.objs)
            self.objs.append(rec)

    def _eligible(self, rec, xy):
        """Not out of tries / time, and no failed plan at (about) this position: a failed plan is tried again only
        after the object moved policy.replan_moved."""
        pc = self.cfg["policy"]
        if rec["no_time"] or rec["tries"] >= int(pc["max_attempts"]) or rec.get("far_short", 0) >= 2:
            return False
        fa = rec["failed_at"]
        return fa is None or np.linalg.norm(np.asarray(xy) - fa) >= float(pc.get("replan_moved", 0.03))

    @staticmethod
    def _variant(rec):
        """Grasp variant of the next try: one step further per verified miss / slip (0 = the default order)."""
        return rec["misses"] + rec["slips"]

    # ------------------------------------------------------------------ grasp verification + abort
    def _watch(self, obs, t):
        """Once per act() while a plan executes. The runner asks for the next chunk while the last tick of the
        previous one executes, so the obs reflects plan tick q_t - 2 (holds only add time: the clock is
        conservative). Monitoring ends when the release is commanded."""
        e = self.q_t - 2
        for side, x in self.execs.items():
            if x["done"]:
                continue
            if self.q_t >= x["k_release"]:
                x["done"] = True
                continue
            cs = (e - x["k_closed"]) / HZ if e >= x["k_closed"] else None
            v = x["mon"].update(np.asarray(obs["hand_q"], float)[ArmKinematics.sl(side)], x["target"], cs)
            if v != x["verdict"]:
                x["verdict"] = v
                self._note(t, f"verify {side}: {v} sig={x['mon'].sig:.4f} ({cs:.2f} s after close)")
            if v != EMPTY:
                continue
            if x["outcome"] is None:
                x["outcome"] = "slip" if x["mon"].slipped else "miss"
                if x["rec"] is not None:
                    x["rec"]["slips" if x["outcome"] == "slip" else "misses"] += 1
            self._abort(side, x, t)

    def _abort(self, side, x, t):
        if self.abort_planner is None or x["tries"] >= MAX_ABORT_TRIES:
            x["done"] = True
            return
        x["tries"] += 1
        t0 = time.perf_counter()
        arm, hand, info = self.abort_planner.abort(side, self.arm_cmd, self.hand_cmd, self.queue_a, self.queue_h,
                                                   neighbours=self.exec_nb.get(side))
        if arm is not None and (self.exec_held is not None or self.exec_parked):  # keep clear of the other arm
            why = self._pair_clear(side, arm, hand)
            if why is not None:
                arm, hand, info = None, None, why
        ms = 1e3 * (time.perf_counter() - t0)
        if arm is None:
            self._note(t, f"abort {side} rejected ({info}; {ms:.0f} ms): plan continues"
                          + (", retry next chunk" if x["tries"] < MAX_ABORT_TRIES else ""))
            return
        n = len(self.queue_a)
        self.queue_a, self.queue_h = arm, hand  # q_t keeps counting: the other arm's columns stay tick-aligned
        if self.exec_held is not None:          # this hand holds nothing from now on
            self.exec_held[0 if side == "left" else 1][self.q_t:] = False
        if self.exec_masks is not None:
            self.exec_masks[0 if side == "left" else 1][self.q_t:] = False
        if side in self.corrs:                  # (a pending correction of this arm has no queue left to shift)
            self.corrs[side]["done"] = True
        x["done"] = x["aborted"] = True
        self._note(t, f"abort {side} ({x['outcome']}): decelerate {info['decel_ticks']} + home {info['home_ticks']} "
                      f"ticks instead of {n} ({(n - len(arm)) / HZ:.1f} s saved, {ms:.0f} ms)")

    def _finish(self, t):
        """Queue drained: per-object outcome of the grasps that were verified; release points of the delivered
        objects (verified held until the release, or unverified: the release was executed) -> drops_at."""
        for side, b, xy in self.exec_drops:
            x = self.execs.get(side)
            if x is None or (x["outcome"] is None and x["verdict"] == HELD and not x["aborted"]):
                self.drops_at.setdefault(b, []).append(xy)
        for k, w in enumerate(self.far_watch):   # far releases carried out to the end: checked at the next look
            x = self.execs.get(w[0])
            if x is None or (x["outcome"] is None and not x["aborted"]):
                self.far_watch[k] = w[:5] + (True,)
        self.exec_drops = []
        for side, x in self.execs.items():
            rec = x["rec"]
            if x["outcome"] is None and x["verdict"] == HELD and rec is not None:
                rec["delivered"] += 1
            if not x["aborted"]:
                what = x["outcome"] or ("held until the release" if x["verdict"] == HELD else "unverified")
                self._note(t, f"{'' if rec is None else 'obj%s ' % rec['id']}{side}: {what}")
        self.execs, self.exec_held, self.exec_parked, self.exec_masks, self.corrs = {}, None, False, None, {}

    # ------------------------------------------------------------------ settle and correct (grasp.correct)
    def _correct(self, obs, t):
        """Once per act() while a plan executes: an arm whose settle phase has reached its measurement tick (_pop
        ends the chunk there; the obs is ~motion lag ticks older, i.e. inside the settle, arm still) gets its
        grasp-point tracking error measured and, if every guard passes, its queued ticks shifted (Planner.correct,
        act()-thread planner). One _note per correction."""
        for side, x in self.corrs.items():
            if not x["done"] and self.q_t >= x["k_meas"]:
                x["done"] = True
                self._correct_arm(side, x, obs, t)

    @staticmethod
    def _correct_guard(cc, err, res, speed):
        """grasp.correct plausibility guards on the measurement (m, m, rad/s): None or why not."""
        if err > float(cc.get("max_err", 0.02)):
            return f"error > max_err {1000 * float(cc.get('max_err', 0.02)):.0f} mm: something in the way?"
        if err < float(cc.get("min_err", 0.001)):
            return "below min_err"
        if speed > float(cc.get("max_speed", 0.1)):
            return f"arm still moving ({speed:.2f} rad/s)"
        if res > 0.5 * err:
            return f"inside the IK joint bounds {1000 * res:.1f} of {1000 * err:.1f} mm would remain"
        return None

    def _correct_arm(self, side, x, obs, t):
        cc, sl, P = self.cfg["grasp"]["correct"], ArmKinematics.sl(side), self.act_planner
        t0 = time.perf_counter()
        meas = np.asarray(obs["arm_q"], float)
        if self.ff is not None and self.ff.learn and np.abs(self.arm_cmd[sl] - x["pose"]).max() <= 1e-9:
            self._ff_sample(obs, t, "grasp", [side])   # BEFORE the shift: measured vs what was sent
        e, dq, f, res = P.correction_dq(side, self.arm_cmd, meas)
        dec = np.array([e @ f, e @ [-f[1], f[0], 0.0], e[2]])
        speed = float(np.abs(np.asarray(obs.get("arm_dq", np.zeros(14)), float)[sl]).max())
        err, info = float(np.linalg.norm(e)), None
        phases = [(n, s - self.q_t, en - self.q_t) for n, s, en in x["phases"] if en > self.q_t]
        why = self._correct_guard(cc, err, res, speed)
        if np.abs(self.arm_cmd[sl] - x["pose"]).max() > 1e-9:
            why = "the arm is not at its settle pose"
        elif not any(n == "settle" for n, _, _ in phases):
            why = "measured after the settle phase"
        elif why is None:
            idx = np.minimum(self.q_t + np.arange(len(self.queue_a)), len(self.exec_masks[0]) - 1)
            held = [m[idx] for m in (self.exec_held if self.exec_held is not None else self.exec_masks)]
            tw = self.cfg.get("two_arm") or {}
            need = float(tw.get("min_clearance", 0.02)) if (self.exec_held is not None or self.exec_parked) else None
            new, info = P.correct(side, self.arm_cmd, self.hand_cmd, self.queue_a, self.queue_h, dq, phases,
                                  place=x["place"], nb=x["nb"], held=held, pair_need=need)
            if new is None:
                why, info = info, None
            else:
                self.queue_a = new
        ms = 1e3 * (time.perf_counter() - t0)
        self.corrections.append({"t": t, "tick": self.q_t, "k_meas": x["k_meas"], "side": side, "err_mm": 1e3 * e,
                                 "dec_mm": 1e3 * dec, "dq": dq.copy(),
                                 "res_mm": 1e3 * res, "applied": why is None, "why": why, "ms": ms,
                                 "ticks": info and info["ticks"]})
        what = (f"applied ({info['ticks']} ticks: blend-in {info['ramp_ticks']}, out {info['blend_out_ticks']}; "
                f"predicted residual {1000 * res:.1f} mm)" if why is None else f"refused ({why})")
        self._note(t, f"correct {side}: grasp point {1000 * err:.1f} mm off (cmd - meas: pelvis x/y/z {1000 * e[0]:+.1f} "
                      f"{1000 * e[1]:+.1f} {1000 * e[2]:+.1f}; along fingers {1000 * dec[0]:+.1f}, across "
                      f"{1000 * dec[1]:+.1f}), dq max {1000 * np.abs(dq).max():.0f} mrad norm "
                      f"{1000 * np.linalg.norm(dq):.0f}: {what}; {ms:.1f} ms")

    # ------------------------------------------------------------------ action plumbing
    def _enqueue(self, a, h):
        self.queue_a, self.queue_h = np.asarray(a, float), np.asarray(h, float)
        self.q_t = 0

    def _pop(self):
        k = int(self.cfg["motion"]["chunk"])
        for x in self.corrs.values():   # a chunk ends at a pending measurement tick: the next act() measures there
            if not x["done"] and self.q_t < x["k_meas"]:
                k = min(k, x["k_meas"] - self.q_t)
        a, h = self.queue_a[:k], self.queue_h[:k]
        self.queue_a, self.queue_h = self.queue_a[k:], self.queue_h[k:]
        self.q_t += len(a)
        if not len(a):
            return self._hold()
        self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
        prev = self._sent()
        out = a.copy() if self.ff is None else self._ff_net(self.ff.commanded(self.act_kin, a, h,
                                                                              self.ff.tick_gains(len(a))))
        if self.ff is not None:
            for i, sl in enumerate((slice(0, 7), slice(7, 14))):   # ticks each arm's sent target has been constant
                ch = np.flatnonzero(np.abs(np.diff(np.vstack([prev[None, sl], out[:, sl]]), axis=0)).max(1) > 0)
                self.const[i] = len(out) - 1 - ch[-1] if len(ch) else self.const[i] + len(out)
        self.out_cmd = out[-1].copy()
        return {"arm_q_target": out, "hand_q_target": h.copy(), "done": False}

    def _sent(self):
        """The last arm target sent (without feed-forward: arm_cmd itself, whoever set it)."""
        return self.arm_cmd if self.ff is None or self.out_cmd is None else self.out_cmd

    def _hold(self, done=False):
        """Hold the last target. While waiting (planning / operator) the hold is `motion.hold_ticks` ticks long, so a
        remote runner needs one request per hold chunk instead of one per tick: the command rate stays at 30 Hz
        even when the planner thread and other processes (sims) load the CPU."""
        k = 1 if done else max(1, int(self.cfg["motion"].get("hold_ticks", 1)))
        self.const += k
        return {"arm_q_target": np.repeat(self._sent()[None], k, 0),
                "hand_q_target": np.repeat(self.hand_cmd[None], k, 0), "done": bool(done)}

    # ------------------------------------------------------------------ sag feed-forward (config sag)
    def attach_ff(self, *planners):
        """Point planners at the policy's feed-forward state (every one that plans / checks / dry-runs ticks)."""
        for p in planners:
            if p is not None:
                p.ff = self.ff

    def _ff_net(self, out):
        """Last net on the commanded ticks (planned ones never need it: their dry run had 0 clips at this gain): each
        tick within the filter's speed and position band from the one before; counted and logged if it acts."""
        sp, cap = self._out_lim, float(self.safety.arm_max_vel) * float(self.safety.dt) - 1e-6
        prev, new = self.out_cmd, out.copy()
        for k in range(len(new)):
            new[k] = np.clip(np.clip(new[k], prev - cap, prev + cap), *sp)
            prev = new[k]
        n = int((np.abs(new - out).max(1) > 1e-9).sum())
        if n:
            self.ff_net += n
            self._note(self._t, f"sag feed-forward: output net changed {n} ticks (max "
                                f"{1000 * np.abs(new - out).max():.1f} mrad; total {self.ff_net})")
        return new

    def _ff_blend(self, t):
        """Settle before a look: end the stretch, blend the applied gain to the current estimate. Returns the blend
        ticks (added to the settle); logs the gains when they change materially."""
        self._ff_report(t, self.ff.flush(self.act_kin))
        ff = self.ff
        n = ff.begin_blend(self.act_kin, self.arm_cmd, self.hand_cmd, 1.0 / HZ)
        g = ff.blend["g1"] if ff.blend else ff.applied
        if np.any(np.abs(g - ff.logged) > float(ff.sc.get("log_frac", 0.1)) * np.abs(ff.prior) + 1e-12):
            ff.logged = g.copy()
            e = ff.est
            self._note(t, f"sag feed-forward: gain/prior -> {self.ff.text(g)} over {n} ticks ({len(e.samples)} "
                          f"samples, {e.refused} refused; trust min {e.trusted().min():.2f})")
        return n

    def _ff_resettle(self, t):
        """Before a look: end the settle's stretch (one sample); if the estimate now moves some commanded joint by
        more than sag.resettle_tol at this pose (first samples, a wrong prior), True: settle again with the blend
        (at most sag.resettle_max times per episode), so this look's plan already uses it."""
        ff = self.ff
        if not ff.learn or ff.stretch is None:
            return False
        self._ff_report(t, ff.flush(self.act_kin))
        if self._resettles >= int(ff.sc.get("resettle_max", 2)):
            return False
        d = float(np.abs(ff.offset(self.act_kin, self.arm_cmd, self.hand_cmd, ff.target())
                         - ff.offset(self.act_kin, self.arm_cmd, self.hand_cmd)).max())
        if d <= float(ff.sc.get("resettle_tol", 0.002)):
            return False
        self._resettles += 1
        return True

    def _ff_observe(self, obs, t):
        """Every act() at home (settle / look / waiting for the operator): both arms' still observations -> the
        current stretch (_ff_sample)."""
        if obs is self._ff_seen:     # (act() calls itself on a state change: one observation counts once)
            return
        self._ff_seen = obs
        if self.state in ("settle", "look", "await", "await_home"):   # (moving ticks fail the still test)
            self._ff_sample(obs, t, "home", ("left", "right"), commit=False)

    def _ff_sample(self, obs, t, src, sides, commit=True):
        """Still test per arm: its sent target constant for >= sag.still_s + sag.obs_lag ticks and measured
        |arm_dq| <= sag.still_speed. Still arms -> SagFF.observe (commit: the stretch is one sample at once)."""
        a, h = np.asarray(obs.get("arm_q"), float), np.asarray(obs.get("hand_q"), float)
        if a.shape != (14,) or h.shape != (14,) or not (np.isfinite(a).all() and np.isfinite(h).all()):
            return
        dq = np.asarray(obs.get("arm_dq", np.zeros(14)), float)
        sc, ok = self.ff.sc, np.zeros(14, bool)
        need = self.ff.still_ticks + int(sc.get("obs_lag", 3))
        for i, side in enumerate(("left", "right")):
            sl = ArmKinematics.sl(side)
            if side in sides and self.const[i] >= need and np.abs(dq[sl]).max() <= float(sc.get("still_speed", 0.05)):
                ok[sl] = True
        if not ok.any():
            return
        self._ff_report(t, self.ff.observe(self.act_kin, a, self.out_cmd, h, ok, src, t))
        if commit:
            self._ff_report(t, self.ff.flush(self.act_kin))

    def _ff_report(self, t, res):
        """Log a refused still sample (a hand resting on something, a collision: not proportional to the torque)."""
        if res is not None and res[2]:
            self._note(t, f"sag sample ({res[0]}, {res[3]} obs) refused for " + "; ".join(
                f"{s}: {w}" for s, w in res[2].items()))

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[classical t={t:6.2f}] {msg}")
