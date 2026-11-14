"""One image for the Aalto Isaac sim and the real G1: detect which world the policy runs in, use that world's config.

    pol = AutoProfilePolicy(lambda profile: ClassicalServerPolicy(profile=profile))   # policy_server.make_policy()
    pol.wait_ready()                  # optional: the policies are built in a background thread

The interface is identical in both worlds, so the server probes at the start of every episode (WorldProbe, ~0.6 s;
config.yaml `autoprofile`) and delegates the rest of the episode to the policy built with that world's profile
(`sim` / `none` = the robot base values). The probe only holds the arms where they were measured and curls the
fingers 0.2 rad from the start pose (<= 1.2 rad/s), inside both worlds' joint limits: no other motion.

Two independent signatures (separate controllers: Dex3 finger PD, arm PD), each voting sim / real / weak:
  hand: max |hand_q - probe posture| over the 14 joints. The sim's fingers (stiffness 100) reach a posture inside
    their soft limits to < 1 mrad; the real ones (kp 1.5) stop 0.02-0.12 rad short (no-load error, thumb_0 bias).
  arm: max |(arm_q - held command) - SIM_SAG_GAIN * gravity torque| over the 14 joints. The sim's arm settles at
    commanded + its fitted PD sag (residual ~0.1 mrad); the real arm (other gains, no feed-forward from the runner,
    friction, or gravity feed-forward as in the teleop data) misses that model by >= 8 mrad.
Decision (decide()): both sim -> sim; both real -> real; real + weak -> real; one sim + one real (contradiction) ->
the clearer vote, if it is >= CLEAR times past its limit and >= OVER times clearer than the other (a real arm with
well-tracking fingers: arm real x6, hand sim x1.7 -> real; a sim finger touching something: arm sim x15 -> sim);
anything else -> sim. Why sim when in
doubt: the sim selection comes first and decides whether the image ever reaches the robot (real config in the sim =
much lower score = no finals); the sim's values are measured and deterministic, so a sim vote on the robot needs
milliradian agreement with the sim's physics that real hardware does not show, while a sim episode can produce one
real-looking value (a finger touching something, the arm bumped). The sim config on the robot is bad too (wrong
right-hand wiring, fast motion) but every command still passes the organisers' safety filter.

Per process: the world does not change, so the first episode's decision stands; later episodes probe again and
switch only on stronger evidence (a `confident` result, or a stronger one than the one that set the decision).
One log line per episode: decision, both values with their thresholds and margins, and the process state.
Before the decision act() returns the probe's targets (= holds); a policy not built yet -> holds until it is.
Hand-over: the delegate's first act() gets the probe's last command as its start pose, and (if it has feed_probe(),
ClassicalServerPolicy) the probe result: its still arm held at a known command is the first sample of the delegate's
sag feed-forward estimate (result["sag"]: held command, measured arm / hand samples, arm_dq).
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from aaltonomous.deploy.policy_server import Policy
from aaltonomous.interface.safety import JointSpec

from .config import load_config, mirror_left_to_right
from .kinematics import ArmKinematics

HZ = 30.0
SIM, REAL, WEAK = "sim", "real", "weak"
STRENGTH = {"confident": 3, "likely": 2, "contradiction": 1, "weak": 1}
SOFT = 0.05   # the sim's soft joint limits: the middle 90 % of each URDF range


def vote(stat, lim):
    """sim / real / weak for one signature value (rad) against {sim_max, real_min}; NaN (no data) = weak."""
    if stat is None or not np.isfinite(stat):
        return WEAK
    return SIM if stat < float(lim["sim_max"]) else REAL if stat > float(lim["real_min"]) else WEAK


def margin(stat, lim, v):
    """How clear a vote is: sim -> sim_max / stat, real -> stat / real_min (x times past its limit); weak -> 0."""
    if v == SIM:
        return float(lim["sim_max"]) / max(stat, 1e-6)
    return stat / float(lim["real_min"]) if v == REAL else 0.0


CLEAR, OVER = 3.0, 2.0   # a contradiction goes to the clearer vote only if its margin is >= CLEAR and >= OVER x the other's


def decide(v_hand, v_arm, m_hand=0.0, m_arm=0.0):
    """(world, how) from the two votes and their margins (margin()); how = confident / likely / contradiction / weak
    (module docstring)."""
    vs = {v_hand, v_arm}
    if len(vs) == 1 and WEAK not in vs:
        return v_hand, "confident"
    if vs == {REAL, WEAK}:
        return REAL, "likely"
    if vs == {SIM, WEAK}:
        return SIM, "likely"
    if vs == {SIM, REAL}:
        (m, v), (mo, _) = sorted(((float(m_hand), v_hand), (float(m_arm), v_arm)), reverse=True)
        return (v if m >= CLEAR and m >= OVER * mo else SIM), "contradiction"
    return SIM, "weak"


def soft_inside(hand_cmd, spec, pad=0.0):
    """Mask of hand commands inside the sim's soft limits (+ pad): only there do the sim's fingers reach them."""
    r = spec.hand_upper - spec.hand_lower
    return (hand_cmd > spec.hand_lower + SOFT * r + pad) & (hand_cmd < spec.hand_upper - SOFT * r - pad)


def probe_posture(ac, spec):
    """(14,) probe hand target (left posture + its mirror), checked to lie inside the sim's soft limits by 0.03."""
    left = np.asarray(ac["hand_probe"], float)
    v = np.r_[left, mirror_left_to_right(left)]
    bad = [spec.hand_names[i] for i in np.flatnonzero(~soft_inside(v, spec, 0.03))]
    if bad:
        raise ValueError(f"autoprofile.hand_probe outside the sim's soft joint limits (+0.03): {bad}")
    return v


def hand_stat(hand_q, hand_cmd, spec):
    """Hand signature (rad) from samples (n, 14) of measured / commanded fingers (command held): max over the joints
    commanded inside the sim's soft limits of |mean error|. NaN without samples, such joints or hand state (all 0)."""
    h, c = np.atleast_2d(np.asarray(hand_q, float)), np.atleast_2d(np.asarray(hand_cmd, float))
    ok = soft_inside(c[-1], spec)
    if not len(h) or not ok.any() or not np.any(h != 0.0):
        return float("nan")
    return float(np.abs((h - c).mean(0))[ok].max())


def arm_stat(arm_q, arm_cmd, hand_q, kin, gain):
    """Arm signature (rad) from samples (n, 14) (command held): max over the 14 joints of |mean residual| of the
    tracking error to the sim's sag model, (measured - commanded) - gain * gravity torque (URDF, pinocchio)."""
    a, c, h = (np.atleast_2d(np.asarray(x, float)) for x in (arm_q, arm_cmd, hand_q))
    if not len(a):
        return float("nan")
    g = []
    for ai, hi in zip(a, h):
        x = kin.q0.copy()
        x[kin.arm_iq], x[kin.hand_iq] = ai, hi
        g.append(kin.pin.computeGeneralizedGravity(kin.model, kin.data, x)[kin.arm_iv].copy())
    return float(np.abs(((a - c) - np.asarray(gain, float) * np.array(g)).mean(0)).max())


class WorldProbe:
    """One episode's probe: act(obs) returns the probe's targets until the decision is due, then None (`result` set).
    Arms: the first measured pose, held. Hands: a ramp from the first measured pose to the probe posture, held.
    Samples every observation in the measurement window; the statistics use the mean error vector over them."""

    def __init__(self, ac, kin, spec, dt=1.0 / HZ):
        self.ac, self.kin, self.spec, self.dt = ac, kin, spec, dt
        self.target = probe_posture(ac, spec)
        self.gain = np.tile(np.asarray(ac["sim_sag_gain"], float), 2)
        self.arm = self.queue = self.t0 = None
        self.last = None            # (arm (14,), hand (14,)) last commanded tick
        self.samples, self.result, self.ticks = [], None, 0

    def _start(self, obs, t):
        sp = self.spec
        self.t0 = t
        self.arm = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
        h0 = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        d = float(np.abs(self.target - h0).max())
        n = max(int(self.ac.get("min_ramp_ticks", 5)), math.ceil(d / (float(self.ac["hand_vel"]) * self.dt) - 1e-9))
        self.queue = list(h0 + (self.target - h0) * (np.arange(1, n + 1) / n)[:, None])
        self.t_meas = t + max(float(self.ac["arm_settle_s"]), n * self.dt + float(self.ac["hand_settle_s"]))
        self.t_end = self.t_meas + float(self.ac["window_s"])
        self.ramp_ticks = n

    def act(self, obs):
        t = obs.get("t")
        t = self.ticks * self.dt if t is None else float(t)
        if self.t0 is None:
            self._start(obs, t)
        elif t >= self.t_meas - 1e-9:
            self._sample(obs, t)
        if self.result is None and ((t >= self.t_end - 1e-9 and len(self.samples) >= int(self.ac["min_samples"]))
                                    or t >= self.t0 + float(self.ac["max_s"])):
            self.result = self._finish(t)
        if self.result is not None:
            return None
        k = max(1, int(self.ac.get("chunk", 3)))
        hand = [self.queue.pop(0) if self.queue else self.target for _ in range(k)]
        return self._action(np.repeat(self.arm[None], k, 0), np.array(hand))

    def hold(self, k=1):
        """k ticks of the last commanded target (waiting for a policy to be built)."""
        a, h = self.last
        return self._action(np.repeat(a[None], k, 0), np.repeat(h[None], k, 0))

    def _action(self, arm, hand):
        self.last = (arm[-1].copy(), hand[-1].copy())
        self.ticks += len(arm)
        return {"arm_q_target": arm, "hand_q_target": hand, "done": False}

    def _sample(self, obs, t):
        a, h = np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float)
        dq = np.asarray(obs.get("arm_dq", np.zeros(14)), float)
        if a.shape == (14,) and h.shape == (14,) and np.isfinite(a).all() and np.isfinite(h).all():
            self.samples.append((t, a, h, dq if dq.shape == (14,) and np.isfinite(dq).all() else np.zeros(14)))

    def _finish(self, t):
        ac, n = self.ac, len(self.samples)
        a, h = np.array([s[1] for s in self.samples]), np.array([s[2] for s in self.samples])
        hs = hand_stat(h, np.repeat(self.target[None], n, 0), self.spec) if n else float("nan")
        as_ = arm_stat(a, np.repeat(self.arm[None], n, 0), h, self.kin, self.gain) if n else float("nan")
        vh, va = vote(hs, ac["hand"]), vote(as_, ac["arm"])
        world, how = decide(vh, va, margin(hs, ac["hand"], vh), margin(as_, ac["arm"], va))
        # the arm's raw static error (no model) at the start pose: on the robot the first measurement of its sag
        raw = float(np.abs((a - self.arm).mean(0)).max()) if n else float("nan")
        # the arm held still at a known command: the delegate's first sag feed-forward sample (feed_probe)
        sag = {"arm_cmd": self.arm.copy(), "arm_q": a, "hand_q": h,
               "arm_dq": np.array([s[3] for s in self.samples]).reshape(-1, 14)} if n else None
        return {"world": world, "how": how, "hand": hs, "arm": as_, "arm_raw": raw, "votes": (vh, va),
                "margins": (margin(hs, ac["hand"], vh), margin(as_, ac["arm"], va)), "n": len(self.samples),
                "t": (self.samples[0][0] - self.t0, self.samples[-1][0] - self.t0) if self.samples else None,
                "t_decide": t - self.t0, "ramp_ticks": self.ramp_ticks, "sag": sag}


class AutoProfilePolicy(Policy):
    """Server-level wrapper: one policy per world (factory(profile), built in a background thread, sim first), the
    probe per episode, delegation to the detected world's policy. ClassicalPolicy itself is unchanged."""
    name = "classical-rung1-auto"

    def __init__(self, factory, config=None, log=print, background=True):
        base = config if isinstance(config, dict) else load_config(config, profile="none")
        self.ac = base["autoprofile"]
        self.action_horizon = int(base["motion"]["chunk"])
        self.spec = JointSpec()
        self.kin = ArmKinematics(base, self.spec)      # own instance: gravity in act() (not shared with any planner)
        probe_posture(self.ac, self.spec)              # a bad probe posture fails here, not in the first episode
        self.log = log or (lambda m: None)
        self.profiles = dict(self.ac["profiles"])      # world -> profile
        self.factory = factory
        self.pols, self.errors = {}, {}
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.world, self.strength, self.history = None, 0, []
        self.episode, self.reset_args, self.reset_ep = 0, ("", None, None), {}
        self.probe, self.active, self.handed = None, None, False
        if background:
            threading.Thread(target=self._build, daemon=True).start()
        else:
            self._build()

    # ------------------------------------------------------------------ policies
    def _build(self):
        for world in sorted(self.profiles, key=lambda w: w != SIM):   # sim first: the selection runs there
            t0 = time.monotonic()
            try:
                p = self.factory(self.profiles[world])
            except Exception as e:  # keep serving with the other one (logged at every use)
                with self.lock:
                    self.errors[world] = repr(e)
                self.log(f"autoprofile: building the {world} policy (profile {self.profiles[world]!r}) failed: {e!r}")
                continue
            with self.lock:
                self.pols[world] = p
            self.log(f"autoprofile: {world} policy (profile {self.profiles[world]!r}) built in "
                     f"{time.monotonic() - t0:.1f} s")
        self.ready.set()

    def wait_ready(self, timeout=None):
        """Block until every policy is built (or failed); True if all were built."""
        self.ready.wait(timeout)
        return self.ready.is_set() and not self.errors

    def _policy(self, world):
        """(world, policy) to delegate to: the detected world's, or the other one if that failed to build;
        (world, None) while it is still being built."""
        with self.lock:
            if world in self.pols:
                return world, self.pols[world]
            if world in self.errors:
                other = next((w for w in self.pols if w != world), None)
                if other is not None:
                    return other, self.pols[other]
        return world, None

    # ------------------------------------------------------------------ Policy API
    def reset(self, instruction, task=None, seed=None):
        self.episode += 1
        self.reset_args = (instruction, task, seed)
        with self.lock:
            built = list(self.pols.items())
        for w, p in built:   # every built one: the next episode may switch (a policy built later: on first use)
            p.reset(instruction, task=task, seed=seed)
            self.reset_ep[w] = self.episode
        self.handed = False
        probe = self.world is None or bool(self.ac.get("every_episode", True))
        self.probe = WorldProbe(self.ac, self.kin, self.spec) if probe else None
        self.active = None if probe else self.world

    def act(self, obs):
        if self.probe is None and self.active is None:   # act() without reset(): as an episode start
            self.reset(obs.get("instruction", ""))
        if self.active is None:
            a = self.probe.act(obs)
            if a is not None:
                return a
            self._decide(self.probe.result)
        world, p = self._policy(self.active)
        if p is None:   # still being built: hold the last target
            if self.probe is None:   # (no probe this episode) hold the first measured pose
                sp, self.probe = self.spec, WorldProbe(self.ac, self.kin, self.spec)
                self.probe.last = (np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper),
                                   np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper))
            return self.probe.hold(int(self.ac.get("chunk", 3)))
        if self.reset_ep.get(world) != self.episode:
            p.reset(self.reset_args[0], task=self.reset_args[1], seed=self.reset_args[2])
            self.reset_ep[world] = self.episode
        waiting = getattr(p, "reset_waiting", None)
        if waiting is not None and waiting():
            # its reset is deferred until the previous episode's planning thread has ended (ClassicalServerPolicy):
            # hold the probe's last command meanwhile, never block (the runner's watchdog allows 1 s without a command)
            if self.probe is None:
                sp, self.probe = self.spec, WorldProbe(self.ac, self.kin, self.spec)
                self.probe.last = (np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper),
                                   np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper))
            return self.probe.hold(int(self.ac.get("chunk", 3)))
        if not self.handed:
            self.handed = True
            feed = getattr(p, "feed_probe", None)   # the probe's still arm -> the delegate's first sag sample
            if feed is not None and self.probe is not None and self.probe.result is not None:
                feed(self.probe.result)
            if self.probe is not None and self.probe.last is not None:
                # the delegate starts from the measured pose: give it the probe's last command instead (the safety
                # filter checks the velocity against the last COMMAND; real fingers can be the whole probe step away)
                arm, hand = self.probe.last
                obs = dict(obs, arm_q=arm.astype(np.float32), hand_q=hand.astype(np.float32))
        return p.act(obs)

    # ------------------------------------------------------------------ decision
    def _decide(self, r):
        old, s = self.world, STRENGTH[r["how"]]
        if r["world"] == self.world:
            self.strength = max(self.strength, s)
        elif self.world is None or r["how"] == "confident" or s > self.strength:
            self.world, self.strength = r["world"], s
        self.history.append(r)
        self.active = self.world
        n = {w: sum(h["world"] == w for h in self.history) for w in (SIM, REAL)}
        what = ("first decision" if old is None else "kept" if old == self.world
                else f"SWITCHED from {old}")
        ac = self.ac

        def sig(name, v, m, lim):
            val = r[name]
            return (f"{name} {val:.4f} rad [sim<{float(lim['sim_max']):g} real>{float(lim['real_min']):g}] "
                    f"{v}" + (f" x{m:.1f}" if v != WEAK else ""))
        span = "no samples" if r["t"] is None else f"{r['n']} samples at {r['t'][0]:.2f}-{r['t'][1]:.2f} s"
        self.log(f"autoprofile ep {self.episode}: {r['world'].upper()} ({r['how']}) | "
                 f"{sig('hand', r['votes'][0], r['margins'][0], ac['hand'])} | "
                 f"{sig('arm', r['votes'][1], r['margins'][1], ac['arm'])} (raw |meas-cmd| "
                 f"{r.get('arm_raw', float('nan')):.4f}) | {span}, decided at {r['t_decide']:.2f} s | "
                 f"process: {self.world} ({what}; episodes sim {n[SIM]} real {n[REAL]}) -> profile "
                 f"{self.profiles.get(self.world)!r}")
