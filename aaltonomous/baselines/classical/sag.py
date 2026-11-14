"""Gravity-sag feed-forward for the arm commands (config `sag`), with the gain learned online.

The arms are position-controlled without gravity compensation in both worlds (sim: PD, ~276-412 Nm/rad; robot: the
runner's G1_29_ArmController, kp 80 shoulders / elbow, 40 wrists, torque feed-forward 0), so a held arm settles at
    measured = commanded + G * tau_g(q)       per joint, G < 0 (rad/Nm), tau_g = URDF gravity torque (arm + hand)
The policy keeps every plan, check and comparison in INTENDED joint space (where the arm should really be) and sends
    commanded = intended + offset,   offset = clip(-G * tau_g(intended), +-sag.max_offset)
so the arm settles where it was planned (sim: residual ~0.2 mrad per joint; held 50 g object not modelled: <= 0.15 Nm
at the shoulder, < 1 mm at the hand, and only while carrying).

    gravity(kin, arm, hand)   (T, 14) Nm; a pinocchio Data of its own per kinematics instance (thread-safe as the
                              instance itself: one per thread, as the planners already are)
    SagEstimator              per joint, robust least squares of (measured - commanded) against tau_g over recent
                              still samples, with the prior as a pseudo-observation, bounds and a trust factor
    SagFF                     the shared state: the APPLIED gain (frozen while a plan is planned or executed; the
                              policy changes it only in the settle before a look, blended), the estimate, offsets,
                              the inverse (intended from a command), still-stretch accumulation, log text

Estimator rule (SagEstimator.fit), per joint j over the last sag.window accepted samples (newest weight 1, each older
sample of that joint x sag.forget):  G_j = argmin  lam (G - G0_j)^2 + sum_i w_i huber_c(r_ij - G tau_ij),  clipped to the bounds
(sag.bounds x the prior), started from the weighted median of the samples' implied gains and the prior (a wrong prior
does not slow it down; a single outlier among good samples does not drag it); lam = prior_weight * min_torque^2 (the
prior counts like prior_weight samples at the least informative torque); a sample's weight counts its torque only up
to sag.lever_torque (no single high-torque sample outweighs the rest). Joints whose samples have |tau| < min_torque
get no information from them and keep the prior. A sample is refused for an arm (logged) if its error cannot be
explained by any gain inside the bounds on 2+ well-loaded joints (hand resting on something, a collision); single
inconsistent joints are dropped. Fallback: if the fitted joint error is not proportional to the torque (robust
residual s vs the predicted sag m; friction / dead band), trust_j = 1 at (s - noise) / m <= misfit[0] down to 0 at misfit[1]: the applied gain is
trust * G, i.e. compensate less, never more; where the data imply a gain nearer zero than the bounds allow, trust =
unbounded fit / that bound (0 for the wrong sign: e.g. a robot that already compensates gravity itself); a joint whose
samples are inconsistent sag.refuse_run times in a row (the probe counts twice) gets trust 0 until one fits again.
"""
from __future__ import annotations

import numpy as np


def gravity(kin, arm, hand):
    """Gravity torque (Nm) of the 14 arm joints at arm (T, 14) / (14,) with hand (T, 14) / (14,) (pelvis fixed, legs /
    waist at kin.q0, right-hand slots as the kinematics model them). Same leading shape as `arm`."""
    data = getattr(kin, "_gdata", None)
    if data is None:     # own Data: never disturbs the FK state the instance's other users rely on
        data = kin._gdata = kin.model.createData()
    A = np.asarray(arm, float)
    one = A.ndim == 1
    A = A.reshape(-1, 14)
    Hh = np.broadcast_to(np.asarray(hand, float).reshape(-1, 14), A.shape)
    q = np.repeat(kin.q0[None], len(A), 0)
    q[:, kin.arm_iq], q[:, kin.hand_iq] = A, Hh
    g, iv = kin.pin.computeGeneralizedGravity, kin.arm_iv
    out = np.array([g(kin.model, data, x)[iv] for x in q]).reshape(-1, 14)
    return out[0] if one else out


def _tile7(v, n=None):
    """Config value: scalar, 7 (LEFT order, the right arm the same) or 14 values -> (14,)."""
    v = np.asarray(v, float).ravel()
    return np.full(14, float(v[0])) if v.size == 1 else np.tile(v, 2) if v.size == 7 else v.reshape(14)


def _wmedian(v, w):
    """Weighted median of v (n,) with weights w (n,) >= 0 (the lower one at an exact half)."""
    i = np.argsort(v)
    c = np.cumsum(w[i])
    return float(v[i][np.searchsorted(c, 0.5 * c[-1])])


class SagEstimator:
    """Online per-joint gain estimate (module docstring). add() one still sample at a time; gain, trust, n (samples
    per joint), info (sum w tau^2 per joint)."""

    def __init__(self, prior, lo, hi, sc):
        self.prior, self.lo, self.hi = (np.asarray(x, float).copy() for x in (prior, lo, hi))
        self.sc = sc
        self.c = _tile7(sc.get("huber", 0.002))
        self.noise = _tile7(sc.get("noise", 0.0005))
        self.min_tau = float(sc.get("min_torque", 0.3))
        self.lam = float(sc.get("prior_weight", 1.0)) * self.min_tau ** 2   # = that many samples at min_torque
        self.reset()

    def reset(self):
        self.samples = []       # accepted: dict(r (14,), tau (14,), ok (14,) bool, src, t)
        self.gain, self.trust = self.prior.copy(), np.ones(14)
        self.n, self.info, self.misfit = np.zeros(14, int), np.zeros(14), np.zeros(14)
        self.refused = 0
        self.run = np.zeros(14, int)        # per joint: inconsistent samples in a row (the probe's count 2)
        self.off = np.zeros(14, bool)       # joints whose data keep contradicting the model: trust 0 until one fits

    def check(self, r, tau, ok):
        """Per-joint (use, inconsistent) masks of a sample and per-arm refusal reasons: a joint is inconsistent if no
        gain inside the bounds explains its error within sag.gate * huber (rad); 2+ such joints refuse the arm."""
        use = ok & (np.abs(tau) >= self.min_tau)
        g = np.clip(np.where(use, r / np.where(use, tau, 1.0), 0.0), self.lo, self.hi)
        bad = use & (np.abs(r - g * tau) > float(self.sc.get("gate", 3.0)) * self.c)
        why = {}
        for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
            if bad[sl].sum() >= 2:
                j = np.flatnonzero(bad[sl])
                why[side] = (f"{len(j)} joints off the model by {1000 * np.abs(r - g * tau)[sl][j].max():.1f} mrad")
                use[sl] = False
        return use & ~bad, bad, why

    def add(self, r, tau, ok, src="", t=0.0):
        """One still sample: r = measured - commanded (14,), tau (14,) at the measured pose, ok (14,) the joints
        sampled (that arm still). Returns (accepted joint mask, {side: why refused})."""
        r, tau = np.asarray(r, float), np.asarray(tau, float)
        use, bad, why = self.check(r, tau, np.asarray(ok, bool))
        self.refused += len(why)
        self.run = np.where(bad, self.run + (2 if src == "probe" else 1), np.where(use, 0, self.run))
        self.off = self.run >= int(self.sc.get("refuse_run", 2))
        if use.any():
            self.samples.append({"r": r.copy(), "tau": tau.copy(), "ok": use, "src": src, "t": float(t)})
            del self.samples[:-int(self.sc.get("window", 16))]
            self.fit()
        return use, why

    def trusted(self):
        """trust (14,) with the fallback for a joint whose samples are inconsistent sag.refuse_run times in a row: 0
        (the model does not describe it: no feed-forward there rather than the prior's)."""
        return np.where(self.off, 0.0, self.trust)

    def fit(self):
        S = self.samples
        if not S:
            self.gain, self.trust = self.prior.copy(), np.ones(14)
            return
        R, T = np.array([s["r"] for s in S]), np.array([s["tau"] for s in S])
        M = np.array([s["ok"] for s in S])
        # per joint: age = newer samples OF THAT JOINT (a sample refused for one arm does not age the other's data);
        # leverage capped: a sample's weight counts its torque up to lever_torque (one high-torque sample, e.g. a
        # grasp settle with the hand on something, cannot outweigh the rest)
        age = np.cumsum(M[::-1], axis=0)[::-1] - 1
        lev = float(self.sc.get("lever_torque", 2.0))
        W = float(self.sc.get("forget", 0.9)) ** np.maximum(age, 0) * M * np.minimum(
            1.0, lev ** 2 / np.maximum(T * T, 1e-12))         # (n, 14), 0 where a joint has no data
        G = self.prior.copy()
        for j in np.flatnonzero(M.any(0)):                    # robust start: weighted median of implied gains + prior
            k = M[:, j]
            G[j] = _wmedian(np.r_[self.prior[j], R[k, j] / T[k, j]], np.r_[self.lam, W[k, j] * T[k, j] ** 2])
        G = np.clip(G, self.lo, self.hi)
        for _ in range(int(self.sc.get("irls", 6))):
            E = R - G * T
            h = W * np.minimum(1.0, self.c / np.maximum(np.abs(E), 1e-12))
            G = np.clip((self.lam * self.prior + (h * T * R).sum(0)) / (self.lam + (h * T * T).sum(0)),
                        self.lo, self.hi)
        E = R - G * T
        h = W * np.minimum(1.0, self.c / np.maximum(np.abs(E), 1e-12))
        hs = h.sum(0)
        s = np.sqrt((h * E * E).sum(0) / np.maximum(hs, 1e-12))
        m = np.sqrt((W * (G * T) ** 2).sum(0) / np.maximum(W.sum(0), 1e-12))
        q = np.maximum(s - self.noise, 0.0) / np.maximum(m, 1e-12)
        q0, q1 = (float(x) for x in self.sc.get("misfit", [0.3, 1.0]))
        n = M.sum(0)
        trust = np.where(n >= int(self.sc.get("min_samples_trust", 3)), np.clip((q1 - q) / (q1 - q0), 0.0, 1.0), 1.0)
        # the data say less sag than the bound nearer zero allows (e.g. a robot that compensates gravity itself):
        # applied = the unbounded fit there (trust = fit / bound, 0 for the wrong sign), never the bound
        free = (self.lam * self.prior + (h * T * R).sum(0)) / (self.lam + (h * T * T).sum(0))
        near = np.where(np.abs(self.lo) < np.abs(self.hi), self.lo, self.hi)
        low = np.where(M.any(0) & (near != 0), np.clip(free / np.where(near != 0, near, 1.0), 0.0, 1.0), 1.0)
        trust = np.minimum(trust, low)
        self.gain, self.trust, self.n, self.info, self.misfit = G, trust, n, (W * T * T).sum(0), q


class SagFF:
    """Feed-forward state shared by the policy and its planners (threads: see the module docstring). on = sag.enabled.
    applied (14,) = the gain every command uses now; version = bumped when it changes (plans checked against an older
    version are re-checked). offset() / commanded() / intended() take the caller's own kinematics instance."""

    def __init__(self, cfg):
        sc = dict(cfg.get("sag") or {})
        self.sc, self.on, self.learn = sc, bool(sc.get("enabled")), bool(sc.get("learn", True))
        self.prior = _tile7(sc.get("prior", 0.0))
        f0, f1 = (float(x) for x in sc.get("bounds", [0.5, 2.0]))
        self.lo, self.hi = np.minimum(self.prior * f0, self.prior * f1), np.maximum(self.prior * f0, self.prior * f1)
        self.max_off = _tile7(sc.get("max_offset", 0.05))
        mv = sc.get("max_cmd_vel")
        self.cap = float(mv if mv is not None else cfg["motion"]["arm_max_vel"])  # commanded joint speed (planning)
        self.still_ticks = int(np.ceil(float(sc.get("still_s", 0.3)) * 30.0 - 1e-9))
        self.est = SagEstimator(self.prior, self.lo, self.hi, sc)
        self.applied, self.version, self.blend, self.stretch = self.prior.copy(), 0, None, None
        self.logged = self.prior.copy()     # applied gain last written to the log (the policy's)

    # ------------------------------------------------------------------ the model
    def offset(self, kin, arm, hand, gain=None):
        """commanded - intended at intended arm (T, 14) / (14,), hand alike; gain (14,) / (T, 14), default applied."""
        g = self.applied if gain is None else np.asarray(gain, float)
        return np.clip(-g * gravity(kin, arm, hand), -self.max_off, self.max_off)

    def commanded(self, kin, arm, hand, gain=None):
        return np.asarray(arm, float) + self.offset(kin, arm, hand, gain)

    def intended(self, kin, cmd, hand, gain=None, iters=30, tol=1e-10):
        """The intended pose whose command is `cmd` (fixed point: the offset changes by <= ~0.4 mrad per mrad of
        pose even at the robot's upper bound, so it contracts)."""
        cmd = np.asarray(cmd, float)
        x = cmd.copy()
        for _ in range(iters):
            x, x0 = cmd - self.offset(kin, x, hand, gain), x
            if np.abs(x - x0).max() < tol:
                break
        return x

    def target(self):
        """The gain the next blend goes to: estimate x trust (learn), else the prior."""
        return np.clip(self.est.gain * self.est.trusted(), np.minimum(self.lo, 0), np.maximum(self.hi, 0)) \
            if self.learn else self.prior.copy()

    # ------------------------------------------------------------------ changing the applied gain
    def set_applied(self, g):
        g = np.asarray(g, float).copy()
        if np.any(g != self.applied):
            self.applied, self.version = g, self.version + 1
        self.blend = None

    def begin_blend(self, kin, arm, hand, dt=1.0 / 30.0):
        """Start moving the applied gain to target() while the arm holds intended `arm`: returns the number of ticks
        it takes (commanded offset changing <= sag.blend_vel rad/s per joint), 0 if the change is below sag.blend_tol
        (then applied at once: a step of that size is far inside the filter's 33 mrad per tick)."""
        g1 = self.target()
        if np.all(g1 == self.applied):
            return 0
        d = float(np.abs((g1 - self.applied) * gravity(kin, arm, hand)).max())   # (unclipped: the gain moves
        if d <= float(self.sc.get("blend_tol", 0.0002)):                         # linearly, the clip only slows it)
            self.set_applied(g1)
            return 0
        n = int(np.ceil(d / (float(self.sc.get("blend_vel", 0.1)) * dt) - 1e-9))
        self.blend = {"g0": self.applied.copy(), "g1": g1, "n": n, "k": 0}
        return n

    def tick_gains(self, n):
        """Gains of the next n popped ticks (n, 14), advancing a running blend (None: none running = applied)."""
        b = self.blend
        if b is None:
            return None
        f = np.minimum(1.0, (b["k"] + np.arange(1, n + 1)) / b["n"])[:, None]
        b["k"] += n
        out = b["g0"] + f * (b["g1"] - b["g0"])
        if b["k"] >= b["n"]:
            self.set_applied(b["g1"])
        return out

    # ------------------------------------------------------------------ still samples
    def observe(self, kin, meas, cmd, hand, ok, src, t):
        """Accumulate one still observation into the current stretch (same command and joints, same source); a
        different one ends the stretch first (flush). Returns flush()'s result or None."""
        out = None
        s = self.stretch
        if s is not None and (s["src"] != src or np.any(s["ok"] != ok) or np.any(s["cmd"][ok] != cmd[ok])):
            out = self.flush(kin)
            s = None
        if s is None:
            s = self.stretch = {"src": src, "ok": np.asarray(ok, bool).copy(), "cmd": np.asarray(cmd, float).copy(),
                                "m": np.zeros(14), "h": np.zeros(14), "n": 0, "t": float(t)}
        s["m"] += meas
        s["h"] += hand
        s["n"] += 1
        return out

    def flush(self, kin):
        """End the current stretch: its mean -> one estimator sample. Returns None (no stretch) or
        (src, accepted mask, refusals, n observations)."""
        s, self.stretch = self.stretch, None
        if s is None or not self.learn:
            return None
        m, h = s["m"] / s["n"], s["h"] / s["n"]
        use, why = self.est.add(m - s["cmd"], gravity(kin, m, h), s["ok"], s["src"], s["t"])
        return s["src"], use, why, s["n"]

    # ------------------------------------------------------------------ text
    def text(self, g=None):
        """'L x1.00 0.98 ... | R ...' = gain / prior per joint (1 = the prior)."""
        g = self.applied if g is None else g
        r = np.where(self.prior != 0, g / np.where(self.prior != 0, self.prior, 1.0), 0.0)
        return "L " + " ".join(f"{v:.2f}" for v in r[:7]) + " | R " + " ".join(f"{v:.2f}" for v in r[7:])
