"""Offline tests of the sag feed-forward (sag.py and its use in planner.py / policy.py / autoprofile.py): the model
(gravity torque, offset, intended <-> commanded), the online estimator (convergence, bounds, outliers, low-torque
joints, wrong priors, the friction fallback), the blend, zero clips with the compensation in plans / aborts / corrections /
pairs / evades (sim and robot priors, gains at their bounds), mock episodes in both worlds, the hand-over from the
world detector, the act() cost, and off = unchanged. Run in the sim container:
    aalto.sh py -m unittest aaltonomous/baselines/classical/test_sag.py -v
"""
import copy
import os
import time
import unittest

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.interface.safety import SafetyFilter
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .mock_world import REAL_SAG_GAIN, SIM_SAG_GAIN
from .perception import Detection
from .planner import Planner, correct_ticks
from .sag import SagEstimator, SagFF, gravity
from .test_classical import run_mock

SC = Scene.load()
CZ, H = SC.counter_z, SC.object_height()
SIM, BASE = load_config(profile="sim"), load_config(profile="none")
GSIM, GREAL = np.tile(SIM_SAG_GAIN, 2), np.tile(REAL_SAG_GAIN, 2)
LOADED = [0, 1, 2, 3, 7, 8, 9, 10]          # shoulders + elbow: > 1 Nm at most poses (wrists: 0.04-1.2 Nm)
# Mock real world: fingers on an object stop this far short of the base (cage) closed posture -- the grasp lab's held
# finger shortfall in the real-hand sim variants is 0.22-0.60 rad (the mock default 0.036 is the stiff sim's squeeze,
# which base verify.empty_below 0.12 would call empty)
REAL_HELD_GAP = 0.3


def det(color, x, y, shape="cylinder"):
    d = Detection(color, np.array([x, y, CZ + H]), 900, 900, "depth")
    d.shape = shape
    return d


def poses(kin, n, seed=0):
    """n random arm poses inside the IK bounds (14,) each."""
    return np.random.default_rng(seed).uniform(kin.lo, kin.hi, (n, 14))


def ff_cfg(cfg, **ov):
    c = copy.deepcopy(cfg)
    for k, v in ov.items():
        c["sag"][k] = v
    return c


def plant(kin, cmd, hand, gain, iters=8):
    """A PD arm without gravity compensation at rest: meas = cmd + gain * tau_g(meas) (fixed point)."""
    m = np.asarray(cmd, float).copy()
    for _ in range(iters):
        m = cmd + gain * gravity(kin, m, hand)
    return m


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = ArmKinematics(SIM)
        cls.post, cls.home = hand_postures(SIM), home_arm(SIM)

    def test_gravity(self):
        """= pinocchio's generalized gravity of the arm joints (hand included); batch = per tick; never touches the
        FK state of the instance (own Data)."""
        k, Q = self.kin, poses(self.kin, 5)
        h = self.post["closed"]
        G = gravity(k, Q, h)
        self.assertEqual(G.shape, (5, 14))
        for q, g in zip(Q, G):
            x = k.q0.copy()
            x[k.arm_iq], x[k.hand_iq] = q, h
            np.testing.assert_allclose(g, k.pin.computeGeneralizedGravity(k.model, k.model.createData(), x)[k.arm_iv])
            np.testing.assert_allclose(gravity(k, q, h), g)
        p0 = k.grasp_pose("left", Q[0])[0]
        k._fk(Q[1])
        before = k.data.oMf[k.palm["left"]].translation.copy()
        gravity(k, Q[2:], h)
        np.testing.assert_array_equal(k.data.oMf[k.palm["left"]].translation, before)
        np.testing.assert_allclose(k.grasp_pose("left", Q[0])[0], p0)
        # the hand's mass is in: open vs closed fingers change the wrist torque
        self.assertGreater(np.abs(gravity(k, Q[0], self.post["open"]) - G[0]).max(), 1e-4)

    def test_offset_round_trip_and_plant(self):
        """commanded = intended + clip(-G tau(intended)); intended(commanded(q)) = q; a PD plant with that gain settles
        at the intended pose (sim prior and robot prior, gains at both bounds)."""
        for cfg, name in ((SIM, "sim"), (BASE, "robot")):
            ff, kin = SagFF(cfg), ArmKinematics(cfg)
            Q = poses(kin, 6, 1)
            h = self.post["open"]
            for g in (ff.prior, ff.lo, ff.hi):
                with self.subTest(name, gain=float(g[0] / ff.prior[0])):
                    ff.set_applied(g)
                    C = ff.commanded(kin, Q, h)
                    off = C - Q
                    np.testing.assert_allclose(off, np.clip(-g * gravity(kin, Q, h), -ff.max_off, ff.max_off))
                    for q, c in zip(Q, C):
                        np.testing.assert_allclose(ff.intended(kin, c, h), q, atol=1e-7)
                        if np.all(np.abs(c - q) < ff.max_off - 1e-6):   # not clipped: lands exactly
                            np.testing.assert_allclose(plant(kin, c, h, g), q, atol=1e-6)
                    self.assertTrue(np.all(np.abs(off) <= ff.max_off + 1e-12))

    def test_config_values(self):
        """Profiles: sim = the fitted sim gains (= the detector's model), base = -1/kp of the runner's controller,
        sim_legacy off; the IK bounds leave max_offset to the filter's position band in both."""
        np.testing.assert_allclose(SIM["sag"]["prior"], SIM_SAG_GAIN)
        np.testing.assert_allclose(SIM["sag"]["prior"], SIM["autoprofile"]["sim_sag_gain"])
        np.testing.assert_allclose(BASE["sag"]["prior"], REAL_SAG_GAIN)
        self.assertTrue(SIM["sag"]["enabled"] and BASE["sag"]["enabled"])
        self.assertFalse(load_config(profile="sim_legacy")["sag"]["enabled"])
        from aaltonomous.interface.safety import JointSpec, SafetyConfig
        sp, m = JointSpec(), SafetyConfig().arm_pos_margin
        for cfg in (SIM, BASE):
            kin, ff = ArmKinematics(cfg), SagFF(cfg)
            room = np.minimum(kin.lo - (sp.arm_lower + m), (sp.arm_upper - m) - kin.hi)
            self.assertTrue(np.all(room >= ff.max_off - 1e-9), (cfg["profile"], room - ff.max_off))
            mag = np.abs(np.c_[ff.lo, ff.hi])
            self.assertTrue(np.all(mag.min(1) <= np.abs(ff.prior)) and np.all(mag.max(1) >= np.abs(ff.prior)))
        from aaltonomous.sim.oracle import ORACLE_OVERRIDES
        self.assertEqual(load_config(profile="none", overrides=ORACLE_OVERRIDES)["sag"], SIM["sag"])

    def test_cost(self):
        """Gravity for a chunk of 10 ticks + the offset: well under a millisecond."""
        ff, kin = SagFF(SIM), self.kin
        Q, h = poses(kin, 10, 2), np.repeat(self.post["open"][None], 10, 0)
        ff.commanded(kin, Q, h)
        t0 = time.perf_counter()
        for _ in range(200):
            ff.commanded(kin, Q, h)
        ms = 1e3 * (time.perf_counter() - t0) / 200
        print(f"\n[test_sag] feed-forward for a 10-tick chunk: {ms:.3f} ms")
        self.assertLess(ms, 1.0)


class TestEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = ArmKinematics(SIM)
        cls.h = hand_postures(SIM)["open"]
        cls.Q = poses(cls.kin, 12, 3)
        cls.T = gravity(cls.kin, cls.Q, cls.h)

    def est(self, prior=GSIM, bounds=(0.5, 2.0), **sc):
        p = np.asarray(prior, float)
        lo, hi = np.minimum(p * bounds[0], p * bounds[1]), np.maximum(p * bounds[0], p * bounds[1])
        return SagEstimator(p, lo, hi, {**SIM["sag"], **sc})

    def feed(self, e, gain, idx, noise=0.0002, seed=0, ok=None):
        rng = np.random.default_rng(seed)
        for i in idx:
            r = gain * self.T[i] + rng.normal(0, noise, 14)
            e.add(r, self.T[i], np.ones(14, bool) if ok is None else ok)
        return e

    def test_right_prior(self):
        e = self.feed(self.est(), GSIM, range(10))
        np.testing.assert_allclose(e.gain[LOADED], GSIM[LOADED], rtol=0.03)
        self.assertTrue(np.all(e.trust == 1.0))

    def test_wrong_priors_converge_in_a_few_samples(self):
        """Prior x0.3 / x3 (bounds 0.25-4 around it): loaded joints within 10 % after 3 samples, 3 % after 8."""
        for f in (0.3, 3.0):
            with self.subTest(prior=f):
                e = self.est(GSIM * f, (0.25, 4.0))
                self.feed(e, GSIM, range(3))
                np.testing.assert_allclose(e.gain[LOADED], GSIM[LOADED], rtol=0.10)
                self.feed(e, GSIM, range(3, 8), seed=1)
                np.testing.assert_allclose(e.gain[LOADED], GSIM[LOADED], rtol=0.03)

    def test_bounds(self):
        """Data from 5x the prior: the estimate stops at the upper bound (2x); never outside [lo, hi]. (Such a sample
        is also outside what any in-bound gain explains: refused as a contact, so the gain stays where it was.)"""
        e = self.est(gate=1e9)        # (gate off: the bound itself is what is tested)
        self.feed(e, 5 * GSIM, range(6))
        np.testing.assert_allclose(e.gain, 2.0 * GSIM)                 # (gains < 0: the bound of larger magnitude)
        e2 = self.est()
        self.feed(e2, 5 * GSIM, range(3))
        self.assertGreater(e2.refused, 0)
        self.assertTrue(np.all((e2.gain >= e2.lo - 1e-15) & (e2.gain <= e2.hi + 1e-15)))

    def test_contact_sample_refused(self):
        """After good samples a sample with the left arm pushed (hand resting on something: 3 joints 20 mrad off) is
        refused for that arm (logged reason) and moves nothing; the right arm's half is used."""
        e = self.feed(self.est(), GSIM, range(6))
        g0 = e.gain.copy()
        r = GSIM * self.T[6]
        r[[0, 1, 3]] += [0.02, -0.02, 0.02]
        use, why = e.add(r, self.T[6], np.ones(14, bool))
        self.assertIn("left", why)
        self.assertNotIn("right", why)
        self.assertFalse(use[:7].any())
        np.testing.assert_allclose(e.gain[:7], g0[:7], rtol=1e-12)

    def test_one_bad_sample_inside_the_bounds(self):
        """A sample the gate lets through (every joint 1.8x the true sag: inside the bounds) after 6 good ones (at a
        pose seen before) moves no loaded joint by more than 12 %; three more good samples bring it back within 3 %."""
        e = self.feed(self.est(), GSIM, range(6))
        e.add(1.8 * GSIM * self.T[2], self.T[2], np.ones(14, bool))
        np.testing.assert_allclose(e.gain[LOADED], GSIM[LOADED], rtol=0.12)
        self.feed(e, GSIM, range(7, 10), seed=2)
        np.testing.assert_allclose(e.gain[LOADED], GSIM[LOADED], rtol=0.03)

    def test_low_torque_joint_keeps_the_prior(self):
        T = self.T.copy()
        T[:, 4] = 0.1                      # left wrist roll: below min_torque in every sample
        e = self.est()
        for i in range(8):
            e.add(3 * GSIM * T[i], T[i], np.ones(14, bool))   # (whatever its error says)
        self.assertEqual(e.gain[4], GSIM[4])
        self.assertEqual(e.n[4], 0)

    def test_friction_compensates_less_never_more(self):
        """Robot parameters, error not proportional to the torque (a dead band: +-10 mrad whatever the load, on top of
        a sag 0.3x the prior): trust drops, the target gain is smaller in magnitude than the estimate, never larger.
        (With the sim's tight residual scale such samples are refused instead: no change at all.)"""
        e = self.est(GREAL, (0.3, 3.0), **{k: v for k, v in BASE["sag"].items() if k not in ("prior", "bounds")})
        rng = np.random.default_rng(4)
        for i in range(10):
            r = 0.3 * GREAL * self.T[i] + 0.010 * rng.choice([-1.0, 1.0], 14)
            e.add(r, self.T[i], np.ones(14, bool))
        self.assertLessEqual(e.refused, 3)
        self.assertLess(e.trust[LOADED].min(), 0.9)
        ff = SagFF(BASE)
        ff.est = e
        tg = ff.target()
        self.assertTrue(np.all(np.abs(tg) <= np.abs(e.gain) + 1e-15))
        self.assertTrue(np.all(np.sign(tg) * np.sign(GREAL) >= 0))

    def test_samples_and_flush(self):
        """observe() accumulates a stretch (same command, same joints); a different command flushes it as ONE sample
        (mean); the stretch's error is measured - commanded."""
        ff, kin = SagFF(SIM), self.kin
        c, ok = self.Q[0], np.r_[np.ones(7, bool), np.zeros(7, bool)]
        m = plant(kin, c, self.h, GSIM)
        for k in range(4):
            self.assertIsNone(ff.observe(kin, m + 1e-5 * k, c, self.h, ok, "home", 0.1 * k))
        res = ff.observe(kin, m, self.Q[1], self.h, ok, "home", 1.0)
        self.assertEqual((res[0], res[3]), ("home", 4))
        self.assertEqual(len(ff.est.samples), 1)
        np.testing.assert_allclose(ff.est.samples[0]["r"][:7], (m - c)[:7] + 1.5e-5, atol=1e-12)
        self.assertFalse(ff.est.samples[0]["ok"][7:].any())


class TestBlend(unittest.TestCase):
    def test_blend(self):
        """A new estimate moves the commanded offset at <= sag.blend_vel, ends exactly at target(), bumps the
        version once; a change below blend_tol is applied at once (n = 0)."""
        kin, post, home = ArmKinematics(SIM), hand_postures(SIM), home_arm(SIM)
        ff = SagFF(SIM)
        q = poses(kin, 1, 5)[0]
        ff.est.gain = ff.prior * 1.8
        v0 = ff.version
        n = ff.begin_blend(kin, q, post["open"])
        self.assertGreater(n, 1)
        prev, offs = ff.offset(kin, q, post["open"]), []
        for _ in range(n):
            g = ff.tick_gains(1)
            offs.append(ff.offset(kin, q, post["open"], g[0]))
        d = np.abs(np.diff(np.vstack([prev, offs]), axis=0)).max()
        self.assertLessEqual(d, SIM["sag"]["blend_vel"] / 30 + 1e-9)
        np.testing.assert_allclose(ff.applied, ff.target())
        self.assertEqual((ff.version, ff.blend, ff.tick_gains(3)), (v0 + 1, None, None))
        ff.est.gain = ff.applied * 1.0001
        self.assertEqual(ff.begin_blend(kin, home, post["open"]), 0)
        np.testing.assert_allclose(ff.applied, ff.target())


def planner(cfg, gain=None, kin=None):
    """Planner with the feed-forward attached, applied gain `gain` (factor of the prior, or None = the prior)."""
    post, home = hand_postures(cfg), home_arm(cfg)
    p = Planner(cfg, SC, post, home, kin or ArmKinematics(cfg))
    p.ff = SagFF(cfg)
    if gain in ("lo", "hi"):
        p.ff.set_applied(p.ff.lo if gain == "lo" else p.ff.hi)
    elif gain is not None:
        p.ff.set_applied(p.ff.prior * gain)
    return p, post, home


class TestZeroClips(unittest.TestCase):
    """Every planned queue passes the safety filter AS SENT (commanded ticks), at the prior and with the gain at
    either bound, sim and robot: plans, aborts, corrections, pairs, evades, the move home."""

    def check_cmd(self, P, arm_q, hand_q, arm, hand, what):
        C = P.commanded(np.vstack([arm_q[None], arm]), np.vstack([hand_q[None], hand]))
        f = SafetyFilter(P.spec, P.safety)
        f.reset(C[0], hand_q, 0.0)
        for k in range(1, len(C)):
            f.filter({"arm_q_target": C[k], "hand_q_target": hand[k - 1]}, k / 30)
        self.assertEqual(f.clipped_steps, 0, what)
        self.assertLessEqual(np.abs(np.diff(C, axis=0)).max() * 30, 1.0 - 1e-6, what)
        self.assertTrue(np.all((C[1:] >= f.arm_lo - 1e-9) & (C[1:] <= f.arm_hi + 1e-9)), what)   # no silent clamp
        return C

    def test_plans_aborts_corrections(self):
        for cfg, name in ((SIM, "sim"), (BASE, "robot")):
            for gain in (None, "lo", "hi"):
                P, post, home = planner(cfg, gain)
                A, _, _ = planner(cfg, gain)          # the act()-thread instances
                A.ff = P.ff
                n_ok = 0
                for d in (det("red", 0.29, 0.14), det("yellow", 0.30, -0.06), det("blue", 0.33, -0.17, "cube")):
                    with self.subTest(name, gain=gain, obj=(d.color, d.top[1])):
                        p = P.plan(d, home, post["open"], relocate=False)
                        if not p.ok:
                            self.assertNotIn("clips", p.reason)
                            continue
                        n_ok += 1
                        self.assertEqual(P.dry_run_safety(home, post["open"], p.arm, p.hand), 0)
                        C = self.check_cmd(P, home, post["open"], p.arm, p.hand, "plan")
                        self.assertLessEqual(np.abs(np.diff(C, axis=0)).max() * 30, P.ff.cap + 1e-6)
                        for k in range(10, len(p.arm) - 10, max(1, len(p.arm) // 6)):   # aborts along the plan
                            a, h, info = A.abort(p.side, p.arm[k - 1], p.hand[k - 1], p.arm[k:], p.hand[k:])
                            if a is None:
                                self.assertNotIn("clips", info)
                            else:
                                self.check_cmd(A, p.arm[k - 1], p.hand[k - 1], a, h, f"abort at {k}")
                        ph = {n: (s, e) for n, s, e in p.phases}
                        if "settle" in ph:                       # a correction at the measurement tick
                            k = ph["close"][0] - correct_ticks(cfg["grasp"]["correct"])[1]
                            phases = [(n, s - k, e - k) for n, s, e in p.phases if e > k and s >= ph["settle"][0]]
                            _, Jp, Jw, _ = A.kin.grasp_jacobian(p.side, p.arm[k - 1])   # grasp point 6 mm up / back
                            dq = np.linalg.lstsq(np.vstack([Jp, Jw]), np.r_[-0.004, 0.0, 0.004, np.zeros(3)],
                                                 rcond=None)[0]
                            # a joint at its IK bound (base: ik.joint_range, the real arm's envelope; the cage's
                            # grasps put the wrist roll there) is not pushed past it -- as correction_dq pins it
                            sl = slice(0, 7) if p.side == "left" else slice(7, 14)
                            dq = np.clip(dq, A.kin.lo[sl] - p.arm[k - 1, sl], A.kin.hi[sl] - p.arm[k - 1, sl])
                            new, info = A.correct(p.side, p.arm[k - 1], p.hand[k - 1], p.arm[k:], p.hand[k:], dq,
                                                  phases)
                            self.assertIsNotNone(new, info)
                            self.check_cmd(A, p.arm[k - 1], p.hand[k - 1], new, p.hand[k:], "correction")
                self.assertGreater(n_ok, 0, (name, gain))

    def test_pairs_evades_and_home(self):
        for gain in (None, "hi", "lo"):
            P, post, home = planner(SIM, gain)
            with self.subTest("pair", gain=gain):
                first = P.plan(det("red", 0.30, 0.15), home, post["open"], relocate=False)
                self.assertTrue(first.ok, first.reason)
                m, p2, why = P.plan_pair(first, det("blue", 0.30, -0.15), "right", home, post["open"])
                self.assertIsNotNone(m, why)
                self.assertEqual(m.safety_clips, 0)
                self.check_cmd(P, home, post["open"], m.arm, m.hand, "pair")
            with self.subTest("evade", gain=gain):
                evades = 0
                for color, y in (("red", -0.16), ("red", -0.08), ("blue", 0.16), ("blue", 0.05)):  # cross-body
                    p = P.plan(det(color, 0.30, y), home, post["open"])
                    if not p.ok:
                        self.assertNotIn("clips", p.reason)
                        continue
                    m, info = P.keep_clear(p, home, post["open"])
                    if m is None:
                        self.assertNotIn("clips", info.get("why", ""))
                        continue
                    evades += bool(m.evade)
                    self.check_cmd(P, home, post["open"], m.arm, m.hand, f"evade {color} {y}")
                self.assertGreater(evades, 0)
            with self.subTest("go_to", gain=gain):
                start = poses(P.kin, 1, 7)[0]
                a, h = P.go_to(start, post["open"], home, post["open"])
                C = P.commanded(np.vstack([start, a]), post["open"])
                self.assertLessEqual(np.abs(np.diff(C, axis=0)).max() * 30, P.ff.cap + 1e-6)

    def test_off_is_bit_identical(self):
        """Feed-forward attached but disabled (or absent): the same plan, tick for tick, and the plain dry run."""
        cfg = ff_cfg(SIM, enabled=False)
        P1, post, home = planner(cfg)
        P0 = Planner(cfg, SC, post, home, P1.kin)
        d = det("red", 0.29, 0.14)
        p0, p1 = P0.plan(d, home, post["open"]), P1.plan(d, home, post["open"])
        np.testing.assert_array_equal(p0.arm, p1.arm)
        np.testing.assert_array_equal(p0.hand, p1.hand)
        self.assertEqual(p0.phases, p1.phases)
        np.testing.assert_array_equal(P1.commanded(p1.arm, p1.hand), p1.arm)


def _mock(world, seed, ov=None, **kw):
    cfg = load_config(profile="sim" if world == "sim" else "none", overrides=ov)
    real = {"held_gap": REAL_HELD_GAP} if world == "real" else {}
    pol, env, info, _ = run_mock(seed, {"world": world, **real, **kw.pop("env_kw", {})}, config=cfg, **kw)
    return pol, env, info


def pre_corr(pol):
    """Grasp-point error (m) at the end of every settle, before any correction (FK(intended) - FK(measured))."""
    return np.array([np.linalg.norm(c["err_mm"]) / 1000 for c in pol.corrections])


class TestEpisodes(unittest.TestCase):
    """Mock worlds (mock_world.SIM_WORLD / REAL_WORLD: the arm settles where kp (cmd - meas) = tau_g(meas)) with the
    feed-forward off / on / from wrong priors: the grasp point before the correction step, deliveries, 0 clips."""

    def check(self, pol, env, info, world):
        self.assertEqual(info["end_reason"], "done", info)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        out = env.backend.outcomes(pol.mapping)
        self.assertEqual(out.count("correct"), len(out), out)
        if pol.ff is not None:
            self.assertEqual(pol.ff_net, 0)
            true = GSIM if world == "sim" else GREAL
            np.testing.assert_allclose(pol.ff.applied[LOADED], true[LOADED], rtol=0.15)
            self.assertEqual(pol.ff.est.refused, 0)

    def test_sim_world(self):
        """Seed 0: without feed-forward the grasp point is 5-12 mm off at the end of every settle; with it < 0.5 mm
        (the correction then has nothing to do: below min_err); the same deliveries, same duration +- 1 s."""
        off = _mock("sim", 0, {"sag.enabled": False})
        on = _mock("sim", 0)
        for r in (off, on):
            self.check(*r, "sim")
        e0, e1 = pre_corr(off[0]), pre_corr(on[0])
        self.assertGreater(e0.min(), 0.005)
        self.assertLess(e1.max(), 0.0005)
        self.assertEqual(sum(c["applied"] for c in on[0].corrections), 0)
        self.assertLess(abs(on[2]["t"] - off[2]["t"]), 1.0)
        notes = [m for _, m in on[0].events if m.startswith("sag")]
        self.assertTrue(notes and "gain/prior" in notes[0], notes)       # logged once at the start (prior: no change)

    def test_real_world(self):
        """The robot as the runner drives it (kp 80 / 40, no feed-forward; dead band, noise, 2-tick delay): without
        feed-forward the grasp point is 2-5 cm off (every correction refused: > max_err); with the robot prior and
        online learning <= 4 mm before the correction."""
        off = _mock("real", 0, {"sag.enabled": False})
        on = _mock("real", 0)
        self.check(*off, "real")
        self.check(*on, "real")
        e0, e1 = pre_corr(off[0]), pre_corr(on[0])
        self.assertGreater(np.median(e0), 0.02)
        self.assertFalse(any(c["applied"] for c in off[0].corrections))
        self.assertLess(e1.max(), 0.004)

    def test_real_world_wrong_priors(self):
        """Prior x0.3 and x3 (bounds 0.25-4): the first settles teach it (re-settle before the first look), every
        grasp point <= 6 mm before the correction, gains within 15 % on the loaded joints, 0 clips."""
        for f in (0.3, 3.0):
            with self.subTest(prior=f):
                pol, env, info = _mock("real", 0, {"sag.prior": list(f * np.asarray(REAL_SAG_GAIN)),
                                                   "sag.bounds": [0.25, 4.0]})
                self.check(pol, env, info, "real")
                self.assertLess(pre_corr(pol).max(), 0.006)
                self.assertTrue(any("gain/prior ->" in m for _, m in pol.events))   # the material change, logged

    def test_real_world_that_does_not_sag(self):
        """The model contradicted (a robot that already compensates gravity: no sag, only the dead band): the
        feed-forward switches itself off (target <= 0.35x the prior on the loaded joints) instead of lifting the hand
        by the prior's 2-5 cm; 0 clips; the later grasp points <= 5 mm off before the correction."""
        pol, env, info = _mock("real", 0, env_kw={"sag_gain": None})
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        out = env.backend.outcomes(pol.mapping)
        self.assertGreaterEqual(out.count("correct"), len(out) - 1, out)
        self.assertTrue(np.all(np.abs(pol.ff.target()[LOADED]) <= 0.35 * np.abs(GREAL[LOADED])), pol.ff.text())
        self.assertLess(pre_corr(pol)[1:].max(), 0.005)

    def test_correction_is_not_learned_as_sag(self):
        """Sim world, prior x0.6, no learning: the corrections are applied (they trim what the prior leaves); with
        learning the estimate ends at the true gain although corrections shifted the queue (samples = measured vs
        what was sent, taken before the shift)."""
        ov = {"sag.prior": list(0.6 * np.asarray(SIM_SAG_GAIN)), "sag.bounds": [0.25, 4.0]}
        fixed, _, info = _mock("sim", 2, {**ov, "sag.learn": False})
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertGreater(sum(c["applied"] for c in fixed.corrections), 0)
        np.testing.assert_allclose(fixed.ff.applied, fixed.ff.prior)
        pol, env, info = _mock("sim", 2, ov)
        self.check(pol, env, info, "sim")
        np.testing.assert_allclose(pol.ff.applied[LOADED], GSIM[LOADED], rtol=0.03)
        srcs = {s["src"] for s in pol.ff.est.samples}
        self.assertTrue({"home", "grasp"} <= srcs, srcs)

    def test_act_cost(self):
        """Feed-forward share of act() (gravity + offset + net per popped chunk): well under 1 ms per tick."""
        t = {}
        for ov, tag in (({"sag.enabled": False}, "off"), (None, "on")):
            acc = [0.0, 0]

            def setup(pol, acc=acc):
                orig = pol._pop

                def timed():
                    t0 = time.perf_counter()
                    out = orig()
                    acc[0] += time.perf_counter() - t0
                    acc[1] += len(out["arm_q_target"])
                    return out
                pol._pop = timed
            _mock("sim", 1, ov, setup=setup)
            t[tag] = 1e3 * acc[0] / max(acc[1], 1)
        print(f"\n[test_sag] _pop per tick: off {t['off']:.4f} ms, on {t['on']:.4f} ms")
        self.assertLess(t["on"] - t["off"], 0.2)


class TestHandover(unittest.TestCase):
    def test_feed_probe(self):
        """A probe result (autoprofile.WorldProbe: held command, measured samples) -> one 'probe' sample, used from the
        first act(): the first target sent is the probe's command itself (no step), the intended start pose is
        where that command makes the arm settle."""
        from .policy import ClassicalPolicy
        cfg = load_config(profile="none", overrides={"sag.prior": list(3 * np.asarray(REAL_SAG_GAIN)),
                                                     "sag.bounds": [0.25, 4.0]})
        pol = ClassicalPolicy(config=cfg, log=None)
        kin, post = pol.act_kin, pol.post
        c = home_arm(SIM)
        m = plant(kin, c, post["open"], GREAL)
        n = 4
        r = {"sag": {"arm_cmd": c, "arm_q": np.repeat(m[None], n, 0), "hand_q": np.repeat(post["open"][None], n, 0),
                     "arm_dq": np.zeros((n, 14))}}
        pol.reset({"instruction": "", "sag_probe": r})
        self.assertEqual([s["src"] for s in pol.ff.est.samples], ["probe"])
        np.testing.assert_allclose(pol.ff.target()[LOADED], GREAL[LOADED], rtol=0.2)    # one sample: x3 -> ~x1
        out = pol.act({"arm_q": c, "hand_q": post["open"], "arm_dq": np.zeros(14), "t": 0.0})
        np.testing.assert_allclose(pol.ff.applied, pol.ff.target())
        i0 = pol.ff.intended(kin, c, post["open"])
        np.testing.assert_allclose(pol.ff.commanded(kin, i0, post["open"]), c, atol=1e-7)
        np.testing.assert_allclose(i0[LOADED], m[LOADED], atol=3e-3)              # intended start = where it is
        self.assertLessEqual(np.abs(out["arm_q_target"][0] - c).max(), 1.0 / 30)   # the move home starts from c
        self.assertIsNone(pol.feed_probe({"sag": None}))

    def test_through_the_wrapper(self):
        """AutoProfilePolicy in the mock real world: the probe's still arm reaches the real-world delegate as its first
        sample; the filter sees no step at the hand-over; the episode ends with 0 clips."""
        from .autoprofile import AutoProfilePolicy
        from .mock_world import make_mock_env
        from .policy_server import ClassicalServerPolicy

        def unthreaded(p):
            p.pol.threaded = False
            return p
        pol = AutoProfilePolicy(lambda p: unthreaded(ClassicalServerPolicy(profile=p)), log=None, background=False)
        env = make_mock_env(seed=1, world="real", cfg=load_config(profile="none"), held_gap=REAL_HELD_GAP)
        obs = env.reset()
        pol.reset(obs["instruction"])
        while True:
            obs, info = env.step(pol.act(obs))
            if info["episode_over"]:
                break
        d = pol.pols["real"].pol
        self.assertEqual(pol.history[-1]["world"], "real")
        self.assertEqual(d.ff.est.samples[0]["src"], "probe")
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        np.testing.assert_allclose(d.ff.applied[LOADED], GREAL[LOADED], rtol=0.15)
        out = env.backend.outcomes(d.mapping)
        self.assertGreaterEqual(out.count("correct"), len(out) - 1, out)


if __name__ == "__main__":
    unittest.main()
