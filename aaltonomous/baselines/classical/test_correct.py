"""Offline tests of settle and correct (grasp.correct): the settle phase, the correction maths, every guard, pairs,
aborts / verification after a correction, and mock episodes with the sim's arm sag (mock_world.SIM_SAG_GAIN). Run in
the sim container:
    aalto.sh py -m unittest aaltonomous/baselines/classical/test_correct.py -v
Profile sim explicitly (the correction is pinned off in sim_legacy, the profile of the older mechanism tests).
"""
import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .mock_world import SIM_SAG_GAIN
from .perception import Detection
from .planner import Planner, correct_ticks, held_masks
from .policy import ClassicalPolicy
from .test_classical import run_mock

SC = Scene.load()
CZ, H = SC.counter_z, SC.object_height()
SIM = load_config(profile="sim")
# the correction's own mechanism: with the sag feed-forward (on in sim, test_sag.py) there would be ~nothing to correct
NO_FF = {"sag.enabled": False}
OFF = {"grasp.correct.enabled": False, **NO_FF}


def det(color, x, y, shape="cylinder"):
    d = Detection(color, np.array([x, y, CZ + H]), 900, 900, "depth")
    d.shape = shape
    return d


def sag(kin, q, h):
    """The mock's static tracking error (SIM_SAG_GAIN * gravity torque) at arm q, hands h (14,)."""
    x = kin.q0.copy()
    x[kin.arm_iq], x[kin.hand_iq] = q, h
    return np.tile(SIM_SAG_GAIN, 2) * kin.pin.computeGeneralizedGravity(kin.model, kin.data, x)[kin.arm_iv]


def rot_deg(kin, side, a, b):
    kin._fk(a)
    Ra = kin.data.oMf[kin.palm[side]].rotation.copy()
    kin._fk(b)
    return float(np.degrees(np.linalg.norm(kin.pin.log3(Ra.T @ kin.data.oMf[kin.palm[side]].rotation))))


class _Fixture(unittest.TestCase):
    """A left-arm pick (profile sim, correction on) and the queue as the policy sees it at the measurement tick."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = SIM
        cls.post, cls.home = hand_postures(SIM), home_arm(SIM)
        cls.pl = Planner(SIM, SC, cls.post, cls.home)
        cls.act = Planner(SIM, SC, cls.post, cls.home, ArmKinematics(SIM))      # the act()-thread instances
        cls.kin = cls.act.kin
        cls.p = cls.pl.plan(det("red", 0.29, 0.14), cls.home, cls.post["open"], relocate=False)
        assert cls.p.ok, cls.p.reason
        cls.side, cls.sl = cls.p.side, ArmKinematics.sl(cls.p.side)
        ph = {n: (s, e) for n, s, e in cls.p.phases}
        cls.ph, cls.ramp = ph, correct_ticks(SIM["grasp"]["correct"])[1]
        cls.k = ph["close"][0] - cls.ramp                                       # measurement tick
        cls.arm_q, cls.hand_q = cls.p.arm[cls.k - 1], cls.p.hand[cls.k - 1]
        cls.rest_a, cls.rest_h = cls.p.arm[cls.k:], cls.p.hand[cls.k:]
        cls.phases = [(n, s - cls.k, e - cls.k) for n, s, e in cls.p.phases if e > cls.k and s >= ph["settle"][0]]
        cls.meas = cls.arm_q + sag(cls.kin, cls.arm_q, cls.hand_q)

    def correct(self, dq, planner=None, **kw):
        return (planner or self.act).correct(self.side, self.arm_q, self.hand_q, self.rest_a, self.rest_h, dq,
                                             self.phases, **kw)

    def dq_for(self, dp):
        """Joint shift moving the commanded grasp point by dp (palm orientation kept)."""
        _, Jp, Jw, _ = self.kin.grasp_jacobian(self.side, self.arm_q)
        return np.linalg.lstsq(np.vstack([Jp, Jw]), np.r_[dp, np.zeros(3)], rcond=None)[0]


class TestSettlePlan(_Fixture):
    def test_settle_phase_and_off_is_unchanged(self):
        """On: `settle` between descend and close, settle_s + obs_lag + ramp_s ticks (sim 6 + 2 + 4), arm at the
        grasp pose, hand preshaped. Off: the same plan without those ticks (every other tick identical)."""
        p = self.p
        names = [n for n, _, _ in p.phases]
        self.assertEqual(names[names.index("descend"):names.index("close") + 1], ["descend", "settle", "close"])
        s, e = self.ph["settle"]
        self.assertEqual(e - s, sum(correct_ticks(SIM["grasp"]["correct"])))
        self.assertEqual(e - s, 12)
        np.testing.assert_allclose(p.arm[s:e], np.repeat(p.arm[s - 1][None], e - s, 0))
        np.testing.assert_allclose(p.hand[s:e, self.sl], np.repeat(self.post["preshape"][self.sl][None], e - s, 0))
        cfg = load_config(profile="sim", overrides=OFF)
        q = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg)).plan(det("red", 0.29, 0.14), self.home,
                                                                       self.post["open"], relocate=False)
        self.assertTrue(q.ok)
        self.assertNotIn("settle", [n for n, _, _ in q.phases])
        keep = np.r_[np.arange(s), np.arange(e, len(p.arm))]
        np.testing.assert_allclose(q.arm, p.arm[keep])
        np.testing.assert_allclose(q.hand, p.hand[keep])
        self.assertEqual(q.phases, [(n, a - (e - s) * (a >= e), b - (e - s) * (b >= e)) for n, a, b in p.phases
                                    if n != "settle"])

    def test_profiles(self):
        """Base: on with the conservative robot values (dwell 12 + 3 + 6 ticks = 0.7 s; the dwell was 9 ticks until
        the slower robot motion of 2026-11-14: grasp.correct.settle_s 0.4); sim: 6 + 2 + 4 = 0.4 s;
        sim_legacy (older mechanism tests) off; the oracle pins sim's."""
        from aaltonomous.sim.oracle import ORACLE_OVERRIDES
        base, legacy = load_config(profile="none"), load_config(profile="sim_legacy")
        self.assertTrue(base["grasp"]["correct"]["enabled"] and SIM["grasp"]["correct"]["enabled"])
        self.assertFalse(legacy["grasp"]["correct"]["enabled"])
        self.assertEqual(correct_ticks(base["grasp"]["correct"]), (15, 6))
        self.assertEqual(correct_ticks(SIM["grasp"]["correct"]), (8, 4))
        orc = load_config(profile="none", overrides=ORACLE_OVERRIDES)["grasp"]["correct"]
        self.assertEqual(orc, SIM["grasp"]["correct"])


class TestCorrectionMath(_Fixture):
    def test_known_joint_offset(self):
        """measured = commanded + a fixed joint offset (+ the sag): e = FK(cmd) - FK(meas); measured + dq has the
        commanded grasp point and palm orientation; with the offset unchanged, commanded + dq puts the hand there
        (sag re-evaluated at the shifted pose: still < 0.3 mm)."""
        delta = np.zeros(14)
        delta[self.sl] = [0.012, -0.008, 0.006, 0.010, 0.0, 0.004, -0.003]
        meas = self.meas + delta
        e, dq, f, res = self.act.correction_dq(self.side, self.arm_q, meas)
        p_cmd, p_meas = self.kin.grasp_pose(self.side, self.arm_q)[0], self.kin.grasp_pose(self.side, meas)[0]
        np.testing.assert_allclose(e, p_cmd - p_meas, atol=1e-9)
        self.assertGreater(np.linalg.norm(e), 0.008)
        self.assertLess(res, 1e-4)
        self.assertAlmostEqual(np.linalg.norm(f), 1.0)
        m2 = meas.copy()
        m2[self.sl] += dq
        self.assertLess(np.linalg.norm(self.kin.grasp_pose(self.side, m2)[0] - p_cmd), 1e-4)
        self.assertLess(rot_deg(self.kin, self.side, m2, self.arm_q), 0.05)
        c2 = self.arm_q.copy()
        c2[self.sl] += dq
        m3 = c2 + delta + sag(self.kin, c2, self.hand_q)
        self.assertLess(np.linalg.norm(self.kin.grasp_pose(self.side, m3)[0] - p_cmd), 3e-4)

    def test_joint_at_its_bound_is_pinned(self):
        """A joint whose shift would leave the IK bounds stays on its bound; the other six still remove the
        position error (residual < 0.3 of it)."""
        _, dq0, _, _ = self.act.correction_dq(self.side, self.arm_q, self.meas)
        j = 3 + int(np.argmax(np.abs(dq0[3:])))      # a wrist / elbow joint that moves
        q = self.arm_q.copy()
        i = self.sl.start + j
        q[i] = self.kin.hi[i] if dq0[j] > 0 else self.kin.lo[i]
        meas = q + (self.meas - self.arm_q)
        e, dq, _, res = self.act.correction_dq(self.side, q, meas)
        self.assertTrue(np.all(q[self.sl] + dq <= self.kin.hi[self.sl] + 1e-12)
                        and np.all(q[self.sl] + dq >= self.kin.lo[self.sl] - 1e-12))
        self.assertLessEqual(dq[j] * np.sign(dq0[j]), 1e-12)
        self.assertLess(res, 0.3 * np.linalg.norm(e))

    def test_vectorised_clearances(self):
        """Planner._clear_many == _counter_clear / _bin_clear per tick."""
        P = np.array([self.kin.hand_points(self.side, a, h) for a, h in zip(self.p.arm[::4], self.p.hand[::4])])
        c, b = self.act._clear_many(P)
        np.testing.assert_allclose(c, [self.act._counter_clear(x) for x in P])
        np.testing.assert_allclose(b, [self.act._bin_clear(x) for x in P])


class TestCorrectGuards(_Fixture):
    def test_applied_weights_speed_and_columns(self):
        """Blend-in ends at the close (w = 1 there), held through the close and hold_lift_s into the lift, blended
        out after; ticks after that and the other arm's columns untouched; no joint faster than motion.arm_max_vel;
        0 clips over the whole new queue."""
        _, dq, _, _ = self.act.correction_dq(self.side, self.arm_q, self.meas)
        new, info = self.correct(dq)
        self.assertIsNotNone(new, info)
        D = new - self.rest_a
        osl = ArmKinematics.sl("right" if self.side == "left" else "left")
        self.assertEqual(np.abs(D[:, osl]).max(), 0.0)
        w = D[:, self.sl] @ dq / (dq @ dq)
        np.testing.assert_allclose(D[:, self.sl], w[:, None] * dq, atol=1e-12)
        c0, l0 = self.ramp, self.ph["lift"][0] - self.k
        hold = int(round(SIM["grasp"]["correct"]["hold_lift_s"] * 30))
        self.assertEqual(info["ramp_ticks"], c0)
        self.assertTrue(np.all(np.diff(w[:c0]) > 0) and w[0] > 0)
        np.testing.assert_allclose(w[c0 - 1:l0 + hold], 1.0)
        self.assertTrue(np.all(np.diff(w[l0 + hold:info["ticks"]]) < 0))
        self.assertGreaterEqual(info["blend_out_ticks"], int(round(SIM["grasp"]["correct"]["blend_out_s"] * 30)))
        np.testing.assert_allclose(w[info["ticks"]:], 0.0)
        step = np.abs(np.diff(np.vstack([self.arm_q[None], new]), axis=0)).max()
        self.assertLessEqual(step, SIM["motion"]["arm_max_vel"] / 30 + 1e-9)
        self.assertEqual(self.act.dry_run_safety(self.arm_q, self.hand_q, new, self.rest_h), 0)

    def test_each_guard_refuses(self):
        _, dq, _, _ = self.act.correction_dq(self.side, self.arm_q, self.meas)
        self.assertIsNotNone(self.correct(dq)[0])
        cases = {"max_dq": self.correct(dq * 0.06 / np.abs(dq).max())}
        i = self.sl.start + 4                       # wrist roll 2 mrad inside its bound from the settle on
        a, r = self.arm_q.copy(), self.rest_a.copy()
        a[i] = r[:, i] = self.kin.hi[i] - 0.002
        out = np.zeros(7)
        out[4] = 0.01
        cases["IK joint bounds"] = self.act.correct(self.side, a, self.hand_q, r, self.rest_h, out, self.phases)
        cases["above the counter"] = self.correct(self.dq_for([0.0, 0.0, -0.006]))
        with mock.patch.object(self.act, "_clear_many", lambda P, rim=None: (np.full(len(P), np.inf),
                                                                             np.full(len(P), 0.001))):
            cases["bin rim"] = self.correct(dq)
        cases["arm-arm clearance"] = self.correct(dq, held=held_masks(self.p), pair_need=1.0)
        with mock.patch.object(self.act, "dry_run_safety", lambda *a: 2):
            cases["safety clips"] = self.correct(dq)
        started = [(n, s - self.ramp, e - self.ramp) for n, s, e in self.phases]
        cases["close has started"] = self.act.correct(self.side, self.arm_q, self.hand_q, self.rest_a[self.ramp:],
                                                      self.rest_h[self.ramp:], dq, started)
        for what, (new, why) in cases.items():
            with self.subTest(what):
                self.assertIsNone(new)
                self.assertIn(what, why)
        self.assertIsNotNone(self.correct(dq, held=held_masks(self.p), pair_need=0.02)[0])   # other arm at home

    def test_neighbour_margin(self):
        """A neighbour 1 mm outside neighbours.margin of the planned hand: a correction towards it is refused, one
        away from it applied."""
        g = self.kin.grasp_pose(self.side, self.arm_q)[0]
        cyl = np.array([[g[0] + 0.09, g[1], 0.0, CZ + H]])
        n = len(self.rest_a)
        D0 = self.act.neighbour_clearance(self.side, self.arm_q, self.hand_q, self.rest_a, self.rest_h, cyl)[1][:40]
        m = float(SIM["neighbours"]["margin"])
        cyl[0, 2] = D0.min() - m - 0.001                          # radius: planned min distance = margin + 1 mm
        self.assertGreater(cyl[0, 2], 0.01, n)
        new, why = self.correct(self.dq_for([0.006, 0.0, 0.0]), nb=cyl)
        self.assertIsNone(new)
        self.assertIn("from a neighbour", why)
        new, why = self.correct(self.dq_for([-0.006, 0.0, 0.0]), nb=cyl)
        self.assertIsNotNone(new, why)

    def test_measurement_guards(self):
        cc, g = SIM["grasp"]["correct"], ClassicalPolicy._correct_guard
        self.assertIsNone(g(cc, 0.008, 0.0, 0.01))
        self.assertIn("max_err", g(cc, 0.025, 0.0, 0.0))
        self.assertIn("min_err", g(cc, 0.0005, 0.0, 0.0))
        self.assertIn("moving", g(cc, 0.008, 0.0, 0.5))
        self.assertIn("would remain", g(cc, 0.008, 0.005, 0.0))


def _plans(store):
    def setup(pol):
        orig = pol._start
        pol._start = lambda plan, t: (store.append(plan), orig(plan, t))[1]
    return setup


def close_errors(env, plans):
    """Per hand close on an object: |measured grasp point - planned one| (m; planned = FK of the plan's tick at the
    close, not the waypoint: the IK may be a few mm off it), nearest planned of that arm."""
    kin = env.backend.kin
    gps = [(p.side, kin.grasp_pose(p.side, p.arm[next(s for n, s, _ in p.phases if n == "close")])[0])
           for m in plans for p in (m.parts or [m]) if "grasp" in p.waypoints]
    out = []
    for c in env.backend.close_log:
        if c["dist"] < 0.03:
            out.append(min(np.linalg.norm(g - c["g"]) for s, g in gps if s == c["side"]))
    return np.array(out)


class TestCorrectEpisodes(unittest.TestCase):
    """Mock world with the sim's arm sag (sag_gain): profile sim with and without the correction (sag feed-forward
    off: the correction alone; with it, test_sag.py)."""

    def run_(self, seed, on, env_kw=None, ticks=None):
        cfg = load_config(profile="sim", overrides=NO_FF if on else OFF)
        plans = []
        pol, env, info, _ = run_mock(seed, {"sag_gain": True, **(env_kw or {})}, config=cfg, setup=_plans(plans),
                                     ticks=ticks)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        return pol, env, info, plans

    def test_sag_removed_at_the_close(self):
        """Seeds 0 and 2: without the correction the grasp point is 5-12 mm from the planned one at the close; with
        it every applied correction leaves < 0.5 mm; every object delivered either way (attach radius 4 cm);
        measured exactly at the measurement tick, a few ms each."""
        for seed in (0, 2):
            with self.subTest(seed=seed):
                pol0, env0, info0, plans0 = self.run_(seed, False)
                pol1, env1, info1, plans1 = self.run_(seed, True)
                e0, e1 = close_errors(env0, plans0), close_errors(env1, plans1)
                self.assertGreater(e0.min(), 0.004)
                self.assertGreater(np.median(e0), 0.006)
                self.assertEqual(pol0.corrections, [])
                c = pol1.corrections
                self.assertEqual(len(c), len(e1))
                self.assertTrue(all(x["tick"] == x["k_meas"] for x in c))
                self.assertGreaterEqual(sum(x["applied"] for x in c), len(c) - 1, [x["why"] for x in c])
                self.assertLess(np.median(e1), 0.0005)
                self.assertLessEqual(np.sum(e1 > 0.001), len(c) - sum(x["applied"] for x in c))
                self.assertLess(max(x["ms"] for x in c), 50.0)
                for env, info, pol in ((env0, info0, pol0), (env1, info1, pol1)):
                    out = env.backend.outcomes(pol.mapping)
                    self.assertEqual((info["end_reason"], out.count("correct")), ("done", len(out)))
                notes = [m for _, m in pol1.events if m.startswith("correct ")]
                self.assertEqual(len(notes), len(c))
                self.assertTrue(all("mm off (cmd - meas: pelvis x/y/z" in m and "along fingers" in m and "mrad" in m
                                    for m in notes), notes)

    def test_offset_misses_hold_with_the_correction(self):
        """A grasp model as sensitive as the pinch (attach within 4.5 mm of the object centre, decided when the
        fingers close): with the sag the hand misses; with the correction it holds."""
        kw = {"attach_radius": 0.0045, "attach_once": True}
        n0 = n1 = n = 0
        for seed in (0, 2):
            _, env0, _, _ = self.run_(seed, False, kw)
            pol1, env1, _, _ = self.run_(seed, True, kw)
            n0 += env0.backend.outcomes(pol1.mapping).count("correct")
            n1 += env1.backend.outcomes(pol1.mapping).count("correct")
            n += len(env1.backend.objects)
        self.assertLessEqual(n0, 0.3 * n, (n0, n1, n))
        self.assertGreaterEqual(n1, 0.8 * n, (n0, n1, n))

    def test_pair_corrects_one_arm_at_a_time(self):
        """Seed 0 starts with a pair: each arm corrected at its own settle; the other arm's columns and every hand
        column untouched by it; the executed ticks keep two_arm.min_clearance in the sphere model."""
        snaps = []

        def setup(pol):
            _plans([])(pol)
            orig = pol._correct_arm

            def wrapped(side, x, obs, t):
                a, h = pol.queue_a.copy(), pol.queue_h.copy()
                orig(side, x, obs, t)
                snaps.append((side, a, h, pol.queue_a.copy(), pol.queue_h.copy(), pol.corrections[-1], pol.exec_held))
            pol._correct_arm = wrapped

        ticks, cfg = [], load_config(profile="sim", overrides=NO_FF)   # (ticks = what was sent: intended here)
        pol, env, info, _ = run_mock(0, {"sag_gain": True}, config=cfg, setup=setup, ticks=ticks)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        paired = [s for s in snaps if s[6] is not None]
        self.assertGreaterEqual(len({s[0] for s in paired}), 2)      # both arms of a pair corrected
        for side, a0, h0, a1, h1, c, _ in snaps:
            osl = ArmKinematics.sl("right" if side == "left" else "left")
            np.testing.assert_array_equal(a1[:, osl], a0[:, osl])
            np.testing.assert_array_equal(h1, h0)
            if c["applied"]:
                changed = np.flatnonzero(np.abs(a1 - a0).max(1) > 0)
                self.assertEqual((changed.min(), changed.max() < c["ticks"]), (0, True))
        kin = ArmKinematics(cfg)
        d = min(kin.arm_arm_distance(a, h, w) for a, h, w in ticks)
        self.assertGreaterEqual(d, float(cfg["two_arm"]["min_clearance"]))

    def test_verification_and_abort_after_a_correction(self):
        """A scripted miss / slip on a corrected grasp: verified as before, the abort decelerates along the corrected
        queue and goes home, the retry delivers; 0 clips."""
        for kw, word in (({"fail_grasps": (0,)}, "(miss)"), ({"slip_grasps": (0,)}, "(slip)")):
            with self.subTest(kw=kw):
                pol, env, info, _ = self.run_(2, True, kw)
                self.assertTrue(pol.corrections[0]["applied"])
                ab = [m for _, m in pol.events if m.startswith("abort")]
                self.assertTrue(ab and word in ab[0], pol.events[-20:])
                out = env.backend.outcomes(pol.mapping)
                self.assertEqual(out.count("correct"), len(out))


if __name__ == "__main__":
    unittest.main()
