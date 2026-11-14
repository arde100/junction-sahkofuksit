"""Offline tests of the two-arm clearance model (sphere model of forearms + Dex3 hands) and of the paired cycle's
planning (merge_plans, Planner.pair / plan_pair). Run:
    ~/miniforge3/envs/aalto/bin/python -m unittest aaltonomous/baselines/classical/test_two_arm.py -v
"""
import copy
import os
import unittest

import numpy as np

# mechanism tests on profile sim_legacy (see test_classical.py): sim's speeds, the grasp these layouts were written for
os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config, mirror_left_to_right
from .kinematics import ArmKinematics
from .perception import Detection
from .planner import PAIR_CHECK_STEP, Planner, checked_masks, held_masks, merge_plans, phase_at
from .verify import grasp_window

CFG = load_config()
SC = Scene.load()
CZ = SC.counter_z
# Every tick of an accepted pair. The scheduler checks every PAIR_CHECK_STEP-th tick (1 since 2026-11-14: with 2 the
# worst unchecked tick was only 0.06 mm below min_clearance on 200 sampled layouts, but away from its minimum the
# distance changes by up to 14.5 mm per tick, each arm's spheres moving up to 20 mm per tick at 0.97 rad/s). The slack
# only matters for a step > 1; the hard bound for unchecked ticks is the per-tick check below.
BRUTE_SLACK = 0.005 if PAIR_CHECK_STEP > 1 else 0.0
ARM_MIRROR = np.array([1, -1, -1, 1, -1, 1, -1], float)  # roll/yaw joints flip sign between the arms


def mirror_arm(q):
    """Arm 14-vector of the mirror-image pose (left <-> right)."""
    return np.r_[q[7:] * ARM_MIRROR, q[:7] * ARM_MIRROR]


def mirror_hand(h):
    """Hand 14-vector of the mirror-image posture (left DDS order <-> right DDS order)."""
    m = mirror_left_to_right  # its own inverse: also maps right order -> left order
    return np.r_[m(h[7:]), m(h[:7])]


class TestTwoArmClearance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = ArmKinematics(CFG)
        cls.post = hand_postures(CFG)
        cls.home = home_arm(CFG)

    def pose(self, yl, yr, z=0.05, x=0.35):
        """Both grasp points at (x, y, CZ + z), top-down IK from home."""
        q = self.home.copy()
        for side, y in (("left", yl), ("right", yr)):
            q, ok, _, _ = self.kin.solve(side, [x, y, CZ + z], q)
            self.assertTrue(ok, f"IK failed for {side} at y={y}")
        return q

    def test_mirror_symmetry(self):
        q = self.pose(0.09, -0.05, z=0.08)
        h = self.post["closed"].copy()
        h[7:] = self.post["open"][7:]
        qm, hm = mirror_arm(q), mirror_hand(h)
        for side, other in (("left", "right"), ("right", "left")):
            c, r = self.kin.arm_spheres(side, q, h, with_object=True)
            cm, rm = self.kin.arm_spheres(other, qm, hm, with_object=True)
            np.testing.assert_allclose(c * [1, -1, 1], cm, atol=1e-4)  # the URDF itself is mirrored to ~10 um
            np.testing.assert_allclose(r, rm)
        self.assertAlmostEqual(self.kin.arm_arm_distance(q, h, (True, False)),
                               self.kin.arm_arm_distance(qm, hm, (False, True)), places=4)

    def test_home_has_large_clearance(self):
        for name in ("open", "closed"):
            self.assertGreater(self.kin.arm_arm_distance(self.home, self.post[name]), 0.10)
            self.assertGreater(self.kin.arm_arm_distance(self.home, self.post[name], (True, True)), 0.05)

    def test_hands_side_by_side_12cm_are_clear(self):
        """Two-arm operation must be possible with the grasp points ~12 cm apart over the object band (fingers
        forward: mesh distance 62 mm, model 42 mm on 2026-11-14)."""
        q = self.pose(0.06, -0.06, z=0.0225)
        for name in ("open", "closed"):
            self.assertGreater(self.kin.arm_arm_distance(q, self.post[name]), 0.03, name)

    def test_same_point_overlaps(self):
        q = self.pose(0.0, 0.0, z=0.10)
        self.assertLess(self.kin.arm_arm_distance(q, self.post["closed"]), 0.0)
        q = self.pose(0.02, -0.02, z=0.10)
        self.assertLess(self.kin.arm_arm_distance(q, self.post["closed"]), 0.01)

    def test_with_object_reduces_distance(self):
        q, h = self.pose(0.05, -0.05, z=0.20, x=0.30), self.post["closed"]
        d0 = self.kin.arm_arm_distance(q, h)
        d1 = self.kin.arm_arm_distance(q, h, (True, False))
        d2 = self.kin.arm_arm_distance(q, h, (True, True))
        self.assertLessEqual(d1, d0)
        self.assertLessEqual(d2, d1)
        self.assertLess(d2, d0 - 0.005)  # here the two objects are the closest pair
        # the object sphere sits at the grasp point
        c, r = self.kin.arm_spheres("left", q, h, with_object=True)
        np.testing.assert_allclose(c[-1], self.kin.grasp_pose("left", q)[0], atol=1e-9)
        self.assertEqual(len(r), len(self.kin.arm_spheres("left", q, h)[1]) + 1)

    def test_trajectory_min_and_tick(self):
        near = self.pose(0.04, -0.04, z=0.10)
        T, k0 = 61, 37
        s = np.clip(1 - np.abs(np.arange(T) - k0) / 30, 0, 1)  # home -> near (at tick 37) -> back
        arm = self.home + s[:, None] * (near - self.home)
        hand = np.tile(self.post["closed"], (T, 1))
        d1, t1 = self.kin.min_arm_arm_distance(arm, hand, step=1)
        self.assertEqual(t1, k0)
        self.assertAlmostEqual(d1, self.kin.arm_arm_distance(near, self.post["closed"]), places=9)
        d2, t2 = self.kin.min_arm_arm_distance(arm, hand, step=2)
        self.assertIn(t2, (k0 - 1, k0 + 1))
        self.assertGreaterEqual(d2, d1)
        # the last tick is always checked
        d3, t3 = self.kin.min_arm_arm_distance(arm[:k0 + 1], hand[:k0 + 1], step=5)
        self.assertEqual(t3, k0)
        # early exit at the first checked tick below the threshold
        thr = 0.5 * (d1 + self.kin.arm_arm_distance(self.home, self.post["closed"]))
        d4, t4 = self.kin.min_arm_arm_distance(arm, hand, step=1, stop_below=thr)
        self.assertLess(d4, thr)
        self.assertLess(t4, k0)
        # object carried by the left hand from tick 30 on: per-tick flags are honoured (= brute force)
        held = np.arange(T) >= 30
        d5, t5 = self.kin.min_arm_arm_distance(arm, hand, step=1, with_object=(held, False))
        brute = [self.kin.arm_arm_distance(arm[k], hand[k], (held[k], False)) for k in range(T)]
        self.assertAlmostEqual(d5, min(brute), places=12)
        self.assertEqual(t5, int(np.argmin(brute)))
        self.assertLessEqual(d5, d1)


class TestPairPlanning(unittest.TestCase):
    """merge_plans / Planner.pair / plan_pair on real plans (sim_legacy: the sim speeds two_arm is tuned for)."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(overrides={"grasp.slots.enabled": False})                  # drop_offsets (slots:
        cls.post, cls.home = hand_postures(cls.cfg), home_arm(cls.cfg)                    # test_grasp_plan.py)
        cls.pl = Planner(cls.cfg, SC, cls.post, cls.home)
        cls.need = float(cls.cfg["two_arm"]["min_clearance"])
        cls.sep = cls.plan("red", 0.30, 0.15), cls.plan("blue", 0.30, -0.15)   # far left / far right

    @classmethod
    def det(cls, color, x, y):
        return Detection(color, np.array([x, y, CZ + SC.object_height()]), 0, 0, "depth")

    @classmethod
    def plan(cls, color, x, y, side=None):
        if side is None:
            p = cls.pl.plan(cls.det(color, x, y), cls.home, cls.post["open"])
        else:
            p = cls.pl.plan_side(cls.det(color, x, y), side, cls.pl.mapping[color], cls.home, cls.post["open"])
        assert p.ok, p.reason
        return p

    def check_every_tick(self, m):
        """Accepted pair: >= min_clearance at the checked ticks, >= min_clearance - BRUTE_SLACK at every tick, and at
        each unchecked tick no lower than the checked tick before it minus how far each arm's spheres moved since
        (the geometric bound that justifies checking every 2nd tick)."""
        kin = self.pl.kin
        hl, hr = held_masks(m)
        cl, cr = checked_masks((hl, hr))
        T, st = len(m.arm), PAIR_CHECK_STEP
        d = np.array([kin.arm_arm_distance(m.arm[k], m.hand[k], (hl[k], hr[k])) for k in range(T)])
        dc = np.array([kin.arm_arm_distance(m.arm[k], m.hand[k], (cl[k], cr[k])) for k in range(0, T, st)])
        self.assertGreaterEqual(dc.min(), self.need)
        self.assertGreaterEqual(d.min(), self.need - BRUTE_SLACK - 1e-9)
        self.assertAlmostEqual(m.clearance[0], min(dc.min(), d[-1]), places=9)
        for k in range(T):
            if k % st == 0:
                continue
            k0 = k - k % st                                   # the checked tick before it
            move = sum(np.linalg.norm(kin.arm_spheres(s, m.arm[k], m.hand[k], True)[0]
                                      - kin.arm_spheres(s, m.arm[k0], m.hand[k0], True)[0], axis=1).max()
                       for s in ("left", "right"))
            self.assertGreaterEqual(d[k], dc[k0 // st] - move - 1e-9, k)
        return d

    def test_merge_columns_lag_phases(self):
        p1, p2 = self.sep
        self.assertEqual((p1.side, p2.side), ("left", "right"))
        lag, L, R = 20, slice(0, 7), slice(7, 14)
        m = merge_plans(p1, p2, lag)
        T = max(len(p1.arm), len(p2.arm) + lag)
        self.assertEqual((m.arm.shape, m.hand.shape, m.side), ((T, 14), (T, 14), "both"))
        self.assertEqual(m.lags, {"left": 0, "right": lag})
        self.assertEqual([q.color for q in m.parts], ["red", "blue"])
        n1, n2 = len(p1.arm), len(p2.arm)
        np.testing.assert_allclose(m.arm[:n1, L], p1.arm[:, L])
        np.testing.assert_allclose(m.hand[:n1, L], p1.hand[:, L])
        np.testing.assert_allclose(m.arm[n1:, L], np.repeat(p1.arm[-1:, L], T - n1, 0))   # padded with its last tick
        np.testing.assert_allclose(m.arm[:lag, R], np.repeat(self.home[None, R], lag, 0))  # delayed: holds the start
        np.testing.assert_allclose(m.hand[:lag, R], np.repeat(self.post["open"][None, R], lag, 0))
        np.testing.assert_allclose(m.arm[lag:lag + n2, R], p2.arm[:, R])
        np.testing.assert_allclose(m.hand[lag:lag + n2, R], p2.hand[:, R])
        np.testing.assert_allclose(m.arm[lag + n2:, R], np.repeat(p2.arm[-1:, R], T - lag - n2, 0))
        np.testing.assert_allclose(m.arm[-1], self.home)            # both arms end at home, hands open
        np.testing.assert_allclose(m.hand[-1], self.post["open"])
        np.testing.assert_allclose(merge_plans(p1, p2, lag, self.home, self.post["open"]).arm, m.arm)  # explicit start
        for side, p, k in (("left", p1, 0), ("right", p2, lag)):
            for n, s, e in p.phases:
                self.assertIn((f"{side}:{n}", s + k, e + k), m.phases)
            # the policy's verification window of this arm = the part's window shifted by its lag
            ph = [(n.split(":")[1], s, e) for n, s, e in m.phases if n.startswith(side + ":")]
            sl = ArmKinematics.sl(side)
            w, w0 = grasp_window(ph, m.hand, sl), grasp_window(p.phases, p.hand, sl)
            self.assertEqual((w[0], w[1]), (w0[0] + k, w0[1] + k))
            ph0 = {n: (s, e) for n, s, e in p.phases}
            h = held_masks(m)[0 if side == "left" else 1]
            self.assertEqual((int(np.argmax(h)), int(np.flatnonzero(h)[-1])),
                             (ph0["close"][0] + k, ph0["open"][1] + k - 1))
        self.assertEqual(phase_at(m, "right", lag - 1), "wait")
        self.assertEqual(phase_at(m, "right", lag), p2.phases[0][0])
        self.assertEqual(phase_at(m, "left", T - 1), "done")
        with self.assertRaises(ValueError):
            merge_plans(p1, p1)

    def test_checked_masks_cover_the_next_tick(self):
        h = np.array([0, 0, 0, 1, 1, 0, 0], bool)
        np.testing.assert_array_equal(checked_masks([h], step=2)[0], [0, 0, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(checked_masks([h], step=1)[0], h)

    def test_separated_pair_at_lag_0(self):
        m, why = self.pl.pair(*self.sep, self.home, self.post["open"])
        self.assertIsNotNone(m, why)
        self.assertEqual((m.lags["right"], m.safety_clips), (0, 0))
        self.assertGreaterEqual(m.clearance[0], self.need)
        self.assertEqual(len(m.arm), max(len(p.arm) for p in self.sep))   # no lag: as long as the longer single plan
        self.check_every_tick(m)

    def test_centre_line_pair_needs_a_lag(self):
        p1, p2 = self.plan("red", 0.30, 0.05), self.plan("blue", 0.30, -0.04)   # 9 cm apart near the centre line
        self.assertEqual((p1.side, p2.side), ("left", "right"))
        m0 = merge_plans(p1, p2, 0)
        d0, k0 = self.pl.kin.min_arm_arm_distance(m0.arm, m0.hand, step=1, with_object=held_masks(m0))
        self.assertLess(d0, self.need)                                  # side by side at lag 0: too close
        m, why = self.pl.pair(p1, p2, self.home, self.post["open"])
        if m is None:
            self.assertIn("no lag", why)
        else:
            self.assertGreater(m.lags["right"], 0)
            self.assertLess(len(m.arm), len(p1.arm) + len(p2.arm))
            self.check_every_tick(m)

    def test_same_bin_pair_uses_the_next_drop_point(self):
        offs = self.cfg["grasp"]["drop_offsets"]
        first = self.plan("yellow", SAME_BIN[0][0], SAME_BIN[0][1])
        m, p2, why = self.pl.plan_pair(first, self.det("yellow", *SAME_BIN[1]), "center", self.home, self.post["open"],
                                       drop_offset=offs[1])
        self.assertTrue(p2.ok, p2.reason)
        self.assertEqual({first.side, p2.side}, {"left", "right"})
        c = self.pl.drop_xy("center", offset=False)
        np.testing.assert_allclose(first.waypoints["drop"][:2], c + offs[0])
        np.testing.assert_allclose(p2.waypoints["drop"][:2], c + offs[1])
        self.assertEqual(self.pl.drop_offsets, {})                      # restored after planning the second arm
        self.assertIsNotNone(m, why)
        d = self.check_every_tick(m)
        # both hands at the centre bin (carry / release / retreat) at the same time: never closer than allowed
        at_bin = [k for k in range(len(m.arm))
                  if all(phase_at(m, s, k) in ("carry", "open", "retreat") for s in ("left", "right"))]
        if at_bin:
            self.assertGreaterEqual(d[at_bin].min(), self.need - BRUTE_SLACK)

    def test_refused_with_reason(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["two_arm"]["min_clearance"] = 0.5                           # no two-arm pose is that far apart
        pl = Planner(cfg, SC, self.post, self.home, kin=self.pl.kin)
        m, why = pl.pair(*self.sep, self.home, self.post["open"])
        self.assertIsNone(m)
        self.assertIn("no lag <= ", why)
        self.assertIn("lag 0.0 s: ", why)                               # first and last lag tried, with what is closest
        self.assertRegex(why, r"tick \d+ \(left \w+, right \w+; \w+ / \w+\)")

    def test_plan_side_fixed_arm(self):
        det = self.det("yellow", 0.30, 0.10)                            # left of the centre line, right arm
        p = self.pl.plan_side(det, "right", "center", self.home, self.post["open"], variant=2,
                              drop_offset=(0.03, -0.03))
        self.assertTrue(p.ok, p.reason)
        self.assertEqual((p.side, p.target_bin, p.variant), ("right", "center", 2))
        np.testing.assert_allclose(p.waypoints["drop"][:2], self.pl.drop_xy("center", offset=False) + [0.03, -0.03])
        np.testing.assert_allclose(p.arm[:, :7], np.repeat(self.home[None, :7], len(p.arm), 0))  # other arm untouched
        q = self.pl.plan_side(self.det("red", 0.30, 0.10), "right", "left", self.home, self.post["open"])
        self.assertFalse(q.ok)                                          # cross-body: the reach test, no relocation
        self.assertIn("cannot reach", q.reason)


SAME_BIN = ((0.30, 0.12), (0.30, -0.12))   # two yellow objects (centre bin), one per arm


def evade_cfg(**ov):
    """Test profile (sim_legacy) + ov with two_arm.avoid_parked on (set on the dict: independent of $AALTO_SET)."""
    cfg = load_config(overrides=ov)
    cfg["two_arm"]["avoid_parked"] = True
    return cfg


class TestParkedArm(unittest.TestCase):
    """Planner.keep_clear / evade: a single-arm plan vs the other arm parked at the (sim) home pose."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = evade_cfg()
        cls.post, cls.home = hand_postures(cls.cfg), home_arm(cls.cfg)
        cls.pl = Planner(cls.cfg, SC, cls.post, cls.home)
        cls.need = float(cls.cfg["two_arm"]["min_clearance"])

    def plan(self, color, x, y, side=None, bin_name=None, variant=0):
        det = Detection(color, np.array([x, y, CZ + SC.object_height()]), 0, 0, "depth")
        if side is None:
            p = self.pl.plan(det, self.home, self.post["open"], variant=variant)
        else:
            p = self.pl.plan_side(det, side, bin_name or self.pl.mapping[color], self.home, self.post["open"],
                                  variant=variant)
        self.assertTrue(p.ok, p.reason)
        return p

    def parked(self, p):
        """Brute force: every tick of the single plan vs the parked arm."""
        kin, (hl, hr) = self.pl.kin, held_masks(p)
        return min(kin.arm_arm_distance(p.arm[k], p.hand[k], (hl[k], hr[k])) for k in range(len(p.arm)))

    def check_evade(self, m, p):
        """Merged evade queue of single plan p: >= min_clearance at EVERY tick (brute force), 0 clips, speed limits,
        both arms home at the end, the idle arm's hand unchanged and its moves above counter / bin rims, the plan's
        own columns = p delayed by its lag (verification window shifted with it), the return not startable earlier."""
        kin, pl, mo = self.pl.kin, self.pl, self.cfg["motion"]
        e, idle = m.evade, m.parts[1]
        self.assertIs(m.parts[0], p)
        self.assertEqual((idle.side, e["side"]), (("right" if p.side == "left" else "left"),) * 2)
        sl, isl, lag, R = ArmKinematics.sl(p.side), ArmKinematics.sl(e["side"]), e["lag"], e["back"]
        hl, hr = held_masks(m)
        self.assertFalse((hr if e["side"] == "right" else hl).any())               # the idle hand holds nothing
        d = np.array([kin.arm_arm_distance(m.arm[k], m.hand[k], (hl[k], hr[k])) for k in range(len(m.arm))])
        self.assertGreaterEqual(d.min(), self.need)
        self.assertAlmostEqual(m.clearance[0], d.min(), places=9)
        self.assertEqual(pl.dry_run_safety(self.home, self.post["open"], m.arm, m.hand), 0)
        self.assertEqual(m.safety_clips, 0)
        full = np.vstack([self.home, m.arm])
        self.assertLessEqual(np.abs(np.diff(full, axis=0)).max(), mo["arm_max_vel"] / 30 + 1e-9)
        np.testing.assert_allclose(m.arm[-1], self.home, atol=1e-9)
        np.testing.assert_allclose(m.hand[-1], self.post["open"])
        np.testing.assert_allclose(m.hand[:, isl], np.repeat(self.post["open"][None, isl], len(m.arm), 0))
        n = len(p.arm)
        np.testing.assert_allclose(m.arm[:lag, sl], np.repeat(self.home[None, sl], lag, 0))
        np.testing.assert_allclose(m.arm[lag:lag + n, sl], p.arm[:, sl])
        np.testing.assert_allclose(m.hand[lag:lag + n, sl], p.hand[:, sl])
        w, w0 = grasp_window([(q.split(":")[1], s, t) for q, s, t in m.phases if q.startswith(p.side + ":")], m.hand,
                             sl), grasp_window(p.phases, p.hand, sl)
        self.assertEqual((w[0], w[1]), (w0[0] + lag, w0[1] + lag))
        self.assertIsNone(grasp_window(idle.phases, idle.hand, isl))
        for k in range(len(m.arm)):                                                 # idle arm: plan checks
            self.assertGreaterEqual(pl.clearance(e["side"], m.arm[k], m.hand[k]), mo["counter_clearance"] - 2e-3)
            self.assertGreaterEqual(pl.bin_clearance(e["side"], m.arm[k], m.hand[k]), 0.003)
        names = ["aside"] + (["clear"] if R > e["aside_ticks"] else []) + ["back"]
        self.assertEqual([x[0] for x in idle.phases], names)
        self.assertLessEqual(R, lag + n)                                            # at the latest: plan over
        if R > e["aside_ticks"]:                                                    # earliest: one tick earlier fails
            a2 = idle.arm[R:, isl]
            e2 = copy.copy(idle)
            e2.arm = np.vstack([idle.arm[:R - 1], np.repeat(idle.arm[-1:], len(idle.arm) - R + 1, 0)])
            e2.arm[R - 1:R - 1 + len(a2), isl] = a2
            e2.hand = np.repeat(idle.hand[:1], len(e2.arm), 0)
            m2 = merge_plans(e2, p, lag, self.home, self.post["open"])
            h2 = held_masks(m2)
            self.assertLess(min(kin.arm_arm_distance(m2.arm[k], m2.hand[k], (h2[0][k], h2[1][k]))
                                for k in range(len(m2.arm))), self.need)
        self.assertEqual(e["added"], len(m.arm) - n)
        self.assertTrue(m.summary().startswith("EVADE") and "moves aside" in m.summary(), m.summary())
        return d

    def test_table(self):
        """The cases of the B1c problem statement: conflicting plans get an evade, clear plans come back as they are."""
        cases = [("red", 0.30, -0.16, True), ("red", 0.30, -0.08, True), ("red", 0.30, 0.00, False),
                 ("blue", 0.30, 0.16, True), ("blue", 0.30, 0.05, True), ("red", 0.30, 0.15, False),
                 ("blue", 0.30, -0.15, False), ("yellow", 0.30, 0.10, False), ("yellow", 0.30, -0.10, False)]
        for color, x, y, conflict in cases:
            with self.subTest(color=color, y=y):
                p = self.plan(color, x, y)
                a, h = p.arm.copy(), p.hand.copy()
                d0 = self.parked(p)
                self.assertEqual(d0 < self.need, conflict, f"{1000 * d0:.0f} mm")
                m, info = self.pl.keep_clear(p, self.home, self.post["open"])
                self.assertAlmostEqual(info["parked"][0], d0, places=9)       # the plan's minimum vs the parked arm
                if not conflict:
                    self.assertIs(m, p)                                    # unchanged, bit for bit
                    np.testing.assert_array_equal(m.arm, a)
                    np.testing.assert_array_equal(m.hand, h)
                    continue
                self.assertIsNotNone(m, info.get("why"))
                self.check_evade(m, p)
                np.testing.assert_array_equal(p.arm, a)                   # the single plan itself is not modified

    def test_cross_body_grid(self):
        """Cross-body picks over the object band (x 0.26-0.33, y to -/+0.17 m), both arms, side and centre bins, and
        counter places at the oracle's relay points: (nearly) every conflicting plan gets an evade."""
        rows, ok, conflicts = [], 0, 0
        for x in (0.26, 0.295, 0.33):
            for y in (-0.17, -0.12, -0.07, -0.02):
                for color, side, bn, sgn, v in (("red", "left", "left", 1, 0), ("blue", "right", "right", -1, 0),
                                                ("yellow", "left", "center", 1, 1),
                                                ("yellow", "right", "center", -1, 1)):
                    det = Detection(color, np.array([x, sgn * y, CZ + SC.object_height()]), 0, 0, "depth")
                    p = self.pl.plan_side(det, side, bn, self.home, self.post["open"], variant=v)
                    if not p.ok:
                        continue
                    m, info = self.pl.keep_clear(p, self.home, self.post["open"])
                    if m is p:
                        continue
                    conflicts += 1
                    if m is not None:
                        ok += 1
                        self.check_evade(m, p)
                    rows.append((color, side, x, sgn * y, round(1000 * info["parked"][0]),
                                 None if m is None else (m.evade["pose"], m.evade["lag"], m.evade["added"]),
                                 info.get("why")))
        for rx, ry in ((0.30, 0.0), (0.30, 0.05), (0.30, -0.05), (0.27, 0.06), (0.27, -0.06)):  # relay / counter places
            for side, oy in (("left", -0.15), ("right", 0.15)):
                det = Detection("blue" if side == "left" else "red", np.array([0.30, oy, CZ + SC.object_height()]),
                                0, 0, "depth")
                p = self.pl._plan_side_hints(det, side, "center", self.home, self.post["open"], place_xy=[rx, ry])
                if not p.ok:
                    continue
                m, info = self.pl.keep_clear(p, self.home, self.post["open"])
                if m is p:
                    continue
                conflicts += 1
                if m is not None:
                    ok += 1
                    self.check_evade(m, p)
                rows.append(("place", side, rx, ry, round(1000 * info["parked"][0]),
                             None if m is None else (m.evade["pose"], m.evade["lag"], m.evade["added"]),
                             info.get("why")))
        print(f"\n[parked grid] {ok}/{conflicts} conflicting plans evaded")
        for r in rows:
            print("  ", r)
        self.assertGreater(conflicts, 10)
        self.assertGreaterEqual(ok, conflicts - 1, [r for r in rows if r[5] is None])

    def test_no_evade_possible(self):
        p = self.plan("red", 0.30, -0.16)
        for poses, word in (([], "no two_arm.clear_poses"), ([list(self.home[7:] * ARM_MIRROR)], "already there"),
                            ([[0.0, 0.35, 0.0, -0.49, 0.0, 0.0, 0.0]], "no lag")):   # (as good as staying home)
            cfg = copy.deepcopy(self.cfg)
            cfg["two_arm"]["clear_poses"] = poses
            pl = Planner(cfg, SC, self.post, self.home, kin=self.pl.kin)
            m, info = pl.keep_clear(p, self.home, self.post["open"])
            self.assertIsNone(m, poses)
            self.assertIn(word, info["why"])
            self.assertLess(info["parked"][0], self.need)

    def test_clear_poses_mirrored_and_validated(self):
        cs = self.pl.clear_poses()
        self.assertGreaterEqual(len(cs), 1)
        kin = self.pl.kin
        for c in cs:
            np.testing.assert_allclose(c[7:], c[:7] * ARM_MIRROR)
            self.assertTrue(np.all(c >= kin.lo) and np.all(c <= kin.hi))
            for name in ("open", "closed"):                                # mirror image of the left pose's spheres
                cl, _ = kin.arm_spheres("left", c, self.post[name])
                cr, _ = kin.arm_spheres("right", c, self.post[name])
                np.testing.assert_allclose(cl * [1, -1, 1], cr, atol=1e-4)
        cfg = copy.deepcopy(self.cfg)
        cfg["two_arm"]["clear_poses"] = [[0.0, 0.35, 0.0, 1.95, 0.0, 0.0, 0.0]]   # elbow past the sim margin
        with self.assertRaises(ValueError):
            Planner(cfg, SC, self.post, self.home, kin=kin).clear_poses()

    def test_profiles(self):
        import os
        env = os.environ.pop("AALTO_SET", None)                        # the files' values, not an experiment's
        try:
            self.assertFalse(load_config(profile="none")["two_arm"]["avoid_parked"])
            self.assertTrue(load_config(profile="sim")["two_arm"]["avoid_parked"])
        finally:
            if env is not None:
                os.environ["AALTO_SET"] = env

    def test_distance_matrix_matches_brute_force(self):
        p = self.plan("red", 0.30, -0.16)
        kin = self.pl.kin
        ca, ra = kin.side_spheres("left", p.arm[::7], p.hand[::7])
        q = self.home.copy()
        idle = np.repeat(self.home[None], 5, 0)
        idle[:, 7:] = self.home[7:] + np.linspace(0, 1, 5)[:, None] * (self.pl.clear_poses()[0][7:] - self.home[7:])
        cb, rb = kin.side_spheres("right", idle, np.repeat(self.post["open"][None], 5, 0))
        held = held_masks(p)[0][::7]
        D = kin.sphere_distance_matrix(ca, ra, cb, rb, wa=held, below=0.05)
        for i in range(len(ca)):
            for j in range(len(cb)):
                q = np.r_[p.arm[7 * i, :7], idle[j, 7:]]
                h = np.r_[p.hand[7 * i, :7], self.post["open"][7:]]
                b = kin.arm_arm_distance(q, h, (held[i], False))
                if b < 0.05:
                    self.assertAlmostEqual(D[i, j], b, places=9)
                else:
                    self.assertLessEqual(D[i, j], b + 1e-9)
                    self.assertGreaterEqual(D[i, j], 0.05)


if __name__ == "__main__":
    unittest.main()
