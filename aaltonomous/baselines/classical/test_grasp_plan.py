"""Offline tests of drop slots (grasp.slots), neighbour-aware plans (neighbours) and face-aligned grasps
(grasp.face_align) in the classical planner / policy. Run in the sim container:
    aalto.sh py -m unittest aaltonomous/baselines/classical/test_grasp_plan.py -v
"""
import copy
import os
import unittest

import numpy as np

# mechanism tests on profile sim_legacy (see test_classical.py); TestSimProfilePlans: profile sim's own grasp
os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics, cylinder_distance
from .perception import Detection
from .planner import Plan, Planner
from .policy import ClassicalPolicy
from .test_classical import NB

SC = Scene.load()
CZ, H = SC.counter_z, SC.object_height()


def det(color, x, y, shape=None, yaw=None, conf=1.0):
    d = Detection(color, np.array([x, y, CZ + H]), 900, 900, "depth", yaw=yaw)
    d.shape, d.shape_conf = shape, conf
    return d


def planner(profile=None, **ov):   # None: $AALTO_PROFILE (sim_legacy here)
    cfg = load_config(profile=profile, overrides=ov)
    return Planner(cfg, SC, hand_postures(cfg), home_arm(cfg)), hand_postures(cfg), home_arm(cfg)


def finger_yaw(pl, p):
    """Palm x axis (finger direction) at the grasp, pelvis yaw (rad)."""
    e = next(e for n, s, e in p.phases if n == "descend")
    pl.kin._fk(p.arm[e - 1])
    x = pl.kin.data.oMf[pl.kin.palm[p.side]].rotation[:, 0]
    return float(np.arctan2(x[1], x[0]))


def wrap(a, per=2 * np.pi):
    return (a + per / 2) % per - per / 2


# ---------------------------------------------------------------------------------------------------- drop slots
class TestSlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pl, cls.post, cls.home = planner()
        cls.pl.prepare()

    def test_geometry_reach_and_spacing(self):
        """Every (bin, arm) that delivers there: 5 slots inside the inner footprint, >= wall_clear from every inner
        wall, >= min_spacing apart, reachable (the reach test), the first two lateral on the near row ~12 cm apart.
        Same for the base (robot) config: the slots come from the scene geometry + that config's reach. (Profile sim:
        TestSimProfilePlans.)"""
        for profile in ("sim_legacy", "none"):
            pl = self.pl if profile == "sim_legacy" else planner("none")[0]
            s = pl.cfg["grasp"]["slots"]
            combos = [("left", "left"), ("center", "left"), ("center", "right"), ("right", "right")]
            for b, side in combos:
                with self.subTest(profile=profile, bin=b, side=side):
                    S = pl.slots(b, side)
                    # (base: the cage's grasp point sits 14 mm nearer the wrist than the old claw's: only the near
                    # row (3) passes the reach test at the scene's box row)
                    self.assertEqual(len(S), s["count"] if profile == "sim_legacy" else 3)
                    lo, hi = SC.bin_inner_box(b)
                    m = SC.pelvis_to_mat(np.c_[S, np.zeros(len(S))])[:, :2]
                    self.assertTrue(np.all(m >= lo[:2] + s["wall_clear"] - 1e-9) and np.all(m <= hi[:2] - s["wall_clear"]
                                                                                            + 1e-9), S)
                    D = np.linalg.norm(S[:, None] - S[None], axis=-1) + np.eye(len(S))
                    self.assertGreaterEqual(D.min(), s["min_spacing"] - 1e-6)
                    self.assertTrue(all(pl._slot_reach(side, p) for p in S))
                    self.assertAlmostEqual(S[0][0], S[1][0])                    # two lateral slots on the near row
                    self.assertGreater(abs(S[0][1] - S[1][1]), 0.09)
                    self.assertLess(S[0][0], np.min(m[:, 1]) + SC.origin[0] + 1e-9 + 1.0)   # (x of row 0 = nearest)
                    self.assertAlmostEqual(S[0][0], S[:, 0].min())
        # centre bin: each arm starts on its own side
        self.assertGreater(self.pl.slots("center", "left")[0][1], 0)
        self.assertLess(self.pl.slots("center", "right")[0][1], 0)

    def test_full_plan_to_every_slot(self):
        """A red object (left bin) planned to each left-arm slot in turn: every plan passes all checks, its release
        point is the slot."""
        S = self.pl.slots("left", "left").copy()
        d = det("red", 0.30, 0.15)
        try:
            for k in range(len(S)):
                self.pl._slots[("left", "left")] = S[k:k + 1]
                p = self.pl.plan(d, self.home, self.post["open"])
                self.assertTrue(p.ok, (k, p.reason))
                self.assertEqual(p.safety_clips, 0)
                np.testing.assert_allclose(p.waypoints["drop"][:2], S[k])
        finally:
            self.pl._slots[("left", "left")] = S

    def test_farthest_free_slot(self):
        pl = self.pl
        S = pl.slots("left", "left")
        pl.drop_used = {}
        np.testing.assert_allclose(pl.drop_slot("left", "left"), S[0])
        got = []
        for _ in range(5):                                   # 5 objects of one colour: 5 distinct places
            p = pl.drop_slot("left", "left")
            got.append(p)
            pl.drop_used = {"left": list(got)}
        np.testing.assert_allclose(np.array(got), S)        # (the preference order is farthest-first)
        D = np.linalg.norm(np.array(got)[:, None] - np.array(got)[None], axis=-1) + np.eye(5)
        self.assertGreaterEqual(D.min(), 0.06 - 1e-6)
        # used points anywhere (oracle: objects really lying there) -> the slot farthest from them
        pl.drop_used = {"left": [S[1] + [0.01, -0.01]]}
        np.testing.assert_allclose(pl.drop_slot("left", "left"), S[0])
        pl.drop_used = {"left": [S[0]]}
        np.testing.assert_allclose(pl.drop_slot("left", "left", extra=[S[1]]), S[2])
        pl.drop_used = {}

    def test_pair_into_one_bin_gets_two_slots(self):
        """plan_pair with the first plan's release point to avoid (as ClassicalPolicy._pair does): the second arm's
        drop is the other end of the centre bin's near row, ~12 cm away; the planner state is restored."""
        pl, home, post = self.pl, self.home, self.post
        first = pl.plan(det("yellow", 0.30, 0.12), home, post["open"])
        self.assertTrue(first.ok, first.reason)
        m, p2, why = pl.plan_pair(first, det("yellow", 0.30, -0.12), "center", home, post["open"],
                                  drop_avoid=[first.waypoints["drop"][:2]])
        self.assertTrue(p2.ok, p2.reason)
        gap = np.linalg.norm(first.waypoints["drop"][:2] - p2.waypoints["drop"][:2])
        self.assertGreater(gap, 0.12)
        self.assertEqual(pl._drop_avoid, [])
        self.assertIsNotNone(m, why)

    def test_off_uses_drop_offsets(self):
        pl, post, home = planner(**{"grasp.slots.enabled": False})
        self.assertEqual(pl._make_slots.__name__, "_make_slots")
        pl.drop_offsets = {"left": (0.03, -0.03)}
        p = pl.plan(det("red", 0.30, 0.15), home, post["open"])
        self.assertTrue(p.ok, p.reason)
        np.testing.assert_allclose(p.waypoints["drop"][:2], pl.drop_xy("left", offset=False) + [0.03, -0.03])
        self.assertEqual(pl._slots, {})                        # never computed


class TestSlotBookkeeping(unittest.TestCase):
    """ClassicalPolicy: release points of DELIVERED objects consume slots; failed / aborted attempts do not."""

    def _policy(self, **ov):
        pol = ClassicalPolicy(log=None, overrides={"two_arm.enabled": False, **ov})
        pol.perception.detect = lambda rgb, depth: []
        return pol

    @staticmethod
    def _plan(pol, b, xy):
        sl = ArmKinematics.sl("left")
        h = np.repeat(pol.post["open"][None], 60, 0)
        h[10:40, sl] = pol.post["closed"][sl]
        return Plan(True, side="left", target_bin=b, waypoints={"grasp": np.zeros(3), "drop": np.r_[xy, 0.1]},
                    ik={"grasp": (True, 0.0, 0.0)}, arm=np.repeat(pol.home[None], 60, 0), hand=h,
                    phases=[("descend", 0, 10), ("close", 10, 20), ("lift", 20, 30), ("open", 40, 50),
                            ("home", 50, 60)])

    def _run(self, pol, b, xy, outcome):
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol._start(self._plan(pol, b, xy), 0.0)
        for x in pol.execs.values():          # what _watch would have concluded
            x["verdict"] = "held" if outcome == "held" else "empty" if outcome else "unknown"
            if outcome in ("miss", "slip"):
                x["outcome"], x["aborted"] = outcome, True
        pol._finish(0.0)

    def test_delivered_consume_failed_do_not(self):
        pol = self._policy()
        self.assertTrue(pol.verify_on)
        self._run(pol, "left", [0.44, 0.20], "held")
        self._run(pol, "left", [0.44, 0.32], "miss")           # aborted: no slot used
        self._run(pol, "left", [0.44, 0.26], "slip")
        self._run(pol, "right", [0.44, -0.20], None)           # no verdict: not verified held
        self._run(pol, "counter", [0.30, 0.0], "held")         # relocation: no bin
        self.assertEqual(list(pol.drops_at), ["left"])
        np.testing.assert_allclose(pol.drops_at["left"], [[0.44, 0.20]])
        self.assertEqual(pol.drops, {"left": 3, "right": 1})   # (drop_offsets still count started drops)
        pol._look({"head_rgb": None, "head_depth": None, "t": 0.0})
        np.testing.assert_allclose(pol.planner.drop_used["left"], [[0.44, 0.20]])
        self.assertEqual(pol.planner.drop_used["right"], [])
        S = pol.planner.slots("left", "left")
        np.testing.assert_allclose(pol.planner.drop_slot("left", "left"), S[1])   # the far end of the near row
        pol.reset()
        self.assertEqual(pol.drops_at, {})

    def test_unverified_release_counts_without_verification(self):
        pol = self._policy(**{"verify.enabled": False})
        self.assertFalse(pol.verify_on)
        self._run(pol, "left", [0.44, 0.20], None)
        np.testing.assert_allclose(pol.drops_at["left"], [[0.44, 0.20]])


class TestSlotEpisode(unittest.TestCase):
    def test_same_colour_objects_end_in_the_bin_below_the_rim(self):
        """Mock episode with stacking (an object dropped within 4.5 cm of one in the bin lands on it: above the
        rim). Seed 0: two red + two blue objects. Slots: all correct; one fixed drop point (offsets off): stacked."""
        from .test_classical import run_mock
        out = {}
        for on in (True, False):
            ov = {"two_arm.enabled": False, "grasp.slots.enabled": on, "grasp.drop_offsets": [[0.0, 0.0]]}
            pol, env, info, starts = run_mock(0, {"stack_radius": 0.045}, overrides=ov)
            self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
            colours = [o["color"] for o in env.backend.objects]
            self.assertTrue(any(colours.count(c) >= 2 for c in set(colours)), colours)
            out[on] = env.backend.outcomes(pol.mapping)
            if on:
                for b, pts in pol.drops_at.items():
                    P = np.array(pts)
                    if len(P) > 1:
                        D = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P))
                        self.assertGreater(D.min(), 0.1)
        self.assertEqual(out[True].count("correct"), len(out[True]), out[True])
        self.assertLess(out[False].count("correct"), len(out[False]), out[False])


# ---------------------------------------------------------------------------------------------------- neighbours
class TestNeighbours(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pl, cls.post, cls.home = planner(**NB)
        cls.margin = float(cls.pl.cfg["neighbours"]["margin"])

    def brute(self, p, nbs):
        """Every tick of plan p: sphere model of its arm vs the neighbours' cylinders (true shape radii)."""
        c, r = self.pl.kin.hand_spheres(p.side, p.arm, p.hand)
        cyl = self.pl.neighbour_cylinders(nbs)[0]
        return cylinder_distance(c, r, cyl, CZ)

    def test_close_neighbour_every_tick_keeps_the_margin(self):
        """A cube 6 cm from a cylinder (the incident: a cube pushed 5.6 cm from a cylinder): where the stock plan
        sweeps through the neighbour, the plan found keeps the margin at every tick (another finger direction);
        where nothing can, the object is blocked by it."""
        tgt = det("red", 0.30, 0.12, "cube", 0.3)
        stock = self.pl.plan(tgt, self.home, self.post["open"])
        self.assertTrue(stock.ok)
        for ang in (150, 180, 0):
            a = np.radians(ang)
            nb = det("blue", 0.30 + 0.06 * np.cos(a), 0.12 + 0.06 * np.sin(a), "cylinder")
            with self.subTest(nb=ang):
                self.assertLess(self.brute(stock, [nb]).min(), 0.0)          # the stock plan overlaps it
                p = self.pl.plan(tgt, self.home, self.post["open"], neighbours=[nb])
                if ang == 0:                                                 # in front: no clear candidate
                    self.assertFalse(p.ok)
                    self.assertTrue(p.blocked and "blue(+0.360,+0.120)" in p.reason, p.reason)
                    continue
                self.assertTrue(p.ok, p.reason)
                self.assertNotEqual(p.hint, stock.hint)
                D = self.brute(p, [nb])
                self.assertGreaterEqual(D.min(), self.margin - 1e-9)
                self.assertAlmostEqual(p.nb_clear, D.min(), places=9)
                self.assertEqual(p.safety_clips, 0)

    def test_blocked_reason_names_the_object(self):
        tgt = det("red", 0.30, 0.12, "cube", 0.0)
        nb = det("blue", 0.356, 0.12, "cylinder")              # 5.6 cm in front: in the fingers' way
        p = self.pl.plan(tgt, self.home, self.post["open"], neighbours=[nb])
        self.assertFalse(p.ok)
        self.assertTrue(p.blocked)
        self.assertIn("blue(+0.356,+0.120)", p.reason)
        self.assertEqual(self.pl._nb, None)                     # restored after the call
        far = det("blue", 0.30, -0.10, "cylinder")
        q = self.pl.plan(tgt, self.home, self.post["open"], neighbours=[far])
        self.assertTrue(q.ok and not q.blocked, q.reason)
        np.testing.assert_allclose(q.arm, self.pl.plan(tgt, self.home, self.post["open"]).arm)   # far: no change

    def test_cylinder_model(self):
        rows, tags = self.pl.neighbour_cylinders([det("red", 0.3, 0.1, "cube"), det("blue", 0.3, 0.0, "cylinder"),
                                                  det("yellow", 0.3, -0.1, "triangle"), det("red", 0.2, 0.0)])
        np.testing.assert_allclose(rows[:, 2], [0.0225 * np.sqrt(2), 0.0225, 0.030, 0.0225 * np.sqrt(2)], atol=1e-6)
        np.testing.assert_allclose(rows[:, 3], CZ + H)
        self.assertEqual(tags[1], "blue(+0.300,+0.000)")
        c = np.array([[[0.3, 0.1 + 0.0318 + 0.01, CZ + 0.02]], [[0.3, 0.1, CZ + H + 0.02]]])   # beside / above
        np.testing.assert_allclose(cylinder_distance(c, np.zeros(1), rows[:1], CZ)[:, 0], [0.01, 0.02], atol=1e-4)
        off = planner(**{"neighbours.enabled": False})[0]
        self.assertIsNone(off.neighbour_cylinders([det("red", 0.3, 0.1)])[0])

    def test_least_bad_rule(self):
        """neighbours.margin so large that every candidate is blocked: plan(least_bad=True) accepts >= floor, keeps
        the candidate with the most clearance (both arms, perpendicular pinch axes first) and says so."""
        pl = planner(**{**NB, "neighbours.margin": 0.08, "neighbours.floor": 0.0, "neighbours.least_bad_good": 1.0})[0]
        tgt, nb = det("yellow", 0.30, 0.10, "cube", 0.0), det("blue", 0.30, 0.0, "cylinder")
        p = pl.plan(tgt, self.home, self.post["open"], neighbours=[nb])
        self.assertTrue(p.blocked and not p.ok, p.reason)
        q = pl.plan(tgt, self.home, self.post["open"], neighbours=[nb], least_bad=True)
        self.assertTrue(q.ok, q.reason)
        self.assertTrue(q.least_bad)
        self.assertGreaterEqual(q.nb_clear, 0.0)
        cands = []
        orig = pl._plan_side_hints

        def spy(d, side, *a, **kw):
            cands.append([c["kind"] for c in pl._candidates(d, side, kw.get("variant", 0))])
            return orig(d, side, *a, **kw)
        pl._plan_side_hints = spy
        q2 = pl.plan(tgt, self.home, self.post["open"], neighbours=[nb], least_bad=True)
        self.assertEqual(len(cands), 2)                                     # both arms planned
        self.assertEqual(cands[0][:2], ["perpendicular"] * 2)
        brute = cylinder_distance(*pl.kin.hand_spheres(q2.side, q2.arm, q2.hand), pl.neighbour_cylinders([nb])[0], CZ)
        self.assertAlmostEqual(brute.min(), q2.nb_clear, places=9)
        quick = planner(**{**NB, "neighbours.margin": 0.08, "neighbours.floor": 0.0})[0]   # least_bad_good 0: stops at
        q3 = quick.plan(tgt, self.home, self.post["open"], neighbours=[nb], least_bad=True)  # the first >= 0 mm
        self.assertTrue(q3.ok and q3.least_bad and q3.nb_clear >= 0.0, q3.reason)
        none = planner(**{**NB, "neighbours.margin": 0.08, "neighbours.floor": 0.07})[0]
        r = none.plan(tgt, self.home, self.post["open"], neighbours=[nb], least_bad=True)
        self.assertFalse(r.ok)
        self.assertTrue(r.blocked)

    def test_abort_home_move_keeps_clear(self):
        """Planner.abort with the plan's neighbour cylinders: a cylinder on the joint-space path home rejects it."""
        pl, post, home = self.pl, self.post, self.home
        p = pl.plan(det("red", 0.30, 0.15), home, post["open"])
        e = next(e for n, s, e in p.phases if n == "lift")
        k = e + 8
        ab = Planner(pl.cfg, SC, post, home, ArmKinematics(pl.cfg))
        a, h, info = ab.abort(p.side, p.arm[k - 1], p.hand[k - 1], p.arm[k:], p.hand[k:])
        self.assertIsNotNone(a, info)
        n = info["decel_ticks"]
        mid = pl.kin.grasp_pose(p.side, a[n + (info["home_ticks"] // 2)])[0]
        cyl = np.array([[mid[0], mid[1], 0.05, mid[2] + 0.05]])                  # an "object" on the path
        a2, _, why = ab.abort(p.side, p.arm[k - 1], p.hand[k - 1], p.arm[k:], p.hand[k:], neighbours=cyl)
        self.assertIsNone(a2)
        self.assertIn("from an object while moving home", why)
        far = np.array([[0.30, -0.20, 0.03, CZ + H]])
        a3, _, info3 = ab.abort(p.side, p.arm[k - 1], p.hand[k - 1], p.arm[k:], p.hand[k:], neighbours=far)
        np.testing.assert_allclose(a3, a)


class TestNeighbourPolicy(unittest.TestCase):
    """Policy order with blocked objects (real planner, stubbed perception)."""

    def _pol(self, dets, **ov):
        pol = ClassicalPolicy(log=None, overrides={**NB, "two_arm.enabled": False, "two_arm.avoid_parked": False, **ov})
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol.perception.detect = lambda rgb, depth: dets
        return pol

    OBS = {"head_rgb": None, "head_depth": None, "t": 0.0}

    def test_blocked_object_waits_for_its_neighbour(self):
        a = det("red", 0.30, 0.12, "cube", 0.0)                    # policy.order largest: first
        a.n_px = 1000
        b = det("yellow", 0.365, 0.12, "cylinder")                 # 6.5 cm in front of it (clear itself)
        pol = self._pol([a, b], **{"policy.order": "largest"})
        kind, plan, _ = pol._look(self.OBS)
        self.assertEqual(kind, "plan")
        ra, rb = pol.objs
        self.assertEqual((ra["color"], rb["color"]), ("red", "yellow"))
        self.assertIs(plan.rec, rb)
        self.assertIsNone(ra["failed_at"])
        self.assertIsNotNone(ra["blocked"])
        self.assertTrue(pol._eligible(ra, ra["xy"]))
        self.assertTrue(any("blocked (" in m and "yellow(" in m for _, m in pol.events))
        n = len(pol.plans)
        pol._look(self.OBS)                                          # same scene: not planned again (moot)
        self.assertEqual(sum(1 for q in pol.plans[n:] if q.color == "red"), 0)
        pol.perception.detect = lambda rgb, depth: [a]               # the neighbour left
        kind, plan, _ = pol._look(self.OBS)
        self.assertEqual((kind, plan.rec), ("plan", ra))
        self.assertIsNone(ra["blocked"])

    def test_least_bad_when_nothing_else(self):
        dets = [det("red", 0.30, 0.12, "cube", 0.0), det("blue", 0.30, 0.03, "cylinder")]
        pol = self._pol(dets, **{"neighbours.margin": 0.08, "neighbours.floor": -0.02})
        kind, plan, _ = pol._look(self.OBS)
        self.assertEqual(kind, "plan")
        self.assertTrue(plan.least_bad)
        self.assertGreaterEqual(plan.nb_clear, -0.02)
        self.assertEqual(len([m for _, m in pol.events if "blocked (" in m]), 2)
        self.assertEqual(len([m for _, m in pol.events if m.startswith("least-bad:")]), 1)
        off = self._pol(dets, **{"neighbours.margin": 0.08, "neighbours.least_bad": False})
        self.assertEqual(off._look(self.OBS)[0], "none")
        self.assertTrue(all(o["failed_at"] is None for o in off.objs))   # waiting, not failed
        low = self._pol(dets, **{"neighbours.margin": 0.08, "neighbours.floor": 0.07})
        self.assertEqual(low._look(self.OBS)[0], "none")
        self.assertTrue(all(o["failed_at"] is None for o in low.objs))      # still blocked, not failed
        self.assertEqual(len([m for _, m in low.events if "no least-bad plan" in m]), 2)

    def test_order_free_space_first(self):
        dets = [det("red", 0.28, 0.12), det("red", 0.33, 0.15), det("blue", 0.30, -0.12)]
        pol = self._pol(dets)
        self.assertEqual([pol._crowded(d, dets) for d in dets], [1, 1, 0])
        pol2 = self._pol(dets, **{"neighbours.order_free": 0.0})
        self.assertEqual([pol2._crowded(d, dets) for d in dets], [0, 0, 0])

    def test_pair_second_arm_avoids_the_first_object(self):
        """two_arm: the second plan's neighbours are every other detection incl. the first plan's object."""
        dets = [det("red", 0.30, 0.15), det("blue", 0.30, -0.15), det("yellow", 0.27, 0.0)]
        pol = self._pol(dets, **{"two_arm.enabled": True})
        seen, orig = [], pol.planner.plan_pair

        def spy(first, d, *a, **kw):
            seen.append((d, kw.get("neighbours")))
            return orig(first, d, *a, **kw)
        pol.planner.plan_pair = spy
        kind, m, _ = pol._look(self.OBS)
        self.assertTrue(seen)
        d2, nbs = seen[0]
        first_det = next(d for d in dets if d.color == m.parts[0].color) if m.parts else None
        self.assertNotIn(d2, nbs)
        self.assertEqual(len(nbs), len(dets) - 1)
        if first_det is not None:
            self.assertIn(first_det, nbs)

    def test_off_is_unchanged(self):
        """neighbours.enabled false: the neighbours argument changes nothing."""
        pl, post, home = planner(**{"neighbours.enabled": False})
        tgt, nb = det("red", 0.30, 0.12), det("blue", 0.356, 0.12, "cylinder")
        p, q = pl.plan(tgt, home, post["open"], neighbours=[nb]), pl.plan(tgt, home, post["open"])
        self.assertTrue(p.ok and q.ok)
        np.testing.assert_array_equal(p.arm, q.arm)
        self.assertIsNone(p.nb)


# ---------------------------------------------------------------------------------------------------- face-aligned
class _OldOraclePlanner(Planner):
    """The oracle's face-aligned planner before it moved into Planner (sim/oracle.py, 2026-11-14), as a reference."""
    TRI_PHI = 105.0

    def _plan_side_hints(self, det, side, bin_name, arm_q, hand_q, variant=0, **kw):
        rej = self._reach_reject(det, side, bin_name, kw.get("place_xy"))
        return rej or self._faces(det, side, bin_name, arm_q, hand_q, variant=variant, **kw)

    def _faces(self, det, side, bin_name, arm_q, hand_q, variant=0, **kw):
        from .planner import tri_shift
        table = {"cube": (0.0, 4), "triangle": (np.pi / 2 + np.radians(self.TRI_PHI), 3)}
        shape, yaw = det.shape, det.yaw
        if shape in table and yaw is not None:
            off, n = table[shape]
            ref = np.radians({"left": -25.0, "right": 25.0}[side])
            cands = []
            for k in range(n):
                a = yaw + off + 2 * np.pi * k / n
                da = abs((a - ref + np.pi) % (2 * np.pi) - np.pi)
                if da <= np.radians(75.0):
                    cands.append((da, a))
            saved = (self.kin.hint, self.kin.w_hint)
            cands = sorted(cands)
            if cands:
                cands = cands[variant % len(cands):] + cands[:variant % len(cands)]
            try:
                for _, a in cands:
                    f = np.array([np.cos(a), np.sin(a), 0.0])
                    d2 = copy.copy(det)
                    if shape == "triangle":
                        d2.top = det.top + tri_shift(a - (yaw + np.pi / 2)) * f
                    self.kin.hint, self.kin.w_hint = f, 3.0
                    p = self._plan_side(d2, side, bin_name, arm_q, hand_q, **kw)
                    if p.ok and self._finger_err(p, side, f) <= 12.0:
                        return p
            finally:
                self.kin.hint, self.kin.w_hint = saved
        return Planner._plan_side_hints(self, det, side, bin_name, arm_q, hand_q, variant=variant, **kw)


class TestFaceAlign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pl, cls.post, cls.home = planner()
        cls.tol = float(cls.pl.cfg["grasp"]["face_align"]["tol_deg"])

    def test_cube_and_triangle_fingers_on_a_face(self):
        """Cubes always, triangles mostly (their 1-2 candidate directions may be out of reach: then the hints)
        get a face-aligned plan whose achieved finger direction is within tol_deg of a face normal / apex + phi."""
        fa = self.pl.cfg["grasp"]["face_align"]
        self.assertTrue(fa["enabled"])
        for shape, period, off in (("cube", np.pi / 2, 0.0),
                                   ("triangle", 2 * np.pi / 3, np.pi / 2 + np.radians(fa["triangle_phi_deg"]))):
            n_face = 0
            for yaw_deg in (-40, -15, 0, 20, 44):
                for x, y in ((0.30, 0.12), (0.30, -0.10)):
                    with self.subTest(shape=shape, yaw=yaw_deg, y=y):
                        d = det("yellow", x, y, shape, np.radians(yaw_deg))
                        p = self.pl.plan(d, self.home, self.post["open"])
                        self.assertTrue(p.ok, p.reason)
                        if p.face is None:
                            self.assertEqual(shape, "triangle")
                            continue
                        n_face += 1
                        self.assertTrue(p.reason.startswith("face-aligned"), p.reason)
                        err = np.degrees(abs(wrap(finger_yaw(self.pl, p) - d.yaw - off, period)))
                        self.assertLessEqual(err, self.tol + 1e-6)
            self.assertGreaterEqual(n_face, 10 if shape == "cube" else 6)

    def test_triangle_grasp_point_at_the_pinch_midpoint(self):
        d = det("yellow", 0.30, 0.12, "triangle", 0.0)
        p = self.pl.plan(d, self.home, self.post["open"])
        self.assertTrue(p.ok and p.face is not None, p.reason)
        a = np.radians(p.face)
        from .planner import tri_shift
        want = d.top[:2] + tri_shift(a - np.pi / 2, 0.030) * np.array([np.cos(a), np.sin(a)])
        np.testing.assert_allclose(p.waypoints["grasp"][:2], want, atol=1e-9)

    def test_fallbacks(self):
        pl, home, post = self.pl, self.home, self.post
        for d in (det("red", 0.30, 0.12),                              # no shape (camera: not sure)
                  det("red", 0.30, 0.12, "cylinder"),                  # no faces
                  det("red", 0.30, 0.12, "cube", None),                # no yaw
                  det("red", 0.30, 0.12, "cube", 0.3, conf=0.3)):      # weak fit: below face_align.min_conf
            p = pl.plan(d, home, post["open"])
            self.assertTrue(p.ok, p.reason)
            self.assertIsNone(p.face)
            self.assertEqual(p.hint, 0)
        no_tri = planner(**{"grasp.face_align.triangle": False})[0]
        self.assertEqual(no_tri.face_dirs(det("red", 0.3, 0.12, "triangle", 0.0), "left"), [])
        self.assertTrue(no_tri.face_dirs(det("red", 0.3, 0.12, "cube", 0.0), "left"))
        base = planner("none")[0]
        self.assertFalse(base.cfg["grasp"]["face_align"]["enabled"])
        self.assertEqual(base.face_dirs(det("red", 0.3, 0.12, "cube", 0.0), "left"), [])

    def test_variants_rotate_faces_then_hints(self):
        d = det("red", 0.30, 0.12, "cube", np.radians(20))
        dirs = [self.pl.face_dirs(d, "left", v) for v in range(3)]
        self.assertEqual(len(dirs[0]), 2)                              # 20 and -70 deg are within 75 of -25
        self.assertEqual(dirs[1], dirs[0][1:] + dirs[0][:1])
        kinds = [c["kind"] for c in self.pl._candidates(d, "left", 1)]
        self.assertEqual(kinds, ["face", "face", "hint", "hint", "hint"])
        self.assertEqual([c["hint"] for c in self.pl._candidates(d, "left", 1)][2:], [1, 2, 0])

    def test_oracle_plans_unchanged(self):
        """Old oracle face-aligned planner vs Planner with grasp.face_align (oracle overrides, no neighbours, no
        slots): the same trajectories for cubes / triangles at several yaws, both arms. The old planner aligned cubes
        and put triangles at 105 deg: those two keys as it had them (the oracle's grasp itself as it is now)."""
        from aaltonomous.sim.oracle import ORACLE_OVERRIDES
        ov = dict(ORACLE_OVERRIDES, **{"grasp.slots.enabled": False, "grasp.face_align.cube": True,
                                       "grasp.face_align.triangle_phi_deg": _OldOraclePlanner.TRI_PHI,
                                       "grasp.face_align.carry_weight": None})   # (it kept the direction to the bin)
        cfg = load_config(overrides=ov)
        post, home = hand_postures(cfg), home_arm(cfg)
        new = Planner(cfg, SC, post, home)
        old = _OldOraclePlanner(cfg, SC, post, home, kin=new.kin)
        n_ok = 0
        for shape in ("cube", "triangle"):
            for yaw_deg, (x, y) in ((-30, (0.30, 0.12)), (10, (0.28, -0.08)), (40, (0.33, 0.02))):
                with self.subTest(shape=shape, yaw=yaw_deg):
                    d = det("blue" if y < 0 else "red", x, y, shape, np.radians(yaw_deg))
                    p, q = new.plan(d, home, post["open"], relocate=False), old.plan(d, home, post["open"],
                                                                                    relocate=False)
                    self.assertEqual(p.ok, q.ok, (p.reason, q.reason))
                    if p.ok:
                        n_ok += 1
                        np.testing.assert_allclose(p.arm, q.arm)
                        np.testing.assert_allclose(p.hand, q.hand)
        self.assertGreaterEqual(n_ok, 5)                                # (not a comparison of two failures)


class TestSimProfilePlans(unittest.TestCase):
    """Planning with profiles.sim's own grasp (thumb-opposed pinch, all-joint IK margins, triangles towards an apex,
    cubes along a face normal): these fail when a re-tune of the sim grasp breaks planning. Measured 2026-11-14."""

    @classmethod
    def setUpClass(cls):
        cls.pl, cls.post, cls.home = planner("sim")

    def test_grid_plans_without_clips(self):
        """Cubes (2 yaws), triangles (3 yaws), cylinders at 8 points over both halves of the object band, own-side
        bins: every plan found (48/48) on its own side with 0 safety clips; triangles face-aligned at an apex
        (phi 0) and cubes along a face normal wherever one is within face_align.max_dev_deg of the arm's natural
        direction, finger direction within tol_deg; cylinders on the stock hints."""
        fa = self.pl.cfg["grasp"]["face_align"]
        self.assertEqual([fa[k] for k in ("enabled", "cube", "triangle", "triangle_phi_deg")], [True, True, True, 0])
        n, ok, aligned, cubes = 0, 0, 0, 0
        for shape, yaws in (("cube", (-30, 30)), ("triangle", (-40, 0, 40)), ("cylinder", (None,))):
            for x in (0.27, 0.32):
                for y in (0.15, 0.07, -0.07, -0.15):
                    for yd in yaws:
                        d = det("red" if y > 0 else "blue", x, y, shape, None if yd is None else np.radians(yd))
                        p = self.pl.plan(d, self.home, self.post["open"], relocate=False)
                        n += 1
                        with self.subTest(shape=shape, x=x, y=y, yaw=yd):
                            if not p.ok:
                                continue
                            ok += 1
                            self.assertEqual((p.side, p.safety_clips), ("left" if y > 0 else "right", 0))
                            if p.face is None:
                                self.assertFalse(p.reason.startswith("face-aligned"), p.reason)
                                self.assertIsNotNone(p.hint)
                                continue
                            self.assertIn(shape, ("triangle", "cube"))      # cylinders: stock hints
                            a = np.radians(p.face)
                            if shape == "triangle":                         # an apex: yaw + 90 + k * 120 deg
                                aligned += 1
                                self.assertAlmostEqual(wrap(a - d.yaw - np.pi / 2, 2 * np.pi / 3), 0.0, places=6)
                            else:                                           # a face normal: yaw + k * 90 deg
                                cubes += 1
                                self.assertAlmostEqual(wrap(a - d.yaw, np.pi / 2), 0.0, places=6)
                            err = np.degrees(abs(wrap(finger_yaw(self.pl, p) - a)))
                            self.assertLessEqual(err, fa["tol_deg"] + 1e-6)
        self.assertGreaterEqual(ok, 0.9 * n)
        self.assertGreaterEqual(aligned, 12)
        self.assertGreaterEqual(cubes, 8, "cube plans that are face-aligned (of 16)")

    def test_standing_prism_grasped_higher(self):
        """grasp.face_align.triangle_height_frac (sim): a standing prism's grasp point sits at that fraction of the
        object height (face-aligned or stock hint), a cube's at grasp.height_frac as before; a lying prism keeps
        lying_height_frac. Base profile: null (unchanged)."""
        g, fa = self.pl.cfg["grasp"], self.pl.cfg["grasp"]["face_align"]
        thf = fa["triangle_height_frac"]
        self.assertIsNotNone(thf)
        self.assertGreater(thf, g["height_frac"])
        H, zc = self.pl.height, SC.counter_z
        for shape, want in (("triangle", thf), ("cube", g["height_frac"])):
            for yd, y in ((-40, 0.15), (0, -0.07), (40, -0.15)):
                with self.subTest(shape=shape, yaw=yd, y=y):
                    p = self.pl.plan(det("red", 0.30, y, shape, np.radians(yd)), self.home, self.post["open"],
                                     relocate=False)
                    self.assertTrue(p.ok, p.reason)
                    self.assertGreaterEqual(p.waypoints["grasp"][2], zc + want * H - 1e-6)   # (raised only to
                    self.assertLess(p.waypoints["grasp"][2], zc + want * H + 0.004)          # clear the counter)
        d = det("red", 0.30, 0.12, "triangle", 0.0)
        d.lying, d.ridge_yaw = True, 0.3
        c = self.pl._candidates(d, "left")
        self.assertTrue(c and all(x["det"].grasp_height_frac == fa["lying_height_frac"] for x in c))
        self.assertIsNone(planner("none")[0].cfg["grasp"]["face_align"]["triangle_height_frac"])

    def test_slots_reachable(self):
        """Drop slots with this grasp: 4 per (bin, arm) (of count 5: the outer second-row slot, y +-0.291 in the side
        bins / the far-side one in the centre bin, is out of reach; sim_legacy / base: 5), inside the bin, reachable."""
        self.pl.prepare()
        s = self.pl.cfg["grasp"]["slots"]
        for b, side in (("left", "left"), ("center", "left"), ("center", "right"), ("right", "right")):
            S = self.pl.slots(b, side)
            self.assertGreaterEqual(len(S), 4, (b, side))
            lo, hi = SC.bin_inner_box(b)
            m = SC.pelvis_to_mat(np.c_[S, np.zeros(len(S))])[:, :2]
            w = s["wall_clear"] - 1e-9
            self.assertTrue(np.all(m >= lo[:2] + w) and np.all(m <= hi[:2] - w), S)
            self.assertTrue(all(self.pl._slot_reach(side, p) for p in S))


class TestTriangleWait(unittest.TestCase):
    """profiles.sim grasp.face_align.triangle_wait. Seed 1009 of the train evaluations (m25o / m27o, 2026-11-14): the
    red triangle's face-aligned left-arm grasp came 17 mm (model) into the blue cube 9 cm to its left, the stock
    forward hint did not, and that unaligned pinch missed 4 times in a row (the triangle stayed on the counter)."""

    TRI, CUBE = (0.321, 0.078), (0.304, 0.169)

    @classmethod
    def setUpClass(cls):
        cls.pl, cls.post, cls.home = planner("sim")

    def _plan(self, pl, shape="triangle", nb=True, **kw):
        d = det("red", *self.TRI, shape, 0.0)
        return pl.plan(d, self.home, self.post["open"], bin_name="left", relocate=False,
                       neighbours=[det("blue", *self.CUBE, "cube", 0.0)] if nb else [], **kw)

    def test_blocked_aligned_triangle_waits(self):
        self.assertTrue(self.pl.cfg["grasp"]["face_align"]["triangle_wait"])
        p = self._plan(self.pl)
        self.assertFalse(p.ok)
        self.assertTrue(p.blocked)
        self.assertIn("aligned triangle grasp blocked", p.reason)
        q = self._plan(self.pl, nb=False)                     # the neighbour gone: the aligned grasp
        self.assertTrue(q.ok)
        self.assertIsNotNone(q.face)
        r = self._plan(self.pl, least_bad=True)               # least-bad pass: planned as before
        self.assertTrue(r.ok)

    def test_off_and_other_shapes_unchanged(self):
        off = planner("sim", **{"grasp.face_align.triangle_wait": False})[0]
        p = self._plan(off)                                   # the old fallback: stock hint, unaligned
        self.assertTrue(p.ok)
        self.assertIsNone(p.face)
        self.assertIsNotNone(p.hint)
        for shape in ("cube", "cylinder"):
            a, b = self._plan(self.pl, shape), self._plan(off, shape)
            self.assertEqual((a.ok, a.hint, a.face), (b.ok, b.hint, b.face), shape)
        self.assertFalse(planner("sim_legacy")[0].cfg["grasp"]["face_align"]["triangle_wait"])
        self.assertFalse(planner("none")[0].cfg["grasp"]["face_align"]["triangle_wait"])

    def test_policy_takes_the_neighbour_first(self):
        tri, cube = det("red", *self.TRI, "triangle", 0.0), det("blue", *self.CUBE, "cube", 0.0)
        cfg = load_config(profile="sim", overrides={"two_arm.enabled": False, "two_arm.avoid_parked": False})
        pol = ClassicalPolicy(cfg, log=None)
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol.perception.detect = lambda rgb, depth: [tri, cube]
        kind, plan, _ = pol._look({"head_rgb": None, "head_depth": None, "t": 0.0})
        self.assertEqual(kind, "plan")
        self.assertEqual(plan.color, "blue")
        rt = next(o for o in pol.objs if o["color"] == "red")
        self.assertIsNotNone(rt["blocked"])
        self.assertIsNone(rt["failed_at"])


class _FakeSceneClient:
    """Privileged 'scene service' over the mock backend's object list (for OraclePolicy offline)."""

    def __init__(self, backend):
        self.b = backend

    def state(self):
        return {"objects": [{"name": f"obj_{i}_{o['shape']}", "active": True, "pos": [float(v) for v in o["pos"]],
                             "color": o["color"], "shape": o["shape"],
                             "quat_xyzw": [0, 0, np.sin(o.get("yaw", 0.0) / 2), np.cos(o.get("yaw", 0.0) / 2)]}
                            for i, o in enumerate(self.b.objects)]}


class TestOracleMock(unittest.TestCase):
    def test_oracle_episode_slots_from_bin_contents(self):
        """OraclePolicy on the mock world with stacking (seed 0: two red, two blue objects; pairs on): everything
        delivered below the rim, 0 clips; objects in one bin end >= 10 cm apart (slot farthest from what lies there,
        prefetched plans included)."""
        from aaltonomous.sim.oracle import ORACLE_OVERRIDES, OraclePolicy
        from .mock_world import make_mock_env
        env = make_mock_env(seed=0, cfg=load_config(overrides=ORACLE_OVERRIDES), stack_radius=0.045)  # oracle's grasp
        pol = OraclePolicy(_FakeSceneClient(env.backend), safety=env.filter.cfg, threaded=False, log=lambda *a: None)
        self.assertTrue(pol.cfg["grasp"]["slots"]["enabled"] and pol.cfg["grasp"]["face_align"]["enabled"])
        self.assertFalse(pol.relay.cfg["grasp"]["slots"]["enabled"])
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        while True:
            obs, info = env.step(pol.act(obs))
            if info["episode_over"]:
                break
        out = env.backend.outcomes(pol.mapping)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertEqual(out.count("correct"), len(out), (out, pol.events[-5:]))
        for b in SC.bin_names:
            P = np.array([o["pos"][:2] for o in env.backend.objects
                          if SC.bin_of(SC.pelvis_to_mat(o["pos"])) == b]).reshape(-1, 2)
            if len(P) > 1:
                D = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P))
                self.assertGreater(D.min(), 0.10, (b, P))


if __name__ == "__main__":
    unittest.main()
