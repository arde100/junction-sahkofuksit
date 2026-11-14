"""Planning time (Q1): the planner's fast paths give the same results, the per-look budget (policy.plan_budget_s)
bounds a look, undeliverable boxes are dropped at once, every look logs a `plan:` line.
    python -m unittest aaltonomous/baselines/classical/test_plan_time.py -v
"""
import os
import re
import time
import unittest

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .mock_world import REAL_COLORS, box_row, make_mock_env
from .perception import Detection
from .planner import Planner
from .policy import ClassicalPolicy
from .test_bins import TABLE
from .test_classical import NB

SC = Scene.load()
CZ, H = SC.counter_z, SC.object_height()


def det(color, x, y, shape=None, yaw=None):
    d = Detection(color, np.array([x, y, CZ + H]), 900, 900, "depth", yaw=yaw)
    d.shape, d.shape_conf = shape, 1.0
    return d


def planner(profile="none", **ov):
    cfg = load_config(profile=profile, overrides=ov)
    return Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))


def run_episode(seed, rows, overrides=None, n_max=None, **env_kw):
    """Robot profile in the mock world (as test_far: ballistic drops, the real camera's colours); the policy's log
    (the per-look `plan:` lines) is kept in pol.logged."""
    cfg = load_config(profile="none", overrides=overrides)
    env = make_mock_env(seed=seed, cfg=cfg, boxes=rows, counter_size=TABLE, ballistic=True, colors=REAL_COLORS,
                        **env_kw)
    logged = []
    pol = ClassicalPolicy(config=cfg, log=logged.append, safety=env.filter.cfg)
    pol.logged = logged
    obs = env.reset()
    if n_max is not None:
        env.backend.objects = env.backend.objects[:n_max]
        obs = env._obs()
    pol.reset({"instruction": obs["instruction"]})
    while True:
        obs, info = env.step(pol.act(obs))
        if info["episode_over"]:
            break
    return pol, env, info


def plan_lines(pol):
    return [m.split("] ", 1)[1] for m in pol.logged if "] plan: " in m]


def wall(line):
    return float(re.match(r"plan: ([0-9.]+) s wall", line).group(1))


class TestSolve(unittest.TestCase):
    """Planner.solve = kin.solve, faster: the arm-length bound and the memo never change a result."""

    @classmethod
    def setUpClass(cls):
        cls.pl = planner()

    def test_reach_bound_is_an_upper_bound(self):
        pl, rng = self.pl, np.random.default_rng(0)
        for side in ("left", "right"):
            o, L = pl.reach_bound(side)
            sl = pl.kin.sl(side)
            far = 0.0
            for _ in range(3000):
                q = np.zeros(14)
                q[sl] = rng.uniform(pl.spec.arm_lower[sl], pl.spec.arm_upper[sl])
                far = max(far, np.linalg.norm(pl.kin.grasp_pose(side, q)[0] - o))
            self.assertLessEqual(far, L + 1e-9)
            self.assertGreater(far, 0.8 * L)            # (not a loose bound)
            self.assertAlmostEqual(L, 0.5526, places=3)  # G1 arm: links 0.411 m + wrist -> grasp point 0.142 m

    def test_same_result_as_kin_solve(self):
        pl, home = self.pl, self.pl.home
        for side, t in (("left", [0.30, 0.15, CZ + 0.03]), ("left", [0.45, 0.0, CZ + 0.16]),
                        ("right", [0.30, 0.20, CZ + 0.03]), ("left", [0.75, 0.3, CZ + 0.10])):
            with self.subTest(side=side, t=t):
                a = pl.solve(side, t, home)
                b = pl.kin.solve(side, t, home)
                self.assertEqual(a[1], b[1])
                if b[1] or np.linalg.norm(np.subtract(t, pl.reach_bound(side)[0])) <= pl.reach_bound(side)[1]:
                    np.testing.assert_allclose(a[0], b[0])
                    self.assertAlmostEqual(a[2], b[2])
                c = pl.solve(side, t, home)              # memo: equal, and a copy
                np.testing.assert_allclose(a[0], c[0])
                c[0][:] = 0
                np.testing.assert_allclose(pl.solve(side, t, home)[0], a[0])

    def test_out_of_reach_is_instant(self):
        pl = self.pl
        t0 = time.perf_counter()
        q, ok, err, _ = pl.solve("left", [0.62, 0.30, CZ + 0.16], pl.home)
        self.assertFalse(ok)
        self.assertGreater(err, 0.0)
        self.assertLess(time.perf_counter() - t0, 0.05)


class TestLeastBadPruning(unittest.TestCase):
    """A least-bad candidate whose grasp pose already has less clearance than the best so far is not planned in
    full: the chosen plan is the same as with every candidate planned."""

    def test_same_plan(self):
        cases = [(det("yellow", 0.30, 0.10, "cube", 0.0), [det("blue", 0.30, 0.0, "cylinder")]),
                 (det("red", 0.28, -0.12, "cylinder"), [det("blue", 0.33, -0.08, "cube", 0.3),
                                                       det("yellow", 0.25, -0.17, "triangle", 0.2)]),
                 (det("blue", 0.30, 0.02, "triangle", 0.4), [det("red", 0.30, 0.07, "cube"),
                                                            det("yellow", 0.30, -0.04, "cylinder")])]
        for prof, ov in (("sim_legacy", {**NB, "neighbours.margin": 0.08, "neighbours.floor": -0.03,
                                         "neighbours.least_bad_good": 1.0}),
                         ("none", {"neighbours.margin": 0.08, "neighbours.floor": -0.03,
                                   "neighbours.least_bad_good": 1.0})):
            a, b = planner(prof, **ov), planner(prof, **ov)
            b.lb_prune = False
            pruned, orig = [], a._plan_side

            def spy(*x, **kw):
                p = orig(*x, **kw)
                pruned.append(p.reason.startswith("least-bad:"))
                return p
            a._plan_side = spy
            for k, (tgt, nb) in enumerate(cases):
                with self.subTest(profile=prof, case=k):
                    p = a.plan(tgt, a.home, a.post["open"], neighbours=nb, least_bad=True)
                    q = b.plan(tgt, b.home, b.post["open"], neighbours=nb, least_bad=True)
                    self.assertEqual((p.ok, p.side), (q.ok, q.side), (p.reason, q.reason))
                    if p.ok:
                        np.testing.assert_allclose(p.arm, q.arm)
                        self.assertAlmostEqual(p.nb_clear, q.nb_clear)
            self.assertTrue(any(pruned), prof)       # (the pruning did happen)


class TestBudget(unittest.TestCase):
    def test_profiles(self):
        self.assertEqual(load_config(profile="none")["policy"]["plan_budget_s"], 2.0)
        self.assertIsNone(load_config(profile="sim")["policy"]["plan_budget_s"])
        self.assertIsNone(load_config(profile="sim_legacy")["policy"]["plan_budget_s"])

    def test_planner_stops_at_the_deadline(self):
        pl = planner()
        pl.deadline = time.perf_counter() - 1.0
        p = pl.plan(det("red", 0.30, 0.15), pl.home, pl.post["open"])
        self.assertTrue(p.budget and not p.ok, p.reason)
        self.assertEqual(pl.kin.hint.tolist(), np.asarray(pl.cfg["grasp"]["finger_hint"], float).tolist())
        pl.deadline = None
        self.assertTrue(pl.plan(det("red", 0.30, 0.15), pl.home, pl.post["open"]).ok)

    def test_tiny_budget_episode(self):
        """A budget too small for any plan: every look stops at it, logs the objects, looks again; each object is
        skipped after policy.budget_max (10) interruptions; the episode ends cleanly, no clip, nothing moved."""
        rows = box_row(0.40, 0, 0.06, 0.05, size=0.21)
        ov = {"policy.plan_budget_s": 0.02, "policy.budget_max": 10}
        pol, env, info = run_episode(0, rows, overrides=ov, n_max=3)
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        ev = [m for _, m in pol.events]
        looks = [m for m in ev if m.startswith("look: ")]
        self.assertEqual(len(plan_lines(pol)), len(looks))
        self.assertEqual(len(pol.look_s), len(looks))
        self.assertFalse(any(m.startswith("plan: ") for m in ev))   # (a log line, not an event)
        self.assertTrue(any("plan budget exceeded: planned again next look" in m for m in ev), ev)
        self.assertTrue(any("plan budget exceeded 10 times: skipped until it moves" in m for m in ev), ev)
        self.assertTrue(any("budget 0.0 s hit" in m for m in plan_lines(pol)))
        self.assertEqual(env.backend.grasp_log, [])

    def test_robot_profile_episode(self):
        """Published boxes, robot profile, 2 s budget: every look within the budget plus about one IK (the loaded
        test machine: 4.5 s); a clean episode."""
        rows = box_row(0.40, 0, 0.06, 0.05, size=0.21)
        pol, env, info = run_episode(1, rows, n_max=3)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        self.assertTrue(plan_lines(pol))
        for m in plan_lines(pol):
            self.assertLess(wall(m), 4.5, m)

    def test_same_plans_within_the_budget(self):
        """A deadline that is not reached (and the failed-candidate memo it switches on) changes no plan: the same
        result as without one, also when planned again (memo hits)."""
        a, b = planner(), planner()
        nb = [det("blue", 0.30, 0.0, "cylinder"), det("yellow", 0.25, -0.12, "cube")]
        for tgt, bn in ((det("red", 0.31, -0.16, "cube"), "left"), (det("yellow", 0.29, 0.09, "cylinder"), "center"),
                        (det("blue", 0.28, 0.17, "cube"), "right")):
            with self.subTest(tgt=tgt.color, bin=bn):
                q = b.plan(tgt, b.home, b.post["open"], bin_name=bn, neighbours=nb)
                for _ in range(2):
                    a.deadline = time.perf_counter() + 600.0
                    p = a.plan(tgt, a.home, a.post["open"], bin_name=bn, neighbours=nb)
                    a.deadline = None
                    self.assertEqual((p.ok, p.side, p.target_bin), (q.ok, q.side, q.target_bin), (p.reason, q.reason))
                    if p.ok:
                        np.testing.assert_allclose(p.arm, q.arm)
        self.assertTrue(a.fail_memo)

    def test_unreachable_boxes_at_once(self):
        """grasp.far off, boxes 0.50 m out (no drop slot in reach): one look, one line per box, no plan, well under a
        second of planning."""
        rows = box_row(0.50, 0, 0.06, 0.05, size=0.21)
        pol, env, info = run_episode(0, rows, n_max=4, overrides={"grasp.far.enabled": False})
        ev = [m for _, m in pol.events]
        self.assertEqual(len(plan_lines(pol)), 1, ev)
        boxes = [m for m in ev if "box: neither arm reaches it" in m]
        self.assertGreaterEqual(len(boxes), 2, ev)
        self.assertEqual(len(set(m.split()[0] for m in boxes)), len(boxes))
        self.assertEqual(pol.plans, [])
        line = plan_lines(pol)[0]
        perc = float(re.search(r"perception ([0-9.]+)", line).group(1))
        self.assertLess(wall(line) - perc, 0.3, line)
        self.assertEqual(env.backend.grasp_log, [])

    def test_far_decides_with_far_on(self):
        """grasp.far on (base): the far module's per-box decision is the only one (one line per box, never the reach
        test's as well); boxes beyond the far envelope too (0.56): nothing released."""
        rows = box_row(0.56, 0, 0.068, 0.02, size=0.215, wall_t=0.004)
        pol, env, info = run_episode(1, rows, n_max=2)
        ev = [m for _, m in pol.events]
        self.assertFalse(any("neither arm reaches it" in m for m in ev), ev)
        far = [m for m in ev if " box out of reach (" in m]
        self.assertTrue(far, ev)
        first = ev.index(next(m for m in ev if m.startswith("look: ")))
        nxt = next((i for i, m in enumerate(ev) if i > first and m.startswith("look: ")), len(ev))
        lines = [m for m in ev[first:nxt] if " box out of reach (" in m]
        self.assertEqual(len(lines), len({m.split()[0] for m in lines}), lines)   # one line per box
        self.assertEqual(env.backend.grasp_log, [])

    def test_bound_holds_on_the_far_envelope(self):
        """Every far release pose of the reach envelope (box at 0.50 / 0.47, both arms) has its grasp point inside
        Planner.reach_bound, with room: the bound never removes a far delivery."""
        from .bins import Box, Boxes
        pl, n, room = planner(), 0, np.inf
        for front in (0.47, 0.50):
            rows = box_row(front, 0, 0.068, 0.02, size=0.215, wall_t=0.004)
            pl.set_boxes(Boxes({b["name"]: Box(b["name"], tuple(b["c"]), b["yaw"], b["size"] / 2, b["size"] / 2,
                                               b["wall_h"], b["wall_t"], b["floor_t"], 1.0, True, "detected")
                                for b in rows}))
            for name in ("left", "center", "right"):
                for side in ("left", "right"):
                    o, L = pl.reach_bound(side)
                    for r in pl.far.envelope(name, side):
                        d = np.linalg.norm(pl.kin.grasp_pose(side, np.asarray(r["q"], float))[0] - o)
                        self.assertLess(d, L - 0.005, (front, name, side, r["v"], r["z"]))
                        n, room = n + 1, min(room, L - d)
        self.assertGreater(n, 20)
        print(f"far envelope: {n} release poses, grasp point >= {1000 * room:.0f} mm inside the arm-length bound")

if __name__ == "__main__":
    unittest.main()
