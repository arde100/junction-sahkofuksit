"""An aborting hand must not open over a box (planner.abort, verify.abort_open_margin).

Grasp verification can be wrong on the real hand: a held object that reads "empty". The abort then moves the arm home
and opens the hand; opened over a box, the object would land in that box, and a wrong box costs a point. These mock
episodes make EVERY grasp read empty (the mock's fingers stop only 0.036 rad short on an object, far below the robot
profile's verify.empty_below) and require that nothing ends in a wrong box."""
import unittest

import numpy as np

from .config import load_config
from .kinematics import ArmKinematics
from .mock_world import REAL_COLORS, make_mock_env
from .policy import ClassicalPolicy

L, C, R = "left", "center", "right"


def episode(text, seed=1, **backend_kw):
    cfg = load_config(profile="none")
    env = make_mock_env(seed=seed, world="real", cfg=cfg, colors=REAL_COLORS, **backend_kw)
    env.fixed_instruction = text
    pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg, time_limit=120.0)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    while True:
        obs, info = env.step(pol.act(obs))
        if info["episode_over"]:
            break
    return pol, env, info


class TestAbortOpensOutsideBoxes(unittest.TestCase):
    def test_wrong_empty_verdicts_never_fill_a_wrong_box(self):
        cases = [("Red goes to the right bin, yellow to the left bin, blue to the center bin.",
                  {"red": R, "yellow": L, "blue": C}),
                 ("Put the purple objects in the left bin, the orange objects in the center bin and the yellow "
                  "objects in the right bin.", {"blue": L, "red": C, "yellow": R})]
        aborts = 0
        for text, internal in cases:
            with self.subTest(text=text):
                pol, env, info = episode(text, held_gap=0.036)            # every held object reads "empty"
                self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
                ev = [e[1] for e in pol.events]
                self.assertTrue(any("verify" in e and "empty" in e for e in ev), ev)
                aborts += sum(e.startswith("abort") and "rejected" not in e for e in ev)
                out = env.backend.outcomes(internal)
                self.assertNotIn("wrong", out, (out, ev))
        self.assertGreater(aborts, 0)                                      # (the rule was exercised)

    def test_hold_ticks(self):
        """_abort_hold_ticks: 0 for a move home that never crosses a box; for one that starts over a box, the hand
        stays closed exactly until the grasp point is verify.abort_open_margin outside it."""
        cfg = load_config(profile="none")
        env = make_mock_env(seed=1, world="real", cfg=cfg, colors=REAL_COLORS)
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg, time_limit=120.0)
        pl, kin = pol.planner, pol.planner.kin
        side, sl = "left", ArmKinematics.sl("left")
        home = np.asarray(pl.home, float)
        self.assertEqual(pl._abort_hold_ticks(side, home, np.repeat(home[sl][None], 5, 0)), 0)
        # a pose with the grasp point over the left box: IK to its centre, 12 cm above the rim
        bx = pl.boxes["left"]
        c = np.asarray(bx.c, float)[:2]
        tgt = np.r_[c, pl.scene.counter_z + bx.rim + 0.12]
        q = home.copy()
        for _ in range(60):                                               # plain damped least squares on the point
            p, J = kin.grasp_jacobian(side, q)[:2]
            q[sl] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), tgt - p)
        self.assertLess(np.linalg.norm(kin.grasp_pose(side, q)[0] - tgt), 0.01)
        a = np.linspace(q[sl], home[sl], 40)                              # joint-space move home, 40 ticks
        k = pl._abort_hold_ticks(side, q, a)
        self.assertGreater(k, 0)
        self.assertLess(k, 40)
        m = float(cfg["verify"]["abort_open_margin"])
        full = home.copy()
        for i, inside in ((k - 1, True), (k, False)):
            full[sl] = a[i]
            self.assertEqual(bool(pl.boxes.inside_any(kin.grasp_pose(side, full)[0][:2], m)), inside)
        old = cfg["verify"]["abort_open_margin"]
        try:
            cfg["verify"]["abort_open_margin"] = None                     # off
            self.assertEqual(pl._abort_hold_ticks(side, q, a), 0)
        finally:
            cfg["verify"]["abort_open_margin"] = old


if __name__ == "__main__":
    unittest.main()
