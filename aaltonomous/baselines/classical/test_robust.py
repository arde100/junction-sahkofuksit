"""Failures inside the policy never reach the runner as errors or invalid targets (policy_server.GuardedPolicy), and a
look / plan that raises is retried instead of ending the episode (ClassicalPolicy._after_look).

    python -m unittest aaltonomous/baselines/classical/test_robust.py
"""
import time
import unittest

import numpy as np

from aaltonomous.interface.safety import JointSpec, SafetyConfig, SafetyFilter

from .config import load_config
from .mock_world import make_mock_env
from .policy import MAX_LOOK_ERRORS
from .policy_server import ClassicalServerPolicy, GuardedPolicy


class Scripted:
    """Inner policy: act() follows a script per call: 'ok' (a 2-tick chunk moving 0.01 rad per tick), 'raise',
    'nan' (a NaN target), 'shape' (wrong shape)."""
    name, action_horizon = "scripted", 2

    def __init__(self, script, reset_raises=False):
        self.script, self.i, self.reset_raises, self.q = list(script), 0, reset_raises, None

    def reset(self, instruction, task=None, seed=None):
        self.i, self.q = 0, None
        if self.reset_raises:
            raise RuntimeError("reset boom")

    def act(self, obs):
        s = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if self.q is None:
            self.q = np.asarray(obs["arm_q"], float).copy()
        if s == "raise":
            raise RuntimeError("act boom")
        if s == "nan":
            return {"arm_q_target": np.full(14, np.nan), "hand_q_target": np.zeros(14)}
        if s == "shape":
            return {"arm_q_target": np.zeros(13), "hand_q_target": np.zeros(14)}
        a = self.q + 0.01 * np.arange(1, 3)[:, None]
        self.q = a[-1].copy()
        return {"arm_q_target": a, "hand_q_target": np.repeat(np.asarray(obs["hand_q"], float)[None], 2, 0)}


def obs0():
    sp = JointSpec()
    return {"arm_q": np.array([0, .35, 0, -.5, 0, 0, 0, 0, -.35, 0, -.5, 0, 0, 0], np.float32),
            "hand_q": np.clip(np.zeros(14), sp.hand_lower, sp.hand_upper).astype(np.float32), "t": 0.0}


def ticks(a):
    A, H = np.atleast_2d(a["arm_q_target"]), np.atleast_2d(a["hand_q_target"])
    return [{"arm_q_target": x, "hand_q_target": H[min(i, len(H) - 1)]} for i, x in enumerate(A)]


class TestGuard(unittest.TestCase):
    def run_script(self, script, n=40, reset_raises=False):
        g = GuardedPolicy(Scripted(script, reset_raises), log=lambda m: None)
        o = obs0()
        g.reset("", task="sort_fixed", seed=0)
        f = SafetyFilter(JointSpec(), SafetyConfig(dt=1 / 30))
        f.reset(o["arm_q"], o["hand_q"], 0.0)
        outs, k = [], 0
        for _ in range(n):
            a = g.act(o)            # never raises here
            outs.append(a)
            for tk in ticks(a):
                k += 1
                self.assertFalse(f.filter(tk, k / 30).clipped, tk)
            if a.get("done"):
                break
        return g, outs

    def test_exception_and_invalid_actions_become_holds_of_the_last_target(self):
        g, outs = self.run_script(["ok", "ok", "raise", "nan", "shape", "ok", "ok"], n=7)
        last_ok = outs[1]["arm_q_target"][-1]
        for a in outs[2:5]:          # the three failures: holds at the last tick returned
            self.assertTrue(np.allclose(a["arm_q_target"], last_ok))
            self.assertFalse(a["done"])
        self.assertTrue(np.allclose(outs[5]["arm_q_target"][0], last_ok + 0.01))   # the inner policy goes on
        self.assertEqual(g.fails, 0)

    def test_failure_before_any_target_holds_the_measured_pose(self):
        g, outs = self.run_script(["raise", "ok"], n=2)
        self.assertTrue(np.allclose(outs[0]["arm_q_target"], obs0()["arm_q"]))

    def test_persistent_failure_ends_the_episode_with_done(self):
        g, outs = self.run_script(["ok", "raise"], n=100)
        self.assertTrue(outs[-1]["done"])
        self.assertEqual(len(outs), 1 + GuardedPolicy.DONE_AFTER)

    def test_reset_failure_holds_and_ends_the_episode(self):
        g, outs = self.run_script(["ok"], n=5, reset_raises=True)
        self.assertEqual(len(outs), 1)
        self.assertTrue(outs[0]["done"])
        self.assertTrue(np.allclose(outs[0]["arm_q_target"], obs0()["arm_q"]))

    def test_no_valid_pose_at_all_raises(self):
        g = GuardedPolicy(Scripted(["raise"]), log=lambda m: None)
        g.reset("")
        o = dict(obs0(), arm_q=np.full(14, np.nan, np.float32))
        with self.assertRaises(RuntimeError):   # the runner then holds its own last command (error reply)
            g.act(o)

    def test_attributes_forwarded(self):
        g = GuardedPolicy(Scripted(["ok"]), log=lambda m: None)
        self.assertEqual((g.name, g.action_horizon, g.script), ("scripted", 2, ["ok"]))


class TestInjectedFailures(unittest.TestCase):
    """The real server policy (sim profile, planning thread as in the image; the mock clock waits for each look so it
    cannot run ahead), mock world: an exception in perception or planning (look thread) is retried and then ends the
    episode cleanly; an exception in act()'s own code (grasp verification, while a plan executes) becomes a hold.
    Every returned target is valid and 0 safety clips."""

    def episode(self, patch, seed=1, max_calls=3000):
        sp = ClassicalServerPolicy(profile="sim", time_limit=120.0)
        self.assertTrue(sp.pol.threaded)
        pol = GuardedPolicy(sp, log=lambda m: None)
        env = make_mock_env(seed=seed, world="sim", cfg=load_config(profile="sim"))
        obs = env.reset()
        pol.reset(obs["instruction"], task="sort_fixed", seed=seed)
        patch(sp.pol)
        info = None
        for _ in range(max_calls):
            job = sp.pol._job
            if job is not None:
                t0 = time.monotonic()
                while job["result"] is None and time.monotonic() - t0 < 120:
                    time.sleep(0.002)
            a = pol.act(obs)
            for x in (a["arm_q_target"], a["hand_q_target"]):
                self.assertTrue(np.isfinite(x).all())
                self.assertEqual(np.atleast_2d(x).shape[1], 14)
            obs, info = env.step(a)
            if info["episode_over"]:
                break
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        return sp.pol, env, info

    @staticmethod
    def raise_first(obj, name, n):
        orig, calls = getattr(obj, name), [0]

        def f(*a, **kw):
            calls[0] += 1
            if calls[0] <= n:
                raise RuntimeError(f"injected {name} failure {calls[0]}")
            return orig(*a, **kw)
        setattr(obj, name, f)
        return calls

    def test_one_perception_failure_is_retried(self):
        calls = []
        pol, env, info = self.episode(lambda p: calls.append(self.raise_first(p.perception, "detect", 1)))
        errs = [m for _, m in pol.events if m.startswith("look/plan error")]
        self.assertEqual(len(errs), 1, errs)
        self.assertIn("looking again", errs[0])
        out = env.backend.outcomes(pol.mapping)
        self.assertGreaterEqual(out.count("correct"), len(out) - 1, out)   # the episode went on and sorted

    def test_persistent_planning_failure_ends_cleanly(self):
        pol, env, info = self.episode(lambda p: self.raise_first(p.planner, "plan", 10 ** 6))
        errs = [m for _, m in pol.events if m.startswith("look/plan error")]
        self.assertEqual(len(errs), MAX_LOOK_ERRORS, errs)
        self.assertIn("stopping", errs[-1])
        self.assertEqual(info["end_reason"], "done")

    def test_act_thread_failure_becomes_a_hold(self):
        # grasp verification runs in act() while a plan executes: make it raise on 3 calls (each one = a held call)
        calls = []
        pol, env, info = self.episode(lambda p: calls.append(self.raise_first(p, "_watch", 3)))
        self.assertGreater(calls[0][0], 3)         # it did raise 3 times (and ran normally after)
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        out = env.backend.outcomes(pol.mapping)
        self.assertGreaterEqual(out.count("correct"), len(out) - 1, out)


if __name__ == "__main__":
    unittest.main()
