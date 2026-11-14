"""Offline tests of the sim / real-robot detection (autoprofile.py): the decision rule, both signatures on recorded
sim and real snippets (testdata/autoprofile_snippets.json), the probe in mock worlds that imitate either world
(mock_world.SIM_WORLD / REAL_WORLD, with noise, contradictions, weak evidence), per-process memory, act() never
blocking while the policies are built, forced profiles, and a full episode in each mock world through the wrapper.
Run in the sim container:
    aalto.sh py -m unittest aaltonomous/baselines/classical/test_autoprofile.py -v
"""
import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.deploy.policy_server import Policy
from aaltonomous.interface.safety import JointSpec

from .autoprofile import (REAL, SIM, WEAK, AutoProfilePolicy, arm_stat, decide, hand_stat, margin, probe_posture,
                          vote)
from .config import load_config
from .kinematics import ArmKinematics
from .mock_world import REAL_WORLD, SIM_SAG_GAIN, make_mock_env

BASE = load_config(profile="none")
AC = BASE["autoprofile"]
SPEC = JointSpec()
SNIPS = json.loads((Path(__file__).parent / "testdata" / "autoprofile_snippets.json").read_text())


class StubPolicy(Policy):
    """Stands in for a ClassicalServerPolicy: holds the pose of its first observation; records what it saw."""

    def __init__(self, profile, delay=0.0):
        time.sleep(delay)
        self.profile, self.obs, self.resets = profile, [], []

    def reset(self, instruction, task=None, seed=None):
        self.resets.append(instruction)
        self.obs, self.home = [], None

    def act(self, obs):
        self.obs.append(obs)
        if self.home is None:
            self.home = (np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float))
        return {"arm_q_target": self.home[0], "hand_q_target": self.home[1], "done": False}


def stub_wrapper(**kw):
    return AutoProfilePolicy(lambda p: StubPolicy(p), log=None, background=False, **kw)


def run_probe(world, seed=0, pol=None, extra_ticks=6, **env_kw):
    """One episode start in a mock world through the wrapper until the decision + extra_ticks delegated ticks.
    Returns the wrapper, the env, the commanded (arm, hand) ticks of the probe and the last info."""
    env = make_mock_env(seed=seed, world=world, cfg=load_config(profile="sim"), **env_kw)
    pol = pol or stub_wrapper()
    obs = env.reset()
    pol.reset(obs["instruction"])
    probe, after, info = [], 0, {}
    for _ in range(200):
        a = pol.act(obs)
        ticks = list(zip(np.atleast_2d(a["arm_q_target"]), np.atleast_2d(a["hand_q_target"])))
        if pol.active is None:
            probe += ticks
        else:
            after += len(ticks)
        obs, info = env.step(a)
        if after >= extra_ticks:
            break
    return pol, env, probe, info


class TestDecision(unittest.TestCase):
    def test_rule_table(self):
        want = {(SIM, SIM): (SIM, "confident"), (REAL, REAL): (REAL, "confident"),
                (REAL, WEAK): (REAL, "likely"), (WEAK, REAL): (REAL, "likely"),
                (SIM, WEAK): (SIM, "likely"), (WEAK, SIM): (SIM, "likely"),
                (SIM, REAL): (SIM, "contradiction"), (REAL, SIM): (SIM, "contradiction"), (WEAK, WEAK): (SIM, "weak")}
        for votes, res in want.items():
            self.assertEqual(decide(*votes), res, votes)

    def test_votes_and_margins(self):
        lim = AC["hand"]
        self.assertEqual(vote(0.002, lim), SIM)
        self.assertEqual(vote(0.05, lim), REAL)
        self.assertEqual(vote(0.018, lim), WEAK)
        self.assertEqual(vote(float("nan"), lim), WEAK)
        self.assertAlmostEqual(margin(0.002, lim, SIM), lim["sim_max"] / 0.002)
        self.assertAlmostEqual(margin(0.05, lim, REAL), 0.05 / lim["real_min"])
        for k in ("hand", "arm"):   # a gap between the two limits
            self.assertLess(AC[k]["sim_max"], AC[k]["real_min"])

    def test_config(self):
        np.testing.assert_allclose(AC["sim_sag_gain"], SIM_SAG_GAIN)   # one model in the detector and the mock
        self.assertEqual(AC["profiles"], {"sim": "sim", "real": "none"})
        v = probe_posture(AC, SPEC)
        self.assertEqual(v.shape, (14,))
        np.testing.assert_allclose(v[[3, 4]], v[[5, 6]])      # index = middle: the wiring difference does not matter
        np.testing.assert_allclose(v[[10, 11]], v[[12, 13]])
        self.assertEqual((v[0], v[7]), (0.0, 0.0))
        with self.assertRaises(ValueError):
            probe_posture(dict(AC, hand_probe=[0.0, 0.0, 0.05, -0.2, -0.2, -0.2, -0.2]), SPEC)   # thumb_2 at the limit

    def test_hand_stat_masks_and_missing_data(self):
        cmd = np.zeros(14)
        cmd[2] = 0.3   # only thumb_2 is inside the soft limits
        h = cmd + 0.001
        h[3] = -0.05   # beyond the soft limit: the sim cannot reach it, not counted
        self.assertAlmostEqual(hand_stat(h[None], cmd[None], SPEC), 0.001)
        self.assertTrue(np.isnan(hand_stat(np.zeros((2, 14)), np.tile(cmd, (2, 1)), SPEC)))   # no hand state
        self.assertTrue(np.isnan(hand_stat(h[None], SPEC.hand_lower[None], SPEC)))             # nothing inside


class TestRecordedSnippets(unittest.TestCase):
    """The signatures on what the sim and the robot really did (fixture: 3 observations each, commands held)."""

    @classmethod
    def setUpClass(cls):
        cls.kin = ArmKinematics(BASE, SPEC)
        cls.gain = np.tile(AC["sim_sag_gain"], 2)

    def stats(self, s):
        n = len(s["arm_q"])
        hs = hand_stat(s["hand_q"], np.tile(s["hand_cmd"], (n, 1)), SPEC)
        as_ = arm_stat(s["arm_q"], np.tile(s["arm_cmd"], (n, 1)), s["hand_q"], self.kin, self.gain)
        return hs, as_

    def test_sim_snippets_vote_sim(self):
        self.assertGreaterEqual(len(SNIPS["sim"]), 4)
        for s in SNIPS["sim"]:
            hs, as_ = self.stats(s)
            self.assertLess(hs, AC["hand"]["sim_max"] / 3, s["src"])   # measured: <= 2.4 mrad (114 episode starts)
            self.assertLess(as_, AC["arm"]["sim_max"] / 10, s["src"])  # measured: <= 0.15 mrad
            self.assertEqual(decide(vote(hs, AC["hand"]), vote(as_, AC["arm"])), (SIM, "confident"))

    def test_real_snippets_vote_real(self):
        self.assertGreaterEqual(len(SNIPS["real_arm"]), 8)
        self.assertGreaterEqual(len(SNIPS["real_hand"]), 5)
        for s in SNIPS["real_arm"]:   # per teleop episode its lowest still window: >= 10 mrad
            self.assertGreater(self.stats(s)[1], AC["arm"]["real_min"] * 1.5, s["src"])
        for s in SNIPS["real_hand"]:  # >= 48 mrad
            self.assertGreater(self.stats(s)[0], AC["hand"]["real_min"] * 1.5, s["src"])


class TestProbe(unittest.TestCase):
    """The probe in the mock worlds (StubPolicy behind the wrapper: no ClassicalPolicy build)."""

    def check(self, world, how, pol, env, probe, info):
        r = pol.history[-1]
        self.assertEqual((r["world"], r["how"]), (world, how), r)
        self.assertEqual(pol.active, world)
        self.assertEqual(env.filter.clipped_steps, 0)
        return r

    def test_sim_world(self):
        for seed, noise in ((0, None), (1, (0.0003, 0.001)), (2, (0.0003, 0.002))):
            r = self.check(SIM, "confident", *run_probe("sim", seed, joint_noise=noise))
            self.assertGreater(min(r["margins"]), 2.0, r)

    def test_real_world(self):
        for seed in (0, 1, 2):   # REAL_WORLD has measurement noise
            r = self.check(REAL, "confident", *run_probe("real", seed))
            self.assertGreater(min(r["margins"]), 1.5, r)

    def test_real_world_variants(self):
        # arm stuck by friction (never moves): measured - commanded = 0, the sim model predicts its sag (~12 mrad at
        # the home pose) -> real. (Why the probe holds the measured pose and does not pre-compensate the sim's sag:
        # then a stuck arm would match the sim's model exactly.)
        r = self.check(REAL, "confident", *run_probe("real", 0, sag_gain=None, arm_deadband=0.1))
        self.assertGreater(r["arm"], 1.5 * AC["arm"]["real_min"])
        self.assertLess(r["arm"], 0.02)
        # fingers without the thumb_0 bias, the smallest no-load error everywhere (data: 9-36 mrad medians)
        self.check(REAL, "confident", *run_probe("real", 1, hand_bias=None, hand_deadband=0.03))

    def test_contradiction_goes_to_sim(self):
        # sim world, one finger stuck (touches something): hand real, arm sim
        db = np.zeros(14)
        db[4] = 0.15
        r = self.check(SIM, "contradiction", *run_probe("sim", 0, hand_deadband=db))
        self.assertEqual(r["votes"], (REAL, SIM))
        # a robot whose arm matched the sim's model to the mrad (not seen in the data): the rule still says sim
        r = self.check(SIM, "contradiction", *run_probe("real", 0, sag_gain=True, arm_deadband=None))
        self.assertEqual(r["votes"], (REAL, SIM))

    def test_weak_evidence(self):
        # fingers 15-20 mrad short (between the limits) + real arm -> real (likely)
        r = self.check(REAL, "likely", *run_probe("real", 0, hand_bias=None, hand_deadband=0.017))
        self.assertEqual(r["votes"], (WEAK, REAL))
        # sim fingers + an arm 1.35x the sim's sag (residual ~4 mrad) -> sim (likely)
        r = self.check(SIM, "likely", *run_probe("sim", 0, sag_gain=1.35 * np.asarray(SIM_SAG_GAIN)))
        self.assertEqual(r["votes"], (SIM, WEAK))

    def test_probe_motion_and_handover(self):
        pol, env, probe, info = run_probe("real", 0)
        r = pol.history[-1]
        arm = np.array([a for a, _ in probe])
        hand = np.array([h for _, h in probe])
        np.testing.assert_allclose(arm, np.repeat(arm[:1], len(arm), 0))   # arms held where first measured
        dh = np.abs(np.diff(hand, axis=0)).max()
        self.assertLessEqual(dh * 30, AC["hand_vel"] + 1e-6)              # <= 1.2 rad/s
        np.testing.assert_allclose(hand[-1], probe_posture(AC, SPEC))
        self.assertLess(r["t_decide"], 0.75)
        self.assertGreaterEqual(r["n"], AC["min_samples"])
        stub = pol.pols[REAL]
        np.testing.assert_allclose(stub.obs[0]["hand_q"], hand[-1], atol=1e-6)   # delegate starts from the last command
        self.assertEqual(env.filter.clipped_steps, 0)                    # the real fingers are far from it: no jump


class TestProcess(unittest.TestCase):
    @staticmethod
    def res(world, how):
        return {"world": world, "how": how, "hand": 0.0, "arm": 0.0, "votes": (WEAK, WEAK), "margins": (0, 0),
                "n": 2, "t": (0.5, 0.6), "t_decide": 0.6, "ramp_ticks": 5}

    def test_memory(self):
        pol, log = stub_wrapper(), []
        pol.log = log.append
        seq = [(SIM, "contradiction", SIM), (REAL, "likely", REAL), (SIM, "contradiction", REAL),
               (SIM, "likely", REAL), (SIM, "confident", SIM), (REAL, "likely", SIM), (REAL, "confident", REAL)]
        for world, how, want in seq:
            pol.reset("x")
            pol._decide(self.res(world, how))
            self.assertEqual(pol.world, want, (world, how))
        self.assertEqual(len(log), len(seq))
        self.assertIn("SWITCHED from sim", log[1])
        self.assertIn("kept", log[2])

    def test_once_per_process(self):
        pol = stub_wrapper()
        pol.ac = dict(pol.ac, every_episode=False)
        run_probe("sim", 0, pol=pol)
        self.assertEqual(pol.world, SIM)
        pol.reset("again")
        self.assertIsNone(pol.probe)
        self.assertEqual(pol.active, SIM)

    def test_episodes_through_the_wrapper(self):
        """Two episodes: each probes, the delegate is reset every episode and starts from the probe's command."""
        pol = stub_wrapper()
        for seed in (0, 1):
            pol, env, probe, info = run_probe("sim", seed, pol=pol)
        self.assertEqual(len(pol.history), 2)
        self.assertEqual(len(pol.pols[SIM].resets), 2)   # reset() per episode (built before the first one)
        self.assertEqual(len(pol.pols[REAL].resets), 2)  # the other one too (the next episode may switch)


class TestNonBlocking(unittest.TestCase):
    def test_act_while_building(self):
        """Policies take 1.5 s to build: act() answers at once (probe, then holds), delegates once built."""
        pol = AutoProfilePolicy(lambda p: StubPolicy(p, delay=0.75), log=None, background=True)
        env = make_mock_env(seed=0, world="sim", cfg=load_config(profile="sim"))
        obs = env.reset()
        pol.reset(obs["instruction"])
        slow, holds, t0 = [], 0, time.monotonic()
        while time.monotonic() - t0 < 10.0:
            c0 = time.perf_counter()
            a = pol.act(obs)
            slow.append(time.perf_counter() - c0)
            built = SIM in pol.pols
            if pol.active is not None and not built:
                holds += 1
                time.sleep(0.02)
            obs, info = env.step(a)
            if built and pol.pols[SIM].obs:
                break
        self.assertTrue(pol.pols[SIM].obs, "never delegated")
        self.assertGreater(holds, 0)            # the decision came before the build finished
        self.assertLess(max(slow), 0.05)
        self.assertEqual(env.filter.clipped_steps, 0)
        self.assertTrue(pol.wait_ready(10.0))

    def test_act_while_building_classical(self):
        """The real ClassicalServerPolicy builds (CPU-bound, GIL) while the probe runs: act() latency stays small.
        (The server itself waits for the build before it listens; this is the margin if it did not.)"""
        from .policy_server import ClassicalServerPolicy
        pol = AutoProfilePolicy(lambda p: ClassicalServerPolicy(profile=p), log=None, background=True)
        env = make_mock_env(seed=0, world="real", cfg=BASE)
        obs = env.reset()
        pol.reset(obs["instruction"])
        lat = []
        while not pol.ready.is_set() and len(lat) < 2000:
            c0 = time.perf_counter()
            a = pol.act(obs) if pol.active is None or REAL not in pol.pols else pol.probe.hold(2)
            lat.append(time.perf_counter() - c0)
            obs, info = env.step(a)
            time.sleep(0.01)
        self.assertTrue(pol.wait_ready(120.0))
        print(f"\n[test_autoprofile] act() during the build: {len(lat)} calls, max {1e3 * max(lat):.0f} ms, "
              f"p99 {1e3 * np.percentile(lat, 99):.0f} ms")
        self.assertLess(max(lat), 0.5)
        self.assertEqual(env.filter.clipped_steps, 0)

    def test_failed_build_falls_back(self):
        def factory(p):
            if p == "sim":
                raise RuntimeError("bad config")
            return StubPolicy(p)
        pol = AutoProfilePolicy(factory, log=None, background=False)
        self.assertFalse(pol.wait_ready(1.0))
        pol, env, probe, info = run_probe("sim", 0, pol=pol)
        self.assertEqual(pol.active, SIM)
        self.assertTrue(pol.pols[REAL].obs)     # served by the policy that exists


class TestForcedProfiles(unittest.TestCase):
    def test_make_policy(self):
        from . import policy_server as ps
        built = []

        class Fake:
            def __init__(self, profile=None):
                built.append(profile)
                self.profile = profile
        with mock.patch.object(ps, "ClassicalServerPolicy", Fake):
            for env, kind in (("sim", Fake), ("none", Fake), ("sim_legacy", Fake), ("auto", AutoProfilePolicy)):
                with mock.patch.dict(os.environ, {"AALTO_PROFILE": env}):
                    p = ps.make_policy()
                    self.assertIsInstance(p, kind, env)
                    if kind is AutoProfilePolicy:
                        self.assertTrue(p.wait_ready(30.0))
                        self.assertEqual({w: q.profile for w, q in p.pols.items()}, {"sim": "sim", "real": "none"})
                    else:
                        self.assertIsNone(p.profile)   # forced: $AALTO_PROFILE inside load_config, no detection
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AALTO_PROFILE", None)
                self.assertIsInstance(ps.make_policy(), Fake)   # unset: the file's default, no detection

    def test_load_config_auto(self):
        with mock.patch.dict(os.environ, {"AALTO_PROFILE": "auto"}):
            self.assertEqual(load_config()["profile"], "sim")            # the file's default
            self.assertEqual(load_config(profile="none")["profile"], "none")

    def test_server_policy_profiles(self):
        from .policy_server import ClassicalServerPolicy
        with mock.patch.dict(os.environ, {"AALTO_PROFILE": "sim_legacy"}):
            self.assertEqual(ClassicalServerPolicy(profile="none").pol.cfg["profile"], "none")
            self.assertEqual(ClassicalServerPolicy().pol.cfg["profile"], "sim_legacy")


def unthreaded(p):
    """Plan in act() (the mock's virtual clock would otherwise run ahead of the planning thread)."""
    p.pol.threaded = False
    return p


class TestEpisodes(unittest.TestCase):
    """A full episode in each mock world through the wrapper with the real ClassicalServerPolicy per world."""

    @classmethod
    def setUpClass(cls):
        from .policy_server import ClassicalServerPolicy
        cls.pol = AutoProfilePolicy(lambda p: unthreaded(ClassicalServerPolicy(profile=p, time_limit=120.0)),
                                    log=None, background=True)
        cls.pol.wait_ready(120.0)

    def episode(self, world, seed):
        cfg = load_config(profile=self.pol.profiles[world])   # the mock grasps with that world's postures
        env = make_mock_env(seed=seed, world=world, cfg=cfg)
        obs = env.reset()
        self.pol.reset(obs["instruction"])
        while True:
            obs, info = env.step(self.pol.act(obs))
            if info["episode_over"]:
                break
        return env, info

    def test_sim_world_episode(self):
        env, info = self.episode(SIM, 1)
        self.assertEqual(self.pol.history[-1]["world"], SIM)
        self.check(env, info, SIM)

    def test_real_world_episode(self):
        env, info = self.episode(REAL, 1)
        self.assertEqual(self.pol.history[-1]["world"], REAL)
        self.check(env, info, REAL)

    def check(self, env, info, world):
        self.assertEqual(info["end_reason"], "done", info)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        out = env.backend.outcomes(self.pol.pols[world].pol.mapping)
        self.assertNotIn("wrong", out)
        self.assertGreaterEqual(out.count("correct"), len(out) - 1, out)


if __name__ == "__main__":
    unittest.main()
