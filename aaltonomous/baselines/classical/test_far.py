"""Far release (far.py, grasp.far) offline: planner level and mock episodes (mock_world ballistic=True). Run:
    python -m unittest aaltonomous/baselines/classical/test_far.py
    python -m aaltonomous.baselines.classical.far map          # the reach map (report)
"""
import os
import unittest

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.scene.scene import Scene

from .bins import Box, Boxes
from .config import hand_postures, home_arm, load_config
from .far import G, flight
from .mock_world import REAL_COLORS, box_row, make_mock_env
from .perception import Detection
from .planner import Planner
from .policy import ClassicalPolicy

SC = Scene.load()
BASE = load_config(profile="none")                 # the robot profile: far on
SIZE, GAP, WALL = 0.21, 0.02, 0.068                # venue: 21 cm boxes, 1-3 cm apart, ~6.8 cm walls
TABLE = [1.4, 1.2]                                 # mat axes: a table larger than the mat (boxes up to x ~0.9)
SIDES = (("left", "left"), ("center", "left"), ("center", "right"), ("right", "right"))


def as_boxes(rows, src="detected", conf=1.0):
    return Boxes({b["name"]: Box(b["name"], tuple(b["c"]), b["yaw"], b["size"] / 2, b["size"] / 2, b["wall_h"],
                                 b["wall_t"], b["floor_t"], conf, True, src) for b in rows}, source=src)


def det(x, y):
    return Detection("red", np.array([x, y, SC.counter_z + SC.object_height()]), 400, 400, "depth")


class TestFlight(unittest.TestCase):
    def test_ballistics(self):
        p, T = flight([0.5, 0.1, 0.3], [0.2, 0.0, 0.0], 0.05)
        self.assertAlmostEqual(T, np.sqrt(2 * 0.25 / G), places=9)
        np.testing.assert_allclose(p, [0.5 + 0.2 * T, 0.1, 0.05])
        p, T = flight([0.5, 0.1, 0.3], [0.1, 0.0, 1.0], 0.1)        # thrown up: up and down again, then 0.2 m
        self.assertAlmostEqual(T, (1.0 + np.sqrt(1.0 + 2 * G * 0.2)) / G, places=9)
        np.testing.assert_allclose(p, [0.5 + 0.1 * T, 0.1, 0.1])


class TestFarPlans(unittest.TestCase):
    """Planner level, robot profile, detected venue-like boxes."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.post, cls.home = BASE, hand_postures(BASE), home_arm(BASE)
        cls.pl = Planner(BASE, SC, cls.post, cls.home)

    def plan(self, rows, box, side, src="detected", conf=1.0, used=()):
        pl = self.pl
        pl.set_boxes(as_boxes(rows, src, conf))
        pl.drop_used = {box: [np.asarray(u, float) for u in used]}
        y = {"left": 0.17, "right": -0.17}.get(box, 0.07 if side == "left" else -0.07)
        return pl.plan(det(0.30, y), self.home, self.post["open"], bin_name=box, relocate=False)

    def check_release(self, p, rows, box, marginal=False):
        """The plan's release: far, 0 clips, landing inside the TRUE box with the configured margins."""
        self.assertTrue(p.ok, p.reason)
        self.assertEqual(p.safety_clips, 0)
        self.assertIsNotNone(p.far)
        f, c = p.far, self.pl.far.c
        b = as_boxes(rows)[box]
        under = b.local(np.asarray(f["release"])[:2])            # straight below the grasp point at the release
        lo, hi = c["release_shift"]
        r = c["object_size"] / 2
        hu, hv = b.inner
        near = -hu + r + c["near_clear"] - (c["max_short"] if marginal else 0.0)
        self.assertGreaterEqual(under[0] + lo, near - 1e-9)                               # clears the near rim
        self.assertEqual(f["marginal"], marginal)
        self.assertLessEqual(under[0] + hi + r + c["land_clear"] + c["far_clear"] + c["pos_err"], hu + 1e-9)
        self.assertLessEqual(abs(under[1]) + c["release_side"] + r + c["side_clear"] + c["pos_err"], hv + 1e-9)
        np.testing.assert_allclose(f["under_uv"], under, atol=1e-9)
        # bookkeeping: the expected landing (mean shift ahead of the release point) is the plan's drop point
        np.testing.assert_allclose(b.local(p.waypoints["drop"][:2]), [under[0] + 0.5 * (lo + hi), under[1]], atol=1e-9)
        self.assertLessEqual(f["tilt_deg"], c["max_tilt_deg"] + 0.5)
        ph = {n: (s, e) for n, s, e in p.phases}
        k = ph["open"][0]
        sl = slice(0, 7) if p.side == "left" else slice(7, 14)
        g = self.pl.kin.grasp_pose(p.side, p.arm[k])[0]
        np.testing.assert_allclose(g, f["release"], atol=2e-3)
        # the hand stays closed while the arm settles there, then opens to `release`
        np.testing.assert_allclose(p.hand[k + f["k_open"] - 1, sl], self.post["closed"][sl])
        np.testing.assert_allclose(p.hand[ph["open"][1] - 1, sl], self.post["release"][sl])
        self.assertGreaterEqual(self.pl.far.bulk_low(p.side, p.arm[ph["carry"][0]:]), c["wrist_clear"] - 1e-9)

    def test_published_row_unchanged(self):
        """Near edge 0.40 (scene.yaml, or detected there): drop slots as before, no far release."""
        rows = box_row(0.40, 0, 0.06, 0.05, size=SIZE, wall_t=0.003)
        for box, side in SIDES:
            with self.subTest(box=box, side=side):
                p = self.plan(rows, box, side)
                self.assertTrue(p.ok, p.reason)
                self.assertIsNone(p.far)
        self.pl.set_boxes(Boxes.from_scene(SC))
        self.assertFalse(self.pl.far.applies("left", "left")[0])

    def test_venue_rows(self):
        """Near edge 0.45 / 0.47 (the venue's 0.50 - 3 to 5 cm), 6.8 cm walls, 2 cm gaps: every box / arm is
        delivered (a stock slot or the far release); every far release lands inside its box. (What reaches how far
        depends on the configured grasp: `python -m ...far map`.)"""
        n_far = 0
        for front in (0.45, 0.47):
            rows = box_row(front, 0, WALL, GAP, size=SIZE, wall_t=0.004)
            for box, side in SIDES:
                with self.subTest(front=front, box=box, side=side):
                    p = self.plan(rows, box, side)
                    self.assertTrue(p.ok, p.reason)
                    if p.far is not None:
                        n_far += 1
                        self.check_release(p, rows, box)
                        self.assertGreater(p.far["margin"], 0.0)
                        self.assertIn("FAR release", p.summary())
        self.assertGreater(n_far, 0)

    def test_yawed_row(self):
        """The organisers' near / turned row (0.46, yawed 7 deg, 10 cm walls): every box is delivered (stock slots or
        far release), every far landing inside its yawed box."""
        rows = box_row(0.46, 7, 0.10, GAP, size=SIZE, wall_t=0.004)
        for box, side in SIDES:
            with self.subTest(box=box, side=side):
                p = self.plan(rows, box, side)
                self.assertTrue(p.ok, p.reason)
                if p.far is not None:
                    self.check_release(p, rows, box)

    def test_marginal_attempted(self):
        """Near edge 0.53: the far release is short of the near rule by less than far.max_short: attempted anyway
        (MARGINAL: a miss lands on the counter in front, 0 points), never at the cost of the far wall / the sides."""
        rows = box_row(0.53, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        for box, side in SIDES:
            with self.subTest(box=box, side=side):
                p = self.plan(rows, box, side)
                self.check_release(p, rows, box, marginal=True)
                self.assertLess(p.far["margin"], 0.0)
                self.assertIn("MARGINAL", p.summary())
                self.assertEqual(self.pl.box_grade(box), "marginal")

    def test_too_far_stays(self):
        """Near edge 0.56: out of the static reach: no plan (the object stays on the counter), and why."""
        rows = box_row(0.56, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        for box, side in SIDES:
            with self.subTest(box=box, side=side):
                p = self.plan(rows, box, side)
                self.assertFalse(p.ok)
                self.assertIn(f"{box} box out of reach (", p.reason)
                self.assertIn("far: short by", p.reason)
        rows = box_row(0.62, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        self.assertIn("far.max_front", self.plan(rows, "left", "left").reason)

    def test_only_detected_confident_boxes(self):
        """A box that is not detected (scene prior) or detected with low confidence gets no far release."""
        rows = box_row(0.47, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        self.assertIsNotNone(self.plan(rows, "center", "left").far)
        p = self.plan(rows, "center", "left", src="scene")
        self.assertIsNone(p.far)
        self.assertFalse(p.ok)
        p = self.plan(rows, "center", "left", conf=0.2)
        self.assertIsNone(p.far)
        self.assertFalse(p.ok)

    def test_sim_profile_off(self):
        for prof in ("sim", "sim_legacy"):
            pl = Planner(load_config(profile=prof), SC, self.post, self.home)
            pl.set_boxes(as_boxes(box_row(0.50, 0, WALL, GAP, size=SIZE, wall_t=0.004)))
            self.assertEqual(pl.far.applies("left", "left"), (False, "off"))

    def test_second_object_lands_apart(self):
        """The next object into the same box: a valid release that keeps at least as far from the one delivered as
        the first release would (spacing is a preference: release_side keeps the releases central)."""
        rows = box_row(0.47, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        p1 = self.plan(rows, "center", "left")
        self.check_release(p1, rows, "center")
        used = p1.waypoints["drop"][:2]
        p2 = self.plan(rows, "center", "left", used=[used])
        self.check_release(p2, rows, "center")
        self.assertGreaterEqual(np.linalg.norm(p2.waypoints["drop"][:2] - used), -1e-9)
        env = self.pl.far.envelope("center", "left")
        self.assertGreater(len(env), 1)


class TestMockBallistics(unittest.TestCase):
    def test_moving_hand_throws(self):
        """ballistic=True: an object released from a moving hand lands where the grasp point's velocity takes it;
        one that comes down on a wall stays on the rim (not in the box)."""
        rows = box_row(0.50, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        env = make_mock_env(seed=0, cfg=BASE, boxes=rows, counter_size=TABLE, ballistic=True)
        b = env.backend
        b.objects = [{"color": "red", "shape": "cube", "yaw": 0.0, "size": SC.size, "height": SC.object_height(),
                      "pos": np.array([0.45, 0.0, SC.counter_z + 0.25])}]
        ob = b.objects[0]
        b._fly(ob, np.array([0.1, 0.0, 0.0]))
        T = np.sqrt(2 * (0.25 - b.objects[0]["height"] / 2) / G)
        self.assertAlmostEqual(ob["pos"][0], 0.45 + 0.1 * T, places=9)
        b._drop(ob)
        self.assertAlmostEqual(ob["pos"][2], SC.counter_z + ob["height"] / 2)      # in front of the box: counter
        ob["pos"] = np.array([0.502, 0.0, SC.counter_z + 0.25])                    # over the near wall: on the rim
        b._drop(ob)
        self.assertAlmostEqual(ob["pos"][2], SC.counter_z + WALL + ob["height"] / 2)
        self.assertEqual(b.outcomes({"red": "center"}), ["counter"])


class TestMissesSeen(unittest.TestCase):
    def test_short_miss_retried_rim_left_alone(self):
        """A far release that comes up short: an object on the counter just in front of the near wall is a detection
        (picked again), one lying on the rim is not (left alone), one in the box neither."""
        from aaltonomous.interface.calibration import Calibration
        from .mock_world import Renderer
        from .perception import Perception
        rows = box_row(0.50, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        c = rows[1]["c"]                                         # the centre box
        h = 0.048

        def ob(x, y, z, col):
            return dict(color=col, shape="cube", yaw=0.1, size=h, height=h, pos=np.array([x, y, z]))
        front = ob(c[0] - SIZE / 2 - 0.02 - h / 2, c[1] + 0.03, SC.counter_z + h / 2, "yellow")
        rim = ob(c[0] - SIZE / 2 + 0.002, c[1] - 0.04, SC.counter_z + WALL + h / 2, "blue")
        inside = ob(c[0] + 0.02, c[1], SC.counter_z + 0.004 + h / 2, "red")
        band = ob(0.30, 0.15, SC.counter_z + h / 2, "red")
        rng = np.random.default_rng(3)
        rgb, depth = Renderer(SC, Calibration.load(), colors=REAL_COLORS, boxes=rows, counter_size=TABLE).render(
            [front, rim, inside, band], rng, 2.0, 0.03)
        p = Perception(BASE, SC, Calibration.load())
        dets = p.detect(rgb, depth)
        self.assertTrue(p.boxes.ok("center"), p.boxes.text())
        for o, seen in ((front, True), (rim, False), (inside, False), (band, True)):
            hit = any(np.linalg.norm(d.top[:2] - o["pos"][:2]) < 0.02 for d in dets)
            self.assertEqual(hit, seen, (o["color"], o["pos"], dets))


def run_episode(seed, rows, n_max=None, overrides=None, **env_kw):
    cfg = load_config(profile="none", overrides=overrides) if overrides else BASE
    # the real camera's colours (the robot profile's classes); held fingers stop 0.3 rad short of the cage's closed
    # posture (grasp verification: finger_shortfall, as test_sag / test_autoprofile)
    env = make_mock_env(seed=seed, cfg=cfg, boxes=rows, counter_size=TABLE, ballistic=True, colors=REAL_COLORS,
                        held_gap=0.3, **env_kw)
    pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
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


class TestFarEpisodes(unittest.TestCase):
    """Robot profile end to end in the mock world (box detection on, ballistic drops, the robot's joint signatures)."""

    def test_venue_row(self):
        """Near edge 0.47, 6.8 cm walls, 2 cm gaps: the far release puts every object into its box, 0 clips; a
        delivered object is not picked again."""
        rows = box_row(0.47, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        pol, env, info = run_episode(0, rows, n_max=3)
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        out = env.backend.outcomes(pol.mapping)
        self.assertEqual(out, ["correct"] * len(out), [m for _, m in pol.events])
        ex = [m for _, m in pol.events if m.startswith("execute")]
        self.assertEqual(len(ex), len(out), ex)
        self.assertTrue(any("FAR release" in m for m in ex), ex)

    def test_beyond_reach_nothing_released(self):
        """Near edge 0.56: nothing is released (every object stays on the counter, none off the table)."""
        rows = box_row(0.56, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        pol, env, info = run_episode(1, rows, n_max=2)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertEqual(env.backend.grasp_log, [])
        self.assertEqual(set(env.backend.outcomes(pol.mapping)), {"counter"})
        ev = [m for _, m in pol.events]
        self.assertTrue(any("box out of reach (left: far: short by" in m for m in ev), ev)
        self.assertTrue(any("whose box is out of reach" in m for m in ev), ev)
        # the gate is cheap: after the first look (reach envelopes computed once per box pose) no look plans
        self.assertLess(max(pol.look_s[1:] or [0.0]), 1.0, pol.look_s)

    def test_look_budget(self):
        """policy.plan_budget_s (the per-look planning budget; look_budget_s is its alias) far below what a far
        plan needs: looks stop at it ('plan budget exceeded'), the next look resumes the search; everything still
        ends in its box."""
        rows = box_row(0.47, 0, WALL, GAP, size=SIZE, wall_t=0.004)
        pol, env, info = run_episode(2, rows, n_max=3, overrides={"policy.plan_budget_s": 0.2})
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        out = env.backend.outcomes(pol.mapping)
        self.assertEqual(out, ["correct"] * len(out), [m for _, m in pol.events])
        self.assertTrue(any("plan budget exceeded" in m for _, m in pol.events), [m for _, m in pol.events])


if __name__ == "__main__":
    unittest.main()
