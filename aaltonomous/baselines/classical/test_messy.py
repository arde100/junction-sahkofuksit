"""Objects as they really lie (lying prisms / cylinders, touching clusters) and the stillness gate, offline (rendered
mock world). Run:
    python -m unittest aaltonomous/baselines/classical/test_messy.py
Accuracy reports: python -m aaltonomous.baselines.classical.eval_messy --mock 40 (or a frame directory).
"""
import os
import unittest

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .mock_world import REAL_COLORS, Renderer, make_mock_env
from .perception import Detection, Perception
from .planner import Planner
from .policy import ClassicalPolicy
from .still import StillGate

SC, CAL = Scene.load(), Calibration.load()
BASE = load_config(profile="none")
SIM = load_config(profile="sim", overrides={"perception.xy_offset": [0.0, 0.0],
                                            "perception.register.fallback_xy_offset": None})
H = SC.object_height()


def obj(color, shape, x, y, yaw=0.0, lying=False, ridge=0.0, size=None):
    s = size or SC.size
    z = SC.counter_z + (s / 3 if lying and shape == "triangle" else s / 2)
    return dict(color=color, shape=shape, yaw=yaw, size=s, height=s, lying=lying, ridge_yaw=ridge,
                pos=np.array([x, y, z]))


def detect(objs, cfg=BASE, seed=0, noise=1.5, holes=0.02, colors=REAL_COLORS):
    rng = np.random.default_rng(seed)
    rgb, depth = Renderer(SC, CAL, colors=colors).render(objs, rng, noise, holes)
    return Perception(cfg, SC, CAL).detect(rgb, depth)


def nearest(dets, o):
    d = min(dets, key=lambda d: np.linalg.norm(d.top[:2] - o["pos"][:2]))
    return d, float(np.linalg.norm(d.top[:2] - o["pos"][:2]))


def ridge_err(a, b):
    return abs((np.degrees(a - b) + 90) % 180 - 90)


class TestLying(unittest.TestCase):
    def test_lying_prism(self):
        """Prisms on a rectangular side at any ridge direction, alone, both profiles: shape triangle, lying, ridge
        direction within 4 deg, centre within 5 mm; never a cube."""
        rng = np.random.default_rng(1)
        n = 0
        for cfg in (BASE, SIM):
            for k in range(8):
                o = obj("red", "triangle", rng.uniform(0.27, 0.33), rng.uniform(-0.15, 0.15), lying=True,
                        ridge=rng.uniform(-np.pi / 2, np.pi / 2))
                d, e = nearest(detect([o], cfg, seed=k), o)
                self.assertNotEqual(d.shape, "cube", d.fit)
                if d.shape is None:
                    continue
                n += 1
                self.assertEqual((d.shape, d.lying), ("triangle", True), d.fit)
                self.assertLess(ridge_err(d.ridge_yaw, o["ridge_yaw"]), 4.0, d.fit)
                self.assertLess(e, 0.005, d.fit)
        self.assertGreaterEqual(n, 14)

    def test_upright_never_lying(self):
        """Upright cubes / cylinders / triangles, 2.5 mm depth noise, any yaw: never lying, never a ridge."""
        rng = np.random.default_rng(2)
        for k in range(10):
            objs = [obj(c, s, 0.27 + 0.03 * (k % 3), y, yaw=rng.uniform(-0.8, 0.8))
                    for c, s, y in (("red", "cube", 0.14), ("yellow", "triangle", 0.0), ("blue", "cylinder", -0.14))]
            for d in detect(objs, BASE, seed=k, noise=2.5):
                self.assertFalse(d.lying, (k, d, d.fit))

    def test_lying_cylinder_is_never_a_cube(self):
        rng = np.random.default_rng(3)
        for k in range(6):
            o = obj("yellow", "cylinder", 0.30, rng.uniform(-0.12, 0.12), lying=True,
                    ridge=rng.uniform(-np.pi / 2, np.pi / 2))
            d, e = nearest(detect([o], BASE, seed=k), o)
            self.assertIn(d.shape, (None, "cylinder"), d.fit)
            if d.shape == "cylinder":
                self.assertTrue(d.lying)
                self.assertLess(ridge_err(d.ridge_yaw, o["ridge_yaw"]), 6.0)


class TestTouching(unittest.TestCase):
    def test_same_colour_cluster_is_split(self):
        """Two to four touching objects of ONE colour (centres 1.05-1.2 sizes apart, any yaw): one detection each,
        centres within 6 mm."""
        rng = np.random.default_rng(4)
        s = SC.size
        for k in range(8):
            n = 2 + k % 3
            c0 = np.array([0.30, rng.uniform(-0.08, 0.08)])
            pts = [c0]
            while len(pts) < n:
                a, f = rng.uniform(-np.pi, np.pi), rng.uniform(1.12, 1.2)
                p = pts[int(rng.integers(len(pts)))] + f * s * np.array([np.cos(a), np.sin(a)])
                if all(np.linalg.norm(p - q) >= 1.12 * s for q in pts) and 0.26 <= p[0] <= 0.34:
                    pts.append(p)
            objs = [obj("yellow", ("cube", "cylinder")[i % 2], p[0], p[1], yaw=rng.uniform(-0.8, 0.8))
                    for i, p in enumerate(pts)]
            dets = detect(objs, BASE, seed=k)
            self.assertEqual(len(dets), n, (k, dets))
            for o in objs:
                d, e = nearest(dets, o)
                self.assertLess(e, 0.006, (k, o["shape"], dets))

    def test_cluster_with_a_lying_prism(self):
        s = SC.size
        objs = [obj("blue", "cube", 0.30, 0.0, yaw=0.3),
                obj("blue", "triangle", 0.30, -1.15 * s, lying=True, ridge=0.2)]
        dets = detect(objs, BASE)
        self.assertEqual(len(dets), 2, dets)
        d, e = nearest(dets, objs[1])
        self.assertLess(e, 0.008)
        self.assertNotEqual(d.shape, "cube")


class TestGrasp(unittest.TestCase):
    """A lying prism is pinched on its two flat ends: fingers along the ridge, low on it; base and sim profiles."""

    def _det(self, ridge):
        d = Detection("red", np.array([0.30, 0.12, SC.counter_z + H]), 500, 500, "depth", shape="triangle",
                      shape_conf=0.9)
        d.lying, d.ridge_yaw = True, ridge
        return d

    def test_finger_directions_and_height(self):
        for prof in ("none", "sim"):
            cfg = load_config(profile=prof)
            pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
            for ridge in (0.0, 0.5, -1.2, 1.5):
                d = self._det(ridge)
                dirs = pl.face_dirs(d, "left")
                self.assertTrue(dirs)
                self.assertTrue(all(ridge_err(a, ridge) < 1e-6 for a in dirs))
                C = pl._candidates(d, "left")
                self.assertEqual([c["kind"] for c in C], ["face"] * len(dirs))   # no stock hints (the slopes)
                self.assertEqual(C[0]["det"].grasp_height_frac, cfg["grasp"]["face_align"]["lying_height_frac"])
            p = pl.plan(self._det(0.1), pl.home, pl.post["open"], bin_name="left")
            self.assertTrue(p.ok, (prof, p.reason))
            self.assertEqual(p.safety_clips, 0)
            e = pl._finger_err(p, p.side, np.array([np.cos(0.1), np.sin(0.1), 0.0]))
            self.assertLess(min(e, 180 - e), 13)
            self.assertLess(p.waypoints["grasp"][2] - SC.counter_z, 0.5 * H)   # lower than an upright grasp


class TestOrder(unittest.TestCase):
    def test_outer_objects_of_a_cluster_first(self):
        pol = ClassicalPolicy(config=load_config(profile="none"), log=None)
        s = SC.size
        dets = [Detection("yellow", np.array([0.30, y, 0.05]), 400, 400, "depth") for y in (-1.1 * s, 0.0, 1.1 * s)]
        dets.append(Detection("yellow", np.array([0.30, 0.15, 0.05]), 400, 400, "depth"))
        c = [pol._crowded(d, dets) for d in dets]
        self.assertEqual(c, [1, 2, 1, 0])


class TestStillGate(unittest.TestCase):
    def _img(self, shift=0.0):
        rgb, _ = Renderer(SC, CAL).render([obj("red", "cube", 0.3, 0.05), obj("blue", "cylinder", 0.3, -0.1)])
        if shift:
            import cv2
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            rgb = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rgb

    def test_still_moving_timeout(self):
        c = BASE["still"]
        a = self._img()
        g = StillGate(c, CAL)
        self.assertIsNone(g.step({"head_rgb": a}, 0.0))
        self.assertIsNone(g.step({"head_rgb": a}, 0.1))
        r = g.step({"head_rgb": a}, 0.2)
        self.assertTrue(r["still"], r)
        self.assertLess(r["motion_px"], 0.05)
        m = g.measure(a, self._img(1.0))
        self.assertAlmostEqual(m["motion_px"], 1.0, delta=0.15)
        g.reset()                                  # always moving: look anyway after timeout_s, logged
        t, r, k = 0.0, None, 0
        while r is None:
            r = g.step({"head_rgb": self._img(0.8 * k)}, t)
            t, k = t + 0.1, k + 1
        self.assertTrue(r.get("timeout"), r)
        self.assertGreaterEqual(r["waited_s"], c["timeout_s"] - 0.11)
        self.assertIn("TIMEOUT", StillGate.text(r))

    def test_featureless_and_mask(self):
        g = StillGate(BASE["still"], CAL)
        flat = np.full((480, 640, 3), 128, np.uint8)
        self.assertIsNone(g.step({"head_rgb": flat}, 0.0))
        r = g.step({"head_rgb": flat}, 0.2)
        self.assertIsNone(r["still"])                     # not measurable: look now
        m = g.mask((480, 640), [(np.array([[0.35, 0.0, 0.10]]), np.array([0.04]))])
        self.assertGreater((m == 0).sum(), 2000)

    def test_policy_waits_for_the_gate(self):
        """Robot profile in the mock world: every look is preceded by a gate line; the first look is not earlier
        than motion.settle_s + still.dt_s after the arms stopped."""
        cfg = load_config(profile="none")
        env = make_mock_env(seed=0, cfg=cfg, colors=REAL_COLORS)
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        while not any(m.startswith("look") for _, m in pol.events):
            obs, info = env.step(pol.act(obs))
        ev = [m for _, m in pol.events]
        i = next(i for i, m in enumerate(ev) if m.startswith("look"))
        self.assertTrue(any(m.startswith("still gate: ") for m in ev[:i]), ev[:i + 1])
        t_home = next(t for t, m in pol.events if m.startswith("to home"))
        t_look = next(t for t, m in pol.events if m.startswith("look"))
        self.assertGreater(t_look - t_home, cfg["motion"]["settle_s"] + cfg["still"]["dt_s"])


class TestEpisode(unittest.TestCase):
    def test_lying_prism_under_a_shape_instruction(self):
        """'Orange prisms left, cubes right' with an orange prism lying on its side: it goes left or stays, never
        right; the cubes go right (robot profile, the mock world in the real camera's colours)."""
        cfg = load_config(profile="none")
        # held fingers stop as far short as the robot profile's verification expects of the real hand (its
        # verify.empty_below 0.12 > the mock's default 0.036: every grasp would read "empty")
        env = make_mock_env(seed=3, cfg=cfg, colors=REAL_COLORS, held_gap=0.3)
        env.fixed_instruction = "Put the orange prisms in the left box and the cubes in the right box."
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
        obs = env.reset()
        s = SC.size
        b = env.backend
        b.objects = [dict(obj("red", "triangle", 0.30, 0.12, lying=True, ridge=0.15), size=s, height=s),
                     dict(obj("yellow", "cube", 0.29, -0.02, yaw=0.2)),
                     dict(obj("red", "cube", 0.31, -0.14, yaw=-0.3))]
        obs = env._obs()
        pol.reset({"instruction": obs["instruction"]})
        while True:
            obs, info = env.step(pol.act(obs))
            if info["episode_over"]:
                break
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        where = [SC.bin_of(SC.pelvis_to_mat(o["pos"])) for o in b.objects]
        self.assertIn(where[0], (None, "left"), [m for _, m in pol.events])
        self.assertEqual(where[1:], ["right", "right"], [m for _, m in pol.events])


if __name__ == "__main__":
    unittest.main()
