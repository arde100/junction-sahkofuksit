"""Offline tests of the classical baseline (no robot). Run:
    ~/miniforge3/envs/aalto/bin/python -m unittest aaltonomous/baselines/classical/test_classical.py -v
"""
import os
import unittest

import numpy as np

# Mechanism tests run on profile sim_legacy (config.yaml: profile sim with the grasp and neighbour values their mock
# seeds and expectations were written for), so tuning profiles.sim does not move them; the sim profile's own grasp is
# tested in TestSimProfile (explicit profile="sim"). Set before anything loads a config ($AALTO_PROFILE wins if set).
os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config, mirror_left_to_right
from .kinematics import ArmKinematics
from .mock_world import Renderer, make_mock_env
from .perception import Perception
from .planner import Planner
from .policy import ClassicalPolicy

SC, CAL, CFG = Scene.load(), Calibration.load(), load_config()
SINGLE = {"two_arm.enabled": False}  # tests whose counts assume one grasp per executed plan (one arm per cycle)
# drop slots, face-aligned grasps, neighbour-aware plans off: mock seeds chosen for the plans before them (their
# own tests: test_grasp_plan.py); with them on the same seeds pair / abort differently
NEW_OFF = {"grasp.slots.enabled": False, "grasp.face_align.enabled": False, "neighbours.enabled": False,
           "neighbours.order_free": 0.0}
# neighbour rule as the neighbour tests expect it (pinned here, not taken from a profile that is being tuned)
NB = {"neighbours.enabled": True, "neighbours.margin": 0.005, "neighbours.floor": -0.012, "neighbours.least_bad": True,
      "neighbours.least_bad_good": 0.0, "neighbours.extra_dirs": True, "neighbours.extra_within": 0.12,
      "neighbours.order_free": 0.07}


def scene_objects(rng, n=3, shapes=("cube", "cylinder")):
    L = SC.cfg["layout"]
    objs, used = [], []
    colors = list(rng.permutation(["red", "yellow", "blue"]))[:n]
    while len(objs) < n:
        x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
        if any(np.hypot(x - a, y - b) < 0.09 for a, b in used):
            continue
        used.append((x, y))
        h = SC.object_height()
        objs.append(dict(color=colors[len(objs)], shape=shapes[len(objs) % len(shapes)], yaw=rng.uniform(-0.8, 0.8),
                         size=SC.size, height=h, pos=SC.mat_to_pelvis([x, y, h / 2])))
    return objs


class TestPerception(unittest.TestCase):
    # the sim profile's xy_offset corrects the Isaac head camera; the mock renders through calibration.yaml exactly
    CFG0 = load_config(overrides={"perception.xy_offset": [0.0, 0.0], "perception.register.fallback_xy_offset": None})

    def setUp(self):
        self.r = Renderer(SC, CAL)
        self.p = Perception(self.CFG0, SC, CAL)

    def _check(self, rgb, depth, objs, tol):
        dets = self.p.detect(rgb, depth)
        self.assertEqual(sorted(d.color for d in dets), sorted(o["color"] for o in objs))
        for o in objs:
            d = next(d for d in dets if d.color == o["color"])
            top = o["pos"] + [0, 0, o["height"] / 2]
            err = np.linalg.norm(d.top - top)
            self.assertLess(err, tol, f"{o['color']} {o['shape']} err {err * 1000:.1f} mm ({d.method})")
        return dets

    def test_top_centre_within_1cm_with_noise_and_holes(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            objs = scene_objects(rng)
            rgb, depth = self.r.render(objs, rng, depth_noise_mm=1.5, hole_frac=0.15)
            dets = self._check(rgb, depth, objs, 0.01)
            self.assertTrue(all(d.method == "depth" for d in dets))

    def test_no_depth_falls_back_to_plane(self):
        rng = np.random.default_rng(1)
        objs = scene_objects(rng)
        rgb, _ = self.r.render(objs)
        dets = self._check(rgb, np.zeros(rgb.shape[:2], np.uint16), objs, 0.025)
        self.assertTrue(all(d.method == "plane" for d in dets))

    def test_bins_are_not_objects(self):
        rgb, depth = self.r.render([])
        self.assertEqual(self.p.detect(rgb, depth), [])

    # ---------------------------------------------------------------- shape + yaw (perception._fit_shape)
    @staticmethod
    def _objs(rng, shapes, n=3):
        L, h = SC.cfg["layout"], SC.object_height()
        objs, used = [], []
        while len(objs) < n:
            x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
            if any(np.hypot(x - a, y - b) < 0.09 for a, b in used):
                continue
            used.append((x, y))
            objs.append(dict(color=("red", "yellow", "blue")[len(objs)], shape=shapes[len(objs) % len(shapes)],
                             yaw=rng.uniform(-np.pi, np.pi), size=SC.size, height=h,
                             pos=SC.mat_to_pelvis([x, y, h / 2])))
        return objs

    def _check_shapes(self, objs, dets, yaw_tol_deg=3.0, xy_tol=0.003):
        found = 0
        for o in objs:
            d = next(d for d in dets if d.color == o["color"])
            if d.shape is None:                      # allowed (the planner falls back), a wrong shape is not
                continue
            found += 1
            self.assertEqual(d.shape, o["shape"], f"{o['color']} {d.fit}")
            if o["shape"] == "cylinder":
                self.assertIsNone(d.yaw)
            else:                                    # oracle convention: cube mod 90, triangle (apex) mod 120
                per = 90.0 if o["shape"] == "cube" else 120.0
                err = (np.degrees(d.yaw - o["yaw"]) + per / 2) % per - per / 2
                self.assertLess(abs(err), yaw_tol_deg, f"{o['shape']} yaw {np.degrees(o['yaw']):.0f}: err {err:.1f}")
            top = o["pos"] + [0, 0, o["height"] / 2]
            self.assertLess(np.linalg.norm(d.top[:2] - top[:2]), xy_tol, o["shape"])   # shape.centre: fitted centre
        return found

    def test_shape_and_yaw_with_noise_and_holes(self):
        """Rendered cubes, cylinders and triangular prisms at any yaw, 1.5 mm depth noise + 10 % holes."""
        rng = np.random.default_rng(5)
        n = found = 0
        for _ in range(6):
            objs = self._objs(rng, ("cube", "triangle", "cylinder"))
            rgb, depth = self.r.render(objs, rng, depth_noise_mm=1.5, hole_frac=0.1)
            found += self._check_shapes(objs, self.p.detect(rgb, depth))
            n += len(objs)
        self.assertGreaterEqual(found, 0.9 * n)

    def test_triangle_apex_direction(self):
        """The same triangle turned by 60 deg (apex where a face was) must give a yaw 60 deg apart (mod 120)."""
        rng = np.random.default_rng(6)
        o = self._objs(rng, ("triangle",), n=1)[0]
        yaws = []
        for dy in (0.0, np.pi / 3):
            rgb, depth = self.r.render([dict(o, yaw=0.3 + dy)])
            d = self.p.detect(rgb, depth)[0]
            self.assertEqual(d.shape, "triangle")
            yaws.append(np.degrees(d.yaw))
        self.assertAlmostEqual(abs(((yaws[1] - yaws[0]) + 60) % 120 - 60), 60.0, delta=2.0)

    def test_occluded_object_never_gets_a_wrong_shape(self):
        """A dark finger hovering 5 mm over the far half of the top face (sim home pose): the occluded directions
        are censored, so the shape is right (or None) and the fitted centre ignores the cut."""
        rng = np.random.default_rng(7)
        n = found = 0
        for shape in ("cube", "triangle", "cylinder") * 3:
            o = self._objs(rng, (shape,), n=1)[0]
            top = o["pos"] + [0, 0, o["height"] / 2]
            finger = dict(color=(30, 30, 30), shape="cube", yaw=rng.uniform(-0.5, 0.5), size=0.03, height=0.012,
                          pos=top + [0.018 + rng.uniform(0, 0.01), rng.uniform(-0.01, 0.01), 0.011])
            rgb, depth = self.r.render([o, finger], rng, depth_noise_mm=1.0, hole_frac=0.02)
            dets = self.p.detect(rgb, depth)
            self.assertEqual([d.color for d in dets], [o["color"]])
            found += self._check_shapes([o], dets, yaw_tol_deg=5.0, xy_tol=0.004)
            n += 1
        self.assertGreaterEqual(found, 2 * n // 3)                # (without the censoring: 4 of these 9)

    def test_recorded_sim_frames(self):
        """Recorded Isaac frames with ground truth (capture_frames.py; skipped when absent): no wrong shape, yaw in
        the oracle's convention. Full evaluation: python -m aaltonomous.baselines.classical.eval_perception <dir>."""
        import json
        from pathlib import Path
        files = sorted(Path(os.environ.get("AALTO_FRAMES", "/data/aaro/frames/nominal")).glob("*.npz"))[:15]
        if not files:
            self.skipTest("no recorded frames")
        n = found = 0
        for f in files:
            z = np.load(f)
            objs = json.loads(str(z["objects"]))
            for d in self.p.detect(z["rgb"], z["depth"]):
                if d.shape is None:
                    continue
                o = min((o for o in objs if o["color"] == d.color),
                        key=lambda o: np.linalg.norm(d.top[:2] - o["pos"][:2]))
                self.assertEqual(d.shape, o["shape"], f"{f.name} {o['name']}")
                found += 1
                if d.yaw is not None:
                    per = 90.0 if d.shape == "cube" else 120.0
                    err = (np.degrees(d.yaw - o["yaw_pelvis_rad"]) + per / 2) % per - per / 2
                    self.assertLess(abs(err), 5.0, f"{f.name} {o['name']}")
            n += len(objs)
        self.assertGreaterEqual(found, 0.8 * n)

    # ---------------------------------------------------------------- geometry first + self-levelling
    def test_geometry_first_is_on(self):
        self.assertTrue(CFG["perception"]["geometry_first"])
        self.assertTrue(CFG["perception"]["level"]["enabled"])

    @staticmethod
    def _hsv_shift(rgb, dh=0, gain=1.0):
        """Whole image: hue + dh (OpenCV units), then brightness x gain (clipped: washed-out tops)."""
        import cv2
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(int)
        hsv[..., 0] = (hsv[..., 0] + dh) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(float) * gain
        return np.clip(out, 0, 255).astype(np.uint8)

    def test_colour_survives_hue_and_brightness_shift(self):
        rng = np.random.default_rng(11)
        for dh, gain in ((-6, 1.0), (6, 1.0), (0, 0.55), (0, 1.5), (5, 1.4), (-5, 0.6)):
            objs = scene_objects(rng, shapes=("cube", "triangle", "cylinder"))
            rgb, depth = self.r.render(objs, rng, depth_noise_mm=1.0, hole_frac=0.02)
            self._check(self._hsv_shift(rgb, dh, gain), depth, objs, 0.01)

    def _one(self, rng, color, shape="cube"):
        o = self._objs(rng, (shape,), n=1)[0]
        o["color"] = color
        return o

    def test_unmatched_colour_is_unknown(self):
        rng = np.random.default_rng(12)
        o = self._one(rng, (40, 170, 60))                       # green: no class within max_dist
        rgb, depth = self.r.render([o], rng, depth_noise_mm=1.0)
        dets = self.p.detect(rgb, depth)
        self.assertEqual([d.color for d in dets], ["unknown"])
        self.assertLess(np.linalg.norm(dets[0].top[:2] - o["pos"][:2]), 0.004)

    def test_bin_orange_is_not_red(self):
        """An object-sized blob in the bins' orange on the counter is not red; a red one of the red class is."""
        rng = np.random.default_rng(13)
        o = self._one(rng, tuple(SC.cfg["bins"]["rgb"]))
        dets = self.p.detect(*self.r.render([o], rng, depth_noise_mm=1.0))
        self.assertEqual([d.color for d in dets], ["unknown"])
        o["color"] = (215, 45, 25)                              # hue ~3 deg: red
        dets = self.p.detect(*self.r.render([o], rng, depth_noise_mm=1.0))
        self.assertEqual([d.color for d in dets], ["red"])

    def test_touching_objects_of_two_colours_are_split(self):
        """A failed grasp pushed one object against another: one depth blob, two colours -> two detections."""
        rng = np.random.default_rng(14)
        for k in range(4):                                      # side by side (pelvis y), one behind the other (x)
            a = self._one(rng, "red", ("cube", "cylinder")[k // 2])
            a["yaw"] = 0.0
            off = [0.0, SC.size, 0.0] if k % 2 == 0 else [SC.size, 0.0, 0.0]
            b = dict(a, color="blue", shape="cube", pos=a["pos"] + off)
            rgb, depth = self.r.render([a, b], rng, depth_noise_mm=1.0, hole_frac=0.02)
            dets, _ = self.p.detect(rgb, depth, debug=True)
            self.assertTrue(all(d.n_votes > 0 for d in dets))   # geometry path
            self._check(rgb, depth, [a, b], 0.012)

    def _tilted(self, pitch_deg, roll_deg, dt, ov=None):
        from scipy.spatial.transform import Rotation
        p = Perception(load_config(overrides={"perception.xy_offset": [0.0, 0.0],
                                              "perception.register.fallback_xy_offset": None, **(ov or {})}), SC, CAL)
        T = p.T0.copy()
        T[:3, :3] = Rotation.from_euler("xy", [roll_deg, pitch_deg], degrees=True).as_matrix() @ T[:3, :3]
        T[:3, 3] += dt
        p.T0 = T
        return p

    def test_self_levelling_removes_tilt_and_height_error(self):
        """Calibration wrong by 1.5 deg pitch, 1 deg roll, 1 cm height (camera rotated about its centre): geometry
        first + levelling vs the colour-first code without levelling (unlevelled, the tilted counter would rise into
        geometry first's height band)."""
        rng = np.random.default_rng(15)
        errs = {}
        for lv in (False, True):
            p = self._tilted(1.5, 1.0, [0.0, 0.0, 0.01], None if lv else
                             {"perception.level.enabled": False, "perception.geometry_first": False,
                              "perception.register.enabled": False})
            e = []
            for _ in range(3):
                objs = scene_objects(rng, shapes=("cube", "cylinder", "triangle"))
                rgb, depth = self.r.render(objs, rng, depth_noise_mm=1.0, hole_frac=0.02)
                dets = p.detect(rgb, depth)
                self.assertEqual(sorted(d.color for d in dets), sorted(o["color"] for o in objs))
                self.assertEqual(p.path, "geometry" if lv else "colour")
                for o in objs:
                    d = next(d for d in dets if d.color == o["color"])
                    e.append(np.linalg.norm(d.top - (o["pos"] + [0, 0, o["height"] / 2])))
            errs[lv] = max(e)
            if lv:
                L = p.level
                self.assertTrue(L["used"], L)
                self.assertAlmostEqual(L["pitch_deg"], -1.5, delta=0.1)
                self.assertAlmostEqual(L["roll_deg"], -1.0, delta=0.1)
        self.assertGreater(errs[False], 0.012)
        self.assertLess(errs[True], 0.004)
        self.assertLess(errs[True], 0.25 * errs[False])

    def test_levelling_guard(self):
        """An implausible correction (8 deg) is not applied: calibration as is, and the reason is reported."""
        p = self._tilted(8.0, 0.0, [0.0, 0.0, 0.0])
        rgb, depth = self.r.render(scene_objects(np.random.default_rng(16)))
        p.detect(rgb, depth)
        self.assertFalse(p.level["used"])
        self.assertTrue(p.level["why"])
        self.assertEqual(p.path, "colour (not levelled)")     # geometry first needs a level counter
        np.testing.assert_allclose(p.T, p.T0)

    # ---------------------------------------------------------------- bin registration (profile sim)
    def _wrong_pose(self, p, dx, dy, yaw_deg, pitch_deg=0.8, dz=0.005):
        """Calibration that perception uses wrong by (dx, dy, dz) and yaw / pitch about the camera centre."""
        from scipy.spatial.transform import Rotation
        T = p.T0.copy()
        T[:3, :3] = Rotation.from_euler("zy", [yaw_deg, pitch_deg], degrees=True).as_matrix() @ T[:3, :3]
        T[:3, 3] += [dx, dy, dz]
        p.T0 = T

    def _pose_left(self, p):
        from .eval_perception import pose_error
        return pose_error(p.T, CAL._T("head"))

    def test_registration_recovers_xy_and_yaw(self):
        rng = np.random.default_rng(18)
        for dx, dy, yaw in ((0.012, -0.008, 1.2), (-0.015, 0.01, -1.8), (0.0, 0.02, 0.5)):
            p = Perception(self.CFG0, SC, CAL)
            self.assertTrue(p.rg["enabled"])
            self._wrong_pose(p, dx, dy, yaw)
            objs = scene_objects(rng, shapes=("cube", "cylinder", "triangle"))
            rgb, depth = self.r.render(objs, rng, depth_noise_mm=1.0, hole_frac=0.02)
            dets = p.detect(rgb, depth)
            self.assertTrue(p.reg["used"], p.reg)
            e = self._pose_left(p)
            self.assertLess(abs(e["yaw"]), 0.05, p.reg)
            self.assertLess(np.hypot(*e["dt"][:2]), 1.5, p.reg)
            for o in objs:
                d = next(d for d in dets if d.color == o["color"])
                self.assertLess(np.linalg.norm(d.top[:2] - o["pos"][:2]), 0.003)

    def test_registration_ignores_objects_in_and_on_bins(self):
        """A sorted object on a bin floor and one lying across a rim are outliers, not walls."""
        rng = np.random.default_rng(19)
        b, h = SC.cfg["bins"], SC.object_height()
        cx, cy = b["centers"]["center"]
        lx, ly = b["centers"]["right"]
        extra = [dict(color="yellow", shape="cube", yaw=0.3, size=SC.size, height=h,
                      pos=SC.mat_to_pelvis([cx + 0.03, cy - 0.02, b["floor_thickness"] + h / 2])),
                 dict(color="blue", shape="cube", yaw=0.0, size=SC.size, height=h,      # across the near wall
                      pos=SC.mat_to_pelvis([lx, ly - b["outer_size"][1] / 2, b["wall_height"] + h / 2]))]
        p = Perception(self.CFG0, SC, CAL)
        self._wrong_pose(p, -0.01, 0.012, -1.0)
        rgb, depth = self.r.render(scene_objects(rng) + extra, rng, depth_noise_mm=1.0, hole_frac=0.02)
        p.detect(rgb, depth)
        self.assertTrue(p.reg["used"], p.reg)
        e = self._pose_left(p)
        self.assertLess(abs(e["yaw"]), 0.05, p.reg)
        self.assertLess(np.hypot(*e["dt"][:2]), 1.5, p.reg)

    def test_fallback_offset_when_registration_is_refused(self):
        """xy_offset belongs to the registered pose: a refused registration uses register.fallback_xy_offset."""
        p = Perception(load_config(overrides={"perception.register.enabled": True, "perception.xy_offset": [0.01, 0.0],
                                              "perception.register.fallback_xy_offset": [0.0, 0.02]}), SC, CAL)
        rgb, depth = self.r.render(scene_objects(np.random.default_rng(21)))
        p.detect(rgb, depth)
        self.assertTrue(p.reg["used"], p.reg)
        np.testing.assert_allclose(p.xy_offset, [0.01, 0.0])
        p.detect(rgb, None)                       # no depth: no registration
        self.assertFalse(p.reg["used"])
        np.testing.assert_allclose(p.xy_offset, [0.0, 0.02])

    def test_registration_off_is_levelling_only(self):
        p = Perception(load_config(overrides={"perception.register.enabled": False}), SC, CAL)
        self._wrong_pose(p, 0.01, 0.01, 1.0)
        rgb, depth = self.r.render(scene_objects(np.random.default_rng(20)))
        p.detect(rgb, depth)
        self.assertEqual(p.reg["why"], "off")
        T = p.T.copy()
        p._level(depth)                                         # the pose detect() used = the levelled one
        np.testing.assert_allclose(T, p.T)
        self.assertGreater(abs(self._pose_left(p)["yaw"]), 0.9)

    def test_registration_guard(self):
        """A bin moved beyond max_shift, or missing: keep the levelled pose and say why."""
        import copy
        for case in ("moved", "missing"):
            sc = copy.deepcopy(SC)
            if case == "moved":                                 # 7 cm further away
                c = sc.cfg["bins"]["centers"]["center"]
                sc.cfg["bins"]["centers"]["center"] = [c[0], c[1] + 0.07]
            else:
                del sc.cfg["bins"]["centers"]["center"]
            p = Perception(self.CFG0, SC, CAL)
            rgb, depth = Renderer(sc, CAL).render(scene_objects(np.random.default_rng(21)))
            p.detect(rgb, depth)
            self.assertFalse(p.reg["used"], (case, p.reg))
            self.assertIn("not where expected", p.reg["why"])
            T = p.T.copy()
            p._level(depth)
            np.testing.assert_allclose(T, p.T)

    def test_sparse_depth_uses_colour_first(self):
        """Depth below min_valid_frac: the HSV path, unchanged (plane fallback for the blobs)."""
        rng = np.random.default_rng(17)
        objs = scene_objects(rng)
        rgb, depth = self.r.render(objs, rng, hole_frac=0.9)
        dets = self._check(rgb, depth, objs, 0.025)
        self.assertTrue(all(d.n_votes == 0 for d in dets))       # (geometry blobs carry their colour votes)


class TestPostures(unittest.TestCase):
    def test_mirror_keeps_thumb_0(self):
        """thumb_0 is not sign-mirrored between the hands (URDF, sim, real data); every other joint is."""
        m = mirror_left_to_right([-0.9, 0.2, 1.0, -0.1, -0.2, -0.3, -0.4])
        np.testing.assert_allclose(m, [-0.9, -0.2, -1.0, 0.3, 0.4, 0.1, 0.2])
        np.testing.assert_allclose(mirror_left_to_right(m), [-0.9, 0.2, 1.0, -0.1, -0.2, -0.3, -0.4])
        kin = ArmKinematics(load_config(profile="none"))
        left = np.r_[[-0.9, 0.2, 1.0, -0.1, -0.2, -0.3, -0.4], np.zeros(7)]
        np.testing.assert_allclose(kin.tips_in_palm("right", np.r_[np.zeros(7), m]) * [1, -1, 1],
                                   kin.tips_in_palm("left", left), atol=0.002)   # the mirrored posture IS the mirror

    def test_mirror_and_limits(self):
        post = hand_postures(CFG)  # raises if outside joint_spec limits
        for v in post.values():
            np.testing.assert_allclose(v[7:], mirror_left_to_right(v[:7]))
        kin = ArmKinematics(CFG)
        for name, v in post.items():  # mirrored postures put the right fingertips at the mirrored place
            np.testing.assert_allclose(kin.tips_in_palm("right", v), kin.tips_in_palm("left", v) * [1, -1, 1], atol=1e-6)


class TestConfig(unittest.TestCase):
    def test_env_overrides_apply_last(self):
        import json
        import os
        import tempfile
        old = os.environ.get("AALTO_SET")
        try:
            os.environ["AALTO_SET"] = json.dumps({"motion.arm_max_vel": 0.33, "policy.max_attempts": 7})
            cfg = load_config(overrides={"motion.arm_max_vel": 0.5})
            self.assertEqual((cfg["motion"]["arm_max_vel"], cfg["policy"]["max_attempts"]), (0.33, 7))
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump({"motion.chunk": 5}, f)
            os.environ["AALTO_SET"] = "@" + f.name
            self.assertEqual(load_config()["motion"]["chunk"], 5)
            os.unlink(f.name)
        finally:
            os.environ.pop("AALTO_SET", None)
            if old is not None:
                os.environ["AALTO_SET"] = old
        self.assertNotIn("aalto_set", load_config())

    def test_profile_inheritance(self):
        import tempfile
        import yaml
        doc = {"a": {"x": 1, "y": 2, "z": 3}, "b": 0,
               "profiles": {"p": {"a.x": 10, "a.y": 20}, "c": {"inherit": "p", "a.y": 200, "b": 1},
                            "d": {"inherit": "c", "a.z": 30}, "loop1": {"inherit": "loop2"},
                            "loop2": {"inherit": "loop1"}, "orphan": {"inherit": "nope"}}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(doc, f)
        try:
            cfg = load_config(f.name, profile="d")                     # parent's keys first, then its own
            self.assertEqual((cfg["a"], cfg["b"], cfg["profile"]), ({"x": 10, "y": 200, "z": 30}, 1, "d"))
            self.assertNotIn("inherit", cfg)
            self.assertEqual(load_config(f.name, profile="p")["a"], {"x": 10, "y": 20, "z": 3})
            self.assertEqual(load_config(f.name, overrides={"a.y": 5}, profile="c")["a"]["y"], 5)   # overrides last
            with self.assertRaisesRegex(ValueError, "cycle: loop1 -> loop2 -> loop1"):
                load_config(f.name, profile="loop1")
            with self.assertRaisesRegex(ValueError, "unknown config profile 'nope'"):
                load_config(f.name, profile="orphan")
        finally:
            os.unlink(f.name)

    def test_sim_legacy_is_sim_with_the_old_grasp(self):
        """sim_legacy = profile sim except the pinned grasp / neighbour keys (so it follows every other sim value)."""
        env = os.environ.pop("AALTO_SET", None)                        # the files' values, not an experiment's
        try:
            sim, old = load_config(profile="sim"), load_config(profile="sim_legacy")
        finally:
            if env is not None:
                os.environ["AALTO_SET"] = env
        old_claw = {"open": [0.0, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4], "preshape": [0.0, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4],
                    "closed": [0.0, -0.1, 1.2, -1.0, -0.6, -1.0, -0.6],
                    "release": [0.0, -0.5, 1.5, -0.05, -0.05, -0.05, -0.05]}   # (the base claw until 2026-11-14)
        for k in ("open", "preshape", "closed", "release"):
            self.assertEqual(old["hand"][k], old_claw[k])
        self.assertIsNone(old["hand"]["right"])
        self.assertEqual(old["grasp"]["point_in_palm"], [0.099, -0.084, 0.015])
        self.assertEqual(old["ik"]["joint_margin"], {"wrist_roll": 0.22, "wrist_pitch": 0.19, "elbow": 0.19})
        self.assertEqual([old["grasp"]["face_align"][k] for k in ("triangle_phi_deg", "cube")], [105, True])
        self.assertEqual([old["neighbours"][k] for k in ("margin", "floor", "order_free")], [0.005, -0.012, 0.07])
        self.assertEqual((old["grasp"]["extra_hints"], old["right_hand_slots"]), ([], None))
        self.assertFalse(old["grasp"]["correct"]["enabled"])           # (sim: on; test_correct.py)
        self.assertFalse(old["sag"]["enabled"])                        # (sim: on; test_sag.py)
        pinned = {"hand", "grasp.point_in_palm", "ik.joint_margin", "grasp.face_align.triangle_phi_deg",
                  "grasp.face_align.cube", "neighbours.margin", "neighbours.floor", "neighbours.order_free", "profile",
                  "grasp.extra_hints", "right_hand_slots", "grasp.face_align.carry_weight",
                  "grasp.correct.enabled", "sag.enabled"}

        def flat(d, pre=""):
            for k, v in d.items():
                if isinstance(v, dict) and pre + k not in pinned:
                    yield from flat(v, pre + k + ".")
                elif pre + k not in pinned:
                    yield pre + k, v
        self.assertEqual(dict(flat(old)), dict(flat(sim)))


class TestIKLimits(unittest.TestCase):
    def test_per_joint_margin(self):
        spec = JointSpec()
        cfg = load_config(overrides={"ik.joint_margin": {"wrist_roll": 0.22, "wrist_pitch": 0.19, "elbow": 0.19}})
        bc = load_config(profile="none")
        base, sim = ArmKinematics(bc), ArmKinematics(cfg)
        i = spec.arm_names.index("right_wrist_roll_joint")
        # (base = the robot: per-joint margins with room for the sag feed-forward, test_sag.py, and the real arms'
        # observed range ik.joint_range, mirrored onto the right arm: TestJointRange)
        self.assertAlmostEqual(base.lo[i], max(spec.arm_lower[i] + bc["ik"]["joint_margin"]["wrist_roll"],
                                               -bc["ik"]["joint_range"]["wrist_roll"][1]))
        self.assertAlmostEqual(sim.lo[i], spec.arm_lower[i] + 0.22)
        self.assertAlmostEqual(sim.hi[i], spec.arm_upper[i] - 0.22)
        j = spec.arm_names.index("left_shoulder_pitch_joint")          # joints without an entry keep limit_margin
        self.assertAlmostEqual(sim.lo[j], spec.arm_lower[j] + 0.08)
        home = home_arm(cfg)
        L = SC.cfg["layout"]
        for x in np.linspace(*L["region_x"], 4):                         # IK results respect the tighter bounds
            p = SC.mat_to_pelvis([x, np.mean(L["region_y"]), 0.5 * SC.object_height()])
            side = "left" if p[1] > 0 else "right"
            q, ok = sim.solve(side, p, home)[:2]
            self.assertTrue(ok)
            self.assertTrue(np.all(q >= sim.lo - 1e-6) and np.all(q <= sim.hi + 1e-6))


class TestJointRange(unittest.TestCase):
    def test_range_is_mirrored_and_intersected(self):
        """ik.joint_range (LEFT convention): the right arm gets [-hi, -lo] on the roll / yaw joints, the same range on
        pitch joints / elbow; intersected with the margins; unknown joints and empty ranges are errors; base = the
        real arms' observed envelope (home inside it), sim / sim_legacy / oracle: none (URDF limits)."""
        from aaltonomous.sim.oracle import ORACLE_OVERRIDES
        spec = JointSpec()
        cfg = load_config(profile="none", overrides={"ik.joint_range": {"wrist_roll": [-0.5, 1.0],
                                                                        "wrist_pitch": [-1.2, -0.2], "elbow": [-5, 5]}})
        kin = ArmKinematics(cfg)
        i = {n: spec.arm_names.index(n) for n in spec.arm_names}
        self.assertEqual((kin.lo[i["left_wrist_roll_joint"]], kin.hi[i["left_wrist_roll_joint"]]), (-0.5, 1.0))
        self.assertEqual((kin.lo[i["right_wrist_roll_joint"]], kin.hi[i["right_wrist_roll_joint"]]), (-1.0, 0.5))
        for s in ("left", "right"):
            self.assertEqual((kin.lo[i[f"{s}_wrist_pitch_joint"]], kin.hi[i[f"{s}_wrist_pitch_joint"]]), (-1.2, -0.2))
            j = i[f"{s}_elbow_joint"]   # wider than the margins: the margins stay
            self.assertAlmostEqual(kin.lo[j], spec.arm_lower[j] + cfg["ik"]["joint_margin"]["elbow"])
        for bad in ({"wrist_twist": [0, 1]}, {"wrist_roll": [1, 0]}, {"wrist_pitch": [3.0, 4.0]}):
            with self.assertRaises(ValueError):
                ArmKinematics(load_config(profile="none", overrides={"ik.joint_range": bad}))
        base = load_config(profile="none")
        kb = ArmKinematics(base)
        home = home_arm(base)
        self.assertTrue(np.all(home >= kb.lo) and np.all(home <= kb.hi))
        q, ok = kb.solve("left", SC.mat_to_pelvis([0.55, 0.15, 0.5 * SC.object_height()]), home)[:2]
        self.assertTrue(ok)
        self.assertTrue(np.all(q >= kb.lo - 1e-9) and np.all(q <= kb.hi + 1e-9))
        for prof in ("sim", "sim_legacy"):
            self.assertIsNone(load_config(profile=prof)["ik"]["joint_range"])
        self.assertIsNone(load_config(profile="none", overrides=ORACLE_OVERRIDES)["ik"]["joint_range"])


class TestIK(unittest.TestCase):
    def test_reaches_points_check_layout_calls_reachable(self):
        """Points over the mat that check_layout's random palm cloud reaches (palm 8 cm above an object centre)
        must be solved by our IK (palm frame, any orientation) within 1 cm. (cloud tolerance 1 cm, not the 3 cm
        check_layout prints with, so 'reachable' means a sampled configuration really got there.)"""
        from aaltonomous.scene.check_layout import palm_cloud, reach_map
        cfg = load_config(overrides={"grasp.point_in_palm": [0.0, 0.0, 0.0]})
        kin = ArmKinematics(cfg)
        home = home_arm(cfg)
        rng = np.random.default_rng(0)
        W, D = SC.cfg["counter"]["mat_size"]
        mat = np.c_[rng.uniform(0, W, 300), rng.uniform(0, D, 300), np.full(300, SC.object_height() / 2)]
        P = SC.mat_to_pelvis(mat)
        ok = reach_map(SC, palm_cloud(40000), P, 0.08, 0.01)
        pts = P[ok][:40] + [0, 0, 0.08]
        self.assertGreater(len(pts), 10)
        hit = 0
        for p in pts:
            side = "left" if p[1] > 0 else "right"
            res = [kin.solve(s, p, home, top_down=False) for s in (side, "right" if side == "left" else "left")]
            hit += min(r[2] for r in res) < 0.01
        self.assertGreaterEqual(hit / len(pts), 0.95, f"{hit}/{len(pts)}")

    def test_top_down_grasp_over_cube_region(self):
        kin, home = ArmKinematics(CFG), home_arm(CFG)
        L = SC.cfg["layout"]
        n = ok = 0
        for x in np.linspace(*L["region_x"], 5):
            for y in np.linspace(*L["region_y"], 3):
                p = SC.mat_to_pelvis([x, y, 0.5 * SC.object_height()])
                n += 1
                ok += any(kin.solve(s, p, home)[1] for s in ("left", "right"))
        self.assertGreaterEqual(ok / n, 0.8, f"top-down grasp IK {ok}/{n} over the cube region")


class TestPlanner(unittest.TestCase):
    def test_plan_respects_speed_and_safety(self):
        from .perception import Detection
        post, home = hand_postures(CFG), home_arm(CFG)
        pl = Planner(CFG, SC, post, home)
        L = SC.cfg["layout"]
        top = SC.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), SC.object_height()])
        p = pl.plan(Detection("yellow", top, 0, 0, "depth"), home, post["open"])
        self.assertTrue(p.ok, p.reason)
        dq = np.abs(np.diff(np.vstack([home, p.arm]), axis=0)).max()
        self.assertLessEqual(dq, CFG["motion"]["arm_max_vel"] / 30 + 1e-9)
        dh = np.abs(np.diff(np.vstack([post["open"], p.hand]), axis=0)).max()
        self.assertLessEqual(dh, CFG["motion"]["hand_max_vel"] / 30 + 1e-9)
        self.assertEqual(p.safety_clips, 0)
        np.testing.assert_allclose(p.arm[-1], home, atol=1e-9)  # ends at home


    def test_cross_body_object_is_relocated(self):
        """Relocation fallback: the right arm puts a red (left bin) object down on the counter where the left arm has a
        full plan from home. With the published box row the left arm reaches the whole object band directly, so the
        fallback is called explicitly here."""
        from .perception import Detection
        cfg = load_config(profile="none")
        post, home = hand_postures(cfg), home_arm(cfg)
        pl = Planner(cfg, SC, post, home)
        top = SC.mat_to_pelvis([0.30, 0.18, SC.object_height()])
        det = Detection("red", top, 0, 0, "depth")
        direct = pl.plan(det, home, post["open"])
        self.assertTrue(direct.ok, direct.reason)
        self.assertEqual((direct.side, direct.target_bin), ("left", "left"))
        p = pl.plan_relocation(det, "left", home, post["open"])
        self.assertTrue(p.ok, p.reason)
        self.assertEqual((p.side, p.target_bin), ("right", "counter"))
        self.assertEqual(p.safety_clips, 0)
        spot = p.waypoints["drop"]
        self.assertLess(abs(spot[2] - p.waypoints["grasp"][2]), 0.01)  # put down at grasp height, not dropped
        fin = pl.plan(Detection("red", np.r_[spot[:2], top[2]], 0, 0, "depth"), home, post["open"], relocate=False)
        self.assertTrue(fin.ok, fin.reason)
        self.assertEqual((fin.side, fin.target_bin), ("left", "left"))


class TestPlannerFeatures(unittest.TestCase):
    """motion.shortcut_tol, motion.joint_carry_first, motion.bin_margin, grasp.reach_check, grasp.drop_offsets."""

    @staticmethod
    def _planner(**ov):
        cfg = load_config(overrides=ov)  # sim_legacy (= sim's tuned feature values, the old grasp) + ov
        post, home = hand_postures(cfg), home_arm(cfg)
        return Planner(cfg, SC, post, home), post, home

    @staticmethod
    def _det(color, mat_xy):
        from .perception import Detection
        return Detection(color, SC.mat_to_pelvis([*mat_xy, SC.object_height()]), 0, 0, "depth")

    def test_profiles(self):
        base, sim = load_config(profile="none"), load_config(profile="sim")
        self.assertEqual([base["motion"][k] for k in ("shortcut_tol", "joint_carry_first", "bin_margin")],
                         [0.01, True, 0.01])                    # (shortcut + joint-space carry in base since H1)
        self.assertEqual([base["grasp"]["reach_check"], base["grasp"]["drop_offsets"]], [True, [[0.0, 0.0]]])
        self.assertEqual([sim["motion"][k] for k in ("shortcut_tol", "joint_carry_first", "bin_margin")],
                         [0.01, True, 0.0])
        self.assertTrue(sim["grasp"]["reach_check"])
        self.assertEqual(len(sim["grasp"]["drop_offsets"]), 5)

    def test_shortcut_keeps_line_and_shortens(self):
        from .planner import Plan
        on, post, home = self._planner(**{"motion.shortcut_tol": 0.01, "motion.joint_carry_first": False})
        off = self._planner(**{"motion.shortcut_tol": None, "motion.joint_carry_first": False})[0]
        # one segment by hand: 6 cm sideways at carry height above the object band
        a = SC.mat_to_pelvis([0.45, 0.155, 0.16])
        q0, ok = on.kin.solve("left", a, home)[:2]
        self.assertTrue(ok)
        d = np.array([0.0, 0.06, 0.0])
        qs_on = on._cartesian("left", a, a + d, q0, "carry", Plan(False))
        qs_off = off._cartesian("left", a, a + d, q0, "carry", Plan(False))
        self.assertEqual(len(qs_on), 1)
        self.assertGreaterEqual(len(qs_off), 3)
        np.testing.assert_allclose(qs_on[-1], qs_off[-1])
        for f in np.linspace(0, 1, 41):
            p = on.kin.grasp_pose("left", q0 + f * (qs_on[0] - q0))[0]
            t = np.clip((p - a) @ d / (d @ d), 0, 1)
            self.assertLess(np.linalg.norm(p - (a + t * d)), 0.01 + 2e-3)  # checked at 10 points, 2 mm slack between
        path = lambda qs: np.abs(np.diff(np.vstack([q0] + list(qs)), axis=0)).max(axis=1).sum()  # time_scale's length
        self.assertLess(path(qs_on), path(qs_off))
        # whole plan (an object whose carry / retreat shortcut; most descend / lift paths are already near-straight)
        det = self._det("yellow", [0.45, 0.155])
        p_on, p_off = on.plan(det, home, post["open"]), off.plan(det, home, post["open"])
        self.assertTrue(p_on.ok and p_off.ok, (p_on.reason, p_off.reason))
        self.assertEqual(p_on.safety_clips, 0)
        self.assertLessEqual(p_on.duration, p_off.duration)   # (equal when no segment of this plan collapses)

    def test_joint_carry_first(self):
        from .planner import Plan
        det = self._det("red", [0.55, 0.15])
        first, post, home = self._planner(**{"motion.joint_carry_first": True, "motion.shortcut_tol": None})
        cart = self._planner(**{"motion.joint_carry_first": False, "motion.shortcut_tol": None})[0]
        p = first.plan(det, home, post["open"])
        self.assertTrue(p.ok, p.reason)
        self.assertEqual(p.safety_clips, 0)
        g = first.cfg["grasp"]
        z_drop = SC.counter_z + SC.cfg["bins"]["wall_height"] + first.height / 2 + g["drop_above_rim"]
        s, e = next((s, e) for n, s, e in p.phases if n == "carry")
        z = [first.kin.grasp_pose(p.side, q)[0][2] for q in p.arm[s:e]]
        self.assertGreater(min(z), z_drop - 5e-3)  # joint-space carry never dips below the drop height
        wp, q_lift = p.waypoints, p.arm[next(e for n, s, e in p.phases if n == "lift") - 1]
        out_j = first._carry(p.side, q_lift, wp["lift"], wp["carry"], z_drop, Plan(False))  # carry = above the bin
        out_c = cart._carry(p.side, q_lift, wp["lift"], wp["carry"], z_drop, Plan(False))
        self.assertLessEqual(len(out_j), 2)        # joint space: target (+ raised midpoint)
        self.assertGreater(len(out_c), 2)          # Cartesian: one IK waypoint per cart_step

    def test_bin_margin(self):
        class Kin:  # one hand point 5 mm outside the left bin's outer footprint, 2 cm below its rim
            def hand_points(self, side, arm_q, hand_q):
                b = SC.cfg["bins"]
                c = np.asarray(b["centers"]["left"], float)
                m = [c[0] + b["outer_size"][0] / 2 + 0.005, c[1], b["wall_height"] - 0.02]
                return SC.mat_to_pelvis(np.array([m]))
        post, home = hand_postures(CFG), home_arm(CFG)
        for profile, want in (("none", -0.02), ("sim", np.inf)):
            pl = Planner(load_config(profile=profile), SC, post, home, kin=Kin())
            self.assertAlmostEqual(pl.bin_clearance("left", home, post["open"]), want)
        self.assertAlmostEqual(pl.bin_clearance("left", home, post["open"], margin=0.01), -0.02)

    def test_drop_offsets_move_the_drop_point(self):
        det = self._det("red", [0.55, 0.15])
        pl, post, home = self._planner(**{"grasp.slots.enabled": False})   # (slots: test_grasp_plan.py)
        c = pl.drop_xy("left")
        pl.drop_offsets = {"left": (0.03, -0.03)}
        np.testing.assert_allclose(pl.drop_xy("left"), c + [0.03, -0.03])
        np.testing.assert_allclose(pl.drop_xy("left", offset=False), c)
        np.testing.assert_allclose(pl.drop_xy("right"), self._planner()[0].drop_xy("right"))
        p = pl.plan(det, home, post["open"])
        self.assertTrue(p.ok, p.reason)
        self.assertEqual((p.target_bin, p.safety_clips), ("left", 0))
        np.testing.assert_allclose(p.waypoints["drop"][:2], c + [0.03, -0.03])

    def test_reach_check_rejects_cross_body_fast(self):
        import time
        det = self._det("red", [0.30, 0.15])
        pl, post, home = self._planner()
        t0 = time.perf_counter()
        p = pl._plan_side_hints(det, "right", "left", home.copy(), post["open"].copy())
        dt = time.perf_counter() - t0
        self.assertFalse(p.ok)
        self.assertIn("cannot reach the left bin", p.reason)
        self.assertLess(dt, 3.0)
        self.assertEqual(pl._reach, {("right", "left"): False})
        self.assertTrue(pl._reaches_bin("right", "right"))
        p2 = pl._plan_side_hints(det, "right", "left", home.copy(), post["open"].copy(), place_xy=[0.3, 0.0])
        self.assertNotIn("cannot reach", p2.reason)   # a counter place is never rejected by the bin reach test
        off = self._planner(**{"grasp.reach_check": False})[0]
        self.assertIsNone(off._reach_reject(det, "right", "left"))

    def test_policy_cycles_drop_offsets_per_bin(self):
        from .planner import Plan
        pol = ClassicalPolicy(log=None)
        offs = pol.cfg["grasp"]["drop_offsets"]
        pol.perception.detect = lambda rgb, depth: []
        obs = {"head_rgb": None, "head_depth": None}

        def start(bin_name):
            pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
            p = Plan(True, side="left", target_bin=bin_name, waypoints={"grasp": np.zeros(3), "drop": np.zeros(3)},
                     ik={"grasp": (True, 0.0, 0.0)}, arm=pol.home[None].copy(), hand=pol.post["open"][None].copy())
            pol._start(p, 0.0)

        pol._look(obs)
        self.assertEqual(pol.planner.drop_offsets, {b: offs[0] for b in SC.bin_names})
        for b in ("left", "left", "counter", "right") + ("left",) * 4:
            start(b)
        pol._look(obs)
        self.assertEqual(pol.planner.drop_offsets, {"left": offs[6 % len(offs)], "center": offs[0], "right": offs[1]})
        pol.reset()
        pol._look(obs)
        self.assertEqual(pol.planner.drop_offsets, {b: offs[0] for b in SC.bin_names})


    def test_variants_rotate_hints_and_alternate_the_centre_arm(self):
        pl, post, home = self._planner()
        det = self._det("red", [0.55, 0.15])                      # left bin: only the hint order changes
        p = [pl.plan(det, home, post["open"], variant=v) for v in range(4)]
        self.assertTrue(all(q.ok for q in p), [q.reason for q in p])
        self.assertEqual([q.variant for q in p], [0, 1, 2, 3])
        self.assertEqual({q.side for q in p}, {"left"})
        self.assertEqual(p[0].hint, 0)
        self.assertNotEqual(p[1].hint, 0)                         # variant 1 starts at hint 1
        self.assertEqual(p[3].hint, p[0].hint)                    # 3 hints: variant 3 = variant 0
        np.testing.assert_allclose(p[3].arm, p[0].arm)
        c = self._det("yellow", [0.45, 0.155])                    # centre bin: odd variants try the other arm first
        q0, q1 = pl.plan(c, home, post["open"]), pl.plan(c, home, post["open"], variant=1)
        self.assertTrue(q0.ok and q1.ok, (q0.reason, q1.reason))
        self.assertNotEqual(q0.side, q1.side)
        self.assertEqual((q0.hint, q1.hint), (0, 0))

    def test_abort_trajectory(self):
        """Planner.abort on a plan 1.2 s after the close (sim profile): decelerate along the plan, joint-space move
        home with the hand opening, other arm untouched, speed / acceleration limits, 0 clips; rejected when the
        hand is still at grasp height (the home move would pass below motion.counter_clearance)."""
        from .verify import grasp_window
        pl, post, home = self._planner()
        p = pl.plan(self._det("red", [0.55, 0.15]), home, post["open"])
        self.assertTrue(p.ok, p.reason)
        sl, osl = ArmKinematics.sl(p.side), ArmKinematics.sl("right" if p.side == "left" else "left")
        kc, kr, closed = grasp_window(p.phases, p.hand, sl)
        np.testing.assert_allclose(closed, post["closed"][sl])
        k = kc + 36
        self.assertLess(k, kr)
        ab = Planner(pl.cfg, SC, post, home, ArmKinematics(pl.cfg))
        a, h, info = ab.abort(p.side, p.arm[k - 1], p.hand[k - 1], p.arm[k:], p.hand[k:])
        self.assertIsNotNone(a, info)
        self.assertLess(len(a), len(p.arm) - k)
        np.testing.assert_allclose(a[-1, sl], home[sl])
        np.testing.assert_allclose(h[-1, sl], post["open"][sl])
        np.testing.assert_allclose(a[:, osl], np.repeat(p.arm[k - 1][None, osl], len(a), 0))
        np.testing.assert_allclose(h[:, osl], np.repeat(p.hand[k - 1][None, osl], len(a), 0))
        mo = pl.cfg["motion"]
        full = np.vstack([p.arm[k - 2:k], a])
        v = np.abs(np.diff(full, axis=0)).max(1)
        self.assertLessEqual(v.max(), mo["arm_max_vel"] / 30 + 1e-9)
        self.assertGreater(v[0], 0.5 * mo["arm_max_vel"] / 30)   # it was moving: no stop in one tick ...
        acc = np.abs(np.diff(full, 2, axis=0)).max(1) * 900
        self.assertLess(acc.max(), 2 * mo["arm_max_acc"])        # ... but a ramp (time warp: <= ~amax on top of plan's)
        dh = np.abs(np.diff(np.vstack([p.hand[k - 1], h]), axis=0)).max()
        self.assertLessEqual(dh, mo["hand_max_vel"] / 30 + 1e-9)
        self.assertEqual(ab.dry_run_safety(p.arm[k - 1], p.hand[k - 1], a, h), 0)
        a2, _, why = ab.abort(p.side, p.arm[kc - 1], p.hand[kc - 1], p.arm[kc:], p.hand[kc:])
        self.assertIsNone(a2)
        self.assertIn("above counter", why)


class TestVerify(unittest.TestCase):
    def test_profiles(self):
        base, sim = load_config(profile="none"), load_config(profile="sim")
        self.assertTrue(base["verify"]["enabled"])                # real robot: the cage's shortfall (H1 grasp lab)
        self.assertEqual([base["verify"][k] for k in ("signal", "after_s", "empty_below", "confirm")],
                         ["finger_shortfall", 1.0, 0.12, 2])
        self.assertEqual([base["policy"][k] for k in ("max_attempts", "same_object_radius")], [2, 0.03])
        self.assertTrue(sim["verify"]["enabled"])
        self.assertEqual([sim["verify"][k] for k in ("after_s", "empty_below", "confirm")], [1.0, 0.008, 2])
        self.assertEqual([sim["policy"][k] for k in ("max_attempts", "same_object_radius")], [5, 0.06])
        self.assertIsNotNone(ClassicalPolicy(log=None, config=base).abort_planner)

    def test_monitor(self):
        from .verify import GraspMonitor
        tgt = np.linspace(-1, 1, 7)
        held, empty = tgt + [0, 0, 0.03, 0, 0, 0, 0], tgt + 0.002
        m = GraspMonitor(after_s=1.0, empty_below=0.008, confirm=2)
        self.assertEqual(m.update(held, tgt, None), "unknown")      # closed posture not commanded yet
        self.assertEqual(m.update(empty, tgt, 0.67), "unknown")     # too early, however empty it looks
        self.assertEqual(m.update(empty, tgt, 1.0), "unknown")      # 1 low reading: not confirmed
        self.assertEqual(m.update(empty, tgt, 1.33), "empty")
        self.assertFalse(m.slipped)
        self.assertEqual(m.update(held, tgt, 1.67), "empty")        # latched
        m = GraspMonitor(after_s=1.0, empty_below=0.008, confirm=2)
        self.assertEqual(m.update(held, tgt, 0.3), "unknown")
        self.assertEqual(m.update(held, tgt, 1.0), "held")
        self.assertAlmostEqual(m.sig, 0.03)
        self.assertEqual(m.update(empty, tgt, 1.33), "held")        # one low reading
        self.assertEqual(m.update(held, tgt, 1.67), "held")
        self.assertEqual(m.update(empty, tgt, 2.0), "held")
        self.assertEqual(m.update(empty, tgt, 2.33), "empty")       # lost during the carry
        self.assertTrue(m.slipped)

    def test_grasp_window(self):
        from .verify import grasp_window
        post = hand_postures(CFG)
        sl = ArmKinematics.sl("right")
        h = np.repeat(post["open"][None], 40, 0)
        h[10:15, sl] = post["open"][sl] + (post["closed"][sl] - post["open"][sl]) * np.linspace(0.2, 1, 5)[:, None]
        h[15:30, sl] = post["closed"][sl]
        phases = [("descend", 0, 10), ("close", 10, 20), ("lift", 20, 25), ("open", 25, 40)]
        k0, kr, tgt = grasp_window(phases, h, sl)
        self.assertEqual((k0, kr), (14, 25))
        np.testing.assert_allclose(tgt, post["closed"][sl])
        self.assertIsNone(grasp_window(phases[:2], h, sl))

    @unittest.skipUnless(os.environ.get("AALTO_VERIFY_LOGS"), "AALTO_VERIFY_LOGS=<dir: episode_*.npz, grasps.json>")
    def test_replay_recorded_grasps(self):
        """Recorded sim episodes (runner logs, 30 Hz) replayed at the policy's chunk granularity on every grid offset:
        every grasp that lost its object is flagged `empty`, no held grasp is."""
        import json
        from pathlib import Path
        from .verify import EMPTY, GraspMonitor
        d = Path(os.environ["AALTO_VERIFY_LOGS"])
        bad = []
        for g in json.loads((d / "grasps.json").read_text()):
            if g.get("obj") is None or "held_at_open" not in g or g["seed"] == 1001:  # 1001: misaligned trace
                continue
            z = np.load(d / f"episode_{g['seed'] - 1000:03d}.npz")
            sl = ArmKinematics.sl(g["side"])
            hc, hold = z["hand_q_cmd"][:, sl], z["hold"].astype(bool)
            i1 = int(np.argmin(np.abs(z["t"] - g["t1"])))
            k0 = i1
            while np.abs(hc[k0 - 1] - hc[i1]).max() < 1e-6:
                k0 -= 1
            p = np.cumsum(~hold)                                  # policy ticks executed up to each row
            for off in range(10):
                m, flagged = GraspMonitor(1.0, 0.008, 2), False
                for r in range(k0, i1 + 1):                       # the closed posture is commanded at rows k0..i1
                    if hold[r] or (p[r] + 1 + off) % 10 or p[r] + 1 >= p[i1]:  # obs 1 tick before a chunk ends
                        continue
                    flagged |= m.update(z["hand_q"][r, sl], hc[i1], (p[r] - p[k0]) / 30.0) == EMPTY
                if flagged == g["held_at_open"]:
                    bad.append((g["seed"], g["side"], round(g["t0"], 1), g["held_at_open"], off))
        self.assertEqual(bad, [])


class TestBookkeeping(unittest.TestCase):
    """Per-object records (_track), candidate order, attempt cap, time budget, with stubbed perception + planner."""

    @staticmethod
    def _det(color, x, y):
        from .perception import Detection
        return Detection(color, np.array([x, y, SC.counter_z + SC.object_height()]), 500, 400, "depth")

    def _policy(self, dets, durations=None, **ov):
        from .planner import Plan
        pol = ClassicalPolicy(log=None, overrides={**SINGLE, **ov})  # stub plans: no real second-arm plan to merge
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol.perception.detect = lambda rgb, depth: dets
        calls = []

        def plan(d, arm, hand, obstacles=(), variant=0, **kw):
            calls.append((d.color, round(float(d.top[0]), 3), variant))
            n = int(30 * (durations or {}).get(d.color, 5.0))
            return Plan(True, side="left", color=d.color, target_bin="left", variant=variant,
                        waypoints={"grasp": d.top.copy()}, ik={"grasp": (True, 0.0, 0.0)},
                        arm=np.repeat(pol.home[None], n, 0), hand=np.repeat(pol.post["open"][None], n, 0))
        pol.planner.plan = plan
        return pol, calls

    def test_tracking_follows_a_pushed_object(self):
        pol, _ = self._policy([])
        r = pol._track([self._det("red", 0.40, 0.10), self._det("blue", 0.40, -0.10)])
        self.assertEqual([o["id"] for o in r], [0, 1])
        r[0]["tries"], r[0]["tried_xy"] = 1, r[0]["xy"].copy()
        r2 = pol._track([self._det("red", 0.445, 0.11), self._det("blue", 0.40, 0.10),   # pushed 4.6 cm; other colour
                         self._det("red", 0.40, -0.10)])                                 # same colour, elsewhere
        self.assertIs(r2[0], r[0])
        self.assertEqual([o["id"] for o in r2], [0, 2, 3])
        np.testing.assert_allclose(r[0]["xy"], [0.445, 0.11])
        r3 = pol._track([self._det("red", 0.42, 0.10), self._det("red", 0.405, 0.10)])  # both near obj0: nearest wins
        self.assertEqual([o["id"] for o in r3], [4, 0])
        n = len(pol.objs)
        pol._track([self._det("yellow", 0.3, 0.0)], commit=False)                       # planning ahead: not stored
        self.assertEqual(len(pol.objs), n)
        r[0]["delivered"] = 1                                                           # obj0 went into its bin:
        r4 = pol._track([self._det("red", 0.405, 0.10)])                                # a red object there is new
        self.assertIsNot(r4[0], r[0])
        self.assertEqual(r4[0]["tries"], 0)

    def test_untried_first_cap_and_variant(self):
        dets = [self._det("red", 0.30, 0.15), self._det("blue", 0.30, 0.05)]  # red is nearer its bin (nearest_bin)
        pol, calls = self._policy(dets, **{"policy.max_attempts": 2})
        obs = {"head_rgb": None, "head_depth": None, "t": 0.0}
        res = pol._look(obs)
        first = res[1].rec
        self.assertEqual(first["color"], "red")
        pol._start(res[1], 0.0)
        first["misses"] += 1                                      # as _watch does after a verified miss
        res = pol._look(obs)
        self.assertIsNot(res[1].rec, first)                       # the object not tried yet comes first
        pol._start(res[1], 0.0)
        res = pol._look(obs)
        self.assertIs(res[1].rec, first)                          # then the retry, with the next variant
        self.assertEqual(res[1].variant, 1)
        pol._start(res[1], 0.0)
        self.assertEqual(first["variants"], [0, 1])
        res = pol._look(obs)                                      # first: 2 tries = cap
        self.assertIsNot(res[1].rec, first)
        pol._start(res[1], 0.0)
        self.assertEqual(pol._look(obs)[0], "none")
        self.assertEqual([c[2] for c in calls], [0, 0, 1, 0])

    def test_time_budget_tries_shorter_plans(self):
        dets = [self._det("red", 0.30, 0.15), self._det("blue", 0.30, -0.15)]
        pol, calls = self._policy(dets, durations={"red": 9.0, "blue": 4.0})
        pol.time_limit = 10.0
        res = pol._look({"head_rgb": None, "head_depth": None, "t": 3.0})   # 7 s left: red (first) does not fit
        self.assertEqual((res[0], res[1].color), ("plan", "blue"))
        self.assertEqual([c[0] for c in calls], ["red", "blue"])
        self.assertTrue(pol.objs[0]["no_time"])
        self.assertEqual(pol._look({"head_rgb": None, "head_depth": None, "t": 7.0})[0], "none")
        pol, _ = self._policy(dets, durations={"red": 3.0, "blue": 3.0})  # planning took long: re-look, never start
        pol.time_limit = 10.0
        res = pol._look({"head_rgb": None, "head_depth": None, "t": 3.0})
        out = pol._after_look(res, 8.0)
        self.assertEqual((pol.state, out["done"]), ("look", False))
        self.assertTrue(res[1].rec["no_time"])


def run_mock(seed, env_kw=None, overrides=None, setup=None, ticks=None, instruction=None, reset_info=None,
             env_setup=None, **pol_kw):
    """ticks: list that gets every executed tick (commanded arm, hand, (left, right) object held before/after it).
    A config dict for the policy also builds the mock world (its grasp point and hand postures decide held / missed).
    instruction: the env's instruction text (default: the fixed one); reset_info: given to pol.reset instead of
    {"instruction": obs["instruction"]}; env_setup(pol, env): called before the episode."""
    cfg = pol_kw.get("config")
    env = make_mock_env(seed=seed, cfg=cfg if isinstance(cfg, dict) else None, **(env_kw or {}))
    if instruction is not None:
        env.fixed_instruction = instruction
    if ticks is not None:
        b, adv = env.backend, env.backend.advance

        def advance(dt):
            before = [b.held[s] is not None for s in ("left", "right")]
            adv(dt)
            ticks.append((b.arm_cmd.copy(), b.hand_cmd.copy(),
                          tuple(h or b.held[s] is not None for h, s in zip(before, ("left", "right")))))
        b.advance = advance
    pol = ClassicalPolicy(log=None, safety=env.filter.cfg, overrides=overrides, **pol_kw)
    if setup:
        setup(pol)
    if env_setup:
        env_setup(pol, env)
    obs = env.reset()
    pol.reset(reset_info if reset_info is not None else {"instruction": obs["instruction"]})
    starts, orig = [], pol._start
    pol._start = lambda plan, t: (starts.append((t, plan.duration)), orig(plan, t))[1]
    while True:
        obs, info = env.step(pol.act(obs))
        if info["episode_over"]:
            break
    return pol, env, info, starts


def shape_outcomes(env, mapping):
    """Per-object outcome with the object's shape (the mock backend's outcomes() is by colour only)."""
    return [env.scene.classify_cube(o["pos"], 0.0, o["color"], mapping, o.get("shape")) for o in env.backend.objects]


def true_shapes(hide=()):
    """env_setup for run_mock: each detection gets the shape of the mock object under it (the policy's bin logic is
    tested, not the shape classifier), None for objects of a colour in `hide` (shape not recognised)."""
    def setup(pol, env):
        det = pol.perception.detect

        def detect(*a, **kw):
            out = det(*a, **kw)
            for d in out if isinstance(out, list) else out[0]:
                o = min(env.backend.objects, key=lambda o: np.linalg.norm(o["pos"][:2] - d.top[:2]))
                d.shape = None if o["color"] in hide else o["shape"]
            return out
        pol.perception.detect = detect
    return setup


class TestInstructionEpisodes(unittest.TestCase):
    """Target bins from the instruction text alone (ClassicalPolicy.reset -> instruction.py, bin_for)."""

    def _clean(self, info):
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])

    def test_permuted_mapping_from_the_text(self):
        """A template sentence and a paraphrase with other mappings: every object ends in its instructed bin."""
        cases = [(0, "The left bin is for blue objects, the center bin for red, the right bin for yellow.",
                  {"blue": "left", "red": "center", "yellow": "right"}),
                 (1, "Yellow and red go on the right; blue goes in the middle.",
                  {"yellow": "right", "red": "right", "blue": "center"})]
        for seed, text, m in cases:
            with self.subTest(seed=seed):
                pol, env, info, starts = run_mock(seed, instruction=text)
                self._clean(info)
                self.assertEqual((pol.mapping, pol.mapping_source), (m, "parsed"))
                self.assertIn("ok: ", pol.events[0][1])
                self.assertEqual(env.backend.outcomes(m), ["correct"] * len(env.backend.objects))

    def test_shape_instruction_leaves_unrecognised_shapes(self):
        """Shape keys: an object whose shape is not recognised (None) has no bin and is never tried; the others go
        to their shape's bin."""
        text = "Cubes in the left bin, cylinders in the middle bin, triangles in the right bin."
        m = {"cube": "left", "cylinder": "center", "triangle": "right"}
        pol, env, info, starts = run_mock(0, instruction=text, env_setup=true_shapes(hide=("red",)))
        self._clean(info)
        self.assertEqual(pol.mapping, m)
        out = shape_outcomes(env, m)
        colors = [o["color"] for o in env.backend.objects]
        self.assertIn("red", colors)
        self.assertEqual(out, ["counter" if c == "red" else "correct" for c in colors])
        self.assertTrue(all(o["tries"] == 0 for o in pol.objs if o["color"] == "red"))
        self.assertTrue(any(m.startswith("nothing left to sort") and "without a bin" in m for _, m in pol.events))

    def test_partly_understood_sorts_only_that(self):
        """Only red's bin is understood (yellow: unknown word, blue: a contradiction): only red objects are sorted."""
        text = "Red objects go in the right bin. Yellow ones go in the purple bin. Blue goes left. Blue goes right."
        pol, env, info, starts = run_mock(1, instruction=text)
        self._clean(info)
        self.assertEqual(pol.mapping, {"red": "right"})
        out = env.backend.outcomes(pol.mapping)
        colors = [o["color"] for o in env.backend.objects]
        self.assertEqual(out, ["correct" if c == "red" else "counter" for c in colors])
        self.assertIn("NOT OK", pol.events[0][1])

    def test_not_understood_sorts_nothing(self):
        pol, env, info, starts = run_mock(0, instruction="Wave at the audience and then take a bow.")
        self.assertEqual(info["end_reason"], "done")
        self.assertEqual((pol.mapping, starts, env.backend.grasp_log), ({}, [], []))
        self.assertEqual(env.backend.outcomes({"default": "left"}), ["counter"] * len(env.backend.objects))
        self.assertIn("sorting nothing", pol.events[0][1])
        self.assertLess(info["t"], 20.0)

    def test_fixed_instruction_as_before(self):
        """The fixed sentence parses to the fixed mapping and runs exactly as the explicit fixed mapping (the code path
        before instruction parsing): same events, times, outcomes."""
        from aaltonomous.interface.tasks import FIXED_MAPPING
        a = run_mock(2, overrides=SINGLE)
        b = run_mock(2, overrides=SINGLE, reset_info={"mapping": dict(FIXED_MAPPING)})
        self.assertEqual((a[0].mapping, a[0].mapping_source, b[0].mapping_source), (FIXED_MAPPING, "parsed", "given"))
        import re

        def ev(pol):   # (planning wall times differ run to run)
            return [(t, re.sub(r"\d+ ms|[\d.]+ s planning", "#", m)) for t, m in pol.events[1:]]
        self.assertEqual(ev(a[0]), ev(b[0]))
        self.assertGreater(len(a[3]), 0)
        self.assertEqual(a[3], b[3])
        self.assertEqual(a[1].backend.outcomes(FIXED_MAPPING), b[1].backend.outcomes(FIXED_MAPPING))


class TestEpisode(unittest.TestCase):
    def _check(self, env, info, pol):
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        out = env.backend.outcomes(pol.mapping)
        self.assertNotIn("wrong", out)
        self.assertNotIn("off", out)
        return out

    @staticmethod
    def _events(pol, word):
        return [(t, m) for t, m in pol.events if m.startswith(word)]

    @staticmethod
    def _first_attempt_s(pol):
        """Time from the first `execute` to the look after it (the attempt incl. settle)."""
        i = next(i for i, (_, m) in enumerate(pol.events) if m.startswith("execute"))
        return next(t for t, m in pol.events[i:] if m.startswith("look")) - pol.events[i][0]

    def test_full_mock_episodes(self):
        """(sim profile: verification on; single-arm cycles, pairs: TestPairEpisode) every grasp verified held,
        delivered, no abort."""
        total_correct = 0
        for seed in (0, 1, 2):
            pol, env, info, starts = run_mock(seed, overrides=SINGLE)
            out = self._check(env, info, pol)
            total_correct += out.count("correct")
            self.assertEqual(self._events(pol, "abort"), [])
            self.assertEqual(len(self._events(pol, "verify")), len(starts))
            self.assertTrue(all("held sig=0.03" in m for _, m in self._events(pol, "verify")))
        self.assertGreaterEqual(total_correct, 3)

    def test_miss_abort_retry_with_another_variant(self):
        pol, env, info, starts = run_mock(0, {"fail_grasps": (0,)}, overrides=NEW_OFF)
        self.assertEqual(self._check(env, info, pol).count("correct"), len(env.backend.objects))
        ab = self._events(pol, "abort")
        self.assertEqual(len(ab), 1)
        self.assertIn("(miss)", ab[0][1])
        rec = next(o for o in pol.objs if o["misses"])
        self.assertEqual((rec["tries"], rec["misses"], rec["variants"], rec["delivered"]), (2, 1, [0, 1], 1))
        self.assertLess(self._first_attempt_s(pol), starts[0][1] - 1.0)  # ended well before its plan would have

    def test_slip_during_carry_aborts(self):
        pol, env, info, starts = run_mock(2, {"slip_grasps": (0,)}, overrides=SINGLE)
        self._check(env, info, pol)
        self.assertEqual(env.backend.grasp_log[0]["outcome"], "slip")
        v = [m for _, m in self._events(pol, "verify")[:2]]
        self.assertTrue(v[0].startswith("verify left: held") or v[0].startswith("verify right: held"), v)
        self.assertIn("empty", v[1])
        self.assertIn("(slip)", self._events(pol, "abort")[0][1])
        self.assertEqual(sum(o["slips"] for o in pol.objs), 1)

    def test_rejected_abort_finishes_the_plan(self):
        def setup(pol):
            pol.abort_planner.dry_run_safety = lambda *a: 1     # every abort trajectory "clips"
        pol, env, info, starts = run_mock(0, {"fail_grasps": (0,)}, setup=setup)
        self._check(env, info, pol)
        ab = self._events(pol, "abort")
        self.assertTrue(ab and all("rejected" in m for _, m in ab), ab)
        self.assertLessEqual(len(ab), 3)
        self.assertGreaterEqual(self._first_attempt_s(pol), starts[0][1])  # the whole plan ran
        rec = next(o for o in pol.objs if o["misses"])            # still a verified miss: retried another way
        self.assertEqual(rec["variants"], [0, 1])

    def test_attempt_cap(self):
        # neighbours off: with them one object of seed 2 is blocked by a neighbour that never leaves (every grasp
        # fails) and is never tried (test_blocked_by_a_neighbour_that_never_leaves)
        pol, env, info, starts = run_mock(2, {"fail_prob": 1.0}, overrides={"policy.max_attempts": 2, **SINGLE,
                                                                             "neighbours.enabled": False})
        self._check(env, info, pol)
        self.assertEqual(info["end_reason"], "done")
        tried = [o for o in pol.objs if o["tries"]]
        self.assertEqual(len(tried), len(env.backend.objects))
        self.assertEqual({(o["tries"], o["misses"]) for o in tried}, {(2, 2)})
        self.assertEqual(sum(o["misses"] for o in tried), len(starts))   # every attempt verified as a miss
        self.assertGreater(len([m for _, m in self._events(pol, "abort") if "(miss)" in m]), len(starts) // 2)

    def test_blocked_by_a_neighbour_that_never_leaves(self):
        """Today's rule, documented (a known weakness): seed 2, every grasp fails, so no object ever leaves. One blue
        object waits for its blue neighbour (~6 cm away; the right arm's best plan is -18 mm from it in the sphere
        model, the left arm cannot reach the right bin). With neighbours.floor -12 mm it is never tried (no least-bad
        plan) and the episode still ends `done`; with floor -25 mm the least-bad rule takes it once nothing else is
        left (tried up to the cap like the others)."""
        for floor, tries in ((-0.012, 0), (-0.025, 2)):
            with self.subTest(floor=floor):
                ov = {"policy.max_attempts": 2, **SINGLE, **NB, "neighbours.floor": floor}
                pol, env, info, starts = run_mock(2, {"fail_prob": 1.0}, overrides=ov)
                self._check(env, info, pol)
                self.assertEqual(info["end_reason"], "done")
                blocked = [o for o in pol.objs if o["blocked"] is not None]
                self.assertEqual([o["color"] for o in blocked], ["blue"])
                rec = blocked[0]
                self.assertEqual((rec["tries"], rec["failed_at"]), (tries, None))
                self.assertTrue(all(o["tries"] == 2 for o in pol.objs if o is not rec), pol.objs)
                tag = f"obj{rec['id']} blue: no least-bad plan either" if tries == 0 else f"least-bad: obj{rec['id']}"
                self.assertTrue(any(m.startswith(tag) for _, m in pol.events), tag)

    def test_time_limit_never_exceeded_by_a_started_plan(self):
        pol, env, info, starts = run_mock(0, time_limit=33.0)
        self._check(env, info, pol)
        self.assertGreaterEqual(len(starts), 2)
        self.assertTrue(all(t + d <= 33.0 for t, d in starts), starts)
        self.assertTrue(any(m.startswith("obj") and "s left" in m for _, m in pol.events))  # skipped: too long

    def test_confirm_gate_holds_until_confirmed(self):
        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, require_confirm=True)
        obs = env.reset()
        pol.reset()
        obs, info = env.step(pol.act(obs))
        self.assertTrue(pol.waiting)          # first: confirm the move to home
        self.assertIn("home", pol.pending_text)
        pol.confirm()
        for _ in range(400):
            obs, info = env.step(pol.act(obs))
            if pol.waiting:
                break
        self.assertTrue(pol.waiting)          # then: confirm the first object
        self.assertIsNotNone(pol.pending)
        q = obs["arm_q"].copy()
        k = int(pol.cfg["motion"]["hold_ticks"])
        for _ in range(60 // k):  # 2 s of waiting: arms hold (k-tick hold chunks), no watchdog stop
            a = pol.act(obs)
            self.assertEqual(a["arm_q_target"].shape, (k, 14))
            self.assertEqual(np.ptp(a["arm_q_target"], axis=0).max(), 0.0)
            obs, info = env.step(a)
        self.assertEqual(info["safety_mode"], "ok")
        self.assertLess(np.abs(obs["arm_q"] - q).max(), 1e-3)
        pol.confirm()
        obs, info = env.step(pol.act(obs))
        self.assertFalse(pol.waiting)
        self.assertEqual(pol.state, "executing")


def pair_cfg(enabled=True, **ov):
    """Test profile (sim_legacy) + ov with two_arm.enabled set on the dict (after $AALTO_SET: these always pair)."""
    cfg = load_config(overrides=ov)
    cfg["two_arm"]["enabled"] = enabled
    return cfg


class TestPairEpisode(unittest.TestCase):
    """Paired two-arm cycles in mock episodes (seed 1: the first look pairs red/left arm + blue/right arm at lag 0,
    the left hand closes first; seed 2: two pairs)."""

    @staticmethod
    def _events(pol, word):
        return [(t, m) for t, m in pol.events if m.startswith(word)]

    def _check(self, env, info, pol, all_correct=True):
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        out = env.backend.outcomes(pol.mapping)
        self.assertNotIn("wrong", out)
        self.assertNotIn("off", out)
        if all_correct:
            self.assertEqual(out.count("correct"), len(out), out)
        return out

    def _first_pair(self, pol):
        i = next(i for i, (_, m) in enumerate(pol.events) if m.startswith("execute"))
        self.assertTrue(pol.events[i][1].startswith("execute pair"), pol.events[i])
        return pol.events[i]

    def test_pairs_deliver_with_fewer_ticks(self):
        res = {}
        for en in (True, False):
            pol, env, info, starts = run_mock(2, config=pair_cfg(en, **NEW_OFF))
            self._check(env, info, pol)
            res[en] = (info["t"], len(self._events(pol, "look")), len(self._events(pol, "execute pair")))
            if en:   # every grasp verified held (one entry per arm of a pair), no abort; both arms end at home
                self.assertEqual(self._events(pol, "abort"), [])
                self.assertEqual(len(self._events(pol, "verify")), len(env.backend.objects))
                self.assertTrue(all(o["tries"] == 1 and o["delivered"] == 1 and o["variants"] == [0] for o in pol.objs))
                self.assertEqual(sum(pol.drops.values()), len(env.backend.objects))
                np.testing.assert_allclose(pol.arm_cmd, pol.home)
        self.assertGreaterEqual(res[True][2], 1)
        self.assertEqual(res[False][2], 0)
        self.assertLess(res[True][0], res[False][0] - 5.0)        # sim seconds to finish the episode
        self.assertLess(res[True][1], res[False][1])              # fewer looks

    @staticmethod
    def _record_plans(plans):
        def setup(pol):
            orig = pol._start
            pol._start = lambda plan, t: (plans.append(plan), orig(plan, t))[1]
        return setup

    def _one_arm_fails(self, env_kw, outcome, retried=True):
        """First grasp of the first pair (seed 1: the left hand) fails: that arm aborts, the other one delivers its
        object in the same cycle; the failed object is retried (next variant) and delivered later."""
        plans = []
        pol, env, info, starts = run_mock(1, env_kw, config=pair_cfg(**NEW_OFF), setup=self._record_plans(plans))
        self._check(env, info, pol, all_correct=retried)
        self._first_pair(pol)
        g0 = env.backend.grasp_log[0]
        self.assertEqual(g0["outcome"], outcome)
        ab = self._events(pol, "abort")
        self.assertEqual(len(ab), 1, ab)
        self.assertTrue(ab[0][1].startswith(f"abort {g0['side']} ({outcome})"), ab)
        first = {p.side: p.rec for p in plans[0].parts}
        other = "right" if g0["side"] == "left" else "left"
        failed, ok = first[g0["side"]], first[other]
        self.assertEqual(next(g for g in env.backend.grasp_log if g["side"] == other)["outcome"], "held")
        self.assertIn((f"obj{ok['id']} {other}: held until the release"), [m for _, m in pol.events])
        self.assertEqual((ok["tries"], ok["variants"], ok["delivered"], ok["misses"] + ok["slips"]), (1, [0], 1, 0))
        self.assertEqual(failed["misses" if outcome == "miss" else "slips"], 1)
        self.assertEqual(sum(o["misses"] + o["slips"] for o in pol.objs), 1)
        if retried:
            self.assertEqual((failed["tries"], failed["variants"], failed["delivered"]), (2, [0, 1], 1))
            self.assertTrue(any(p is not plans[0] and failed is q.rec for p in plans for q in (p.parts or [p])))

    def test_miss_on_one_arm_of_a_pair(self):
        self._one_arm_fails({"fail_grasps": (0,)}, "miss")

    def test_slip_on_one_arm_of_a_pair(self):
        # the mock drops the slipped object where it is (here x 0.37 m, past the object band: no plan, left there)
        self._one_arm_fails({"slip_grasps": (0,)}, "slip", retried=False)

    def test_abort_that_violates_the_clearance_is_rejected(self):
        calls = []

        def setup(pol):
            def close(arm, hand, step=2, with_object=None, stop_below=None):   # the abort check's kinematics only
                calls.append((len(arm), [np.array(w) for w in with_object], pol.q_t,
                              [m.copy() for m in pol.exec_held]))
                return 0.0, 3
            pol.abort_planner.kin.min_arm_arm_distance = close
        pol, env, info, starts = run_mock(1, {"fail_grasps": (0,)}, config=pair_cfg(), setup=setup)
        self._check(env, info, pol)
        ab = self._events(pol, "abort")
        self.assertTrue(ab and all("rejected" in m for _, m in ab), ab)
        self.assertLessEqual(len(ab), 3)
        self.assertEqual(len(calls), len([m for _, m in ab if "rejected (arm-arm clearance 0 mm < 20 mm" in m]))
        self.assertGreaterEqual(len(calls), 1)
        side = env.backend.grasp_log[0]["side"]
        i, j = (0, 1) if side == "left" else (1, 0)
        for n, wo, q_t, held in calls:      # aborting hand: empty; the other: its planned mask for these ticks
            self.assertFalse(wo[i].any())
            want = held[j][np.minimum(q_t + np.arange(n), len(held[j]) - 1)]
            np.testing.assert_array_equal(wo[j][:len(want)] | want, wo[j])  # (checked_masks: may add the next tick)
            self.assertTrue(want.any())                                     # the other hand holds its object then
        missed = next(o for o in pol.objs if o["misses"])                   # the whole plan ran; retried later
        self.assertEqual((missed["tries"], missed["variants"]), (2, [0, 1]))

    def test_time_limit_with_pairs(self):
        pol, env, info, starts = run_mock(2, config=pair_cfg(), time_limit=24.0)
        self._check(env, info, pol, all_correct=False)
        self.assertGreaterEqual(len(starts), 2)
        self.assertTrue(all(t + d <= 24.0 for t, d in starts), starts)

    def test_pair_degrades_to_its_first_plan_when_late(self):
        pol = ClassicalPolicy(config=pair_cfg(), log=None)
        env = make_mock_env(seed=1)
        env.reset()
        obs = {"head_rgb": None, "head_depth": None, "t": 0.0}
        obs.update(env.backend.images())
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        res = pol._look(obs)
        m = res[1]
        self.assertTrue(m.parts, m.summary())
        first = m.parts[0]
        self.assertLess(first.duration, m.duration)
        pol.state = "look"
        pol._after_look(res, pol.time_limit - 0.5 * (first.duration + m.duration))   # only the first plan fits
        self.assertEqual(pol.state, "executing")
        self.assertEqual(list(pol.execs), [first.side])
        self.assertIsNone(pol.exec_held)
        self.assertEqual(len(pol.queue_a) + pol.q_t, len(first.arm))
        self.assertEqual([o["tries"] for o in (first.rec, m.parts[1].rec)], [1, 0])

    def test_confirm_covers_the_pair(self):
        env = make_mock_env(seed=1)
        pol = ClassicalPolicy(config=pair_cfg(), log=None, require_confirm=True)
        obs = env.reset()
        pol.reset()
        obs, info = env.step(pol.act(obs))
        pol.confirm()                                              # the move to home
        for _ in range(400):
            obs, info = env.step(pol.act(obs))
            if pol.waiting:
                break
        self.assertTrue(pol.waiting and pol.pending.parts)
        self.assertTrue(pol.pending_text.startswith("PAIR"), pol.pending_text)
        for p in pol.pending.parts:
            self.assertIn(f"{p.color} {p.side} arm", pol.pending_text)
        pol.confirm()                                              # one confirmation: both arms
        obs, info = env.step(pol.act(obs))
        self.assertEqual((pol.waiting, pol.state), (False, "executing"))
        self.assertEqual(set(pol.execs), {"left", "right"})
        self.assertTrue(all(p.rec["tries"] == 1 for p in pol.pending.parts))


def parked_cfg(pairs=False, **ov):
    """Test profile + ov with two_arm.avoid_parked on and pairing as given (set on the dict, after $AALTO_SET)."""
    cfg = pair_cfg(pairs, **ov)
    cfg["two_arm"]["avoid_parked"] = True
    return cfg


EVADE_SEEDS = (6, 3)   # mock layouts with evades (single-arm cycles: seed 6 single + 3 evades, seed 3: 3 + 2)


class TestParkedEpisode(unittest.TestCase):
    """two_arm.avoid_parked in mock episodes: every executed tick (single plans, evades, pairs, aborts) keeps
    two_arm.min_clearance in the sphere model, with the objects the mock hands really hold."""

    kin = None

    @classmethod
    def setUpClass(cls):
        cls.kin = ArmKinematics(load_config())   # the episodes' profile (grasp point = the held-object sphere)

    @staticmethod
    def _events(pol, word):
        return [(t, m) for t, m in pol.events if m.startswith(word)]

    def _check(self, env, info, pol, ticks, all_correct=True):
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        out = env.backend.outcomes(pol.mapping)
        self.assertNotIn("wrong", out)
        self.assertNotIn("off", out)
        if all_correct:
            self.assertEqual(out.count("correct"), len(out), out)
        need = float(pol.cfg["two_arm"]["min_clearance"])
        d = np.array([self.kin.arm_arm_distance(a, h, w) for a, h, w in ticks])
        self.assertGreater(len(d), 100)
        self.assertGreaterEqual(d.min(), need, f"tick {int(np.argmin(d))}")
        return out

    def test_episodes_keep_clear(self):
        """Single-arm cycles and pairs: evades happen, everything is delivered with one try each, every grasp is
        verified once (the evading arm has nothing to verify), both arms end at home."""
        n_evade = 0
        for seed, pairs in ((EVADE_SEEDS[0], False), (EVADE_SEEDS[1], False), (9, True)):   # seed 9: pair + 2 evades
            ticks = []
            pol, env, info, starts = run_mock(seed, config=parked_cfg(pairs), ticks=ticks)
            self._check(env, info, pol, ticks)
            ex = self._events(pol, "execute")
            n_evade += len([m for _, m in ex if m.startswith("execute evade")])
            self.assertEqual(self._events(pol, "abort"), [])
            self.assertEqual(len(self._events(pol, "verify")), len(env.backend.objects))
            self.assertTrue(all(o["tries"] == 1 and o["delivered"] == 1 for o in pol.objs), pol.objs)
            self.assertEqual(sum(pol.drops.values()), len(env.backend.objects))
            np.testing.assert_allclose(pol.arm_cmd, pol.home)
            for _, m in self._events(pol, "evade:"):
                self.assertNotIn("skipped", m)
        self.assertGreaterEqual(n_evade, 7)

    def test_miss_during_an_evade_aborts(self):
        """Seed 6: the second grasp is the first evaded pick; it misses -> that arm aborts (checked against the
        evading arm's queued motion), the other arm still returns home; the object is retried and delivered."""
        ticks = []
        pol, env, info, starts = run_mock(EVADE_SEEDS[0], {"fail_grasps": (1,)}, config=parked_cfg(), ticks=ticks)
        self._check(env, info, pol, ticks)
        g = env.backend.grasp_log[1]
        self.assertEqual(g["outcome"], "miss")
        ab = self._events(pol, "abort")       # (an abort too close to the evading arm is rejected, retried next chunk)
        self.assertTrue(ab and ab[-1][1].startswith(f"abort {g['side']} (miss)"), ab)
        self.assertTrue(all("rejected (arm-arm clearance" in m for _, m in ab[:-1]), ab)
        last = [m for t, m in pol.events if m.startswith("execute") and t <= ab[0][0]][-1]
        self.assertTrue(last.startswith("execute evade"), last)
        rec = next(o for o in pol.objs if o["misses"])
        self.assertEqual((rec["tries"], rec["misses"], rec["variants"], rec["delivered"]), (2, 1, [0, 1], 1))
        self.assertEqual(len(self._events(pol, "verify")), len(env.backend.grasp_log))

    def test_no_evade_skips_the_object(self):
        def setup(pol):
            pol.planner._clear = []                                        # no clear pose: every evade fails
        ticks = []
        pol, env, info, starts = run_mock(EVADE_SEEDS[0], config=parked_cfg(), setup=setup, ticks=ticks)
        out = self._check(env, info, pol, ticks, all_correct=False)    # nothing unsafe executed
        sk = [m for _, m in self._events(pol, "evade:") if "skipped, no evade" in m]
        self.assertEqual(len(sk), 3, sk)                                   # seed 6: the 3 cross-body objects
        self.assertEqual(self._events(pol, "execute evade"), [])
        self.assertEqual(info["end_reason"], "done")
        skipped = [o for o in pol.objs if o["failed_at"] is not None]
        self.assertTrue(len(skipped) == 3 and all(o["tries"] == 0 for o in skipped), pol.objs)
        self.assertEqual(out.count("correct"), len(out) - 3)
        # one look: the conflicting candidate is skipped, the next one (clear) is planned and returned
        from .perception import Detection
        pol = ClassicalPolicy(config=parked_cfg(**{"policy.order": "largest", **NEW_OFF}), log=None)  # (9 cm: blocked)
        pol.planner._clear = []
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        top = SC.counter_z + SC.object_height()
        dets = [Detection("red", np.array([0.30, -0.16, top]), 900, 900, "depth"),   # left arm, over the right one
                Detection("blue", np.array([0.30, -0.07, top]), 500, 500, "depth")]  # right arm, its own side
        pol.perception.detect = lambda rgb, depth: dets
        kind, plan, n = pol._look({"head_rgb": None, "head_depth": None, "t": 0.0})
        self.assertEqual((kind, plan.color, plan.side), ("plan", "blue", "right"))
        self.assertFalse(plan.evade or plan.parts)                          # clear as it is
        red = pol.objs[0]
        self.assertEqual((red["color"], red["tries"]), ("red", 0))
        np.testing.assert_allclose(red["failed_at"], [0.30, -0.16])
        self.assertFalse(pol._eligible(red, red["xy"]))                     # until it moves

    def test_single_abort_is_checked_against_the_parked_arm(self):
        calls = []

        def setup(pol):
            def close(arm, hand, step=2, with_object=None, stop_below=None):
                calls.append([np.array(w) for w in with_object])
                return 0.0, 3
            pol.abort_planner.kin.min_arm_arm_distance = close
        ticks = []
        pol, env, info, starts = run_mock(0, {"fail_grasps": (0,)}, config=parked_cfg(), setup=setup, ticks=ticks)
        self._check(env, info, pol, ticks)
        ab = self._events(pol, "abort")
        self.assertTrue(ab and all("rejected (arm-arm clearance 0 mm < 20 mm" in m for _, m in ab), ab)
        self.assertEqual(len(calls), len(ab))
        self.assertTrue(all(not w.any() for c in calls for w in c))         # parked arm and aborting hand: no object

    def test_off_is_todays_behaviour(self):
        """avoid_parked off: the same layout executes a plan that overlaps the parked arm in the model (that is what
        this feature exists for); no evade, no skip."""
        ticks = []
        cfg = pair_cfg(False)
        cfg["two_arm"]["avoid_parked"] = False
        pol, env, info, starts = run_mock(EVADE_SEEDS[0], config=cfg, ticks=ticks)
        self.assertEqual(self._events(pol, "evade"), [])
        self.assertEqual(self._events(pol, "execute evade"), [])
        d = min(self.kin.arm_arm_distance(a, h, w) for a, h, w in ticks)
        self.assertLess(d, float(cfg["two_arm"]["min_clearance"]))

    def test_confirm_says_the_other_arm_moves_aside(self):
        env = make_mock_env(seed=EVADE_SEEDS[0])
        pol = ClassicalPolicy(config=parked_cfg(), log=None, require_confirm=True)
        obs = env.reset()
        pol.reset()
        obs, info = env.step(pol.act(obs))
        pol.confirm()                                                      # the move to home
        for _ in range(3000):                                              # confirm plans until one is an evade
            obs, info = env.step(pol.act(obs))
            if pol.waiting:
                if pol.pending.evade:
                    break
                pol.confirm()
        self.assertTrue(pol.waiting and pol.pending.evade)
        self.assertTrue(pol.pending_text.startswith("EVADE") and "moves aside" in pol.pending_text, pol.pending_text)
        pol.confirm()
        obs, info = env.step(pol.act(obs))
        self.assertEqual((pol.waiting, pol.state), (False, "executing"))
        self.assertEqual(list(pol.execs), [pol.pending.parts[0].side])     # only the picking arm is verified
        self.assertIsNotNone(pol.exec_held)


class TestSimProfile(unittest.TestCase):
    """profiles.sim's own grasp, explicitly (profile="sim"; the classes above run on sim_legacy): the properties the
    thumb-opposed pinch (sim grasp lab, 2026-11-14) relies on. These SHOULD fail when a re-tune breaks one of them.
    Planning with it: test_grasp_plan.TestSimProfilePlans."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(profile="sim")
        cls.kin, cls.post = ArmKinematics(cls.cfg), hand_postures(cls.cfg)

    def test_hand_postures_inside_the_sim_soft_limits(self):
        """Every posture of both hands (profile and ORACLE_OVERRIDES) in the middle 90 % of each hand joint's URDF
        range (the sim's soft_joint_pos_limit_factor 0.9: beyond it a posture never arrives and reads as a held
        object); hand.right explicit and not the mirror (the sim's right-hand wiring)."""
        from aaltonomous.sim.oracle import ORACLE_OVERRIDES
        spec = JointSpec()
        r = spec.hand_upper - spec.hand_lower
        lo, hi = spec.hand_lower + 0.05 * r, spec.hand_upper - 0.05 * r
        oracle = hand_postures(load_config(profile="none", overrides=ORACLE_OVERRIDES))
        self.assertEqual(set(self.post), {"open", "preshape", "closed", "release"})
        for src, post in (("sim", self.post), ("oracle", oracle)):
            for name, v in post.items():
                bad = [spec.hand_names[i] for i in np.flatnonzero((v < lo - 1e-9) | (v > hi + 1e-9))]
                self.assertEqual(bad, [], (src, name))
        for name, v in self.post.items():
            np.testing.assert_allclose(oracle[name], v, err_msg=name)    # the oracle pins the sim grasp
            self.assertIn(name, self.cfg["hand"]["right"])
            if not np.allclose(v[3:5], v[5:7]):   # one finger straight, one bent: the sim's slot order differs from
                self.assertFalse(np.allclose(v[7:], mirror_left_to_right(v[:7])), name)   # the spec's (mirror's)
        self.assertFalse(np.allclose(self.post["closed"][3:5], self.post["closed"][5:7]))

    def test_closed_thumb_opposite_the_index(self):
        """Left hand, closed (URDF tips in the palm frame, z = across the palm): the thumb tip on the index tip's side
        (measured 0.1 mm apart in z), the middle finger out of the pinch, the grasp point between the two tips along
        the pinch (62 % of the way from the thumb) and at most 1 cm from their line (8 mm above it: the object's
        centre sits higher in the hand than the fingertips that pinch it), the gap smaller than the object (33.6 mm)."""
        index, middle, thumb = self.kin.tips_in_palm("left", self.post["closed"])
        self.assertLess(abs(index[2] - thumb[2]), 0.003)
        self.assertGreater(abs(middle[2] - thumb[2]), 0.04)
        g, u = np.asarray(self.cfg["grasp"]["point_in_palm"], float), index - thumb
        s = (g - thumb) @ u / (u @ u)
        self.assertTrue(0.3 < s < 0.7, s)
        self.assertLess(np.linalg.norm(thumb + s * u - g), 0.010)
        self.assertLess(abs((thumb + s * u - g)[2]), 0.002)       # (on the pinch line across the palm)
        self.assertLess(np.linalg.norm(u), SC.size)

    def test_right_hand_model_is_the_mirror_image(self):
        """The kinematic model must be the hand that moves: joint_spec.yaml orders the right hand as index [3, 4],
        middle [5, 6], the sim wires it as middle [3, 4], index [5, 6], and the explicit hand.right is written for
        the sim. right_hand_slots (profiles.sim) tells ArmKinematics which joint each slot drives; without it the
        modelled right hand closed its middle finger, 57 mm across the palm from the thumb, and every right-hand
        check (fingertips / counter / bin rims, neighbours, arm-arm clearance) used another hand's shape."""
        self.assertEqual(list(self.cfg["right_hand_slots"]), [0, 1, 2, 5, 6, 3, 4])
        plain = ArmKinematics(load_config(profile="sim", overrides={"right_hand_slots": None}))
        self.assertGreater(np.abs(plain.tips_in_palm("right", self.post["closed"]) * [1, -1, 1]
                                  - plain.tips_in_palm("left", self.post["closed"])).max(), 0.05)   # (the old model)
        for name, v in self.post.items():
            np.testing.assert_allclose(self.kin.tips_in_palm("right", v) * [1, -1, 1], self.kin.tips_in_palm("left", v),
                                       atol=0.002, err_msg=name)

    def test_ik_margins_cover_every_arm_joint(self):
        """ik.joint_margin names all 7 arm joints, each >= 5 % of its URDF range on both arms (the sim's soft limits:
        IK targets beyond them are never reached)."""
        spec = JointSpec()
        self.assertEqual(set(self.cfg["ik"]["joint_margin"]), {n[5:-6] for n in spec.arm_names[:7]})  # left_..._joint
        r = spec.arm_upper - spec.arm_lower
        m = np.minimum(self.kin.lo - spec.arm_lower, spec.arm_upper - self.kin.hi)
        self.assertEqual([n for n, a, w in zip(spec.arm_names, m, r) if a < 0.05 * w - 1e-9], [])

    def test_mock_episodes(self):
        """Mock world built from the sim profile (its grasp point decides held / missed; the closure is measured along
        the profile's own open -> closed, so this posture is modelled as any other): pairs, evades, neighbours and
        verification as tuned; seeds 0-2: every object delivered on its first try, every grasp verified held, 0 clips,
        both arms >= two_arm.min_clearance apart in the sphere model at every tick."""
        n_pair = n_evade = 0
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                ticks, cfg = [], load_config(profile="sim")
                pol, env, info, starts = run_mock(seed, config=cfg, ticks=ticks)
                self.assertEqual(info["end_reason"], "done")
                self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
                out = env.backend.outcomes(pol.mapping)
                self.assertEqual(out.count("correct"), len(out), out)
                self.assertEqual([g["outcome"] for g in env.backend.grasp_log], ["held"] * len(out))
                v = [m for _, m in pol.events if m.startswith("verify")]
                self.assertTrue(len(v) == len(out) and all(": held" in m for m in v), v)
                self.assertTrue(all(o["tries"] == 1 and o["delivered"] == 1 for o in pol.objs), pol.objs)
                d = min(self.kin.arm_arm_distance(a, h, w) for a, h, w in ticks)
                self.assertGreaterEqual(d, float(cfg["two_arm"]["min_clearance"]))
                ex = [m for _, m in pol.events if m.startswith("execute")]
                n_pair += sum(m.startswith("execute pair") for m in ex)
                n_evade += sum(m.startswith("execute evade") for m in ex)
        self.assertGreaterEqual(n_pair, 1)                                 # (measured: 3 pairs, 2 evades)
        self.assertGreaterEqual(n_evade, 1)


if __name__ == "__main__":
    unittest.main()
