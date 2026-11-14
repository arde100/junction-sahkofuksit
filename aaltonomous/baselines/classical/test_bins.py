"""Box detection (bins.py) and everything downstream of it, offline (rendered mock world). Run:
    python -m unittest aaltonomous/baselines/classical/test_bins.py
    python -m aaltonomous.baselines.classical.test_bins grid      # the full synthetic grid (report, ~5 min)
    python -m aaltonomous.baselines.classical.test_bins reach     # drop slots left vs row distance / wall height
    python -m aaltonomous.baselines.classical.test_bins frames [dir ...]   # recorded sim frames vs scene.yaml
"""
import json
import os
import sys
import unittest

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .bins import Box, Boxes, BoxTracker
from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .mock_world import REAL_COLORS, Renderer, box_row, make_mock_env
from .perception import Detection, Perception
from .planner import Planner
from .policy import ClassicalPolicy

SC, CAL = Scene.load(), Calibration.load()
BASE = load_config(profile="none")                 # the robot profile: perception.boxes.mode on
TABLE = [1.4, 1.2]                                 # mat axes: a table larger than the mat (boxes up to x ~0.9)
SIZE = 0.215                                       # robot data: 0.21-0.22


def pitched(deg):
    """The true head pose when the camera looks `deg` further down than calibration.yaml (about its optical x axis,
    i.e. its own centre; robot data: 2.8 deg)."""
    from scipy.spatial.transform import Rotation
    T = CAL._T("head").copy()
    T[:3, :3] = T[:3, :3] @ Rotation.from_euler("x", -deg, degrees=True).as_matrix()
    return T


def as_boxes(rows, rim=None):
    """box_row() dicts -> bins.Boxes (the truth, as the planner would get it)."""
    return Boxes({b["name"]: Box(b["name"], tuple(b["c"]), b["yaw"], b["size"] / 2, b["size"] / 2,
                                 b["wall_h"] if rim is None else rim, b["wall_t"], b["floor_t"], 1.0, True, "detected")
                  for b in rows}, source="detected")


def objects(rng, rows, n_band=3, n_in=1, size=0.05):
    """n_band objects in the layout band (x 0.26-0.35), n_in on the floors of random boxes."""
    L, out, used = SC.cfg["layout"], [], []
    cols = ["red", "yellow", "blue"]
    while len(out) < n_band:
        x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
        if any(np.hypot(x - a, y - b) < 0.09 for a, b in used):
            continue
        used.append((x, y))
        out.append(dict(color=cols[len(out) % 3], shape=("cube", "cylinder")[len(out) % 2], yaw=rng.uniform(-0.8, 0.8),
                        size=size, height=size, pos=SC.mat_to_pelvis([x, y, size / 2])))
    for k in range(n_in if len(rows) else 0):
        b = rows[int(rng.integers(len(rows)))]
        m = b["size"] / 2 - b["wall_t"] - size / 2 - 0.005
        uv = rng.uniform(-m, m, 2)
        c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
        xy = b["c"] + [c * uv[0] - s * uv[1], s * uv[0] + c * uv[1]]
        out.append(dict(color=cols[k % 3], shape="cube", yaw=rng.uniform(-0.8, 0.8), size=size, height=size,
                        pos=np.r_[xy, SC.counter_z + b["floor_t"] + size / 2]))
    return out


def arm_spheres(profile):
    """The arms' sphere model at that profile's home pose (what the head camera sees of them; the self-mask)."""
    cfg = load_config(profile=profile)
    kin, q, h = ArmKinematics(cfg), home_arm(cfg), hand_postures(cfg)["open"]
    return [(kin.hand_spheres(s, q, h)[0][0], kin.hand_spheres(s, q, h)[1]) for s in ("left", "right")]


def detect_case(front, yaw, wall, gap, pitch=0.0, seed=0, n_band=3, n_in=1, arms=None, cfg=BASE, rows=None,
                noise=2.0, holes=0.03):
    """Render one layout and detect. Returns (perception, truth rows, objects, errors {name: (dxy mm, dyaw deg,
    drim mm, dside mm)}, missing names, wrong names)."""
    rng = np.random.default_rng(seed)
    rows = box_row(front, yaw, wall, gap, size=SIZE, wall_t=0.004) if rows is None else rows
    objs = objects(rng, rows, n_band, n_in)
    r = Renderer(SC, CAL, colors=REAL_COLORS, boxes=rows, T=pitched(pitch) if pitch else None, counter_size=TABLE)
    rgb, depth = r.render(objs, rng, noise, holes, spheres=arms)
    p = Perception(cfg, SC, CAL)
    p.arm_spheres = arms
    p.dets = p.detect(rgb, depth)
    truth = {b["name"]: b for b in rows}
    errs, missing, wrong = {}, [], []
    for n in ("left", "center", "right"):
        if not p.boxes.ok(n):
            missing.append(n)
            continue
        b, t = p.boxes[n], truth.get(n)
        if t is None or np.linalg.norm(np.subtract(b.c, t["c"])) > 0.05:
            wrong.append(n)
            continue
        errs[n] = (1000 * np.linalg.norm(np.subtract(b.c, t["c"])), np.degrees(b.yaw - t["yaw"]),
                   1000 * (b.rim - t["wall_h"]), 1000 * (2 * b.hv - t["size"]))
    return p, rows, objs, errs, missing, wrong


# ---------------------------------------------------------------------- unit tests
class TestBoxGeometry(unittest.TestCase):
    def test_scene_boxes_are_scene_yaml(self):
        """Boxes.from_scene = scene.yaml's bins: the same footprint test as the code before (mat axes)."""
        B = Boxes.from_scene(SC)
        b = SC.cfg["bins"]
        rng = np.random.default_rng(0)
        P = rng.uniform([0.3, -0.5], [0.7, 0.5], (4000, 2))
        pm = SC.pelvis_to_mat(np.c_[P, np.zeros(len(P))])
        for name, ctr in b["centers"].items():
            old = np.all(np.abs(pm[:, :2] - np.asarray(ctr)) <= np.asarray(b["outer_size"]) / 2 + 0.01, axis=1)
            np.testing.assert_array_equal(B[name].inside(P, 0.01), old)
            self.assertAlmostEqual(B[name].rim, b["wall_height"])
        np.testing.assert_allclose(B["left"].c, [0.505, 0.26], atol=1e-9)

    def test_frames(self):
        bx = Box("x", (0.6, 0.1), np.radians(14), 0.1, 0.1, 0.1, 0.004)
        uv = np.random.default_rng(1).uniform(-0.2, 0.2, (50, 2))
        np.testing.assert_allclose(bx.local(bx.pelvis(uv)), uv, atol=1e-12)
        self.assertLess(bx.gap(bx.pelvis([0.0, 0.0])), 0)
        self.assertAlmostEqual(float(bx.gap(bx.pelvis([0.13, 0.0]))), 0.03)
        np.testing.assert_allclose(bx.front, bx.pelvis([-0.1, 0.0]))


class TestDetect(unittest.TestCase):
    TOL = (6.0, 1.0, 4.0)                  # centre mm, yaw deg, rim mm

    def _ok(self, case, errs, missing, wrong, need=("left", "center", "right")):
        self.assertEqual(wrong, [], case)
        self.assertEqual([n for n in need if n in missing], [], case)
        for n, (dc, dy, dr, _) in errs.items():
            self.assertLess(dc, self.TOL[0], (case, n, errs))
            self.assertLess(abs(dy), self.TOL[1], (case, n, errs))
            self.assertLess(abs(dr), self.TOL[2], (case, n, errs))

    def test_scene_layout(self):
        """scene.yaml's boxes (5 cm gaps, 6 cm walls) as the sim draws them: all three, at their place."""
        p, rows, objs, errs, missing, wrong = detect_case(0.40, 0, 0.06, 0.05, rows=box_row(size=0.21))
        self._ok("scene", errs, missing, wrong)
        # not the one inside; rendered in the real camera's colours (REAL_COLORS), named by the robot profile's classes
        seen = {"red": "orange", "yellow": "yellow", "blue": "purple"}
        self.assertEqual(sorted(d.color for d in p.dets), sorted(seen[o["color"]] for o in objs[:3]))

    def test_robot_like_layouts(self):
        """The robot data's layouts (row at 0.46-0.62, yaw -10 ... +14 deg, 2 cm gaps, 10 cm walls) with the camera
        looking 2.8 deg further down than calibrated (levelling must absorb it), objects in front and in boxes."""
        for k, (front, yaw, wall, gap, pitch) in enumerate([(0.46, 0, 0.10, 0.02, 2.8), (0.50, 14, 0.10, 0.02, 2.8),
                                                            (0.56, -10, 0.10, 0.02, 0.0), (0.53, 8, 0.10, 0.02, 2.8),
                                                            (0.50, -7, 0.06, 0.05, 2.8)]):
            with self.subTest(front=front, yaw=yaw, wall=wall, gap=gap, pitch=pitch):
                p, rows, objs, errs, missing, wrong = detect_case(front, yaw, wall, gap, pitch, seed=k)
                self._ok((front, yaw, wall, gap, pitch), errs, missing, wrong)
                if pitch:
                    self.assertAlmostEqual(p.level["pitch_deg"], pitch, delta=0.15)

    def test_sim_home_arms_over_the_boxes(self):
        """Profile sim's home pose puts the hands over the side boxes' near walls: the self-mask removes them."""
        p, rows, objs, errs, missing, wrong = detect_case(0.40, 0, 0.06, 0.05, rows=box_row(size=0.21),
                                                          arms=arm_spheres("sim"))
        self._ok("sim home", errs, missing, wrong)

    def test_fewer_boxes_named_only_when_certain(self):
        """Fewer than three boxes seen: named by position when the row is centred (centre_y_max), the rest not
        found; a row off centre: not named, kept for the masks only (nothing to deliver); centre_y_max null: two
        neighbours are not named either."""
        rows = box_row(0.46, 0, 0.10, 0.02, size=SIZE, wall_t=0.004, names=("center", "right"))
        p, _, _, errs, missing, wrong = detect_case(0, 0, 0, 0, rows=rows)
        self.assertEqual((wrong, missing), ([], ["left"]))
        self.assertIn("left not found", p.boxes.why)
        off = box_row(0.46, 0, 0.10, 0.02, size=SIZE, wall_t=0.004, y0=0.12, names=("center",))
        p, _, _, errs, missing, wrong = detect_case(0, 0, 0, 0, rows=off)
        self.assertEqual(missing, ["left", "center", "right"])
        self.assertEqual(len(p.boxes.more), 1)                  # kept for the masks
        strict = load_config(profile="none", overrides={"perception.boxes.centre_y_max": None})
        p, _, _, errs, missing, wrong = detect_case(0, 0, 0, 0, rows=rows, cfg=strict)
        self.assertEqual((wrong, missing), ([], ["left", "center", "right"]), p.box_det["why"])
        p, _, _, errs, missing, wrong = detect_case(0, 0, 0, 0, rows=[])
        self.assertEqual(missing, ["left", "center", "right"])

    def test_no_boxes_found_on_objects_alone(self):
        """No boxes on the table: objects (and their clusters) are no boxes."""
        rng = np.random.default_rng(3)
        objs = objects(rng, [], 4, 0)
        for o in objs[:2]:                                       # two of them further out, where boxes would be
            o["pos"] = o["pos"] + [0.25, 0.0, 0.0]
        rgb, depth = Renderer(SC, CAL, boxes=[], counter_size=TABLE).render(objs, rng, 2.0, 0.03)
        p = Perception(BASE, SC, CAL)
        p.detect(rgb, depth)
        self.assertEqual([n for n in p.boxes.names if p.boxes.ok(n)], [])
        self.assertEqual(p.box_det["res"]["boxes"] if p.box_det["res"] else [], [])

    def test_recorded_sim_frames(self):
        """Recorded Isaac frames (skipped when absent), profile sim (registration, boxes logged): all three boxes
        within 3 mm / 1 deg / 2 mm of scene.yaml; the objects found are the same as without box detection."""
        from pathlib import Path
        files = []
        for d in ("nominal", "train"):
            files += sorted(Path(os.environ.get("AALTO_FRAMES_DIR", "/data/aaro/frames"), d).glob("*.npz"))[:8]
        if not files:
            self.skipTest("no recorded frames")
        cfg = load_config(profile="sim")
        p, p0 = Perception(cfg, SC, CAL), Perception(load_config(profile="sim", overrides={
            "perception.boxes.mode": "off"}), SC, CAL)
        p.arm_spheres = arm_spheres("sim")
        truth = Boxes.from_scene(SC)
        for f in files:
            z = np.load(f)
            p.reset_boxes()
            dets = p.detect(z["rgb"], z["depth"])
            self.assertEqual([(d.color, tuple(np.round(d.top, 4))) for d in dets],
                             [(d.color, tuple(np.round(d.top, 4))) for d in p0.detect(z["rgb"], z["depth"])])
            res = p.box_det["res"]
            self.assertTrue(res["named"], (f.name, res["why"]))
            for b in res["boxes"]:
                t = truth[b.name]
                self.assertLess(np.linalg.norm(np.subtract(b.c, t.c)), 0.003, (f.name, b.name))
                self.assertLess(abs(np.degrees(b.yaw)), 1.0, (f.name, b.name))
                self.assertLess(abs(b.rim - t.rim), 0.002, (f.name, b.name))


class TestTracker(unittest.TestCase):
    def _res(self, boxes, named=True, why=None):
        return {"boxes": boxes, "named": named, "why": why}

    def test_fuse_match_reset(self):
        bc = BASE["perception"]["boxes"]
        tr = BoxTracker(bc, Boxes.from_scene(SC))
        self.assertFalse(any(tr.boxes.ok(n) for n in tr.boxes.names))
        mk = [Box(n, (0.6, y), 0.1, 0.1075, 0.1075, 0.1, 0.004, conf=0.9, ok=True, src="detected")
              for n, y in (("left", 0.24), ("center", 0.0), ("right", -0.24))]
        tr.update(self._res(mk))
        moved = [Box(b.name, (b.c[0] + 0.004, b.c[1]), b.yaw, b.hu, b.hv, b.rim, b.wall, conf=0.9, ok=True,
                     src="detected") for b in mk]
        B = tr.update(self._res(moved))
        self.assertAlmostEqual(B["left"].c[0], 0.602, places=6)           # averaged
        unnamed = [Box(None, (0.61, 0.0), 0.1, 0.1075, 0.1075, 0.1, 0.004, conf=0.9, ok=False, src="detected")]
        B = tr.update(self._res(unnamed, named=False, why="1 box: name uncertain"))
        self.assertTrue(B.ok("center") and B.ok("left"))                   # matched by position; left kept
        far = [Box("left", (0.70, 0.24), 0.1, 0.1075, 0.1075, 0.1, 0.004, conf=0.9, ok=True, src="detected")]
        B = tr.update(self._res(far))
        self.assertAlmostEqual(B["left"].c[0], 0.70)                       # re-placed (logged)
        self.assertTrue(tr.events and "re-placed" in tr.events[0])
        tr.reset()
        self.assertFalse(tr.boxes.ok("left"))


class TestDownstream(unittest.TestCase):
    """Planner and perception masks with detected (moved, yawed, taller) boxes."""

    @classmethod
    def setUpClass(cls):
        cfg = load_config(profile="none")
        cls.cfg, cls.post, cls.home = cfg, hand_postures(cfg), home_arm(cfg)
        cls.pl = Planner(cfg, SC, cls.post, cls.home)
        cls.rows = box_row(0.43, 8, 0.08, 0.02, size=SIZE, wall_t=0.004)
        cls.B = as_boxes(cls.rows)

    def test_slots_inside_the_yawed_box(self):
        pl, s = self.pl, self.cfg["grasp"]["slots"]
        pl.set_boxes(self.B)
        for b, side in (("left", "left"), ("center", "left"), ("center", "right"), ("right", "right")):
            with self.subTest(bin=b, side=side):
                S = pl.slots(b, side)
                self.assertGreater(len(S), 0)
                L = self.B[b].local(S)
                lim = self.B[b].inner - s["wall_clear"] + 1e-9
                self.assertTrue(np.all(np.abs(L) <= lim), L)
                self.assertTrue(all(pl._slot_reach(side, p, self.B[b].rim) for p in S))

    def test_slots_follow_the_box(self):
        pl = self.pl
        pl.set_boxes(self.B)
        S0 = pl.slots("left", "left")
        nudged = Boxes(dict(self.B.named), source="detected")
        b = self.B["left"]
        nudged.named["left"] = Box("left", (b.c[0] + 0.002, b.c[1]), b.yaw, b.hu, b.hv, b.rim, b.wall, ok=True)
        pl.set_boxes(nudged)
        self.assertIs(pl.slots("left", "left"), S0)                         # within box_tol: kept
        moved = Boxes(dict(self.B.named), source="detected")
        moved.named["left"] = Box("left", (b.c[0] - 0.02, b.c[1] + 0.01), b.yaw, b.hu, b.hv, b.rim, b.wall, ok=True)
        pl.set_boxes(moved)
        S1 = pl.slots("left", "left")
        self.assertTrue(np.all(np.abs(moved["left"].local(S1)) <= moved["left"].inner))
        self.assertGreater(np.abs(S1[0] - S0[0]).max(), 0.009)
        pl.set_boxes(self.B)
        self.assertIs(pl.slots("left", "left"), S0)                         # back: the earlier slots again

    def test_plan_uses_rim_and_refuses_unknown_box(self):
        pl, zc, g = self.pl, SC.counter_z, self.cfg["grasp"]
        pl.set_boxes(self.B)
        d = Detection("red", np.array([0.30, 0.15, zc + pl.height]), 400, 400, "depth")
        p = pl.plan(d, self.home, self.post["open"], bin_name="left")
        self.assertTrue(p.ok, p.reason)
        self.assertEqual(p.safety_clips, 0)
        self.assertAlmostEqual(p.waypoints["drop"][2], zc + 0.08 + pl.height / 2 + g["drop_above_rim"])
        self.assertTrue(self.B["left"].inside(p.waypoints["drop"][:2]))
        lost = Boxes(dict(self.B.named), source="detected")
        b = self.B["left"]
        lost.named["left"] = Box("left", b.c, b.yaw, b.hu, b.hv, b.rim, b.wall, ok=False)
        pl.set_boxes(lost)
        p = pl.plan(d, self.home, self.post["open"], bin_name="left")
        self.assertFalse(p.ok)
        self.assertIn("left box not found", p.reason)
        pl.set_boxes(Boxes.from_scene(SC))

    def test_masks_follow_the_detected_boxes(self):
        """An object on the floor of a moved box is in that box (not an object to pick), the moved box's walls are
        no objects, the objects in front of it are found."""
        rows = box_row(0.53, 10, 0.10, 0.02, size=SIZE, wall_t=0.004)
        b = rows[0]
        inside = dict(color="red", shape="cube", yaw=0.2, size=0.05, height=0.05,
                      pos=np.r_[b["c"] + [0.03, -0.02], SC.counter_z + 0.003 + 0.025])
        band = objects(np.random.default_rng(4), rows, 2, 0)
        rgb, depth = Renderer(SC, CAL, boxes=rows, counter_size=TABLE).render([inside] + band,
                                                                                np.random.default_rng(4), 2.0, 0.03)
        p = Perception(BASE, SC, CAL)
        dets = p.detect(rgb, depth)
        self.assertTrue(p.boxes.ok("left"), p.boxes.text())
        self.assertEqual(len(dets), 2, dets)
        for o in band:
            self.assertTrue(any(np.linalg.norm(d.top[:2] - o["pos"][:2]) < 0.01 for d in dets), dets)


# ---------------------------------------------------------------------- end-to-end mock episodes
def run_episode(seed, rows, profile="none", overrides=None, n_max=None, **env_kw):
    cfg = load_config(profile=profile, overrides=overrides)
    if profile == "none":   # held fingers stop 0.3 rad short of the base cage's closed posture (verify: >= 0.12 held)
        env_kw.setdefault("held_gap", 0.3)
    env = make_mock_env(seed=seed, cfg=cfg, boxes=rows, counter_size=TABLE, **env_kw)
    pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
    obs = env.reset()
    if n_max is not None:                           # (shorter episodes)
        env.backend.objects = env.backend.objects[:n_max]
        obs = env._obs()
    pol.reset({"instruction": obs["instruction"]})
    while True:
        obs, info = env.step(pol.act(obs))
        if info["episode_over"]:
            break
    return pol, env, info


class TestEpisodes(unittest.TestCase):
    """Robot profile (detection on) in the mock world with boxes elsewhere than scene.yaml."""

    def _clean(self, info):
        self.assertIn(info["end_reason"], ("done", "time_limit"))
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])

    def test_moved_yawed_taller_boxes(self):
        """Row 3 cm further, yawed 8 deg, 8 cm walls, 2 cm gaps: every object ends in its box."""
        rows = box_row(0.43, 8, 0.08, 0.02, size=SIZE, wall_t=0.004)
        pol, env, info = run_episode(0, rows, n_max=3)
        self._clean(info)
        out = env.backend.outcomes(pol.mapping)
        self.assertEqual(out, ["correct"] * len(out), [m for _, m in pol.events])
        look = next(m for _, m in pol.events if m.startswith("look"))
        self.assertIn("; boxes L (", look)

    def test_no_boxes_nothing_released(self):
        """No boxes in view: nothing is picked (no guess at scene.yaml's places), the episode ends after the extra
        looks."""
        pol, env, info = run_episode(1, [], n_max=3)
        self._clean(info)
        self.assertEqual(env.backend.grasp_log, [])
        self.assertEqual(set(env.backend.outcomes(pol.mapping)), {"counter"})
        self.assertTrue(any("whose box was not found" in m for _, m in pol.events))
        self.assertTrue(any(m.startswith("looking again for the boxes") for _, m in pol.events))

    def test_one_box_missing(self):
        """The left box is not there: objects for it stay, the others are delivered; a single box off centre: its
        name is uncertain, nothing is delivered."""
        rows = box_row(0.43, 0, 0.08, 0.02, size=SIZE, wall_t=0.004, names=("center", "right"))
        pol, env, info = run_episode(2, rows)
        self._clean(info)
        for o, out in zip(env.backend.objects, env.backend.outcomes(pol.mapping)):
            self.assertEqual(out, "counter" if pol.mapping.get(o["color"]) == "left" else "correct", o["color"])
        rows = box_row(0.43, 0, 0.08, 0.02, size=SIZE, wall_t=0.004, y0=0.12, names=("center",))
        pol, env, info = run_episode(2, rows, n_max=3)
        self._clean(info)
        self.assertEqual(env.backend.grasp_log, [])
        self.assertNotIn("wrong", env.backend.outcomes(pol.mapping))


# ---------------------------------------------------------------------- reports (not unit tests)
def grid(log=print, seeds=1):
    """The synthetic grid: front edge x yaw x wall x gap x camera pitch error, objects in front and in the boxes."""
    rows, t = [], []
    for front in (0.40, 0.46, 0.50, 0.56, 0.62):
        for yaw in (-10, 0, 14):
            for wall in (0.06, 0.10):
                for gap in (0.02, 0.05):
                    for pitch in (0.0, 2.8):
                        for s in range(seeds):
                            p, _, _, errs, missing, wrong = detect_case(front, yaw, wall, gap, pitch,
                                                                        seed=hash((front, yaw, wall, gap, pitch, s))
                                                                        % 2 ** 31)
                            t.append(p.box_det.get("ms", 0.0))
                            rows.append(dict(case=(front, yaw, wall, gap, pitch), errs=errs, missing=missing,
                                             wrong=wrong, why=p.boxes.why, lv=p.level.get("pitch_deg")))
                            big = {n: np.round(e, 1).tolist() for n, e in errs.items()
                                   if e[0] > 5 or abs(e[1]) > 1 or abs(e[2]) > 4}
                            if missing or wrong or big:
                                res = p.box_det.get("res") or {}
                                rej = "; ".join(f"{s['c']} {s['why']}" for s in res.get("cands", []) if not s["ok"]
                                                and (s.get("cover") or {}).get("front", 0) > 0.3)
                                log(f"  {front:.2f} {yaw:+3d} {wall:.2f} {gap:.2f} {pitch:.1f}: missing {missing} "
                                    f"wrong {wrong} large {big} ({p.boxes.why}); rejected: {rej}")
    E = np.array([e for r in rows for e in r["errs"].values()])
    n = 3 * len(rows)
    log(f"{len(rows)} layouts, {n} boxes: found + named {len(E)}, missing {sum(len(r['missing']) for r in rows)}, "
        f"WRONG {sum(len(r['wrong']) for r in rows)}; box ms mean {np.mean(t):.1f} p95 {np.percentile(t, 95):.1f}")
    for j, nm in enumerate(("centre |dxy| mm", "yaw deg", "rim mm", "side mm")):
        a = np.abs(E[:, j])
        log(f"  {nm:<16} mean {a.mean():5.2f}  p95 {np.percentile(a, 95):5.2f}  worst {a.max():5.2f}")
    for key, i, vals in (("front", 0, (0.40, 0.46, 0.50, 0.56, 0.62)), ("yaw", 1, (-10, 0, 14)),
                         ("pitch", 4, (0.0, 2.8)), ("wall", 2, (0.06, 0.10)), ("gap", 3, (0.02, 0.05))):
        for v in vals:
            sub = [r for r in rows if r["case"][i] == v]
            e = np.array([e for r in sub for e in r["errs"].values()]).reshape(-1, 4)
            log(f"  {key} {v}: found {len(e)}/{3 * len(sub)}, centre p95 {np.percentile(e[:, 0], 95) if len(e) else 0:.2f}"
                f" mm, yaw p95 {np.percentile(np.abs(e[:, 1]), 95) if len(e) else 0:.2f} deg, rim p95 "
                f"{np.percentile(np.abs(e[:, 2]), 95) if len(e) else 0:.2f} mm")
    return rows


def reach(log=print):
    """Drop slots left (count per box / arm) as the row moves away and the walls grow, base and sim profiles."""
    for prof in ("none", "sim"):
        cfg = load_config(profile=prof)
        pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
        log(f"profile {prof}: slots.max_tilt_deg {cfg['grasp']['slots']['max_tilt_deg']}")
        for front, wall, yaw in ((0.40, 0.06, 0), (0.40, 0.10, 0), (0.43, 0.10, 0), (0.46, 0.10, 0), (0.50, 0.10, 0),
                                 (0.56, 0.10, 0), (0.50, 0.10, 14), (0.56, 0.10, 14)):
            pl.set_boxes(as_boxes(box_row(front, yaw, wall, 0.02, size=SIZE, wall_t=0.004)))
            part = [f"{b[0]}/{s[0]} {len(pl.slots(b, s))}" for b, s in (("left", "left"), ("center", "left"),
                                                                           ("center", "right"), ("right", "right"))]
            log(f"  front {front:.2f} wall {wall:.2f} yaw {yaw:+d}: " + ", ".join(part))


def frames(dirs, log=print):
    """Recorded sim frames: detected boxes vs scene.yaml (profile sim: registered; and levelled only, as on the
    robot: then the sim's camera pose error is in it)."""
    from pathlib import Path
    truth = Boxes.from_scene(SC)
    arms = arm_spheres("sim")
    for tag, ov in (("sim (registered)", {}), ("levelled only", {"perception.register.enabled": False,
                                                                 "perception.boxes.mode": "on"})):
        p = Perception(load_config(profile="sim", overrides=ov), SC, CAL)
        p.arm_spheres = arms
        for d in dirs:
            E, named, n, t = [], 0, 0, []
            for f in sorted(Path(d).glob("*.npz")):
                z = np.load(f)
                p.reset_boxes()
                p.detect(z["rgb"], z["depth"])
                res = p.box_det["res"] or {"boxes": [], "named": False}
                n += 1
                named += bool(res["named"]) and len(res["boxes"]) == 3
                t.append(p.box_det["ms"])
                for b in res["boxes"]:
                    if b.name:
                        tb = truth[b.name]
                        E.append([1000 * np.linalg.norm(np.subtract(b.c, tb.c)), np.degrees(b.yaw),
                                  1000 * (b.rim - tb.rim)])
            E = np.abs(np.array(E))
            log(f"{tag} {d}: {n} frames, all three named {named}; centre mm mean {E[:, 0].mean():.2f} p95 "
                f"{np.percentile(E[:, 0], 95):.2f} worst {E[:, 0].max():.2f}; yaw deg p95 {np.percentile(E[:, 1], 95):.2f}"
                f" worst {E[:, 1].max():.2f}; rim mm p95 {np.percentile(E[:, 2], 95):.2f} worst {E[:, 2].max():.2f};"
                f" ms mean {np.mean(t):.1f} p95 {np.percentile(t, 95):.1f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if what == "grid":
        grid()
    elif what == "reach":
        reach()
    elif what == "frames":
        frames(sys.argv[2:] or ["/data/aaro/frames/nominal", "/data/aaro/frames/train"])
    else:
        print(json.dumps({"usage": __doc__}))
