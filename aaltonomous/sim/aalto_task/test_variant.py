"""Offline tests for sim variants (variant.py) and the evaluator's colour-word renaming (no Isaac):
python -m unittest aaltonomous/sim/aalto_task/test_variant.py"""
import colorsys
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aaltonomous.interface import tasks
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene
from aaltonomous.sim.aalto_task import dr, layout as L, variant as V

# upstream dex3_state.get_robot_girl_joint_names() (the published state order), copied from the sim image
UPSTREAM_STATE_NAMES = [f"{s}_hand_{f}_joint" for s in ("left", "right")
                        for f in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]


def _write(d):
    import yaml
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    return f.name


class TestUnset(unittest.TestCase):
    """No variant = the organisers' sim: every helper returns its input / upstream's value."""

    def test_load_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AALTO_VARIANT", None)
            self.assertIsNone(V.load())
        for ref in ("", "off"):
            self.assertIsNone(V.load(ref))
        self.assertIsNone(V.summary(None))

    def test_identity(self):
        sc = Scene.load()
        self.assertEqual(V.scene(None).cfg, sc.cfg)
        for c in sc.colors:   # exactly scene_cfg's old expression
            self.assertEqual(V.object_rgb(None, sc, c), tuple(x / 255.0 for x in sc.colors[c]))
        self.assertEqual(V.color_override(None, "counter", (0.55, 0.45, 0.35)), (0.55, 0.45, 0.35))
        self.assertEqual(V.light(None, "key", "intensity", 60000.0), 60000.0)
        self.assertIsNone(V.mat_tint(None))
        T = np.eye(4)
        self.assertIs(V.camera_T(None, T), T)
        self.assertEqual(V.soft_limit_factor(None, 0.9), 0.9)
        self.assertEqual(V.wiring(None), "upstream")
        self.assertEqual(V.actuator_overrides(None, None), [])
        self.assertEqual(V.rename(tasks.FIXED_INSTRUCTION, {}), tasks.FIXED_INSTRUCTION)

    def test_hand_gains_alone(self):
        # AALTO_HAND_GAINS without a variant: the old hook's semantics (all hand joints, given keys only)
        o = V.actuator_overrides(None, "1.5,0.2,0.002,1.4")
        self.assertEqual(o, [("hands", {"stiffness": 1.5, "damping": 0.2, "armature": 0.002, "effort_limit": 1.4})])
        act = SimpleNamespace(stiffness={".*": 100.0}, damping={".*": 10.0}, armature={".*": 0.1}, effort_limit=300,
                              joint_effort_limit=None)
        V.apply_actuator(act, *o[0])
        self.assertEqual((act.stiffness, act.damping, act.armature), ({".*": 1.5}, {".*": 0.2}, {".*": 0.002}))
        self.assertEqual((act.joint_effort_limit, act.effort_limit), ({".*": 1.4}, None))
        act = SimpleNamespace(stiffness=None, damping=None, armature={".*": 0.1})   # 2 values: armature kept
        V.apply_actuator(act, *V.actuator_overrides(None, "2,0.3")[0])
        self.assertEqual((act.stiffness, act.damping, act.armature), ({".*": 2.0}, {".*": 0.3}, {".*": 0.1}))


class TestRealLike(unittest.TestCase):
    def setUp(self):
        self.v = V.load("real_like")
        self.spec = JointSpec()

    def test_loads(self):
        v = self.v
        self.assertEqual(v["name"], "real_like")
        self.assertEqual(V.wiring(v), "spec")
        self.assertEqual(V.soft_limit_factor(v, 0.9), 1.0)
        self.assertEqual(v["color_words"], {"red": "orange", "blue": "purple"})
        json.dumps(V.summary(v))                                  # scene service info is JSON

    def test_arm_and_hand_gains(self):
        A = dict(V.actuator_overrides(self.v))
        for n in self.spec.arm_names:
            wrist = "wrist" in n
            self.assertEqual(V.resolve(A["arms"]["stiffness"], n), 40.0 if wrist else 80.0)
            self.assertEqual(V.resolve(A["arms"]["damping"], n), 1.5 if wrist else 3.0)
        for n in self.spec.hand_names:
            self.assertEqual(V.resolve(A["hands"]["stiffness"], n), 1.5)
            self.assertEqual(V.resolve(A["hands"]["damping"], n), 0.2)
            self.assertEqual(V.resolve(A["hands"]["effort_limit"], n), 2.45 if "thumb_0" in n else 1.4)
        # effort limits = joint_spec (URDF) values
        import yaml
        spec = yaml.safe_load(V.SPEC.read_text())
        for j in spec["arm"] + spec["hand"]:
            g = "arms" if j in spec["arm"] else "hands"
            self.assertEqual(V.resolve(A[g]["effort_limit"], j["name"]), j["effort"], j["name"])

    def test_hand_gains_env_wins(self):
        o = V.actuator_overrides(self.v, "3,0.5")
        self.assertEqual(o[-1], ("hands", {"stiffness": 3.0, "damping": 0.5}))
        act = SimpleNamespace(stiffness=None, damping=None, armature=None, effort_limit=300, joint_effort_limit=None)
        for g, p in o:
            if g == "hands":
                V.apply_actuator(act, g, p)
        self.assertEqual((act.stiffness, act.damping), ({".*": 3.0}, {".*": 0.5}))
        self.assertEqual(act.armature, {".*": 0.002})            # the variant's, not overridden by 2 values
        self.assertEqual(act.joint_effort_limit[".*_hand_thumb_0_joint"], 2.45)


class TestValidate(unittest.TestCase):
    def bad(self, d):
        with self.assertRaises(ValueError):
            V.load(_write(d))

    def test_rejects(self):
        self.bad({"robots": {}})                                                  # typo
        self.bad({"robot": {"right_hand_wiring": "mirror"}})
        self.bad({"robot": {"actuators": {"legs": {"stiffness": 1}}}})
        self.bad({"robot": {"actuators": {"arms": {"stifness": 1}}}})
        self.bad({"robot": {"actuators": {"arms": {"stiffness": {".*_shoulder_.*_joint": 80}}}}})   # elbow, wrist
        self.bad({"robot": {"actuators": {"hands": {"damping": {".*": 1, ".*_thumb_0_joint": 2}}}}})  # twice
        self.bad({"colors": {"objects": {"purple": [1, 2, 3]}}})                  # keys are internal names
        self.bad({"colors": {"bins": [1, 2]}})
        self.bad({"color_words": {"red": "yellow"}})                              # two colours would be "yellow"
        self.bad({"color_words": {"green": "lime"}})
        self.bad({"color_words": {"red": "dark red"}})
        self.bad({"scene": {"bins": {"centres": {}}}})                            # not in scene.yaml
        self.bad({"lighting": {"sun": {}}})

    def test_waist_group(self):
        self.assertEqual(V.spec_names("waist"), ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
        v = V.load(_write({"robot": {"actuators": {"waist": {"stiffness": 300.0, "damping": 3.0}}}}))
        self.assertEqual(V.actuator_overrides(v), [("waist", {"stiffness": 300.0, "damping": 3.0})])
        self.bad({"robot": {"actuators": {"waist": {"stiffness": {"waist_yaw_joint": 300.0}}}}})

    def test_minimal(self):
        v = V.load(_write({"color_words": {"red": "blue", "blue": "red"}}))        # a swap is fine
        self.assertEqual(v["name"], Path(v["_path"]).stem)
        self.assertEqual(V.actuator_overrides(v), [])
        self.assertEqual(V.wiring(v), "upstream")


class TestWiring(unittest.TestCase):
    def test_tables(self):
        up, spec = V.hand_slots("upstream"), V.hand_slots("spec")
        self.assertEqual(up["left"], spec["left"])                                # left: sim == robot
        diff = {n for n in spec["right"] if spec["right"][n] != up["right"][n]}
        self.assertEqual(diff, {f"right_hand_{f}_joint" for f in ("index_0", "index_1", "middle_0", "middle_1")})
        self.assertEqual(spec["right"]["right_hand_index_0_joint"], 3)
        self.assertEqual(up["right"]["right_hand_index_0_joint"], 5)

    def test_state_order(self):
        self.assertEqual(V.state_order("upstream"), UPSTREAM_STATE_NAMES)
        # spec wiring publishes exactly the interface's hand_q order (JointSpec.hand_names = AaltoEnv hand_q)
        self.assertEqual(V.state_order("spec"), JointSpec().hand_names)

    def test_command_path(self):
        """Mimic DDSActionProvider.get_action: joint (mapping key) <- message[slot]."""
        names = JointSpec().hand_names
        for w in ("upstream", "spec"):
            slots = V.hand_slots(w)["right"]
            for k in range(7):
                msg = np.zeros(7)
                msg[k] = 1.0
                moved = [n for n in UPSTREAM_STATE_NAMES[7:] if msg[slots[n]] == 1.0]
                self.assertEqual(len(moved), 1)
                if w == "spec":
                    self.assertEqual(moved[0], names[7 + k])
        self.assertEqual(len(V.UPSTREAM_STATE_IDX), 14)


class TestRender(unittest.TestCase):
    def test_object_colors_and_dr(self):
        v, sc = V.load("real_like"), Scene.load()
        self.assertEqual(L.object_names(V.scene(v)), L.object_names(sc))         # names stay internal
        base = V.object_rgb(v, sc, "red")
        self.assertEqual(base, tuple(c / 255 for c in v["colors"]["objects"]["red"]))
        # DR (hooks._apply_dr) jitters the colour read from the shader = the variant's render colour
        cfg = dr.load(str(Path(dr.DR_DIR) / "train.yaml"))
        P = dr.sample(cfg, 7, L.object_names(sc), sc.bin_names)
        o = P["objects"]["obj_red_cube_0"]
        new = dr.jitter_color(base, o["dh"], o["s"], o["v"])
        h_new, h_base = colorsys.rgb_to_hsv(*new)[0], colorsys.rgb_to_hsv(*base)[0]
        h_red = colorsys.rgb_to_hsv(*V.object_rgb(None, sc, "red"))[0]
        self.assertLess(abs(h_new - h_base), cfg["objects"]["hue_jitter"] + 1e-9)
        self.assertGreater(abs(h_new - h_red), 0.03)                              # orange, not jittered red
        np.testing.assert_allclose(dr.jitter_color(base, 0.0, 1.0, 1.0), base, atol=1e-12)   # DR off

    def test_camera_offset(self):
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.0, 0.4]
        T2 = V.camera_T({"head_camera": {"pos_offset": [0.0, 0.01, -0.02], "rot_offset_deg": [0, 0, 90]}}, T)
        np.testing.assert_allclose(T2[:3, 3], [0.1, 0.01, 0.38])
        np.testing.assert_allclose(T2[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3], np.eye(3))                          # input untouched

    def test_scene_overlay_scores_new_bins(self):
        sc0 = Scene.load()
        v = V.load(_write({"scene": {"bins": {"centers": {"left": [0.75, 0.38]}}, "counter": {"grid_cm": 10}}}))
        sc = V.scene(v)
        self.assertEqual(sc.cfg["bins"]["centers"]["left"], [0.75, 0.38])
        self.assertEqual(sc.cfg["bins"]["centers"]["right"], sc0.cfg["bins"]["centers"]["right"])
        lo, hi = sc.bin_inner_box("left")
        p = sc.mat_to_pelvis((lo + hi) / 2 * np.array([1, 1, 0]) + [0, 0, 0.03])
        objs = [{"name": "a", "color": "red", "shape": "cube", "active": True, "pos": p, "speed": 0.0}]
        self.assertEqual(L.evaluate(sc, "sort_fixed", objs)["outcomes"][0]["outcome"], "correct")


class TestColorWords(unittest.TestCase):
    W = {"red": "orange", "blue": "purple"}

    def test_rename(self):
        self.assertEqual(V.rename("Red goes left, RED too; redo the blue ones. Blue!", self.W),
                         "Orange goes left, ORANGE too; redo the purple ones. Purple!")
        self.assertEqual(V.rename("red to blue", {"red": "blue", "blue": "red"}), "blue to red")   # simultaneous
        self.assertEqual(V.rename(tasks.FIXED_INSTRUCTION, self.W),
                         "Put the orange objects in the left bin, the yellow objects in the center bin "
                         "and the purple objects in the right bin.")
        self.assertEqual(V.rename(tasks.INSERT_INSTRUCTION, self.W), tasks.INSERT_INSTRUCTION)

    def test_templates_consistent(self):
        """rename(instruction in internal words) == the same template written with the told words, for every
        template x colour order x bin order; the inverse gives the original back; and the mapping the told text
        states, translated back, is the internal mapping the evaluator scores with."""
        inv = V.inverse(self.W)
        for tpl in tasks.TEMPLATES:
            for colors in itertools.permutations(tasks.COLORS):
                for bins in itertools.permutations(tasks.BINS):
                    w = {f"c{i}": c for i, c in enumerate(colors)}
                    w.update({f"C{i}": c.capitalize() for i, c in enumerate(colors)})
                    w.update({f"b{i}": b for i, b in enumerate(bins)})
                    text = tpl.format(**w)
                    told = [self.W.get(c, c) for c in colors]
                    t = {f"c{i}": c for i, c in enumerate(told)}
                    t.update({f"C{i}": c.capitalize() for i, c in enumerate(told)})
                    t.update({f"b{i}": b for i, b in enumerate(bins)})
                    self.assertEqual(V.rename(text, self.W), tpl.format(**t))
                    self.assertEqual(V.rename(V.rename(text, self.W), inv), text)
                    stated = dict(zip(told, bins))                              # what a reader gets from the text
                    self.assertEqual({inv.get(k, k): b for k, b in stated.items()}, dict(zip(colors, bins)))
                    self.assertEqual(V.told_mapping(dict(zip(colors, bins)), self.W), stated)

    def test_evaluator(self):
        from aaltonomous.interface.env import AaltoEnv
        from aaltonomous.sim import evaluate as E
        info = {"variant": V.summary(V.load("real_like")), "dr": None}
        self.assertEqual(E.color_words("auto", info), self.W)
        self.assertEqual(E.color_words("auto", {"dr": None, "variant": None}), {})
        self.assertEqual(E.color_words("off", info), {})
        self.assertEqual(E.color_words("red=orange,blue=purple", {}), self.W)
        with self.assertRaises(ValueError):
            E.color_words("red=yellow", {})
        for task in ("sort_fixed", "sort_language", "insert_shapes"):
            for seed in range(12):
                a = AaltoEnv(backend="mock", task=task, seed=seed)
                b = AaltoEnv(backend="mock", task=task, seed=seed)
                E.rename_instructions(b, self.W)
                oa, ob = a.reset(), b.reset()
                self.assertEqual(ob["instruction"], V.rename(oa["instruction"], self.W))
                self.assertEqual(b.instruction, ob["instruction"])
                self.assertEqual((b.mapping, b.layout), (a.mapping, a.layout))   # scoring + layout: internal
                self.assertNotRegex(ob["instruction"].lower(), r"\b(red|blue)\b")
                self.assertEqual(b._obs()["instruction"], ob["instruction"])      # every later obs too
        E.rename_instructions(a, {})                                             # no-op
        self.assertEqual(a.reset.__func__, AaltoEnv.reset)
        # --policy oracle gets the internal words back
        seen = []
        c = E.LocalClient(SimpleNamespace(reset=seen.append), words=V.inverse(self.W))
        c.reset("sort_fixed", V.rename(tasks.FIXED_INSTRUCTION, self.W), 0)
        self.assertEqual(seen, [{"instruction": tasks.FIXED_INSTRUCTION}])
        c = E.LocalClient(SimpleNamespace(reset=seen.append))
        c.reset("sort_fixed", "x red", 0)
        self.assertEqual(seen[-1], {"instruction": "x red"})


# ---------------------------------------------------------------------------------------------- R2: drop-2 setup
from aaltonomous.sim.aalto_task import boxes as B   # noqa: E402

D2 = ("real_d2", "real_d2_far", "real_d2_near")


def _old_bin_parts(sc, name):
    """scene_cfg.bin_cfgs before R2, verbatim: (part, centre pelvis, size pelvis)."""
    b = sc.cfg["bins"]
    ox, oy = b["outer_size"]
    t, ft, h = b["wall_thickness"], b["floor_thickness"], b["wall_height"]
    cx, cy = b["centers"][name]
    parts = {"floor": ((cx, cy, ft / 2), (ox, oy, ft)),
             "wall_xlo": ((cx - ox / 2 + t / 2, cy, h / 2), (t, oy, h)),
             "wall_xhi": ((cx + ox / 2 - t / 2, cy, h / 2), (t, oy, h)),
             "wall_ylo": ((cx, cy - oy / 2 + t / 2, h / 2), (ox - 2 * t, t, h)),
             "wall_yhi": ((cx, cy + oy / 2 - t / 2, h / 2), (ox - 2 * t, t, h))}
    return [(k, sc.mat_to_pelvis(c), (s[1], s[0], s[2])) for k, (c, s) in parts.items()]


def _old_counter(sc, th=0.04):
    c = sc.cfg["counter"]
    mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
    return sc.mat_to_pelvis([mat[0] / 2, mat[1] / 2, -th / 2]), (size[1], size[0], th)


def _in_box(sc, name, local_xy, z):
    """Mat point at box-frame xy (x depth, y width) of a VariantScene box, height z above the counter."""
    x, y, th = sc.poses[name]
    c, s = np.cos(th), np.sin(th)
    p = [x + c * local_xy[0] - s * local_xy[1], y + s * local_xy[0] + c * local_xy[1], sc.counter_z + z]
    return sc.pelvis_to_mat(np.array(p))


class TestUnsetR2(unittest.TestCase):
    """Without a variant (and with real_like, which has no boxes / table): every R2 path is the old one."""

    def test_plain_scene(self):
        for v in (None, V.load("real_like")):
            sc = V.scene(v)
            self.assertIs(type(sc), Scene)
            self.assertEqual(sc.cfg, Scene.load().cfg)
            self.assertEqual(V.joint_limits(v), {})
            self.assertEqual(V.object_mesh_scale(sc), (1.0, 1.0, 1.0))
            self.assertEqual(V.object_physics(v, "mass", 0.05), 0.05)
            self.assertEqual(V.table_roughness(v, 0.8), 0.8)
            for n in sc.bin_names:   # exactly the old sim geometry (same floats, same order, no yaw)
                new, old = B.bin_parts(sc, n), _old_bin_parts(sc, n)
                self.assertEqual([k for k, *_ in new], [k for k, *_ in old])
                for (_, c, s, yaw), (_, c0, s0) in zip(new, old):
                    self.assertEqual(tuple(c), tuple(c0))
                    self.assertEqual(tuple(float(x) for x in s), tuple(float(x) for x in s0))
                    self.assertEqual(yaw, 0.0)
            c, s = B.counter_box(sc, 0.04)
            c0, s0 = _old_counter(sc)
            self.assertEqual((tuple(c), tuple(s)), (tuple(c0), tuple(float(x) for x in s0)))
        self.assertEqual(V.color_override(V.load("real_like"), "bin_floor", (0.1, 0.2, 0.3)), (0.1, 0.2, 0.3))

    def test_camera_untouched(self):
        T = np.eye(4)
        self.assertIs(V.camera_T(V.load("real_like"), T), T)


class TestInherit(unittest.TestCase):
    def test_real_d2_files(self):
        base = V.load("real_like")
        for name, chain in (("real_d2", ["real_like"]), ("real_d2_far", ["real_d2", "real_like"]),
                            ("real_d2_near", ["real_d2", "real_like"])):
            v = V.load(name)
            self.assertEqual((v["name"], v["inherit"]), (name, chain))
            self.assertEqual(v["robot"]["actuators"], base["robot"]["actuators"])
            self.assertEqual(V.wiring(v), "spec")
            self.assertEqual(v["color_words"], base["color_words"])
            self.assertEqual(v["colors"]["objects"], base["colors"]["objects"])
            self.assertNotEqual(v["notes"], base["notes"])
            self.assertEqual(v["boxes"]["gap"], 0.02)                       # from real_d2 in all three
            json.dumps(V.summary(v))
        self.assertEqual(V.load("real_d2_far")["boxes"]["front_x"], 0.56)
        self.assertEqual(V.load("real_d2_near")["boxes"]["yaw_deg"], 7.0)

    def test_merge_and_cycles(self):
        self.assertEqual(V.deep_merge({"a": {"b": 1, "c": [1, 2]}, "d": 1}, {"a": {"c": [3]}, "e": 2}),
                         {"a": {"b": 1, "c": [3]}, "d": 1, "e": 2})
        d = Path(tempfile.mkdtemp())
        import yaml
        (d / "a.yaml").write_text(yaml.safe_dump({"inherit": "b.yaml", "mat": False}))
        (d / "b.yaml").write_text(yaml.safe_dump({"inherit": "a.yaml"}))
        with self.assertRaises(ValueError):
            V.load(str(d / "a.yaml"))
        (d / "b.yaml").write_text(yaml.safe_dump({"name": "bee", "notes": "x", "mat": True,
                                                  "color_words": {"red": "orange"}}))
        v = V.load(str(d / "a.yaml"))                                         # relative to the child's folder
        self.assertEqual((v["name"], v.get("notes"), v["mat"], v["inherit"]), ("a", None, False, ["b"]))
        self.assertEqual(v["color_words"], {"red": "orange"})


class TestBoxes(unittest.TestCase):
    def test_row_geometry(self):
        D, W = 0.21, 0.21
        sc = V.scene(V.load("real_d2"))
        self.assertIsInstance(sc, B.VariantScene)
        self.assertEqual(sc.cfg["bins"]["wall_height"], 0.10)
        for n, y in (("right", -(W + 0.02)), ("center", 0.0), ("left", W + 0.02)):
            np.testing.assert_allclose(sc.poses[n], (0.50 + D / 2, y, 0.0), atol=1e-12)
            np.testing.assert_allclose(sc.mat_to_pelvis([*sc.cfg["bins"]["centers"][n], 0])[:2], sc.poses[n][:2],
                                       atol=1e-12)                             # cfg centres follow (mat coords)
        self.assertEqual(list(sc.cfg["bins"]["centers"]), list(Scene.load().cfg["bins"]["centers"]))   # order kept
        for name, front, yaw in (("real_d2_far", 0.56, -10.0), ("real_d2_near", 0.46, 7.0)):
            sc = V.scene(V.load(name))
            r = B.report(sc)
            np.testing.assert_allclose(r["boxes"]["center"]["front_mid"], [front, 0.0], atol=1e-4)
            for n in sc.bin_names:
                self.assertAlmostEqual(np.degrees(sc.poses[n][2]), yaw)
            self.assertAlmostEqual(B.poly_gap(sc.footprint("left"), sc.footprint("center")), 0.02, places=9)
            self.assertAlmostEqual(B.poly_gap(sc.footprint("center"), sc.footprint("right")), 0.02, places=9)
            # CCW sign: + = the left end nearer
            self.assertEqual(r["boxes"]["left"]["near_x"] < r["boxes"]["right"]["near_x"], yaw > 0)
            self.assertEqual(r["problems"], [])

    def test_yaw0_equals_scene(self):
        """A VariantScene at the scene.yaml poses scores exactly like scene.py (and its inner box is the same)."""
        sc0 = Scene.load()
        sc = V.scene(V.load(_write({"boxes": {}})))
        self.assertIsInstance(sc, B.VariantScene)
        rng = np.random.default_rng(0)
        b = sc0.cfg["bins"]
        for n in sc0.bin_names:
            for a, c in zip(sc.bin_inner_box(n), sc0.bin_inner_box(n)):
                np.testing.assert_allclose(a, c, atol=1e-12)
            for (k, c, s, yaw), (_, c0, s0) in zip(B.bin_parts(sc, n), _old_bin_parts(sc0, n)):
                np.testing.assert_allclose(c, c0, atol=1e-12)
                np.testing.assert_allclose(s, s0, atol=1e-12)
            lo, hi = sc0.bin_inner_box(n)
            for _ in range(300):                       # points around the bin, away from FP-exact boundaries
                p = rng.uniform(lo - 0.03, hi + 0.03)
                if np.min(np.abs(np.r_[p - lo, p - hi])) < 1e-6:
                    continue
                self.assertEqual(sc.bin_of(p), sc0.bin_of(p))
                pp = sc.mat_to_pelvis(p)
                for v in (0.0, 0.05):
                    self.assertEqual(sc.classify_cube(pp, v, "red", tasks.FIXED_MAPPING),
                                     sc0.classify_cube(pp, v, "red", tasks.FIXED_MAPPING))
        self.assertEqual(b["centers"].keys(), sc.cfg["bins"]["centers"].keys())

    def test_yawed_membership_by_hand(self):
        """scene.yaml row yawed 10 deg about the centre box's front-wall middle (0.40, 0): inner half-width
        a = 0.105 - 0.003. Points written out in the box frame, scored through scene.py's classify_cube."""
        v = V.load(_write({"boxes": {"front_x": 0.40, "yaw_deg": 10.0}}))
        sc = V.scene(v)
        x, y, th = sc.poses["center"]
        self.assertAlmostEqual(th, np.radians(10.0))
        np.testing.assert_allclose([x, y], [0.40 + 0.105 * np.cos(th), 0.105 * np.sin(th)], atol=1e-12)
        a, c, s = 0.102, np.cos(np.radians(10.0)), np.sin(np.radians(10.0))
        M = dict(tasks.FIXED_MAPPING)                 # red -> left, yellow -> center, blue -> right
        z = 0.03

        def P(lx, ly):                                 # box frame -> pelvis, written out
            return np.array([x + c * lx - s * ly, y + s * lx + c * ly, sc.counter_z + z])
        cases = [((0.0, 0.0), "center"), ((0.0, a - 0.002), "center"), ((0.0, a + 0.002), None),
                 ((a - 0.002, 0.0), "center"), ((a + 0.002, 0.0), None), ((-a + 0.002, 0.0), "center"),
                 ((-a - 0.002, 0.0), None),
                 ((a - 0.003, a - 0.003), "center"),   # just inside a rotated corner: axis-aligned would say out
                 ((0.104, 0.05), None)]                # just outside the rotated far wall: axis-aligned says in
        for (lx, ly), want in cases:
            p = P(lx, ly)
            self.assertEqual(sc.bin_of(sc.pelvis_to_mat(p)), want, (lx, ly))
            self.assertEqual(sc.classify_cube(p, 0.0, "yellow", M), "correct" if want else "counter", (lx, ly))
            self.assertEqual(sc.classify_cube(p, 0.0, "red", M), "wrong" if want else "counter", (lx, ly))
        # the last two: scene.py's axis-aligned box (same centre) gets them the other way round
        hx = hy = a
        for (lx, ly), want_aa in (((a - 0.003, a - 0.003), False), ((0.104, 0.05), True)):
            d = P(lx, ly)[:2] - [x, y]
            self.assertEqual(bool(abs(d[0]) <= hx and abs(d[1]) <= hy), want_aa)
        p = P(0.0, 0.0)
        self.assertEqual(sc.classify_cube(p, 0.03, "yellow", M), "counter")               # not at rest
        self.assertEqual(sc.classify_cube(p + [0, 0, 0.031], 0.0, "yellow", M), "counter")   # above the rim (0.06)
        self.assertEqual(sc.classify_cube(p - [0, 0, 0.0285], 0.0, "yellow", M), "counter")  # below the floor top
        sc.cfg["success"]["inside_margin"] = 0.01
        self.assertIsNone(sc.bin_of(sc.pelvis_to_mat(P(0.0, a - 0.005))))
        self.assertEqual(sc.bin_of(sc.pelvis_to_mat(P(0.0, a - 0.015))), "center")

    def test_inner_box_conservative(self):
        for name in D2 + (_write({"boxes": {"yaw_deg": 25.0, "front_x": 0.50}, "table": {"x": [0.09, 1.0], "y": [-0.8, 0.8]}}),):
            sc = V.scene(V.load(name))
            for n in sc.bin_names:
                lo, hi = sc.bin_inner_box(n)
                ah = sc.inner_half()
                for cx in (lo[0], hi[0]):
                    for cy in (lo[1], hi[1]):
                        d = np.abs(sc.to_box(n, sc.mat_to_pelvis([cx, cy, 0.0])))
                        self.assertTrue(np.all(d <= ah + 1e-9), (name, n, d, ah))
                        self.assertTrue(np.any(d >= ah - 1e-9))        # largest: touches a wall
                self.assertEqual((lo[2], hi[2]), (sc.cfg["bins"]["floor_thickness"], sc.cfg["bins"]["wall_height"]))

    def test_table_and_off(self):
        sc, sc0 = V.scene(V.load("real_d2")), Scene.load()
        far = sc.pelvis_to_mat(np.array([0.95, 0.0, sc.counter_z + 0.03]))
        self.assertTrue(sc.on_counter(far))
        self.assertFalse(sc0.on_counter(sc0.pelvis_to_mat(np.array([0.95, 0.0, sc0.counter_z + 0.03]))))
        self.assertFalse(sc.on_counter(sc.pelvis_to_mat(np.array([1.05, 0.0, sc.counter_z + 0.03]))))
        self.assertFalse(sc.on_counter(sc.pelvis_to_mat(np.array([0.5, 0.0, sc.counter_z - 0.05]))))
        c, s = B.counter_box(sc, 0.04)
        np.testing.assert_allclose(c, [0.545, 0.0, sc.counter_z - 0.02])
        np.testing.assert_allclose(s, [0.91, 1.5, 0.04])
        # the far part of the left box (beyond scene.yaml's counter edge 0.69) is still "in the box"
        p = sc.mat_to_pelvis(_in_box(sc, "left", (0.09, 0.0), 0.03))
        self.assertGreater(p[0], 0.69)
        self.assertEqual(sc.classify_cube(p, 0.0, "red", dict(tasks.FIXED_MAPPING)), "correct")

    def test_per_box_and_centers(self):
        v = V.load(_write({"boxes": {"per_box": {"left": {"dx": 0.01, "dy": -0.005, "yaw_deg": 3.0}}}}))
        sc, sc0 = V.scene(v), V.scene(V.load(_write({"boxes": {}})))
        np.testing.assert_allclose(np.subtract(sc.poses["left"], sc0.poses["left"]), [0.01, -0.005, np.radians(3)],
                                   atol=1e-12)
        self.assertEqual(sc.poses["right"], sc0.poses["right"])
        v = V.load(_write({"boxes": {"centers": {"left": [0.55, 0.25], "center": [0.55, 0.0], "right": [0.57, -0.25]},
                                     "yaw_deg": 2.0}}))
        self.assertEqual(V.scene(v).poses["right"], (0.57, -0.25, np.radians(2.0)))

    def test_rejects(self):
        bad = TestValidate.bad
        bad(self, {"boxes": {"front": 0.5}})
        bad(self, {"boxes": {"per_box": {"middle": {"dx": 0.0}}}})
        bad(self, {"boxes": {"per_box": {"left": {"dz": 0.0}}}})
        bad(self, {"boxes": {"centers": {"left": [0.6, 0.2]}}})
        bad(self, {"boxes": {"gap": -0.01}})                                    # boxes overlap
        bad(self, {"boxes": {"front_x": 0.33}})                                 # on the object band
        bad(self, {"boxes": {"front_x": 0.60}})                                 # off scene.yaml's counter
        bad(self, {"boxes": {"front_x": 0.50}, "table": {"x": [0.09, 0.70]}})   # off the given table
        bad(self, {"table": {"x": [0.0, 1.0]}})                                 # into the robot
        bad(self, {"table": {"x": [0.5, 0.4]}})
        bad(self, {"table": {"roughness": 2.0}})
        bad(self, {"table": {"colour": [1, 2, 3]}})
        bad(self, {"object_physics": {"mass": 0.0}})
        bad(self, {"object_physics": {"density": 1.0}})
        bad(self, {"head_camera": {"rot_offset_optical_deg": [1.0, 2.0]}})
        bad(self, {"scene": {"objects": {"size": 0.0}}})

    def test_insert_yawed(self):
        sc = V.scene(V.load(_write({"boxes": {"yaw_deg": 8.0, "front_x": 0.42}})))
        x, y, th = sc.poses[sc.lid_bin]
        p = np.array([x, y, sc.counter_z + sc.cfg["bins"]["floor_thickness"] + 0.02])
        self.assertEqual(sc.classify_insert(p, 0.0, "cube"), "inserted")
        self.assertEqual(sc.poses[sc.lid_bin][2], th)                          # cfg / poses untouched
        self.assertAlmostEqual(sc.cfg["bins"]["yaw_deg"][sc.lid_bin], 8.0)


class TestObjectsAndLayout(unittest.TestCase):
    def test_size_overlay(self):
        for name in D2:
            sc = V.scene(V.load(name))
            self.assertEqual((sc.size, sc.object_height()), (0.052, 0.052))
            np.testing.assert_allclose(V.object_mesh_scale(sc), [0.052 / 0.045] * 3)
            self.assertEqual(V.object_physics(V.load(name), "mass", 0.05), 0.02)
            kw = sc.layout_kwargs()
            self.assertEqual((kw["n_min"], kw["n_max"]), (4, 6))
            r = max(sc.object_dims(s)["circumradius"] for s in sc.cfg["objects"]["shapes"])
            self.assertGreaterEqual(kw["min_spacing"], 2 * r)                  # no spawn overlap at any yaw
            for seed in range(150):
                lay = L.random_layout(sc, seed)
                self.assertTrue(4 <= len(lay) <= 6)
                poses, active = L.assign(sc, lay)
                for n in active:
                    p = poses[n][0]
                    self.assertAlmostEqual(p[2], sc.counter_z + 0.026 + 0.001)
                    self.assertTrue(sc.on_counter(sc.pelvis_to_mat(p)))
                    self.assertIsNone(sc.bin_of(sc.pelvis_to_mat(p)))
                    disc = B.rect(p[:2], (r, r), 0.0)
                    for bn in sc.bin_names:                                   # nothing spawns in / under a box
                        self.assertGreater(B.poly_gap(disc, sc.footprint(bn)), 0.03)

    def test_band_in_view(self):
        """Every corner of the (wider) object band, at the object top, is in the head image: with calibration.yaml
        (what the policy assumes) and with the variant's tilted sim camera (what it gets)."""
        from aaltonomous.interface.calibration import Calibration
        for name in D2:
            v, sc = V.load(name), V.scene(V.load(name))
            _, (x0, x1), (y0, y1) = B.band_rect(sc)
            z = sc.counter_z + sc.object_height()
            pts = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1)])
            cal = Calibration.load()
            self.assertTrue(cal.visible("head", pts).all(), name)
            cal.cams["head"].T_pelvis_optical = V.camera_T(v, cal.cams["head"].T_pelvis_optical)
            self.assertTrue(cal.visible("head", pts).all(), name)

    def test_mesh_scale_height_ratio(self):
        sc = V.scene(V.load(_write({"scene": {"objects": {"size": 0.05, "height_ratio": 0.8}}})))
        np.testing.assert_allclose(V.object_mesh_scale(sc), [0.05 / 0.045, 0.05 / 0.045, 0.04 / 0.045])

    def test_evaluator_samples_with_sim_scene(self):
        """evaluate.py sets env.scene = variant.scene(info.variant): AaltoEnv then samples 4-6 objects (real_d2)."""
        from aaltonomous.interface.env import AaltoEnv
        info = {"variant": V.summary(V.load("real_d2"))}
        n = set()
        for seed in range(30):
            env = AaltoEnv(backend="mock", task="sort_fixed", seed=seed)
            env.scene = V.scene(info["variant"])
            env.reset()
            n.add(len(env.layout))
        self.assertTrue(n <= {4, 5, 6} and len(n) > 1)
        a = V.scene(info["variant"])
        b = V.scene(V.load("real_d2"))
        self.assertEqual((a.poses, a.table, a.cfg), (b.poses, b.table, b.cfg))   # summary rebuilds the same scene


class TestJointLimits(unittest.TestCase):
    def test_real_d2(self):
        for name in D2:
            self.assertEqual(V.joint_limits(V.load(name)), {"left_hand_thumb_0_joint": (-0.26, None),
                                                            "right_hand_thumb_0_joint": (-0.22, None)})

    def test_regex(self):
        v = V.load(_write({"robot": {"joint_limits": {".*_hand_thumb_0_joint": [-0.2, None],
                                                      "left_elbow_joint": [None, 1.0]}}}))
        self.assertEqual(V.joint_limits(v), {"left_hand_thumb_0_joint": (-0.2, None), "right_hand_thumb_0_joint": (-0.2, None),
                                             "left_elbow_joint": (None, 1.0)})
        self.assertEqual(V.joint_limits(v, ["left_hand_thumb_1_joint"]), {})
        bad = TestValidate.bad
        bad(self, {"robot": {"joint_limits": {".*_thumb0_joint": [-0.2, None]}}})            # typo: no joint
        bad(self, {"robot": {"joint_limits": {".*_thumb_0_joint": [-0.2, None], "left_.*": [None, 1.0]}}})   # twice
        bad(self, {"robot": {"joint_limits": {"left_hand_thumb_0_joint": [0.3, 0.2]}}})
        bad(self, {"robot": {"joint_limits": {"left_hand_thumb_0_joint": -0.2}}})
        bad(self, {"robot": {"joint_limits": {"left_hand_thumb_0_joint": [1.2, None]}}})        # above the URDF upper

    def test_usd_authoring(self):
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("no pxr")
        st = Usd.Stage.CreateInMemory()
        root = st.DefinePrim("/World/envs/env_0/Robot", "Xform")
        for n in ("left_hand_thumb_0_joint", "right_hand_thumb_0_joint", "left_hand_thumb_1_joint"):
            j = UsdPhysics.RevoluteJoint.Define(st, f"/World/envs/env_0/Robot/joints/{n}")
            j.CreateLowerLimitAttr(-60.0)
            j.CreateUpperLimitAttr(60.0)
        lim = V.joint_limits(V.load("real_d2"))
        cfg = SimpleNamespace(spawn=SimpleNamespace(func=lambda path, c, translation=None, orientation=None, **k: root))
        self.assertEqual(V.install_joint_limits(cfg, V.load("real_d2")), lim)
        self.assertIs(cfg.spawn.func("/World/envs/env_.*/Robot", cfg.spawn), root)
        J = lambda n: UsdPhysics.RevoluteJoint(st.GetPrimAtPath(f"/World/envs/env_0/Robot/joints/{n}"))
        self.assertAlmostEqual(J("left_hand_thumb_0_joint").GetLowerLimitAttr().Get(), np.degrees(-0.26), places=4)
        self.assertAlmostEqual(J("right_hand_thumb_0_joint").GetLowerLimitAttr().Get(), np.degrees(-0.22), places=4)
        self.assertEqual(J("left_hand_thumb_0_joint").GetUpperLimitAttr().Get(), 60.0)     # null = kept
        self.assertEqual(J("left_hand_thumb_1_joint").GetLowerLimitAttr().Get(), -60.0)    # not matched
        cfg2 = SimpleNamespace(spawn=SimpleNamespace(func="untouched"))
        self.assertEqual(V.install_joint_limits(cfg2, V.load("real_like")), {})
        self.assertEqual(cfg2.spawn.func, "untouched")
        lone = Usd.Stage.CreateInMemory().DefinePrim("/R", "Xform")                       # joint missing: loud
        cfg3 = SimpleNamespace(spawn=SimpleNamespace(func=lambda *a, **k: lone))
        V.install_joint_limits(cfg3, V.load("real_d2"))
        with self.assertRaises(RuntimeError):
            cfg3.spawn.func("/R", cfg3.spawn)


class TestCameraR2(unittest.TestCase):
    def test_optical_pitch(self):
        import yaml
        with open(V.REPO / "aaltonomous/interface/calibration.yaml") as f:
            T = np.asarray(yaml.safe_load(f)["cameras"]["head"]["T_pelvis_optical"], float)
        T2 = V.camera_T(V.load("real_d2"), T)
        down = lambda M: np.degrees(np.arcsin(-M[2, 2]))       # optical z below the horizon
        self.assertAlmostEqual(down(T2) - down(T), 2.8, places=4)      # calibration.yaml: 6-digit rotation
        np.testing.assert_allclose(T2[:3, 3], T[:3, 3])
        np.testing.assert_allclose(T[:3, :3].T @ T2[:3, :3], V.rotvec([-2.8, 0, 0]), atol=1e-5)
        np.testing.assert_allclose(T2[:3, 0], T[:3, 0], atol=1e-12)       # rotation about optical x: x axis kept
        # a pelvis-frame pitch about +y gives the same here (optical x = pelvis -y): the two keys agree
        T3 = V.camera_T({"head_camera": {"rot_offset_deg": [0, 2.8, 0]}}, T)
        np.testing.assert_allclose(T3[:3, :3], T2[:3, :3], atol=1e-6)


# ---------------------------------------------------------------------------------------------- R3: compliant waist
ROBOT_USD = Path("/opt/aalto/third_party/unitree_sim_isaaclab/assets/robots/g1-29dof-dex3-base-fix-usd/"
                 "g1_29dof_with_dex3_base_fix.usd")
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def _pxr():
    try:
        import pxr  # noqa: F401
        return True
    except ImportError:
        return False


def _cal_T():
    import yaml
    with open(V.REPO / "aaltonomous/interface/calibration.yaml") as f:
        return np.asarray(yaml.safe_load(f)["cameras"]["head"]["T_pelvis_optical"], float)


class TestWaist(unittest.TestCase):
    def test_real_like_waist(self):
        for name in ("real_like_waist", "real_d2_waist"):     # the soft waist lives in its own variant files
            v = V.load(name)
            A = dict(V.actuator_overrides(v))
            self.assertEqual(V.spec_names("waist"), list(WAIST))
            for n in WAIST:
                self.assertEqual(V.resolve(A["waist"]["stiffness"], n), 300.0)
                self.assertEqual(V.resolve(A["waist"]["damping"], n), 3.0)
                self.assertEqual(V.resolve(A["waist"]["effort_limit"], n), 88.0 if "yaw" in n else 50.0)   # URDF
            self.assertNotIn("armature", A["waist"])                             # upstream's (USD)
            self.assertEqual(V.camera_parent(v), "torso_link")
        self.assertEqual(V.load("real_d2_waist")["head_camera"]["rot_offset_optical_deg"], [-2.8, 0.0, 0.0])   # merged
        for name in ("real_like",) + D2 + ("real_d2_yaw",):   # ... and the grasp test beds keep the rigid waist
            self.assertNotIn("waist", dict(V.actuator_overrides(V.load(name))))
            self.assertEqual(V.camera_parent(V.load(name)), "pelvis")
        act = SimpleNamespace(stiffness={".*": 10000.0}, damping={".*": 10000.0}, effort_limit=1000.0,
                              joint_effort_limit=None)
        for k, p in A["waist"].items():
            V.apply_actuator(act, "waist", {k: p})
        self.assertEqual((act.stiffness, act.damping, act.effort_limit), ({".*": 300.0}, {".*": 3.0}, None))
        self.assertEqual(act.joint_effort_limit, {"waist_yaw_joint": 88.0, "waist_(roll|pitch)_joint": 50.0})

    def test_unset_mount(self):
        T = _cal_T()
        for v in (None, V.load("sim_pyo"), V.load(_write({"mat": False}))):
            self.assertEqual(V.camera_parent(v), "pelvis")
            parent, T2 = V.camera_mount(v, T)                  # no USD needed for the organisers' mount
            self.assertEqual(parent, "pelvis")
            np.testing.assert_array_equal(T2, V.camera_T(v, T))
        self.assertIs(V.camera_mount(None, T)[1], T)
        self.assertIsNone(V.body_yaw_base_deg(None))
        self.assertIsNone(V.body_yaw(V.load("real_d2"), [{"color": "red", "x": 0.3, "y": 0.15}]))

    def test_validate(self):
        bad = TestValidate.bad
        bad(self, {"head_camera": {"parent": "head_link"}})
        bad(self, {"body_yaw": {"deg": 1.0, "sigma": 1.0}})
        bad(self, {"body_yaw": {"jitter_deg": -1.0}})
        bad(self, {"robot": {"actuators": {"waist": {"stiffness": {"waist_yaw_joint": 300.0}}}}})   # roll, pitch

    @unittest.skipUnless(_pxr() and ROBOT_USD.exists(), "needs pxr + the sim's robot USD (container)")
    def test_link_constants_match_usd(self):
        """variant.LINK_POSE_ZERO (used at run time, no USD read) == the sim's robot USD."""
        for link in ("pelvis", "torso_link", "d435_link"):
            np.testing.assert_allclose(V.link_T_zero(link), V.link_T_pelvis(ROBOT_USD, link), atol=1e-9)

    def test_no_pxr_before_kit(self):
        """scene_cfg runs camera_mount before Kit starts: importing pxr there broke Kit's USD (R3 live). The pure task
        modules + the mount of every variant must not pull pxr into a fresh process."""
        import subprocess
        import sys
        code = ("import sys, numpy as np\n"
                "from aaltonomous.sim.aalto_task import variant as V, boxes, layout, dr\n"
                "for n in ('real_like_waist', 'real_d2_waist', 'real_d2_yaw', 'real_venue'):\n"
                "    v = V.load(n); V.camera_mount(v, np.eye(4)); V.scene(v); V.joint_limits(v)\n"
                "bad = sorted(m for m in sys.modules if m == 'pxr' or m.startswith('pxr.'))\n"
                "print('PXR', bad); sys.exit(1 if bad else 0)\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(V.REPO), capture_output=True, text=True,
                           env={**os.environ, "AALTO_VARIANT": "real_like_waist"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_usd_links_and_mount(self):
        Tt = V.link_T_zero("torso_link")
        np.testing.assert_allclose(Tt[:3, 3], [-0.003963, 0.0, 0.044], atol=1e-6)
        np.testing.assert_allclose(Tt[:3, :3], np.eye(3), atol=1e-9)
        # calibration.yaml's head camera IS the URDF / USD d435_link at waist 0 (ROS optical axes): the organisers'
        # pelvis mount equals the torso mount as long as the waist is rigid
        Td, Tc = V.link_T_zero("d435_link"), _cal_T()
        np.testing.assert_allclose(Td[:3, 3], Tc[:3, 3], atol=2e-5)
        np.testing.assert_allclose(Td[:3, :3].T @ Tc[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=2e-5)
        for name in ("real_like_waist", "real_d2_waist"):
            v = V.load(name)
            parent, off = V.camera_mount(v, Tc)
            self.assertEqual(parent, "torso_link")
            np.testing.assert_allclose(Tt @ off, V.camera_T(v, Tc), atol=1e-12)     # same pose at waist 0
            np.testing.assert_allclose(off[:3, 3], [0.057623, 0.01753, 0.42987], atol=2e-5)
        v = V.load(_write({"head_camera": {"parent": "d435_link"}}))
        _, off = V.camera_mount(v, Tc)
        np.testing.assert_allclose(off[:3, 3], 0.0, atol=2e-5)
        if _pxr() and ROBOT_USD.exists():
            with self.assertRaises(ValueError):
                V.link_T_pelvis(ROBOT_USD, "no_such_link")


class TestBodyYaw(unittest.TestCase):
    def test_seeded_by_layout(self):
        v = V.load("real_d2_yaw")
        self.assertEqual(V.body_yaw_base_deg(v), 0.0)
        sc = V.scene(v)
        lays = [L.random_layout(sc, s) for s in range(1500)]
        y = np.degrees([V.body_yaw(v, lay) for lay in lays])
        self.assertEqual(V.body_yaw(v, lays[3]), V.body_yaw(v, [dict(o) for o in lays[3]]))    # same layout, same yaw
        self.assertLess(abs(np.std(y) - 1.0), 0.08)
        self.assertLess(abs(np.mean(y)), 0.08)
        self.assertLessEqual(np.abs(y).max(), 3.0 + 1e-9)
        self.assertIsNone(V.body_yaw(v, []))                       # parking / homing keeps the current yaw
        v = V.load(_write({"body_yaw": {"deg": 2.5}}))
        self.assertAlmostEqual(np.degrees(V.body_yaw(v, lays[0])), 2.5)
        self.assertEqual(V.body_yaw_base_deg(v), 2.5)


class TestProbeHelpersR3(unittest.TestCase):
    def test_prim_box_yawed(self):
        """The R2 probe read 0.282 / 0.261 for yawed 0.21 floors: sizes must be along the prim's own axes."""
        if not _pxr():
            self.skipTest("no pxr")
        from pxr import Gf, Usd, UsdGeom
        from aaltonomous.sim.aalto_task import hooks as Hk
        st = Usd.Stage.CreateInMemory()
        for k, yaw in enumerate((0.0, 7.0, -10.0, 25.0)):
            x = UsdGeom.Xform.Define(st, f"/World/envs/env_0/B{k}")
            x.AddTranslateOp().Set(Gf.Vec3d(0.6, 0.1 * k, 0.8))
            x.AddOrientOp().Set(Gf.Quatf(float(np.cos(np.radians(yaw) / 2)), 0.0, 0.0, float(np.sin(np.radians(yaw) / 2))))
            c = UsdGeom.Cube.Define(st, f"/World/envs/env_0/B{k}/geometry/mesh")
            c.GetSizeAttr().Set(0.003)
            c.AddScaleOp().Set(Gf.Vec3f(70.0, 70.0, 1.0))          # IsaacLab cuboid: cube of min size, scaled
            b = Hk._prim_box(st, f"/World/envs/env_0/B{k}", np.zeros(3))
            np.testing.assert_allclose(b["size"], [0.21, 0.21, 0.003], atol=1e-5)
            self.assertAlmostEqual(b["yaw_deg"], yaw, places=3)
            np.testing.assert_allclose(b["pos"], [0.6, 0.1 * k, 0.8 - L.PELVIS_W[2]], atol=1e-6)
        self.assertIsNone(Hk._prim_box(st, "/World/nothing", np.zeros(3)))

    def test_quat_and_pixels(self):
        from aaltonomous.sim import probe_variant as P
        from aaltonomous.sim.aalto_task import hooks as Hk
        th = np.radians(5.0)
        np.testing.assert_allclose(Hk.quat_xyzw_R([0, 0, np.sin(th / 2), np.cos(th / 2)]), V.rotvec([0, 0, 5.0]), atol=1e-12)
        # body yaw re-poses a pelvis-mounted head camera: the mount (as scene_cfg builds it) turned by the yaw
        v = V.load("real_d2_yaw")
        Tm = Hk.head_camera_pelvis_T(v)
        np.testing.assert_allclose(Tm, V.camera_T(v, _cal_T()))
        Rw = V.rotvec([0, 0, 1.4]) @ Tm[:3, :3]
        np.testing.assert_allclose(Hk.quat_xyzw_R(Hk.R_quat_xyzw(Rw)), Rw, atol=1e-5)   # calibration: 6 digits
        T = _cal_T()
        K = {"fx": 604.8, "fy": 604.3, "cx": 325.0, "cy": 250.9}
        pts = [np.array([0.30, 0.0, 0.036]), np.array([0.55, 0.0, 0.01])]
        self.assertEqual(P.pixel_shift(T, T, K, pts), [0.0, 0.0])
        T2 = V.camera_T({"head_camera": {"rot_offset_optical_deg": [-0.1, 0, 0]}}, T)   # pitch 0.1 deg
        for d in P.pixel_shift(T, T2, K, pts):
            self.assertAlmostEqual(d, 604.3 * np.radians(0.1), delta=0.1)
        # expected camera: pelvis mount turned with the body yaw (table frame)
        r = {"pelvis_in_table": {"pos": [0.0, 0.0, 0.0], "yaw_deg": 2.0}}
        Te, bend = P.expected_camera(None, T, r)
        np.testing.assert_allclose(Te[:3, :3], V.rotvec([0, 0, 2.0]) @ T[:3, :3], atol=1e-12)
        np.testing.assert_allclose(bend, np.eye(4))

    def test_expected_camera_torso(self):
        from aaltonomous.sim import probe_variant as P
        v, T = V.load("real_like_waist"), _cal_T()
        Tt = V.link_T_zero("torso_link")
        tilt = V.rotvec([0, 1.0, 0])                                    # torso pitched 1 deg about the waist
        r = {"pelvis_in_table": {"pos": [0, 0, 0], "yaw_deg": 0.0}, "usd_path": str(ROBOT_USD),
             "links": {"torso_link": {"pos": list(tilt @ Tt[:3, 3]), "R": tilt.tolist()}}}
        Te, bend = P.expected_camera(v, V.camera_T(v, T), r)
        np.testing.assert_allclose(bend[:3, :3], tilt, atol=1e-12)
        np.testing.assert_allclose(Te[:3, :3], tilt @ T[:3, :3], atol=1e-9)
        np.testing.assert_allclose(Te[:3, 3], tilt @ T[:3, 3], atol=1e-9)    # pivot = waist joint at the pelvis origin

    def test_reach_pose(self):
        from aaltonomous.interface.safety import JointSpec, SafetyConfig, WorkspaceChecker
        from aaltonomous.sim import probe_variant as P
        spec = JointSpec()
        ws = WorkspaceChecker(spec, SafetyConfig())
        home = np.array([L.ARM_HOME[n] for n in spec.arm_names])
        frames = list(SafetyConfig().workspace_frames)
        for side in ("left", "right"):
            q = P.reach_pose(spec, ws, home, side, 0.01 + 0.10 + 0.08)
            self.assertIsNotNone(q)
            self.assertEqual(ws.violation(q), 0.0)
            k = frames.index(f"{side}_hand_palm_link")
            p, p0 = ws.positions(q)[k], ws.positions(home)[k]
            self.assertGreater(p[0], p0[0] + 0.1)                      # a real forward reach
            self.assertGreaterEqual(p[2], 0.19)
            other = [i for i, n in enumerate(spec.arm_names) if not n.startswith(side)]
            np.testing.assert_array_equal(q[other], home[other])


# ---------------------------------------------------------------------------------------------- R4: messy layouts
from aaltonomous.sim.aalto_task import messy as MS   # noqa: E402

MESSY = ("real_venue_messy", "sim_messy")


def _pelvis_poly(sc, P):
    """Mat-coords CCW polygon -> pelvis-frame CCW polygon (the mat -> pelvis map is a mirror: reverse the order)."""
    return np.array([sc.mat_to_pelvis([x, y, 0.0])[:2] for x, y in P])[::-1]


class TestMessy(unittest.TestCase):
    def test_unset_unchanged(self):
        for v in (None, V.load("real_venue"), V.load("real_d2")):
            sc = V.scene(v)
            self.assertFalse(MS.active(v))
            for seed in range(20):
                lay = L.random_layout(sc, seed)
                self.assertIs(MS.messy_layout(v, sc, lay), lay)
                poses, act = L.assign(sc, lay)
                flagged, _ = L.assign(sc, [{**o, "lying": False} for o in lay])
                for n, (p, q) in poses.items():                    # upright: the old pose formula exactly
                    np.testing.assert_array_equal(p, flagged[n][0])
                    np.testing.assert_array_equal(q, flagged[n][1])
                for ob, n in zip(lay, act):
                    np.testing.assert_array_equal(poses[n][1], L.yaw_quat_xyzw(L.mat_yaw_to_pelvis(ob["yaw"])))

    def test_validate(self):
        bad = TestValidate.bad
        bad(self, {"layout": {"lying_prism": 0.5}})
        bad(self, {"layout": {"region_x": [0.3, 0.6]}})                  # scene.yaml knobs go under scene.layout
        bad(self, {"layout": {"lying_prism_prob": 1.5}})
        bad(self, {"layout": {"cluster_prob": 0.5, "cluster_spacing": [0.9, 1.2]}})
        bad(self, {"layout": {"cluster_prob": 0.5, "cluster_n": [1, 3]}})
        for name in MESSY:
            v = V.load(name)
            self.assertTrue(MS.active(v))
            self.assertGreater(min(MS.band_clearance(v, V.scene(v)).values()), 0.0)
        self.assertEqual(V.load("real_venue_messy")["inherit"][0], "real_venue")

    def test_lying_dims(self):
        rows = {}
        for s in (0.045, 0.048, 0.052):
            sc = V.scene(V.load(_write({"scene": {"objects": {"size": s}}})))
            c, across, along, top = MS.lying_dims(sc, "triangle")
            rows[s] = (c, across, along, top)
            self.assertAlmostEqual(c, s / 3)                                   # centroid one apothem above the side
            self.assertAlmostEqual(across, 2 * s / np.sqrt(3))                 # side length
            self.assertAlmostEqual(along, s)                                   # prism length (height_ratio 1)
            self.assertAlmostEqual(top, s)                                     # ridge = the triangle's altitude
        np.testing.assert_allclose(rows[0.048], [0.016, 0.0554256, 0.048, 0.048], atol=1e-6)
        with self.assertRaises(ValueError):
            MS.lying_dims(Scene.load(), "cube")

    def test_lying_mesh_rests_on_a_face(self):
        """The real STL, rotated as layout.assign spawns it: lowest points = a whole rectangular face at counter
        height, top = the altitude (minus the fillet at the ridge), long axis along ridge_yaw."""
        try:
            import trimesh
        except ImportError:
            self.skipTest("no trimesh")
        m = trimesh.load(str(V.REPO / "aaltonomous/scene/assets/object_triangle.stl"), force="mesh")
        self.assertTrue(m.is_convex)                                         # PhysX convex hull = the mesh itself
        sc = Scene.load()                                                    # STL exported at scene.yaml's 45 mm
        P0 = m.vertices * 0.001
        P0[:, 2] -= (P0[:, 2].min() + P0[:, 2].max()) / 2                    # scene_cfg center_z
        for side in range(3):
            for ridge in (0.0, 0.7, -2.0):
                R = MS.lying_R("triangle", ridge, side)
                P = P0 @ R.T + [0, 0, MS.spawn_height(sc, {"shape": "triangle", "lying": True})]
                self.assertAlmostEqual(P[:, 2].min(), 0.0, places=5)
                bottom = P[P[:, 2] < 1e-5]
                ax = np.array([np.cos(ridge), np.sin(ridge)])
                along = bottom[:, :2] @ ax
                across = bottom[:, :2] @ [-ax[1], ax[0]]
                self.assertGreater(np.ptp(along), 0.04)                      # a face, not an edge
                self.assertGreater(np.ptp(across), 0.03)
                self.assertGreater(P[:, 2].max(), 0.040)
                self.assertLessEqual(P[:, 2].max(), sc.size + 1e-6)
                np.testing.assert_allclose(R @ [0, 0, 1], [*ax, 0.0], atol=1e-12)
                t = MS.pose_truth(MS.quat_xyzw(R))
                self.assertTrue(t["lying"])
                self.assertAlmostEqual(t["tilt_deg"], 90.0, places=3)
                self.assertAlmostEqual(((t["ridge_yaw"] - ridge) + np.pi / 2) % np.pi - np.pi / 2, 0.0, places=4)
                fp = MS.footprint(sc, {"shape": "triangle", "x": 0.4, "y": 0.15, "lying": True, "ridge_yaw": ridge})
                pp = _pelvis_poly(sc, fp)                                    # the mesh lies inside its footprint
                c = sc.mat_to_pelvis([0.4, 0.15, 0.0])[:2]
                from aaltonomous.scene.scene import in_convex
                self.assertTrue(all(in_convex(pp, q + c) for q in P[:, :2]))
        up = MS.pose_truth(L.yaw_quat_xyzw(0.3))
        self.assertEqual((up["lying"], up["tilt_deg"], up["ridge_yaw"]), (False, 0.0, None))

    def test_assign_lying(self):
        sc = V.scene(V.load("real_venue"))
        ob = {"color": "blue", "shape": "triangle", "x": 0.4, "y": 0.15, "yaw": 0.0, "lying": True, "ridge_yaw": 0.5}
        poses, act = L.assign(sc, [ob])
        p, q = poses[act[0]]
        self.assertAlmostEqual(p[2], sc.counter_z + float(sc.cfg["objects"]["size"]) / 3 + 0.001)   # (venue: 47 mm)
        np.testing.assert_allclose(MS.R_of_quat(q), MS.lying_R("triangle", 0.5, 0), atol=1e-12)
        with self.assertRaises(ValueError):
            L.assign(sc, [{**ob, "shape": "cube"}])                          # a cube cannot "lie"
        poses, act = L.assign(sc, [{**ob, "shape": "cylinder"}])
        self.assertAlmostEqual(poses[act[0]][0][2], sc.counter_z + sc.size / 2 + 0.001)

    def test_sampling(self):
        for name in MESSY:
            v = V.load(name)
            sc = V.scene(v)
            Lc = sc.cfg["layout"]
            lo, hi = (s * sc.size for s in v["layout"]["cluster_spacing"])
            n_tri = n_lying = n_cl = 0
            import copy
            vs = sc if V.is_variant_scene(sc) else B.VariantScene(copy.deepcopy(sc.cfg))
            for seed in range(400):
                lay = L.random_layout(sc, seed)
                pl = MS.messy_layout(v, sc, lay)
                self.assertEqual(pl, MS.messy_layout(v, sc, [dict(o) for o in lay]))     # deterministic
                self.assertIs(MS.messy_layout(v, sc, pl), pl)                            # applied once
                self.assertEqual([(o["color"], o["shape"]) for o in pl], [(o["color"], o["shape"]) for o in lay])
                for o in pl:
                    self.assertIn("lying", o)
                    self.assertTrue(MS._in_region(sc, o))
                    n_tri += o["shape"] == "triangle"
                    n_lying += o["lying"]
                    self.assertTrue(not o["lying"] or o["shape"] == "triangle")
                    fp = _pelvis_poly(sc, MS.footprint(sc, o))
                    for bn in vs.bin_names:                                  # never touching / under a box
                        self.assertGreater(B.poly_gap(fp, vs.footprint(bn)), 0.0)
                for i, a in enumerate(pl):                                   # no interpenetration at spawn
                    for b in pl[i + 1:]:
                        self.assertGreaterEqual(B.poly_gap(MS.footprint(sc, a), MS.footprint(sc, b)),
                                                MS.GAP_MIN - 1e-9)
                        if not (a.get("cluster") and b.get("cluster")):     # others keep the sampler's spacing
                            self.assertGreaterEqual(np.hypot(a["x"] - b["x"], a["y"] - b["y"]),
                                                    Lc["min_spacing"] - 1.5e-3)   # sampler rounds to mm
                cl = [o for o in pl if o.get("cluster")]
                if cl:
                    n_cl += 1
                    self.assertTrue(2 <= len(cl) <= 4)
                    for o in cl:                                             # each member touches (nearly) another
                        d = min(np.hypot(o["x"] - b["x"], o["y"] - b["y"]) for b in cl if b is not o)
                        self.assertLessEqual(d, hi + 2e-4)
                if seed < 10:
                    _ = L.assign(sc, pl)                                     # the sim can place it
            self.assertLess(abs(n_lying / n_tri - 0.5), 0.07, name)
            self.assertGreater(n_cl / 400, 0.40, name)                       # 0.5 minus the rare misfits

    def test_evaluator_install(self):
        v = V.summary(V.load("real_venue_messy"))
        sent = []
        env = SimpleNamespace(scene=V.scene(v), backend=SimpleNamespace(reset_scene=lambda lay, ins: sent.append(lay)))
        self.assertTrue(MS.install(env, v))
        self.assertFalse(MS.install(SimpleNamespace(), V.summary(V.load("real_venue"))))
        for seed in range(30):
            lay = L.random_layout(env.scene, seed)
            ref = MS.messy_layout(v, env.scene, [dict(o) for o in lay])
            env.backend.reset_scene(lay, "x")
            self.assertEqual(lay, ref)                                       # env.layout updated in place
            self.assertIs(sent[-1], lay)
            self.assertIs(MS.messy_layout(v, env.scene, lay), lay)           # the sim keeps it as placed

    def test_probe_seed_search(self):
        from aaltonomous.sim import probe_variant as P
        v = V.load("real_venue_messy")
        s = P.messy_seeds(v, V.scene(v))
        self.assertIn((True, True), s)


class TestCameraFollow(unittest.TestCase):
    """R3 live: a camera prim under a robot link keeps its spawn pose; the runtime writes it every step instead."""

    def test_when(self):
        from aaltonomous.sim.aalto_task import hooks as Hk
        for name, want in ((None, False), ("real_like", False), ("real_d2", False), ("sim_messy", False),
                           ("real_like_waist", True), ("real_d2_waist", True), ("real_d2_yaw", True)):
            self.assertEqual(Hk.camera_follows(None if name is None else V.load(name)), want, name)

    def test_world_pose(self):
        from aaltonomous.sim import probe_variant as P
        from aaltonomous.sim.aalto_task import hooks as Hk
        v = V.load("real_like_waist")
        parent, mount = V.camera_mount(v, Hk.head_camera_pelvis_T(None))
        self.assertEqual(parent, "torso_link")
        Tz = V.link_T_zero("torso_link")
        # waist at zero: the camera where calibration.yaml puts it (world = pelvis frame + PELVIS_W, no yaw)
        p, R = Hk.camera_world_pose(Tz[:3, 3] + L.PELVIS_W, [0, 0, 0, 1], mount)
        np.testing.assert_allclose(p - L.PELVIS_W, _cal_T()[:3, 3], atol=1e-9)
        np.testing.assert_allclose(R, _cal_T()[:3, :3], atol=1e-9)
        # torso pitched 2 deg about the waist joint (at the pelvis origin): the camera turns with it, = the probe's
        # expected_camera; DR jitter goes on top in the world frame
        tilt = V.rotvec([0, 2.0, 0])
        q = Hk.R_quat_xyzw(tilt)
        p, R = Hk.camera_world_pose(tilt @ Tz[:3, 3], q, mount)
        r = {"pelvis_in_table": {"pos": [0, 0, 0], "yaw_deg": 0.0},
             "links": {"torso_link": {"pos": list(tilt @ Tz[:3, 3]), "R": tilt.tolist()}}}
        Te, _ = P.expected_camera(v, V.camera_T(v, _cal_T()), r)
        np.testing.assert_allclose(p, Te[:3, 3], atol=1e-7)                # via a quaternion: float32-ish
        np.testing.assert_allclose(R, Te[:3, :3], atol=1e-6)
        J = (V.rotvec([0.3, 0, 0]), np.array([0.001, 0, 0]))
        pj, Rj = Hk.camera_world_pose(tilt @ Tz[:3, 3], q, mount, J)
        np.testing.assert_allclose(Rj, J[0] @ R, atol=1e-12)
        np.testing.assert_allclose(pj, p + J[1], atol=1e-12)
        # yawed pelvis mount (body_yaw): turned about the vertical through the pelvis
        v = V.load("real_d2_yaw")
        parent, mount = V.camera_mount(v, Hk.head_camera_pelvis_T(None))
        Rz = V.rotvec([0, 0, 1.3])
        p, R = Hk.camera_world_pose([0, 0, 0], Hk.R_quat_xyzw(Rz), mount)
        np.testing.assert_allclose(R, Rz @ V.camera_T(v, _cal_T())[:3, :3], atol=1e-6)

    def test_image_shift(self):
        """Phase correlation sign = the content's motion from image 0 to image 1 (what _uv(T1) - _uv(T0) predicts)."""
        import cv2
        from aaltonomous.sim import probe_variant as P
        rng = np.random.default_rng(0)
        img = cv2.GaussianBlur(rng.uniform(0, 255, (480, 640)).astype(np.float32), (0, 0), 3)
        img = np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        for du, dv in ((3, 0), (0, -4), (-2, 5)):
            moved = np.roll(img, (dv, du), axis=(0, 1))                  # content moves +du (right), +dv (down)
            m, resp = P.image_shift(img, moved, (320, 240))
            np.testing.assert_allclose(m, [du, dv], atol=0.2)
        self.assertEqual(P.image_shift(img, img, (5, 240))[0], None)      # window leaves the image


# ---------------------------------------------------------------------------------------------- m24 fixes
def _fake_runtime(vname, cam=False):
    """An AaltoRuntime without Isaac: just what place() / camera_image_check() touch."""
    import torch
    from aaltonomous.sim.aalto_task import hooks as Hk
    rt = object.__new__(Hk.AaltoRuntime)
    rt.variant = V.load(vname)
    rt.sc = V.scene(rt.variant)
    rt.names = L.object_names(rt.sc)
    rt.dr_params, rt.layout, rt.active, rt.body_yaw_deg = None, [], [], None
    rt.cam_follow = False

    class Obj:
        def write_root_pose_to_sim_index(self, root_pose, env_ids):
            self.pose = root_pose

        def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
            pass
    env = SimpleNamespace(scene={n: Obj() for n in rt.names})
    env.scene = type("S", (dict,), {})(env.scene)
    env.scene.env_origins = torch.zeros((1, 3))
    return rt, env


class TestExplicitResetPlacedAsGiven(unittest.TestCase):
    """m24 live: `aalto check` in sim_messy failed its drop test: the scene service's `reset` (explicit layouts:
    check_scene's drops, grasp labs) went through the variant's messy transform (cluster / lying) like a sampled one."""

    def test_reset_vs_reset_random(self):
        rt, env = _fake_runtime("sim_messy")
        drop = [{"color": "red", "shape": "cube", "x": 0.71, "y": 0.365, "yaw": 0.2, "z": 0.10},
                {"color": "blue", "shape": "cylinder", "x": 0.19, "y": 0.365, "yaw": 0.0, "z": 0.10},
                {"color": "yellow", "shape": "triangle", "x": 0.22, "y": 0.345, "yaw": 0.5, "z": 0.16},
                {"color": "red", "shape": "triangle", "x": 0.45, "y": -0.30, "yaw": 0.0, "z": 0.05}]
        self.assertNotEqual(MS.messy_layout(rt.variant, rt.sc, drop), drop)   # this layout WOULD be transformed
        rt.place(env, [dict(o) for o in drop])
        self.assertEqual(rt.layout, drop)                                      # explicit: exactly as given
        for seed in range(5):
            lay = L.random_layout(rt.sc, seed)
            rt.place(env, [dict(o) for o in lay], sampled=True)
            self.assertEqual(rt.layout, MS.messy_layout(rt.variant, rt.sc, lay))   # sampled: messy applies


class TestCameraImageCheck(unittest.TestCase):
    """The sim's start-up self-check: rendered depth of the counter vs the written / the spawn camera pose."""

    def _depth(self, T, K, z, shape=(480, 640)):
        from aaltonomous.sim.aalto_task import hooks as Hk
        D = np.zeros(shape, np.uint16)
        for v in range(0, shape[0]):
            for u in range(0, shape[1], 4):
                d = Hk.counter_depth(T, K, u, v, z)
                if d is not None and 0.1 < d < 10:
                    D[v, u:u + 4] = int(round(d * 1000))
        return D

    def test_verdicts(self):
        import yaml
        from aaltonomous.sim.aalto_task import hooks as Hk
        rt, env = _fake_runtime("real_like_waist")
        rt.cam_parent, rt.cam_mount = V.camera_mount(rt.variant, Hk.head_camera_pelvis_T(None))
        with open(V.REPO / "aaltonomous/interface/calibration.yaml") as f:
            K = yaml.safe_load(f)["cameras"]["head"]["intrinsics"]
        Tz = V.link_T_zero("torso_link")
        tilt = V.rotvec([0, 1.7, 0])                                   # the home sag seen live
        p, R = Hk.camera_world_pose(tilt @ Tz[:3, 3] + L.PELVIS_W, Hk.R_quat_xyzw(tilt), rt.cam_mount)
        rt._cam_last = (p, R)
        Tw = np.eye(4)
        Tw[:3, :3], Tw[:3, 3] = R, p
        Ts = np.eye(4)
        Ts[:3, :3], Ts[:3, 3] = (Tz @ rt.cam_mount)[:3, :3], (Tz @ rt.cam_mount)[:3, 3] + L.PELVIS_W
        z = rt.sc.counter_z + L.PELVIS_W[2]
        for T, want in ((Tw, "follows"), (Ts, "does NOT follow")):
            rt._latest = (None, self._depth(T, K, z), 0.0)
            r = Hk.camera_image_check(rt, env)
            self.assertTrue(r["verdict"].startswith(want), r)
            self.assertGreater(abs(r["depth_resid_spawn_mm"] - r["depth_resid_written_mm"]), 4.0, r)


class TestPixelTools(unittest.TestCase):
    def test_verdict_and_compare(self):
        import cv2
        from aaltonomous.sim import probe_variant as P
        from aaltonomous.sim.aalto_task import hooks as Hk
        rng = np.random.default_rng(1)
        img = cv2.GaussianBlur(rng.uniform(0, 255, (480, 640)).astype(np.float32), (0, 0), 3)
        img = np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        self.assertIs(P.pixel_verdict("t", img, np.roll(img, 6, axis=0), (320, 240), np.array([0.0, 6.0])), True)
        self.assertIs(P.pixel_verdict("t", img, img, (320, 240), np.array([0.0, 6.0])), False)   # did not move
        self.assertIsNone(P.pixel_verdict("t", img, img, (10, 240), np.array([0.0, 0.0])))       # off the image
        flat = np.full_like(img, 128)
        self.assertIsNone(P.pixel_verdict("t", flat, flat, (320, 240), np.array([0.0, 0.0])))    # no texture
        # compare: two shots whose expected cameras differ by a small pitch; B's image = A's moved as predicted
        d = tempfile.mkdtemp()
        sc = V.scene(V.load("real_like"))
        import copy
        VS = B.VariantScene(copy.deepcopy(sc.cfg))
        refs = np.array([P.box_front_top(sc, VS, n) for n in ("left", "center", "right")])
        T = _cal_T() + 0.0
        T[:3, 3] += L.PELVIS_W * 0                                    # table frame = pelvis frame here
        TB = V.camera_T({"head_camera": {"rot_offset_optical_deg": [-0.5, 0, 0]}}, T)
        K = {"fx": 604.8, "fy": 604.3, "cx": 325.0, "cy": 250.9}
        pred = P._uv(TB, K, refs[1]) - P._uv(T, K, refs[1])
        moved = np.roll(img, int(round(pred[1])), axis=0)
        meta = json.dumps({"variant": "x"})
        np.savez(f"{d}/a.npz", rgb=img, depth=np.zeros((480, 640)), T=T, refs=refs, K=json.dumps(K), meta=meta)
        np.savez(f"{d}/b.npz", rgb=moved, depth=np.zeros((480, 640)), T=TB, refs=refs, K=json.dumps(K), meta=meta)
        np.savez(f"{d}/c.npz", rgb=img, depth=np.zeros((480, 640)), T=TB, refs=refs, K=json.dumps(K), meta=meta)
        self.assertTrue(P.camera_compare(f"{d}/a.npz", f"{d}/b.npz"))
        self.assertFalse(P.camera_compare(f"{d}/a.npz", f"{d}/c.npz"))   # the image did not follow the pose
        self.assertTrue(Hk.camera_follows(V.load("real_like_waist")))


if __name__ == "__main__":
    unittest.main()
