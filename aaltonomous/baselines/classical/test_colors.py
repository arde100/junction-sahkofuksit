"""Open colour vocabulary: instructions in other colour words (parser), the words matched to the world's colour classes
(colors.py), object_bin with episode classes, perception with multi-hue classes (synthetic pixels, real-robot blob
statistics from data drop 2, recorded sim frames hue-shifted to orange / purple) and mock episodes with the robot
configuration and real-like object colours. Run in the sim container:
    aalto.sh py -m unittest aaltonomous/baselines/classical/test_colors.py -v
"""
import csv
import itertools
import json
import os
import re
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")

from aaltonomous.interface import tasks

from .colors import Calibrator, hue_dist, match_colors
from .instruction import BINS, HUE, _COLOR, mapping_colors, object_bin, parse_instruction
from .test_instruction import DEV, HELD

L, C, R = "left", "center", "right"
SIM_WORLD = ({"red": [0.0], "yellow": [25.0], "blue": [113.0]}, [12.0])    # scene.yaml (profile sim)
ROBOT_WORLD = ({"orange": [12.0], "yellow": [30.0], "purple": [111.0]}, [4.0])   # config base (data drop 2)
TESTDATA = Path(__file__).parent / "testdata"


def recolour(text, words):
    """Colour words replaced (case and the plural kept): recolour("Reds left", {"red": "orange"}) = "Oranges left"."""
    def sub(m):
        w = m.group(0)
        lw = w.lower()
        base, s = (lw[:-1], "s") if lw.endswith("s") and lw[:-1] in words else (lw, "")
        if base not in words:
            return w
        r = words[base] + s
        return r.upper() if w.isupper() and len(w) > 1 else r.capitalize() if w[0].isupper() else r
    return re.sub(r"[A-Za-z]+", sub, text)


def rekey(m, words):
    """Expected mapping with the colour keys renamed (canonical names of the new words)."""
    def c(k):
        return _COLOR[words[k]] if k in words else k
    return {((c(k[0]), k[1]) if isinstance(k, tuple) else c(k)): v for k, v in m.items()}


def match(text_or_mapping, world, **kw):
    if isinstance(text_or_mapping, str):
        r = parse_instruction(text_or_mapping)
        return r, match_colors(mapping_colors(r.mapping), *world, words=r.words, **kw)
    return None, match_colors(mapping_colors(text_or_mapping), *world, **kw)


class TestOpenVocabulary(unittest.TestCase):
    TRIPLES = [("orange", "yellow", "purple"), ("amber", "gold", "violet"), ("crimson", "yellow", "navy"),
               ("magenta", "lemon", "tangerine"), ("green", "pink", "cyan"), ("lilac", "orange", "blue")]

    def test_templates_in_other_colours(self):
        """Every training template x colour / bin permutation with other colour triples: the canonical mapping,
        ok, no problem."""
        n = 0
        for words in self.TRIPLES:
            for tpl in tasks.TEMPLATES:
                for cs in itertools.permutations(words):
                    for bs in itertools.permutations(BINS):
                        w = {f"c{i}": c for i, c in enumerate(cs)}
                        w.update({f"C{i}": c.capitalize() for i, c in enumerate(cs)})
                        w.update({f"b{i}": b for i, b in enumerate(bs)})
                        s = tpl.format(**w)
                        r = parse_instruction(s)
                        self.assertEqual(r.mapping, {_COLOR[c]: b for c, b in zip(cs, bs)}, s)
                        self.assertTrue(r.ok, s)
                        self.assertEqual(r.problems, [], s)
                        n += 1
        self.assertEqual(n, 6 * 288)

    def test_paraphrase_sets_in_orange_and_purple(self):
        """The dev and held-back paraphrase sets with red -> orange, blue -> purple (and red -> crimson, blue ->
        violet): exact mapping (typo sentences of the replaced words excluded)."""
        for words in ({"red": "orange", "blue": "purple"}, {"red": "crimson", "blue": "violet"}):
            fails, n = [], 0
            for s, exp in DEV + HELD:
                if re.search(r"\b(rde|bleu)\b", s, re.I):
                    continue
                t = recolour(s, words)
                got = parse_instruction(t).mapping
                n += 1
                if got != rekey(exp, words):
                    fails.append((t, rekey(exp, words), got))
            for t, e, g in fails:
                print(f"\nFAIL {t!r}\n  expected {e}\n  got      {g}")
            self.assertEqual(fails, [], words)
            self.assertGreater(n, 150)

    def test_orange_objects_and_orange_bins(self):
        """'orange' is an object colour unless a bin noun follows directly."""
        cases = [("Put the orange objects in the left bin.", {"orange": L}),
                 ("Orange goes left, purple goes right.", {"orange": L, "purple": R}),
                 ("Put the orange cubes into the orange box on the right.", {("orange", "cube"): R}),
                 ("The three orange boxes are the bins: orange in the left one, yellow in the middle one.",
                  {"orange": L, "yellow": C}),
                 ("Put the purple ones in the left orange bin and the orange ones in the right orange bin.",
                  {"purple": L, "orange": R}),
                 ("Put everything into the orange plastic box on the left.", {"default": L}),
                 ("Put the green objects in the left bin.", {"green": L}),
                 ("Violet objects go in the middle, amber ones on the left, everything else on the right.",
                  {"purple": C, "orange": L, "default": R}),
                 ("Put the dark blue items in the left bin and the light purple items in the right bin.",
                  {"blue": L, "purple": R}),
                 ("Put the purpel cubes in the rigth bin.", {("purple", "cube"): R})]
        for s, exp in cases:
            r = parse_instruction(s)
            self.assertEqual(r.mapping, exp, f"{s}: {r.summary()}")
            self.assertTrue(r.ok, s)
        r = parse_instruction("Violet left, lilac right.")      # one colour, two bins: a conflict, not two keys
        self.assertEqual(r.mapping, {})
        self.assertEqual(parse_instruction("Put the white objects in the left bin.").mapping, {})   # no hue
        self.assertEqual(parse_instruction("Violet goes left.").words, {"purple": ["violet"]})

    def test_achromatic_is_unknown_colour(self):
        r = parse_instruction("Black objects go left, red objects go right.")
        self.assertEqual(r.mapping, {"red": R})
        self.assertIn("unknown_colour", [p["kind"] for p in r.problems])


class TestMatch(unittest.TestCase):
    def test_sim_red_yellow_blue_as_before(self):
        _, ec = match(tasks.FIXED_INSTRUCTION, SIM_WORLD)
        self.assertEqual((ec.refs, ec.reject, ec.problems), (SIM_WORLD[0], [12.0], []))

    def test_robot_own_words(self):
        _, ec = match("Put the orange objects in the left bin, the yellow objects in the center bin and the purple "
                      "objects in the right bin.", ROBOT_WORLD)
        self.assertEqual((ec.refs, ec.reject, ec.problems), (ROBOT_WORLD[0], [4.0], []))

    def test_robot_told_red_yellow_blue(self):
        """Objects orange / purple, words red / blue: merged with the unnamed classes (measured hue kept); red's
        lexicon 0 joins as well, which drops the bin-wall reject 4 (an instruction colour that close to it)."""
        _, ec = match(tasks.FIXED_INSTRUCTION, ROBOT_WORLD)
        self.assertEqual(ec.refs, {"red": [12.0, 0.0], "yellow": [30.0], "blue": [111.0, 115.0]})
        self.assertEqual(ec.alias, {"yellow": "yellow", "orange": "red", "purple": "blue"})
        self.assertEqual((ec.reject, ec.problems), ([], []))

    def test_sim_config_orange_purple_objects(self):
        """Sim configuration (red / blue classes) but objects and words orange / purple: merged, lexicon hues added
        (13 is 12 from yellow 25: kept), the bins' reject 12 dropped (1 from orange 13)."""
        _, ec = match("Orange goes left, yellow in the middle, purple on the right.", SIM_WORLD)
        self.assertEqual(ec.refs, {"orange": [0.0, 13.0], "yellow": [25.0], "purple": [113.0, 135.0]})
        self.assertEqual(ec.reject, [])

    def test_synonyms_of_the_world_class(self):
        """violet / lilac / lavender / magenta = purple: the robot's world class with its measured hue only."""
        for w in ("violet", "lilac", "lavender", "magenta", "purple"):
            _, ec = match(f"Put the {w} objects in the left bin and the orange objects in the right bin.", ROBOT_WORLD)
            self.assertEqual(ec.refs, {"purple": [111.0], "orange": [12.0], "yellow": [30.0]}, w)
            _, ec = match(f"Put the {w} objects in the left bin.", SIM_WORLD)
            self.assertEqual(ec.refs["purple"], [113.0, 135.0], w)       # = the sim's blue objects
            self.assertNotIn("blue", ec.refs)

    def test_blue_and_purple_on_the_robot(self):
        """Both named, one measured class (111) and blue's lexicon 115: inseparable -> one combined class whose
        objects get a bin only if blue and purple share it."""
        r, ec = match("Blue goes left, purple goes right, yellow in the middle.", ROBOT_WORLD)
        self.assertIn("blue/purple", ec.refs)
        self.assertEqual([p["kind"] for p in ec.problems], ["inseparable"])
        self.assertIsNone(object_bin(r.mapping, "blue/purple", None, ec.names))
        self.assertEqual(object_bin(r.mapping, "yellow", None, ec.names), C)
        r, ec = match("Blue and purple go left, yellow right.", ROBOT_WORLD)
        self.assertEqual(object_bin(r.mapping, "blue/purple", "cube", ec.names), L)   # same bin: either way right
        _, ec = match("Blue goes left, purple goes right.", SIM_WORLD)                 # sim: 22 apart, separable
        self.assertEqual(ec.refs, {"blue": [113.0], "purple": [135.0], "red": [0.0], "yellow": [25.0]})

    def test_ambiguous_merge(self):
        """A colour between two unnamed classes (runner-up within `ambiguous`): no merge; its lexicon-only class
        is then too close to both and combined with them: those objects get no bin, the far ones the default."""
        r, ec = match("Orange goes left, everything else goes right.", SIM_WORLD)   # 13: red 13, yellow 12 away
        self.assertEqual([p["kind"] for p in ec.problems], ["ambiguous", "inseparable"])
        self.assertIn("orange/red/yellow", ec.refs)
        self.assertIsNone(object_bin(r.mapping, "orange/red/yellow", None, ec.names))
        self.assertEqual(object_bin(r.mapping, "blue", None, ec.names), R)
        # two words, one class at the same distance (claimants tie): neither takes it
        _, ec = match({"green": L, "cyan": R}, ({"teal": [75.0], "red": [0.0]}, []))
        self.assertEqual(ec.problems[0]["kind"], "ambiguous")
        self.assertNotIn("green", [k for k, v in ec.refs.items() if 75.0 in v and "/" not in k])

    def test_unmatched_colours(self):
        """A colour of no world class and no class near: its own lexicon class (finds nothing unless such objects
        exist); pink on the robot: 32 from orange (> merge_max), no merge. No hue (white): no class, a warning."""
        r, ec = match("Green goes left, everything else right.", ROBOT_WORLD)
        self.assertEqual(ec.refs["green"], [60.0])
        self.assertEqual(object_bin(r.mapping, "orange", None, ec.names), R)
        _, ec = match({"pink": L}, ROBOT_WORLD)
        self.assertEqual(ec.refs["pink"], [160.0])
        _, ec = match({"white": L, "red": R}, ROBOT_WORLD)
        self.assertEqual([p["kind"] for p in ec.problems], ["no_hue"])
        self.assertNotIn("white", ec.refs)

    def test_one_colour_and_everything_else(self):
        r, ec = match("Put the purple objects in the left bin and everything else in the right bin.", SIM_WORLD)
        self.assertEqual(ec.names, ["purple", "red", "yellow"])
        self.assertEqual([object_bin(r.mapping, c, None, ec.names) for c in ("purple", "red", "yellow", "unknown")],
                         [L, R, R, None])
        r, ec = match("Put the red objects in the left bin and everything else in the right bin.", ROBOT_WORLD)
        self.assertEqual(ec.names, ["red", "yellow", "purple"])          # red = the orange objects
        self.assertEqual(object_bin(r.mapping, "purple", "cube", ec.names), R)

    def test_given_and_fixed_mappings(self):
        _, ec = match({"orange": L, "yellow": C, "purple": R}, ROBOT_WORLD)
        self.assertEqual(ec.refs, ROBOT_WORLD[0])
        _, ec = match(dict(tasks.FIXED_MAPPING), ROBOT_WORLD)
        self.assertEqual(sorted(ec.refs), ["blue", "red", "yellow"])
        _, ec = match({"cube": L, "default": R}, ROBOT_WORLD)            # no colour named: the world's classes
        self.assertEqual(ec.refs, ROBOT_WORLD[0])

    def test_multi_hue_world_class(self):
        _, ec = match({"orange": L}, ({"orange": [9.0, 13.0], "yellow": 30}, [2, 7]))
        self.assertEqual(ec.refs["orange"], [9.0, 13.0])
        self.assertEqual(ec.reject, [2.0])                             # 7: 2 from orange 9, dropped


class TestFixedAndShapes(unittest.TestCase):
    """The robot's sort_fixed mapping (organisers, 2026-11-14) and instructions with shape and colour + shape keys."""

    def test_fixed_mappings(self):
        try:
            from .config import fixed_mapping, load_config
        except ImportError as e:
            self.skipTest(f"needs yaml: {e}")
        robot = fixed_mapping(load_config(profile="none"))
        self.assertEqual(robot, {"yellow": L, "purple": C, "orange": R})
        self.assertEqual(fixed_mapping(load_config(profile="sim")), tasks.FIXED_MAPPING)
        _, ec = match(robot, ROBOT_WORLD)
        self.assertEqual(ec.refs, ROBOT_WORLD[0])
        self.assertEqual([object_bin(robot, c, None, ec.names) for c in ("orange", "yellow", "purple")], [R, L, C])

    def test_sort_fixed_stale_text_on_the_robot(self):
        """Task sort_fixed on the robot profile: a text in this world's colour words is followed; the simulator's old
        sentence (red / blue) or an unreadable one gives the organisers' fixed mapping; sort_language and the sim
        profile always follow the text."""
        try:
            from .policy import ClassicalPolicy
        except ImportError as e:
            self.skipTest(f"needs the policy's dependencies: {e}")
        old = tasks.FIXED_INSTRUCTION
        new = "Put the yellow objects in the left bin, the purple objects in the center bin and the orange objects in the right bin."
        other = "Put the orange objects in the left bin, the yellow objects in the center bin and the purple objects in the right bin."
        robot = ClassicalPolicy(overrides=None, config=__import__("aaltonomous.baselines.classical.config", fromlist=["x"]).load_config(profile="none"))
        want = {"yellow": L, "purple": C, "orange": R}
        for text, task, m, src in ((old, "sort_fixed", want, "fixed"), (new, "sort_fixed", want, "parsed"),
                                   ("please sort them", "sort_fixed", want, "fixed"),
                                   (other, "sort_fixed", {"orange": L, "yellow": C, "purple": R}, "parsed"),
                                   (old, "sort_language", {"red": L, "yellow": C, "blue": R}, "parsed"),
                                   (old, None, {"red": L, "yellow": C, "blue": R}, "parsed")):
            got = robot._mapping({"instruction": text, "task": task})
            self.assertEqual((got[0], got[1]), (m, src), (text, task))
        sim = ClassicalPolicy(config=__import__("aaltonomous.baselines.classical.config", fromlist=["x"]).load_config(profile="sim"))
        got = sim._mapping({"instruction": other, "task": "sort_fixed"})
        self.assertEqual((got[0], got[1]), ({"orange": L, "yellow": C, "purple": R}, "parsed"))

    def test_shape_keys_with_calibration(self):
        """Shape keys need the shape: an object whose bin depends on its shape and has none stays; colours are
        calibrated from the scene as with colour-only instructions."""
        cases = [("Purple cylinders left, everything else right.",
                  {("purple", "cylinder"): L, "default": R},
                  [("purple", "cylinder", L), ("purple", "cube", R), ("purple", None, None), ("orange", None, R),
                   ("yellow", "triangle", R), ("unknown", "cylinder", None)]),
                 ("Cubes in the middle, orange prisms right, the rest left.",
                  {"cube": C, ("orange", "triangle"): R, "default": L},
                  [("orange", "cube", C), ("orange", "triangle", R), ("orange", None, None), ("yellow", None, None),
                   ("yellow", "cylinder", L), ("purple", "cube", C), ("unknown", "cube", C)])]
        for text, mapping, objs in cases:
            r = parse_instruction(text)
            self.assertEqual(r.mapping, mapping, text)
            for world, calib, hues in ((ROBOT_WORLD, None, [12, 30, 111]), (SIM_WORLD, SIMP, [22.2, 28.4, 142.8])):
                cal = Calibrator(mapping_colors(r.mapping), *world, None, calib, r.words)
                cal.update([(h, 1000.0, 2.0) for h in hues])
                self.assertTrue({"orange", "yellow", "purple"} <= set(cal.ec.names) or world is SIM_WORLD, cal.ec.refs)
                for col, shape, want in objs:
                    if world is SIM_WORLD and col not in cal.ec.names and col != "unknown":
                        continue                     # (the sim's prior names its yellow / purple classes otherwise)
                    self.assertEqual(object_bin(r.mapping, col, shape, cal.ec.names), want, (text, col, shape))


class TestObjectBinClasses(unittest.TestCase):
    def test_episode_classes(self):
        m, cl = {"orange": L, "yellow": C, "purple": R}, ["orange", "yellow", "purple"]
        self.assertEqual([object_bin(m, c, None, cl) for c in cl], [L, C, R])
        self.assertIsNone(object_bin(m, "red", None, cl))          # a colour no key names (no default)
        self.assertEqual(object_bin({**m, "default": C}, "red", None, cl), C)
        self.assertIsNone(object_bin({**m, "default": C}, "rubbish", None, cl))   # unknown name: any class
        self.assertIsNone(object_bin(m, "unknown", "cube", cl))
        self.assertEqual(object_bin({"default": C}, "unknown", None, cl), C)
        m = {"orange": L, "default": R}
        self.assertIsNone(object_bin(m, "orange/red", None, ["orange/red", "yellow"]))
        self.assertEqual(object_bin(m, "yellow", None, ["orange/red", "yellow"]), R)
        self.assertEqual(object_bin({"orange": L, "red": L}, "orange/red", None, ["orange/red"]), L)
        self.assertIsNone(object_bin({"cube": L}, "orange", None, cl))  # shape key needs the shape

    def test_summary_with_classes(self):
        r = parse_instruction("Orange goes left, yellow goes right.")
        self.assertIn("no bin: purple", r.summary(["orange", "yellow", "purple"]))


def _perception(profile="none"):
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    from .config import load_config
    from .perception import Perception
    return Perception(load_config(profile=profile), Scene.load(), Calibration.load())


def _blob(h, s, v, n, rng, sd=1.5):
    hh = (rng.normal(h, sd, n)) % 180
    return np.c_[hh, np.clip(rng.normal(s, 10, n), 0, 255), np.clip(rng.normal(v, 10, n), 0, 255)].astype(np.uint8)


class TestPerceptionClasses(unittest.TestCase):
    """Classification with episode classes (Perception.set_classes) on synthetic HSV pixels (no image)."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.p = _perception()
        except ImportError as e:
            raise unittest.SkipTest(f"needs cv2 / yaml: {e}")

    def vote(self, pix, ec):
        self.p.set_classes(ec.refs, ec.reject, ec.alias)
        cl = self.p._cls
        return self.p._vote(self.p._classify(pix, cl), pix[:, 1], cl)[0]

    def test_multi_hue_class(self):
        rng = np.random.default_rng(0)
        _, ec = match(tasks.FIXED_INSTRUCTION, ROBOT_WORLD)             # red = [12, 0], blue = [111, 115]
        self.assertEqual(self.vote(_blob(12, 160, 190, 300, rng), ec), "red")
        self.assertEqual(self.vote(_blob(1, 160, 190, 300, rng), ec), "red")
        self.assertEqual(self.vote(_blob(111, 100, 170, 300, rng), ec), "blue")
        self.assertEqual(self.vote(_blob(30, 110, 140, 300, rng), ec), "yellow")
        self.assertEqual(self.vote(_blob(60, 160, 190, 300, rng), ec), "unknown")      # green: no class near
        self.assertEqual(self.vote(_blob(105, 160, 40, 300, rng), ec), "unknown")      # black hand: dark (val_min)
        _, ec = match({"orange": L}, ROBOT_WORLD)
        self.assertEqual(self.vote(_blob(1, 180, 120, 300, rng), ec), "unknown")       # bin wall red: reject 4
        self.p.set_classes(*self.p.world)

    def test_colour_first_uses_episode_classes(self):
        """No depth (colour-first fallback), robot configuration: masks from the episode's classes, detections named
        by them (the base `hsv` ranges know only red / yellow / blue)."""
        from aaltonomous.interface.calibration import Calibration
        from aaltonomous.scene.scene import Scene
        from .mock_world import REAL_COLORS, Renderer
        sc = Scene.load()
        L_ = sc.cfg["layout"]
        x0, y0 = np.mean(L_["region_x"]), np.mean(L_["region_y"])
        h = sc.object_height()
        objs = [dict(color=c, shape="cube", yaw=0.2, size=sc.size, height=h, pos=sc.mat_to_pelvis([x0 + dx, y0, h / 2]))
                for c, dx in (("red", -0.12), ("yellow", 0.0), ("blue", 0.12))]
        rgb, _ = Renderer(sc, Calibration.load(), colors=REAL_COLORS).render(objs)
        for text, want in ((INSTR["OYP"], ["orange", "purple", "yellow"]),
                           (tasks.FIXED_INSTRUCTION, ["blue", "red", "yellow"])):
            _, ec = match(text, ROBOT_WORLD)
            self.p.set_classes(ec.refs, ec.reject, ec.alias)
            dets = self.p.detect(rgb, None)
            self.assertTrue(self.p.path.startswith("colour"))
            self.assertEqual(sorted(d.color for d in dets), want, dets)
        self.p.set_classes(*self.p.world)

    def test_hue_estimate(self):
        rng = np.random.default_rng(1)
        h, sd, (wh, ch) = self.p._hue(_blob(2, 150, 150, 2000, rng, sd=3))
        self.assertAlmostEqual(h, 1.5, delta=0.5)                                  # (uint8 floor)
        self.assertAlmostEqual(sd, 3.0, delta=0.5)
        self.assertEqual(int(ch.sum()), 2000)
        self.assertAlmostEqual(self.p._hue(_blob(179, 150, 150, 2000, rng, sd=3))[0], 178.5, delta=0.5)  # wraps
        self.assertIsNone(self.p._hue(_blob(50, 20, 150, 100, rng))[0])
        # clipped pixels (V >= clip_v) never enter the estimate: a lit face reading 8 units too yellow
        lit = _blob(20, 120, 254, 500, rng)
        lit[:, 2] = 254
        self.assertAlmostEqual(self.p._hue(np.r_[_blob(12, 160, 180, 500, rng), lit])[0], 12.0, delta=1.0)

    def test_real_robot_blob_statistics(self):
        """Data drop 2 (head camera, 189 object blobs): each blob as lit + shaded + median pixels; classes for the
        robot's own words, red / yellow / blue, and synonyms: >= 90 % of the blobs right, and the same blobs right
        under every wording (the words only rename the classes)."""
        with (TESTDATA / "real_drop2_blobs.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 189)
        rng = np.random.default_rng(2)
        pix = [np.r_[_blob(float(r["Hb"]), float(r["Sb"]), float(r["Vb"]), 100, rng),
                     _blob(float(r["Hs"]), float(r["Ss"]), float(r["Vs"]), 100, rng),
                     _blob(float(r["H_med"]), float(r["S_med"]), float(r["V_med"]), 100, rng)] for r in rows]
        ok = {}
        for words in ({"orange": "orange", "yellow": "yellow", "purple": "purple"},
                      {"orange": "red", "yellow": "yellow", "purple": "blue"},
                      {"orange": "amber", "yellow": "gold", "purple": "violet"}):
            text = (f"Put the {words['orange']} objects in the left bin, the {words['yellow']} objects in the center "
                    f"bin and the {words['purple']} objects in the right bin.")
            r, ec = match(text, ROBOT_WORLD)
            want = {k: _COLOR[w] for k, w in words.items()}
            got = [self.vote(x, ec) for x in pix]
            ok[words["orange"]] = [g == want[r_["cls"]] for g, r_ in zip(got, rows)]
            wrong = sum(g not in ("unknown", want[r_["cls"]]) for g, r_ in zip(got, rows))
            print(f"\nreal blobs, words {words}: {sum(ok[words['orange']])}/{len(rows)} right, {wrong} wrong class")
            self.assertGreaterEqual(np.mean(ok[words["orange"]]), 0.9)
        self.assertEqual(ok["orange"], ok["amber"])
        self.assertGreaterEqual(np.mean(np.array(ok["red"]) == np.array(ok["orange"])), 0.97)
        self.p.set_classes(*self.p.world)


class TestRecordedFramesRecoloured(unittest.TestCase):
    """Recorded Isaac frames (capture_frames.py; skipped when absent) with the red objects' pixels hue-shifted to
    orange (H 14, next to the orange bins at 12) and the blue ones to purple (+23): the sim configuration told
    'orange / yellow / purple' (classes matched per episode) finds them under those names."""

    def test_shifted_frames(self):
        import cv2
        files = sorted(Path(os.environ.get("AALTO_FRAMES", "/data/aaro/frames/nominal")).glob("*.npz"))[:30]
        if not files:
            self.skipTest("no recorded frames")
        p = _perception("sim")
        ec = match_colors(["orange", "yellow", "purple"], *p.world)
        p.set_classes(ec.refs, ec.reject, ec.alias)
        name = {"red": "orange", "yellow": "yellow", "blue": "purple"}
        n = right = wrong = extra = 0
        for f in files:
            z = np.load(f)
            objs = json.loads(str(z["objects"]))
            hsv = cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2HSV)
            h, s = hsv[..., 0].astype(int), hsv[..., 1]
            red = (s >= 50) & ((h <= 6) | (h >= 174))
            blue = (s >= 50) & (h >= 98) & (h <= 128)
            hsv[..., 0] = np.where(red, 14, np.where(blue, h + 23, h)).astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            dets = p.detect(rgb, z["depth"])
            used = set()
            for o in objs:
                d = min(dets, key=lambda d: np.linalg.norm(d.top[:2] - np.asarray(o["pos"])[:2]), default=None)
                if d is None or np.linalg.norm(d.top[:2] - np.asarray(o["pos"])[:2]) > 0.03:
                    continue
                used.add(id(d))
                n += 1
                right += d.color == name[o["color"]]
                wrong += d.color not in ("unknown", name[o["color"]])
            extra += sum(id(d) not in used for d in dets)
        print(f"\nrecoloured frames: {right}/{n} objects right, {wrong} wrong, {extra} extra detections")
        self.assertEqual(wrong, 0)
        self.assertGreaterEqual(right, 0.95 * n)
        self.assertEqual(extra, 0)


class TestRealColourEpisodes(unittest.TestCase):
    """Mock episodes, robot configuration (profile none), objects rendered in the real camera's colours
    (mock_world.REAL_COLORS: internal red = orange, blue = purple): told in the robot's words and in the sim's."""

    def episode(self, text, seed=1, profile="none", colors=None):
        from .config import load_config
        from .mock_world import REAL_COLORS, make_mock_env
        from .policy import ClassicalPolicy
        cfg = load_config(profile=profile)
        # (robot: held fingers stop 0.3 rad short of the base cage's closed posture, as in the grasp lab; the mock's
        # default 0.036 is the stiff sim's squeeze, which base verify.empty_below 0.12 calls empty)
        env = make_mock_env(seed=seed, world="real" if profile == "none" else None, cfg=cfg,
                            colors=REAL_COLORS if colors is None else colors,
                            **({"held_gap": 0.3} if profile == "none" else {}))
        env.fixed_instruction = text
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg, time_limit=120.0)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        while True:
            obs, info = env.step(pol.act(obs))
            if info["episode_over"]:
                break
        return pol, env, info

    def check(self, pol, env, info, internal):
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        out = env.backend.outcomes(internal)
        self.assertNotIn("wrong", out, pol.events)
        self.assertGreaterEqual(out.count("correct"), len(out) - 1, (out, pol.events))

    def test_robot_words(self):
        pol, env, info = self.episode("Put the purple objects in the left bin, the orange objects in the center bin "
                                      "and the yellow objects in the right bin.")
        self.assertEqual(pol.colors.names, ["purple", "orange", "yellow"])
        self.check(pol, env, info, {"blue": L, "red": C, "yellow": R})

    def test_sim_words_on_robot_colours(self):
        pol, env, info = self.episode("Red goes to the right bin, yellow to the left bin, blue to the center bin.")
        self.assertEqual(pol.colors.alias["orange"], "red")
        self.check(pol, env, info, {"red": R, "yellow": L, "blue": C})

    def test_sim_configuration_variant_colours(self):
        """Profile sim, objects rendered as the real_like sim variant does (orange / purple), words orange / purple."""
        var = {"red": (240, 120, 20), "blue": (110, 50, 160)}
        pol, env, info = self.episode("Put the orange objects in the left bin, the yellow objects in the center bin "
                                      "and the purple objects in the right bin.", profile="sim_legacy", colors=var)
        self.check(pol, env, info, {"red": L, "yellow": C, "blue": R})


# ---------------------------------------------------------------------------------------------------- calibration
SIMP_PAIR = 7
SIMP = {"sigma_measured": 3.5, "max_shift": 0, "keep_measured": False, "link": 5,   # profile sim's (config.yaml)
        "pair_max": SIMP_PAIR, "pair_cost": 6, "sigma_lexicon": 9}
INTERNAL = {"red": "orange", "yellow": "yellow", "blue": "purple"}   # what our sim variants call its objects
INSTR = {"O": "Put the orange objects in the left bin.",
         "O+": "Put the orange objects in the left bin and everything else in the right bin.",
         "P": "Put the purple objects in the right bin.",
         "P+": "Put the purple objects in the right bin and everything else in the middle bin.",
         "OP": "Put the orange objects in the left bin and the purple objects in the right bin.",
         "OP+": "Orange goes left, purple goes right, everything else goes in the middle.",
         "OYP": "Put the orange objects in the left bin, the yellow objects in the center bin and the purple objects "
                "in the right bin."}


def pure_look(text, world, hues, calib=None, cal=None):
    """Calibrator alone on blobs [hue]; each object labelled as perception would: the override, else the nearest
    class of the calibrated classes if within 17, 4 nearer than any other class and nearer than a reject hue + its
    margin (sim 8, robot 6). -> (labels, bins, cal, line)."""
    r = parse_instruction(text)
    cal = cal or Calibrator(mapping_colors(r.mapping), *world, None, calib, r.words)
    _, line, over = cal.update([(h, 1000.0, 2.0) for h in hues])
    labels, rm = [], 8 if world is SIM_WORLD else 6
    for i, h in enumerate(hues):
        d = sorted((min(hue_dist(h, x) for x in v), k) for k, v in cal.ec.refs.items())
        rej = min((hue_dist(h, x) + rm for x in cal.ec.reject), default=float("inf"))
        ok = d[0][0] <= 17 and (len(d) < 2 or d[1][0] - d[0][0] >= 4) and d[0][0] < rej
        labels.append(over.get(i, d[0][1] if ok else "unknown"))
    return labels, [object_bin(r.mapping, c, None, cal.ec.names) for c in labels], cal, line


class TestCalibrator(unittest.TestCase):
    """colors.Calibrator on blob hues: clusters, one-to-one assignment, guards (pure Python)."""

    def test_clusters(self):
        from .colors import clusters
        cl = clusters([(1, 10, 1), (178, 10, 1), (6, 10, 1), (25, 10, 1), (30, 10, 1), (113, 10, 1), (60, 5, 20)],
                      7.5, 8)
        self.assertEqual(sorted(len(i) for _, _, i in cl), [1, 2, 3])     # (178, 1, 6) wrap; 25-30; 113; 60 too wide
        self.assertAlmostEqual(min(cl, key=lambda c: -len(c[2]))[0], 1.7, delta=0.2)

    def test_sim_orange_without_yellow(self):
        """The C1 weak point: "orange" not next to "yellow" on the sim prior (red 0 / yellow 25) was ambiguous; the
        scene decides it at every shade, for both of our sim variants' palettes."""
        for o, y, pu in ((10, 25, 125), (14, 25, 136), (19, 30, 150)):
            labels, bins, cal, line = pure_look(INSTR["O+"], SIM_WORLD, [o, y, pu], SIMP)
            self.assertEqual(bins, [L, R, R], (o, y, pu, labels, line))
            labels, bins, _, line = pure_look(INSTR["OP"], SIM_WORLD, [o, o + 1, y, pu], SIMP)
            self.assertEqual(bins, [L, L, None, R], (o, y, pu, labels, line))

    def test_sim_tone_mapped_orange(self):
        """The sim camera shows orange albedos ~8 units yellower (19-24) next to yellow at 25-30: two clusters, the
        lower one is orange; alone between the priors, nothing is decided."""
        for o, y in ((17, 29), (19, 29), (21, 29), (23, 29), (24, 30)):
            ok = y - o > SIMP["link"]               # one cluster: cannot be told apart
            labels, bins, cal, line = pure_look(INSTR["OYP"], SIM_WORLD, [o, y, y + 1, 107], SIMP)
            self.assertEqual(bins, [L if ok else None, C, C, R], (o, y, labels, line))
            labels, bins, cal, line = pure_look(INSTR["O+"], SIM_WORLD, [o, o + 1, y], SIMP)
            if y - (o + 1) > SIMP["link"]:
                self.assertEqual(bins, [L, L, R], (o, y, labels, line))
            else:                                   # (the yellow may stay too: never a wrong bin)
                self.assertTrue(bins[:2] == [None, None] and bins[2] in (R, None), (o, y, labels, line))
        for o in (20, 22, 24):
            for text in (INSTR["OYP"], INSTR["O+"]):
                labels, bins, cal, line = pure_look(text, SIM_WORLD, [o], SIMP)
                self.assertEqual(bins, [None], (o, text, labels, line))

    def test_sim_red_yellow_blue_unchanged(self):
        for hues in ([0, 25, 113], [3, 178, 22, 28, 110, 116], [7, 25]):
            labels, bins, cal, line = pure_look(tasks.FIXED_INSTRUCTION, SIM_WORLD, hues, SIMP)
            self.assertEqual(cal.ec.refs, SIM_WORLD[0], line)
            self.assertEqual(labels, [{0: "red", 3: "red", 178: "red", 7: "red", 22: "yellow", 25: "yellow",
                                       28: "yellow", 110: "blue", 113: "blue", 116: "blue"}[h] for h in hues])

    def test_robot_unchanged_and_white_balance(self):
        """Robot classes: the nominal scene changes nothing; the whole scene shifted by +-8 is followed (each class
        to its cluster); one object between orange and yellow is decided by nobody."""
        labels, bins, cal, line = pure_look(INSTR["OYP"], ROBOT_WORLD, [12, 30, 111])
        self.assertEqual((labels, line), (["orange", "yellow", "purple"], None))
        for dl in (-8, 8):
            labels, bins, cal, line = pure_look(INSTR["OYP"], ROBOT_WORLD, [12 + dl, 30 + dl, 111 + dl])
            self.assertEqual(bins, [L, C, R], (dl, line))
        for h in (20, 21, 22):
            labels, bins, _, line = pure_look(INSTR["OYP"], ROBOT_WORLD, [h])
            self.assertEqual(bins, [None], (h, labels, line))

    def test_single_colour_scenes(self):
        """One colour in the scene: decided only when it is clearly one class's."""
        cases = [(SIM_WORLD, SIMP, "Orange left, yellow right.", 21, None),   # between orange 13 and yellow 25
                 (SIM_WORLD, SIMP, "Orange left, yellow right.", 14, L),
                 # the sim camera shows orange albedos anywhere in ~10-27: alone, a yellow-ish object may be either
                 # (decided by an earlier look's two clusters: test_persists_and_never_flips)
                 (SIM_WORLD, SIMP, "Orange left, yellow right.", 25, None),
                 (SIM_WORLD, SIMP, "Orange left, yellow right.", 31, None),
                 (SIM_WORLD, SIMP, INSTR["O+"], 19, None),                   # alone: orange 13 or yellow 25?
                 (SIM_WORLD, SIMP, INSTR["O+"], 13, L),
                 (SIM_WORLD, SIMP, INSTR["P+"], 136, R),
                 (ROBOT_WORLD, None, INSTR["OYP"], 12, L), (ROBOT_WORLD, None, INSTR["OYP"], 30, C),
                 (ROBOT_WORLD, None, INSTR["OYP"], 111, R), (ROBOT_WORLD, None, INSTR["OYP"], 21, None)]
        for world, cp, text, h, want in cases:
            labels, bins, _, line = pure_look(text, world, [h, h + 1], cp)
            self.assertEqual(bins, [want, want], (text, h, labels, line))

    def test_distractor(self):
        """An object of a colour the instruction does not mention, no "everything else": it stays, and the named
        colours are calibrated as without it."""
        for d in (60, 165, 2, 85):
            labels, bins, cal, line = pure_look(INSTR["OYP"], ROBOT_WORLD, [12, 30, 111, d])
            self.assertEqual(bins, [L, C, R, None], (d, labels, line))
            self.assertEqual([cal.ec.refs[k] for k in ("orange", "yellow", "purple")], [[12.0], [30.0], [111.0]])
            labels, bins, cal, line = pure_look(INSTR["OYP"], SIM_WORLD, [19, 30, 150, d], SIMP)
            # (red 2 on the sim prior: orange's measured partner is the old red: 2 and 19 both could be orange)
            self.assertEqual(bins, [None if d == 2 else L, C, R, None], (d, labels, line))

    def test_persists_and_never_flips(self):
        """Observations persist (later looks see fewer objects) and a later cluster far from them is not taken."""
        labels, bins, cal, line = pure_look(INSTR["O+"], SIM_WORLD, [19, 30, 150], SIMP)
        self.assertEqual(bins, [L, R, R], line)
        labels, bins, cal, line = pure_look(INSTR["O+"], SIM_WORLD, [20], SIMP, cal)   # alone now: still orange,
        self.assertEqual((bins, cal.ec.refs["orange"]), ([L], [19.5]), line)        # refined
        labels, bins, cal, line = pure_look(INSTR["O+"], SIM_WORLD, [28], SIMP, cal)   # the yellow one, alone
        self.assertEqual((bins, cal.ec.refs["orange"]), ([R], [19.5]), line)

    def test_later_looks_keep_the_decision(self):
        """m16_pyo: the first look calibrates orange 22.2 / yellow 28.4 / purple 142.8; after yellow and purple are
        delivered the orange cluster alone must still be orange (it was "orange or yellow, not clear"), and mirrored:
        only yellow left, only purple, one object."""
        first = [22.2, 22.5, 28.4, 142.8, 142.5]
        for text in (INSTR["OYP"], INSTR["O+"], INSTR["OP"]):
            _, bins0, cal0, _ = pure_look(text, SIM_WORLD, first, SIMP)
            want = dict(zip(first, bins0))
            for rest in ([22.2, 22.5], [22.0], [28.4], [142.2], [28.4, 142.2], [22.2, 142.2]):
                import copy
                cal = copy.deepcopy(cal0)
                _, bins, cal, line = pure_look(text, SIM_WORLD, rest, SIMP, cal)
                self.assertEqual(bins, [want[min(first, key=lambda f: abs(f - h))] for h in rest], (text, rest, line))
        _, bins, cal, line = pure_look(INSTR["OYP"], ROBOT_WORLD, [20, 38, 119])   # robot: shifted scene ...
        self.assertEqual(bins, [L, C, R], line)
        _, bins, cal, line = pure_look(INSTR["OYP"], ROBOT_WORLD, [20.5], None, cal)   # ... the orange one alone
        self.assertEqual(bins, [L], line)

    def test_undecided_first_look_may_decide_later(self):
        """Nothing was decided (one cluster between the priors): a later look with two clusters decides."""
        _, bins, cal, _ = pure_look(INSTR["OYP"], SIM_WORLD, [22], SIMP)
        self.assertEqual(bins, [None])
        _, bins, cal, line = pure_look(INSTR["OYP"], SIM_WORLD, [22, 28.5], SIMP, cal)
        self.assertEqual(bins, [L, C], line)

    def test_count_check(self):
        """Three objects in one class (the task has <= 2 per colour): logged, the outlier is unknown."""
        from types import SimpleNamespace
        from .colors import calibrate_look

        class P:
            def set_classes(self, *a):
                pass

            def relabel(self, dets):
                pass
        cal = Calibrator(["orange", "yellow"], *ROBOT_WORLD)
        dets = [SimpleNamespace(color="yellow", hue=h, hue_sd=2.0, hue_hist=(None, np.ones(1) * 500))
                for h in (30, 31, 23)]
        _, line = calibrate_look(cal, P(), dets)
        self.assertEqual([d.color for d in dets], ["yellow", "yellow", "unknown"], line)
        self.assertIn("3 objects called yellow", line)


def _shift_frame(rgb, o, y, pu, clip=False):
    """Recorded sim frame with the objects recoloured: red pixels -> hue o (+ their own offset), yellow -> y, blue ->
    pu (the bins' orange 12 untouched). clip: the brightest third of the orange pixels over-exposed as in the bright
    sim variant (red channel clipped: hue + 7, S 70, V 255)."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s = hsv[..., 0].astype(int), hsv[..., 1]
    red = (s >= 50) & ((h <= 6) | (h >= 174))
    yel = (s >= 50) & (h >= 18) & (h <= 34)
    blue = (s >= 50) & (h >= 98) & (h <= 128)
    hn = np.where(red, (np.where(h >= 90, h - 180, h) + o), np.where(yel, h + y - 25, np.where(blue, h + pu - 113, h)))
    hsv[..., 0] = (hn % 180).astype(np.uint8)
    if clip and red.any():
        lit = red & (hsv[..., 2] >= np.percentile(hsv[..., 2][red], 67))
        hsv[..., 0][lit], hsv[..., 1][lit], hsv[..., 2][lit] = (o + 7) % 180, 70, 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _frames(n):
    return sorted(Path(os.environ.get("AALTO_FRAMES", "/data/aaro/frames/nominal")).glob("*.npz"))[:n]


def _score(dets, objs, mapping, names, word=INTERNAL):
    """Per ground-truth object: 'right' (its bin, or no bin when it has none), 'left' (no bin but it has one),
    'wrong' (another bin, or a bin it should not get)."""
    out = []
    for o in objs:
        want = mapping.get(word[o["color"]], mapping.get("default"))
        d = min(dets, key=lambda d: np.linalg.norm(d.top[:2] - np.asarray(o["pos"])[:2]), default=None)
        got = None
        if d is not None and np.linalg.norm(d.top[:2] - np.asarray(o["pos"])[:2]) <= 0.03:
            got = object_bin(mapping, d.color, d.shape, names)
        out.append("right" if got == want else "left" if got is None else "wrong")
    return out


class TestCalibratedFrames(unittest.TestCase):
    """Recorded Isaac frames (skipped when absent), objects hue-shifted to orange / purple shades, through detect +
    calibrate_look as the policy does (one look, a fresh episode per frame)."""

    @classmethod
    def setUpClass(cls):
        cls.files = _frames(int(os.environ.get("AALTO_CAL_FRAMES", 10)))
        if not cls.files:
            raise unittest.SkipTest("no recorded frames")
        from .config import load_config
        cls.cfg = load_config(profile="sim")

    def run_look(self, p, text, rgb, depth, calib=True):
        from .colors import calibrate_look
        r = parse_instruction(text)
        c = self.cfg["perception"]["color"]
        cal = Calibrator(mapping_colors(r.mapping), *p.world, c.get("match"), c["calib"], r.words)
        p.set_classes(cal.ec.refs, cal.ec.reject, cal.ec.alias)
        dets = p.detect(rgb, depth)
        if calib:
            calibrate_look(cal, p, dets)
        return dets, r.mapping, cal.ec.names

    def test_shades_and_instructions(self):
        p = _perception("sim")
        tot = {"right": 0, "left": 0, "wrong": 0}
        rows = []
        combos = [(o, 30 if o == 19 else 25, pu, False) for o in (10, 14, 19) for pu in (125, 136, 150)]
        combos += [(19, 25, 136, False), (14, 25, 136, True), (19, 30, 150, True)]   # orange 6 from yellow; clipped
        for o, y, pu, clip in combos:
            fr = [(z, _shift_frame(z["rgb"], o, y, pu, clip)) for z in (np.load(f) for f in self.files)]
            for key, text in INSTR.items():
                cnt = {"right": 0, "left": 0, "wrong": 0}
                for z, rgb in fr:
                    dets, m, names = self.run_look(p, text, rgb, z["depth"])
                    for s in _score(dets, json.loads(str(z["objects"])), m, names):
                        cnt[s] += 1
                rows.append(f"O{o} Y{y} P{pu}{' clip' if clip else ''} {key:<4} right {cnt['right']:3d} left "
                            f"{cnt['left']:3d} wrong {cnt['wrong']:2d}")
                for k in tot:
                    tot[k] += cnt[k]
        print("\n" + "\n".join(rows) + f"\ntotal {tot}")
        self.assertEqual(tot["wrong"], 0)

    def test_red_yellow_blue_identical_or_better(self):
        """Unshifted frames, the sim's own words: C2 (obj_margin + calibration) vs C1 (neither)."""
        p2, p1 = _perception("sim"), _perception("sim")
        p1.cfg["color"]["obj_margin"] = 0
        res = {}
        for tag, p, cal in (("C1", p1, False), ("C2", p2, True)):
            cnt = {"right": 0, "left": 0, "wrong": 0}
            for f in _frames(int(os.environ.get("AALTO_CAL_FRAMES_RYB", 60))):
                z = np.load(f)
                dets, m, names = self.run_look(p, tasks.FIXED_INSTRUCTION, z["rgb"], z["depth"], calib=cal)
                for s in _score(dets, json.loads(str(z["objects"])), m, names, {c: c for c in INTERNAL}):
                    cnt[s] += 1
            res[tag] = cnt
        print(f"\nred/yellow/blue frames: {res}")
        self.assertLessEqual(res["C2"]["wrong"], res["C1"]["wrong"])
        self.assertGreaterEqual(res["C2"]["right"], res["C1"]["right"])


class TestRealBlobsShifted(unittest.TestCase):
    """Data drop 2 blobs, frame by frame per episode (sessions e0-15, e16-20, e21-33, e34-47), all hues shifted by
    -8 ... +8: C1 (fixed references, vote only), C2 without calibration (vote + obj_margin), C2 calibrated."""

    def test_shifts(self):
        from types import SimpleNamespace
        from .colors import calibrate_look
        from .config import load_config
        p = _perception("none")
        c = load_config(profile="none")["perception"]["color"]
        with (TESTDATA / "real_drop2_blobs.csv").open() as f:
            rows = list(csv.DictReader(f))
        frames = {}
        for r in rows:
            frames.setdefault(int(r["ep"]), {}).setdefault(int(r["f"]), []).append(r)
        said, m = ["orange", "yellow", "purple"], {"orange": L, "yellow": C, "purple": R}
        res = {}
        for dl in (-8, -4, 0, 4, 8):
            rng = np.random.default_rng(3)
            cnt = {m: {"right": 0, "left": 0, "wrong": 0} for m in ("C1", "C2 fixed", "C2 calibrated")}
            for ep in sorted(frames):
                cal = Calibrator(said, *p.world, c.get("match"), c["calib"])
                for fno in sorted(frames[ep]):
                    blobs = frames[ep][fno]
                    pix = [np.r_[_blob(float(r["Hb"]) + dl, float(r["Sb"]), float(r["Vb"]), 100, rng),
                                 _blob(float(r["Hs"]) + dl, float(r["Ss"]), float(r["Vs"]), 100, rng),
                                 _blob(float(r["H_med"]) + dl, float(r["S_med"]), float(r["V_med"]), 100, rng)]
                           for r in blobs]
                    for mode in cnt:
                        ec = cal.ec if mode == "C2 calibrated" else cal.prior
                        p.set_classes(ec.refs, ec.reject, ec.alias)
                        cl = p._cls
                        dets = []
                        for x in pix:
                            col, frac, nv = p._vote(p._classify(x, cl), x[:, 1], cl)
                            col = "unknown" if nv < c["min_votes"] or frac < c["min_frac"] else col
                            h, sd, hh = p._hue(x)
                            if mode != "C1":
                                col = p._decide(col, h, cl)
                            dets.append(SimpleNamespace(color=col, hue=h, hue_sd=sd, hue_hist=hh))
                        if mode == "C2 calibrated":
                            calibrate_look(cal, p, dets)
                        for d, r in zip(dets, blobs):
                            got = object_bin(m, d.color, None, ec.names if mode != "C2 calibrated" else cal.ec.names)
                            cnt[mode]["right" if got == m[r["cls"]] else "left" if got is None else "wrong"] += 1
            res[dl] = cnt
            print(f"\nreal blobs shift {dl:+d}: " + "; ".join(f"{m}: {v['right']} right {v['left']} left "
                                                             f"{v['wrong']} wrong" for m, v in cnt.items()))
        p.set_classes(*p.world)
        for dl, cnt in res.items():
            self.assertLessEqual(cnt["C2 calibrated"]["wrong"], cnt["C1"]["wrong"], dl)
        self.assertLessEqual(res[0]["C2 calibrated"]["wrong"], 2)


class TestTwoLookFrames(unittest.TestCase):
    """Rendered sim variants (orange / yellow / purple objects; skipped when absent): a first look on the whole frame,
    then a second with only one colour / one object left (eval_colors --two-look): no wrong bin, and no object the
    first look sorted right is lost in the second (m16_pyo: the orange ones were re-opened once alone)."""

    def test_sets(self):
        from .eval_colors import run
        root = Path(os.environ.get("AALTO_FRAMES_ROOT", "/data/aaro/frames"))
        sets = [s for s in ("var_sim_pyo", "var_sim_pyo_train") if (root / s).is_dir()]
        if not sets:
            self.skipTest("no rendered variant frames")
        for s in sets:
            cnt, _, _, _, second = run(root / s, "sim", INTERNAL, limit=int(os.environ.get("AALTO_CAL_FRAMES", 10)),
                                       log=lambda *a: None, two_look=True)
            for key, v in second.items():
                self.assertEqual((v["wrong"], v["lost"]), (0, 0), (s, key, v))
                self.assertEqual(cnt[key]["wrong"], 0, (s, key, cnt[key]))


class TestCalibratedEpisodes(unittest.TestCase):
    """Mock episodes through the policy (calibration in its look)."""

    episode = TestRealColourEpisodes.episode
    check = TestRealColourEpisodes.check
    VAR_A = {"red": (240, 120, 20), "blue": (110, 50, 160)}                     # H 14 / 25 / 136
    VAR_B = {"red": (230, 150, 30), "yellow": (220, 220, 40), "blue": (180, 40, 200)}   # H ~19 / 30 / ~148

    def test_sim_orange_without_yellow(self):
        pol, env, info = self.episode(INSTR["O+"], profile="sim_legacy", colors=self.VAR_B)
        self.check(pol, env, info, {"red": L, "default": R})
        self.assertTrue(any(m.startswith("colour calibration") for _, m in pol.events))

    def test_sim_pyo_like_all_looks(self):
        """Orange seen at ~21 next to yellow at ~27 (var_sim_pyo-like), all three named: the first look calibrates
        and the later looks (yellow / purple delivered) keep it: every object sorted (m16_pyo left the oranges)."""
        pol, env, info = self.episode(INSTR["OYP"], profile="sim_legacy",
                                      colors={"red": (230, 170, 30), "yellow": (220, 200, 30), "blue": (150, 40, 200)})
        self.check(pol, env, info, {"red": L, "yellow": C, "blue": R})
        looks = [m for _, m in pol.events if m.startswith("look")]
        self.assertGreater(len(looks), 1)
        self.assertFalse(any("orange/yellow" in m or "yellow/orange" in m for m in looks[1:]), looks)

    def test_sim_orange_and_purple(self):
        pol, env, info = self.episode(INSTR["OP"], profile="sim_legacy", colors=self.VAR_A)
        out = env.backend.outcomes({"red": L, "blue": R})
        self.assertNotIn("wrong", out)
        cols = [o["color"] for o in env.backend.objects]
        self.assertEqual([x for x, c in zip(out, cols) if c == "yellow"], ["counter"] * cols.count("yellow"))

    def test_robot_white_balance(self):
        """Robot configuration, every object hue +8 (white balance)."""
        pol, env, info = self.episode(INSTR["OYP"], colors={"red": (240, 165, 20), "yellow": (176, 210, 40),
                                                            "blue": (60, 65, 200)})   # H ~20 / 36 / 119
        self.check(pol, env, info, {"red": L, "yellow": C, "blue": R})

    def test_distractor_stays(self):
        """Yellow objects rendered green, the instruction names orange and purple only: they stay on the counter."""
        pol, env, info = self.episode(INSTR["OP"], profile="sim_legacy",
                                      colors={**self.VAR_A, "yellow": (40, 170, 60)})
        out = env.backend.outcomes({"red": L, "blue": R})
        self.assertNotIn("wrong", out, pol.events)
        cols = [o["color"] for o in env.backend.objects]
        self.assertEqual([x for x, c in zip(out, cols) if c == "yellow"], ["counter"] * cols.count("yellow"))
        self.assertGreaterEqual(sum(x == "correct" for x, c in zip(out, cols) if c != "yellow"),
                                sum(c != "yellow" for c in cols) - 1)


if __name__ == "__main__":
    unittest.main()
