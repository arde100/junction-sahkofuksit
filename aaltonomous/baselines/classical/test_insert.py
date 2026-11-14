"""Tests of the rung-3 mode (insert.py): recognising the insert task, the lid model vs Scene.lid_holes, the reach
finding (no hole reachable: nothing attempted) and mock episodes (insert mode moves nothing; sort tasks unchanged).
    python -m unittest aaltonomous/baselines/classical/test_insert.py -v
"""
import os
import unittest

import numpy as np

os.environ.setdefault("AALTO_PROFILE", "sim_legacy")   # as test_classical (the mock episodes reuse its run_mock)

from aaltonomous.interface import tasks
from aaltonomous.scene.scene import Scene, in_convex, shape_polygon

from .config import hand_postures, home_arm, load_config
from .insert import LidModel, decide, read_insert, reach_report, select_mode
from .instruction import parse_instruction
from .planner import Planner
from .test_instruction import DEV, HELD, REFUSE

SC = Scene.load()

INSERT = [
    tasks.INSERT_INSTRUCTION,
    "Insert every shape into the matching hole of the lid.",
    "Post each shape through its slot in the lid.",
    "Drop each piece through the hole that matches its shape.",
    "Put the objects through the holes in the lid.",
    "Push each object through the cut-out of the same shape in the cover.",
    "Put each object into the hole of the same shape.",
    "Fit every block into the right hole in the lid.",          # "right hole" = the correct one, not a bin
    "Each object goes through the opening in the lid that has its shape.",
]
MIXED = [   # lid / hole wording + a colour, a bin position, a mapping or a negation: not understood, sort nothing
    "Put each object through the matching hole in the lid on the left box.",
    "Put the red objects through the lid holes and the rest in the right bin.",
    "Put the red cubes in the left bin and everything else through the lid.",
    "Do not put anything through the holes in the lid.",
    "Put the blue objects through the holes in the lid.",
]


def _sentences(items):
    return [x[0] if isinstance(x, tuple) else x for x in items]


class TestRecognition(unittest.TestCase):
    def test_insert_paraphrases(self):
        for t in INSERT:
            with self.subTest(t=t):
                self.assertEqual(read_insert(t).kind, "insert", read_insert(t).summary())

    def test_mixed_is_not_understood(self):
        for t in MIXED:
            with self.subTest(t=t):
                self.assertEqual(read_insert(t).kind, "mixed", read_insert(t).summary())

    def test_sort_instructions_are_not_insert(self):
        """Every sort sentence of the instruction tests (templates, dev / held-back sets, refusals except the insert
        instruction itself) and a bin called a slot stay with the sorter."""
        rng = np.random.default_rng(0)
        texts = [tasks.FIXED_INSTRUCTION] + [tasks.make_instruction("sort_language", rng)[0] for _ in range(200)]
        texts += _sentences(DEV) + _sentences(HELD) + [t for t in REFUSE if t != tasks.INSERT_INSTRUCTION]
        texts.append("Drop the red ones into the left slot, yellow into the middle slot, blue into the right slot.")
        for t in texts:
            with self.subTest(t=t):
                self.assertIsNone(read_insert(t).kind, read_insert(t).summary())
                self.assertIsNone(select_mode({"instruction": t}))

    def test_select_mode(self):
        t = tasks.INSERT_INSTRUCTION
        self.assertEqual(select_mode({"task": "insert_shapes", "instruction": t}).source, "task")
        self.assertEqual(select_mode({"task": "insert_shapes", "instruction": "Sort by colour."}).source, "task")
        self.assertEqual(select_mode({"instruction": t}).source, "instruction")
        self.assertEqual(select_mode({"instruction": MIXED[0]}).source, "mixed")
        self.assertFalse(select_mode({"instruction": MIXED[0]}).understood)
        for task in ("sort_fixed", "sort_language"):   # an explicit sort task id: never the insert mode
            self.assertIsNone(select_mode({"task": task, "instruction": t}))
        self.assertIsNone(select_mode({"instruction": t, "mapping": {"red": "left"}}))
        self.assertIsNone(select_mode({"task": "insert_shapes", "instruction": t}, {"insert": {"recognise": False}}))
        self.assertIsNone(select_mode({}))

    def test_old_parse_would_drop_on_the_lid(self):
        """Why mixed sorts nothing: the sort parser reads this paraphrase as everything -> left (the lid bin)."""
        self.assertEqual(parse_instruction(MIXED[0]).mapping, {"default": "left"})


class TestLidModel(unittest.TestCase):
    def test_holes_match_the_scene(self):
        lid = LidModel(SC)
        holes = {h["shape"]: h for h in SC.lid_holes()}
        self.assertEqual(set(lid.holes), set(holes))
        for s, h in lid.holes.items():
            np.testing.assert_allclose(SC.pelvis_to_mat(np.r_[h["xy"], SC.counter_z])[:2], holes[s]["center"],
                                       atol=1e-9)
        np.testing.assert_allclose(lid.holes["cylinder"]["xy"], [0.5745, 0.1905], atol=1e-3)
        np.testing.assert_allclose(lid.holes["cube"]["xy"], [0.5745, 0.3295], atol=1e-3)
        np.testing.assert_allclose(lid.holes["triangle"]["xy"], [0.522, 0.26], atol=1e-3)
        self.assertAlmostEqual(lid.z_top - SC.counter_z, 0.063, places=6)

    def test_required_yaw_fits_the_hole(self):
        """The object outline (Detection yaw convention = scene.shape_polygon rot, pelvis frame) at the hole's yaw
        lies inside the hole; 15 deg off it does not (cube, triangle)."""
        lid = LidModel(SC)
        for s, h in lid.holes.items():
            a = SC.object_dims(s)["apothem"]
            for dyaw, fits in ((0.0, True), (np.radians(15), s == "cylinder")):
                yaw = (h["yaw"] or 0.0) + dyaw
                for k in range(3):   # every symmetric equivalent
                    P = (lid.PERIOD[s] or 0.0) * k
                    obj = h["xy"] + shape_polygon(s, a, yaw + P)
                    with self.subTest(shape=s, dyaw=dyaw, k=k):
                        self.assertEqual(all(in_convex(self._ccw(h["poly"]), p) for p in obj), fits)
                        self.assertAlmostEqual(lid.yaw_error(s, yaw + P), dyaw if s != "cylinder" else 0.0)

    @staticmethod
    def _ccw(poly):   # the mat -> pelvis map is a mirror: the hole polygon is clockwise in the pelvis frame
        x, y = poly[:, 0], poly[:, 1]
        return poly if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0 else poly[::-1]

    def test_on_lid(self):
        lid = LidModel(SC)
        self.assertTrue(lid.on_lid(lid.holes["triangle"]["xy"]))
        self.assertFalse(lid.on_lid([0.30, 0.0]))


class TestReach(unittest.TestCase):
    """The finding behind 'nothing attempted': no arm holds a top-down release above any hole (sim profile)."""

    def test_no_hole_reachable(self):
        cfg = load_config(profile="sim")
        pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
        rep = reach_report(pl, LidModel(SC), cfg["insert"])
        for s, arms in rep.items():
            for side, (ok, err, tilt) in arms.items():
                with self.subTest(shape=s, side=side):
                    self.assertFalse(ok)
                    self.assertGreater(err, 0.03)   # (left arm: triangle ~38 mm, cylinder ~77, cube ~105 mm short)
        dec = decide(rep, cfg["insert"])
        self.assertTrue(all(not a for a, _ in dec.values()))
        self.assertTrue(all("out of reach" in why for _, why in dec.values()))
        # a reachable hole is still not attempted (no delivery implemented)
        fake = {"cube": {"left": (True, 0.001, 3.0)}}
        self.assertFalse(decide(fake, cfg["insert"])["cube"][0])


class TestEpisodes(unittest.TestCase):
    """Mock episodes (no lid in the mock world: the insert mode moves nothing, so none is needed)."""

    def test_insert_task_moves_nothing(self):
        """Task id insert_shapes, an insert paraphrase, a mixed one (that the sort parser reads as everything ->
        left): no plan is executed, every object stays where it was placed, 0 clips, done."""
        from .test_classical import run_mock
        for text, task in ((tasks.INSERT_INSTRUCTION, "insert_shapes"), (MIXED[0], None), (MIXED[0], "insert_shapes"),
                           (INSERT[2], None)):
            with self.subTest(text=text, task=task):
                rec = {}

                def env_setup(pol, env):
                    orig = env.backend.reset_scene

                    def reset_scene(*a, **kw):
                        out = orig(*a, **kw)
                        rec["before"] = np.array([o["pos"] for o in env.backend.objects], float)
                        return out
                    env.backend.reset_scene = reset_scene
                pol, env, info, starts = run_mock(0, instruction=text, reset_info={"instruction": text, "task": task},
                                                  overrides={"insert.enabled": True}, env_setup=env_setup)
                self.assertEqual(starts, [])
                self.assertEqual(pol.mapping, {})
                self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
                self.assertEqual(info["end_reason"], "done")
                after = np.array([o["pos"] for o in env.backend.objects], float)
                np.testing.assert_allclose(after, rec["before"], atol=1e-9)
                self.assertTrue(pol.events[0][1].startswith("rung 3") if select_mode(
                    {"instruction": text, "task": task}).understood else "not understood" in pol.events[0][1])
                if pol.insert.understood:   # the reach / decision line of the first look
                    self.assertTrue(any("rung 3 lid on the left bin" in m and "out of reach" in m
                                        for _, m in pol.events))

    def test_server_passes_the_task(self):
        """policy_server.reset(instruction, task): the evaluator's task id selects the mode."""
        from .policy_server import ClassicalServerPolicy
        s = ClassicalServerPolicy(profile="sim")
        s.reset(tasks.INSERT_INSTRUCTION, task="insert_shapes", seed=0)
        self.assertEqual((s.pol.insert.source, s.pol.mapping), ("task", {}))
        s.reset(tasks.FIXED_INSTRUCTION, task="sort_fixed", seed=0)
        self.assertIsNone(s.pol.insert)
        self.assertEqual(s.pol.mapping, dict(tasks.FIXED_MAPPING))

    def test_sort_task_unchanged(self):
        """The fixed instruction with task id sort_fixed: mapping and outcome as without the insert mode."""
        from .test_classical import run_mock
        text = tasks.FIXED_INSTRUCTION
        pol, env, info, starts = run_mock(0, instruction=text, reset_info={"instruction": text, "task": "sort_fixed"})
        self.assertIsNone(pol.insert)
        self.assertEqual((pol.mapping, pol.mapping_source), (dict(tasks.FIXED_MAPPING), "parsed"))
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertEqual(env.backend.outcomes(tasks.FIXED_MAPPING), ["correct"] * len(env.backend.objects))


if __name__ == "__main__":
    unittest.main()
