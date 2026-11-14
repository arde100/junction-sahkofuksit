"""instruction_sets.py (no sim): determinism, text / mapping consistency, colour-word renaming, serialisation, and the
evaluator glue (default behaviour unchanged).  python -m unittest aaltonomous/sim/test_instruction_sets.py"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import unittest

import numpy as np

from aaltonomous.interface import tasks
from aaltonomous.interface.tasks import BINS, COLORS, SHAPES
from aaltonomous.scene.scene import Scene, resolve_bin
from aaltonomous.sim import instruction_sets as IS
from aaltonomous.sim.aalto_task import variant as V

W = {"red": "orange", "blue": "purple"}
N = 150


def layouts(n=N):
    sc = Scene.load()
    return [(s, IS.episode_layout(s, sc)) for s in range(n)]


class TestInstructionSets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = layouts()
        cls.D = {(name, dis): [(s, L, IS.draw(name, s, L, dis)) for s, L in cls.L]
                 for name in IS.CHOICES for dis in (False, True) if not (dis and name == "unseen")}

    def test_deterministic(self):
        for (name, dis), ds in self.D.items():
            for s, L, d in ds[:40]:
                self.assertEqual(IS.draw(name, s, L, dis), d)
            self.assertGreater(len({d.text for _, _, d in ds}), N // 3, name)   # seeds differ

    def test_mapping_values_and_keys(self):
        for (name, dis), ds in self.D.items():
            for s, L, d in ds:
                for k, v in d.mapping.items():
                    self.assertTrue(v in BINS or (v is None and d.set == "negation"), (d, k, v))
                    ok = k == "default" or k in COLORS or k in SHAPES or (
                        isinstance(k, tuple) and k[0] in COLORS and k[1] in SHAPES)
                    self.assertTrue(ok, (d, k))
                self.assertEqual(d.present, IS.types_of(L))

    def test_text_names_every_key(self):
        """Generated from the mapping: every colour / shape key's words are in the text, a default key comes with a
        rest / except phrase, and colour-only sets name colours only."""
        shape_words = {s: {w.split()[-2] if w.endswith((" pieces", " ones", " blocks")) else w
                           for w in IS.SHAPE_NOUN[s] + IS.NOVEL_NOUN[s]} | set(IS.SHAPE_ADJ[s]) | {IS.NOVEL_ADJ[s]}
                       for s in SHAPES}
        for (name, dis), ds in self.D.items():
            for s, L, d in ds:
                t = d.text.lower()
                for k in d.mapping:
                    if k == "default":
                        self.assertRegex(t, r"else|rest|other|remaining|left goes|except|but |apart|is not", d)
                        continue
                    c, sh = (k if isinstance(k, tuple) else (k, None) if k in COLORS else (None, k))
                    if c:
                        self.assertRegex(t, rf"\b{c}\b", d)
                    if sh:
                        self.assertTrue(any(w in t for w in shape_words[sh]) or IS.SHAPE_ONE[sh] in t, d)
                if d.set == "unseen":
                    self.assertEqual(sorted(d.mapping), sorted(COLORS))
                    self.assertEqual(sorted(d.mapping.values()), sorted(BINS))
                    self.assertNotIn(d.text, self.training_texts())
                if d.set == "shape":
                    self.assertFalse([k for k in d.mapping if k in COLORS or isinstance(k, tuple)], d)
                if d.set == "partial":
                    self.assertNotIn("default", d.mapping)
                    self.assertTrue(d.stay(IS.TYPES), d)

    @staticmethod
    def training_texts():
        out = set()
        for tpl in tasks.TEMPLATES:
            for c in itertools.permutations(COLORS):
                for b in itertools.permutations(BINS):
                    w = {f"c{i}": x for i, x in enumerate(c)}
                    w.update({f"C{i}": x.capitalize() for i, x in enumerate(c)})
                    w.update({f"b{i}": x for i, x in enumerate(b)})
                    out.add(tpl.format(**w))
        return out

    def test_unseen_templates(self):
        self.assertGreaterEqual(len(IS.UNSEEN), 30)
        for tpl, order in IS.UNSEEN:
            self.assertNotIn(tpl, tasks.TEMPLATES)
            self.assertTrue(all(f"{{c{i}}}" in tpl for i in range(3)), tpl)

    def test_mixed_no_conflict_on_table(self):
        """A type a colour clause and a shape clause send to different bins is never on the table; the override
        (colour, shape) key differs from its colour's bin."""
        for dis in (False, True):
            for s, L, d in self.D[("mixed", dis)]:
                cols = [k for k in d.mapping if k in COLORS]
                shs = [k for k in d.mapping if k in SHAPES]
                for c in cols:
                    for sh in shs:
                        if d.mapping[c] != d.mapping[sh]:
                            self.assertNotIn((c, sh), d.present, d)
                            self.assertNotIn((c, sh), d.valid, d)
                for k in [k for k in d.mapping if isinstance(k, tuple)]:
                    self.assertNotEqual(d.mapping[k], d.mapping[k[0]], d)
                    self.assertIn("but ", d.text.lower())

    def test_both_split(self):
        n = 0
        for s, L, d in self.D[("both", False)]:
            tup = [k for k in d.mapping if isinstance(k, tuple)]
            self.assertTrue(tup and "default" in d.mapping, d)
            if d.kind == "split":
                n += 1
                same = [(a, b) for a in tup for b in tup if a[0] == b[0] and a[1] != b[1]
                        and d.mapping[a] != d.mapping[b]]
                self.assertTrue(same, d)
        self.assertGreater(n, N // 5)

    def test_distractor(self):
        for (name, dis), ds in self.D.items():
            if not dis:
                continue
            for s, L, d in ds:
                self.assertNotIn("default", d.mapping, d)
                self.assertTrue(IS.distracts(d.mapping, d.present), d)
                cs, ss = IS.mentioned(d.mapping)
                self.assertTrue(any(d.bin(c, sh) is None and (c not in cs or sh not in ss) for c, sh in d.present))
        with self.assertRaises(ValueError):
            IS.draw("unseen", 0, None, True)
        with self.assertRaises(ValueError):
            IS.draw("nonsense", 0)

    def test_negation_meanings(self):
        kinds = set()
        for s, L, d in self.D[("negation", False)]:
            kinds.add(d.kind)
            t = d.text.lower()
            if d.kind == "stay_all":
                x = [k for k, v in d.mapping.items() if v is None]
                self.assertEqual(len(x), 1)
                self.assertIsNotNone(d.mapping["default"])
                self.assertTrue(d.stay(IS.TYPES))
            if d.kind == "stay_clauses":
                self.assertRegex(t, r"not|don't|leave|stay")
                self.assertTrue(d.stay(IS.TYPES))
            if d.kind == "except":
                self.assertEqual(len(set(d.mapping.values())), 2)
            if d.kind == "correction":
                self.assertRegex(t, r", not ")
        self.assertEqual(kinds, {"except", "stay_all", "stay_clauses", "correction"})

    def test_color_words(self):
        """Renamed, the texts say orange / purple and never red / blue (no plural colour nouns rename() would miss);
        the told mapping is the internal one renamed."""
        for (name, dis), ds in self.D.items():
            for s, L, d in ds:
                self.assertNotRegex(d.text.lower(), r"\b(red|blue|yellow)[a-z]|\band but\b|\s[,.;]|\.\.", d)
                told = V.rename(d.text, W)
                self.assertNotRegex(told.lower(), r"\b(red|blue)\b", told)
                for c in [k for k in d.mapping if k in W]:
                    self.assertRegex(told.lower(), rf"\b{W[c]}\b")
                self.assertEqual(V.rename(told, V.inverse(W)), d.text)

    def test_serialisation(self):
        from aaltonomous.sim.aalto_task.layout import parse_mapping
        for (name, dis), ds in self.D.items():
            for s, L, d in ds[:50]:
                j = IS.mapping_json(d.mapping)
                self.assertEqual(json.loads(json.dumps(j)), j)
                self.assertEqual(IS.mapping_from_json(j), d.mapping)
                self.assertEqual(parse_mapping(j), d.mapping)     # what the scene service scores with
                json.dumps(d.to_json())
        self.assertEqual(IS.mapping_json({("red", "cube"): "left", "default": None}),
                         {"red/cube": "left", "default": None})

    def test_scoring_of_objects_without_bin(self):
        """The organisers' classify_cube: no bin -> wrong in any box, counter when left."""
        sc = Scene.load()
        m = {"yellow": None, "red": "left", ("blue", "cube"): "right"}
        on_mat = sc.mat_to_pelvis([0.3, 0.15, 0.02])
        for name in sc.bin_names:
            lo, hi = sc.bin_inner_box(name)
            p = sc.mat_to_pelvis((lo + hi) / 2)
            self.assertEqual(sc.classify_cube(p, 0.0, "yellow", m, "cube"), "wrong")
            self.assertEqual(sc.classify_cube(p, 0.0, "blue", m, "cylinder"), "wrong")   # no key at all
            self.assertEqual(sc.classify_cube(p, 0.0, "red", m, "cube"), "correct" if name == "left" else "wrong")
        self.assertEqual(sc.classify_cube(on_mat, 0.0, "yellow", m, "cube"), "counter")
        self.assertEqual(sc.score("sort_language", ["counter", "wrong", "correct"]), 1)

    def test_trial_fields_and_summary(self):
        d = IS.Draw("partial", "colours", "Put the red ones in the left bin.", {"red": "left"},
                    present=(("red", "cube"), ("yellow", "cube")))
        objs = [{"name": "red_cube_0", "color": "red", "shape": "cube"},
                {"name": "yellow_cube_0", "color": "yellow", "shape": "cube"},
                {"name": "blue_cube_0", "color": "blue", "shape": "cube"}]
        outc = [{"name": "red_cube_0", "outcome": "correct"}, {"name": "yellow_cube_0", "outcome": "counter"},
                {"name": "blue_cube_0", "outcome": "wrong"}]
        f = IS.trial_fields(d, objs, outc, W)
        self.assertEqual((f["n_must_stay"], f["n_stayed"]), (2, 1))
        self.assertEqual(f["instruction_told"], "Put the orange ones in the left bin.")
        self.assertEqual(f["mapping_told"], {"orange": "left"})
        t = {**f, "score": 1, "max_score": 2}
        r = IS.summarize([t], "partial", False)
        self.assertEqual((r["objects_must_stay"], r["objects_stayed"], r["objects_wrong"], r["objects_left"]),
                         (2, 1, 1, 1))
        rows = dict(IS.table_rows({**r, "objects_correct": 1}))
        self.assertEqual(rows["had to stay / stayed"], "2 / 1")
        json.dumps(t)


class TestEvaluatorGlue(unittest.TestCase):
    def test_install_on_mock_env(self):
        """Layout as without the option; instruction + env.mapping from draw(); renamed after; seeds reproducible."""
        from aaltonomous.interface.env import AaltoEnv
        from aaltonomous.sim import evaluate as E
        for name in ("all", "both", "unseen"):
            for seed in range(8):
                a = AaltoEnv(backend="mock", task="sort_language", seed=0)
                b = AaltoEnv(backend="mock", task="sort_language", seed=0)
                IS.install(b, name)
                E.rename_instructions(b, W)
                for env in (a, b):
                    env.rng, env.instruction_seed = np.random.default_rng(seed), seed
                oa, ob = a.reset(), b.reset()
                self.assertEqual(b.layout, a.layout)                  # the layout draw is untouched
                d = IS.draw(name, seed, b.layout)
                self.assertEqual(b.mapping, d.mapping)
                self.assertEqual(ob["instruction"], V.rename(d.text, W))
                self.assertEqual(b._obs()["instruction"], ob["instruction"])
                self.assertIs(b.instruction_draw.mapping, b.mapping)

    def test_default_unchanged(self):
        from aaltonomous.interface.env import AaltoEnv
        from aaltonomous.sim import evaluate as E
        a = AaltoEnv(backend="mock", task="sort_language", seed=3)
        self.assertEqual(a.reset.__func__, AaltoEnv.reset)
        ap_err = []
        orig = E.evaluate
        try:
            E.evaluate = lambda a: ap_err.append(a) or {"gate": {"passed": True}}
            self.assertEqual(E.main(["--task", "sort_language"]), 0)
            self.assertIsNone(ap_err[-1].instructions)
            self.assertFalse(ap_err[-1].distractor)
            self.assertEqual(E.main(["--task", "sort_language", "--instructions", "all", "--distractor"]), 0)
            for bad in (["--instructions", "shape"], ["--task", "sort_language", "--instructions", "x"],
                        ["--task", "sort_language", "--distractor"],
                        ["--task", "sort_language", "--instructions", "unseen", "--distractor"]):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    E.main(bad)
        finally:
            E.evaluate = orig
        # the table without the option has no instruction rows
        r = {"trials": 1, "task": "t", "variant": "v", "score_mean": 0, "score_std": 0, "score_frac": 0,
             "objects_correct": 0, "objects": 1, "object_success": 0, "episodes_all_correct": 0, "outcome_counts": {},
             "end_reasons": {}, "safety_clipped_steps_total": 0, "safety_clipped_steps_max": 0, "effective_hz_min": 0,
             "command_hz_min": 0, "gate": {"passed": True, "failed_trials": []}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.print_table(r)
        self.assertNotIn("had to stay", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
