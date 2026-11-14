"""Tests of the instruction reader (instruction.py): templates, paraphrases, refusals. Pure Python, no sim:
    python -m unittest aaltonomous/baselines/classical/test_instruction.py -v
Policy-level tests (mock world) are in test_classical.py (TestInstructionEpisodes).
"""
import itertools
import unittest

from aaltonomous.interface import tasks

from .instruction import BINS, COLORS, SHAPES, object_bin, parse_instruction, resolve_bin

L, C, R = "left", "center", "right"
FIX = {"red": L, "yellow": C, "blue": R}

# Development set: written before / while the parser was built (the parser was developed against it).
# (sentence, expected mapping)
DEV = [
    # ---- colour, object first
    ("Red objects go in the right bin, yellow ones in the left bin and blue ones in the middle bin.",
     {"red": R, "yellow": L, "blue": C}),
    ("Please put all the blue items into the left bin. The yellow items go to the right bin. Red items belong in the "
     "centre bin.", {"blue": L, "yellow": R, "red": C}),
    ("red on the left, yellow in the middle, blue on the right", FIX),
    ("Blue -> left, red -> right, yellow -> middle", {"blue": L, "red": R, "yellow": C}),
    ("Sort by colour: reds to the right, blues to the left, yellows in the centre.", {"red": R, "blue": L, "yellow": C}),
    ("Could you please put the red pieces in the bin on the left, the blue pieces in the bin in the middle and the "
     "yellow pieces in the bin on the right? Thanks!", {"red": L, "blue": C, "yellow": R}),
    ("Yellow goes to the left, blue goes to the right, and red goes in the middle.", {"yellow": L, "blue": R, "red": C}),
    ("Red in the left box, blue in the right box, yellow in the middle box.", {"red": L, "blue": R, "yellow": C}),
    ("Put the yellow objects into the orange box on the left, the red objects into the orange box in the middle and "
     "the blue objects into the orange box on the right.", {"yellow": L, "red": C, "blue": R}),
    ("There are three orange bins in front of you. Put red objects in the right one, yellow objects in the left one, "
     "and blue objects in the middle one.", {"red": R, "yellow": L, "blue": C}),
    ("Place each blue object in the center container and each red object in the left container; yellow objects go "
     "in the right container.", {"blue": C, "red": L, "yellow": R}),
    ("Red goes on the right side, blue on the left side, and yellow in the middle.", {"red": R, "blue": L, "yellow": C}),
    ("Red into the centre, yellow into the left, blue into the right.", {"red": C, "yellow": L, "blue": R}),
    ("Hi robot! Today we are sorting by colour. Please drop the blue things in the left bin, the red things in the "
     "right bin and the yellow things in the middle bin.", {"blue": L, "red": R, "yellow": C}),
    ("Red left, blue right, yellow centre.", {"red": L, "blue": R, "yellow": C}),
    ("red=left, yellow=middle, blue=right", FIX),
    ("The blue objects belong in the right-hand bin, the yellow ones in the left-hand bin and the red ones in the "
     "centre.", {"blue": R, "yellow": L, "red": C}),
    ("Put the yellow stuff on the left and the blue stuff on the right; red goes in between.",
     {"yellow": L, "blue": R, "red": C}),
    ("Kindly place all blue items in the rightmost container, and all yellow items in the leftmost container. Red "
     "items go to the central container.", {"blue": R, "yellow": L, "red": C}),
    ("Red items → left bin, yellow items → middle bin, blue items → right bin.", FIX),
    ("The robot should put red objects into the bin on its left, yellow objects into the bin on its right, and blue "
     "objects into the middle bin.", {"red": L, "yellow": R, "blue": C}),
    ("Throw the reds in the left bin and the blues in the right bin, yellows in the middle.",
     {"red": L, "blue": R, "yellow": C}),
    ("red objects -> center, blue objects -> left, yellow objects -> right", {"red": C, "blue": L, "yellow": R}),
    ("Each red item goes in the right bin; each blue item goes in the left bin; each yellow item goes in the middle "
     "bin.", {"red": R, "blue": L, "yellow": C}),
    ("Red objects: the bin on the left. Blue objects: the bin on the right. Yellow objects: the bin in the middle.",
     {"red": L, "blue": R, "yellow": C}),
    ("Red belongs left and blue belongs right, while yellow belongs in the middle.", {"red": L, "blue": R, "yellow": C}),
    ("Blue things in the left bin, yellow things in the right bin. Red things in the middle bin.",
     {"blue": L, "yellow": R, "red": C}),
    ("Put the blue objects into the left bin and the yellow objects into the right bin. Put the red objects into the "
     "bin in the middle.", {"blue": L, "yellow": R, "red": C}),
    ("Sort the objects so that red ends up in the left bin, blue in the middle bin and yellow in the right bin.",
     {"red": L, "blue": C, "yellow": R}),
    ("The objects that are red go on the right, the ones that are yellow go in the middle, and the ones that are blue "
     "go on the left.", {"red": R, "yellow": C, "blue": L}),
    ("red → middle; blue → left; yellow → right", {"red": C, "blue": L, "yellow": R}),
    ("Put red things to the left and yellow things to the right. Blue things go in the middle.",
     {"red": L, "yellow": R, "blue": C}),
    ("The red pieces belong to the left box, the yellow pieces to the box in the middle and the blue pieces to the "
     "right box.", FIX),
    # ---- colour, bin first
    ("The right bin is for red, the left bin is for blue, and the middle bin is for yellow.",
     {"red": R, "blue": L, "yellow": C}),
    ("Use the leftmost bin for yellow things, the rightmost for red things and the central one for blue things.",
     {"yellow": L, "red": R, "blue": C}),
    ("Left: blue. Middle: red. Right: yellow.", {"blue": L, "red": C, "yellow": R}),
    ("The bin on the right takes the yellow blocks; the one in the middle takes the red blocks; the left one takes "
     "the blue blocks.", {"yellow": R, "red": C, "blue": L}),
    ("Into the left bin go the blue ones, into the middle bin the yellow ones, and into the right bin the red ones.",
     {"blue": L, "yellow": C, "red": R}),
    ("The middle bin gets the red objects, the left bin gets the yellow objects and the right bin gets the blue "
     "objects.", {"red": C, "yellow": L, "blue": R}),
    ("Left bin: red and blue. Right bin: yellow.", {"red": L, "blue": L, "yellow": R}),
    ("To the left go the red objects, to the right the blue ones, and to the middle the yellow ones.",
     {"red": L, "blue": R, "yellow": C}),
    ("Left for blue, right for red, middle for yellow.", {"blue": L, "red": R, "yellow": C}),
    ("On the left: yellow. On the right: blue. In the middle: red.", {"yellow": L, "blue": R, "red": C}),
    ("In the left bin, put the yellow objects. In the right bin, put the red objects. In the middle bin, put the blue "
     "objects.", {"yellow": L, "red": R, "blue": C}),
    ("For red, use the right bin; for blue, use the left bin; for yellow, use the middle bin.",
     {"red": R, "blue": L, "yellow": C}),
    ("Left bin – red. Centre bin – blue. Right bin – yellow.", {"red": L, "blue": C, "yellow": R}),
    ("Left bin = red, middle bin = yellow, right bin = blue", FIX),
    ("The leftmost box gets the blue pieces, the rightmost box gets the yellow pieces, and the middle box gets the "
     "red pieces.", {"blue": L, "yellow": R, "red": C}),
    # ---- lists / respectively
    ("Red, yellow and blue objects go to the right, left and middle bins respectively.",
     {"red": R, "yellow": L, "blue": C}),
    ("Put the red, yellow and blue objects in the left, center and right bins.", FIX),
    ("1. Red -> right bin\n2. Yellow -> center bin\n3. Blue -> left bin", {"red": R, "yellow": C, "blue": L}),
    ("- red: left\n- yellow: right\n- blue: middle", {"red": L, "yellow": R, "blue": C}),
    ("Sort the items by colour. Red items: left bin. Yellow items: middle bin. Blue items: right bin.", FIX),
    ("Red: right bin; blue: centre bin; yellow: left bin.", {"red": R, "blue": C, "yellow": L}),
    ("Instructions:\n1) red objects -> left bin\n2) blue objects -> right bin\n3) yellow objects -> middle bin",
     {"red": L, "blue": R, "yellow": C}),
    ("* Left bin: yellow\n* Middle bin: blue\n* Right bin: red", {"yellow": L, "blue": C, "red": R}),
    ("Left bin:\n- red objects\n- blue objects\nRight bin:\n- yellow objects", {"red": L, "blue": L, "yellow": R}),
    ("The red ones? Left. The yellow ones? Right. The blue ones? Middle.", {"red": L, "yellow": R, "blue": C}),
    # ---- shared bins
    ("Put red and yellow in the left bin and blue in the right bin.", {"red": L, "yellow": L, "blue": R}),
    ("Red and blue go left, yellow goes right.", {"red": L, "blue": L, "yellow": R}),
    ("The left bin is for red and yellow objects; the right bin is for blue ones.", {"red": L, "yellow": L, "blue": R}),
    ("The red objects should be placed in the left bin, while the yellow and blue objects should be placed in the "
     "right bin.", {"red": L, "yellow": R, "blue": R}),
    ("Blue objects are to be put in the middle bin; red and yellow objects are to be put in the left bin.",
     {"blue": C, "red": L, "yellow": L}),
    ("The yellow and the blue objects go into the right bin, the red objects into the left one.",
     {"yellow": R, "blue": R, "red": L}),
    ("Red goes left. Yellow goes left too. Blue goes right.", {"red": L, "yellow": L, "blue": R}),
    ("Red objects go in the left bin, and so do the yellow ones. Blue ones go in the right bin.",
     {"red": L, "yellow": L, "blue": R}),
    ("Red, yellow and blue all go in the left bin.", {"red": L, "yellow": L, "blue": L}),
    ("Put the red and the yellow objects in the right bin, the blue objects in the left bin.",
     {"red": R, "yellow": R, "blue": L}),
    ("The middle bin is for blue and red objects, the left bin for yellow objects.", {"blue": C, "red": C, "yellow": L}),
    ("Yellow and red into the right bin, blue into the left bin.", {"yellow": R, "red": R, "blue": L}),
    # ---- "everything else"
    ("All red objects belong in the middle bin. Everything else goes into the left bin.", {"red": C, "default": L}),
    ("Move every yellow piece to the far right bin and every other piece to the far left bin.",
     {"yellow": R, "default": L}),
    ("Everything goes in the left bin.", {"default": L}),
    ("Put all the objects in the right bin.", {"default": R}),
    ("Yellow objects in the left bin, all the others in the right bin.", {"yellow": L, "default": R}),
    ("Blue to the left; the rest to the right.", {"blue": L, "default": R}),
    ("Put the red ones on the left. Whatever is left goes in the middle.", {"red": L, "default": C}),
    ("Yellow goes in the right bin and all remaining objects go in the left bin.", {"yellow": R, "default": L}),
    ("Red: left bin. Anything else: right bin.", {"red": L, "default": R}),
    ("Put the yellow ones in the middle, and whatever's left on the right.", {"yellow": C, "default": R}),
    ("Blue objects to the left, the remaining ones to the right.", {"blue": L, "default": R}),
    ("Red goes left; others go right.", {"red": L, "default": R}),
    ("If an object is red, put it in the left bin; if it is blue, put it in the right bin; otherwise put it in the "
     "middle bin.", {"red": L, "blue": R, "default": C}),
    ("Put red objects in the left bin, yellow objects in the right bin and blue objects in the remaining bin.",
     {"red": L, "yellow": R, "blue": C}),
    # ---- shapes
    ("Cubes in the left bin, cylinders in the middle, triangles on the right.", {"cube": L, "cylinder": C, "triangle": R}),
    ("Put the cubes in the right bin, the cylinders in the left bin and the triangular prisms in the middle bin.",
     {"cube": R, "cylinder": L, "triangle": C}),
    ("The left bin is for cubes, the middle bin is for cylinders and the right bin is for triangles.",
     {"cube": L, "cylinder": C, "triangle": R}),
    ("All the cylinders go to the right, all the cubes to the left and the prisms in the centre.",
     {"cylinder": R, "cube": L, "triangle": C}),
    ("Round objects go in the left bin, square ones in the right bin, and triangular ones in the middle bin.",
     {"cylinder": L, "cube": R, "triangle": C}),
    ("Wedges on the left, cans in the middle, cubes on the right.", {"triangle": L, "cylinder": C, "cube": R}),
    ("Cubes go left and everything else goes right.", {"cube": L, "default": R}),
    ("Put every cylinder into the middle bin, and the other shapes into the left bin.", {"cylinder": C, "default": L}),
    ("The triangles belong in the right bin. Cubes and cylinders belong in the left bin.",
     {"triangle": R, "cube": L, "cylinder": L}),
    ("Use the middle bin for the cubes, the left bin for the triangles and the right bin for the cylinders.",
     {"cube": C, "triangle": L, "cylinder": R}),
    ("Square blocks to the left, round blocks to the right, triangular blocks in the middle.",
     {"cube": L, "cylinder": R, "triangle": C}),
    ("Cubes: right. Cylinders: left. Prisms: middle.", {"cube": R, "cylinder": L, "triangle": C}),
    ("Please sort the objects by shape: cubes in the centre bin, cylinders in the right bin, triangles in the left bin.",
     {"cube": C, "cylinder": R, "triangle": L}),
    ("The cube-shaped objects go into the left bin, the cylindrical ones into the middle bin and the triangular ones "
     "into the right bin.", {"cube": L, "cylinder": C, "triangle": R}),
    ("All triangular prisms go into the left bin; all cylinders go into the right bin; all cubes go into the middle "
     "bin.", {"triangle": L, "cylinder": R, "cube": C}),
    ("Discs left, wedges right, cubes centre.", {"cylinder": L, "triangle": R, "cube": C}),
    ("Cylinders go in the left bin, everything else in the right bin.", {"cylinder": L, "default": R}),
    # ---- mixed colour / shape
    ("Red cubes go left, everything else goes right.", {("red", "cube"): L, "default": R}),
    ("Put the blue cylinders in the middle bin and all other objects in the left bin.",
     {("blue", "cylinder"): C, "default": L}),
    ("Yellow triangles in the right bin, red objects in the left bin, the rest in the middle bin.",
     {("yellow", "triangle"): R, "red": L, "default": C}),
    ("The left bin is for red cubes and blue cylinders; the right bin is for everything else.",
     {("red", "cube"): L, ("blue", "cylinder"): L, "default": R}),
    ("Red and blue cubes go in the left bin, and all the remaining objects go in the right bin.",
     {("red", "cube"): L, ("blue", "cube"): L, "default": R}),
    ("Put cubes in the left bin, but red ones go in the right bin.", {"cube": L, "red": R}),
    ("Blue objects in the middle, yellow cylinders on the left, everything else on the right.",
     {"blue": C, ("yellow", "cylinder"): L, "default": R}),
    ("Cylinders go to the left bin, red cubes to the right bin, and any other object to the middle bin.",
     {"cylinder": L, ("red", "cube"): R, "default": C}),
    ("Triangles left; blue right; the rest centre.", {"triangle": L, "blue": R, "default": C}),
    # ---- spelling, register, context
    ("Put the red objects in the centre bin, the blue objects in the left bin and the yellow objects in the right bin.",
     {"red": C, "blue": L, "yellow": R}),
    ("Place the yellow coloured objects in the centre, the red coloured in the left and blue coloured in the right.",
     {"yellow": C, "red": L, "blue": R}),
    ("Sort the colors: blue in the center, yellow on the left, red on the right.", {"blue": C, "yellow": L, "red": R}),
    ("Red objects are to be placed in the middle bin, yellow objects in the right bin, and blue objects in the left "
     "bin.", {"red": C, "yellow": R, "blue": L}),
    ("The blue items must be dropped into the left bin; the red items must be dropped into the right bin; the yellow "
     "items must be dropped into the centre bin.", {"blue": L, "red": R, "yellow": C}),
    ("I want the yellow things on the right, the blue things on the left and the red things in the middle.",
     {"yellow": R, "blue": L, "red": C}),
    ("Your task is to sort the objects: red ones go in the right bin, yellow ones in the middle bin and blue ones in "
     "the left bin.", {"red": R, "yellow": C, "blue": L}),
    ("PUT THE BLUE OBJECTS IN THE RIGHT BIN AND THE RED OBJECTS IN THE LEFT BIN. THE YELLOW OBJECTS GO IN THE CENTER "
     "BIN.", {"blue": R, "red": L, "yellow": C}),
    ("Hello! The three orange boxes in front of you are the bins. Please put red objects in the left box, blue ones "
     "in the right box and yellow ones in the middle box. Thank you!", FIX),
    ("This is a sorting task. Be careful with the objects. Blue goes to the middle bin, red to the right bin and "
     "yellow to the left bin.", {"blue": C, "red": R, "yellow": L}),
    ("When you are ready, start by putting the red items in the right bin; after that, the blue items in the middle "
     "bin; finally, the yellow items in the left bin.", {"red": R, "blue": C, "yellow": L}),
    ("Drop yellow in the left container, red in the central container and blue in the right container.",
     {"yellow": L, "red": C, "blue": R}),
    ("Red in the left basket, blue in the middle basket, yellow in the right basket.", {"red": L, "blue": C, "yellow": R}),
    ("Put red objects in the left-most bin, blue in the right-most bin and yellow in the centre-most bin.",
     {"red": L, "blue": R, "yellow": C}),
    ("Sort them: blue in the middle, red on the right, yellow on the left.", {"blue": C, "red": R, "yellow": L}),
    # ---- typos
    ("Put the yelow objects in the left bin, the red objects in the rigth bin and the blue objects in the midle bin.",
     {"yellow": L, "red": R, "blue": C}),
    ("Bleu cubes go to the left bin, everything else to the right bin.", {("blue", "cube"): L, "default": R}),
    ("Cubes left, cylinders right, traingles centre.", {"cube": L, "cylinder": R, "triangle": C}),
]

# Must not give a confident mapping (r.ok False): contradiction, negation, nonsense, another task, unknown colour.
REFUSE = [
    "Red goes left. Red goes right.",
    "Put the red objects in the left bin, not the right bin.",
    "Put everything except the red objects in the bin.",   # ("... in the left bin" is read now: red stays, the
    # rest goes left: TestStayAndExcept)
    "Don't put anything in the middle bin.",
    "Do not sort the blue objects into the right bin.",
    "Wave to the audience.",
    "Stack the cubes on top of each other.",
    "Banana helicopter purple sky.",
    "Put the white objects in the left bin.",   # (green / orange objects: open vocabulary, test_colors.py)
    "Put the red objects in the left or the right bin.",
    "Put the red objects in the bin.",
    "Put the objects in the bins.",
    "Red goes in the left bin and also in the right bin.",
    "Put the red objects in the first bin and the blue objects in the second bin.",
    tasks.INSERT_INSTRUCTION,
    "Sort the objects by colour.",
    "Put red objects in the left bin unless they are cubes.",
    "",
    "Left, right, center.",
]

# Held-back set: written AFTER the parser was finished, then run once (result: TestParaphrases.test_held_back_set).
HELD = [
    ("Please sort the objects by colour: the blue ones belong in the leftmost bin, the red ones in the rightmost bin "
     "and the yellow ones in the bin in the middle.", {"blue": L, "red": R, "yellow": C}),
    ("Each yellow object should be dropped into the bin on the right; red objects go into the middle bin; anything "
     "blue goes to the left bin.", {"yellow": R, "red": C, "blue": L}),
    ("Bins are named by position. Left: yellow objects. Centre: blue objects. Right: red objects.",
     {"yellow": L, "blue": C, "red": R}),
    ("Take every red item and put it in the middle bin, then put the yellow items in the left bin and the blue items "
     "in the right bin.", {"red": C, "yellow": L, "blue": R}),
    ("The cubes belong in the middle bin and the cylinders in the right bin; put the triangular prisms in the left "
     "bin.", {"cube": C, "cylinder": R, "triangle": L}),
    ("Sort by shape — cylinders on the left, cubes on the right, prisms in the middle.",
     {"cylinder": L, "cube": R, "triangle": C}),
    ("Red objects should go into the right-hand bin. All the others go into the left-hand bin.",
     {"red": R, "default": L}),
    ("Put the yellow cylinders in the left bin, and everything else in the middle one.",
     {("yellow", "cylinder"): L, "default": C}),
    ("Your job: blue → middle bin, yellow → right bin, red → left bin.", {"blue": C, "yellow": R, "red": L}),
    ("Put the objects into the bins as follows. Red: middle. Blue: left. Yellow: right.",
     {"red": C, "blue": L, "yellow": R}),
    ("Use the bin on the left for the blue and the yellow objects, and the one on the right for the red objects.",
     {"blue": L, "yellow": L, "red": R}),
    ("All cubes go to the left bin regardless of their colour; all cylinders go to the right bin; all triangles go to "
     "the centre bin.", {"cube": L, "cylinder": R, "triangle": C}),
    ("I'd like the red pieces in the centre, the yellow pieces on the right-hand side and the blue pieces on the "
     "left-hand side.", {"red": C, "yellow": R, "blue": L}),
    ("Blue and yellow objects go together in the right bin, and red objects go in the left bin.",
     {"blue": R, "yellow": R, "red": L}),
    ("Into the middle bin: everything that is red. Into the right bin: everything that is yellow. Into the left bin: "
     "everything that is blue.", {"red": C, "yellow": R, "blue": L}),
    ("The rightmost bin is for blue things, the leftmost for yellow things, and red things go in the central bin.",
     {"blue": R, "yellow": L, "red": C}),
    ("Move the triangles to the right bin and the cubes to the left bin. The cylinders go in the middle.",
     {"triangle": R, "cube": L, "cylinder": C}),
    ("Put red blocks in the left box, yellow blocks in the centre box and blue blocks in the right box.", FIX),
    ("Place the red cubes in the right bin and all remaining objects in the left bin.",
     {("red", "cube"): R, "default": L}),
    ("Red, blue and yellow go to the middle, left and right bins respectively.", {"red": C, "blue": L, "yellow": R}),
    ("Robot, please put the yellow objects in the left bin. After that, place the red ones in the right bin, and "
     "finally the blue ones in the middle bin.", {"yellow": L, "red": R, "blue": C}),
    ("Left bin — red and yellow. Right bin — blue.", {"red": L, "yellow": L, "blue": R}),
    ("Every blue piece belongs on the right, every red piece on the left, and every yellow piece in the middle.",
     {"blue": R, "red": L, "yellow": C}),
    ("The yellow objects go in the right bin. Put the blue objects in the right bin too. The red objects go in the "
     "left bin.", {"yellow": R, "blue": R, "red": L}),
    ("Sort the toys: red ones to the middle container, blue ones to the left container, yellow ones to the right "
     "container.", {"red": C, "blue": L, "yellow": R}),
    ("Cylinders belong in the left bin. Everything else, put in the right bin.", {"cylinder": L, "default": R}),
    ("The blue cylinders and the red cubes go in the middle bin; the rest goes in the left bin.",
     {("blue", "cylinder"): C, ("red", "cube"): C, "default": L}),
    ("Red objects in the middle, please. Blue ones on the left side and yellow ones on the right side.",
     {"red": C, "blue": L, "yellow": R}),
    ("Put the round objects in the left bin, the square objects in the middle bin and the triangular objects in the "
     "right bin.", {"cylinder": L, "cube": C, "triangle": R}),
    ("Drop the blue items in the bin to the right, the yellow items in the bin to the left, and the red items in the "
     "bin in the centre.", {"blue": R, "yellow": L, "red": C}),
    ("Here is how to sort them. Red goes in the left bin. Yellow and blue both go in the right bin.",
     {"red": L, "yellow": R, "blue": R}),
    ("Please place the red cylinders and the red triangles in the right bin, the red cubes in the left bin, and all "
     "other objects in the middle bin.",
     {("red", "cylinder"): R, ("red", "triangle"): R, ("red", "cube"): L, "default": C}),
]


def _template_sentences():
    for tpl in tasks.TEMPLATES:
        for cs in itertools.permutations(COLORS):
            for bs in itertools.permutations(BINS):
                w = {f"c{i}": c for i, c in enumerate(cs)}
                w.update({f"C{i}": c.capitalize() for i, c in enumerate(cs)})
                w.update({f"b{i}": tasks.bin_phrase(b) for i, b in enumerate(bs)})
                yield tpl.format(**w), dict(zip(cs, bs))


def run_set(cases):
    """[(sentence, expected, got Parse)] of the failures."""
    out = []
    for s, exp in cases:
        r = parse_instruction(s)
        if r.mapping != exp:
            out.append((s, exp, r))
    return out


class TestTemplates(unittest.TestCase):
    def test_every_template_sentence(self):
        """All 8 x 6 x 6 training sentences (as tasks.make_instruction builds them) and the fixed one: exact
        mapping, ok, no problem at all."""
        n = 0
        for s, exp in _template_sentences():
            r = parse_instruction(s)
            self.assertEqual(r.mapping, exp, s)
            self.assertTrue(r.ok, s)
            self.assertEqual(r.problems, [], s)
            n += 1
        self.assertEqual(n, 288)
        r = parse_instruction(tasks.FIXED_INSTRUCTION)
        self.assertEqual((r.mapping, r.ok, r.problems), (tasks.FIXED_MAPPING, True, []))

    def test_make_instruction(self):
        """tasks.make_instruction draws (as the env does): parsed == its mapping."""
        import numpy as np
        rng = np.random.default_rng(0)
        for _ in range(200):
            s, m = tasks.make_instruction("sort_language", rng)
            self.assertEqual(parse_instruction(s).mapping, m, s)

    def test_evidence(self):
        r = parse_instruction(tasks.FIXED_INSTRUCTION)
        ev = {k["key"]: k["evidence"] for k in r.keys}
        self.assertEqual(ev["red"], "red objects in the left bin")
        self.assertEqual({k["conf"] for k in r.keys}, {1.0})


class TestParaphrases(unittest.TestCase):
    def test_dev_set(self):
        """Development paraphrases: every one passes (report: failures printed)."""
        self.assertGreaterEqual(len(DEV), 120)
        fails = run_set(DEV)
        for s, exp, r in fails:
            print(f"\nDEV FAIL: {s!r}\n  expected {exp}\n  got      {r.mapping}  [{r.summary()}]")
        print(f"\ndev set: {len(DEV) - len(fails)}/{len(DEV)} exact")
        self.assertEqual(len(fails), 0)

    def test_held_back_set(self):
        """Written after the parser was finished; first (only) run before this test existed: 32/32 exact."""
        self.assertGreaterEqual(len(HELD), 30)
        fails = run_set(HELD)
        for s, exp, r in fails:
            print(f"\nHELD FAIL: {s!r}\n  expected {exp}\n  got      {r.mapping}  [{r.summary()}]")
        self.assertEqual(len(fails), 0)

    def test_refusals(self):
        for s in REFUSE:
            r = parse_instruction(s)
            self.assertFalse(r.ok, f"{s!r}: {r.summary()}")

    def test_partial_negation_keeps_the_rest(self):
        """A negated clause is dropped (and makes the result not ok), the other clauses stay."""
        r = parse_instruction("Put the red objects in the left bin, not the right bin.")
        self.assertEqual(r.mapping, {"red": L})
        self.assertIn("negation", [p["kind"] for p in r.problems])

    def test_conflict_dropped(self):
        r = parse_instruction("Red goes left, blue goes right. Red goes in the middle.")
        self.assertEqual(r.mapping, {"blue": R})
        self.assertEqual([p["kind"] for p in r.problems], ["conflict"])

    def test_box_and_block(self):
        """'box(es)' = bins, never cubes; 'blocks' = any object; 'square blocks' = cubes."""
        self.assertEqual(parse_instruction("Put the red blocks in the left box.").mapping, {"red": L})
        self.assertEqual(parse_instruction("Square blocks go in the right box.").mapping, {"cube": R})
        self.assertEqual(parse_instruction("Put everything into the orange box on the left.").mapping, {"default": L})

    def test_left_and_right_senses(self):
        self.assertEqual(parse_instruction("Red goes right away into the left bin.").mapping, {"red": L})
        self.assertEqual(parse_instruction("Using your left hand, put red in the right bin.").mapping, {"red": R})
        self.assertEqual(parse_instruction("Red goes left, and anything left over goes right.").mapping,
                         {"red": L, "default": R})
        self.assertEqual(parse_instruction("Red goes to the left bin. As for yellow, the middle bin. Blue: whichever "
                                           "bin is left.").mapping, {"red": L, "yellow": C, "blue": R})

    def test_unknown_adjective_is_not_everything(self):
        """'the objects' alone = everything, but an unknown word before the noun may be a misspelt colour."""
        self.assertEqual(parse_instruction("Put the objects in the left bin.").mapping, {"default": L})
        self.assertEqual(parse_instruction("Put the shiny objects in the left bin.").mapping, {})
        self.assertEqual(parse_instruction("Rde objects go left.").mapping, {"red": L})

    def test_typo_not_on_real_words(self):
        """'lift' / 'light' / 'let' are never read as left / right."""
        r = parse_instruction("Lift the red objects and let them down in the left bin.")
        self.assertEqual(r.mapping, {"red": L})
        self.assertEqual(parse_instruction("Lift the light blue objects into the bin.").mapping, {})

    def test_one_shape_warning(self):
        r = parse_instruction("Put the red cubes in the left bin, the yellow cubes in the middle and the blue cubes on "
                              "the right.")
        self.assertEqual(r.mapping, {("red", "cube"): L, ("yellow", "cube"): C, ("blue", "cube"): R})
        self.assertIn("one_shape", [p["kind"] for p in r.problems])


class TestCountedAndBetween(unittest.TestCase):
    """Bins counted with a stated direction ("the second bin from the left"; the direction carries over to later
    ordinals), and the bin between the other two (= center)."""
    COUNTED = [
        ("Place the red objects in the first bin from the left, the yellow objects in the second, and the blue objects "
         "in the third.", FIX),
        ("Counting from the right, the first bin is for red, the second for yellow and the third for blue.",
         {"red": R, "yellow": C, "blue": L}),
        ("Starting from the left, the first bin takes yellow objects, the second takes blue and the third takes red.",
         {"yellow": L, "blue": C, "red": R}),
        ("Red goes in the third bin from the left; blue goes in the first bin counting from the right.",
         {"red": R, "blue": R}),
        ("Put blue in the second bin from the right and red in the first one from the left.", {"blue": C, "red": L}),
        ("Put red in the first bin from the right, blue in the third, yellow in the second.",
         {"red": R, "blue": L, "yellow": C}),
        ("Red goes in the second bin from the left. Blue goes in the first.", {"red": C, "blue": L}),
        ("Cubes go in the 1st bin from the left, cylinders in the 2nd and triangles in the 3rd.",
         {"cube": L, "cylinder": C, "triangle": R}),
    ]
    BETWEEN = [
        ("The yellow objects go in the bin furthest to the left, the red ones in the bin furthest to the right, and "
         "the blue ones in the one between them.", {"yellow": L, "red": R, "blue": C}),
        ("Yellow in the leftmost bin, red in the rightmost bin, blue in the bin between the other two.",
         {"yellow": L, "red": R, "blue": C}),
        ("Red left, yellow right, blue in the one in between.", {"red": L, "yellow": R, "blue": C}),
        ("Blue goes in the bin between the left and right bins, red goes left.", {"blue": C, "red": L}),
        ("The bin between the others is for cubes; everything else goes right.", {"cube": C, "default": R}),
    ]
    AMBIGUOUS = [   # no direction stated: ordinal keys are dropped (not ok)
        ("Put red in the first bin, yellow in the second bin and blue in the third bin.", {}),
        ("The first bin is for red, the second for yellow and the third for blue.", {}),
        ("Red in the first bin and blue in the second.", {}),
        ("The left bin is for red, the second for yellow.", {"red": L}),
    ]

    def test_counted(self):
        for s, exp in self.COUNTED:
            self.assertEqual(parse_instruction(s).mapping, exp, s)

    def test_between(self):
        for s, exp in self.BETWEEN:
            self.assertEqual(parse_instruction(s).mapping, exp, s)

    def test_ambiguous_ordinals(self):
        for s, exp in self.AMBIGUOUS:
            r = parse_instruction(s)
            self.assertEqual(r.mapping, exp, s)
            self.assertFalse(r.ok, s)

    def test_not_a_bin(self):
        """'first' as a step word and 'between' about objects name no bin."""
        self.assertEqual(parse_instruction("First, red goes left; then yellow goes right.").mapping,
                         {"red": L, "yellow": R})
        self.assertEqual(parse_instruction("In the first step put red in the left bin.").mapping, {"red": L})
        self.assertEqual(parse_instruction("Leave a gap between the objects and put red in the left bin.").mapping,
                         {"red": L})
        self.assertEqual(parse_instruction("Red objects go between the cubes.").mapping, {})


N = None   # "stays on the table" (scene.resolve_bin: a key with value None = no bin)


class TestStayAndExcept(unittest.TestCase):
    """'Stay' is an outcome: 'everything but / except / apart from X', 'X stay ...', 'leave X ...', 'don't touch X',
    'leave the rest' (no default). Written for P1 (not the evaluator's instruction_sets sentences)."""
    CASES = [
        # A: everything but X
        ("Put everything but the cubes in the right bin.", {"default": R, "cube": N}),
        ("Move all objects but the yellow ones to the middle box.", {"default": C, "yellow": N}),
        ("All but the cylinders go left.", {"default": L, "cylinder": N}),
        ("Everything apart from the blue pieces belongs in the left bin.", {"default": L, "blue": N}),
        ("Aside from the triangles, put everything in the middle.", {"triangle": N, "default": C}),
        ("Put everything other than red in the right box.", {"default": R, "red": N}),
        ("Everything that isn't yellow goes left.", {"default": L, "yellow": N}),
        ("Put everything except the red objects in the left bin.", {"default": L, "red": N}),
        ("Everything except the red cubes goes in the left bin; the red cubes go right.",
         {"default": L, ("red", "cube"): R}),
        ("Everything goes in the middle bin except the blue ones, which go in the left bin.",
         {"default": C, "blue": L}),
        ("Everything goes into the right box except the cylinders, which stay where they are.",
         {"default": R, "cylinder": N}),
        # B: X stay, the rest -> B
        ("The yellow ones remain on the table, and all other objects go in the right bin.",
         {"yellow": N, "default": R}),
        ("Cylinders stay where they are; put everything else in the middle box.", {"cylinder": N, "default": C}),
        ("Blue stays. The rest goes left.", {"blue": N, "default": L}),
        # C: leave X, the rest -> B
        ("Leave the cubes alone and put all the other objects in the left bin.", {"cube": N, "default": L}),
        ("Keep the red pieces on the table; everything else goes in the centre box.", {"red": N, "default": C}),
        ("Don't touch the triangles, but move everything else to the right bin.", {"triangle": N, "default": R}),
        ("Ignore the blue objects and put the rest in the right bin.", {"blue": N, "default": R}),
        # D: bin first, then a stay / rest clause: no default
        ("The middle bin is for yellow objects; leave the rest where they are.", {"yellow": C, "default": N}),
        ("Use the right box for the cylinders, and the other objects stay on the table.",
         {"cylinder": R, "default": N}),
        ("The left bin takes the red ones. Nothing else needs to move.", {"red": L, "default": N}),
        ("Into the left bin go the blue pieces; do not move anything else.", {"blue": L, "default": N}),
        # a stay beats the broader keys for its objects
        ("Red goes left, but leave the red triangles on the table.", {"red": L, ("red", "triangle"): N}),
        ("Cubes go in the middle bin, everything else in the right one; don't move the yellow cubes.",
         {"cube": C, "default": R, ("yellow", "cube"): N}),
        ("Red goes left. Do not move the cubes.", {"red": L, "cube": N, ("red", "cube"): N}),
    ]

    def test_cases(self):
        for s, exp in self.CASES:
            r = parse_instruction(s)
            self.assertEqual(r.mapping, exp, f"{s}: {r.summary()}")

    def test_object_bin_with_stays(self):
        m = {"red": N, "default": L}
        self.assertEqual([object_bin(m, "red", "cube"), object_bin(m, "blue", None), object_bin(m, "unknown", "cube")],
                         [None, L, None])
        self.assertEqual(resolve_bin(m, "red", "cube"), None)

    def test_negation_with_a_bin_is_no_stay(self):
        """'do not move X to the right bin' does not say where X goes: a negation (not ok), not a stay."""
        r = parse_instruction("Do not move the blue objects to the right bin.")
        self.assertEqual((r.mapping, r.ok), ({}, False))
        r = parse_instruction("Leave the red ones in the left bin.")       # (keep / leave X in a bin: put it there)
        self.assertEqual(r.mapping, {"red": L})

    def test_conflict_with_a_default_stays(self):
        """A key dropped as a conflict falls to no broader key: its objects stay (they would otherwise get the
        default)."""
        r = parse_instruction("Red goes left. Red goes right. Everything else goes in the middle.")
        self.assertEqual(r.mapping, {"default": C, "red": N})

    def test_two_bins_in_two_clauses(self):
        """'... into the right-hand bin and the left bin is for everything else': two clauses, not one term."""
        r = parse_instruction("Blue goes to the bin on the left, but the round blue ones go to the right-hand bin and "
                              "the left bin is for everything else.")
        self.assertEqual(r.mapping, {"blue": L, ("blue", "cylinder"): R, "default": L})
        r = parse_instruction("Red goes in the left-hand bin and the right bin takes the cubes.")
        self.assertEqual(r.mapping, {"red": L, "cube": R})
        self.assertEqual(parse_instruction("The left and right bins are for red.").mapping, {})   # (as before)

    def test_trailing_except_after_the_bin(self):
        """'everything ... in the right box, apart from / except X': X stays (it went to the default's box before:
        -1 per object); an explicit bin for X still wins."""
        for text, m in (("Place all items into the right box, apart from the yellow ones.", {"default": R, "yellow": N}),
                        ("Put everything in the right box excluding the purple cubes.",
                         {"default": R, ("purple", "cube"): N}),
                        ("Put everything in the centre bin apart from the yellow ones and the cubes.",
                         {"default": C, "yellow": N, "cube": N}),
                        ("Put everything in the right box, except for the yellow ones.", {"default": R, "yellow": N}),
                        ("Put everything in the right box, except for the yellow ones, which go left.",
                         {"default": R, "yellow": L}),
                        ("Put everything in the right box except the yellow ones. The yellow ones go left.",
                         {"default": R, "yellow": L})):
            with self.subTest(text=text):
                r = parse_instruction(text)
                self.assertEqual(r.mapping, m)
                self.assertTrue(r.ok, r.problems)

    def test_ramp_is_a_prism(self):
        self.assertEqual(parse_instruction("Put the ramps in the middle tray and the cans on the right.").mapping,
                         {"triangle": C, "cylinder": R})


class TestUnread(unittest.TestCase):
    """A word in object position the lexicon cannot read: that phrase gives no key, and every key that could take its
    objects (a default always could) is dropped; its bin is never given to another phrase. Never a wrong bin."""
    CASES = [
        ("The knobbly pieces go left and everything else goes right.", {}),
        ("Put the hexagonal ones in the left bin, the cubes in the middle bin, and the rest in the right bin.",
         {"cube": C}),
        ("The fluted red pieces go to the right box; put the yellow ones in the middle; the others go left.",
         {"yellow": C}),
        ("Red goes left, and the star shaped pieces go right.", {}),
        ("Cubes left, the egg shaped ones in the middle, triangles right.", {"cube": L, "triangle": R}),
        ("The left box is for the knobbly pieces, the middle box for yellow.", {}),   # (knobbly yellow ones?)
        ("The left box is for the knobbly blue pieces, the middle box for yellow.", {"yellow": C}),
        ("Put the spotted blue ones in the left bin and the red ones in the right bin.", {"red": R}),
    ]

    def test_cases(self):
        for s, exp in self.CASES:
            r = parse_instruction(s)
            self.assertEqual(r.mapping, exp, f"{s}: {r.summary()}")
            self.assertFalse(r.ok, s)
            self.assertIn("unread", [p["kind"] for p in r.problems], s)

    def test_harmless_adjective(self):
        """Next to both a colour and a shape an unknown word is an adjective, not a missing shape / colour."""
        r = parse_instruction("Put the shiny red cubes in the left bin.")
        self.assertEqual((r.mapping, r.ok), ({("red", "cube"): L}, True))

    def test_new_shape_words(self):
        for s, exp in (("The boxy pieces go left.", {"cube": L}), ("Barrel-shaped pieces go in the middle.",
                                                                   {"cylinder": C}),
                       ("Put the three-sided ones on the right.", {"triangle": R}),
                       ("The box-shaped red objects go in the left bin.", {("red", "cube"): L})):
            self.assertEqual(parse_instruction(s).mapping, exp, s)


class TestOracle(unittest.TestCase):
    """sim/oracle.py reads instructions with this parser first (needs the sim packages: skipped without them)."""

    def setUp(self):
        try:
            from aaltonomous.sim import oracle
        except ImportError as e:
            self.skipTest(f"oracle not importable here: {e}")
        self.oracle = oracle

    def test_reproduces_the_template_table(self):
        table = self.oracle._parse_table()
        self.assertEqual(len(table), 288)
        for s, m in table.items():
            self.assertEqual(self.oracle.parse_instruction(s), m, s)
        self.assertEqual(self.oracle.parse_instruction(tasks.FIXED_INSTRUCTION), tasks.FIXED_MAPPING)
        self.assertIsNone(self.oracle.parse_instruction("Wave at the audience."))

    def test_privileged_fallback(self):
        priv = {"cube": L, "default": R}
        pol = self.oracle.OraclePolicy(None, privileged_mapping=lambda: dict(priv), threaded=False, log=None)
        pol.reset({"instruction": "Wave at the audience."})
        self.assertEqual((pol.mapping, pol.mapping_source), (priv, "privileged"))
        pol.reset({"instruction": "Blue goes left, red goes right, yellow in the middle."})
        self.assertEqual((pol.mapping, pol.mapping_source), ({"blue": L, "red": R, "yellow": C}, "parsed"))
        pol.reset({"instruction": ""})
        self.assertEqual((pol.mapping, pol.mapping_source), (tasks.FIXED_MAPPING, "fixed"))
        pol.reset({"instruction": "x", "mapping": {"red": C}})
        self.assertEqual((pol.mapping, pol.mapping_source), ({"red": C}, "given"))


class TestObjectBin(unittest.TestCase):
    def test_resolve_bin_same_as_scene(self):
        try:
            from aaltonomous.scene.scene import resolve_bin as scene_resolve
        except ImportError:   # no yaml (plain python on a laptop)
            self.skipTest("aaltonomous.scene needs yaml")
        maps = [FIX, {"cube": L, "default": R}, {("red", "cube"): L, "red": C, "cube": R, "default": L}, {}]
        for m in maps:
            for c, s in itertools.product(COLORS, SHAPES + (None,)):
                self.assertEqual(resolve_bin(m, c, s), scene_resolve(m, c, s))

    def test_unknown_shape_or_colour(self):
        self.assertEqual(object_bin(FIX, "red", None), L)                     # colour mapping: shape not needed
        self.assertIsNone(object_bin(FIX, "unknown", "cube"))
        m = {"cube": L, "cylinder": C, "triangle": R}
        self.assertIsNone(object_bin(m, "red", None))                         # needs the shape
        self.assertEqual(object_bin(m, "unknown", "cube"), L)                 # colour not needed
        m = {("red", "cube"): L, "default": R}
        self.assertIsNone(object_bin(m, "red", None))                         # red cube or other red object
        self.assertEqual(object_bin(m, "blue", None), R)                      # every blue object: default
        self.assertIsNone(object_bin(m, "unknown", "cube"))
        self.assertEqual(object_bin({"cube": L, "default": L}, "red", None), L)   # every key the same bin
        self.assertEqual(object_bin({"default": C}, "unknown", None), C)
        self.assertIsNone(object_bin({}, "red", "cube"))
