"""Richer sort_language instructions for our evaluator (E1): each draw is an instruction TEXT plus the TRUE mapping
it means, in the organisers' mapping format (scene.resolve_bin keys: colour, shape, (colour, shape), "default";
values left / center / right, or None = "this one stays on the counter").

    d = draw("both", seed=1000, layout=env.layout)    # Draw(set, kind, text, mapping, valid)
    install(env, "all")                               # evaluate.py --instructions all: env.reset() hands these out
    python -m aaltonomous.sim.instruction_sets --show 3 --n 500          # dry run of our parser (no sim)
    python -m aaltonomous.sim.instruction_sets --world robot --color-words red=orange,blue=purple

Generated FROM the mapping (keys and bins first, then words), so text and mapping agree by construction. Texts use
the internal colour words (red / yellow / blue): the evaluator's --color-words renames them as any instruction, so
the sets work unchanged in the orange / yellow / purple variants (no plural colour nouns like "reds": rename() only
replaces whole words). Per trial the rng is default_rng([trial seed, SALT]): same seed + layout = same draw, and the
layout itself is AaltoEnv's (drawn before, unchanged).

Sets (bins named by position only; "boxes" / "bins" / "container" are bins, never objects):
  shape     shape keys only: all three shapes (one bin each or two sharing one), or 1-2 shapes + a rest clause
  both      (colour, shape) keys + a rest clause, half of them splitting one colour by shape ("red cubes left, red
            cylinders right, everything else in the middle")
  mixed     colour clauses and shape clauses (+ rest or not), or a colour clause overridden for one of its shapes
            ("blue goes right, but the blue cubes go left"). Never a conflict for an object ON THE TABLE: a type
            that a colour clause and a shape clause send to different bins is not in the layout. resolve_bin's
            precedence ((colour, shape) > colour > shape > default) is the override's English meaning; `valid` = the
            types the text is unambiguous for (all but the absent conflicting ones)
  partial   1-2 colours / shapes / types named, no rest clause (sometimes "leave the rest where it is"): the others
            must stay
  negation  "everything except X goes left; X right" ({X: right, default: left}), "put everything but X in the left
            bin" ({X: None, default: left}), "... Do not move X." (X absent from the mapping), "red goes left, not
            right" (a correction)
  unseen    colour-only mappings in 40 paraphrases that are NOT tasks.TEMPLATES (bins first, passive, polite, lists,
            "respectively", "from left to right", counted bins "the second box from the left", "the remaining box")
  all       one of the above per episode (seeded)
Distractor (distractor=True): no rest clause and at least one object on the table whose colour or shape the text
does not mention at all (it simply has no bin). Not for `unseen` (its paraphrases name all three colours).

Scoring (the organisers' rule, unchanged: scene.classify_cube with the true mapping): an object whose bin is None is
"wrong" (-1) in any box and "counter" (0) if left; staying is the best it can do (best episode = +2 per object that
has a bin).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field

import numpy as np

from aaltonomous.interface.tasks import BINS, COLORS, SHAPES
from aaltonomous.scene.scene import resolve_bin

SALT = 0xE1                     # instruction rng: default_rng([trial seed, SALT]), independent of the layout draws
SETS = ("shape", "both", "mixed", "partial", "negation", "unseen")
CHOICES = SETS + ("all",)
TYPES = tuple((c, s) for c in COLORS for s in SHAPES)

# ------------------------------------------------------------------------------------------------- wording
BIN_NP = {"left": ("the left bin", "the left box", "the bin on the left", "the box on the left", "the leftmost box",
                   "the left-hand bin"),
          "center": ("the center bin", "the middle bin", "the middle box", "the box in the middle", "the centre box",
                     "the central bin"),
          "right": ("the right bin", "the right box", "the bin on the right", "the box on the right",
                    "the rightmost box", "the right-hand bin")}
BIN_NP_MORE = {"left": ("the first box from the left", "the third bin from the right", "the bin on the far left",
                        "the left-most container"),
               "center": ("the second box from the left", "the second bin from the right",
                          "the box between the other two", "the bin in the centre"),
               "right": ("the first box from the right", "the third bin from the left", "the bin on the far right",
                         "the last box on the right")}
BIN_WORD = {"left": ("left",), "center": ("middle", "center", "centre"), "right": ("right",)}
SHAPE_NOUN = {"cube": ("cubes", "square blocks", "square ones", "cube-shaped pieces"),
              "cylinder": ("cylinders", "round ones", "round pieces", "cylindrical pieces", "circular ones"),
              "triangle": ("triangles", "triangular prisms", "wedges", "triangular ones", "prisms")}
SHAPE_ADJ = {"cube": ("square", "cubic"), "cylinder": ("round", "cylindrical"), "triangle": ("triangular",)}
SHAPE_ONE = {"cube": "cube", "cylinder": "cylinder", "triangle": "triangle"}
# fair English that instruction.py's lexicon does not know (2026-11-14): used in P_NOVEL of the shape phrases
NOVEL_NOUN = {"cube": ("boxy pieces",), "cylinder": ("barrel-shaped pieces",), "triangle": ("three-sided pieces",)}
NOVEL_ADJ = {"cube": "boxy", "cylinder": "barrel-shaped", "triangle": "three-sided"}
P_NOVEL = 0.1
COL_NP = (("the {c} objects", 1), ("the {c} ones", 1), ("{c} objects", 1), ("the {c} pieces", 1),
          ("all {c} objects", 1), ("anything {c}", 0), ("everything {c}", 0), ("{c}", 0))
# (first clause, later clauses of a list): one frame per instruction
FRAMES = (("put {O} in {B}", "{O} in {B}"), ("{O} {go} in {B}", "{O} in {B}"), ("{O} {go} into {B}", "{O} {go} into {B}"),
          ("move {O} to {B}", "{O} to {B}"), ("{B} is for {O}", "{B} for {O}"), ("{O} {belong} in {B}", "{O} {belong} in {B}"),
          ("place {O} into {B}", "{O} into {B}"), ("{O} should go in {B}", "{O} in {B}"), ("use {B} for {O}", "{B} for {O}"),
          ("{O} {go} to {B}", "{O} to {B}"))
REST = ("everything else goes in {B}", "all other objects go in {B}", "the rest go in {B}", "put everything else in {B}",
        "whatever is left goes in {B}", "anything else goes into {B}", "{B} is for everything else",
        "the remaining objects go to {B}", "all the others go in {B}")
LEAVE = ("leave everything else where it is", "leave the rest on the table", "the other objects stay where they are",
         "do not move anything else", "nothing else needs to move")
ONLY = ("only move {O}: put them in {B}", "just put {O} in {B} and leave the rest", "{O} {go} in {B}; leave everything "
        "else alone", "put {O} in {B}. That is all", "sort only {O}: they go in {B}")
EXCEPT = ("everything except {X} goes in {B0}; {X} go in {B1}", "put all objects except {X} in {B0}, and put {X} in {B1}",
          "apart from {X}, everything goes in {B0}. {X} go in {B1}", "put everything other than {X} in {B0} and {X} in {B1}",
          "{X} go in {B1}. Everything that is not {xa} goes in {B0}", "everything goes in {B0} except {X}, which go in {B1}")
STAY_ALL = ("put everything except {X} in {B0}", "put everything but {X} in {B0}",
            "everything goes in {B0} except {X}, which stay where they are",
            "leave {X} on the table and put everything else in {B0}", "{X} stay on the table; all other objects go in {B0}",
            "move everything that is not {xa} to {B0}")
STAY = ("do not move {X}", "leave {X} where they are", "{X} stay on the table", "don't touch {X}",
        "{X} should not be moved")
# unseen colour paraphrases: (template, bin order) with {c0..2} colours, {B0..2} bin phrases, {b0..2} position words;
# order None = any permutation, "lcr" / "rcl" = the template fixes the bins left -> right / right -> left
UNSEEN = (
    ("could you put the {c0} objects in {B0}, the {c1} ones in {B1} and the {c2} ones in {B2}, please?", None),
    ("please sort the pieces by colour: {c0} ones go in {B0}, {c1} ones in {B1}, {c2} ones in {B2}.", None),
    ("{B0} gets everything {c0}, {B1} gets everything {c1}, and {B2} gets everything {c2}.", None),
    ("I would like the {c0} pieces in {B0}. The {c1} pieces should go in {B1}. Finally, the {c2} pieces belong in {B2}.",
     None),
    ("the {c0} objects should be placed in {B0}, the {c1} objects in {B1}, and the {c2} objects in {B2}.", None),
    ("{c0} objects are to be put in {B0}; {c1} objects are to be put in {B1}; {c2} objects are to be put in {B2}.", None),
    ("into {B0} go the {c0} objects, into {B1} the {c1} ones, and into {B2} the {c2} ones.", None),
    ("start with the {c0} objects and put them in {B0}. Then put the {c1} objects in {B1} and the {c2} objects in {B2}.",
     None),
    ("from left to right, the boxes are for {c0}, {c1} and {c2} objects.", "lcr"),
    ("the {c0}, {c1} and {c2} objects go into the {b0}, {b1} and {b2} boxes respectively.", None),
    ("put all {c0} things into {B0}, all {c1} things into {B1} and all {c2} things into {B2}.", None),
    ("{c0}: {B0}. {c1}: {B1}. {c2}: {B2}.", None),
    ("{B0}: {c0}. {B1}: {c1}. {B2}: {c2}.", None),
    ("sort by colour. {c0} goes in {B0}, {c1} goes in {B1}, {c2} goes in {B2}.", None),
    ("each {c0} object belongs in {B0}, each {c1} object in {B1} and each {c2} object in {B2}.", None),
    ("take every {c0} object and drop it into {B0}; do the same with the {c1} ones and {B1}, and with the {c2} ones "
     "and {B2}.", None),
    ("{c0} things go to {B0}, while {c1} things go to {B1} and {c2} things to {B2}.", None),
    ("put the {c0} ones in {B0} and the {c1} ones in {B1}. The {c2} ones go in the remaining box.", None),
    ("your task: {c0} into {B0}, {c1} into {B1}, {c2} into {B2}.", None),
    ("we need the {c0} items in {B0}, the {c1} items in {B1}, and the {c2} items in {B2}.", None),
    ("pick up each {c0} object and place it in {B0}. Every {c1} object goes in {B1}, and every {c2} object goes in "
     "{B2}.", None),
    ("{B1} is where the {c1} objects go. The {c0} ones go in {B0} and the {c2} ones in {B2}.", None),
    ("put the {c2} objects in {B2} first, then the {c0} ones in {B0}, and the {c1} ones in {B1} last.", None),
    ("move anything {c0} to {B0}, anything {c1} to {B1} and anything {c2} to {B2}.", None),
    ("if an object is {c0}, put it in {B0}; if it is {c1}, put it in {B1}; if it is {c2}, put it in {B2}.", None),
    ("the {c0} pieces belong in {B0}. {B1} takes the {c1} pieces, and {B2} takes the {c2} pieces.", None),
    ("group the objects by colour: {c0} in {B0}, {c1} in {B1}, {c2} in {B2}.", None),
    ("kindly place the {c0} blocks into {B0}, the {c1} blocks into {B1} and the {c2} blocks into {B2}. Thank you!", None),
    ("here is the plan: {c0} goes to {B0}, {c1} goes to {B1}, and {c2} goes to {B2}.", None),
    ("all the {c0} stuff in {B0}, all the {c1} stuff in {B1}, all the {c2} stuff in {B2}.", None),
    ("working from left to right, fill the boxes with {c0}, {c1} and {c2} objects.", "lcr"),
    ("from right to left, the boxes take the {c0}, {c1} and {c2} objects.", "rcl"),
    ("the box for {c0} objects is {B0}, the box for {c1} objects is {B1}, and the box for {c2} objects is {B2}.", None),
    ("sort everything: {c0} objects to {B0}, {c1} objects to {B1}, and {c2} objects to {B2}.", None),
    ("{c0} in {B0}, {c1} in {B1}, {c2} in {B2}, please.", None),
    ("put {c0} objects in {B0}. Put {c1} objects in {B1}. Put {c2} objects in {B2}.", None),
    ("the objects that are {c0} go in {B0}, those that are {c1} go in {B1} and those that are {c2} go in {B2}.", None),
    ("don't mix the colours: {c0} into {B0}, {c1} into {B1}, {c2} into {B2}.", None),
    ("in {B0}, put the {c0} objects. In {B1}, the {c1} ones. In {B2}, the {c2} ones.", None),
    ("the {b0} box takes {c0}, the {b1} box takes {c1}, and the {b2} box takes {c2}.", None),
)


def _p(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _perm(rng, seq):
    return [str(x) for x in rng.permutation(list(seq))]


def _cap(t):
    """Capitalise each sentence's first letter."""
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)


def _bin(rng, b, more=False):
    return _p(rng, BIN_NP[b] + (BIN_NP_MORE[b] if more else ()))


def _noun(rng, s):
    return _p(rng, NOVEL_NOUN[s] if rng.random() < P_NOVEL else SHAPE_NOUN[s])


def _obj(rng, key, plural=False):
    """Noun phrase for a mapping key -> (text, plural)."""
    if isinstance(key, tuple):
        c, s = key
        if rng.random() < 0.25:     # adjective first: "the round red pieces" (usual English adjective order)
            adj = NOVEL_ADJ[s] if rng.random() < P_NOVEL else _p(rng, SHAPE_ADJ[s])
            return f"the {adj} {c} {_p(rng, ('pieces', 'ones', 'objects'))}", 1
        return f"the {c} {_noun(rng, s)}", 1
    if key in SHAPES:
        n = _noun(rng, key)
        det = ("the {n}", "all the {n}") if "ones" in n else ("the {n}", "{n}", "all {n}", "all the {n}")
        return _p(rng, det).format(n=n), 1
    t, pl = _p(rng, [f for f in COL_NP if f[1]] if plural else COL_NP)
    return t.format(c=key), pl


def _xphrase(rng, key):
    """Plural phrase for a negated key + its predicate form ('not red' / 'not a cube')."""
    if key in SHAPES:
        return f"the {_noun(rng, key)}", f"a {SHAPE_ONE[key]}"
    return _p(rng, ("the {c} objects", "the {c} ones", "the {c} pieces")).format(c=key), key


def _clauses(rng, items, group=0.3):
    """[(key, bin)] -> [(object phrase, plural, bin phrase, prefix)]; keys sharing a bin are sometimes joined."""
    out, done = [], set()
    for i, (k, b) in enumerate(items):
        if i in done:
            continue
        objs = [_obj(rng, k)]
        if rng.random() < group:
            for j in range(i + 1, len(items)):
                if j not in done and items[j][1] == b:
                    objs.append(_obj(rng, items[j][0], plural=True))
                    done.add(j)
        pl = objs[0][1] if len(objs) == 1 else 1
        out.append((" and ".join(o for o, _ in objs), pl, _bin(rng, b), ""))
    return out


def _say(frame, o, pl, b):
    return frame.format(O=o, B=b, go="go" if pl else "goes", belong="belong" if pl else "belongs")


def _render(rng, clauses, rest=None, tail=None):
    """Clauses -> text with one frame, as a list ("a, b and c"), with semicolons or as sentences; rest = bin phrase of
    an 'everything else' clause (last); tail = a closing clause ('leave the rest where it is')."""
    first, nxt = _p(rng, FRAMES)
    style = _p(rng, ("list", "list", "semi", "sent"))
    parts = [pre + _say(first if i == 0 or style == "sent" else nxt, o, pl, b)
             for i, (o, pl, b, pre) in enumerate(clauses)]
    if rest:
        parts.append(_p(rng, REST).format(B=rest))
    if tail:
        parts.append(tail)
    if style == "sent" or len(parts) == 1:
        text = ". ".join(parts) + "."
    elif style == "semi":
        text = "; ".join(parts) + "."
    else:
        last = ", " if parts[-1].startswith("but ") else _p(rng, (" and ", ", and "))   # never "and but"
        text = ", ".join(parts[:-1]) + last + parts[-1] + "."
    return _cap(text)


# ------------------------------------------------------------------------------------------------- sets
def _gen_shape(rng, present, distractor):
    sh = _perm(rng, SHAPES)
    kind = "subset" if distractor else _p(rng, ("bijective", "bijective", "shared", "rest"))
    if kind == "bijective":
        names, bins = sh, _perm(rng, BINS)
    elif kind == "shared":
        b = _perm(rng, BINS)
        names, bins = sh, [b[0], b[1], b[int(rng.integers(2))]]
    else:
        names = sh[:int(rng.integers(1, 3))]
        bins = [_p(rng, BINS) for _ in names]
    m = dict(zip(names, bins))
    rest = None
    if kind == "rest":
        rest = _p(rng, [b for b in BINS if b not in bins] or list(BINS))
        m["default"] = rest
    tail = _p(rng, LEAVE) if kind == "subset" and rng.random() < 0.4 else None
    return kind, _render(rng, _clauses(rng, list(zip(names, bins))), rest and _bin(rng, rest), tail), m, TYPES


def _gen_both(rng, present, distractor):
    pres = sorted(present)
    items = []
    split = [c for c in COLORS if len({s for c2, s in pres if c2 == c}) >= 2]
    kind = "split" if split and rng.random() < 0.5 else "plain"
    if kind == "split":           # one colour, two shapes, two bins
        c = _p(rng, split)
        s1, s2 = [str(s) for s in rng.permutation(sorted({s for c2, s in pres if c2 == c}))[:2]]
        b = _perm(rng, BINS)
        items = [((c, s1), b[0]), ((c, s2), b[1])]
    n = int(rng.integers(1, 4)) - len(items)
    others = [t for t in pres if t not in dict(items)]
    for i in rng.permutation(len(others))[:max(0, n)]:
        items.append((others[int(i)], _p(rng, BINS)))
    if rng.random() < 0.5:        # the order in the text need not follow the colours
        items = [items[int(i)] for i in rng.permutation(len(items))]
    m = dict(items)
    rest = None if distractor else _p(rng, BINS)
    if rest:
        m["default"] = rest
    return kind, _render(rng, _clauses(rng, items), rest and _bin(rng, rest)), m, TYPES


def _gen_mixed(rng, present, distractor):
    for _ in range(200):
        override = rng.random() < 0.3
        cols = _perm(rng, COLORS)[:int(rng.integers(1, 3))]
        shs = [] if override else _perm(rng, SHAPES)[:int(rng.integers(1, 3))]
        m = {k: _p(rng, BINS) for k in cols + shs}
        extra = None
        if override:              # "blue goes right, but the blue cubes go left": (colour, shape) wins
            c = cols[0]
            on = sorted(s for c2, s in present if c2 == c) or list(SHAPES)
            s = _p(rng, on)
            extra = ((c, s), _p(rng, [b for b in BINS if b != m[c]]))
        conflict = {(c, s) for c in cols for s in shs if m[c] != m[s]}   # English conflict: must not be on the table
        if not conflict & set(present):
            break
    else:
        raise RuntimeError("mixed: no conflict-free draw for this layout")
    items = list(m.items())
    items = [items[int(i)] for i in rng.permutation(len(items))]
    cl = _clauses(rng, items, group=0.0)
    if extra:
        m[extra[0]] = extra[1]
        o, pl = _obj(rng, extra[0])
        at = next(i for i, (k, _) in enumerate(items) if k == extra[0][0])
        cl.insert(at + 1, (o, pl, _bin(rng, extra[1]), "but "))
    rest = None if distractor or rng.random() < 0.5 else _p(rng, BINS)
    if rest:
        m["default"] = rest
    return ("override" if extra else "colour+shape"), _render(rng, cl, rest and _bin(rng, rest)), m, \
        tuple(t for t in TYPES if t not in conflict)


def _gen_partial(rng, present, distractor):
    kind = _p(rng, ("colours", "shapes", "types", "only"))
    if kind == "colours":
        keys = _perm(rng, COLORS)[:int(rng.integers(1, 3))]
    elif kind == "shapes":
        keys = _perm(rng, SHAPES)[:int(rng.integers(1, 3))]
    elif kind == "types":
        pres = sorted(present)
        keys = [pres[int(i)] for i in rng.permutation(len(pres))[:int(rng.integers(1, 3))]]
    else:
        keys = [_p(rng, list(COLORS) + list(SHAPES) + sorted(present))]
    m = {k: _p(rng, BINS) for k in keys}
    if kind == "only":
        o, pl = _obj(rng, keys[0], plural=True)
        text = _cap(_say(_p(rng, ONLY), o, pl, _bin(rng, m[keys[0]])) + ".")
    else:
        text = _render(rng, _clauses(rng, list(m.items())), None, _p(rng, LEAVE) if rng.random() < 0.5 else None)
    return kind, text, m, TYPES


def _gen_negation(rng, present, distractor):
    kind = _p(rng, ("stay_clauses",) if distractor else ("except", "stay_all", "stay_clauses", "correction"))
    attr = COLORS if rng.random() < 0.6 or kind == "correction" else SHAPES
    vals = _perm(rng, attr)
    x = vals[0]
    xp, xa = _xphrase(rng, x)
    b = _perm(rng, BINS)
    if kind == "except":
        m = {x: b[1], "default": b[0]}
        text = _p(rng, EXCEPT).format(X=xp, xa=xa, B0=_bin(rng, b[0]), B1=_bin(rng, b[1]))
    elif kind == "stay_all":
        m = {x: None, "default": b[0]}
        text = _p(rng, STAY_ALL).format(X=xp, xa=xa, B0=_bin(rng, b[0]))
    elif kind == "stay_clauses":
        others = vals[1:1 + int(rng.integers(1, 3))]
        m = {k: _p(rng, BINS) for k in others}
        body = _render(rng, _clauses(rng, list(m.items())))
        stay = _cap(_p(rng, STAY).format(X=xp) + ".")
        text = f"{stay} {body}" if rng.random() < 0.3 else f"{body} {stay}"
        return kind, text, m, TYPES
    else:                         # correction: "red goes in the left bin, not in the right one"
        m = dict(zip(vals, b))
        o, pl = _obj(rng, x)
        wrong = _bin(rng, _p(rng, [q for q in BINS if q != m[x]]))
        first = _say("{O} {go} in {B}", o, pl, _bin(rng, m[x])) + f", not {_p(rng, ('in ', ''))}{wrong}"
        text = _cap(first + ". " + _render(rng, _clauses(rng, [(k, m[k]) for k in vals[1:]])))
        return kind, text, m, TYPES
    return kind, _cap(text + "."), m, TYPES


def _gen_unseen(rng, present, distractor):
    i = int(rng.integers(len(UNSEEN)))
    tpl, order = UNSEEN[i]
    cols = _perm(rng, COLORS)
    bins = list(BINS) if order == "lcr" else list(BINS)[::-1] if order == "rcl" else _perm(rng, BINS)
    w = {f"c{k}": c for k, c in enumerate(cols)}
    w.update({f"B{k}": _bin(rng, b, more=True) for k, b in enumerate(bins)})
    w.update({f"b{k}": _p(rng, BIN_WORD[b]) for k, b in enumerate(bins)})
    return f"t{i:02d}", _cap(tpl.format(**w)), dict(zip(cols, bins)), TYPES


GEN = {"shape": _gen_shape, "both": _gen_both, "mixed": _gen_mixed, "partial": _gen_partial,
       "negation": _gen_negation, "unseen": _gen_unseen}


# ------------------------------------------------------------------------------------------------- draws
@dataclass
class Draw:
    set: str
    kind: str
    text: str                  # internal colour words (the evaluator renames them for the policy)
    mapping: dict              # TRUE mapping (resolve_bin keys; value None = stays)
    valid: tuple = TYPES       # object types the text is unambiguous for (mixed: not the absent conflicting ones)
    present: tuple = field(default_factory=tuple)   # types on the table (layout), sorted

    def bin(self, color, shape):
        return resolve_bin(self.mapping, color, shape)

    def stay(self, types=None):
        """Types (of `types`, default the table's) that must stay on the counter."""
        return [t for t in (self.present if types is None else types) if self.bin(*t) is None]

    def to_json(self):
        return {"set": self.set, "kind": self.kind, "text": self.text, "mapping": mapping_json(self.mapping)}


def types_of(layout):
    return tuple(sorted({(o["color"], o.get("shape", "cube")) for o in layout})) if layout else TYPES


def mentioned(mapping):
    """(colours, shapes) the mapping's keys name (None-valued keys too: 'do not move the yellow ones' names yellow)."""
    cs = {k[0] if isinstance(k, tuple) else k for k in mapping if k != "default" and k not in SHAPES}
    ss = {k[1] if isinstance(k, tuple) else k for k in mapping if isinstance(k, tuple) or k in SHAPES}
    return cs, ss


def distracts(mapping, present):
    """True if some object on the table has no bin and its colour or shape is not mentioned at all."""
    cs, ss = mentioned(mapping)
    return any(resolve_bin(mapping, c, s) is None and (c not in cs or s not in ss) for c, s in present)


def draw(name, seed, layout=None, distractor=False) -> Draw:
    """One (instruction, true mapping) of set `name` for trial `seed` and this layout (None: any of the 9 types may
    be on the table). Deterministic in (name, seed, layout, distractor)."""
    if name not in CHOICES:
        raise ValueError(f"unknown instruction set {name!r} (choose from {', '.join(CHOICES)})")
    if distractor and name == "unseen":
        raise ValueError("--distractor needs a set without a full colour mapping (not 'unseen')")
    rng = np.random.default_rng([int(seed), SALT])
    present = types_of(layout)
    if name == "all":
        name = _p(rng, [s for s in SETS if not (distractor and s == "unseen")])
    for _ in range(200):
        kind, text, m, valid = GEN[name](rng, present, distractor)
        if not distractor or distracts(m, present):
            return Draw(name, kind, text, m, tuple(valid), present)
    raise RuntimeError(f"{name}: no distractor draw for this layout")


def mapping_json(m):
    """Tuple keys -> 'colour/shape' (as SceneClient.evaluate / aalto_task.layout.parse_mapping); None stays null."""
    return None if m is None else {("/".join(k) if isinstance(k, tuple) else k): v for k, v in m.items()}


def mapping_from_json(m):
    return {(tuple(k.split("/")) if "/" in k else k): v for k, v in (m or {}).items()}


# ------------------------------------------------------------------------------------------------- evaluator glue
def install(env, name, distractor=False):
    """After every env.reset() (layout drawn by AaltoEnv as before) the instruction and env.mapping (the scored, true
    mapping) come from draw(name, env.instruction_seed, env.layout). Install BEFORE evaluate.rename_instructions so
    the colour words of these texts are renamed too."""
    orig = env.reset

    def reset():
        obs = orig()
        d = draw(name, getattr(env, "instruction_seed", 0), env.layout, distractor)
        env.instruction = obs["instruction"] = d.text
        env.mapping, env.instruction_draw = d.mapping, d
        return obs
    env.reset = reset


def trial_fields(d, objects, outcomes, words=None):
    """eval.json fields of one trial: set, texts, true mapping (serialised), per object its true bin + outcome, and
    how many objects had to stay / stayed. objects: scene service state (name, color, shape)."""
    from aaltonomous.sim.aalto_task.variant import rename, told_mapping
    by = {o["name"]: o for o in objects}
    per = []
    for r in outcomes:
        o = by.get(r["name"], {})
        per.append({"name": r["name"], "color": o.get("color"), "shape": o.get("shape"),
                    "bin": resolve_bin(d.mapping, o.get("color"), o.get("shape")), "outcome": r["outcome"]})
    stay = [p for p in per if p["bin"] is None]
    out = {"instruction_set": d.set, "instruction_kind": d.kind, "instruction_internal": d.text,
           "instruction_told": rename(d.text, words or {}), "mapping": mapping_json(d.mapping), "expected": per,
           "n_must_stay": len(stay), "n_stayed": sum(p["outcome"] == "counter" for p in stay)}
    if words:
        out["mapping_told"] = mapping_json(told_mapping(d.mapping, words))
    return out


def summarize(trials, name, distractor):
    """Summary fields: objects correct / wrong / left / off, had to stay / stayed, and the same per actual set."""
    def agg(ts):
        oc = [p["outcome"] for t in ts for p in t["expected"]]
        return {"trials": len(ts), "score": sum(t["score"] for t in ts), "max_score": sum(t["max_score"] for t in ts),
                "correct": oc.count("correct"), "wrong": oc.count("wrong"), "left": oc.count("counter"),
                "off": oc.count("off"), "must_stay": sum(t["n_must_stay"] for t in ts),
                "stayed": sum(t["n_stayed"] for t in ts)}
    ts = [t for t in trials if "expected" in t]
    a = agg(ts)
    return {"instructions": name, "distractor": bool(distractor), "objects_wrong": a["wrong"],
            "objects_left": a["left"], "objects_off": a["off"], "objects_must_stay": a["must_stay"],
            "objects_stayed": a["stayed"],
            "by_set": {s: agg([t for t in ts if t["instruction_set"] == s])
                       for s in sorted({t["instruction_set"] for t in ts})}}


def table_rows(r):
    rows = [("instructions", f"{r['instructions']}{' + distractor' if r['distractor'] else ''}"),
            ("objects correct / wrong / left / off", f"{r['objects_correct']} / {r['objects_wrong']} / "
             f"{r['objects_left']} / {r['objects_off']}"),
            ("had to stay / stayed", f"{r['objects_must_stay']} / {r['objects_stayed']}")]
    for s, v in r["by_set"].items():
        rows.append((f"  {s}", f"score {v['score']:+d}/{v['max_score']}  correct {v['correct']} wrong {v['wrong']} "
                               f"left {v['left']} off {v['off']}  stay {v['stayed']}/{v['must_stay']}  "
                               f"({v['trials']} trials)"))
    return rows


# ------------------------------------------------------------------------------------------------- dry run
def episode_layout(seed, scene=None):
    """The layout AaltoEnv(seed) draws for sort_language (nominal scene.yaml unless `scene`): instruction first."""
    from aaltonomous.interface import tasks
    from aaltonomous.scene.scene import Scene
    rng = np.random.default_rng(seed)
    tasks.make_instruction("sort_language", rng)
    return tasks.sample_layout(rng, **(scene or Scene.load()).layout_kwargs())


def world(name):
    """(world classes {class: [hues]}, reject hues, match params) as the classical policy has them: sim = profile
    sim (scene.yaml red / yellow / blue), robot = base config (orange / yellow / purple)."""
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.perception import world_classes
    from aaltonomous.scene.scene import Scene
    cfg = load_config(profile="sim" if name == "sim" else "none")
    refs, rej = world_classes(cfg["perception"], Scene.load())
    return refs, rej, (cfg["perception"].get("color") or {}).get("match")


def world_class(c, refs, words):
    """The world class that internal colour c's objects are: same name / told word, else the nearest hue."""
    from aaltonomous.baselines.classical.colors import hue_dist
    from aaltonomous.baselines.classical.instruction import HUE
    for n in (c, (words or {}).get(c)):
        if n in refs:
            return n
    return min(refs, key=lambda k: min(hue_dist(HUE[c], h) for h in refs[k]))


def judge(d, words, wld):
    """Our parser (instruction.parse_instruction) + the policy's colour matching (colors.match_colors, prior: no
    look calibration) + object_bin on the told text, against the true mapping, per type d.valid ∩ the table.
    Returns (verdict exact | subset | wrong, parse, {type: (true bin, parsed bin)} of the differing types)."""
    from aaltonomous.baselines.classical.colors import match_colors
    from aaltonomous.baselines.classical.instruction import mapping_colors, object_bin, parse_instruction
    from aaltonomous.sim.aalto_task.variant import rename
    refs, rej, mp = wld
    r = parse_instruction(rename(d.text, words or {}))
    ec = match_colors(mapping_colors(r.mapping), refs, rej, mp, r.words)
    diff = {}
    for c, s in d.valid:
        det = ec.alias.get(world_class(c, refs, words), "unknown")
        got, want = object_bin(r.mapping, det, s, ec.names), d.bin(c, s)
        if got != want:
            diff[(c, s)] = (want, got)
    on = {t: v for t, v in diff.items() if t in d.present}
    if any(g is not None for _, g in on.values()):
        return "wrong", r, diff
    return ("subset" if on else "exact"), r, diff


def _fmt_types(diff):
    return ", ".join(f"{c} {s}: true {w} parsed {g}" for (c, s), (w, g) in sorted(diff.items()))


def dry_run(sets, n=500, show=3, seed=0, words=None, wld_name="sim", distractor=False, out=None, max_list=None):
    from aaltonomous.sim.aalto_task.variant import rename
    wld = world(wld_name)
    rows, cases = [], []
    for name in sets:
        print(f"\n=== {name}  (world {wld_name}, colour words {words or 'off'}"
              f"{', distractor' if distractor else ''}) ===")
        cnt = {"exact": 0, "subset": 0, "wrong": 0, "wrong_novel": 0, "wrong_off_table": 0, "not_ok": 0}
        for i in range(n):
            sd = seed + i
            d = draw(name, sd, episode_layout(sd), distractor)
            v, r, diff = judge(d, words, wld)
            cnt[v] += 1
            novel = any(w in d.text.lower() for w in NOVEL_ADJ.values())
            cnt["wrong_novel"] += v == "wrong" and novel
            cnt["not_ok"] += not r.ok
            off = {t: x for t, x in diff.items() if t not in d.present and x[1] is not None}
            cnt["wrong_off_table"] += v != "wrong" and bool(off)
            told = rename(d.text, words or {})
            if i < show:
                print(f"  [{d.set}/{d.kind}] {told}\n      true {mapping_json(d.mapping)}  table "
                      f"{', '.join(f'{c} {s}' for c, s in d.present)}\n      parsed {mapping_json(r.mapping)} -> {v}")
            if v == "wrong" or off:
                cases.append({"set": d.set, "kind": d.kind, "seed": sd, "text": told, "true": mapping_json(d.mapping),
                              "parsed": mapping_json(r.mapping), "on_table": v == "wrong", "novel": novel,
                              "present": [f"{c}/{s}" for c, s in d.present], "diff": _fmt_types(diff),
                              "summary": r.summary()})
        rows.append((name, cnt))
    print(f"\n{'set':<10} {'n':>4} {'exact':>6} {'subset':>7} {'WRONG':>6} {'(novel word)':>13} "
          f"{'wrong off-table only':>21} {'parse not ok':>13}")
    for name, c in rows:
        print(f"{name:<10} {n:>4} {c['exact']:>6} {c['subset']:>7} {c['wrong']:>6} {c['wrong_novel']:>13} "
              f"{c['wrong_off_table']:>21} {c['not_ok']:>13}")
    dang = [x for x in cases if x["on_table"]]
    print(f"\nDANGEROUS (parsed to a wrong bin for a type on the table): {len(dang)}")
    for x in dang[:max_list]:
        print(f"  [{x['set']}/{x['kind']} seed {x['seed']}] {x['text']}\n      true {x['true']}\n      parsed "
              f"{x['parsed']}\n      {x['diff']}  (table: {', '.join(x['present'])})")
    if out:
        with open(out, "w") as f:
            json.dump({"rows": dict(rows), "cases": cases}, f, indent=1)
    return rows, cases


def main(argv=None):
    from aaltonomous.sim.aalto_task.variant import parse_words
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=",".join(SETS), help=f"comma list of {', '.join(CHOICES)}")
    ap.add_argument("--n", type=int, default=500, help="samples per set (seeds --seed ... --seed + n - 1)")
    ap.add_argument("--show", type=int, default=3, help="samples printed per set")
    ap.add_argument("--seed", type=int, default=0, help="first seed (1000 = the evaluator's default first trial)")
    ap.add_argument("--color-words", default="off", help="e.g. red=orange,blue=purple (as evaluate.py)")
    ap.add_argument("--world", default="sim", choices=["sim", "robot"], help="the policy's colour classes")
    ap.add_argument("--distractor", action="store_true")
    ap.add_argument("--json", default=None, help="write counts + every wrong case here")
    ap.add_argument("--max-list", type=int, default=None, help="dangerous cases printed (default all)")
    a = ap.parse_args(argv)
    sets = [s for s in a.sets.split(",") if s]
    if a.distractor:
        sets = [s for s in sets if s != "unseen"]
    dry_run(sets, a.n, a.show, a.seed, parse_words(a.color_words), a.world, a.distractor, a.json, a.max_list)


if __name__ == "__main__":
    main()
