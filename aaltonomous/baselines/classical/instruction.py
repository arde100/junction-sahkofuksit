"""Instruction -> bin mapping (rung 2, sort_language): a small rule-based reader. Pure Python, no model, ~1 ms.

    r = parse_instruction("Red and blue go left, everything else in the middle.")
    r.mapping    # {"red": "left", "blue": "left", "default": "center"}: scene.resolve_bin keys, values tasks.BINS
    r.ok         # True: something understood and no error-level problem
    r.keys       # per key: bin, evidence (the clause it came from), conf
    r.problems   # conflicts, negation, bins without objects, objects without bins, unknown colours, ...
    r.summary()  # one log line
    object_bin(r.mapping, "red", None)   # bin of one object (shape / colour may be unknown), or None

How it reads (structure, not one pattern per phrasing):
  1. Tokens: lower case, contractions expanded ("don't" -> "do not"), hyphenated words split ("left-hand" -> "left
     hand"). Each word is looked up in a small lexicon: colour, other colour, shape noun / adjective, generic object
     noun, "everything else" marker, position (left / center / right + synonyms), bin noun, negation, separator. Words
     whose meaning depends on their neighbours are decided here: "left" in "what is left" / "left over" = the rest;
     "right" in "right away" / "the right bins" (plural, correct ones) is no position; "left hand" / "right arm"
     without a bin noun after it is the robot's arm; "one(s)" after a position is a bin ("the right one"), after a
     colour an object; "can" is a cylinder only after a determiner / colour; "orange" is the bins' colour unless an
     object noun follows ("orange cubes": an unknown object colour).
     Typos: an unknown word of >= 3 letters one edit (substitution, insertion, deletion, transposition) from exactly
     one colour / position / shape word of >= 4 letters with the same first letter is read as that word ("yelow",
     "rigth", "midle", "traingle"), unless it is a real word listed in _REAL ("lift", "light" ... never reach it
     anyway: different first letter or listed). "red" is too short to correct (only "rde" is read as red).
  2. Bin terms: a position word with its modifiers / bin noun ("the left-most box", "the bin on the right" reads the
     position), lists of positions joined by and / or ("the left and right bins") into one multi-bin term, "(from)
     left to right" = the three bins in that order. "the other / remaining / last bin" = the one bin no other clause
     used; "the same bin" = the previous clause's. Counted bins need a direction: "the second bin from the left",
     "counting from the right, the first ..." (the direction carries over to later ordinals: "... the yellow in the
     second"); without one "the first / second bin" is ambiguous (the key is dropped). "The one between them / the
     other two", "in between", "between the left and right bins" = center.
  3. Object terms (noun phrases): runs of colour / shape / object words with fillers ("the red ones", "objects that
     are blue", "every triangular prism"); adjectives joined by and / or / comma distribute over the noun ("red and
     blue cubes" = red cubes + blue cubes); a new noun starts a new phrase ("red cubes and blue cylinders"). Keys:
     colour x shape if both, else colour, else shape, else "default" for everything / the rest / all other objects /
     all objects; a bare "the objects" is "default" only alone with one bin ("put the objects in the left bin") and
     right after a determiner (not "shiny objects": an unknown word there may be a misspelt colour).
  4. Clauses: sentences (. ! ? newline, bullets) split at , ; : and / or / then / but / while ... (not the ones
     consumed in 2 / 3). A clause with a negation word (not, no, except, without, instead, ...) is dropped (problem);
     if it names objects ("red goes left, but not the cubes"), every key covering any of them is dropped as well.
  5. Pairing, per sentence: "respectively" (or one multi-bin term and as many objects: implicit, a warning) zips
     objects and bins in order. Otherwise a clause with objects and bins pairs them by order and proximity: object
     first ("red goes left") -> each bin takes the objects since the previous bin; bin first ("the left bin is for
     red") -> each bin takes the objects up to the next one. A clause with only objects joins the next clause's bin
     (object-first lists: "red, then blue cubes go left"), or the previous one after a bin-first clause ("the left
     bin is for red, and blue") or with "too / also / as well"; a clause with only a bin takes the pending objects
     ("red: left") or the next clause's ("left bin: red"), else (after an object-first clause) the previous clause's
     objects ("red goes left and into the right bin" -> a conflict). A text before ":" with only objects is a header
     ("Sort the objects: ...") and pairs only with a bare bin after the colon ("Red: left").
  6. Result: a key paired with two bins (or a multi-bin term) is dropped (conflict). ok = something understood and no
     error (conflict, ambiguous bin, negation in a clause with a bin or a colour / shape, nothing understood).
     Warnings: objects without a bin, a bin without objects, unknown colour, pairing in order assumed, colour and
     shape keys that disagree for some object (resolve_bin: the colour wins), every key naming one shape.
     conf per key: 1.0, x0.8 typo-corrected / implicit "respectively" / "the objects" read as everything, x0.7 "the
     other bin".
Readings: "box(es)", "container", "basket", ... are bins, never objects or cubes (the bins are orange boxes);
"block(s)" is a generic object word ("the red blocks" = red objects; "square blocks" = cubes); "square" / "cuboid" /
"rectangular" = cube, "round" / "circle" / "disc" / "can" = cylinder, "prism" / "wedge" / "pyramid" = triangle ("prism"
gives way to another shape word in the same phrase: "square prism" = cube). Shape words are always shape keys:
"the red cubes" is (red, cube), never "red" (literal reading; a warning when every key names the same shape).
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from aaltonomous.interface.tasks import BINS, COLORS, SHAPES

# ---------------------------------------------------------------------------------------------------- lexicon
_COLOR = {"red": "red", "reds": "red", "crimson": "red", "scarlet": "red",
          "yellow": "yellow", "yellows": "yellow", "golden": "yellow", "gold": "yellow",
          "blue": "blue", "blues": "blue", "navy": "blue", "azure": "blue"}
_OTHER_COLOR = {"green", "greens", "purple", "violet", "pink", "white", "black", "brown", "grey", "gray", "cyan",
                "magenta", "beige", "silver", "turquoise", "oranges"}        # "orange": the bins' colour (contextual)
# shape words: (shape, is a noun); "prism" is weak (another shape word in the phrase wins)
_SHAPE = {"cube": ("cube", 1), "cubes": ("cube", 1), "cuboid": ("cube", 1), "cuboids": ("cube", 1),
          "squares": ("cube", 1), "dice": ("cube", 1), "cubic": ("cube", 0), "cubical": ("cube", 0),
          "square": ("cube", 0), "rectangular": ("cube", 0), "rectangle": ("cube", 1), "rectangles": ("cube", 1),
          "cylinder": ("cylinder", 1), "cylinders": ("cylinder", 1), "cylindrical": ("cylinder", 0),
          "round": ("cylinder", 0), "circular": ("cylinder", 0), "circle": ("cylinder", 1), "circles": ("cylinder", 1),
          "disc": ("cylinder", 1), "discs": ("cylinder", 1), "disk": ("cylinder", 1), "disks": ("cylinder", 1),
          "puck": ("cylinder", 1), "pucks": ("cylinder", 1), "cans": ("cylinder", 1), "tube": ("cylinder", 1),
          "tubes": ("cylinder", 1),
          "triangle": ("triangle", 1), "triangles": ("triangle", 1), "triangular": ("triangle", 0),
          "prism": ("triangle", 1), "prisms": ("triangle", 1), "wedge": ("triangle", 1), "wedges": ("triangle", 1),
          "pyramid": ("triangle", 1), "pyramids": ("triangle", 1)}
_WEAK_SHAPE = {"prism", "prisms"}
_OBJ = {"object", "objects", "item", "items", "thing", "things", "piece", "pieces", "block", "blocks", "shape",
        "shapes", "toy", "toys", "part", "parts", "element", "elements", "stuff", "article", "articles", "colour",
        "colours", "color", "colors"}
_OBJ_PLURAL_ALL = {"objects", "items", "things", "pieces", "blocks", "shapes", "toys", "parts", "elements", "articles"}
_DEF = {"else", "rest", "remaining", "remainder", "remains", "other", "others", "leftover", "leftovers", "everything",
        "anything", "whatever", "otherwise"}
_ALL = {"all", "every", "each", "any"}
_POS = {"left": "left", "leftmost": "left", "lefthand": "left",
        "center": "center", "centre": "center", "middle": "center", "central": "center", "centermost": "center",
        "centremost": "center", "mid": "center", "centered": "center", "centred": "center",
        "right": "right", "rightmost": "right", "righthand": "right"}
_POSMOD = {"most", "hand", "side", "end", "far", "very"}
_BINN = {"bin", "bins", "box", "boxes", "container", "containers", "basket", "baskets", "tray", "trays", "crate",
         "crates", "bucket", "buckets", "tub", "tubs", "receptacle", "receptacles", "compartment", "compartments",
         "bowl", "bowls", "side", "sides", "slot", "slots", "spot", "pile", "piles", "area"}
_BINN_PLURAL = {w for w in _BINN if w.endswith("s") and w not in ("bins",)} | {"bins"}
_NEG = {"not", "no", "never", "except", "excluding", "without", "unless", "nor", "neither", "none", "nothing",
        "besides", "apart", "instead", "avoid"}
_SEP = {",", ";", ":", "and", "or", "then", "while", "whereas", "but", "plus", "(", ")", "-", "after", "next",
        "finally", "lastly", "first", "afterwards", "/", "&", "+"}
_JOIN = {",", "and", "or", "/", "&", "+"}            # may join adjectives / positions into one term
_SENT = {".", "!", "?", "\n"}
_TOO = {"too", "also", "likewise", "aswell"}
_PRON = {"them", "it", "these", "those", "they"}
_OTHERBIN = {"other", "remaining", "last", "third"}  # + bin noun: "the other bin" (third: after two named bins)
_ORDINAL = {"first", "second", "1st", "2nd", "3rd"}
_ORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3}
_ORD_END = {"", ",", ".", ";", ":", "and", "!", "?", ")", "\n", "-"}   # "... and the blue ones in the third."
_ORD_VERB = {"for", "gets", "get", "takes", "take", "is", "holds", "receives", "should", "will", "must", "contains"}
_BETWEEN_FILL = ("the", "two", "other", "others", "outer", "them", "those", "these", "both", "ones", "one", "bins",
                 "boxes", "containers")


def _between(w, i):
    """'between' names the centre bin: 'in between', 'between them / the other two / the others', 'between the
    left and right bins' (not 'a gap between the objects')."""
    j = i + 1
    while j < len(w) and w[j] in _BETWEEN_FILL:
        j += 1
    named = any(v in ("two", "other", "others", "outer", "them", "those", "these", "both") for v in w[i + 1:j])
    return (i > 0 and w[i - 1] == "in") or named or (j < len(w) and w[j] in ("left", "right"))
_FILL = {"the", "a", "an", "of", "that", "which", "are", "is", "those", "these", "its", "their", "coloured", "colored",
         "shaped", "sized", "small", "little", "big", "large", "tiny", "plastic", "wooden", "foam", "solid", "painted",
         "bright", "dark", "light", "rounded", "single", "individual", "both", "kind", "kinds", "type", "types"}
_NP_FILL = _FILL                                     # words a noun phrase may contain ("objects that are red")
_CAN_NEXT = {"", ",", ".", ";", ":", "and", "or", "in", "into", "on", "to", "-", "goes", "belongs", "is", "should",
             "must", "will"}                     # "the red can goes left" (noun) vs "the yellow can go left" (modal)
_DET = {"the", "a", "an", "each", "every", "any", "all", "this", "that", "these", "those", "your", "my", "its"}
# real words one edit from a key word (with its first letter): never typo-corrected
_REAL = {"lift", "loft", "lest", "lent", "let", "lifts", "lefty", "rights", "righty", "rigid", "censer", "muddle",
         "meddle", "blew", "blur", "blub", "blues", "cute", "cure", "cubs", "cuba", "cub", "cubed", "cures", "prim",
         "prims", "prise", "wedged", "yell", "cent", "cents", "lefts", "ride", "rite", "prisms", "trial", "tribe",
         "mild", "mile", "midst", "cyan", "lead", "leaf", "bled", "blot", "clue", "cubit", "wedgy", "riot", "rig",
         "rich", "ridge", "might", "midge", "midday", "middy", "pyre", "yeller", "yellows", "boxes", "cylinders"}
_TYPO_TARGETS = ("yellow", "blue", "left", "right", "center", "centre", "middle", "leftmost", "rightmost", "cube",
                 "cubes", "cylinder", "cylinders", "triangle", "triangles", "triangular", "cylindrical", "prism",
                 "prisms", "wedge", "wedges", "circle", "circles", "square", "squares")
_KNOWN = (set(_COLOR) | _OTHER_COLOR | set(_SHAPE) | _OBJ | _DEF | _ALL | set(_POS) | _POSMOD | _BINN | _NEG | _SEP
          | _TOO | _PRON | _OTHERBIN | _ORDINAL | _FILL | _DET | _REAL | {"orange", "can", "one", "ones", "same", "than",
                                                                         "respectively", "hand", "arm", "arms"})


def _edit1(a, b):
    """True if a and b are one substitution, adjacent transposition, insertion or deletion apart."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        return len(d) == 1 or (len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]])
    s, t = (a, b) if len(a) < len(b) else (b, a)
    i = next((i for i in range(len(s)) if s[i] != t[i]), len(s))
    return s[i:] == t[i + 1:]


def _typo(w):
    """The key word an unknown word is a one-edit typo of, or None (ambiguous / too short / a listed real word)."""
    if len(w) < 3 or not w.isalpha() or w in _KNOWN:
        return None
    hits = {t for t in _TYPO_TARGETS if len(t) >= 4 and t[0] == w[0] and len(w) >= len(t) - 1 and _edit1(w, t)}
    if w == "rde":                                   # "red": 3 letters, only its transposition is no real word
        hits.add("red")
    return hits.pop() if len(hits) == 1 else None


# ---------------------------------------------------------------------------------------------------- tokens
@dataclass
class Tok:
    kind: str          # COL BADCOL SHAPE OBJ DEF ALL POS BINN OTHERBIN ORDBIN SAMEBIN NEG SEP SENT TOO PRON RESP
                       # FILL OTHER
    val: object        # colour / shape / bin / separator ...
    word: str
    a: int             # char span in the original text
    b: int
    typo: str = ""     # the word as written, if typo-corrected
    noun: bool = False


_TOKEN = re.compile(r"[a-z0-9]+|[.,;:!?()\n/&+]|[-–—]")
_SUFFIX = {"s": "is", "re": "are", "ll": "will", "d": "would", "ve": "have", "m": "am"}


def _tokens(text):
    """[(word, start, end)]: lower case, hyphenated words split ("left-hand" -> "left", "hand"), contractions
    expanded ("don't" -> "do", "not"; "what's" -> "what", "is"). Spans index the original text (evidence)."""
    s = re.sub(r"(?<=[a-z])[-‐](?=[a-z])", " ", text.lower())   # same length: spans stay valid
    raw = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(s)]
    out, i = [], 0
    while i < len(raw):
        w, a, b = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else None
        if nxt and nxt[1] == b + 1 and s[b] in "'’" and (nxt[0] in _SUFFIX or (nxt[0] == "t" and w.endswith("n"))):
            if nxt[0] == "t":
                out += [({"can": "ca", "won": "wo"}.get(w, w[:-1]), a, b), ("not", b, nxt[2])]
            else:
                out += [(w, a, b), (_SUFFIX[nxt[0]], b, nxt[2])]
            i += 2
            continue
        out.append(("-" if w in "–—" else w, a, b))
        i += 1
    return out


def _classify(words):
    n = len(words)
    w = [x[0] for x in words]
    toks = []
    for i, (word, a, b) in enumerate(words):
        prev = w[i - 1] if i else ""
        prev2 = w[i - 2] if i > 1 else ""
        nxt = w[i + 1] if i + 1 < n else ""
        nxt2 = w[i + 2] if i + 2 < n else ""
        typo, x = "", word
        if x not in _KNOWN and not x.isdigit() and x not in _SENT:
            t = _typo(x)
            if t is not None:
                typo, x = word, t

        def T(kind, val=None, **kw):
            return Tok(kind, val, x, a, b, typo, **kw)

        if x in _SENT:
            toks.append(T("SENT", x))
        elif x in ("left", "right") and (nxt in ("hand", "hands", "arm", "arms", "gripper")
                                         and nxt2 not in _BINN and nxt2 not in ("one", "ones")):
            toks.append(T("OTHER"))                                    # the robot's arm, not a bin
        elif x == "left" and (prev in ("what", "whatever", "anything", "everything", "nothing", "still")
                              or (prev in ("is", "are", "remain", "remains", "be")
                                  and (prev2 in ("what", "whatever", "that", "which", "anything", "everything", "all",
                                                 "nothing", "else", "still", "stuff") or prev2 in _BINN))
                              or nxt in ("over", "behind", "unsorted", "on", "lying", "standing")):
            if any(v in _BINN for v in w[max(0, i - 3):i]):
                toks.append(T("OTHERBIN", "left"))                     # "whichever bin is left": the remaining bin
            else:
                toks.append(T("DEF", "rest"))                          # "what is left", "anything left over"
        elif x == "right" and (nxt in ("away", "now", "after", "before", "here", "there", "way", "order", "place",
                                       "places", "answer")
                               or (prev in ("that", "all") and nxt in _SENT | {""})
                               or (nxt in _BINN_PLURAL and not (prev in _JOIN or prev2 in _JOIN))):
            toks.append(T("OTHER"))                                    # "right away", "the right bins" (correct)
        elif x in _POS:
            toks.append(T("POS", _POS[x]))
        elif x == "between" and _between(w, i):
            toks.append(T("BETWEEN", x))                               # "the one between them": the centre bin
        elif x in _ORD and (nxt in _BINN or nxt in ("one", "from", "counting", "starting")
                            or (prev == "the" and (nxt in _ORD_END or nxt in _ORD_VERB))):
            toks.append(T("ORD", _ORD[x]))                             # "the second bin (from the left)"
        elif x == "so" and nxt in ("do", "does", "should", "must", "will", "are", "is", "go", "goes"):
            toks.append(T("TOO", x))                                   # "and so do the yellow ones"
        elif x in _COLOR:
            toks.append(T("COL", _COLOR[x]))
        elif x == "orange" or x in _OTHER_COLOR:
            j = i + 1
            while j < n and w[j] in _NP_FILL | {"and", ","} | set(_COLOR) | _OTHER_COLOR | {"orange"}:
                j += 1
            obj = j < n and (w[j] in _OBJ or w[j] in _SHAPE or w[j] in ("one", "ones", "cans"))
            toks.append(T("BADCOL", x) if (obj or x != "orange") and not (j < n and w[j] in _BINN) else T("OTHER"))
        elif x in _SHAPE:
            s, noun = _SHAPE[x]
            toks.append(T("SHAPE", s, noun=bool(noun)))
        elif x == "can" and (prev in _DET or prev in _COLOR or prev in _OTHER_COLOR) and nxt in _CAN_NEXT:
            toks.append(T("SHAPE", "cylinder", noun=True))
        elif x in ("one", "ones"):
            if prev in _POS or (prev in _POSMOD and prev2 in _POS):
                toks.append(T("BINN", x))                              # "the right one"
            elif prev in _COLOR or prev in _OTHER_COLOR or prev in _DEF or prev in _ALL or prev in _NP_FILL \
                    or prev == "orange" or (prev in _SHAPE and not _SHAPE[prev][1]):
                toks.append(T("OBJ", x, noun=True))
            else:
                toks.append(T("OTHER"))                                # "one by one", "one at a time"
        elif x in _OTHERBIN and (nxt in _BINN or (nxt in ("one",) and x != "other")) and nxt not in _BINN_PLURAL:
            toks.append(T("OTHERBIN", x))                              # "the remaining bin", "the last one"
        elif x == "same" and (nxt in _BINN or nxt in ("one", "place", "spot")):
            toks.append(T("SAMEBIN", x))
        elif x == "same" and nxt in ("for", "with", "applies", "goes"):
            toks.append(T("TOO", x))                                   # "same for yellow"
        elif x == "ahead" and prev == "straight":
            toks.append(T("POS", "center"))
        elif x == "other" and nxt == "than":
            toks.append(T("NEG", x))
        elif x in _DEF:
            toks.append(T("DEF", x))
        elif x in _ALL:
            toks.append(T("ALL", x))
        elif x in _OBJ:
            toks.append(T("OBJ", x, noun=True))
        elif x in _BINN:
            toks.append(T("BINN", x))
        elif x in _NEG:
            toks.append(T("NEG", x))
        elif x in _TOO or (x == "well" and prev == "as"):
            toks.append(T("TOO", x))
        elif x == "respectively" or (x == "order" and prev in ("same", "that", "this")):
            toks.append(T("RESP", x))
        elif x in _SEP:
            toks.append(T("SEP", x))
        elif x in _PRON:
            toks.append(T("PRON", x))
        elif x in _FILL or x in _POSMOD:
            toks.append(T("FILL", x))
        else:
            toks.append(T("OTHER", x))
    return toks


# ---------------------------------------------------------------------------------------------------- terms
@dataclass
class NP:
    colors: list = field(default_factory=list)
    shapes: list = field(default_factory=list)
    bad: list = field(default_factory=list)
    generic: bool = False      # object noun ("objects", "ones", "blocks")
    plural_all: bool = False   # a plural generic noun ("the objects": all of them when alone with one bin)
    default: bool = False      # everything / the rest / all other objects ...
    alldet: bool = False       # all / every / each / any
    a: int = 0
    b: int = 0
    typo: list = field(default_factory=list)
    i0: int = 0
    i1: int = 0

    def keys(self):
        """Mapping keys this phrase names (empty: none, e.g. only an unknown colour / a bare 'the one')."""
        sh = list(dict.fromkeys(self.shapes))
        if self.colors and sh:
            return [(c, s) for c in dict.fromkeys(self.colors) for s in sh]
        if self.colors:
            return list(dict.fromkeys(self.colors))
        if sh:
            return sh
        if self.bad:
            return []
        if self.default or (self.alldet and self.generic):
            return ["default"]
        return []


@dataclass
class BT:
    bins: list                 # bin names; "?other" / "?same" / "?ordinal" placeholders
    a: int = 0
    b: int = 0
    i0: int = 0
    i1: int = 0
    typo: list = field(default_factory=list)


def _terms(toks):
    """Tokens -> units: NP / BT objects and the remaining structural tokens (SEP, SENT, NEG, TOO, PRON, RESP)."""
    units, i, n = [], 0, len(toks)
    direction = None           # "from the left / right" of the last counted bin (carries over to later ordinals)
    while i < n:
        t = toks[i]
        if t.word in ("counting", "starting", "numbered", "numbering", "going") and i + 1 < n \
                and toks[i + 1].word == "from":         # "counting from the right, the first bin ...": a direction
            m = i + 2
            while m < n and toks[m].word in ("the", "your", "my", "its", "far"):
                m += 1
            if m < n and toks[m].kind == "POS" and toks[m].val in ("left", "right"):
                direction = toks[m].val
                if m + 2 < n and toks[m + 1].word == "to" and toks[m + 2].kind == "POS":
                    m += 2                              # "numbered from left to right"
                i = m + 1
                continue
        if (t.kind == "POS" and t.val in ("left", "right") and i + 2 < n and toks[i + 1].word == "to"
                and toks[i + 2].kind == "POS" and {t.val, toks[i + 2].val} == {"left", "right"}):
            order = list(BINS) if t.val == "left" else list(BINS)[::-1]      # "(from) left to right: red, ..."
            units.append(BT(order, t.a, toks[i + 2].b, i, i + 2))
            i += 3
            continue
        if t.kind == "POS":
            bt = BT([t.val], t.a, t.b, i, i, [t.typo] if t.typo else [])
            j, noun_after_first = i + 1, False
            while True:
                while j < n and toks[j].kind == "FILL" and toks[j].word in _POSMOD | {"of"}:
                    j += 1
                if j < n and toks[j].kind == "BINN":
                    noun_after_first = True
                    bt.b, bt.i1 = toks[j].b, j
                    j += 1
                k = j
                if k < n and toks[k].kind == "SEP" and toks[k].val in _JOIN:
                    m = k + 1
                    while m < n and toks[m].kind == "FILL" and toks[m].word in ("the", "a", "far", "very"):
                        m += 1
                    if m < n and toks[m].kind == "POS" and not (toks[k].val == "," and noun_after_first):
                        bt.bins.append(toks[m].val)
                        bt.b, bt.i1 = toks[m].b, m
                        if toks[m].typo:
                            bt.typo.append(toks[m].typo)
                        j = m + 1
                        continue
                break
            if bt.i1 < i:
                bt.i1 = i
            bt.b = max(bt.b, toks[bt.i1].b)
            units.append(bt)
            i = bt.i1 + 1
            continue
        if t.kind == "BETWEEN":                  # centre; swallow "the other two" / "the left and right bins"
            j = i + 1
            while j < n and (toks[j].word in _BETWEEN_FILL or (toks[j].kind == "POS" and toks[j].val != "center")
                             or (toks[j].word in ("and", "&") and j + 1 < n and toks[j + 1].word in ("the", "left",
                                                                                                    "right"))):
                j += 1
            units.append(BT(["center"], t.a, toks[j - 1].b, i, j - 1))
            i = j
            continue
        if t.kind == "ORD":                      # counted bins: need a direction, stated here or before
            j = i + 1 if i + 1 < n and (toks[i + 1].kind == "BINN" or toks[i + 1].word == "one") else i
            k = j + 1
            while k < n and toks[k].word in ("counting", "starting", "going", "as", "seen", "looking"):
                k += 1
            if k < n and toks[k].word == "from":
                m = k + 1
                while m < n and toks[m].word in ("the", "your", "my", "its", "far"):
                    m += 1
                if m < n and toks[m].kind == "POS" and toks[m].val in ("left", "right"):
                    direction = toks[m].val
                    j = m + 1 if m + 1 < n and toks[m + 1].word in ("side", "end", "hand") else m
            if direction is not None:
                order = list(BINS) if direction == "left" else list(BINS)[::-1]
                ph = order[t.val - 1]
            else:                                # "the third bin" (no direction): the one left over, as before
                ph = "?other" if t.val == 3 and j > i else "?ordinal"
            units.append(BT([ph], t.a, toks[j].b, i, j))
            i = j + 1
            continue
        if t.kind in ("OTHERBIN", "SAMEBIN"):
            ph = {"OTHERBIN": "?other", "SAMEBIN": "?same"}[t.kind]
            j = i + 1 if i + 1 < n and toks[i + 1].kind in ("BINN", "FILL", "OTHER") else i
            units.append(BT([ph], t.a, toks[j].b, i, j))
            i = j + 1
            continue
        if t.kind in ("COL", "BADCOL", "SHAPE", "OBJ", "DEF", "ALL"):
            np_ = NP(a=t.a, b=t.b, i0=i, i1=i)
            j, has_noun, last_adj = i, False, False
            while j < n:
                u = toks[j]
                if u.kind in ("COL", "BADCOL", "SHAPE", "OBJ", "DEF", "ALL"):
                    if u.kind == "COL":
                        np_.colors.append(u.val)
                    elif u.kind == "BADCOL":
                        np_.bad.append(u.val)
                    elif u.kind == "SHAPE":
                        np_.shapes.append((u.val, u.word in _WEAK_SHAPE))
                    elif u.kind == "OBJ":
                        np_.generic = True
                        np_.plural_all |= u.word in _OBJ_PLURAL_ALL
                    elif u.kind == "DEF":
                        np_.default = True
                    else:
                        np_.alldet = True
                    if u.typo:
                        np_.typo.append(u.typo)
                    has_noun |= u.noun
                    last_adj = u.kind in ("COL", "BADCOL") or (u.kind == "SHAPE" and not u.noun)
                    np_.b, np_.i1 = u.b, j
                    j += 1
                    continue
                if u.kind == "FILL" and u.word in _NP_FILL:
                    j += 1
                    continue
                if u.kind == "SEP" and u.val in _JOIN and not has_noun and last_adj:   # "red and blue (cubes)"
                    m = j + 1
                    while m < n and toks[m].kind == "FILL" and toks[m].word in ("the", "a", "an", "both"):
                        m += 1
                    if m < n and (toks[m].kind in ("COL", "BADCOL") or (toks[m].kind == "SHAPE"
                                                                       and not toks[m].noun)):
                        j = m
                        continue
                break
            sh = [s for s, weak in np_.shapes if not weak] or [s for s, _ in np_.shapes]
            np_.shapes = sh
            # "the objects" = all of them, but not "rde objects" / "shiny objects" (an unknown word may be a colour)
            np_.plural_all &= i > 0 and toks[i - 1].word in ("the", "these", "those", "all", "any", "every", "each")
            units.append(np_)
            i = np_.i1 + 1
            continue
        if t.kind in ("SEP", "SENT", "NEG", "TOO", "PRON", "RESP"):
            units.append(t)
        i += 1
    return units


# ---------------------------------------------------------------------------------------------------- result
@dataclass
class Parse:
    text: str
    mapping: dict              # key -> bin (conflicting keys dropped)
    keys: list                 # [{"key", "bin", "evidence", "conf"}]
    problems: list             # [{"kind", "severity": "error" | "warning", "msg", "text"}]

    @property
    def ok(self):
        return bool(self.mapping) and not any(p["severity"] == "error" for p in self.problems)

    def unsorted(self):
        """Object types (colour, shape) that get no bin from this mapping."""
        return [(c, s) for c in COLORS for s in SHAPES if object_bin(self.mapping, c, s) is None]

    def summary(self):
        if not self.mapping:
            what = "nothing understood"
        else:
            what = ", ".join(f"{_kname(k['key'])}->{k['bin']}" + (f" ({k['conf']:.1f})" if k["conf"] < 1 else "")
                             for k in self.keys)
        un = set(self.unsorted())
        names = [c if all((c, s) in un for s in SHAPES) else " ".join(
            [c, "/".join(s for s in SHAPES if (c, s) in un)]) for c in COLORS if any((c, s) in un for s in SHAPES)]
        rest = f"; no bin: {', '.join(names)}" if self.mapping and un else ""
        probs = "".join(f"; {p['severity']}: {p['msg']}" for p in self.problems)
        return f"{'ok' if self.ok else 'NOT OK'}: {what}{rest}{probs}"


def _kname(k):
    return " ".join(k) if isinstance(k, tuple) else str(k)


def resolve_bin(mapping, color, shape=None):
    """scene.scene.resolve_bin (same rule, kept here so this module needs no yaml / scene): (colour, shape), colour,
    shape, then "default"; None if no key applies."""
    for k in ((color, shape), color, shape, "default"):
        if k is not None and k in mapping:
            return mapping[k]
    return None


def object_bin(mapping, color, shape=None):
    """Bin of one object, or None (it stays on the counter for now): resolve_bin over every colour / shape the object
    may have (colour "unknown" / None -> any of COLORS, shape None -> any of SHAPES); one bin for all of them -> that
    bin, else None. Never a guess: a colour-only mapping needs no shape; a shape key needs the shape."""
    cs = [color] if color in COLORS else list(COLORS)
    ss = [shape] if shape in SHAPES else list(SHAPES)
    if color in COLORS and shape in SHAPES:
        return resolve_bin(mapping, color, shape)
    bins = {resolve_bin(mapping, c, s) for c in cs for s in ss}
    return bins.pop() if len(bins) == 1 else None


def _clauses(sent):
    """One sentence's units -> clauses {items, neg, too, pron, resp, sep (separator before), end (after)}. Phrases
    that name nothing (a bare "the one") are left out, so they do not decide the clause's order."""
    def new(sep):
        return {"items": [], "neg": False, "too": False, "pron": False, "resp": False, "sep": sep, "end": None}
    segs, seg = [], new(None)
    for u in sent:
        if isinstance(u, Tok) and u.kind == "SEP":
            seg["end"] = u.val
            segs.append(seg)
            seg = new(u.val)
        elif isinstance(u, Tok):
            key = {"NEG": "neg", "TOO": "too", "PRON": "pron", "RESP": "resp"}.get(u.kind)
            if key:
                seg[key] = True
        elif isinstance(u, BT) or u.keys() or u.plural_all or u.bad:
            seg["items"].append(u)
    segs.append(seg)
    return [s for s in segs if s["items"] or s["neg"] or s["too"] or s["pron"] or s["resp"]]


def _objects(k):
    """The (colour, shape) types a mapping key covers."""
    if k == "default":
        return set(itertools.product(COLORS, SHAPES))
    if isinstance(k, tuple):
        return {k}
    return {(k, s) for s in SHAPES} if k in COLORS else {(c, k) for c in COLORS}


def _overlap(a, b):
    return bool(_objects(a) & _objects(b))


def _span(items):
    return min(x.a for x in items), max(x.b for x in items)


class _Reader:
    """Pairs object phrases with bin terms (module docstring, step 5) and collects the assignments."""

    def __init__(self, text):
        self.text = text
        self.assigned = {}         # key -> [(bin | tuple of bins | "?placeholder", evidence, conf)]
        self.problems = []
        self.last_bin = None       # bin of the last pairing ("the same bin", "too")
        self.prev_pend = []        # objects the previous sentence left without a bin ("The red ones? Left.")
        self.header = None         # a bin heading a list on the next lines ("Left bin:\n- red\n- blue")
        self.neg_keys = []         # keys named in negated clauses

    def prob(self, kind, sev, msg, items=()):
        a, b = _span(items) if items else (0, 0)
        self.problems.append({"kind": kind, "severity": sev, "msg": msg, "text": self.text[a:b].strip()})

    def give(self, objs, bt, conf=1.0):
        """Object phrases objs -> bin term bt. Generic phrases count only alone ("put the objects in the left bin" =
        everything); phrases with an unknown colour give nothing (a warning)."""
        for o in objs:
            if o.bad:
                self.prob("unknown_colour", "warning", f"colour {'/'.join(o.bad)} is not red / yellow / blue", [o])
        spec = [o for o in objs if o.keys()]
        if not spec and objs and all(o.plural_all and not o.bad for o in objs):
            spec, conf = [NP(default=True)], conf * 0.8
        if not spec:
            return
        bins = [(self.last_bin or "?same") if x == "?same" else x for x in bt.bins]
        val = bins[0] if len(bins) == 1 else tuple(bins)
        if any(o.typo for o in spec) or bt.typo:
            conf *= 0.8
        a, b = _span(list(objs) + [bt])
        for o in spec:
            for k in o.keys():
                self.assigned.setdefault(k, []).append((val, self.text[a:b].strip(), conf))
        if isinstance(val, str) and val in BINS:
            self.last_bin = val

    def sentence(self, sent):
        segs = []
        for s in _clauses(sent):
            if not s["neg"]:
                segs.append(s)
                continue
            strong = any(isinstance(x, BT) or x.colors or x.shapes for x in s["items"])
            if strong:   # "red goes left, but not the cubes": what it excludes may be inside another clause's key
                self.neg_keys += [k for x in s["items"] if isinstance(x, NP) for k in x.keys()]
            if s["items"]:
                self.prob("negation", "error" if strong else "warning", "negation: clause ignored" if strong
                          else "negation (no bin / colour / shape in it) ignored", s["items"])
        if not segs:
            return
        prev_pend, self.prev_pend = self.prev_pend, []
        if self._zip(segs):
            self.header = None
            if prev_pend:
                self.prob("object_without_bin", "warning", "objects without a bin", prev_pend)
            return
        pend, pend_b = [], self.header     # [(phrase, before a colon)], a bin waiting for objects
        last_dir, last_objs, last_bo = None, [], None
        for k, s in enumerate(segs):
            O = [x for x in s["items"] if isinstance(x, NP)]
            B = [x for x in s["items"] if isinstance(x, BT)]
            nxt = segs[k + 1] if k + 1 < len(segs) else None
            if O and B:
                self.header = pend_b = None
                if isinstance(s["items"][0], NP):      # object first: a bin takes the objects since the previous bin
                    grp = [o for o, hdr in pend if not hdr]
                    for x in s["items"]:
                        if isinstance(x, NP):
                            grp.append(x)
                        elif grp:
                            self.give(grp, x)
                            last_objs, grp = grp, []
                        else:                          # "the bin to the left of the middle one": the first counts
                            self.prob("bin_ignored", "warning", "second bin term ignored", [x])
                    pend = [(o, False) for o in grp]   # trailing objects wait for the next bin
                    last_dir = "OB"
                else:                                  # bin first: a bin takes the objects up to the next bin
                    if any(not h for _, h in pend):
                        self.prob("object_without_bin", "warning", "objects without a bin", [o for o, _ in pend])
                    pend, bt, grp = [], None, []
                    for x in s["items"]:
                        if isinstance(x, BT):
                            if bt is not None and grp:
                                self.give(grp, bt)
                            elif bt is not None:                 # "the bin left of the middle one is for red"
                                self.prob("bin_ignored", "warning", "second bin term ignored", [x])
                                continue
                            bt, grp = x, []
                        else:
                            grp.append(x)
                    if grp:
                        self.give(grp, bt)
                        last_objs = grp
                    elif nxt is not None:
                        pend_b = bt
                    last_dir, last_bo = "BO", bt
            elif O:
                if (pend_b is None and nxt is not None and not any(isinstance(x, NP) for x in nxt["items"])
                        and any(isinstance(x, BT) for x in nxt["items"])):
                    pend += [(o, s["end"] == ":") for o in O]   # a bare bin follows: "the same with blue, but right"
                elif s["too"] and self.last_bin is not None:    # "blue too", "and so do the yellow ones"
                    self.give(O, BT([self.last_bin], *_span(O)))
                elif pend_b is not None:                         # "Left bin: red" / the header's list
                    self.give(O, pend_b)
                    if pend_b is not self.header:
                        pend_b = None
                elif last_dir == "BO" and last_bo is not None and s["sep"] in _JOIN:
                    self.give(O, last_bo)                        # "the left bin is for red, and blue"
                else:
                    pend += [(o, s["end"] == ":") for o in O]
                last_objs = O
            elif B:
                bt = B[0]
                if len(B) > 1:
                    self.prob("bin_without_object", "warning", "several bins without an object", B)
                if pend:                                         # "Red: left", "for red, use the right bin"
                    last_objs = [o for o, _ in pend]
                    self.give(last_objs, bt)
                    pend = []
                elif prev_pend and k == 0 and (len(segs) == 1 or s["pron"]):
                    self.give(prev_pend, bt)                     # "The red ones? Left." / "... Put them left."
                    last_objs, prev_pend = prev_pend, []
                elif nxt is not None and any(isinstance(x, NP) for x in nxt["items"]) and not any(
                        isinstance(x, BT) for x in nxt["items"]):
                    pend_b = bt                                  # "In the left bin, put red"
                elif last_dir == "OB" and last_objs:
                    self.give(last_objs, bt)                     # "red goes in the left bin, and into the right"
                elif len(segs) == 1:
                    self.header = pend_b = bt                    # "Left bin:" heading the next lines
                else:
                    self.prob("bin_without_object", "warning", "bin without an object", [bt])
        if pend_b is not None and pend_b is not self.header:
            self.prob("bin_without_object", "warning", "bin without an object", [pend_b])
        if prev_pend:
            self.prob("object_without_bin", "warning", "objects without a bin", prev_pend)
        self.prev_pend = [o for o, _ in pend if o.keys()]

    def _zip(self, segs):
        """'respectively', or one multi-bin term as the only bin term: objects and bins in order. True if handled."""
        objs = [x for s in segs for x in s["items"] if isinstance(x, NP)]
        bts = [x for s in segs for x in s["items"] if isinstance(x, BT)]
        resp = any(s["resp"] for s in segs)
        if not ((resp and bts) or (len(bts) == 1 and len(bts[0].bins) > 1)):
            return False
        items = [k for o in objs for k in o.keys()]
        bins = [b for bt in bts for b in bt.bins]
        if items and len(items) == len(bins) > 1:
            if not resp:
                self.prob("in_order", "warning", "objects and bins paired in the order given (respectively assumed)",
                          objs + bts)
            a, b = _span(objs + bts)
            for k, bn in zip(items, bins):
                self.assigned.setdefault(k, []).append((bn, self.text[a:b].strip(), 1.0 if resp else 0.8))
            self.last_bin = bins[-1]
        elif items:
            self.prob("pairing", "error", f"{len(items)} object(s) for {len(bins)} bin(s): cannot pair", objs + bts)
        else:
            self.prob("bin_without_object", "warning", "bins named without objects", bts)
        return True

    def result(self):
        if self.prev_pend:
            self.prob("object_without_bin", "warning", "objects without a bin", self.prev_pend)
        used = {v for vs in self.assigned.values() for v, _, _ in vs if v in BINS}
        free = [b for b in BINS if b not in used]
        keys, mapping, problems = [], {}, list(self.problems)
        for k, vs in self.assigned.items():
            vs = [(free[0], ev, c * 0.7) if v == "?other" and len(free) == 1 else (v, ev, c) for v, ev, c in vs]
            vals = list(dict.fromkeys(v for v, _, _ in vs))
            ev = " | ".join(dict.fromkeys(e for _, e, _ in vs))
            if len(vals) == 1 and vals[0] in BINS:
                mapping[k] = vals[0]
                keys.append({"key": k, "bin": vals[0], "evidence": ev, "conf": round(min(c for _, _, c in vs), 2)})
                continue
            if any(isinstance(v, tuple) for v in vals):
                why = "several bins in one phrase (" + ", ".join("/".join(v) if isinstance(v, tuple) else v
                                                                 for v in vals) + ")"
            elif any(v not in BINS for v in vals):
                why = "bin not clear (" + ", ".join(v.lstrip("?") for v in vals) + ")"
            else:
                why = "assigned to " + " and ".join(vals)
            problems.append({"kind": "conflict", "severity": "error", "msg": f"{_kname(k)}: {why} -> dropped",
                             "text": ev})
        for k in [k for k in mapping if any(_overlap(k, q) for q in self.neg_keys)]:
            ev = next(e["evidence"] for e in keys if e["key"] == k)
            problems.append({"kind": "negated", "severity": "error", "text": ev,
                             "msg": f"{_kname(k)}: overlaps a negated phrase -> dropped"})
            del mapping[k]
            keys = [e for e in keys if e["key"] != k]
        cols = {k: v for k, v in mapping.items() if k in COLORS}
        for sh, b in [(k, v) for k, v in mapping.items() if k in SHAPES]:
            for c, bc in cols.items():
                if bc != b and (c, sh) not in mapping:
                    problems.append({"kind": "overlap", "severity": "warning", "text": "",
                                     "msg": f"{c} {sh}s: {c} -> {bc} and {sh} -> {b}; the colour wins"})
        if not mapping:
            problems.append({"kind": "nothing", "severity": "error", "msg": "no object / bin pair understood",
                             "text": self.text.strip()})
        tup = [k for k in mapping if isinstance(k, tuple)]
        if len(tup) >= 2 and len({s for _, s in tup}) == 1 and all(isinstance(k, tuple) or k == "default"
                                                                    for k in mapping):
            problems.append({"kind": "one_shape", "severity": "warning", "text": "",
                             "msg": f"every key names {tup[0][1]}s: other shapes get no bin"})
        return Parse(self.text, mapping, keys, problems)


def parse_instruction(text) -> Parse:
    """Instruction text -> Parse (mapping, per-key evidence, problems). See the module docstring."""
    text = text or ""
    units = _terms(_classify(_tokens(text)))
    rd, cur = _Reader(text), []
    for u in units + [Tok("SENT", "\n", "\n", len(text), len(text))]:
        if isinstance(u, Tok) and u.kind == "SENT":
            rd.sentence(cur)
            cur = []
        else:
            cur.append(u)
    return rd.result()
