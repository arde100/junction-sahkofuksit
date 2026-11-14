"""Rung 3 (insert_shapes): recognise the task, model the lid, decide what to attempt. Today: nothing (see below).

    m = select_mode({"task": task, "instruction": text})   # InsertMode or None (= rungs 1-2, unchanged)
    lid = LidModel(scene)                                  # holes in the pelvis frame, required object yaw per shape
    rep = reach_report(planner, lid, cfg["insert"])         # per hole + arm: can a top-down release above it be held?
    decide(rep, cfg["insert"])                              # -> {shape: (attempt, why)}

Mode selection (select_mode, called by ClassicalPolicy.reset): the explicit task id wins: "insert_shapes" = insert
mode, "sort_fixed" / "sort_language" = never. Without a task id the text decides (read_insert): a hole word (hole /
slot / opening / cut-out / aperture) together with a lid word or a through / insert / post / fit verb, or a lid word
with through / insert / post, is the insert task; the same with a colour, a bin position or a colour -> bin mapping in it (instruction.parse_instruction) is MIXED,
and so is a negated one ("do not ... through the holes"): not understood (without a lid word it is a sort
instruction as before: "drop the red ones into the left slot"). Both kinds sort nothing: the lid lies on the
left bin, every object a sorter drops there lands on the lid (scene.classify_insert: forced, -1), and the lid's
instruction names no bin, so no object may get one (a paraphrase like "... the matching hole of the lid on the left
box" reads as everything -> left to instruction.py).

Why nothing is attempted (measured 2026-11-14, kinematics.py IK of the planner, sim profile; aaltonomous/sim/
insert_lab.py for the drops): the holes sit in the FAR half of the left box (pelvis frame, scene.lid_holes): cylinder
(0.575, +0.19), cube (0.575, +0.33), triangle (0.522, +0.26), lid top 63 mm above the mat. With the object's bottom a
few mm above the lid the grasp point is at z = counter + 0.089; there the left arm's grasp point reaches x <= 0.49
(y 0.19), 0.48 (y 0.26), 0.46 (y 0.33) in ANY orientation, the same with the joint margins at URDF - 0.05 rad (the
arm is straight: geometry, not margins); the right arm reaches none of the holes. Short by ~3.5 cm (triangle, with
the best pinch offset), 8.5 cm (cylinder), 11.5 cm (cube). decide() therefore only attempts a shape whose hole
reach_report() finds reachable AND for which a delivery exists (insert.deliver; none is implemented), so the mode
leaves every object on the counter (0 points, never -1). reach_report() re-checks the reach from the scene + config
of the run, so a lid elsewhere is reported (not attempted) instead of silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .instruction import _COLOR, _NEG, _OTHER_COLOR, _POS, _tokens, parse_instruction

INSERT_TASK = "insert_shapes"
SORT_TASKS = ("sort_fixed", "sort_language")

_HOLE = {"hole", "holes", "slot", "slots", "opening", "openings", "cutout", "cutouts", "aperture", "apertures",
         "orifice", "orifices"}
_LID = {"lid", "lids", "cover", "covers", "plate", "board", "panel"}
_VERB = {"through", "insert", "inserting", "inserted", "post", "posting", "fit", "fits", "fitting", "thread", "pass",
         "passing", "push", "poke", "drop", "slide", "feed", "into"}
_MATCH = {"same", "matching", "match", "matches", "corresponding", "correspond", "fits", "fit", "fitting", "its",
          "their", "own", "correct", "proper", "appropriate", "respective"}
_CUT = {"cut"}   # "cut-out" (hyphen split by _tokens) -> "cut out"
_THROUGH = {"through", "insert", "inserting", "inserted", "post", "posting"}   # with a lid word, no hole word needed


@dataclass
class InsertRead:
    """read_insert() result. kind: "insert" (the rung-3 task) | "mixed" (lid / hole words with a colour, a bin
    position, a mapping or a negation: not understood) | None (no lid / hole wording)."""
    text: str
    kind: str | None
    cues: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    def summary(self):
        c = ",".join(self.cues) or "-"
        return f"{self.kind or 'not insert'} (cues {c})" + "".join(f"; {p}" for p in self.problems)


def read_insert(text):
    """Is `text` the insert task? See the module docstring (structure: hole word + lid word / insert verb)."""
    toks = [w for w, _, _ in _tokens(text or "")]
    words = set(toks)
    hole = sorted(words & _HOLE) + (["cut out"] if any(a in _CUT and b == "out" for a, b in zip(toks, toks[1:]))
                                    else [])
    ctx = sorted(words & (_LID | _VERB))
    if not hole and not (words & _LID and words & _THROUGH):   # "... everything else through the lid"
        return InsertRead(text, None)
    if not ctx:
        return InsertRead(text, None, hole, ["hole word without a lid / insert verb"])
    cues = hole + ctx + sorted(words & _MATCH)
    probs = []
    cols = sorted(words & (set(_COLOR) | _OTHER_COLOR))
    if cols:
        probs.append(f"colour words {cols}")
    pos = []
    for i, w in enumerate(toks):   # "the right hole" = the correct one, not a bin position
        if w in _POS and not (w == "right" and set(toks[i + 1:i + 3]) & (_HOLE | {"one", "ones", "shape"})):
            pos.append(w)
    if pos:
        probs.append(f"bin positions {sorted(set(pos))}")
    neg = sorted(words & _NEG)
    if neg:
        probs.append(f"negation {neg}")
    m = parse_instruction(text).mapping if (cols or pos) else {}   # (else only "the right hole" could give one)
    if m:
        probs.append(f"a bin mapping ({len(m)} keys)")
    if probs and not words & _LID:   # "drop red into the left slot": a sort instruction (slot = bin), as before
        return InsertRead(text, None, cues, probs + ["no lid word: read as a sort instruction"])
    return InsertRead(text, "mixed" if probs else "insert", cues, probs)


@dataclass
class InsertMode:
    """This episode is (or may be) the rung-3 task: no object gets a bin. source: "task" (the reset's task id) |
    "instruction" (read_insert) | "mixed" (not understood)."""
    source: str
    read: InsertRead | None
    task: str | None = None

    @property
    def understood(self):
        return self.source in ("task", "instruction")

    def note(self):
        r = "" if self.read is None else f"; instruction {self.read.text!r}: {self.read.summary()}"
        if self.understood:
            return f"rung 3 (insert_shapes, from the {self.source}{r}): no bins, every object stays on the counter " \
                   f"unless its hole is attempted (insert.py)"
        return f"lid / hole instruction not understood{r}: sorting nothing (the lid lies on a bin)"


def select_mode(info, cfg=None):
    """ClassicalPolicy.reset hook: InsertMode for this reset, or None (rungs 1-2: the sorter as before). info: the
    reset dict (task, instruction, mapping). An explicit info["mapping"] or a sort task id -> None; config
    insert.recognise false -> None (the instruction is parsed as before)."""
    ic = (cfg or {}).get("insert") or {}
    if not ic.get("recognise", True) or info.get("mapping") is not None:
        return None
    task, text = info.get("task"), (info.get("instruction") or "").strip()
    if task in SORT_TASKS:
        return None
    r = read_insert(text) if text else None
    if task == INSERT_TASK:
        return InsertMode("task", r, task)
    if r is None or r.kind is None:
        return None
    return InsertMode("instruction" if r.kind == "insert" else "mixed", r, task)


# ---------------------------------------------------------------------------------------------------- lid model
class LidModel:
    """The lid of scene.yaml (public, fixed) in the pelvis frame: per shape the hole centre (xy), its polygon, the
    object yaw the hole needs (Detection convention: cube face normals at yaw + k 90 deg, triangle apexes at yaw + 90
    + k 120 deg) with its symmetry period, the lid top / underside z and the release height of the grasp point."""

    PERIOD = {"cube": np.pi / 2, "triangle": 2 * np.pi / 3, "cylinder": None}

    def __init__(self, scene):
        self.scene = sc = scene
        self.bin = sc.lid_bin
        z0 = sc.counter_z
        self.z_under, self.z_top = z0 + sc.lid_z[0], z0 + sc.lid_z[1]
        self.holes = {}
        for h in sc.lid_holes():
            c = sc.mat_to_pelvis(np.r_[h["center"], 0.0])[:2]
            poly = sc.mat_to_pelvis(np.c_[h["poly"], np.zeros(len(h["poly"]))])[:, :2]
            self.holes[h["shape"]] = {"xy": c, "poly": poly, "yaw": self._hole_yaw(h["shape"], poly - c)}

    def _hole_yaw(self, shape, rel):
        """Object yaw (Detection convention) that matches the hole outline rel (n, 2, around its centre)."""
        if shape == "cube":     # a face normal = the direction of an edge midpoint
            m = 0.5 * (rel + np.roll(rel, -1, axis=0))
            return float(np.arctan2(m[0, 1], m[0, 0]) % (np.pi / 2))
        if shape == "triangle":  # an apex = a vertex; Detection: apexes at yaw + 90 deg
            return float((np.arctan2(rel[0, 1], rel[0, 0]) - np.pi / 2) % (2 * np.pi / 3))
        return None

    def yaw_error(self, shape, yaw):
        """Smallest rotation (rad, signed) from object yaw `yaw` to a yaw that fits the hole (0 for a cylinder)."""
        P = self.PERIOD[shape]
        if P is None or yaw is None:
            return 0.0
        d = (yaw - self.holes[shape]["yaw"]) % P
        return float(d - P if d > P / 2 else d)

    def release_z(self, gap, dz_grasp=0.0):
        """Grasp-point z for a release with the object's bottom `gap` above the lid top (object centre at the grasp
        point + dz_grasp below it)."""
        return self.z_top + gap + self.scene.object_height() / 2 + dz_grasp

    def on_lid(self, xy, margin=0.0):
        """xy (pelvis) over the lid plate (the lid bin's outer footprint + margin)."""
        b = self.scene.cfg["bins"]
        c = self.scene.mat_to_pelvis(np.r_[b["centers"][self.bin], 0.0])[:2]
        h = np.asarray(b["outer_size"], float)[::-1] / 2 + margin   # mat (x, y) half sizes -> pelvis (x, y)
        return bool(np.all(np.abs(np.asarray(xy, float)[:2] - c) <= h))


def reach_report(planner, lid, ic=None, arms=("left", "right")):
    """Per hole and arm: can the arm hold its grasp point top-down (approach tilt <= insert.max_tilt_deg) above the
    hole centre at carry height and at the release height (object bottom insert.gap above the lid)? IK of the
    planner (its config, joint margins), from home. {shape: {side: (ok, pos_err m, tilt deg)}}."""
    ic = ic or {}
    kin, home = planner.kin, planner.home
    g = planner.cfg["grasp"]
    sc = planner.scene
    z_rel = lid.release_z(float(ic.get("gap", 0.003)))
    z_carry = sc.counter_z + max(g["carry_height"], float(sc.cfg["bins"]["wall_height"]) + planner.height + 0.03)
    tilt_max = float(ic.get("max_tilt_deg", 10.0))
    tol = float(planner.cfg["ik"]["pos_tol"])
    out = {}
    for shape, h in lid.holes.items():
        out[shape] = {}
        for side in arms:
            q1, ok1, e1, t1 = kin.solve(side, np.r_[h["xy"], z_carry], home)
            q2, ok2, e2, t2 = kin.solve(side, np.r_[h["xy"], z_rel], q1)
            e, t = max(e1, e2), float(np.degrees(max(t1, t2)))
            out[shape][side] = (bool(e <= tol and t <= tilt_max), float(e), t)
    return out


def decide(report, ic=None):
    """{shape: (attempt?, why)}: attempted only if insert.attempt.<shape> is on, some arm reaches the hole
    (reach_report) and a delivery is implemented (none is: insert.deliver)."""
    ic = ic or {}
    want = ic.get("attempt") or {}
    out = {}
    for shape, arms in report.items():
        best = min(arms.items(), key=lambda kv: kv[1][1])
        side, (ok, e, t) = best
        if not want.get(shape, False):
            out[shape] = (False, "insert.attempt off")
        elif not ok:
            out[shape] = (False, f"hole out of reach ({side} arm {1000 * e:.0f} mm short, tilt {t:.0f} deg)")
        else:
            out[shape] = (False, f"{side} arm reaches the hole but no delivery is implemented")
    return out


def look_note(planner, cfg):
    """One log line for the first look of an insert episode: reach_report + decide (cached on the planner: the IK
    takes ~1-3 s, in the planning thread). '' if insert.enabled is off."""
    ic = cfg.get("insert") or {}
    if not ic.get("enabled"):
        return ""
    if getattr(planner, "_insert", None) is None:
        lid = LidModel(planner.scene)
        rep = reach_report(planner, lid, ic)
        planner._insert = (lid, rep, decide(rep, ic))
    lid, rep, dec = planner._insert
    return f"rung 3 lid on the {lid.bin} bin: {report_text(rep, dec)}"


def report_text(report, decision):
    return "; ".join(f"{s}: " + ", ".join(f"{side} {'ok' if ok else f'{1000 * e:.0f} mm off'}" for side, (ok, e, _)
                                          in arms.items()) + f" -> {'ATTEMPT' if decision[s][0] else 'no'} "
                     f"({decision[s][1]})" for s, arms in report.items())
