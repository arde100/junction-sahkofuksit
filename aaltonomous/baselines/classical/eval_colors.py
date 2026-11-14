"""Colour decisions on frames with ground truth (capture_frames.py .npz): per instruction type, objects sorted right /
left on the counter / wrong as the policy's look would decide them (detect + colors.calibrate_look, a fresh episode
per frame), the hue estimate per true colour, and what the calibration decided per frame.

    python -m aaltonomous.baselines.classical.eval_colors /data/aaro/frames/var_sim_pyo --profile sim \
        [--words red=orange,blue=purple] [--set perception.color.calib.link=5] [--limit N] [--per-frame] [--two-look]
--words: what the instruction calls each internal (ground-truth) colour (default red=orange, yellow=yellow,
blue=purple; "red=red,blue=blue" for the stock sim). --set: config overrides (dotted key=json value).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

INSTR = {"all": "Put the {o} objects in the left bin, the {y} objects in the center bin and the {p} objects in the "
                "right bin.",
         "o": "Put the {o} objects in the left bin.",
         "o+": "Put the {o} objects in the left bin and everything else in the right bin.",
         "p": "Put the {p} objects in the right bin.",
         "p+": "Put the {p} objects in the right bin and everything else in the middle bin.",
         "op": "Put the {o} objects in the left bin and the {p} objects in the right bin.",
         "op+": "{O} goes left, {p} goes right, everything else goes in the middle."}
MATCH_R = 0.03


def run(frames, profile, words, overrides=None, limit=None, log=print, per_frame=False, two_look=False):
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    from .colors import Calibrator, calibrate_look
    from .config import load_config
    from .instruction import mapping_colors, object_bin, parse_instruction
    from .perception import Perception
    cfg = load_config(profile=profile, overrides=overrides)
    p = Perception(cfg, Scene.load(), Calibration.load())
    c = cfg["perception"]["color"]
    w = {"o": words["red"], "y": words["yellow"], "p": words["blue"], "O": words["red"].capitalize()}
    files = sorted(Path(frames).glob("*.npz"), key=lambda f: (len(f.stem), f.stem))[:limit]
    cnt = {k: {"right": 0, "left": 0, "wrong": 0} for k in INSTR}
    missed, hues, calib, second = [], {}, [], {}
    for f in files:
        z = np.load(f)
        objs = json.loads(str(z["objects"]))
        for key, tpl in INSTR.items():
            r = parse_instruction(tpl.format(**w))
            cal = Calibrator(mapping_colors(r.mapping), *p.world, c.get("match"), c.get("calib"), r.words)
            p.set_classes(cal.ec.refs, cal.ec.reject, cal.ec.alias)
            dets = p.detect(z["rgb"], z["depth"])
            on = (c.get("calib") or {}).get("enabled")
            _, line = calibrate_look(cal, p, dets) if on else (False, None)
            first = {}
            for o in objs:
                want = r.mapping.get(words[o["color"]], r.mapping.get("default"))
                d = min(dets, key=lambda d: np.linalg.norm(d.top[:2] - np.asarray(o["pos"])[:2]), default=None)
                hit = d is not None and np.linalg.norm(d.top[:2] - np.asarray(o["pos"])[:2]) <= MATCH_R
                got = object_bin(r.mapping, d.color, d.shape, cal.ec.names) if hit else None
                cnt[key]["right" if got == want else "left" if got is None else "wrong"] += 1
                if hit:
                    first[id(o)] = (o, d, got, want)
            # second look (two_look): only some objects left (one colour, or one object), same detections re-voted
            # with the classes the first look calibrated; the calibrator must keep its decisions
            for keep in ((("red",), ("yellow",), ("blue",), "one") if two_look and on else ()):
                rest = [v for v in first.values() if (v[0]["color"] in keep if keep != "one" else v is
                                                       next(iter(first.values())))]
                if not rest:
                    continue
                cal2 = copy.deepcopy(cal)
                p.set_classes(cal2.ec.refs, cal2.ec.reject, cal2.ec.alias)
                d2 = [copy.deepcopy(d) for _, d, _, _ in rest]
                p.relabel(d2)
                calibrate_look(cal2, p, d2)
                for (o, _, got1, want), d in zip(rest, d2):
                    got = object_bin(r.mapping, d.color, d.shape, cal2.ec.names)
                    s2 = second.setdefault(key, {"right": 0, "left": 0, "wrong": 0, "lost": 0, "gained": 0})
                    s2["right" if got == want else "left" if got is None else "wrong"] += 1
                    s2["lost"] += got1 == want and got != want
                    s2["gained"] += got1 != want and got == want
                if key == "all":
                    if not hit:
                        missed.append(f"{f.stem}: {o.get('name', '')} {o['color']} {o.get('shape', '')} at "
                                      f"({o['pos'][0]:+.3f},{o['pos'][1]:+.3f})")
                    elif d.hue is not None:
                        hues.setdefault(o["color"], []).append(d.hue)
            if key in ("all", "o+") and (line or per_frame):
                calib.append(f"{f.stem} {key}: {line or '-'}")
    log(f"{frames}  profile {profile}  words {words}  {len(files)} frames"
        + (f"  overrides {overrides}" if overrides else ""))
    for k, v in cnt.items():
        s = second.get(k)
        log(f"  {k:<4} right {v['right']:3d}  left {v['left']:3d}  wrong {v['wrong']:3d}" + (
            f"   | second look: right {s['right']:3d}  left {s['left']:3d}  wrong {s['wrong']:3d}  "
            f"lost {s['lost']:3d}  gained {s['gained']:3d}" if s else ""))
    for col, hs in sorted(hues.items()):
        log(f"  hue {col:<7} ({words[col]}): median {np.median(hs):5.1f}  min {min(hs):5.1f}  max {max(hs):5.1f}  "
            f"n {len(hs)}")
    log(f"  missed (no detection within {100 * MATCH_R:.0f} cm, instruction 'all'): {len(missed)}"
        + "".join(f"\n    {m}" for m in missed))
    if per_frame:
        log("  calibration per frame:" + "".join(f"\n    {x}" for x in calib))
    return cnt, hues, missed, calib, second


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--profile", default="sim")
    ap.add_argument("--words", default="red=orange,yellow=yellow,blue=purple")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=json value (config override)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-frame", action="store_true", help="print the calibration line of every frame")
    ap.add_argument("--two-look", action="store_true", help="also a second look per frame with one colour / one "
                    "object left (the first look's detections, re-voted): must keep the first look's decisions")
    a = ap.parse_args(argv)
    words = dict(x.split("=") for x in a.words.split(","))
    ov = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
    run(a.frames, a.profile, words, ov or None, a.limit, per_frame=a.per_frame, two_look=a.two_look)


if __name__ == "__main__":
    main()
