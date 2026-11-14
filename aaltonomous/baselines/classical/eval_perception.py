"""ANALYSIS TOOL (reads ground truth; never import this from a policy): head-camera perception vs the recorded sim
frames of aaltonomous/sim/capture_frames.py (<seed>.npz with rgb, depth, objects = ground truth).

    python -m aaltonomous.baselines.classical.eval_perception /data/aaro/frames/nominal [--corrupt train|aaro_stress]
        [--color-jitter train|aaro_stress] [--cam-error 0.01,1] [--variants old,geom] [--limit N] [--worst 10]
        [--dump out.json]

Runs the classical Perception once per variant per frame (config overrides on top of config.yaml):
    old    = colour first, no self-levelling (the code before geometry first)
    colour = colour first + self-levelling          geom0 = geometry first, no levelling
    geom   = geometry first + self-levelling        geomreg = geom + bin registration (= profile sim as is)
Default variants: old,geom,geomreg. Detections are matched to the true objects by nearest
xy whatever their colour (Hungarian, <= 4 cm; 6 cm with --cam-error). Reports missed / extra detections, the colour
confusion matrix (truth x detected incl. unknown / missed), the top centre error (perceived - true, pelvis frame;
truth = object centre + half its height) split into x / y / z, overall and by position, the levelling (correction,
inliers, fallbacks; with --cam-error the pose error left after it: tilt / yaw / height / in-plane shift), the shape
confusion matrix, yaw errors (cube modulo 90 deg, triangle modulo 120 deg = apex directions), run time, worst cases.
--corrupt: camera noise of a DR config (aaltonomous/sim/dr/<name>.yaml `camera:`: rgb gain + noise, depth noise,
  holes; NOT the pose jitter) with dr.corrupt, seeded per frame.
--color-jitter: the colour part of a DR config applied to the recorded image, seeded per frame (dr.sample): every
  true object's pixels (3D point within 45 mm of its centre, 3 mm above the counter up to its top + 15 mm) get the
  object's hue / saturation / value change (dr.jitter_color's dh, s, v, in HSV); then in linear light (sRGB gamma
  2.2): bin pixels x bin_rgb, counter pixels x mat_scale + mat_bias, the whole image x a global per-channel light gain
  0.6 * key_scale * key_rgb + 0.4 * dome_scale * dome_rgb (a guess at how the two lights mix; no moving shadows or
  highlights), clipped (washed-out tops); then camera rgb gain + noise (dr.corrupt, no depth noise).
--cam-error POS,DEG: the head T_pelvis_optical perception uses is perturbed per frame (seeded): camera centre + U(+-POS)
  per axis, rotation U(+-DEG) per axis about the camera centre (as the sim's camera pose DR does).
Frames recorded with DR (e.g. /data/aaro/frames/train) already carry all of this: run them without these options.
Config overrides as for the policy: AALTO_SET='{"perception.shape.band": 0.004}'.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config, _set
from .perception import SHAPES, Perception

MATCH_R = 0.04
NEAR_X = 0.295                 # pelvis x splitting the object band (0.26-0.33) into near / far row
SIDE_Y = 0.06                  # |pelvis y| beyond this = left / right, else centre
PERIOD = {"cube": 90.0, "triangle": 120.0}
VARIANTS = {"old": {"perception.geometry_first": False, "perception.level.enabled": False,
                    "perception.register.enabled": False, "perception.xy_offset": [0.0, 0.0]},
            "colour": {"perception.geometry_first": False, "perception.level.enabled": True,
                       "perception.register.enabled": False, "perception.xy_offset": [0.0, 0.0]},
            "geom0": {"perception.geometry_first": True, "perception.level.enabled": False,
                      "perception.register.enabled": False, "perception.xy_offset": [0.0, 0.0]},
            "geom": {"perception.geometry_first": True, "perception.level.enabled": True,
                     "perception.register.enabled": False, "perception.xy_offset": [0.0, 0.0]},
            "geomreg": {"perception.geometry_first": True, "perception.level.enabled": True,
                        "perception.register.enabled": True}}


def _wrap(a, per):
    return (a + per / 2) % per - per / 2


def _match(dets, objs, r=MATCH_R):
    """Nearest xy, colour-agnostic (a wrong colour shows in the confusion matrix, not as missed + extra)."""
    from scipy.optimize import linear_sum_assignment
    pairs = []
    if dets and objs:
        C = np.array([[np.linalg.norm(d.top[:2] - np.asarray(o["pos"][:2])) for o in objs] for d in dets])
        pairs = [(i, j) for i, j in zip(*linear_sum_assignment(C)) if C[i, j] <= r]
    ud, uo = {i for i, _ in pairs}, {j for _, j in pairs}
    return pairs, [i for i in range(len(dets)) if i not in ud], [j for j in range(len(objs)) if j not in uo]


def _where(p):
    row = "near" if p[0] < NEAR_X else "far"
    side = "left" if p[1] > SIDE_Y else "right" if p[1] < -SIDE_Y else "centre"
    return row, side


def _rot(rv):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rv).as_matrix()


def cam_error(T, pos_m, deg, seed):
    """Perturbed camera pose: centre + U(+-pos_m) per axis, rotation U(+-deg) per axis about the camera centre."""
    rng = np.random.default_rng([int(seed) if str(seed).isdigit() else 0, 77])
    T2 = T.copy()
    T2[:3, :3] = _rot(np.radians(rng.uniform(-deg, deg, 3))) @ T[:3, :3]
    T2[:3, 3] = T[:3, 3] + rng.uniform(-pos_m, pos_m, 3)
    return T2


def pose_error(T, T_true):
    """Pose error of T vs T_true: tilt of the vertical (deg), yaw (deg), camera centre shift (mm, x y z)."""
    R = T[:3, :3] @ T_true[:3, :3].T
    tilt = float(np.degrees(np.arctan2(np.hypot(R[0, 2], R[1, 2]), R[2, 2])))   # (arccos: 0.08 deg of rounding)
    yaw = float(np.degrees(np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])))
    return {"tilt": tilt, "yaw": yaw, "dt": (1000 * (T[:3, 3] - T_true[:3, 3])).tolist()}


def color_jitter(rgb, depth, objs, dr_cfg, seed, sc, cal):
    """The DR's colour knobs applied to a recorded frame (see the module doc)."""
    import cv2
    from aaltonomous.sim.aalto_task import dr
    prm = dr.sample(dr_cfg, seed, [o["name"] for o in objs], sc.bin_names)
    f = rgb.astype(np.float32) / 255.0
    d = depth.astype(float)
    vv, uu = np.nonzero((d > 150) & (d < 2000))
    P = cal.pixel_to_pelvis("head", uu, vv, d[vv, uu])
    h = P[:, 2] - sc.counter_z - 0.005          # (the sim's head camera sees the counter ~5 mm high, see levelling)
    for o in objs:                              # neighbours are >= 8 cm apart: a 45 mm disc is one object (+ finger)
        j = prm["objects"][o["name"]]
        sel = (np.linalg.norm(P[:, :2] - np.asarray(o["pos"][:2]), axis=1) <= 0.045) & (h >= 0.003) & \
              (h <= sc.object_height(o["shape"]) + 0.015)
        if not sel.any():
            continue
        hsv = cv2.cvtColor(f[vv[sel], uu[sel]][None], cv2.COLOR_RGB2HSV)   # float: H 0-360, S, V 0-1
        hsv[..., 0] = (hsv[..., 0] + 360.0 * j["dh"]) % 360.0
        hsv[..., 1] = np.clip(hsv[..., 1] * j["s"], 0, 1)
        hsv[..., 2] = hsv[..., 2] * j["v"] ** (1 / 2.2)                  # albedo scale, in sRGB
        f[vv[sel], uu[sel]] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]
    lin = np.clip(f, 0, None) ** 2.2                                    # reflectance / light: linear light
    pm = sc.pelvis_to_mat(P)
    half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2
    for name, ctr in sc.cfg["bins"]["centers"].items():
        sel = np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1) & \
              (h <= sc.cfg["bins"]["wall_height"] + 0.01)
        lin[vv[sel], uu[sel]] *= np.asarray(prm["textures"]["bin_rgb"][name], np.float32)
    T = prm["textures"]
    sel = np.abs(h) < 0.004
    lin[vv[sel], uu[sel]] = lin[vv[sel], uu[sel]] * np.asarray(T["mat_scale"], np.float32) + 0.5 * T["mat_bias"]
    L = prm["lighting"]
    gain = 0.6 * L["key_intensity_scale"] * np.asarray(L["key_rgb"]) + \
        0.4 * L["dome_intensity_scale"] * np.asarray(L["dome_rgb"])
    out = (np.clip(lin * gain.astype(np.float32), 0, 1) ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)
    cam = dict(prm["camera"], depth_noise_mm=0.0, depth_noise_rel=0.0, depth_hole_frac=0.0)
    out, _ = dr.corrupt(out, depth, cam, np.random.default_rng(cam["stream_seed"]))
    return out, {"gain": np.round(gain, 3).tolist()}


def run(frames, corrupt=None, limit=None, variants=("old", "geom"), cam_err=None, cjit=None, log=print):
    sc, cal = Scene.load(), Calibration.load()
    cfg = load_config()
    pers = {}
    for v in variants:
        c = copy.deepcopy(cfg)
        for k, val in VARIANTS[v].items():
            _set(c, k, val)
        pers[v] = Perception(c, sc, cal)
    T_cal = cal._T("head").copy()
    from aaltonomous.sim.aalto_task import dr
    dr_cfg = dr.load(corrupt) if corrupt and corrupt != "none" else None
    cj_cfg = dr.load(cjit) if cjit and cjit != "none" else None
    match_r = 0.06 if cam_err else MATCH_R
    files = sorted(Path(frames).glob("*.npz"), key=lambda p: (len(p.stem), p.stem))[:limit]
    objs_out, frames_out = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        rgb, depth = z["rgb"], z["depth"]
        seed = int(f.stem) if f.stem.isdigit() else f.stem
        objs = json.loads(str(z["objects"]))
        fr = {"seed": seed, "n_obj": len(objs)}
        if cj_cfg is not None:
            rgb, fr["cjit"] = color_jitter(rgb, depth, objs, cj_cfg, seed if isinstance(seed, int) else 0, sc, cal)
        if dr_cfg is not None:
            cam = dr.sample(dr_cfg, seed if isinstance(seed, int) else 0)["camera"]
            rgb, depth = dr.corrupt(rgb, depth, cam, np.random.default_rng(cam["stream_seed"]))
        T_used = cam_error(T_cal, cam_err[0], cam_err[1], seed) if cam_err else T_cal
        if cam_err:
            fr["cam_err"] = pose_error(T_used, T_cal)
        res = {}
        for tag, per in pers.items():
            per.T0 = T_used
            t0 = time.perf_counter()
            dets = per.detect(rgb, depth)
            fr[f"t_{tag}"] = time.perf_counter() - t0
            fr[f"level_{tag}"] = dict(per.level, pose=pose_error(per.T, T_cal), path=per.path, reg=dict(per.reg))
            pairs, extra, missed = _match(dets, objs, match_r)
            res[tag] = (dets, dict((j, i) for i, j in pairs))
            fr[f"extra_{tag}"] = [dets[i].color for i in extra]
            fr[f"missed_{tag}"] = [objs[j]["color"] for j in missed]
            fr[f"extra_xy_{tag}"] = [np.round(dets[i].top, 4).tolist() for i in extra]
        frames_out.append(fr)
        for j, o in enumerate(objs):
            pos = np.asarray(o["pos"], float)
            truth = pos + [0.0, 0.0, sc.object_height(o["shape"]) / 2]
            rec = {"seed": seed, "name": o["name"], "color": o["color"], "shape": o["shape"],
                   "yaw_true": float(o["yaw_pelvis_rad"]), "true": truth.tolist(), "where": _where(pos)}
            for tag, (dets, m) in res.items():
                d = dets[m[j]] if j in m else None
                rec[tag] = None if d is None else {
                    "top": d.top.tolist(), "method": d.method, "yaw": d.yaw, "shape": d.shape, "color": d.color,
                    "conf": d.shape_conf, "fit": None if d.fit is None else
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.fit.items()
                     if not k.startswith("hull")}}
            objs_out.append(rec)
    return objs_out, frames_out


# ---------------------------------------------------------------------- report
def _pos_stats(errs):
    e = np.asarray(errs) * 1000
    if not len(e):
        return "n=0"
    n = np.linalg.norm(e[:, :2], axis=1)
    return (f"n={len(e):3d} bias=({e[:, 0].mean():+5.1f},{e[:, 1].mean():+5.1f},{e[:, 2].mean():+5.1f})  "
            f"std=({e[:, 0].std():4.1f},{e[:, 1].std():4.1f},{e[:, 2].std():4.1f})  |xy| p50 {np.median(n):4.1f} "
            f"p95 {np.percentile(n, 95):4.1f} max {n.max():4.1f} mm")


def _yaw_stats(err_deg):
    e = np.abs(np.asarray(err_deg))
    if not len(e):
        return "n=0"
    return (f"n={len(e):3d} median {np.median(e):4.1f} p90 {np.percentile(e, 90):5.1f} max {e.max():5.1f} deg, "
            f"<=10 deg {100 * np.mean(e <= 10):5.1f} %")


def _ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():+6.2f} sd {x.std():5.2f} |max| {np.abs(x).max():5.2f}" if len(x) else "n=0"


def report(objs, frames, log=print):
    tags = [k[2:] for k in frames[0] if k.startswith("t_")]
    log(f"{len(frames)} frames, {len(objs)} objects; variants {', '.join(tags)}")
    log("\n== detection (matched by xy whatever the colour; missed per true colour, extra per detected colour)")
    cols = sorted({o["color"] for o in objs})
    for tag in tags:
        miss = {c: sum(fr[f"missed_{tag}"].count(c) for fr in frames) for c in cols}
        exc = sorted({c for fr in frames for c in fr[f"extra_{tag}"]})
        ext = {c: sum(fr[f"extra_{tag}"].count(c) for fr in frames) for c in exc}
        wrong = sum(1 for r in objs if r[tag] and r[tag]["color"] != r["color"])
        log(f"  {tag:<6} missed {sum(miss.values()):3d} {miss}   extra {sum(ext.values()):3d} {ext}   "
            f"wrong colour {wrong}")
    if len(tags) > 1:
        a, b = tags[0], tags[-1]
        diff = [fr["seed"] for fr in frames if (sorted(fr[f"missed_{a}"]), sorted(fr[f"extra_{a}"])) !=
                (sorted(fr[f"missed_{b}"]), sorted(fr[f"extra_{b}"]))]
        log(f"  frames whose detection list differs {a} vs {b}: {diff if diff else 'none'}")

    log("\n== colour (rows = truth; columns = detected colour of the matched detection)")
    cc = ["red", "yellow", "blue", "unknown"]
    for tag in tags:
        log(f"  -- {tag}" + " " * 7 + "".join(f"{c:>9}" for c in (*cc, "other", "missed")) + "   correct")
        for c in cols:
            rs = [r for r in objs if r["color"] == c]
            cnt = [sum(1 for r in rs if r[tag] and r[tag]["color"] == k) for k in cc]
            cnt += [sum(1 for r in rs if r[tag] and r[tag]["color"] not in cc), sum(1 for r in rs if not r[tag])]
            ok = sum(1 for r in rs if r[tag] and r[tag]["color"] == c)
            log(f"     {c:<10} " + "".join(f"{k:>9d}" for k in cnt) + f"   {100 * ok / max(len(rs), 1):5.1f} %")

    log("\n== self-levelling + bin registration (corrections; pose = pose used - calibration.yaml)")
    if "cam_err" in frames[0]:
        pe = [fr["cam_err"] for fr in frames]
        log(f"  applied error : tilt {_ms([p['tilt'] for p in pe])} deg, yaw {_ms([p['yaw'] for p in pe])} deg, "
            f"dz {_ms([p['dt'][2] for p in pe])} mm, |dxy| {_ms([np.hypot(*p['dt'][:2]) for p in pe])} mm")
    for tag in tags:
        L = [fr[f"level_{tag}"] for fr in frames]
        used = [x for x in L if x["used"]]
        whys = {}
        for x in L:
            if not x["used"]:
                whys[x["why"]] = whys.get(x["why"], 0) + 1
        paths = {}
        for x in L:
            paths[x["path"]] = paths.get(x["path"], 0) + 1
        log(f"  {tag:<6} used {len(used)}/{len(L)}" + (f" (else: {whys})" if whys else "") + f"; path {paths}")
        if used:
            log(f"         correction roll {_ms([x['roll_deg'] for x in used])} deg, pitch "
                f"{_ms([x['pitch_deg'] for x in used])} deg, dz {_ms([x['dz_mm'] for x in used])} mm; inliers "
                f"{int(np.median([x['n_inliers'] for x in used]))} ({np.median([x['inlier_frac'] for x in used]):.2f})"
                f", rms {np.median([x['rms_mm'] for x in used]):.2f} mm")
        R = [x["reg"] for x in L if x["reg"]["why"] != "off"]
        if R:
            ru = [x for x in R if x["used"]]
            rw = {}
            for x in R:
                if not x["used"]:
                    rw[x["why"]] = rw.get(x["why"], 0) + 1
            log(f"         registration used {len(ru)}/{len(R)}" + (f" (else: {rw})" if rw else ""))
            if ru:
                log(f"         registration dx {_ms([x['dx_mm'] for x in ru])} mm, dy "
                    f"{_ms([x['dy_mm'] for x in ru])} mm, dyaw {_ms([x['dyaw_deg'] for x in ru])} deg; inliers "
                    f"{int(np.median([x['n_inliers'] for x in ru]))} of {int(np.median([x['n_points'] for x in ru]))}"
                    f", rms {np.median([x['rms_mm'] for x in ru]):.2f} mm")
        p = [x["pose"] for x in L]     # (vs calibration.yaml; the sim's own camera differs from it by a constant)
        log(f"         pose left: tilt {_ms([q['tilt'] for q in p])} deg, yaw {_ms([q['yaw'] for q in p])} deg, "
            f"dx {_ms([q['dt'][0] for q in p])} mm, dy {_ms([q['dt'][1] for q in p])} mm, "
            f"dz {_ms([q['dt'][2] for q in p])} mm, |dxy| {_ms([np.hypot(*q['dt'][:2]) for q in p])} mm")

    log("\n== top centre error, perceived - true (mm, pelvis frame; bias / std = x, y, z)")
    last = tags[-1]
    variants = [(t, (lambda t: lambda r: r[t] and r[t]["top"])(t)) for t in tags]
    for t in tags:                     # shape accepted: top = fitted centre (+ perception.xy_offset)
        variants.append((f"fit xy ({t})", (lambda t: lambda r: r[t] and r[t]["fit"] and r[t]["shape"] and
                                           r[t]["top"])(t)))
    groups = [("all", lambda r: True)] + [(f"{s}", (lambda s: lambda r: r["shape"] == s)(s)) for s in SHAPES]
    groups += [(f"{row} {side}", (lambda a, b: lambda r: tuple(r["where"]) == (a, b))(row, side))
               for row in ("near", "far") for side in ("left", "centre", "right")]
    for vname, get in variants:
        log(f"  -- {vname}" + (" (objects with an accepted shape only)" if vname.startswith("fit") else ""))
        for gname, g in groups:
            errs = [np.subtract(get(r), r["true"]) for r in objs if g(r) and get(r)]
            log(f"    {gname:<13} {_pos_stats(errs)}")

    log("\n== shape (rows = truth; columns = detected)")
    for tag in tags:
        log(f"  -- {tag}    " + "".join(f"{c:>10}" for c in (*SHAPES, "None", "missed")))
        for s in SHAPES:
            rs = [r for r in objs if r["shape"] == s]
            cnt = [sum(1 for r in rs if r[tag] and r[tag]["shape"] == c) for c in SHAPES]
            cnt += [sum(1 for r in rs if r[tag] and r[tag]["shape"] is None), sum(1 for r in rs if not r[tag])]
            log(f"    {s:<10} " + "".join(f"{c:>10d}" for c in cnt))
        ok = [r for r in objs if r[tag] and (r[tag]["fit"] or {}).get("rms_mm")]   # a fit was made
        conf_ok = [r[tag]["conf"] for r in ok if r[tag]["fit"]["best"] == r["shape"]]
        conf_bad = [r[tag]["conf"] for r in ok if r[tag]["fit"]["best"] != r["shape"]]
        n_occ = sum(1 for r in objs if r[tag] and r[tag]["fit"] and not r[tag]["fit"].get("rms_mm"))
        n_sh = sum(1 for r in objs if r[tag] and r[tag]["shape"])
        log(f"    shape found {n_sh}/{len(objs)} ({100 * n_sh / max(len(objs), 1):.0f} %); no fit (outline occluded) "
            f"{n_occ}" + (f"; best-fit right {len(conf_ok)} (conf p5 {np.percentile(conf_ok, 5):.2f}), wrong "
                           f"{len(conf_bad)}" if conf_ok else ""))

    log("\n== yaw (objects whose shape was detected right)")
    for tag in tags:
        for s in ("cube", "triangle"):
            e = [_wrap(np.degrees(r[tag]["yaw"] - r["yaw_true"]), PERIOD[s]) for r in objs
                 if r["shape"] == s and r[tag] and r[tag]["shape"] == s and r[tag]["yaw"] is not None]
            log(f"  {tag:<6} {s:<9} modulo {PERIOD[s]:.0f}: {_yaw_stats(e)}")

    log("\n== run time per frame (detect(), this machine)")
    for tag in tags:
        t = np.array([fr[f"t_{tag}"] for fr in frames]) * 1000
        log(f"  {tag:<6} mean {t.mean():5.1f} ms  p50 {np.median(t):5.1f}  p95 {np.percentile(t, 95):5.1f}  "
            f"worst {t.max():5.1f} ms")

    log(f"\n== worst cases ({last})")
    bad, first = [], tags[0]
    for fr in frames:
        for c, xy in zip(fr[f"extra_{last}"], fr[f"extra_xy_{last}"]):
            bad.append((100, fr["seed"], f"{c} ?", f"extra detection at {xy}"))
    for r in objs:
        a, nm = r[last], r["name"]
        if a is None:
            bad.append((90, r["seed"], nm, "missed"))
            continue
        if a["color"] != r["color"]:
            bad.append((85, r["seed"], nm, f"colour {a['color']}"))
        f = a["fit"] or {}
        info = (f"rms {f.get('rms_mm')} conf {a['conf']:.2f} fill {f.get('fill')} free {f.get('free')} "
                f"shrink {f.get('shrink_mm')}")
        if a["shape"] is not None and a["shape"] != r["shape"]:
            bad.append((80, r["seed"], nm, f"shape {a['shape']} ({info})"))
        elif a["shape"] is None:
            bad.append((40 + 10 * a["conf"], r["seed"], nm, f"shape None ({info})"))
        if a["shape"] == r["shape"] and a["yaw"] is not None:
            e = _wrap(np.degrees(a["yaw"] - r["yaw_true"]), PERIOD[r["shape"]])
            if abs(e) > 5:
                bad.append((20 + abs(e), r["seed"], nm, f"yaw error {e:+.1f} deg ({info})"))
        ex = 1000 * np.linalg.norm(np.subtract(a["top"][:2], r["true"][:2]))
        if ex > 10:
            bad.append((40 + ex, r["seed"], nm, f"top xy error {ex:.1f} mm ({a['method']}, shape {a['shape']})"))
    for sev, seed, nm, what in sorted(bad, key=lambda b: -b[0])[:report.worst]:
        log(f"  seed {seed}  {nm:<22} {what}")


report.worst = 10


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", help="directory of <seed>.npz (capture_frames.py)")
    ap.add_argument("--corrupt", default=None, help="DR config whose camera noise to apply (train | aaro_stress | "
                                                    "a yaml path; none = off)")
    ap.add_argument("--color-jitter", default=None, help="DR config whose colour / lighting knobs to apply")
    ap.add_argument("--cam-error", default=None, help="POS_M,DEG: perturb the head extrinsics perception uses")
    ap.add_argument("--variants", default=None, help=f"comma list of {', '.join(VARIANTS)}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--dump", default=None, help="write per-object / per-frame records as JSON")
    a = ap.parse_args(argv)
    ce = tuple(float(x) for x in a.cam_error.split(",")) if a.cam_error else None
    var = a.variants.split(",") if a.variants else ["old", "geom", "geomreg"]
    objs, frames = run(a.frames, a.corrupt, a.limit, var, ce, a.color_jitter)
    print(f"frames {a.frames}  corrupt {a.corrupt or 'none'}  color-jitter {a.color_jitter or 'none'}  "
          f"cam-error {a.cam_error or 'none'}")
    report.worst = a.worst
    report(objs, frames)
    if a.dump:
        Path(a.dump).write_text(json.dumps({"objects": objs, "frames": frames}, default=float))


if __name__ == "__main__":
    main()
