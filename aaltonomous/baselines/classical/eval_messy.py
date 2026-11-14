"""ANALYSIS TOOL (reads ground truth; never import this from a policy): perception on objects as they really lie
(prisms / cylinders lying on a side, touching clusters) vs truth, on recorded sim frames (capture_frames.py: objects
carry `lying`, `ridge_yaw_pelvis_rad`) or on rendered mock frames (messy.py layouts, mock_world.Renderer).

    python -m aaltonomous.baselines.classical.eval_messy /data/aaro/frames/var_sim_messy [--profile sim]
    python -m aaltonomous.baselines.classical.eval_messy --mock 40 [--size 0.048] [--profile none]

Per object group (upright / lying, alone / in a cluster = another object within 1.4 sizes): found, missed, merged
(one detection matched while another true object lies within 1.2 sizes of it, unmatched), centre error (top xy vs
the true centre xy), shape right / unknown / WRONG (a lying prism called anything but a lying triangle, a cube called
a triangle, ...), lying flag, ridge yaw error (mod 180). Extra detections, run time.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .eval_perception import _match
from .perception import Perception

MESSY = {"lying_prism_prob": 0.5, "lying_cylinder_prob": 0.0, "cluster_prob": 1.0, "cluster_spacing": [1.05, 1.25],
         "cluster_n": [2, 4]}


def mock_frames(n, size=None, seed=0, messy=None, noise=1.5, holes=0.02, colors=None):
    """n rendered frames of messy layouts (sim/aalto_task/messy.py): [(name, rgb, depth, objects)] with objects
    {name, color, shape, pos (centre, pelvis), yaw_pelvis_rad, lying, ridge_yaw_pelvis_rad}."""
    import copy
    from aaltonomous.interface.tasks import sample_layout
    from aaltonomous.sim.aalto_task.messy import messy_layout, spawn_height
    from .mock_world import Renderer
    sc = Scene.load()
    if size:
        sc = Scene(copy.deepcopy(sc.cfg))
        sc.cfg["objects"]["size"] = float(size)
    r = Renderer(sc, Calibration.load(), colors=colors)
    out = []
    for k in range(n):
        rng = np.random.default_rng([seed, k])
        lay = messy_layout({"layout": messy or MESSY}, sc, sample_layout(rng, **sc.layout_kwargs()))
        objs = []
        for i, c in enumerate(lay):
            pos = sc.mat_to_pelvis([c["x"], c["y"], spawn_height(sc, c)])
            objs.append({"name": f"{c['color']}_{c.get('shape', 'cube')}_{i}", "color": c["color"],
                         "shape": c.get("shape", "cube"), "pos": pos.tolist(),
                         "yaw_pelvis_rad": float(c.get("yaw", 0.0)),      # (the mock world's convention)
                         "lying": bool(c.get("lying")), "ridge_yaw_pelvis_rad": c.get("ridge_yaw"),
                         "cluster": bool(c.get("cluster"))})
        ren = [dict(color=o["color"], shape=o["shape"], yaw=o["yaw_pelvis_rad"], size=sc.size,
                    height=sc.object_height(o["shape"]), pos=np.asarray(o["pos"]), lying=o["lying"],
                    ridge_yaw=o["ridge_yaw_pelvis_rad"] or 0.0) for o in objs]
        rgb, depth = r.render(ren, rng, noise, holes)
        out.append((f"mock{k}", rgb, depth, objs))
    return out, sc


def recorded(d):
    for f in sorted(Path(d).glob("*.npz"), key=lambda p: (len(p.stem), p.stem)):
        z = np.load(f, allow_pickle=False)
        yield f.stem, z["rgb"], z["depth"], json.loads(str(z["objects"]))


def evaluate(frames, sc, profile="none", overrides=None, log=print):
    cfg = load_config(profile=profile, overrides=overrides)
    p = Perception(cfg, sc, Calibration.load())
    size = sc.size
    rows, extra, times = [], 0, []
    for name, rgb, depth, objs in frames:
        p.reset_boxes()
        t0 = time.perf_counter()
        dets = p.detect(rgb, depth)
        times.append(1000 * (time.perf_counter() - t0))
        objs = [dict(o, pos=np.asarray(o["pos"], float)) for o in objs]
        pairs, ex, missed = _match(dets, objs, 0.03)
        extra += len(ex)
        m = {j: i for i, j in pairs}
        P = np.array([o["pos"][:2] for o in objs])
        for j, o in enumerate(objs):
            dd = np.linalg.norm(P - P[j], axis=1)
            dd[j] = np.inf
            crowd = bool(dd.min() < 1.4 * size) if len(objs) > 1 else False
            d = dets[m[j]] if j in m else None
            merged = d is None and any(np.linalg.norm(dets[i].top[:2] - o["pos"][:2]) < 1.2 * size for i in m.values())
            r = {"frame": name, "shape": o["shape"], "lying": bool(o.get("lying")), "crowd": crowd,
                 "found": d is not None, "merged": merged}
            if d is not None:
                r["err_mm"] = 1000 * float(np.linalg.norm(d.top[:2] - o["pos"][:2]))
                r["det_shape"], r["det_lying"] = d.shape, bool(getattr(d, "lying", False))
                ry = getattr(d, "ridge_yaw", None)
                if r["lying"] and ry is not None and o.get("ridge_yaw_pelvis_rad") is not None:
                    e = np.degrees(ry - o["ridge_yaw_pelvis_rad"])
                    r["ridge_err"] = float(abs((e + 90) % 180 - 90))
                ok_shape = d.shape == o["shape"] and r["det_lying"] == r["lying"]
                r["shape_out"] = "unknown" if d.shape is None else "right" if ok_shape else "WRONG"
                r["fit"] = {k: v for k, v in (d.fit or {}).items() if k in ("profile", "rms_mm", "best", "fill", "free",
                                                                            "bad")}
            rows.append(r)
    report(rows, extra, times, log)
    return rows


def report(rows, extra, times, log=print):
    log(f"{len(rows)} objects; extra detections {extra}; detect ms mean {np.mean(times):.0f} p95 "
        f"{np.percentile(times, 95):.0f}")
    groups = [("upright alone", lambda r: not r["lying"] and not r["crowd"]),
              ("upright in cluster", lambda r: not r["lying"] and r["crowd"]),
              ("lying alone", lambda r: r["lying"] and not r["crowd"]),
              ("lying in cluster", lambda r: r["lying"] and r["crowd"])]
    for g, f in groups:
        R = [r for r in rows if f(r)]
        if not R:
            continue
        F = [r for r in R if r["found"]]
        e = np.array([r["err_mm"] for r in F]) if F else np.zeros(0)
        so = {k: sum(r.get("shape_out") == k for r in F) for k in ("right", "unknown", "WRONG")}
        ry = [r["ridge_err"] for r in F if "ridge_err" in r]
        log(f"  {g:<20} n {len(R):3d}  found {len(F):3d}  missed {len(R) - len(F):2d} (merged "
            f"{sum(r['merged'] for r in R):2d})  centre mm p50 {np.median(e) if len(e) else 0:4.1f} p95 "
            f"{np.percentile(e, 95) if len(e) else 0:5.1f} max {e.max() if len(e) else 0:5.1f}  shape {so}"
            + (f"  ridge deg p50 {np.median(ry):.1f} max {max(ry):.1f}" if ry else ""))
    for r in [r for r in rows if r.get("shape_out") == "unknown"][:8]:
        log(f"    unknown {r['frame']} {r['shape']}{' lying' if r['lying'] else ''}"
            f"{' (cluster)' if r['crowd'] else ''}: {r.get('fit')}")
    W = [r for r in rows if r.get("shape_out") == "WRONG"]
    for r in W[:12]:
        log(f"    WRONG {r['frame']} {r['shape']}{' lying' if r['lying'] else ''} -> {r['det_shape']}"
            f"{' lying' if r['det_lying'] else ''}{' (cluster)' if r['crowd'] else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="*")
    ap.add_argument("--mock", type=int, default=0)
    ap.add_argument("--size", type=float, default=None)
    ap.add_argument("--lying-cyl", type=float, default=0.0, help="mock: probability of a cylinder lying")
    ap.add_argument("--noise", type=float, default=1.5, help="mock: depth noise sd (mm)")
    ap.add_argument("--profile", default="none")
    ap.add_argument("--set", default=None, help="JSON config overrides")
    ap.add_argument("--colors", default=None,
                    help="mock: real (mock_world.REAL_COLORS; default for profile none) | scene")
    a = ap.parse_args(argv)
    ov = json.loads(a.set) if a.set else None
    if a.mock:
        from .mock_world import REAL_COLORS
        real = (a.colors or ("real" if a.profile == "none" else "scene")) == "real"
        fr, sc = mock_frames(a.mock, a.size, colors=REAL_COLORS if real else None,
                             messy=dict(MESSY, lying_cylinder_prob=a.lying_cyl), noise=a.noise)
        log = print
        log(f"== mock {a.mock} frames (size {sc.size}) profile {a.profile}")
        evaluate(fr, sc, a.profile, ov)
    for d in a.frames:
        print(f"== {d} profile {a.profile}")
        evaluate(recorded(d), Scene.load(), a.profile, ov)


if __name__ == "__main__":
    main()
