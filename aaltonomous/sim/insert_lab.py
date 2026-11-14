"""ANALYSIS TOOL (privileged sim state; never import this from a policy): rung-3 lid measurements in the
Aalto-Insert sim (TASK=Aalto-Insert-G129-Dex3-Joint) through the scene service only (no robot commands).

    python -m aaltonomous.sim.insert_lab info
    python -m aaltonomous.sim.insert_lab drops [--offsets 0,3,5,7] [--dirs 8] [--yaws 0,6] [--gap 0.003] [--out F]

info:  the lid / hole geometry from scene.yaml (mat and pelvis frame) and the sim's info.
drops: drop tolerance. Per case one object of each shape is placed above its own hole, at rest, offset by (d mm in
       direction k * 360 / dirs deg in the PELVIS frame, yaw offset deg), its bottom `gap` above the lid top; after
       --settle sim seconds the scene service's insert_shapes outcome is recorded (inserted / forced / ...).
       Models a release from a perfectly still hand: no finger drag, no swing.
Rows (JSON lines) -> --out; a table per shape (inserted fraction per offset / yaw) on stdout.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aaltonomous.scene.scene import Scene

COLOR = {"cube": "red", "cylinder": "yellow", "triangle": "blue"}
BASE_YAW_MAT = {"cube": 0.0, "cylinder": 0.0, "triangle": np.pi}   # check_scene: the orientation that fits each hole


def wait_sim(c, secs, timeout=60.0):
    t0 = float(c.ping()["sim_t"])
    w0 = time.time()
    while float(c.ping()["sim_t"]) - t0 < secs and time.time() - w0 < timeout:
        time.sleep(0.1)


def info(sc, c):
    z_under, z_top = sc.lid_z
    print(f"lid on the {sc.lid_bin} bin, plate z (mat) {z_under:.3f} .. {z_top:.3f}, counter z (pelvis) "
          f"{sc.counter_z:+.3f}, clearance {sc.cfg['lid']['hole_clearance']}")
    for h in sc.lid_holes():
        m = h["center"]
        p = sc.mat_to_pelvis(np.r_[m, z_top])
        ext = sc.mat_to_pelvis(np.c_[h["poly"], np.zeros(len(h["poly"]))])[:, :2] - p[:2]
        print(f"  {h['shape']:<8} mat ({m[0]:.4f}, {m[1]:.4f})  pelvis ({p[0]:+.4f}, {p[1]:+.4f}, top z {p[2]:+.4f})  "
              f"r {np.hypot(*p[:2]):.3f}  pelvis extent x {ext[:, 0].min():+.4f}..{ext[:, 0].max():+.4f} "
              f"y {ext[:, 1].min():+.4f}..{ext[:, 1].max():+.4f}")
    if c is not None:
        print("sim info:", json.dumps(c.info()))


def drops(sc, c, a):
    holes = {h["shape"]: np.asarray(h["center"], float) for h in sc.lid_holes()}
    z = sc.lid_z[1] + sc.object_height() / 2 + a.gap
    offs = [float(x) for x in a.offsets.split(",")]
    yaws = [float(x) for x in a.yaws.split(",")]
    shapes = a.shapes.split(",")
    cases = []
    for yw in yaws:
        for d in offs:
            for k in (range(a.dirs) if d > 0 else [0]):
                cases.append((d, 360.0 * k / a.dirs if d > 0 else 0.0, yw))
    rows = []
    out = open(a.out, "a") if a.out else None
    for i, (d, ang, yw) in enumerate(cases):
        dxp, dyp = 1e-3 * d * np.cos(np.radians(ang)), 1e-3 * d * np.sin(np.radians(ang))   # pelvis frame
        lay = []
        for s in shapes:
            h = holes[s]
            lay.append({"color": COLOR[s], "shape": s, "x": float(h[0] + dyp), "y": float(h[1] + dxp),
                        "yaw": float(BASE_YAW_MAT[s] - np.radians(yw)), "z": float(z)})   # mat yaw -> pelvis +yw
        act = c.reset(lay)["active"]
        wait_sim(c, a.settle)
        st = {o["name"]: o for o in c.state()["objects"]}
        ev = c.evaluate("insert_shapes")
        by = {o["name"]: o["outcome"] for o in ev["outcomes"]}
        for s, n in zip(shapes, act):
            p = np.asarray(st[n]["pos"], float)
            r = {"type": "drop", "shape": s, "d_mm": d, "dir_deg": ang, "yaw_deg": yw, "gap": a.gap,
                 "outcome": by[n], "pos": p.tolist(), "speed": st[n]["speed"]}
            rows.append(r)
            if out:
                out.write(json.dumps(r) + "\n")
        if out:
            out.flush()
        print(f"case {i + 1}/{len(cases)} d {d:.1f} dir {ang:5.1f} yaw {yw:+.1f}: "
              + " ".join(f"{s}={by[n]}" for s, n in zip(shapes, act)), flush=True)
    print("\ninserted fraction (shape, yaw deg, offset mm: n_inserted/n; other outcomes)")
    for s in shapes:
        for yw in yaws:
            parts = []
            for d in offs:
                rr = [r for r in rows if r["shape"] == s and r["yaw_deg"] == yw and r["d_mm"] == d]
                ins = sum(r["outcome"] == "inserted" for r in rr)
                oth = sorted({r["outcome"] for r in rr} - {"inserted"})
                parts.append(f"{d:g}mm {ins}/{len(rr)}" + (f" ({','.join(oth)})" if oth else ""))
            print(f"  {s:<8} yaw {yw:+5.1f}: " + "  ".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["info", "drops"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--offsets", default="0,3,5,7")
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--yaws", default="0,6")
    ap.add_argument("--shapes", default="cube,cylinder,triangle")
    ap.add_argument("--gap", type=float, default=0.003)
    ap.add_argument("--settle", type=float, default=1.2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from aaltonomous.sim.scene_client import SceneClient
    sc = Scene.load()
    c = SceneClient(a.host, timeout_ms=10000)
    if a.cmd == "info":
        info(sc, c)
    else:
        drops(sc, c, a)


if __name__ == "__main__":
    main()
