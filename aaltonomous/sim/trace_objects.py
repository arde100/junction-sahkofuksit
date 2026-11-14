"""ANALYSIS TOOL (privileged sim state; never import this from a policy): poll the scene service while an
evaluation runs and log the active objects' poses over time, so a failed object can be classified afterwards
(never touched / grasp missed / slipped in the lift or carry / dropped beside the bin / knocked off).

    python -m aaltonomous.sim.trace_objects --out /data/aaro/logs/trace.jsonl [--hz 4]      # Ctrl-C / SIGTERM to stop
    python -m aaltonomous.sim.trace_objects --summary /data/aaro/logs/trace.jsonl           # per-episode object summary

One JSON line per sample: {"sim_t", "ep", "t" (sim s since this layout appeared), "objects": [{name, pos (pelvis),
yaw_deg, tilt_deg, speed}]}. `ep` counts layouts: it goes up by one every time a new non-empty layout is placed
(evaluate.py: one per trial; the empty layout during arm homing is skipped).
Second sim instance: AALTO_SCENE_PORT as for every other client.
"""
from __future__ import annotations

import argparse
import json
import signal
import time

import numpy as np


def _yaw_tilt(q):
    x, y, z, w = q
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    tilt = np.degrees(2 * np.arcsin(min(1.0, float(np.hypot(x, y)))))
    return round(float(yaw), 1), round(float(tilt), 1)


def trace(out, hz, host):
    from aaltonomous.sim.scene_client import SceneClient
    sc = SceneClient(host)
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    ep, last_key, t0 = -1, None, 0.0
    with open(out, "a") as f:
        while not stop:
            try:
                sim_t = sc.ping()["sim_t"]
                st = sc.state()
            except (TimeoutError, RuntimeError):
                time.sleep(1.0)
                continue
            except KeyboardInterrupt:
                break
            objs = [o for o in st["objects"] if o["active"]]
            key = json.dumps(st.get("layout"), sort_keys=True) if objs else None
            if key is not None and key != last_key:
                ep, t0 = ep + 1, sim_t
                f.write(json.dumps({"ep": ep, "sim_t": sim_t, "layout": st.get("layout")}) + "\n")
            last_key = key          # None while no object is active (arm homing): the next layout is a new episode
            if objs:
                rows = []
                for o in objs:
                    yaw, tilt = _yaw_tilt(o["quat_xyzw"])
                    rows.append({"name": o["name"], "pos": [round(v, 4) for v in o["pos"]], "yaw": yaw, "tilt": tilt,
                                 "speed": round(o["speed"], 3)})
                f.write(json.dumps({"ep": ep, "sim_t": sim_t, "t": round(sim_t - t0, 3), "objects": rows}) + "\n")
                f.flush()
            try:
                time.sleep(1.0 / hz)
            except KeyboardInterrupt:
                break


def summary(path, lift=0.03):
    """Per episode and object: start xy, when it first rose `lift` m above its start height (lifted), max height,
    when it came down again, final position."""
    from aaltonomous.scene.scene import Scene
    sc = Scene.load()
    eps = {}
    for line in open(path):
        r = json.loads(line)
        if "objects" not in r:
            continue
        for o in r["objects"]:
            eps.setdefault(r["ep"], {}).setdefault(o["name"], []).append((r["t"], *o["pos"], o["tilt"], o["speed"]))
    for ep in sorted(eps):
        print(f"--- layout {ep}")
        for name, rows in sorted(eps[ep].items()):
            a = np.array(rows)
            t, p = a[:, 0], a[:, 1:4]
            z0 = p[0, 2]
            up = np.flatnonzero(p[:, 2] > z0 + lift)
            moved = np.flatnonzero(np.linalg.norm(p[:, :2] - p[0, :2], axis=1) > 0.01)
            pm = sc.pelvis_to_mat(p[-1])
            where = "off" if not sc.on_counter(pm) else (sc.bin_of(pm) or "counter")
            lifts = []
            if len(up):                      # contiguous runs above the lift height = lift episodes
                cuts = np.flatnonzero(np.diff(up) > 1)
                for s, e in zip(np.r_[0, cuts + 1], np.r_[cuts, len(up) - 1]):
                    i, j = up[s], up[e]
                    lifts.append(f"{t[i]:.1f}-{t[j]:.1f}s zmax+{100 * (p[i:j + 1, 2].max() - z0):.0f}cm "
                                 f"down@({p[min(j + 1, len(p) - 1), 0]:+.2f},{p[min(j + 1, len(p) - 1), 1]:+.2f})")
            print(f"  {name:22s} start=({p[0, 0]:+.3f},{p[0, 1]:+.3f}) first_move="
                  f"{'%.1fs' % t[moved[0]] if len(moved) else '-':>6} lifts=[{'; '.join(lifts)}] "
                  f"final=({p[-1, 0]:+.3f},{p[-1, 1]:+.3f},{p[-1, 2]:+.3f}) tilt={a[-1, 4]:.0f} -> {where}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hz", type=float, default=4.0, help="samples per WALL second")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    else:
        trace(a.out, a.hz, a.host)
