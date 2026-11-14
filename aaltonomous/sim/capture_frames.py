"""ANALYSIS TOOL (privileged sim state; never import this from a policy): save head RGB-D frames of random layouts
together with the ground-truth object poses, for offline perception tests (shape, yaw, robustness to DR).

    python -m aaltonomous.sim.capture_frames --out /data/aaro/frames/nominal --seeds 3000-3099 [--home]

Per seed: the scene service places a random layout (and, when the sim runs with AALTO_DR, applies that seed's
randomization), the objects settle, then one head RGB-D frame is saved as <out>/<seed>.npz with
  rgb (480,640,3) uint8, depth (480,640) uint16 mm, arm_q (14,), objects = json list of {name, color, shape, pos
  (pelvis), quat_xyzw, yaw_pelvis_rad, lying, tilt_deg, ridge_yaw_pelvis_rad (lying: long axis, mod pi)}, layout =
  json (as placed: a variant's `layout:` adds lying / ridge_yaw / cluster), dr = json (the DR params or null).
--home first moves the arms to the sim home pose through the evaluator's homing routine (needs DDS; without it the
arms stay wherever they are, which is the home pose after any finished evaluation).
Second sim instance: AALTO_SCENE_PORT / AALTO_RGBD_PORT / AALTO_DDS_DOMAIN as for every other client.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="3000-3049")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--settle", type=float, default=1.0, help="sim seconds to wait after placing a layout")
    ap.add_argument("--home", action="store_true")
    a = ap.parse_args(argv)

    from aaltonomous.interface.head_rgbd import HeadRGBDClient
    from aaltonomous.sim.scene_client import SceneClient
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = SceneClient(a.host)
    info = sc.info()
    arm_q = np.full(14, np.nan)
    if a.home:
        from aaltonomous.interface.env import AaltoEnv
        from aaltonomous.sim import evaluate as ev
        ev.apply_instance_env()
        env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, realtime=True, img_server_ip=a.host, head_rgbd=False,
                       cameras=False)
        ev.home_arms(env, sc)
        arm_q = np.asarray(env.backend.joints()[0], float)
        env.close()
    rgbd = HeadRGBDClient(a.host, int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    n = 0
    for seed in _seeds(a.seeds):
        dr = sc.randomize(seed).get("params") if info.get("dr") else None   # only to record the parameters:
        lay = sc.reset_random(seed)["layout"]       # reset_random applies the same seed's DR again, then places
        t0 = sc.ping()["sim_t"]
        while sc.ping()["sim_t"] - t0 < a.settle:
            time.sleep(0.05)
        f0 = rgbd.get()
        fid0 = f0[3] if f0 is not None else -1
        t_wait = time.monotonic()
        while time.monotonic() - t_wait < 10.0:      # a frame rendered after the settle
            f = rgbd.get()
            if f is not None and f[3] > fid0 + 2:
                break
            time.sleep(0.05)
        else:
            print(f"seed {seed}: no fresh frame")
            continue
        rgb, depth = f[0], f[1]
        objs = []
        for o in sc.state()["objects"]:
            if not o["active"]:
                continue
            q = o["quat_xyzw"]
            yaw = float(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
            # lying / tilt_deg / ridge_yaw (rad, pelvis frame, mod pi): from the settled pose (scene service state)
            objs.append({"name": o["name"], "color": o["color"], "shape": o["shape"], "pos": o["pos"],
                         "quat_xyzw": q, "yaw_pelvis_rad": yaw, "lying": o.get("lying", False),
                         "tilt_deg": o.get("tilt_deg"), "ridge_yaw_pelvis_rad": o.get("ridge_yaw")})
        np.savez_compressed(out / f"{seed}.npz", rgb=rgb, depth=depth, arm_q=arm_q, objects=json.dumps(objs),
                            layout=json.dumps(lay), dr=json.dumps(dr))
        n += 1
        print(f"seed {seed}: {len(objs)} objects, rgb mean {rgb.mean():.0f}, depth valid {100 * np.mean(depth > 0):.0f} %",
              flush=True)
    sc.reset([])
    rgbd.close()
    print(f"saved {n} frames to {out}")
    os._exit(0)


if __name__ == "__main__":
    main()
