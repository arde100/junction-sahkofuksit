"""Check a running Aalto-* sim against its variant (AALTO_VARIANT, aalto_task/variant.py): gains, limits, Dex3
wiring, arm sag, finger response, scene geometry. Needs the sim's scene service `robot` / `geometry` commands
(hooks.py); wiring / sag / finger move the robot.

    aalto py -m aaltonomous.sim.probe_variant params          # info + live gains / limits vs expected (no motion)
    aalto py -m aaltonomous.sim.probe_variant scene [--no-image]   # boxes / table / objects / camera in force vs
                                                              # the variant, objects at rest; head-image HSV of
                                                              # objects / bins / counter vs the real drop-2 medians
    aalto py -m aaltonomous.sim.probe_variant wiring [--both] # each Dex3 slot -> which joint moves, state readback
    aalto py -m aaltonomous.sim.probe_variant sag             # arm tracking error at home + a raised pose (mm)
    aalto py -m aaltonomous.sim.probe_variant finger          # right index step response through the DDS slot
    aalto py -m aaltonomous.sim.probe_variant waist [--side right]  # one arm home -> forward reach -> home: waist
                                                              # angles, torso tilt, head camera deg / mm / px, palm
                                                              # shift the arm joints can't see, settling time
    aalto py -m aaltonomous.sim.probe_variant camera --save A.npz [--seed N]   # home head image + expected camera
    aalto py -m aaltonomous.sim.probe_variant camera --compare A.npz B.npz     # offline: does the IMAGE move as the
                                                              # camera poses say (e.g. real_like vs real_like_waist)
    aalto py -m aaltonomous.sim.probe_variant all             # params, scene, wiring, sag, finger, waist

Expected values: the variant reported by `info` (+ AALTO_HAND_GAINS on top), else upstream's preset / scene.yaml.
Exit code 0 = every check passed (sag / finger / HSV only print numbers). Objects are parked first (sc.reset([])).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time

import numpy as np

# upstream unitree.py G129_CFG_WITH_DEX3_BASE_FIX (the organisers' sim); effort / armature None = USD value
UPSTREAM = {"arms": {"stiffness": {".*_shoulder_.*_joint": 300.0, ".*_elbow_joint": 400.0, ".*_wrist_.*_joint": 400.0},
                     "damping": {".*_shoulder_.*_joint": 3.0, ".*_elbow_joint": 2.5, ".*_wrist_.*_joint": 2.5}},
            "hands": {"stiffness": 100.0, "damping": 10.0, "armature": 0.1, "effort_limit": 300.0},
            "waist": {"stiffness": 10000.0, "damping": 10000.0}}   # rigid (effort 1000 via an alias, armature = USD)
UPSTREAM_SOFT = 0.9


def expected(info):
    from aaltonomous.sim.aalto_task import variant as Vm
    V = info.get("variant")
    exp = {g: dict(p) for g, p in UPSTREAM.items()}
    for g, p in Vm.actuator_overrides(V, info.get("hand_gains")):
        exp[g].update(p)
    return exp, Vm.soft_limit_factor(V, UPSTREAM_SOFT), Vm.wiring(V)


def params(sc, spec, info):
    from aaltonomous.sim.aalto_task.variant import resolve
    exp, soft, wiring = expected(info)
    print(f"variant: {(info.get('variant') or {}).get('name')}  hand_gains: {info.get('hand_gains')}  "
          f"wiring: {wiring}  dr: {(info.get('dr') or {}).get('name')}")
    r = sc.robot()
    J, ok = r["joints"], True
    print(f"soft_joint_pos_limit_factor: cfg {r['soft_joint_pos_limit_factor']}  expected {soft}"
          + (f"  errors {r['errors']}" if r["errors"] else ""))
    ok &= abs(r["soft_joint_pos_limit_factor"] - soft) < 1e-6
    print(f"{'joint':28s} {'stiff':>8s} {'damp':>7s} {'armat':>8s} {'effort':>7s} {'soft/usd':>8s} "
          f"{'usd-urdf lo/hi':>15s}  check")
    names = [(n, "arms", lo, hi) for n, lo, hi in zip(spec.arm_names, spec.arm_lower, spec.arm_upper)]
    names += [(n, "hands", lo, hi) for n, lo, hi in zip(spec.hand_names, spec.hand_lower, spec.hand_upper)]
    for n, g, lo, hi in names:
        j = J[n]
        (plo, phi), (slo, shi) = j["pos_limits"], j["soft_limits"]
        frac = (shi - slo) / (phi - plo)
        bad = []
        for key in ("stiffness", "damping", "armature", "effort_limit"):
            e = resolve(exp[g].get(key), n)
            if e is not None and (j.get(key) is None or abs(j[key] - e) > 1e-4 * max(1.0, abs(e))):
                bad.append(f"{key} {j.get(key)} != {e}")
        if abs(frac - soft) > 1e-3:
            bad.append(f"soft fraction {frac:.3f} != {soft}")
        ok &= not bad
        f = lambda x, s: format(x, s) if x is not None else "-"
        print(f"{n:28s} {f(j['stiffness'], '8.3f')} {f(j['damping'], '7.3f')} {f(j['armature'], '8.5f')} "
              f"{f(j['effort_limit'], '7.2f')} {frac:8.3f} {plo - lo:+7.3f}/{phi - hi:+7.3f}  "
              f"{'ok' if not bad else 'FAIL ' + '; '.join(bad)}")
    for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):      # rigid upstream; variant: soft
        j = J.get(n)
        if j is None:
            ok = False
            print(f"{n:28s} not in the articulation  FAIL")
            continue
        bad = [f"{key} {j.get(key)} != {e}" for key in ("stiffness", "damping", "armature", "effort_limit")
               for e in [resolve(exp["waist"].get(key), n)]
               if e is not None and (j.get(key) is None or abs(j[key] - e) > 1e-4 * max(1.0, abs(e)))]
        ok &= not bad
        f = lambda x, s: format(x, s) if x is not None else "-"
        print(f"{n:28s} {f(j['stiffness'], '8.1f')} {f(j['damping'], '7.2f')} {f(j['armature'], '8.5f')} "
              f"{f(j['effort_limit'], '7.1f')}  pos {np.degrees(j['pos']):+.3f} deg (target "
              f"{np.degrees(j['pos_target']):+.3f})  {'ok' if not bad else 'FAIL ' + '; '.join(bad)}")
    from aaltonomous.sim.aalto_task.variant import joint_limits
    lim = joint_limits(info.get("variant"))
    print(f"joint limit overrides: variant {lim or 'none'}  (sim reports {r.get('joint_limit_overrides')})")
    for n, want in lim.items():
        live = J[n]["pos_limits"]
        bad = [f"{s} {g:+.4f} != {w:+.4f}" for s, g, w in zip(("lo", "hi"), live, want) if w is not None and abs(g - w) > 2e-3]
        ok &= not bad
        print(f"  {n:28s} live [{live[0]:+.4f}, {live[1]:+.4f}]  soft [{J[n]['soft_limits'][0]:+.4f}, "
              f"{J[n]['soft_limits'][1]:+.4f}]  want {want}  {'ok' if not bad else 'FAIL ' + '; '.join(bad)}")
    return ok


# real head camera, drop 2 (handoffs/drop2/REPORT.md): OpenCV HSV medians (H 0-180)
REAL_HSV = {"red": (11, 158, 192), "yellow": (30, 112, 135), "blue": (111, 102, 176), "bin_wall": (4.9, 179, 116),
            "bin_floor": (31, 38, 119), "counter": (0, 23, 146)}


def _quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rotvec_deg(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if a < 1e-9:
        return np.zeros(3)
    return np.degrees(a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]))


def _hsv(rgb, u, v, r=3):
    import cv2
    h, w = rgb.shape[:2]
    u, v = int(round(u)), int(round(v))
    if not (r <= u < w - r and r <= v < h - r):
        return None
    p = cv2.cvtColor(np.ascontiguousarray(rgb[v - r:v + r + 1, u - r:u + r + 1]), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    a = np.radians(p[:, 0].astype(float) * 2)                    # circular median-ish: mean angle (red wraps)
    hue = (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180
    return round(float(hue), 1), int(np.median(p[:, 1])), int(np.median(p[:, 2]))


def _T(pos, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R, float), np.asarray(pos, float)
    return T


def _Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def expected_camera(V, Te, r):
    """Where the head camera must be, in the TABLE frame, given the live robot (`robot` reply r): T_table_pelvis
    (body yaw) @ T_pelvis_parent(live) @ T_parent_optical, with T_parent_optical = T_pelvis_parent(zero)^-1 @ Te
    (the mount scene_cfg built). Returns (T, bend = T_pelvis_parent(live) @ T_pelvis_parent(zero)^-1)."""
    from aaltonomous.sim.aalto_task import variant as Vm
    pt = r.get("pelvis_in_table") or {"pos": [0, 0, 0], "yaw_deg": 0.0}
    Ttp = _T(pt["pos"], _Rz(np.radians(pt["yaw_deg"])))
    parent = Vm.camera_parent(V)
    if parent == "pelvis":
        return Ttp @ Te, np.eye(4)
    L = r["links"][parent]
    bend = _T(L["pos"], L["R"]) @ np.linalg.inv(Vm.link_T_zero(parent))
    return Ttp @ bend @ Te, bend


def scene_probe(sc, info, host, image=True):
    """Geometry the sim really built vs the variant's scene (variant.scene of `info`), objects at rest, camera pose,
    and (image) the head camera's HSV of objects / bins / counter next to the real drop-2 medians."""
    import yaml
    from aaltonomous.sim.aalto_task import boxes as Bx, variant as Vm
    V = info.get("variant")
    S = Vm.scene(V)
    VS = S if Vm.is_variant_scene(S) else Bx.VariantScene(copy.deepcopy(S.cfg))
    rep = Bx.report(VS)
    g = sc.call(cmd="geometry")
    ok, D, W, t = True, *Bx.box_dims(S.cfg)
    print(f"variant: {(V or {}).get('name')} (inherit {(V or {}).get('inherit')})  scene: {type(S).__name__}  "
          f"dr: {(info.get('dr') or {}).get('name')}")
    tol_p, tol_yaw = 0.002, 0.2
    print(f"boxes: outer D x W {D:.3f} x {W:.3f}, wall height {S.cfg['bins']['wall_height']:.3f}, wall {t:.3f} m")
    live = {}
    for n in S.bin_names:
        P = g["bins"][n]
        fl = P["floor"]
        x, y, yaw = VS.poses[n]
        live[n] = (fl["pos"][0], fl["pos"][1], np.radians(fl["yaw_deg"]))
        hgt = max(P[k]["size"][2] for k in ("wall_xlo", "wall_xhi", "wall_ylo", "wall_yhi"))
        dims = sorted(fl["size"][:2])
        bad = []
        if np.hypot(fl["pos"][0] - x, fl["pos"][1] - y) > tol_p:
            bad.append("centre")
        if abs(((fl["yaw_deg"] - np.degrees(yaw)) + 180) % 360 - 180) > tol_yaw:
            bad.append("yaw")
        if abs(hgt - S.cfg["bins"]["wall_height"]) > 1e-3 or abs(dims[0] - min(D, W)) > 1e-3 or abs(dims[1] - max(D, W)) > 1e-3:
            bad.append("size")
        if any(abs(P[k]["yaw_deg"] - fl["yaw_deg"]) > tol_yaw for k in P):
            bad.append("walls not yawed with the floor")
        ok &= not bad
        print(f"  {n:6s} centre live ({fl['pos'][0]:+.4f}, {fl['pos'][1]:+.4f}) want ({x:+.4f}, {y:+.4f})  yaw live "
              f"{fl['yaw_deg']:+6.2f} want {np.degrees(yaw):+6.2f} deg  wall h {hgt:.3f}  floor {dims[0]:.3f} x "
              f"{dims[1]:.3f}  {'ok' if not bad else 'FAIL ' + ', '.join(bad)}")
    LV = Bx.VariantScene(copy.deepcopy(S.cfg), live, VS.table_box())
    lr = Bx.report(LV)
    gap = lambda s, a, b: Bx.poly_gap(s.footprint(a), s.footprint(b))
    print(f"  live: centre box front-wall middle x {lr['boxes']['center']['front_mid'][0]:.4f} (want "
          f"{rep['boxes']['center']['front_mid'][0]:.4f}); gaps left-center {gap(LV, 'left', 'center'):.4f} "
          f"center-right {gap(LV, 'center', 'right'):.4f} (want {gap(VS, 'left', 'center'):.4f} / "
          f"{gap(VS, 'center', 'right'):.4f}); nearest box point x: "
          + ", ".join(f"{n} {b['near_x']:.3f}" for n, b in lr["boxes"].items()))
    print(f"  object band centres x {rep['band_x']} y {rep['band_y']} (+ circumradius) clearance to boxes: "
          + ", ".join(f"{n} {b['band_clearance']:.3f}" for n, b in lr["boxes"].items()) + f"  problems: {lr['problems'] or 'none'}")
    ok &= not lr["problems"]
    # table
    (tx0, tx1), (ty0, ty1) = VS.table_box()
    C = g["counter"]
    cx = (C["pos"][0] - C["size"][0] / 2, C["pos"][0] + C["size"][0] / 2)
    cy = (C["pos"][1] - C["size"][1] / 2, C["pos"][1] + C["size"][1] / 2)
    bad = abs(cx[0] - tx0) > 1e-3 or abs(cx[1] - tx1) > 1e-3 or abs(cy[0] - ty0) > 1e-3 or abs(cy[1] - ty1) > 1e-3
    rough = (g["colors"].get("counter") or {}).get("roughness")
    want_r = Vm.table_roughness(V, 0.8)
    bad |= rough is None or abs(rough - want_r) > 1e-3
    mat_want = (V or {}).get("mat", True)
    bad |= g["mat"] != mat_want
    ok &= not bad
    print(f"table: x [{cx[0]:.3f}, {cx[1]:.3f}] y [{cy[0]:.3f}, {cy[1]:.3f}] want x [{tx0:.3f}, {tx1:.3f}] y [{ty0:.3f}, "
          f"{ty1:.3f}]; top z {C['pos'][2] + C['size'][2] / 2:+.4f} (scene {S.counter_z:+.4f}); roughness {rough} want "
          f"{want_r}; mat {g['mat']} want {mat_want}; colour {(g['colors'].get('counter') or {}).get('diffuse')}  "
          f"{'ok' if not bad else 'FAIL'}")
    # objects: mesh size = the STL (scene.yaml size) x the variant's scale; mass
    import trimesh
    sx = Vm.object_mesh_scale(S)
    want_m = Vm.object_physics(V, "mass", 0.05)
    for shape, b in g["objects"].items():
        if "size" not in b:
            ok = False
            print(f"object {shape}: no prim on the stage  FAIL")
            continue
        m = trimesh.load(str(Vm.REPO / f"aaltonomous/scene/assets/object_{shape}.stl"), force="mesh")
        want = np.sort((m.bounds[1] - m.bounds[0]) * 0.001 * np.asarray(sx))[::-1]
        got = np.sort(np.asarray(b["size"]))[::-1]
        bad = np.abs(got - want).max() > 5e-4 or (b.get("mass") is not None and abs(b["mass"] - want_m) > 1e-4)
        ok &= not bad
        print(f"object {shape:8s} mesh {np.round(got * 1000, 1)} mm want {np.round(want * 1000, 1)}  mass {b.get('mass')} "
              f"want {want_m}  {'ok' if not bad else 'FAIL'}")
    # camera: live pose vs calibration.yaml and vs the variant's (moved with the live torso / body yaw, see
    # expected_camera: a soft waist sags a little under gravity even at home)
    cal = yaml.safe_load(open(Vm.REPO / "aaltonomous/interface/calibration.yaml"))["cameras"]["head"]
    Tc = np.asarray(cal["T_pelvis_optical"], float)
    Te = Vm.camera_T(V, Tc)
    Te0, same = Te, None
    try:
        r = sc.robot("waist")
        if r.get("camera"):     # the camera and its parent link from the SAME step (geometry is a later request: a
            Tr = _T(r["camera"]["pos"], r["camera"]["R"])   # still-swinging soft torso would read as an error there)
            Tx = expected_camera(V, Te0, {**r, "pelvis_in_table": None})[1] @ Te0
            same = (float(np.linalg.norm(_rotvec_deg(Tx[:3, :3].T @ Tr[:3, :3]))), float(np.linalg.norm(Tx[:3, 3] - Tr[:3, 3])))
        if r.get("camera_image_check") is not None:
            print(f"head camera image self-check (sim, depth of the counter): {r['camera_image_check']}")
        Te, bend = expected_camera(V, Te, r)
        print(f"head camera parent {Vm.camera_parent(V)} (sim: {(r.get('camera') or {}).get('parent')}); live waist "
              f"{ {k.replace('_joint', ''): round(float(np.degrees(v)), 3) for k, v in (r.get('waist') or {}).items()} } deg, "
              f"parent link bend {np.round(_rotvec_deg(bend[:3, :3]), 3)} deg / {np.round(bend[:3, 3] * 1000, 2)} mm, "
              f"body yaw {r.get('body_yaw_deg')}")
    except (RuntimeError, KeyError, TimeoutError) as e:     # older sim without body_info: the pelvis mount
        print(f"head camera: no robot body info ({type(e).__name__}: {e}); expecting the pelvis mount")
    if g.get("camera"):
        q = np.asarray(g["camera"]["quat_xyzw_ros"], float)
        cands = {"xyzw": _quat_R(q), "wxyz": _quat_R([q[1], q[2], q[3], q[0]])}
        conv, R = min(cands.items(), key=lambda kv: np.linalg.norm(_rotvec_deg(Te[:3, :3].T @ kv[1])))
        pos = np.asarray(g["camera"]["pos"], float)
        err_r = np.linalg.norm(_rotvec_deg(Te[:3, :3].T @ R))
        err_p = np.linalg.norm(pos - Te[:3, 3])
        bad = err_r > 0.05 or err_p > 1e-3
        if same is not None:
            bad = same[0] > 0.05 or same[1] > 1e-3
            print(f"head camera vs its parent link x mount, same step: {same[0]:.3f} deg, {same[1] * 1000:.2f} mm  "
                  f"{'ok' if not bad else 'FAIL'} (the line below compares a later reading: informative only)")
        ok &= not bad
        print(f"head camera vs calibration.yaml: rotation (optical frame) {np.round(_rotvec_deg(Tc[:3, :3].T @ R), 3)} deg, "
              f"position {np.round((pos - Tc[:3, 3]) * 1000, 1)} mm; vs the variant's: {err_r:.3f} deg, {err_p * 1000:.2f} mm "
              f"(quat read as {conv})  {'ok' if not bad else 'FAIL'}")
        print(f"  optical axis down from horizontal: live {np.degrees(np.arcsin(-R[2, 2])):.2f} deg, calibration "
              f"{np.degrees(np.arcsin(-Tc[2, 2])):.2f} deg")
    else:
        print(f"head camera: not reported ({g.get('errors')})")
        ok, R, pos = False, None, None
    print(f"lights: {g.get('lights')}")
    # objects at rest on the counter (size / counter height physically), then the head image
    h = S.object_height()
    lay = [{"color": col, "shape": "cube", "x": x, "y": 0.155, "yaw": 0.0, "lying": False}   # lying given: the
           for col, x in zip(("red", "yellow", "blue"), (0.35, 0.45, 0.55))]                  # variant layout: keeps it
    act = sc.reset(lay)["active"]
    time.sleep(2.0)
    st = {o["name"]: o for o in sc.state()["objects"]}
    zs = [st[n]["pos"][2] for n in act]
    bad = max(abs(z - (S.counter_z + h / 2)) for z in zs) > 0.003 or max(st[n]["speed"] for n in act) > 0.01
    ok &= not bad
    print(f"objects at rest: centre z {np.round(zs, 4)} want {S.counter_z + h / 2:.4f} (counter + size/2)  "
          f"{'ok' if not bad else 'FAIL'}")
    if image and R is not None:
        _image_hsv(sc, S, VS, g, h, R, pos, cal, host)
    ok &= messy_probe(sc, V, S)
    sc.reset_random(0)
    return ok


def messy_seeds(V, S, n=500):
    """Seeds whose placed layout (what reset_random(seed) places) has a lying object / a cluster / both."""
    from aaltonomous.sim.aalto_task import messy
    from aaltonomous.sim.aalto_task.layout import random_layout
    out = {}
    for seed in range(n):
        pl = messy.messy_layout(V, S, random_layout(S, seed))
        k = (any(o.get("lying") for o in pl), any(o.get("cluster") for o in pl))
        out.setdefault(k, seed)
        if (True, True) in out:
            break
    return out


def messy_probe(sc, V, S, settle_s=2.0):
    """Variant `layout:` (messy.py): place reset_random(seed) for a seed with lying objects + a cluster, and per
    object compare the settled pose with the placed layout: upright / lying, rest height, tilt, ridge yaw, how far it
    moved (a touching cluster must not be pushed apart), speed; plus the minimum centre distance. No-op without keys."""
    from aaltonomous.sim.aalto_task import messy
    from aaltonomous.sim.aalto_task.layout import random_layout
    if not messy.active(V):
        return True
    seeds = messy_seeds(V, S)
    seed = seeds.get((True, True), seeds.get((True, False), seeds.get((False, True))))
    if seed is None:
        print("messy layout: no seed with a lying object or a cluster in 500")
        return False
    want = messy.messy_layout(V, S, random_layout(S, seed))
    r = sc.reset_random(seed)
    same = [{k: o.get(k) for k in ("color", "shape", "lying", "ridge_yaw", "cluster")} for o in r["layout"]] == \
           [{k: o.get(k) for k in ("color", "shape", "lying", "ridge_yaw", "cluster")} for o in want]
    time.sleep(settle_s)
    st = {o["name"]: o for o in sc.state()["objects"]}
    ok = same
    print(f"messy layout (variant layout: {messy.cfg(V)}): seed {seed}, {len(r['active'])} objects; sim and probe "
          f"agree on the placed layout: {same}; band clearance to the boxes {messy.band_clearance(V, S)}")
    got_xy = []
    for ob, n in zip(r["layout"], r["active"]):
        o = st[n]
        p = np.asarray(o["pos"], float)
        m = S.pelvis_to_mat(p)
        got_xy.append({"x": m[0], "y": m[1]})
        z_want = S.counter_z + messy.spawn_height(S, ob)
        moved = float(np.hypot(m[0] - ob["x"], m[1] - ob["y"]))
        tilt_want = 90.0 if ob.get("lying") else 0.0
        bad = []
        if bool(o["lying"]) != bool(ob.get("lying")):
            bad.append("lying")
        if abs(o["tilt_deg"] - tilt_want) > 3.0:
            bad.append("tilt")
        if abs(p[2] - z_want) > 0.003:
            bad.append("height")
        if moved > 0.005:
            bad.append("moved")
        if o["speed"] > 0.01:
            bad.append("moving")
        rid = ""
        if ob.get("lying") and o.get("ridge_yaw") is not None:
            d = (np.degrees(o["ridge_yaw"] - ob["ridge_yaw"]) + 90.0) % 180.0 - 90.0
            rid = f" ridge yaw {np.degrees(o['ridge_yaw']):+6.1f} deg (placed {d:+.1f} off)"
            if abs(d) > 3.0:
                bad.append("ridge")
        ok &= not bad
        print(f"  {n:22s} {'LYING ' if ob.get('lying') else 'upright'}{' cluster' if ob.get('cluster') else '        '}"
              f" rest z {p[2]:+.4f} (want {z_want:+.4f}) tilt {o['tilt_deg']:5.1f} deg moved {moved * 1000:4.1f} mm "
              f"speed {o['speed']:.3f}{rid}  {'ok' if not bad else 'FAIL ' + ', '.join(bad)}")
    print(f"  min centre distance: placed {messy.min_centre_distance(r['layout']) * 1000:.1f} mm, settled "
          f"{messy.min_centre_distance(got_xy) * 1000:.1f} mm (object size {S.size * 1000:.0f} mm)")
    return ok


def _fresh_frame(cli, after=None, timeout=10.0):
    """Next head RGB-D frame (rgb, depth, stamp, frame_id) rendered after frame id `after` (+2: the request round trip)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = cli.get()
        if f is not None and (after is None or f[3] > after + 2):
            return f
        time.sleep(0.05)
    return None


def _image_hsv(sc, S, VS, g, h, R, pos, cal, host, xs=(0.35, 0.45, 0.55), cols=("red", "yellow", "blue")):
    """Head RGB-D frames: HSV at known points projected with the LIVE camera pose; a patch counts only if its depth
    matches the point (not occluded by a hand / wall). The three test cubes are placed three times with the colours
    rotated over the positions and the samples merged, so a hand over one position cannot hide a colour."""
    import os
    from aaltonomous.interface.head_rgbd import HeadRGBDClient
    K = cal["intrinsics"]
    cli = HeadRGBDClient(host, int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    shots = []
    for k in range(len(cols)):
        lay = [{"color": c, "shape": "cube", "x": x, "y": 0.155, "yaw": 0.0, "lying": False}
               for c, x in zip(cols[k:] + cols[:k], xs)]
        prev = cli.get()
        act = sc.reset(lay)["active"]
        time.sleep(1.5)
        f = _fresh_frame(cli, None if prev is None else prev[3])
        if f is not None:
            shots.append((f, {o["name"]: o for o in sc.state()["objects"]}, act))
    cli.close()
    if not shots:
        print("head RGB-D: no frame")
        return
    from aaltonomous.sim.aalto_task.boxes import box_dims
    D, W, t = box_dims(S.cfg)
    hb, ft = float(S.cfg["bins"]["wall_height"]), float(S.cfg["bins"]["floor_thickness"])

    def box_pt(name, bx, by, z):
        x, y, th = VS.poses[name]
        c, s = np.cos(th), np.sin(th)
        return np.array([x + c * bx - s * by, y + s * bx + c * by, S.counter_z + z])

    fixed = {"counter": [np.array([0.38, 0.0, S.counter_z]), np.array([0.40, 0.22, S.counter_z]),
                         np.array([0.40, -0.22, S.counter_z])],
             "bin_wall": [box_pt("center", D / 2 - t - 0.001, dy, 0.6 * hb) for dy in (-0.04, 0.0, 0.04)],
             "bin_floor": [box_pt("center", D / 2 - t - 0.03, dy, ft + 0.0005) for dy in (-0.04, 0.0, 0.04)]}
    samples = []                                                # (key, point, rgb, depth)
    for (rgb_k, depth_k, *_), st, act in shots:
        for k, P in fixed.items():
            samples += [(k, p, rgb_k, depth_k) for p in P]
        for n in act:
            samples.append((n.split("_")[1], np.asarray(st[n]["pos"], float) + [0, 0, h / 2], rgb_k, depth_k))
    print(f"head image HSV (OpenCV H 0-180) at projected points (live camera; {len(shots)} shots, colours rotated over "
          f"the test positions), real drop-2 medians in brackets:")
    for k in dict.fromkeys(s_[0] for s_ in samples):
        vals = []
        for _, p, rgb, depth in (s_ for s_ in samples if s_[0] == k):
            c = R.T @ (p - pos)
            if c[2] <= 0:
                continue
            u, v = K["fx"] * c[0] / c[2] + K["cx"], K["fy"] * c[1] / c[2] + K["cy"]
            hsv = _hsv(rgb, u, v)
            if hsv is None:
                continue
            d = depth[int(round(v)), int(round(u))]
            if d == 0 or abs(d / 1000.0 - c[2]) > 0.015:
                continue                                          # occluded or off the surface
            vals.append(hsv)
        alb = (g["colors"].get(k) or {}).get("diffuse")
        ah = None
        if alb:
            import colorsys
            ah = round(colorsys.rgb_to_hsv(*alb)[0] * 180, 1)
        if vals:
            a = np.radians(np.array([v[0] for v in vals]) * 2)
            hue = (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180
            print(f"  {k:9s} H {hue:5.1f}  S {int(np.median([v[1] for v in vals])):3d}  V {int(np.median([v[2] for v in vals])):3d}"
                  f"   real {REAL_HSV.get(k)}   albedo {alb} (H {ah})  n {len(vals)}")
        else:
            print(f"  {k:9s} no unoccluded sample in any shot (hidden by a hand / arm?)   albedo {alb} (H {ah})")


def _env(a):
    from aaltonomous.interface.env import AaltoEnv
    return AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=1e6, realtime=True, img_server_ip=a.host,
                    head_rgbd=False, cameras=False)


def _move(env, arm=None, hand=None, secs=1.0, vmax=(0.6, 2.5), trace=None):
    """Ramp to (arm, hand) at vmax rad/s (arm, hand) through the safety filter, then hold for secs.
    trace: list to append (t, hand_q) per tick."""
    a0, h0 = env.filter.arm_cmd.copy(), env.filter.hand_cmd.copy()
    a1 = a0 if arm is None else np.asarray(arm, float)
    h1 = h0 if hand is None else np.asarray(hand, float)
    n = int(np.ceil(max(np.abs(a1 - a0).max() / vmax[0], np.abs(h1 - h0).max() / vmax[1]) / env.dt)) or 1
    obs = None
    for i in range(n + int(secs / env.dt)):
        f = min(1.0, (i + 1) / n)
        obs, info = env.step({"arm_q_target": a0 + f * (a1 - a0), "hand_q_target": h0 + f * (h1 - h0)})
        if trace is not None:
            trace.append((obs["t"], obs["hand_q"].copy(), env.filter.hand_cmd.copy()))
        if info["episode_over"]:
            raise RuntimeError(f"episode ended: {info['end_reason']} {info.get('clip_reasons')}")
    return obs


def _far(q, lo, hi, frac=0.6, margin=0.05):
    """Target frac of the way from q to the farther end of [lo + margin, hi - margin]."""
    lo, hi = lo + margin, hi - margin
    end = hi if abs(hi - q) >= abs(lo - q) else lo
    return q + frac * (end - q)


def wiring(env, sc, spec, info, both=False):
    _, _, w = expected(info)
    print(f"wiring probe (sim reports right_hand_wiring={w}; expected: slot k moves joint_spec hand_names[k] and "
          f"rt/dex3 state slot k reads that joint)")
    ok = True
    home = env.filter.hand_cmd.copy()
    for k in range(0 if both else 7, 14):
        before = {n: v["pos"] for n, v in sc.robot("hand")["joints"].items()}
        tgt = home.copy()
        tgt[k] = _far(home[k], spec.hand_lower[k], spec.hand_upper[k])
        obs = _move(env, hand=tgt, secs=1.0)
        J = sc.robot("hand")["joints"]
        moved = max(J, key=lambda n: abs(J[n]["pos"] - before[n]))
        want = spec.hand_names[k]
        d_state = abs(float(obs["hand_q"][k]) - J[want]["pos"])
        good = moved == want and d_state < 0.03
        ok &= good
        print(f"  slot {'LR'[k // 7]}{k % 7}: cmd {tgt[k]:+.2f}  moved {moved:26s} ({J[moved]['pos'] - before[moved]:+.2f})"
              f"  want {want:26s} state slot {obs['hand_q'][k]:+.2f} vs joint {J[want]['pos']:+.2f}  "
              f"{'ok' if good else 'FAIL'}")
        _move(env, hand=home, secs=0.7)
    return ok


def sag(env, sc, spec, poses):
    from aaltonomous.interface.safety import SafetyConfig, WorkspaceChecker
    ws = WorkspaceChecker(spec, SafetyConfig())
    frames = SafetyConfig().workspace_frames
    for name, q in poses:
        if ws.violation(q) > 0:
            print(f"  pose {name}: outside the safety workspace, skipped")
            continue
        _move(env, arm=q, secs=2.0)
        J = sc.robot("shoulder|elbow|wrist")["joints"]
        tgt = np.array([J[n]["pos_target"] for n in spec.arm_names])
        pos = np.array([J[n]["pos"] for n in spec.arm_names])
        d = np.linalg.norm(ws.positions(pos) - ws.positions(tgt), axis=1) * 1000
        err = pos - tgt
        print(f"  pose {name}: frame displacement measured vs target (mm): "
              + "  ".join(f"{f} {x:.1f}" for f, x in zip(frames, d)))
        print("    joint error (mrad): " + "  ".join(f"{n.replace('_joint', '')} {1000 * e:+.0f}"
                                                      for n, e in zip(spec.arm_names, err) if abs(e) > 0.002))


def reach_pose(spec, ws, home, side, z_min):
    """One arm's forward reach (the other at home): over a grid of shoulder pitch x elbow, the pose that puts the
    palm farthest forward while inside the safety workspace and with the palm >= z_min (pelvis frame)."""
    from aaltonomous.interface.safety import SafetyConfig
    k = list(SafetyConfig().workspace_frames).index(f"{side}_hand_palm_link")
    isp, iel = spec.arm_names.index(f"{side}_shoulder_pitch_joint"), spec.arm_names.index(f"{side}_elbow_joint")
    best = None
    for sp in np.linspace(-0.2, -1.4, 13):
        for el in np.linspace(-0.8, 0.8, 17):
            q = np.array(home, float)
            q[isp], q[iel] = sp, el
            if ws.violation(q) > 0:
                continue
            p = ws.positions(q)[k]
            if p[2] >= z_min and (best is None or p[0] > best[0]):
                best = (p[0], q)
    return None if best is None else best[1]


def pixel_shift(T0, T1, K, pts):
    """Image displacement (px) of pelvis-frame points between camera poses T0 -> T1 (4x4 T_pelvis_optical)."""
    def uv(T, p):
        c = T[:3, :3].T @ (np.asarray(p, float) - T[:3, 3])
        return np.array([K["fx"] * c[0] / c[2] + K["cx"], K["fy"] * c[1] / c[2] + K["cy"]])
    return [float(np.linalg.norm(uv(T1, p) - uv(T0, p))) for p in pts]


def _body(r, spec, ws):
    """From a `robot` reply: waist (rad), torso tilt (rotvec deg, pelvis frame), camera T (pelvis frame) and the palms'
    measured positions minus FK of the measured arm joints with a RIGID waist at 0 (WorkspaceChecker = the URDF)."""
    from aaltonomous.interface.safety import SafetyConfig
    q = np.array([r["joints"][n]["pos"] for n in spec.arm_names])
    fk = dict(zip(SafetyConfig().workspace_frames, ws.positions(q)))
    L = r["links"]
    resid = {s: np.asarray(L[f"{s}_hand_palm_link"]["pos"], float) - fk[f"{s}_hand_palm_link"] for s in ("left", "right")}
    cam = _T(r["camera"]["pos"], r["camera"]["R"]) if r.get("camera") else None
    return {"t": r.get("sim_t"), "waist": dict(r.get("waist") or {}), "torso": _rotvec_deg(np.asarray(L["torso_link"]["R"])),
            "cam": cam, "resid": resid, "q": q, "r": r}


def image_shift(img0, img1, center, half=(80, 60)):
    """Sub-pixel shift (du, dv) of the content of img1 relative to img0 in a window around `center` (u, v) by phase
    correlation (grayscale, Hanning window); (None, 0) if the window leaves the image or has no texture."""
    import cv2
    h, w = img0.shape[:2]
    u, v = int(round(center[0])), int(round(center[1]))
    u0, u1, v0, v1 = u - half[0], u + half[0], v - half[1], v + half[1]
    if u0 < 0 or v0 < 0 or u1 > w or v1 > h:
        return None, 0.0
    g = [cv2.cvtColor(np.ascontiguousarray(im[v0:v1, u0:u1]), cv2.COLOR_RGB2GRAY).astype(np.float32) if im.ndim == 3
         else im[v0:v1, u0:u1].astype(np.float32) for im in (img0, img1)]
    if min(float(x.std()) for x in g) < 3.0:      # no texture (flat table / a hand filling it): nothing to correlate
        return None, 0.0
    win = cv2.createHanningWindow((g[0].shape[1], g[0].shape[0]), cv2.CV_32F)
    (du, dv), resp = cv2.phaseCorrelate(g[0], g[1], win)
    return np.array([du, dv]), float(resp)


def box_front_top(S, VS, name):
    """Table-frame point: the middle of a box's near wall top edge (high contrast, rarely occluded)."""
    from aaltonomous.sim.aalto_task.boxes import box_dims
    bx, by, bth = VS.poses[name]
    D = box_dims(S.cfg)[0]
    return np.array([bx - D / 2 * np.cos(bth), by - D / 2 * np.sin(bth), S.counter_z + float(S.cfg["bins"]["wall_height"])])


def pixel_verdict(label, img0, img1, center, pred, min_resp=0.3, tol=(1.0, 0.25)):
    """Measured image shift (phase correlation around `center`) vs the predicted one: ok / FAIL / UNDECIDED (window
    off the image, no texture or correlation response < min_resp: not trusted). Returns True / False / None (undecided)."""
    meas, resp = image_shift(img0, img1, center)
    if meas is None or resp < min_resp:
        why = "window off the image or without texture" if meas is None else f"response {resp:.2f} < {min_resp}"
        print(f"         {label}: UNDECIDED ({why})")
        return None
    bad = np.linalg.norm(meas - pred) > max(tol[0], tol[1] * np.linalg.norm(pred))
    print(f"         {label} (u, v {np.round(center, 1)}): image moved {np.round(meas, 2)} px (response {resp:.2f}), the "
          f"camera poses predict {np.round(pred, 2)} px  {'ok' if not bad else 'FAIL'}")
    return not bad


def _uv(T, K, p):
    c = T[:3, :3].T @ (np.asarray(p, float) - T[:3, 3])
    return np.array([K["fx"] * c[0] / c[2] + K["cx"], K["fy"] * c[1] / c[2] + K["cy"]])


def waist_probe(env, sc, spec, info, side="left", hold=2.5, px_tol=0.5, waist_tol=0.001, host="127.0.0.1",
                cam_tol=(0.1, 0.002)):
    """Waist compliance: one arm home -> forward reach -> home through the normal command interface (safety filter,
    0.6 rad/s). Prints waist angles, torso tilt, head camera pose change (deg, mm, px) and the palm's displacement
    the arm joints cannot see (measured - FK of the measured arm joints with a rigid waist), and the settling time
    after each move (camera within px_tol of its final image position AND waist pitch within waist_tol rad)."""
    import yaml
    from aaltonomous.interface.safety import SafetyConfig, WorkspaceChecker
    from aaltonomous.sim.aalto_task import variant as Vm
    from aaltonomous.sim.aalto_task.layout import ARM_HOME
    V = info.get("variant")
    S = Vm.scene(V)
    ws = WorkspaceChecker(spec, SafetyConfig())
    K = yaml.safe_load(open(Vm.REPO / "aaltonomous/interface/calibration.yaml"))["cameras"]["head"]["intrinsics"]
    exp, _, _ = expected(info)
    wexp = exp.get("waist") or {"stiffness": 10000.0, "damping": 10000.0}
    print(f"waist probe ({side} arm): variant {(V or {}).get('name')}, waist actuators {wexp}, camera parent "
          f"{Vm.camera_parent(V)}")
    home = np.array([ARM_HOME[n] for n in spec.arm_names])
    reach = reach_pose(spec, ws, home, side, S.counter_z + float(S.cfg["bins"]["wall_height"]) + 0.08)
    if reach is None:
        print("  no reach pose inside the safety workspace")
        return False
    pts = [np.array([0.30, 0.0, S.counter_z + S.object_height() / 2]), np.array([0.55, 0.0, S.counter_z])]
    hand = env.filter.hand_cmd.copy()
    Te = Vm.camera_T(V, np.asarray(yaml.safe_load(open(Vm.REPO / "aaltonomous/interface/calibration.yaml"))
                                   ["cameras"]["head"]["T_pelvis_optical"], float))
    ok = True
    # pixel check: the box on the far side from the reaching arm (out of its way), front wall top middle
    from aaltonomous.sim.aalto_task import boxes as Bx
    VS = S if Vm.is_variant_scene(S) else Bx.VariantScene(copy.deepcopy(S.cfg))
    P_ref = box_front_top(S, VS, "center")     # in the image for either arm; above the reaching hand's path
    import os
    from aaltonomous.interface.head_rgbd import HeadRGBDClient
    cli = HeadRGBDClient(host, int(os.environ.get("AALTO_RGBD_PORT", "55560")))

    def cam_check(name, b):
        """The camera the sim reports vs where the live parent link carries the mount (pelvis frame)."""
        nonlocal ok
        if b["cam"] is None:
            print(f"         {name}: no camera pose reported  FAIL")
            ok = False
            return
        _, bend = expected_camera(V, Te, b["r"])
        Tx = bend @ Te
        dr = float(np.linalg.norm(_rotvec_deg(Tx[:3, :3].T @ b["cam"][:3, :3])))
        dp = float(np.linalg.norm(Tx[:3, 3] - b["cam"][:3, 3]))
        bad = dr > cam_tol[0] or dp > cam_tol[1]
        ok &= not bad
        print(f"         {name}: head camera vs the live {Vm.camera_parent(V)} x mount: {dr:.3f} deg, {dp * 1000:.2f} mm  "
              f"{'ok' if not bad else 'FAIL (the camera does not follow its parent link)'}")

    def snap():
        prev = cli.get()
        f = _fresh_frame(cli, None if prev is None else prev[3])
        return None if f is None else f[0]

    def hold_trace(q):
        tr = []
        for _ in range(int(hold / env.dt)):
            _, inf = env.step({"arm_q_target": q, "hand_q_target": hand})
            if inf["episode_over"]:
                raise RuntimeError(f"episode ended: {inf['end_reason']}")
            tr.append(_body(sc.robot("waist|shoulder|elbow|wrist"), spec, ws))
        return tr

    wp = lambda b: b["waist"].get("waist_pitch_joint", 0.0)

    def settle(tr):
        """Sim seconds from the command reaching its target (one tick before tr[0]) until camera and waist stay put;
        None = still moving at the end of the hold."""
        fin = tr[-1]
        bad = [i for i, b in enumerate(tr) if abs(wp(b) - wp(fin)) > waist_tol
               or (b["cam"] is not None and max(pixel_shift(fin["cam"], b["cam"], K, pts)) > px_tol)]
        if not bad:
            return 0.0
        if bad[-1] >= len(tr) - 3 or tr[0]["t"] is None:
            return None
        return round(tr[bad[-1] + 1]["t"] - (tr[0]["t"] - env.dt), 3)

    def show(name, b, ref=None):
        w = {k.replace("_joint", ""): round(float(np.degrees(v)), 3) for k, v in b["waist"].items()}
        s = (f"  {name:6s} waist {w} deg; torso tilt (pelvis frame rotvec) {np.round(b['torso'], 3)} deg; palm vs "
             f"rigid-waist FK: left {np.round(b['resid']['left'] * 1000, 1)} right {np.round(b['resid']['right'] * 1000, 1)} mm")
        print(s)
        if ref is not None:
            d = (b["resid"][side] - ref["resid"][side]) * 1000
            line = f"         vs home: {side} palm moved {np.round(d, 1)} mm ({np.linalg.norm(d):.1f}) beyond what its joints say"
            if b["cam"] is not None and ref["cam"] is not None:
                dR = _rotvec_deg(ref["cam"][:3, :3].T @ b["cam"][:3, :3])
                dt = (b["cam"][:3, 3] - ref["cam"][:3, 3]) * 1000
                px = pixel_shift(ref["cam"], b["cam"], K, pts)
                line += (f"; head camera rotated {np.round(dR, 3)} deg (optical frame, |{np.linalg.norm(dR):.3f}|), "
                         f"moved {np.round(dt, 2)} mm -> image shift {px[0]:.1f} px at the object band, {px[1]:.1f} px "
                         f"at the boxes' front")
            print(line)

    _move(env, arm=home, secs=0.3)
    h0 = hold_trace(home)
    show("home", h0[-1])
    cam_check("home", h0[-1])
    img_home = snap()
    _move(env, arm=reach, secs=0.0)
    tr = hold_trace(reach)
    show("reach", tr[-1], h0[-1])
    cam_check("reach", tr[-1])
    img_reach = snap()
    if img_home is not None and img_reach is not None:      # pixels: does the IMAGE move as the torso predicts?
        Th, Tr = (expected_camera(V, Te, b["r"])[0] for b in (h0[-1], tr[-1]))
        pred = _uv(Tr, K, P_ref) - _uv(Th, K, P_ref)
        ok &= bool(pixel_verdict("image check (home -> reach) at the centre box's front edge", img_home, img_reach,
                                 _uv(Th, K, P_ref), pred))       # undecided counts as not passed
    else:
        ok = False
        print("         image check: no head RGB-D frame  UNDECIDED (counts as not passed)")
    pk = max(tr, key=lambda b: abs(wp(b) - wp(tr[-1])))
    print(f"         reach pose {side}: shoulder pitch {reach[spec.arm_names.index(f'{side}_shoulder_pitch_joint')]:+.2f}, elbow "
          f"{reach[spec.arm_names.index(f'{side}_elbow_joint')]:+.2f} rad; settled {settle(tr)} s after the command "
          f"stopped (camera < {px_tol} px, waist pitch < {waist_tol * 1000:.0f} mrad; None = not within {hold} s); "
          f"largest waist pitch deviation from final during the hold {np.degrees(wp(pk) - wp(tr[-1])):+.3f} deg")
    _move(env, arm=home, secs=0.0)
    tb = hold_trace(home)
    show("back", tb[-1], h0[-1])
    cam_check("back", tb[-1])
    print(f"         settled {settle(tb)} s after the command stopped")
    cli.close()
    return ok


def finger(env, spec, k=10):
    """Step (at the safety filter's hand speed) on hand slot k; time to 63 / 90 % of the step on the state slot."""
    home = env.filter.hand_cmd.copy()
    tgt = home.copy()
    tgt[k] = _far(home[k], spec.hand_lower[k], spec.hand_upper[k], 0.5)
    tr = []
    _move(env, hand=tgt, secs=1.5, vmax=(0.6, 2.9), trace=tr)
    t0, step = tr[0][0], tgt[k] - home[k]
    frac = [(t - t0, (q[k] - home[k]) / step) for t, q, _ in tr]
    t_cmd = next((round(t - t0, 2) for t, _, c in tr if abs(c[k] - tgt[k]) < 1e-3), None)
    t63 = next((round(t, 2) for t, f in frac if f >= 0.63), None)
    t90 = next((round(t, 2) for t, f in frac if f >= 0.90), None)
    print(f"  slot {k} ({spec.hand_names[k]} per joint_spec): step {home[k]:+.2f} -> {tgt[k]:+.2f} rad, command "
          f"there at {t_cmd} s; state 63 % at {t63} s, 90 % at {t90} s; final error {tr[-1][1][k] - tgt[k]:+.3f} rad")
    _move(env, hand=home, secs=0.5)


def camera_save(env, sc, info, path, host, seed=None, hold=2.0):
    """Head image at home (objects parked, arms at the sim home, `hold` s for the waist to settle) + where the camera
    should be (expected_camera, table frame) -> npz, for camera_compare across variants / episodes. seed: first
    reset_random(seed) (a body-yaw variant draws that episode's yaw), then park."""
    import os
    import yaml
    from aaltonomous.interface.head_rgbd import HeadRGBDClient
    from aaltonomous.sim.aalto_task import boxes as Bx, variant as Vm
    from aaltonomous.sim.aalto_task.layout import ARM_HOME
    V = info.get("variant")
    S = Vm.scene(V)
    VS = S if Vm.is_variant_scene(S) else Bx.VariantScene(copy.deepcopy(S.cfg))
    if seed is not None:
        sc.reset_random(seed)
    sc.reset([])
    spec = env.filter.spec
    _move(env, arm=np.array([ARM_HOME[n] for n in spec.arm_names]), secs=hold)
    r = sc.robot("waist")
    cal = yaml.safe_load(open(Vm.REPO / "aaltonomous/interface/calibration.yaml"))["cameras"]["head"]
    T = expected_camera(V, Vm.camera_T(V, np.asarray(cal["T_pelvis_optical"], float)), r)[0]
    cli = HeadRGBDClient(host, int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    prev = _fresh_frame(cli)
    f = _fresh_frame(cli, None if prev is None else prev[3])
    cli.close()
    if f is None:
        print("camera save: no head RGB-D frame")
        return False
    refs = np.array([box_front_top(S, VS, n) for n in ("left", "center", "right")])
    np.savez_compressed(path, rgb=f[0], depth=f[1], T=T, refs=refs, K=json.dumps(cal["intrinsics"]),
                        meta=json.dumps({"variant": (V or {}).get("name"), "body_yaw_deg": r.get("body_yaw_deg"),
                                         "waist": r.get("waist"), "camera_image_check": r.get("camera_image_check")}))
    print(f"camera save -> {path}: variant {(V or {}).get('name')}, body yaw {r.get('body_yaw_deg')}, waist "
          f"{ {k: round(float(np.degrees(v)), 3) for k, v in (r.get('waist') or {}).items()} } deg, image self-check "
          f"{r.get('camera_image_check')}")
    return True


def camera_compare(a_path, b_path):
    """Does the head IMAGE move between two saved shots (camera_save) as the camera poses say? At the three boxes'
    near-wall top edges: phase-correlation shift vs the shift the two expected camera poses predict. E.g. real_like
    vs real_like_waist at home (the soft waist's sag, ~1.7 deg, ~18 px at the box row), or two real_d2_yaw episodes."""
    A, B = np.load(a_path), np.load(b_path)
    K = json.loads(str(A["K"]))
    ma, mb = json.loads(str(A["meta"])), json.loads(str(B["meta"]))
    print(f"camera compare: A {ma}\n                B {mb}")
    dR = _rotvec_deg(A["T"][:3, :3].T @ B["T"][:3, :3])
    print(f"  expected camera B vs A: rotated {np.round(dR, 3)} deg (optical frame), moved "
          f"{np.round((B['T'][:3, 3] - A['T'][:3, 3]) * 1000, 2)} mm")
    res = []
    for name, P in zip(("left", "center", "right"), A["refs"]):
        pred = _uv(B["T"], K, P) - _uv(A["T"], K, P)
        res.append(pixel_verdict(f"{name} box front edge", A["rgb"], B["rgb"], _uv(A["T"], K, P), pred))
    decided = [r for r in res if r is not None]
    ok = bool(decided) and all(decided)       # >= 1 decided reference, none against (side boxes may leave the image)
    print("IMAGE FOLLOWS THE CAMERA POSE" if ok else "IMAGE DOES NOT FOLLOW THE CAMERA POSE" if decided
          else "UNDECIDED: no reference could be correlated")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["params", "scene", "wiring", "sag", "finger", "waist", "camera", "all"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--both", action="store_true", help="wiring: also the left hand")
    ap.add_argument("--side", default="left", choices=["left", "right"], help="waist: the arm that reaches")
    ap.add_argument("--no-image", action="store_true", help="scene: skip the head-image HSV")
    ap.add_argument("--save", help="camera: save the home head image + expected camera pose to this .npz")
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"), help="camera: compare two saved shots (no sim)")
    ap.add_argument("--seed", type=int, default=None, help="camera --save: reset_random(seed) first (body yaw)")
    a = ap.parse_args(argv)
    if a.what == "camera" and a.compare:
        return 0 if camera_compare(*a.compare) else 1
    from aaltonomous.interface.safety import JointSpec
    from aaltonomous.sim.evaluate import apply_instance_env, home_arms
    from aaltonomous.sim.scene_client import SceneClient
    apply_instance_env()
    sc, spec = SceneClient(a.host, timeout_ms=10000), JointSpec()
    info = sc.info()
    ok = True
    if a.what in ("params", "all"):
        ok &= params(sc, spec, info)
    if a.what in ("scene", "all"):
        ok &= scene_probe(sc, info, a.host, image=not a.no_image)
    if a.what in ("params", "scene"):
        print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
        return 0 if ok else 1
    env = _env(a)
    try:
        home_arms(env, sc)
        if a.what in ("wiring", "all"):
            ok &= wiring(env, sc, spec, info, a.both)
        if a.what in ("sag", "all"):
            from aaltonomous.sim.aalto_task.layout import ARM_HOME
            home = np.array([ARM_HOME[n] for n in spec.arm_names])
            raised = home.copy()
            for side in ("left", "right"):     # upper arms forward ~35 deg, forearms near level: the long lever
                raised[spec.arm_names.index(f"{side}_shoulder_pitch_joint")] = -0.6
                raised[spec.arm_names.index(f"{side}_elbow_joint")] = -0.1
            sag(env, sc, spec, [("home", home), ("raised", raised)])
            _move(env, arm=home, secs=0.5)
        if a.what in ("finger", "all"):
            finger(env, spec)
        if a.what in ("waist", "all"):
            ok &= waist_probe(env, sc, spec, info, a.side, host=a.host)
        if a.what == "camera":
            if not a.save:
                raise SystemExit("camera: --save PATH (in the sim) or --compare A.npz B.npz (offline)")
            ok &= camera_save(env, sc, info, a.save, a.host, a.seed)
    finally:
        env.close()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
