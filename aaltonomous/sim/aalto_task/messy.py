"""Objects as they really lie (variant `layout:` section): triangular prisms (optionally cylinders) lying on a
rectangular side, and touching clusters. Pure numpy, unit-tested offline (test_variant.py).

A layout is still tasks.sample_layout's [{color, shape, x, y, yaw}] (mat coords); messy_layout() turns it into the
placed layout, deterministically from the layout itself (crc32 of its JSON), so whoever applies it gets the same
scene: the sim for the layouts it samples itself (scene service reset_random) and the evaluator for AaltoEnv's
(install(): it hands the sim the placed layout and logs it). An explicit scene service `reset` is placed exactly as
given (check_scene's drops, grasp labs). It is applied once: a layout whose entries carry `lying` is taken as placed.
Added keys per entry:
  lying: bool            lying prisms / cylinders: on a rectangular side (prism: triangular ends vertical, ridge on
                         top; height = the triangle's altitude = size; footprint = side x length)
  ridge_yaw: rad         the lying object's long axis (= the ridge) as a yaw about z in the PELVIS frame, mod pi
  side: 0 | 1 | 2        which rectangular side is down (the sides are identical; kept for the record)
  cluster: true          member of the touching cluster (centre distances cluster_spacing x size, any relative yaw)
A request may set `lying` / `ridge_yaw` / `side` itself (scene service `reset`, e.g. a grasp lab): placed as given.
Frames: the layout's x / y / yaw are mat coords (yaw CCW from mat +x, the upright meshes' convention, layout.py);
ridge_yaw is pelvis. Footprints below are in mat coords, CCW polygons (boxes.poly_gap).
"""
from __future__ import annotations

import json
import zlib

import numpy as np

from aaltonomous.scene.scene import shape_polygon

from .boxes import poly_gap, rect

LAYOUT_KEYS = ("lying_prism_prob", "lying_cylinder_prob", "cluster_prob", "cluster_spacing", "cluster_n")
LYING_SHAPES = ("triangle", "cylinder")
GAP_MIN = 0.001          # m between footprints at spawn (the real meshes' fillets add a little more)
SIDE_NORMAL_DEG = (-90.0, 30.0, 150.0)   # outward normals of the prism's rectangular sides in the mesh xy plane
                                          # (apex towards mesh +y, origin at the triangle's centroid: export.py)


def cfg(V):
    return (V or {}).get("layout") or {}


def active(V):
    M = cfg(V)
    return any(float(M.get(k) or 0.0) > 0 for k in ("lying_prism_prob", "lying_cylinder_prob", "cluster_prob"))


def validate(M):
    for k in ("lying_prism_prob", "lying_cylinder_prob", "cluster_prob"):
        if M.get(k) is not None and not 0.0 <= float(M[k]) <= 1.0:
            raise ValueError(f"variant: layout.{k} must be in [0, 1]")
    s = M.get("cluster_spacing")
    if s is not None and (len(s) != 2 or not 1.0 <= float(s[0]) <= float(s[1])):
        raise ValueError("variant: layout.cluster_spacing must be [lo, hi] with 1 <= lo <= hi (x object size)")
    n = M.get("cluster_n")
    if n is not None and (len(n) != 2 or not 2 <= int(n[0]) <= int(n[1])):
        raise ValueError("variant: layout.cluster_n must be [lo, hi] with 2 <= lo <= hi")


# ---------------------------------------------------------------------------------------------- lying geometry
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


R_BASE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # mesh x -> pelvis y, y -> z (up), z -> x


def lying_R(shape, ridge_yaw, side=0):
    """Rotation (pelvis frame) of a lying object: mesh z (the extrusion axis) horizontal along ridge_yaw, the prism's
    rectangular side `side` facing down (its outward normal -> -z); cylinder: just the axis horizontal."""
    a = 0.0 if shape == "cylinder" else np.radians(-90.0 - SIDE_NORMAL_DEG[int(side) % 3])
    return rot_z(float(ridge_yaw)) @ R_BASE @ rot_z(a)


def quat_xyzw(R):
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def R_of_quat(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def lying_dims(sc, shape):
    """(centre height above the counter, footprint across the axis, footprint along the axis, top height)."""
    d = sc.object_dims(shape)
    if shape == "triangle":                       # on a side: the centroid is one apothem above it, apex = altitude
        return d["apothem"], d["side"], d["height"], d["width"]
    if shape == "cylinder":
        return d["width"] / 2, d["width"], d["height"], d["width"]
    raise ValueError(f"only {LYING_SHAPES} can lie, not {shape!r}")


def footprint(sc, ob):
    """Horizontal outline (mat coords, CCW) of a layout entry as spawned."""
    shape = ob.get("shape", "cube")
    c = (float(ob["x"]), float(ob["y"]))
    if ob.get("lying"):
        _, across, along, _ = lying_dims(sc, shape)
        # ridge along pelvis (cos r, sin r) = mat (sin r, cos r): mat angle pi/2 - r
        return rect(c, (along / 2, across / 2), np.pi / 2 - float(ob.get("ridge_yaw", 0.0)))
    return np.asarray(c) + shape_polygon(shape, sc.object_dims(shape)["apothem"], float(ob.get("yaw", 0.0)))


def spawn_height(sc, ob):
    """Centre height above the counter (mat z) of a layout entry at rest."""
    if ob.get("lying"):
        return lying_dims(sc, ob.get("shape", "cube"))[0]
    return sc.object_height(ob.get("shape")) / 2


def pose_truth(quat_xyzw):
    """From an object's orientation: tilt of its extrusion axis from vertical (deg; 0 upright, 90 lying), lying
    (tilt > 45), and the axis' yaw (rad, mod pi, in the frame of the quaternion) when lying, else None."""
    a = R_of_quat(quat_xyzw)[:, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(a[2])))))
    lying = tilt > 45.0
    ridge = None
    if lying:
        ridge = float(np.arctan2(a[1], a[0]))
        ridge = (ridge + np.pi / 2) % np.pi - np.pi / 2
    return {"tilt_deg": round(tilt, 3), "lying": lying, "ridge_yaw": None if ridge is None else round(ridge, 5)}


# ---------------------------------------------------------------------------------------------- sampling
def _clear(sc, ob, others, gap=GAP_MIN):
    P = footprint(sc, ob)
    return all(poly_gap(P, footprint(sc, o)) >= gap for o in others)


def _in_region(sc, ob):
    L = sc.cfg["layout"]
    return L["region_x"][0] <= ob["x"] <= L["region_x"][1] and L["region_y"][0] <= ob["y"] <= L["region_y"][1]


def messy_layout(V, sc, layout):
    """The placed layout (see the module doc). Unchanged (the same list) without `layout:` keys, for an empty layout
    or for one that is already placed (an entry has `lying`)."""
    M = cfg(V)
    if not active(V) or not layout or any("lying" in o for o in layout):
        return layout
    rng = np.random.default_rng(zlib.crc32(json.dumps(layout, sort_keys=True, default=str).encode() + b"messy"))
    out = [dict(o) for o in layout]
    prob = {"triangle": float(M.get("lying_prism_prob") or 0.0), "cylinder": float(M.get("lying_cylinder_prob") or 0.0)}
    for o in out:                                   # same draws for every entry: a seed's stream does not depend on shapes
        u, side, r = rng.random(), int(rng.integers(3)), rng.uniform(-np.pi, np.pi)
        o["lying"] = bool(u < prob.get(o.get("shape", "cube"), 0.0))
        if o["lying"]:
            o["ridge_yaw"], o["side"] = round(float(r), 4), side
    if rng.random() < float(M.get("cluster_prob") or 0.0) and len(out) >= 2:
        lo, hi = (int(v) for v in M.get("cluster_n") or (2, 4))
        n = min(len(out), int(rng.integers(lo, hi + 1)))
        idx = [int(i) for i in rng.choice(len(out), n, replace=False)]
        placed = _cluster(sc, out, idx, M, rng)
        if placed is not None:
            out = placed
    return out


def _cluster(sc, out, idx, M, rng, restarts=30, tries=60):
    """Place out[idx] as a touching cluster in the layout region, then move any other object that now collides
    (resampled in the region at the sampler's min_spacing). None if it does not fit (layout kept, no cluster)."""
    L = sc.cfg["layout"]
    s_lo, s_hi = (float(v) for v in M.get("cluster_spacing") or (1.05, 1.25))
    size = sc.size
    yaw_lo, yaw_hi = np.radians(L["yaw_range_deg"])
    for _ in range(restarts):
        new = [dict(o) for o in out]
        a0 = new[idx[0]]
        a0["x"], a0["y"] = (round(float(rng.uniform(*L["region_x"])), 4), round(float(rng.uniform(*L["region_y"])), 4))
        group = [a0]
        ok = True
        for j in idx[1:]:
            ob = new[j]
            for _ in range(tries):
                m = group[int(rng.integers(len(group)))]
                th, d = rng.uniform(-np.pi, np.pi), rng.uniform(s_lo, s_hi) * size
                ob["x"], ob["y"] = round(float(m["x"] + d * np.cos(th)), 4), round(float(m["y"] + d * np.sin(th)), 4)
                ob["yaw"] = round(float(rng.uniform(yaw_lo, yaw_hi)), 3)
                if ob.get("lying"):
                    ob["ridge_yaw"] = round(float(rng.uniform(-np.pi, np.pi)), 4)
                if _in_region(sc, ob) and _clear(sc, ob, group):
                    group.append(ob)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        for i in idx:
            new[i]["cluster"] = True
        rest = [i for i in range(len(new)) if i not in idx]
        fixed = list(group)
        for i in rest:                              # others: keep, or move clear of everything placed so far
            ob = new[i]
            for k in range(tries * 2):
                if k:
                    ob["x"], ob["y"] = (round(float(rng.uniform(*L["region_x"])), 4),
                                        round(float(rng.uniform(*L["region_y"])), 4))
                if all(np.hypot(ob["x"] - o["x"], ob["y"] - o["y"]) >= L["min_spacing"] for o in fixed) \
                        and _clear(sc, ob, fixed):
                    fixed.append(ob)
                    break
            else:
                ok = False
                break
        if ok:
            return new
    return None


def reach_radius(V, sc):
    """Largest centre-to-footprint distance of any object as this variant may spawn it (upright circumradius, or the
    half diagonal of a lying footprint)."""
    M = cfg(V)
    r = max(sc.object_dims(s)["circumradius"] for s in sc.cfg["objects"]["shapes"])
    for shape, k in (("triangle", "lying_prism_prob"), ("cylinder", "lying_cylinder_prob")):
        if float(M.get(k) or 0.0) > 0 and shape in sc.cfg["objects"]["shapes"]:
            _, across, along, _ = lying_dims(sc, shape)
            r = max(r, float(np.hypot(across, along)) / 2)
    return r


def band_clearance(V, sc):
    """{bin: distance} from the object band (layout region of the centres, grown by reach_radius) to each box's
    footprint: > 0 = nothing this variant spawns can touch a box or lie under its front."""
    import copy
    from .boxes import VariantScene
    vs = sc if isinstance(sc, VariantScene) else VariantScene(copy.deepcopy(sc.cfg))
    L = sc.cfg["layout"]
    lo = sc.mat_to_pelvis([L["region_x"][0], L["region_y"][0], 0.0])
    hi = sc.mat_to_pelvis([L["region_x"][1], L["region_y"][1], 0.0])
    (x0, x1), (y0, y1) = sorted((lo[0], hi[0])), sorted((lo[1], hi[1]))
    r = reach_radius(V, sc)
    band = rect(((x0 + x1) / 2, (y0 + y1) / 2), ((x1 - x0) / 2 + r, (y1 - y0) / 2 + r), 0.0)
    return {n: round(poly_gap(band, vs.footprint(n)), 4) for n in vs.bin_names}


def min_centre_distance(layout):
    p = np.array([[o["x"], o["y"]] for o in layout], float)
    if len(p) < 2:
        return None
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    return float(d[np.triu_indices(len(p), 1)].min())


# ---------------------------------------------------------------------------------------------- evaluator
def install(env, V):
    """Evaluator side: every layout AaltoEnv sends to the sim goes through messy_layout first (in place, so env.layout
    = the trial log = what the sim places; the sim sees `lying` and does not apply it again). No-op without keys."""
    if not active(V):
        return False
    orig = env.backend.reset_scene

    def reset_scene(layout, instruction):
        layout[:] = messy_layout(V, env.scene, list(layout))
        return orig(layout, instruction)
    env.backend.reset_scene = reset_scene
    return True
