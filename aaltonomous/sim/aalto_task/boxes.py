"""Box row and table footprint of a sim variant (variant.py `boxes:` / `table:`). Pure numpy, unit-tested offline.

scene.py (organisers', not edited) treats the bins as axis-aligned in mat coordinates and the counter as centred on
the mat. A variant that moves / yaws the boxes or enlarges the table gets a VariantScene: a Scene subclass with
the same cfg (the moved bin centres written back into cfg bins.centers, mat coords, so every consumer of the cfg
drops at the right place) plus per-box poses in the PELVIS frame, and overrides of the membership tests:
  bin_of           exact: centre inside the YAWED inner walls (inside_margin), floor <= z <= rim = the organisers'
                   rule (classify_cube calls it: correct / wrong need it, at rest is checked there as before)
  bin_inner_box    conservative: the largest axis-aligned box (mat) inside the yawed inner footprint, so planners
                   that sample release points in it stay inside; identical to scene.py's at yaw 0
  on_counter       the table footprint (pelvis x / y ranges) instead of counter_size centred on the mat
  classify_insert  rung 3 in the lid bin's own frame (point and track rotated about its centre)
Box frame: x = depth (away from the robot at yaw 0), y = width (robot's left). Yaw about pelvis +z, CCW seen from
above: + = the row's LEFT end comes nearer the robot (the drop-2 report quotes yaw with the opposite sign).
Without `boxes` / `table` a variant keeps the plain Scene (variant.scene), i.e. nothing here runs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from aaltonomous.scene.scene import Scene

ROW = ("right", "center", "left")          # along pelvis +y at yaw 0
BOX_KEYS = ("front_x", "y", "gap", "yaw_deg", "centers", "per_box")
PER_BOX_KEYS = ("dx", "dy", "yaw_deg")
TABLE_KEYS = ("x", "y", "roughness")
MIN_TABLE_X = 0.05                         # the counter slab must stay clear of the robot's pelvis / thighs


def rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def box_dims(cfg):
    """(depth D along box x, width W along box y, wall t) of every bin. scene.yaml outer_size is in MAT axes:
    [x_mat = along the robot's lateral = width, y_mat = away from the robot = depth]."""
    b = cfg["bins"]
    W, D = (float(v) for v in b["outer_size"])
    return D, W, float(b["wall_thickness"])


def mat_centres_pelvis(sc):
    """{name: (x, y)} pelvis frame of the cfg bins.centers."""
    return {n: tuple(sc.mat_to_pelvis([cx, cy, 0.0])[:2]) for n, (cx, cy) in sc.cfg["bins"]["centers"].items()}


def row_poses(sc, B):
    """`boxes:` dict -> {name: (x, y, yaw_rad)} in the pelvis frame. Defaults come from the scene (after the
    variant's scene overlay): front_x / y = the centre box's front-wall midpoint / centre, gap = the cfg spacing."""
    D, W, _ = box_dims(sc.cfg)
    cur = mat_centres_pelvis(sc)
    th = np.radians(float(B.get("yaw_deg", 0.0)))
    if B.get("centers"):
        if set(B["centers"]) != set(sc.bin_names):
            raise ValueError(f"variant: boxes.centers needs exactly {sorted(sc.bin_names)}")
        poses = {n: (float(B["centers"][n][0]), float(B["centers"][n][1]), th) for n in sc.bin_names}
    else:
        if set(sc.bin_names) != set(ROW):
            raise ValueError(f"variant: boxes row needs bins {ROW}, scene has {sc.bin_names}")
        gap = float(B.get("gap", cur["left"][1] - cur["center"][1] - W))
        F = np.array([float(B.get("front_x", cur["center"][0] - D / 2)), float(B.get("y", cur["center"][1]))])
        n, u = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])   # box x, box y
        c = F + D / 2 * n
        poses = {name: (*(c + k * (W + gap) * u), th) for k, name in zip((-1, 0, 1), ROW)}
    for name, d in (B.get("per_box") or {}).items():
        if name not in poses:
            raise ValueError(f"variant: boxes.per_box.{name}: no such bin {sorted(poses)}")
        x, y, t = poses[name]
        poses[name] = (x + float(d.get("dx", 0.0)), y + float(d.get("dy", 0.0)), t + np.radians(float(d.get("yaw_deg", 0.0))))
    return {n: tuple(float(v) for v in p) for n, p in poses.items()}


def rect(center, half, yaw):
    """CCW corners (4, 2) of a rectangle: half = (half along its x, half along its y), rotated by yaw."""
    h = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * np.asarray(half, float)
    return np.asarray(center, float) + h @ rot2(yaw).T


def _separated(P, Q):
    """Separating-axis test for CCW convex polygons (edge normals of both)."""
    for A, B in ((P, Q), (Q, P)):
        e = np.roll(A, -1, axis=0) - A
        for i in range(len(A)):
            if np.min((B - A[i]) @ np.array([e[i, 1], -e[i, 0]])) > 0:     # outward normal of a CCW edge
                return True
    return False


def poly_gap(P, Q):
    """Distance between CCW convex polygons (0 if they touch or overlap)."""
    if not _separated(P, Q):
        return 0.0

    def seg(p, a, b):
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
        return np.linalg.norm(p - (a + t * (b - a)))
    return float(min(seg(p, A[i], A[(i + 1) % len(A)]) for A, B in ((P, Q), (Q, P)) for p in B for i in range(len(A))))


@dataclass
class VariantScene(Scene):
    """Scene with per-box pelvis poses {name: (x, y, yaw_rad)} and an optional table footprint ((x0, x1), (y0, y1)),
    pelvis frame. cfg bins.centers is rewritten from the poses (mat coords)."""
    poses: dict = None
    table: tuple = None

    def __post_init__(self):
        super().__post_init__()
        if self.poses is None:
            self.poses = {n: (*c, 0.0) for n, c in mat_centres_pelvis(self).items()}
        o = self.origin
        b = self.cfg["bins"]
        if set(self.poses) != set(b["centers"]):
            raise ValueError(f"VariantScene: poses for {sorted(self.poses)}, scene bins {sorted(b['centers'])}")
        names = list(b["centers"])               # keep scene.yaml's order (bin_names: DR sampling order, logs)
        b["centers"] = {n: [float(self.poses[n][1] - o[1]), float(self.poses[n][0] - o[0])] for n in names}
        b["yaw_deg"] = {n: float(np.degrees(self.poses[n][2])) for n in names}   # info only (scene.py ignores it)

    # ------------------------------------------------------------ box frame
    def box_pose(self, name):
        return self.poses[name]

    def to_box(self, name, xy_pelvis):
        """Pelvis xy -> box frame xy (x = depth, y = width)."""
        x, y, t = self.poses[name]
        return rot2(-t) @ (np.asarray(xy_pelvis, float)[:2] - [x, y])

    def inner_half(self):
        D, W, t = box_dims(self.cfg)
        return np.array([D / 2 - t, W / 2 - t])

    def footprint(self, name, inner=False):
        D, W, t = box_dims(self.cfg)
        half = self.inner_half() if inner else np.array([D / 2, W / 2])
        x, y, th = self.poses[name]
        return rect((x, y), half, th)

    # ------------------------------------------------------------ membership (scene.py overrides)
    def bin_of(self, p_mat):
        p_mat = np.asarray(p_mat, float)
        m = float(self.cfg["success"]["inside_margin"])
        b = self.cfg["bins"]
        if not float(b["floor_thickness"]) <= p_mat[2] <= float(b["wall_height"]):
            return None
        p = self.mat_to_pelvis(p_mat)
        a = self.inner_half() - m
        for name in self.bin_names:
            if np.all(np.abs(self.to_box(name, p)) <= a):
                return name
        return None

    def bin_inner_box(self, name):
        ax, ay = self.inner_half()                       # box x (depth), box y (width)
        th = abs(self.poses[name][2])
        c, s = np.cos(th), np.sin(th)
        k = min(ax / (ax * c + ay * s), ay / (ax * s + ay * c))
        X, Y = k * ax, k * ay                            # pelvis x / y half extents
        b = self.cfg["bins"]
        cm = np.asarray(b["centers"][name], float)
        half = np.array([Y, X])                          # mat x = pelvis y
        return np.r_[cm - half, b["floor_thickness"]], np.r_[cm + half, b["wall_height"]]

    def on_counter(self, p_mat):
        if self.table is None:
            return super().on_counter(p_mat)
        p = self.mat_to_pelvis(np.asarray(p_mat, float))
        (x0, x1), (y0, y1) = self.table
        inside = x0 <= p[0] <= x1 and y0 <= p[1] <= y1
        return bool(inside and p_mat[2] > -float(self.cfg["success"]["off_counter_drop"]))

    def classify_insert(self, pos_pelvis, speed, shape, track_pelvis=None, beside_margin=0.05):
        """Rung 3 in the lid bin's frame: rotate the point (and track) about the lid bin centre by -yaw and score
        with this scene's boxes unyawed (on_counter is then approximate within the lid bin's reach of a table edge)."""
        x, y, t = self.poses[self.lid_bin]
        if t == 0.0:
            return super().classify_insert(pos_pelvis, speed, shape, track_pelvis, beside_margin)
        R = rot2(-t)

        def un(p):
            p = np.array(p, float)
            p[..., :2] = (p[..., :2] - [x, y]) @ R.T + [x, y]
            return p
        flat = VariantScene(copy.deepcopy(self.cfg), {n: (*p[:2], 0.0) for n, p in self.poses.items()}, self.table)
        return Scene.classify_insert(flat, un(pos_pelvis), speed, shape,
                                     None if track_pelvis is None else un(track_pelvis), beside_margin)

    # ------------------------------------------------------------ table
    def table_box(self):
        """((x0, x1), (y0, y1)) pelvis frame of the counter top."""
        if self.table is not None:
            return self.table
        return counter_extent(Scene(self.cfg))


def counter_extent(sc):
    c = sc.cfg["counter"]
    mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
    lo, hi = sc.mat_to_pelvis(np.r_[mat / 2 - size / 2, 0.0]), sc.mat_to_pelvis(np.r_[mat / 2 + size / 2, 0.0])
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


# ---------------------------------------------------------------------------------------------- sim geometry
def bin_parts(sc, name):
    """[(part, centre pelvis (3,), size pelvis (3,), yaw rad)] of one bin: floor + 4 walls. Plain Scene: exactly
    scene_cfg's original mat-axis formula; VariantScene: the same parts built in the box frame and yawed. Part names
    (wall_xlo = mat x low = the box's right wall, ...) are kept so DR finds the prims."""
    b = sc.cfg["bins"]
    t, ft, h = b["wall_thickness"], b["floor_thickness"], b["wall_height"]
    if not isinstance(sc, VariantScene):
        ox, oy = b["outer_size"]
        cx, cy = b["centers"][name]
        parts = {  # (centre mat xyz, size mat xyz)
            "floor": ((cx, cy, ft / 2), (ox, oy, ft)),
            "wall_xlo": ((cx - ox / 2 + t / 2, cy, h / 2), (t, oy, h)),
            "wall_xhi": ((cx + ox / 2 - t / 2, cy, h / 2), (t, oy, h)),
            "wall_ylo": ((cx, cy - oy / 2 + t / 2, h / 2), (ox - 2 * t, t, h)),
            "wall_yhi": ((cx, cy + oy / 2 - t / 2, h / 2), (ox - 2 * t, t, h)),
        }
        return [(k, sc.mat_to_pelvis(c), np.array([s[1], s[0], s[2]], float), 0.0) for k, (c, s) in parts.items()]
    D, W, _ = box_dims(sc.cfg)
    x, y, th = sc.poses[name]
    local = {  # (centre box xyz, size box xyz); box x = pelvis x at yaw 0, so this IS the formula above rotated
        "floor": ((0.0, 0.0, ft / 2), (D, W, ft)),
        "wall_xlo": ((0.0, -W / 2 + t / 2, h / 2), (D, t, h)),
        "wall_xhi": ((0.0, W / 2 - t / 2, h / 2), (D, t, h)),
        "wall_ylo": ((-D / 2 + t / 2, 0.0, h / 2), (t, W - 2 * t, h)),
        "wall_yhi": ((D / 2 - t / 2, 0.0, h / 2), (t, W - 2 * t, h)),
    }
    out = []
    for k, (c, s) in local.items():
        xy = np.array([x, y]) + rot2(th) @ np.asarray(c[:2], float)
        out.append((k, np.array([xy[0], xy[1], sc.counter_z + c[2]]), np.asarray(s, float), float(th)))
    return out


def counter_box(sc, thickness):
    """(centre pelvis (3,), size pelvis (3,)) of the counter slab: scene_cfg's original formula for a plain Scene."""
    if isinstance(sc, VariantScene) and sc.table is not None:
        (x0, x1), (y0, y1) = sc.table
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2, sc.counter_z - thickness / 2]), np.array([x1 - x0, y1 - y0, thickness])
    c = sc.cfg["counter"]
    mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
    return sc.mat_to_pelvis([mat[0] / 2, mat[1] / 2, -thickness / 2]), np.array([size[1], size[0], thickness])


def yaw_quat_xyzw(th):
    return (0.0, 0.0, float(np.sin(th / 2)), float(np.cos(th / 2)))


# ---------------------------------------------------------------------------------------------- checks / report
def band_rect(sc):
    """Object band (layout region of the centres) in the pelvis frame grown by the largest object circumradius:
    the footprint any spawned object can cover. -> (corners (4, 2), (x0, x1), (y0, y1))."""
    L = sc.cfg["layout"]
    r = max(sc.object_dims(s)["circumradius"] for s in sc.cfg["objects"]["shapes"])
    lo = sc.mat_to_pelvis([L["region_x"][0], L["region_y"][0], 0.0])
    hi = sc.mat_to_pelvis([L["region_x"][1], L["region_y"][1], 0.0])
    (x0, x1), (y0, y1) = sorted((lo[0], hi[0])), sorted((lo[1], hi[1]))
    return rect(((x0 + x1) / 2, (y0 + y1) / 2), ((x1 - x0) / 2 + r, (y1 - y0) / 2 + r), 0.0), (x0, x1), (y0, y1)


def report(sc):
    """Geometry summary of a VariantScene (also the checks validate() enforces). Distances in m, angles in deg."""
    D, W, t = box_dims(sc.cfg)
    band, bx, by = band_rect(sc)
    (tx0, tx1), (ty0, ty1) = sc.table_box()
    tab = rect(((tx0 + tx1) / 2, (ty0 + ty1) / 2), ((tx1 - tx0) / 2, (ty1 - ty0) / 2), 0.0)
    out = {"boxes": {}, "band_x": [round(float(v), 4) for v in bx], "band_y": [round(float(v), 4) for v in by],
           "object_size": sc.size, "wall_height": float(sc.cfg["bins"]["wall_height"]), "outer_D_W": [D, W],
           "table_x": [tx0, tx1], "table_y": [ty0, ty1], "problems": []}
    names = list(sc.bin_names)
    for n in names:
        F = sc.footprint(n)
        x, y, th = sc.poses[n]
        out["boxes"][n] = {"center": [round(x, 4), round(y, 4)], "yaw_deg": round(float(np.degrees(th)), 3),
                           "front_mid": [round(float(v), 4) for v in np.array([x, y]) - D / 2 * np.array([np.cos(th), np.sin(th)])],
                           "near_x": round(float(F[:, 0].min()), 4), "band_clearance": round(poly_gap(F, band), 4)}
        if out["boxes"][n]["band_clearance"] <= 0.0:
            out["problems"].append(f"box {n} overlaps the object band (+ circumradius): objects could spawn in it")
        if not (np.all(F[:, 0] >= tx0) and np.all(F[:, 0] <= tx1) and np.all(F[:, 1] >= ty0) and np.all(F[:, 1] <= ty1)):
            out["problems"].append(f"box {n} is not on the table {sc.table_box()} (its far part would score 'off')")
    for i, a in enumerate(names):
        for b2 in names[i + 1:]:
            g = poly_gap(sc.footprint(a), sc.footprint(b2))
            out["boxes"][a][f"gap_{b2}"] = round(g, 4)
            if g <= 0.0:
                out["problems"].append(f"boxes {a} and {b2} overlap")
    if poly_gap(band, tab) > 0 or not (bx[0] >= tx0 and bx[1] <= tx1 and by[0] >= ty0 and by[1] <= ty1):
        out["problems"].append("object band is not on the table")
    if tx0 < MIN_TABLE_X:
        out["problems"].append(f"table near edge x {tx0} < {MIN_TABLE_X}: the slab would cut into the robot")
    return out
