"""A tiny kinematic "world" for the mock backend so the classical baseline can be tested end to end offline.

The stock MockBackend returns blank images and has no objects. RenderMockBackend adds:
  * a ray-cast head RGB-D render (counter, coloured bins, coloured objects) through calibration.yaml,
  * kinematic grasping: an object within `attach_radius` of a closing hand's grasp point sticks to it, and drops
    straight down (into a bin or onto the counter) when the hand opens. Fingers on an object stop `held_gap` rad
    short of the closed posture (the sim's squeeze, what grasp verification reads); an empty hand reaches it.
  * scripted failures (grasp events numbered from 0 per episode, in grasp_log): `fail_grasps` / `fail_prob` miss
    (the object is pushed by `miss_push` instead), `slip_grasps` hold for `slip_after_s` and then drop.
  * stacking (`stack_radius` > 0): an object dropped into a bin within that distance (xy) of one already there
    lands on top of it (two 45 mm objects: centre 7 cm up, above the 6 cm rim = not "in").
  * optional static arm tracking error: `sag_gain` (7, rad/Nm; True = SIM_SAG_GAIN): the arms settle at
    commanded + gain * gravity torque (the sim's PD sag), `arm_offset` (14,): + a fixed joint offset. arm_q (what the
    policy measures and where the hand really is) includes it; close_log has one row per hand close (t, side, grasp
    point measured / commanded, nearest object and its distance to the measured grasp point). attach_once: an object
    attaches only at the first closed tick (else also later, e.g. while a closed hand that missed is lifted).
    sag_iters (presets: 3): the torque at the sagged pose itself (kp (cmd - meas) = tau_g(meas), as a PD arm settles;
    the policy's feed-forward lands exactly there); 1 = at the commanded-tracking state (the older tests' numbers).
  * optional real-robot joint signatures (what autoprofile.py tells the worlds apart by; REAL_WORLD / SIM_WORLD =
    presets, make_mock_env(world="real" | "sim")): `cmd_delay` ticks of command latency, `hand_tau` (s) finger lag,
    `hand_deadband` / `arm_deadband` (rad, scalar or per joint): a joint stops that far short of its command (no-load
    error, friction; a smaller command change does not move it), `hand_bias` (14,): a constant finger offset,
    `joint_noise` (arm, hand) rad sd on the measured joints; `hand_soft_limits`: the fingers stop at the middle 90 % of
    their URDF range (the sim's soft limits).
  * other box layouts (`boxes` = box_row(front_x, yaw_deg, wall_h, gap, ...): renders, drops and outcomes use them),
    a larger table (`counter_size`), a true head pose that differs from calibration.yaml (`head_T`), and the arms'
    sphere model in the head image (`render_arms`; else the arms are invisible).
No physics, no contacts. Not a substitute for Isaac (WP3); it only exercises the
perception -> IK -> trajectory -> AaltoEnv/SafetyFilter path. AaltoEnv itself is used unchanged:

    env = make_mock_env(seed=0)       # AaltoEnv(backend="mock") with its backend swapped for this one
"""
from __future__ import annotations

import numpy as np

from aaltonomous.interface.backends import H, W, MockBackend
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

FLOOR_Z = -0.75
MAT_RGB = (175, 175, 170)
FLOOR_RGB = (60, 60, 60)
# Static tracking error of the Aalto sim's arms (PD, no gravity compensation): measured - commanded joint = k * gravity
# torque (pinocchio, URDF masses) per joint, LEFT order, same k for the right arm. Fitted 2026-11-14 on 88 still
# stretches (>= 1 s, arm + hand commanded constant) of the m2_cam_train evaluation: residual sd <= 0.2 mrad per joint
# (stiffness ~276-412 Nm/rad); out of sample at 67 grasp poses the grasp point within 1.4 mm mean of the measured.
SIM_SAG_GAIN = (-0.00362, -0.00346, -0.00354, -0.00280, -0.00077, -0.00286, -0.00242)   # rad / Nm
# The robot as the runner drives it (backends.DDSBackend -> xr_teleoperate G1_29_ArmController): arm kp 80 (shoulders,
# elbow) / 40 (wrists), no gravity feed-forward -> static error = -gravity torque / kp (sign as SIM_SAG_GAIN). Not
# measured on the robot (the teleop data had feed-forward on): a model from the gains only.
REAL_SAG_GAIN = (-1 / 80, -1 / 80, -1 / 80, -1 / 80, -1 / 40, -1 / 40, -1 / 40)
# Real Dex3 at kp 1.5 (data drop 1, free fingers at rest, median |measured - commanded| per joint, LEFT then RIGHT DDS
# order): a joint stops this far short of its command; left thumb_0 has a constant +0.075 bias instead. Lag 70-150 ms.
REAL_HAND_DEADBAND = (0.0, 0.033, 0.015, 0.029, 0.019, 0.027, 0.020, 0.036, 0.028, 0.011, 0.015, 0.009, 0.015, 0.013)
REAL_HAND_BIAS = (0.075,) + (0.0,) * 13
REAL_WORLD = dict(sag_gain=REAL_SAG_GAIN, arm_deadband=0.003, hand_deadband=REAL_HAND_DEADBAND,
                  hand_bias=REAL_HAND_BIAS, cmd_delay=2, hand_tau=0.08, joint_noise=(0.0005, 0.002), sag_iters=3)


def held_gap_for(cfg):
    """How far short of the commanded closed posture (rad on the joint with the largest travel) the mock's fingers
    stop on a held object: a property of the grasp, not of the world. The robot profile's cage (verify.signal
    finger_shortfall) closes 0.3-0.6 rad past the object: held objects read 0.22-0.60 on that signal in the grasp
    lab (real-like sim variants, finger kp 1.5), an empty close <= 0.09; 0.35 gives 0.14-0.32 on the four finger
    joints. The sim profile's pinch: 0.036 (with that value every object held in the cage read "empty" against
    verify.empty_below 0.12)."""
    return 0.35 if ((cfg or {}).get("verify") or {}).get("signal") == "finger_shortfall" else 0.036
SIM_WORLD = dict(sag_gain=True, hand_soft_limits=True, sag_iters=3)
# Object colours as the real head camera sees them (data drop 2, OpenCV H): internal red = orange (H ~12), yellow
# (H ~28), blue = the "purple" objects (H 111, blue-violet). For RenderMockBackend(colors=...).
REAL_COLORS = {"red": (240, 110, 20), "yellow": (210, 200, 40), "blue": (60, 102, 200)}


def _yaw_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def box_row(front_x=0.40, yaw_deg=0.0, wall_h=0.06, gap=0.05, size=0.21, y0=0.0, wall_t=0.003, floor_t=0.003,
            rgb=(230, 110, 30), names=("left", "center", "right")):
    """Three boxes in a row (Renderer / RenderMockBackend `boxes`): the centre box's near wall midpoint at pelvis
    (front_x, y0), the row (and every box) yawed by yaw_deg, `gap` between neighbours. Defaults = scene.yaml."""
    a = np.radians(yaw_deg)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c0 = np.array([front_x, y0]) + size / 2 * u
    out = []
    for k, n in zip((1, 0, -1), ("left", "center", "right")):
        if n in names:
            out.append({"name": n, "c": c0 + k * (size + gap) * v, "yaw": a, "size": size, "wall_h": wall_h,
                        "wall_t": wall_t, "floor_t": floor_t, "rgb": rgb})
    return out


def _rest_height(sc, c):
    """Centre height above the counter of a layout entry at rest (lying: messy.lying_dims)."""
    if c.get("lying"):
        from aaltonomous.sim.aalto_task.messy import spawn_height
        return spawn_height(sc, c)
    return sc.object_height(c.get("shape")) / 2


def _box_local(b, xy):
    d = np.asarray(xy, float)[..., :2] - b["c"]
    c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


class Renderer:
    def __init__(self, scene: Scene, cal: Calibration, cam="head", colors=None, boxes=None, T=None, counter_size=None):
        """colors: {internal colour: rgb} rendered instead of scene.yaml's (e.g. REAL_COLORS; names stay internal).
        boxes: box_row() dicts instead of scene.yaml's bins (any pose / yaw / wall height); T: the TRUE camera pose
        (calibration.yaml's if None; perception keeps using calibration.yaml: a camera error); counter_size: mat-axes
        size of the table top (scene.yaml's if None; the robot's table is larger than the mat)."""
        self.scene, self.cal, self.cam = scene, cal, cam
        c = cal.cams[cam]
        T = cal._T(cam) if T is None else np.asarray(T, float)
        self.boxes, self.counter_size = boxes, counter_size
        self.K, self.R, self.o = c.K, T[:3, :3], T[:3, 3]
        u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
        ray_opt = np.stack([(u - c.K[0, 2]) / c.K[0, 0], (v - c.K[1, 2]) / c.K[1, 1], np.ones_like(u)], -1)
        self.rays = ray_opt @ self.R.T             # pelvis-frame directions; ray param t == optical depth (m)
        self.colors = {k: np.asarray(v, float) for k, v in {**scene.colors, **(colors or {})}.items()}
        self._bg = self._background()

    # ---------------------------------------------------------------- primitives
    def _plane(self, z):
        dz = self.rays[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - self.o[2]) / dz
        return np.where((dz < -1e-9) & (t > 0), t, np.inf)

    def _background(self):
        sc = self.scene
        rgb = np.zeros((H, W, 3), float)
        zb = np.full((H, W), np.inf)
        t_floor = self._plane(FLOOR_Z)
        rgb[np.isfinite(t_floor)] = FLOOR_RGB
        zb = np.minimum(zb, t_floor)
        t = self._plane(sc.counter_z)
        p = self.o + t[..., None] * self.rays
        pm = sc.pelvis_to_mat(np.where(np.isfinite(p), p, 0.0))
        mat = np.asarray(sc.cfg["counter"]["mat_size"], float)
        size = np.asarray(sc.cfg["counter"]["counter_size"] if self.counter_size is None else self.counter_size, float)
        lo, hi = mat / 2 - size / 2, mat / 2 + size / 2
        on = np.isfinite(t) & np.all((pm[..., :2] >= lo) & (pm[..., :2] <= hi), axis=-1) & (t < zb)
        rgb[on] = MAT_RGB
        rgb[on] += ((np.floor(pm[on, 0] / 0.05) + np.floor(pm[on, 1] / 0.05)) % 2)[:, None] * 12.0  # grid
        zb[on] = t[on]
        if self.boxes is not None:
            for bx in self.boxes:
                self._draw_box(rgb, zb, bx)
            return rgb, zb
        b = sc.cfg["bins"]
        ox, oy = np.asarray(b["outer_size"], float) / 2
        wt, wh, ft = b["wall_thickness"], b["wall_height"], b["floor_thickness"]
        for name, (cx, cy) in b["centers"].items():
            col = list(b["rgb"])
            c = sc.mat_to_pelvis([cx, cy, 0.0])
            parts = [((0, 0, ft / 2), (oy, ox, ft / 2)),                 # mat x = pelvis y, mat y = pelvis x
                     ((oy - wt / 2, 0, wh / 2), (wt / 2, ox, wh / 2)), ((-oy + wt / 2, 0, wh / 2), (wt / 2, ox, wh / 2)),
                     ((0, ox - wt / 2, wh / 2), (oy, wt / 2, wh / 2)), ((0, -ox + wt / 2, wh / 2), (oy, wt / 2, wh / 2))]
            for off, half in parts:
                self._box(rgb, zb, c + np.asarray(off), np.eye(3), np.asarray(half), col)
        return rgb, zb

    def _draw_box(self, rgb, zb, b):
        """A box_row() box: floor + 4 walls (outer side `size`, walls wall_t thick, wall_h high) at its pose."""
        a, wt, wh, ft = b["size"] / 2, b["wall_t"], b["wall_h"], b["floor_t"]
        R = _yaw_R(b["yaw"])
        base = np.r_[b["c"], self.scene.counter_z]
        parts = [((0, 0, ft / 2), (a, a, ft / 2)), ((-a + wt / 2, 0, wh / 2), (wt / 2, a, wh / 2)),
                 ((a - wt / 2, 0, wh / 2), (wt / 2, a, wh / 2)), ((0, -a + wt / 2, wh / 2), (a, wt / 2, wh / 2)),
                 ((0, a - wt / 2, wh / 2), (a, wt / 2, wh / 2))]
        for off, half in parts:
            self._box(rgb, zb, base + R @ np.asarray(off, float), R, np.asarray(half, float),
                      np.asarray(b.get("rgb", (230, 110, 30)), float))

    def _sphere(self, rgb, zb, center, r, col):
        win = self._pixels(np.asarray(center, float), np.eye(3), np.full(3, r))
        if win is None:
            return
        d = self.rays[win]
        o = self.o - center
        bq = (d * o).sum(-1)
        disc = bq * bq - (d * d).sum(-1) * (o @ o - r * r)
        with np.errstate(invalid="ignore"):
            t = (-bq - np.sqrt(disc)) / (d * d).sum(-1)
        hit = (disc >= 0) & (t > 0)
        self._write(rgb, zb, win, hit, np.where(hit, t, np.inf), np.full(hit.shape, 0.5), col)

    def _pixels(self, center, R, half):
        corners = center + (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half) @ R.T
        p = (corners - self.o) @ self.R
        with np.errstate(divide="ignore", invalid="ignore"):
            uvz = np.stack([self.K[0, 0] * p[:, 0] / p[:, 2] + self.K[0, 2],
                            self.K[1, 1] * p[:, 1] / p[:, 2] + self.K[1, 2], p[:, 2]], -1)
        if np.any(uvz[:, 2] <= 0.01):
            return None
        u0, v0 = np.floor(uvz[:, :2].min(0)).astype(int) - 1
        u1, v1 = np.ceil(uvz[:, :2].max(0)).astype(int) + 2
        u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
        if u0 >= u1 or v0 >= v1:
            return None
        return slice(v0, v1), slice(u0, u1)

    def _box(self, rgb, zb, center, R, half, col):
        win = self._pixels(center, R, half)
        if win is None:
            return
        d = self.rays[win] @ R                    # into the box frame
        o = (self.o - center) @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / d, (half - o) / d
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        tmin, tmax = tn.max(-1), tf.min(-1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
        axis = tn.argmax(-1)
        nz = np.abs((np.eye(3)[axis] @ R.T)[..., 2])
        self._write(rgb, zb, win, hit, tmin, nz, col)

    def _cylinder(self, rgb, zb, center, radius, hz, col):
        half = np.array([radius, radius, hz])
        win = self._pixels(center, np.eye(3), half)
        if win is None:
            return
        d = self.rays[win]
        o = self.o - center
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = 2 * (o[0] * d[..., 0] + o[1] * d[..., 1])
        cc = o[0] ** 2 + o[1] ** 2 - radius ** 2
        disc = b * b - 4 * a * cc
        with np.errstate(invalid="ignore", divide="ignore"):
            ts = (-b - np.sqrt(disc)) / (2 * a)
            tcap = (hz - o[2]) / d[..., 2]
        side = (disc >= 0) & (ts > 0) & (np.abs(o[2] + ts * d[..., 2]) <= hz)
        pc = o + tcap[..., None] * d
        cap = (tcap > 0) & (pc[..., 0] ** 2 + pc[..., 1] ** 2 <= radius ** 2)
        t = np.where(cap & (~side | (tcap < ts)), tcap, np.where(side, ts, np.inf))
        nz = np.where(t == tcap, 1.0, 0.0)
        self._write(rgb, zb, win, np.isfinite(t), t, nz, col)

    def _prism(self, rgb, zb, center, poly, hz, col):
        """Vertical prism over the convex CCW polygon `poly` (N, 2; pelvis xy relative to center), z +-hz: the ray
        enters at the last entering plane and leaves at the first exiting one (convex polyhedron)."""
        r = float(np.linalg.norm(poly, axis=1).max())
        win = self._pixels(center, np.eye(3), np.array([r, r, hz]))
        if win is None:
            return
        d = self.rays[win]
        o = self.o - center
        e = np.roll(poly, -1, axis=0) - poly
        n = np.c_[e[:, 1], -e[:, 0], np.zeros(len(poly))]                   # outward side normals (CCW polygon)
        n = np.r_[n / np.linalg.norm(n, axis=1, keepdims=True), [[0, 0, 1.0], [0, 0, -1.0]]]
        off = np.r_[(n[:-2, :2] * poly).sum(1), hz, hz]                      # inside: n . p <= off
        with np.errstate(divide="ignore", invalid="ignore"):
            dn = d @ n.T
            t = (off - o @ n.T) / dn
        t_in = np.where(dn < 0, t, -np.inf)
        t_out = np.where(dn > 0, t, np.inf)
        inside0 = np.all(o @ n.T <= off)
        tmin, tmax = t_in.max(-1), t_out.min(-1)
        hit = (tmin <= tmax) & (tmin > 0) & ~inside0
        nz = np.abs(n[t_in.argmax(-1), 2])
        self._write(rgb, zb, win, hit, tmin, nz, col)

    def _convex(self, rgb, zb, center, n, off, r, col):
        """Convex polyhedron {p: n . (p - center) <= off} (n (k, 3) unit outward normals) within radius r."""
        win = self._pixels(center, np.eye(3), np.full(3, r))
        if win is None:
            return
        d = self.rays[win]
        o = self.o - center
        with np.errstate(divide="ignore", invalid="ignore"):
            dn = d @ n.T
            t = (off - o @ n.T) / dn
        t_in = np.where(dn < 0, t, -np.inf)
        t_out = np.where(dn > 0, t, np.inf)
        tmin, tmax = t_in.max(-1), t_out.min(-1)
        hit = (tmin <= tmax) & (tmin > 0) & ~np.all(o @ n.T <= off)
        self._write(rgb, zb, win, hit, tmin, np.abs(n[t_in.argmax(-1), 2]), col)

    def _lying(self, rgb, zb, ob, col):
        """A prism / cylinder lying on a rectangular side (messy.py): its axis horizontal along ob["ridge_yaw"]
        (pelvis), ob["pos"] = its centroid. Prism: triangle of altitude `size` across, `height` long, ridge on top."""
        s, L = ob["size"], ob.get("height", ob["size"])
        r = float(ob.get("ridge_yaw", 0.0))
        a, nn = np.array([np.cos(r), np.sin(r), 0.0]), np.array([-np.sin(r), np.cos(r), 0.0])
        c = np.asarray(ob["pos"], float)
        if ob.get("shape") == "cylinder":                      # rays in the cylinder's frame (axis = local z)
            R = np.c_[nn, [0.0, 0.0, 1.0], a]
            win = self._pixels(c, R, np.array([s / 2, s / 2, L / 2]))
            if win is None:
                return
            d, o = self.rays[win] @ R, (self.o - c) @ R
            A = d[..., 0] ** 2 + d[..., 1] ** 2
            B = 2 * (o[0] * d[..., 0] + o[1] * d[..., 1])
            disc = B * B - 4 * A * (o[0] ** 2 + o[1] ** 2 - s * s / 4)
            with np.errstate(invalid="ignore", divide="ignore"):
                ts = (-B - np.sqrt(disc)) / (2 * A)
                tc = (np.sign(o[2]) * L / 2 - o[2]) / d[..., 2]
            side = (disc >= 0) & (ts > 0) & (np.abs(o[2] + ts * d[..., 2]) <= L / 2)
            pc = o + tc[..., None] * d
            cap = (tc > 0) & (pc[..., 0] ** 2 + pc[..., 1] ** 2 <= s * s / 4)
            t = np.where(cap & (~side | (tc < ts)), tc, np.where(side, ts, np.inf))
            nzs = np.abs((o + np.where(np.isfinite(t), t, 0)[..., None] * d)[..., 1]) / (s / 2)
            self._write(rgb, zb, win, np.isfinite(t), t, np.where(t == tc, 0.0, nzs), col)
            return
        k = np.sqrt(3) / 2
        n = np.array([a, -a, [0.0, 0.0, -1.0], k * nn + [0, 0, 0.5], -k * nn + [0, 0, 0.5]])
        off = np.array([L / 2, L / 2, s / 3, s / 3, s / 3])
        self._convex(rgb, zb, c, n, off, float(np.hypot(L / 2, s)), col)

    @staticmethod
    def _write(rgb, zb, win, hit, t, nz, col):
        z = zb[win]
        m = hit & (t < z)
        if not m.any():
            return
        shade = 0.6 + 0.4 * nz
        sub = rgb[win]
        sub[m] = col * shade[m][:, None]
        z[m] = t[m]
        rgb[win], zb[win] = sub, z

    # ---------------------------------------------------------------- frame
    def render(self, objects, rng=None, depth_noise_mm=0.0, hole_frac=0.0, spheres=None):
        """spheres: [(centres (n, 3), radii (n,))] drawn dark (the arms' sphere model: what the head camera sees of
        the forearms and hands)."""
        rgb, zb = self._bg[0].copy(), self._bg[1].copy()
        for C, Rr in spheres or ():
            for c, r in zip(np.asarray(C, float).reshape(-1, 3), np.asarray(Rr, float).reshape(-1)):
                self._sphere(rgb, zb, c, float(r), np.array([35.0, 35.0, 40.0]))
        from aaltonomous.scene.scene import shape_polygon
        for ob in objects:  # yaw = about pelvis z, scene.shape_polygon convention (triangle apex at yaw + 90 deg)
            col = self.colors[ob["color"]] if isinstance(ob["color"], str) else np.asarray(ob["color"], float)
            s, h = ob["size"], ob.get("height", ob["size"])
            if ob.get("lying") and ob.get("shape") in ("triangle", "cylinder"):
                self._lying(rgb, zb, ob, col)
            elif ob.get("shape", "cube") in ("cylinder", "circle"):
                self._cylinder(rgb, zb, ob["pos"], s / 2, h / 2, col)
            elif ob.get("shape") == "triangle":  # equilateral prism, altitude = size, centroid at pos
                self._prism(rgb, zb, np.asarray(ob["pos"], float), shape_polygon("triangle", s / 3, ob.get("yaw", 0.0)),
                            h / 2, col)
            else:  # cube
                self._box(rgb, zb, ob["pos"], _yaw_R(ob.get("yaw", 0.0)), np.array([s / 2, s / 2, h / 2]), col)
        depth = np.where(np.isfinite(zb), zb * 1000.0, 0.0)
        if rng is not None and depth_noise_mm > 0:
            depth = depth + rng.normal(0, depth_noise_mm, depth.shape)
        if rng is not None and hole_frac > 0:
            depth[rng.random(depth.shape) < hole_frac] = 0
        # depth rounded to mm like the sim (hooks.py); truncating reads 0.5 mm short, which self-levelling sees as
        # a 0.1 deg tilt
        return np.clip(rgb, 0, 255).astype(np.uint8), np.round(np.clip(depth, 0, 65535)).astype(np.uint16)


class RenderMockBackend(MockBackend):
    name = "mock"

    def __init__(self, scene, cal, kin, postures, attach_radius=0.04, depth_noise_mm=1.0, hole_frac=0.02, seed=0,
                 held_gap=0.036, fail_grasps=(), fail_prob=0.0, miss_push=(0.02, 0.0), slip_grasps=(), slip_after_s=1.6,
                 stack_radius=0.0, sag_gain=None, arm_offset=None, attach_once=False, cmd_delay=0, hand_tau=None,
                 hand_deadband=None, arm_deadband=None, hand_bias=None, joint_noise=None, hand_soft_limits=False,
                 sag_iters=1, colors=None, boxes=None, counter_size=None, head_T=None, render_arms=False,
                 ballistic=False, messy=None, **kw):
        """colors: rendered object colours (names stay internal). boxes / counter_size / head_T: the true boxes
        (box_row(); drops and outcomes use them), table and camera pose (see Renderer); render_arms: the head image
        shows the arms' sphere model at the measured pose. ballistic: a released object keeps the grasp point's
        velocity (last tick, measured arm) and flies to the counter level before it drops (point mass, no drag, no
        bounce); with `boxes`, one that comes down on a wall stays on its rim. messy: a variant's `layout:` section
        (lying_prism_prob, cluster_prob, ...): layouts placed by sim/aalto_task/messy.py (lying prisms, touching
        clusters)."""
        super().__init__(**kw)
        self.ballistic, self._g_last = bool(ballistic), {}
        self.boxes, self.counter_size, self.render_arms = boxes, counter_size, bool(render_arms)
        self.messy, self.layout_placed = messy, None
        self.sag_iters = max(1, int(sag_iters))

        def vec(v):
            return None if v is None else np.broadcast_to(np.asarray(v, float), (14,)).copy()
        self.cmd_delay, self.hand_tau = int(cmd_delay), hand_tau
        self.hand_deadband, self.arm_deadband, self.hand_bias = vec(hand_deadband), vec(arm_deadband), vec(hand_bias)
        self.joint_noise = tuple(float(x) for x in joint_noise) if joint_noise else None
        self.noise_rng = np.random.default_rng(seed + 104729)   # own stream: renders / failures as without noise
        self.hand_soft = None
        if hand_soft_limits:
            from aaltonomous.interface.safety import JointSpec
            sp = JointSpec()
            r = sp.hand_upper - sp.hand_lower
            self.hand_soft = (sp.hand_lower + 0.05 * r, sp.hand_upper - 0.05 * r)
        self._cmds = []                                       # cmd_delay: commands not applied yet
        self._joint_model = bool(self.cmd_delay or hand_tau or hand_soft_limits or any(
            v is not None for v in (self.hand_deadband, self.arm_deadband, self.hand_bias)))
        self.sag_gain = None if sag_gain is None or sag_gain is False else np.tile(
            np.asarray(SIM_SAG_GAIN if sag_gain is True else sag_gain, float), 2)
        self.arm_offset = None if arm_offset is None else np.asarray(arm_offset, float)
        self._ideal = None                                    # tracking error on: the arm state without it
        self.attach_once = bool(attach_once)                  # attach only at the first closed tick (no later
                                                              # attach while a closed hand moves, e.g. lifts)
        self.scene, self.kin, self.post = scene, kin, postures
        self.renderer = Renderer(scene, cal, colors=colors, boxes=boxes, T=head_T, counter_size=counter_size)
        self.attach_radius = attach_radius
        self.depth_noise_mm, self.hole_frac = depth_noise_mm, hole_frac
        self.rng = np.random.default_rng(seed)
        self.fail_rng = np.random.default_rng(seed + 7919)   # own stream: the render noise stays as without failures
        self.held_gap, self.fail_grasps, self.fail_prob = held_gap, set(fail_grasps), fail_prob
        self.miss_push, self.slip_grasps = np.asarray(miss_push, float), set(slip_grasps)
        self.slip_after_s = slip_after_s
        self.stack_radius = float(stack_radius)               # > 0: an object dropped this close to one in the bin
        self.objects, self.held = [], {"left": None, "right": None}
        self._cache = None
        self._reset_grasps()

    def _reset_grasps(self):
        self.t, self.grasp_log = 0.0, []         # grasp_log: {t, side, obj, outcome: held | miss | slip}
        self._event = {"left": None, "right": None}  # outcome of this hand's current close (None: hand open)
        self.close_log, self._closing = [], {"left": False, "right": False}

    def _tracking(self, q):
        """Static joint error at arm state q (14,): sag_gain * gravity torque (+ arm_offset). sag_iters > 1: the torque
        at the sagged pose itself (fixed point; a PD arm settles where kp (cmd - meas) = tau_g(meas)); 1 = at q."""
        off = np.zeros(14) if self.arm_offset is None else self.arm_offset.copy()
        if self.sag_gain is not None:
            k = self.kin
            x = k.q0.copy()
            x[k.hand_iq] = self.hand_q
            s = np.zeros(14)
            for i in range(self.sag_iters):
                x[k.arm_iq] = q if i == 0 else q + off + s
                s = self.sag_gain * k.pin.computeGeneralizedGravity(k.model, k.data, x)[k.arm_iv]
            off += s
        return off

    def reset_scene(self, layout, instruction):
        super().reset_scene(layout, instruction)
        sc = self.scene
        if self.messy:                          # lying prisms / touching clusters as the sim places them (messy.py)
            from aaltonomous.sim.aalto_task.messy import messy_layout
            layout = messy_layout({"layout": self.messy}, sc, layout)
        self.layout_placed = layout
        self.objects = [{"color": c["color"], "shape": c.get("shape", "cube"), "yaw": c.get("yaw", 0.0),
                         "size": sc.size, "height": sc.object_height(c.get("shape")), "lying": bool(c.get("lying")),
                         "ridge_yaw": c.get("ridge_yaw", 0.0),
                         "pos": sc.mat_to_pelvis([c["x"], c["y"], _rest_height(sc, c)])} for c in layout]
        self.held = {"left": None, "right": None}
        self._cache = None
        self._reset_grasps()

    def _closure(self, side):
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        o, c = self.post["open"][sl], self.post["closed"][sl]
        return float((self.hand_q[sl] - o) @ (c - o) / ((c - o) @ (c - o)))

    @staticmethod
    def _short(x, cmd, db):
        """Where a joint at x heads for command cmd if it stops db short of it (a smaller change: it stays)."""
        if db is None:
            return cmd
        d = cmd - x
        return np.where(np.abs(d) <= db, x, cmd - db * np.sign(d))

    def _joints_step(self, dt):
        """advance() of the joints with the real-robot / sim signatures (module docstring)."""
        self._cmds.append((self.arm_cmd.copy(), self.hand_cmd.copy()))
        del self._cmds[:-(self.cmd_delay + 1)]
        arm_cmd, hand_cmd = self._cmds[0]
        a = 1.0 - np.exp(-dt / self.tau)
        x = self.arm_q if self._ideal is None else self._ideal
        new = x + a * (self._short(x, arm_cmd, self.arm_deadband) - x)
        self.arm_dq = (new - x) / dt
        self._ideal = new
        err = self.sag_gain is not None or self.arm_offset is not None
        self.arm_q = new + self._tracking(new) if err else new.copy()
        ah = a if not self.hand_tau else 1.0 - np.exp(-dt / self.hand_tau)
        tgt = hand_cmd if self.hand_bias is None else hand_cmd + self.hand_bias
        h = self.hand_q + ah * (self._short(self.hand_q, tgt, self.hand_deadband) - self.hand_q)
        self.hand_q = h if self.hand_soft is None else np.clip(h, *self.hand_soft)

    def joints(self):
        a, dq, h = super().joints()
        if self.joint_noise:   # measurement noise only (the physical state, grasps and renders stay as they are)
            sa, sh = self.joint_noise
            n = self.noise_rng.normal(0.0, 1.0, (3, 14))
            a, dq, h = a + sa * n[0], dq + 20 * sa * n[1], h + sh * n[2]
        return a, dq, h

    def advance(self, dt):
        if self._joint_model:
            self._joints_step(dt)
        elif self.sag_gain is None and self.arm_offset is None:
            super().advance(dt)
        else:  # the first-order lag acts on the arm state without the error; measured = that + the static error
            self._ideal = self.arm_q.copy() if self._ideal is None else self._ideal
            self.arm_q = self._ideal
            super().advance(dt)
            self._ideal = self.arm_q.copy()
            self.arm_q = self._ideal + self._tracking(self._ideal)
        self.t += dt
        for side in ("left", "right"):
            g, _ = self.kin.grasp_pose(side, self.arm_q)
            f = self._closure(side)
            h = self.held[side]
            if f < 0.3:
                self._event[side] = None
                self._closing[side] = False
            first = f > 0.6 and not self._closing[side]
            if first:
                self._closing[side] = True
                taken = [v[0] for s, v in self.held.items() if v and s != side]
                d = [(np.linalg.norm(o["pos"] - g), i) for i, o in enumerate(self.objects) if i not in taken]
                dist, i = min(d) if d else (np.inf, None)
                self.close_log.append({"t": round(self.t, 3), "side": side, "g": g.copy(), "obj": i, "dist": dist,
                                       "g_cmd": self.kin.grasp_pose(side, self.arm_cmd)[0], "pos": None if i is None
                                       else self.objects[i]["pos"].copy()})
            if h is None and f > 0.6 and self._event[side] is None and (first or not self.attach_once):
                free = [i for i, ob in enumerate(self.objects) if i not in [v[0] for v in self.held.values() if v]]
                if free:
                    i = min(free, key=lambda i: np.linalg.norm(self.objects[i]["pos"] - g))
                    if np.linalg.norm(self.objects[i]["pos"] - g) < self.attach_radius:
                        k = len(self.grasp_log)
                        miss = k in self.fail_grasps or (self.fail_prob > 0 and self.fail_rng.random() < self.fail_prob)
                        self._event[side] = "miss" if miss else "slip" if k in self.slip_grasps else "held"
                        self.grasp_log.append({"t": round(self.t, 3), "side": side, "obj": i,
                                               "outcome": self._event[side]})
                        if miss:  # squeezed out: pushed along the counter, the hand closes on nothing
                            self.objects[i]["pos"] = self.objects[i]["pos"] + np.r_[self.miss_push, 0.0]
                        else:
                            t_slip = self.t + self.slip_after_s if k in self.slip_grasps else None
                            self.held[side] = (i, self.objects[i]["pos"] - g, t_slip)
            elif h is not None:
                i, off, t_slip = h
                self.objects[i]["pos"] = g + off
                if f < 0.3 or (t_slip is not None and self.t >= t_slip):
                    if self.ballistic and side in self._g_last:
                        self._fly(self.objects[i], (g - self._g_last[side]) / dt)
                    self._drop(self.objects[i])
                    self.held[side] = None
            self._g_last[side] = g
            if self.held[side] is not None:  # fingers on the object stop held_gap short of the closed posture
                sl = slice(0, 7) if side == "left" else slice(7, 14)
                d = self.post["closed"][sl] - self.post["open"][sl]
                f_stop = 1.0 - self.held_gap / np.abs(d).max()
                if self._closure(side) > f_stop:
                    self.hand_q[sl] = self.hand_q[sl] - (self._closure(side) - f_stop) * d

    def _on_counter(self, pm):
        c = self.scene.cfg["counter"]
        mat = np.asarray(c["mat_size"], float)
        size = np.asarray(c["counter_size"] if self.counter_size is None else self.counter_size, float)
        return bool(np.all(pm[:2] >= mat / 2 - size / 2) and np.all(pm[:2] <= mat / 2 + size / 2)
                    and pm[2] > -float(self.scene.cfg["success"]["off_counter_drop"]))

    def _true_bin(self, pos):
        """Name of the true box (self.boxes) whose inner footprint holds pelvis xy pos (below its rim), else None."""
        z = pos[2] - self.scene.counter_z
        for b in self.boxes:
            L = np.abs(_box_local(b, pos))
            if np.all(L <= b["size"] / 2 - b["wall_t"]) and z <= b["wall_h"]:
                return b["name"]
        return None

    def _fly(self, ob, v):
        """ballistic: the object leaves with velocity v (3,) and flies until its centre is at counter level + half
        its height (then _drop puts it where it came down)."""
        dz = ob["pos"][2] - (self.scene.counter_z + ob["height"] / 2)
        if dz > 0:
            T = (v[2] + np.sqrt(v[2] ** 2 + 2 * 9.81 * dz)) / 9.81
            ob["pos"] = np.r_[ob["pos"][:2] + np.asarray(v, float)[:2] * T, ob["pos"][2]]

    def _drop(self, ob):
        sc = self.scene
        pm = sc.pelvis_to_mat(ob["pos"])
        ob["lying"] = False                     # (a released object lands upright: kinematic mock)
        if self.boxes is not None:              # the true boxes (any pose): inner footprint -> its floor
            for b in self.boxes:
                if np.all(np.abs(_box_local(b, ob["pos"])) <= b["size"] / 2 - b["wall_t"]):
                    ob["pos"] = np.r_[ob["pos"][:2], sc.counter_z + b["floor_t"] + ob["height"] / 2]
                    return
            for b in self.boxes if self.ballistic else ():   # came down on a wall: stays on the rim
                if np.all(np.abs(_box_local(b, ob["pos"])) <= b["size"] / 2):
                    ob["pos"] = np.r_[ob["pos"][:2], sc.counter_z + b["wall_h"] + ob["height"] / 2]
                    return
            z = ob["height"] / 2 if self._on_counter(np.r_[pm[:2], 0.0]) else -1.0
            ob["pos"] = sc.mat_to_pelvis([pm[0], pm[1], z])
            return
        for name in sc.bin_names:
            lo, hi = sc.bin_inner_box(name)
            if np.all(pm[:2] >= lo[:2]) and np.all(pm[:2] <= hi[:2]):
                z = lo[2]
                if self.stack_radius > 0:  # lands on an object already there (centre above the rim = not "in")
                    for o in self.objects:
                        q = sc.pelvis_to_mat(o["pos"])
                        if o is not ob and np.linalg.norm(q[:2] - pm[:2]) < self.stack_radius and q[2] < hi[2] + 0.03:
                            z = max(z, q[2] + o["height"] / 2)
                ob["pos"] = sc.mat_to_pelvis([pm[0], pm[1], z + ob["height"] / 2])
                return
        z = ob["height"] / 2 if sc.on_counter(np.r_[pm[:2], 0.0]) else -1.0
        ob["pos"] = sc.mat_to_pelvis([pm[0], pm[1], z])

    def images(self):
        key = tuple(np.round(np.concatenate([o["pos"] for o in self.objects]), 4)) if self.objects else ()
        sph = None
        if self.render_arms:
            sph = [self.kin.hand_spheres(s, self.arm_q, self.hand_q) for s in ("left", "right")]
            sph = [(c[0], r) for c, r in sph]
            key = key + tuple(np.round(np.r_[self.arm_q, self.hand_q], 3))
        if self._cache is None or self._cache[0] != key:
            rgb, depth = self.renderer.render(self.objects, self.rng, self.depth_noise_mm, self.hole_frac, spheres=sph)
            self._cache = (key, rgb, depth)
        out = super().images()
        out["head_rgb"], out["head_depth"] = self._cache[1].copy(), self._cache[2].copy()
        return out

    def outcomes(self, mapping):
        if self.boxes is not None:              # the true boxes: centre in an inner footprint below the rim
            from aaltonomous.scene.scene import resolve_bin
            out = []
            for o in self.objects:
                b = self._true_bin(o["pos"])
                out.append("off" if not self._on_counter(self.scene.pelvis_to_mat(o["pos"])) else "counter" if b is None
                           else "correct" if resolve_bin(mapping, o["color"]) == b else "wrong")
            return out
        return [self.scene.classify_cube(o["pos"], 0.0, o["color"], mapping) for o in self.objects]  # colour rule (rung 1)


def make_mock_env(scene_path=None, seed=0, cfg=None, realtime=False, world=None, **backend_kw):
    """AaltoEnv(backend="mock") with the render/grasp mock backend. scene_path overrides scene.yaml. world "sim" /
    "real": that world's joint signatures (SIM_WORLD / REAL_WORLD; explicit backend_kw win), arms starting at the
    task's home pose (profile sim's home_arm_q = layout.ARM_HOME), hands open."""
    from aaltonomous.interface.env import AaltoEnv
    from .config import hand_postures, load_config
    from .kinematics import ArmKinematics
    cfg = cfg or load_config()
    if world is not None:
        preset = {"sim": SIM_WORLD, "real": REAL_WORLD}[world]
        backend_kw = {"arm_q0": load_config(profile="sim")["home_arm_q"], **preset, **backend_kw}
    backend_kw.setdefault("held_gap", held_gap_for(cfg))      # (explicit held_gap wins)
    env = AaltoEnv(backend="mock", task="sort_fixed", seed=seed, realtime=realtime)
    if scene_path is not None:
        env.scene = Scene.load(scene_path)
    env.backend = RenderMockBackend(env.scene, Calibration.load(), ArmKinematics(cfg), hand_postures(cfg),
                                    seed=seed, **backend_kw)
    return env
