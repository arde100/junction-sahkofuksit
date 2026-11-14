"""Arm kinematics for the classical baseline: FK of a grasp point on each Dex3 + damped-least-squares IK.

Model: the full G1 URDF (the one the safety filter uses), fixed base at the pelvis, legs/waist at 0 (the stiff
pose the safety filter and calibration assume). Each arm is solved alone (7 joints); the other arm is untouched.

Grasp point = a point fixed in `<side>_hand_palm_link` (config `grasp.point_in_palm`, right side mirrored in y),
where the object centre should sit between thumb and fingers. `auto` = midway between the index/middle tips and the thumb tip in
the `closed` posture (from the URDF): `closed` leaves a gap a bit smaller than the object, so it is squeezed there.
Orientation: palm-frame x = finger direction; the fingers curl and the thumb points along palm -y (left; +y right).
Default grasp = "claw from above": that curl axis points straight down (palm faces the counter), fingers roughly
forward. IK: position (3) + approach axis down (2, hard-ish) + finger direction hint (weak); yaw is otherwise free.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from aaltonomous.interface.safety import DEFAULT_URDF, JointSpec

SIDES = ("left", "right")
DOWN = np.array([0.0, 0.0, -1.0])
# Distal finger links and the fingertip in each link frame (from the Dex3 STL extents: 52 mm along the link;
# the thumb tip points along -y on the left hand, +y on the right).
FINGER_LINKS = ("hand_index_1_link", "hand_middle_1_link", "hand_thumb_2_link")
TIP = {"left": ([0.052, 0, 0], [0.052, 0, 0], [0, -0.052, 0]), "right": ([0.052, 0, 0], [0.052, 0, 0], [0, 0.052, 0])}
# Two-arm clearance model (no meshes at run time): spheres (x, y, z, r) covering the URDF collision meshes of the LEFT
# forearm + Dex3 (elbow link to fingertips), in the frame of the joint each link hangs on (the palm is fixed to
# wrist_yaw). Right arm = y mirrored (its meshes are mirror images up to ~1 mm; the radii cover both sides). Fitted
# offline to the meshes: centres placed to minimise how far the spheres stick out, then every (subdivided) mesh
# triangle lies inside one sphere and radii are rounded up (+0.5 mm): the model never reports more clearance than
# the meshes have. Checked against coal mesh distances on 12000 two-arm grasp/transit poses (2026-11-14): never
# optimistic, at least 2.8 mm and typically 13 mm conservative (p95 19 mm) when the hands are within 6 cm.
ARM_SPHERES = {
    "elbow_joint": [  # elbow link
        (0.0536, 0.0022, -0.0128, 0.0430), (0.0795, 0.0019, -0.0098, 0.0365), (0.0114, 0.0178, -0.0009, 0.0450),
        (0.0611, 0.0095, -0.0025, 0.0365),
    ],
    "wrist_roll_joint": [  # wrist roll link
        (0.0171, -0.0002, 0.0000, 0.0350), (0.0353, -0.0002, 0.0011, 0.0330),
    ],
    "wrist_pitch_joint": [  # wrist pitch link
        (0.0460, 0.0033, -0.0023, 0.0360), (-0.0005, 0.0003, -0.0028, 0.0370), (0.0006, -0.0015, 0.0043, 0.0355),
        (0.0456, -0.0042, -0.0013, 0.0370),
    ],
    "wrist_yaw_joint": [  # wrist yaw link (first sphere) + palm (14)
        (0.0121, 0.0000, -0.0002, 0.0395), (0.0868, 0.0106, -0.0107, 0.0250), (0.1019, 0.0028, -0.0253, 0.0260),
        (0.0652, 0.0021, -0.0160, 0.0230), (0.0537, -0.0032, -0.0060, 0.0230), (0.0772, 0.0018, 0.0192, 0.0245),
        (0.0786, 0.0087, 0.0196, 0.0255), (0.1111, 0.0024, 0.0186, 0.0300), (0.0572, 0.0082, -0.0108, 0.0255),
        (0.0818, -0.0002, -0.0047, 0.0240), (0.0809, 0.0034, -0.0231, 0.0245), (0.0563, 0.0040, 0.0104, 0.0255),
        (0.0990, 0.0034, 0.0088, 0.0260), (0.1010, 0.0014, 0.0275, 0.0265), (0.1115, 0.0024, -0.0182, 0.0295),
    ],
    "hand_thumb_0_joint": [  # thumb base
        (-0.0007, -0.0134, 0.0000, 0.0200),
    ],
    "hand_thumb_1_joint": [  # thumb proximal (a box in the URDF)
        (-0.0010, -0.0404, 0.0000, 0.0170), (-0.0010, -0.0245, 0.0000, 0.0170),
    ],
    "hand_thumb_2_joint": [  # thumb distal
        (-0.0005, -0.0279, 0.0005, 0.0165), (-0.0005, -0.0077, -0.0072, 0.0170), (-0.0013, -0.0078, 0.0086, 0.0165),
        (-0.0022, -0.0397, -0.0006, 0.0140),
    ],
    "hand_index_0_joint": [  # index proximal
        (0.0024, -0.0014, 0.0000, 0.0180), (0.0227, -0.0020, 0.0006, 0.0220), (0.0419, 0.0024, 0.0004, 0.0220),
    ],
    "hand_index_1_joint": [  # index distal
        (0.0205, -0.0006, 0.0067, 0.0155), (0.0391, 0.0023, 0.0000, 0.0165), (0.0021, -0.0003, -0.0001, 0.0175),
        (0.0200, 0.0003, -0.0058, 0.0165),
    ],
    "hand_middle_0_joint": [  # middle proximal (same mesh as index, fitted separately)
        (0.0315, -0.0067, 0.0000, 0.0175), (0.0123, 0.0005, -0.0073, 0.0210), (0.0411, 0.0045, 0.0000, 0.0215),
        (0.0126, 0.0003, 0.0075, 0.0215),
    ],
    "hand_middle_1_joint": [  # middle distal
        (0.0302, 0.0008, 0.0002, 0.0175), (0.0408, 0.0022, -0.0003, 0.0135), (0.0076, 0.0010, 0.0089, 0.0165),
        (0.0077, 0.0004, -0.0069, 0.0180),
    ],
}
ARM_MIRROR = np.array([1, -1, -1, 1, -1, 1, -1], float)  # arm 7-vector left <-> right: roll / yaw joints flip sign
OBJECT_RADIUS = 0.040  # held object (45 mm cube, any yaw/tilt) centred on the grasp point: half space diagonal 39 mm


class ArmKinematics:
    def __init__(self, cfg, spec: JointSpec | None = None, urdf=DEFAULT_URDF, lower_body_q=None):
        import pinocchio as pin
        self.pin = pin
        self.cfg = cfg
        self.spec = spec or JointSpec()
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q0 = pin.neutral(self.model)
        for name, val in (lower_body_q or {}).items():
            self.q0[self._iq(name)] = val
        self.arm_iq = np.array([self._iq(n) for n in self.spec.arm_names])
        self.arm_iv = np.array([self.model.joints[self.model.getJointId(n)].idx_v for n in self.spec.arm_names])
        self.hand_iq = np.array([self._iq(n) for n in self.spec.hand_names])
        order = cfg.get("right_hand_slots")  # which right-hand joint (index into the spec's 7 names) each command slot
        if order is not None:                # drives, if not the spec's order: every FK below then models the hand
            if sorted(int(i) for i in order) != list(range(7)):   # that really moves (clearances, spheres, tips)
                raise ValueError(f"right_hand_slots must be a permutation of 0..6, got {order}")
            self.hand_iq[7:] = self.hand_iq[7:][np.asarray(order, int)]
        self.palm = {s: self.model.getFrameId(f"{s}_hand_palm_link") for s in SIDES}
        self.fingers = {s: [self.model.getFrameId(f"{s}_{f}") for f in FINGER_LINKS] for s in SIDES}
        self.offset = {"left": np.zeros(3), "right": np.zeros(3)}
        m = np.full(len(self.spec.arm_names), float(cfg["ik"]["limit_margin"]))
        for key, v in (cfg["ik"].get("joint_margin") or {}).items():  # per joint, e.g. {wrist_roll: 0.22}: a robot / sim
            for i, n in enumerate(self.spec.arm_names):               # whose joint stops short of the URDF limit
                if n.endswith(f"_{key}_joint"):
                    m[i] = float(v)
        lo, hi = self.spec.arm_lower + m, self.spec.arm_upper - m
        self.lo, self.hi = lo, hi
        p = cfg["grasp"]["point_in_palm"]
        if isinstance(p, str) and p == "auto":
            from .config import hand_postures
            closed = hand_postures(cfg, self.spec)["closed"]
            t = self.tips_in_palm("left", closed)
            p = 0.5 * (t[:2].mean(0) + t[2])  # midway between the finger pads and the thumb tip
        p = np.asarray(p, float)
        mirror = np.array([1.0, -1.0, 1.0])
        self.offset = {"left": p, "right": p * mirror}
        a = np.asarray(cfg["grasp"]["approach_axis_in_palm"], float)
        self.axis = {"left": a / np.linalg.norm(a), "right": a * mirror / np.linalg.norm(a)}
        self.hint = np.asarray(cfg["grasp"].get("finger_hint", [1.0, 0.0, 0.0]), float)
        self.w_hint = float(cfg["grasp"].get("finger_hint_weight", 0.0))
        self._init_spheres()

    def _init_spheres(self):
        """Flat sphere table of both arms (+ one held-object sphere per hand): parent joint, centre in it, radius."""
        jid, cen, rad, rows = [], [], [], {}
        for side in SIDES:
            m = np.array([1.0, 1.0 if side == "left" else -1.0, 1.0])
            start = len(rad)
            for joint, spheres in ARM_SPHERES.items():
                for s in spheres:
                    jid.append(self.model.getJointId(f"{side}_{joint}"))
                    cen.append(np.asarray(s[:3], float) * m)
                    rad.append(s[3])
            palm = self.model.frames[self.palm[side]]  # object at the grasp point, carried by the palm's joint
            jid.append(palm.parentJoint)
            cen.append(palm.placement.act(self.offset[side]))
            rad.append(OBJECT_RADIUS)
            rows[side] = (np.arange(start, len(rad) - 1), np.arange(start, len(rad)))  # (without, with object)
        joints, k = np.unique(jid, return_inverse=True)
        self._sph_joints = [int(j) for j in joints]  # pinocchio's oMi only takes python ints
        self._sph_k, self._sph_c, self._sph_r, self._sph_rows = k, np.array(cen), np.array(rad), rows
        e = np.cumsum([len(s) for s in ARM_SPHERES.values()] + [1])  # side_spheres() columns per link (+ object)
        self._sph_groups = list(zip(np.r_[0, e[:-1]].tolist(), e.tolist()))
        r = self._sph_r  # radius sums of every left/right pair, per (left holds, right holds)
        self._sph_rsum = {(wl, wr): r[rows["left"][wl], None] + r[None, rows["right"][wr]]
                          for wl in (0, 1) for wr in (0, 1)}

    def tips_in_palm(self, side, hand_q):
        """Fingertip positions (3, 3) in the palm frame for a hand posture (arm pose irrelevant)."""
        self._fk(np.zeros(14), hand_q)
        M = self.data.oMf[self.palm[side]]
        tips = self._tips(side)
        return (tips - M.translation) @ M.rotation

    def _tips(self, side):
        return np.array([self.data.oMf[f].translation + self.data.oMf[f].rotation @ np.asarray(t, float)
                         for f, t in zip(self.fingers[side], TIP[side])])

    def _iq(self, name):
        return self.model.joints[self.model.getJointId(name)].idx_q

    @staticmethod
    def sl(side):
        return slice(0, 7) if side == "left" else slice(7, 14)

    # ------------------------------------------------------------------ FK
    def _fk(self, arm_q, hand_q=None):
        q = self.q0.copy()
        q[self.arm_iq] = arm_q
        if hand_q is not None:
            q[self.hand_iq] = hand_q
        self.pin.framesForwardKinematics(self.model, self.data, q)
        return q

    def grasp_pose(self, side, arm_q):
        """(grasp point position (3,), approach axis (3,)) in the pelvis frame for the full arm_q (14,)."""
        self._fk(arm_q)
        M = self.data.oMf[self.palm[side]]
        return M.translation + M.rotation @ self.offset[side], M.rotation @ self.axis[side]

    def grasp_jacobian(self, side, arm_q):
        """Grasp point (3,), its position Jacobian (3, 7) and the palm's angular Jacobian (3, 7) w.r.t. `side`'s arm
        joints, palm rotation (3, 3); pelvis frame, full arm_q (14,)."""
        pin = self.pin
        q = self._fk(arm_q)
        fid = self.palm[side]
        M = self.data.oMf[fid]
        r = M.rotation @ self.offset[side]
        J6 = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J6 = J6[:, self.arm_iv[self.sl(side)]]
        return M.translation + r, J6[:3] - _skew(r) @ J6[3:], J6[3:], M.rotation.copy()

    def hand_points(self, side, arm_q, hand_q):
        """Pelvis-frame points of one hand: palm, grasp point, distal finger link origins, fingertips (8, 3)."""
        self._fk(arm_q, hand_q)
        M = self.data.oMf[self.palm[side]]
        pts = [M.translation.copy(), M.translation + M.rotation @ self.offset[side]]
        pts += [self.data.oMf[f].translation.copy() for f in self.fingers[side]]
        return np.vstack([np.array(pts), self._tips(side)])

    # ------------------------------------------------------------------ two-arm clearance
    def _sphere_centres(self, arm_q, hand_q):
        """Pelvis-frame centres of the whole sphere table: one joint-only FK (no frames) + one batched transform."""
        q = self.q0.copy()
        q[self.arm_iq] = arm_q
        q[self.hand_iq] = hand_q
        self.pin.forwardKinematics(self.model, self.data, q)
        H = np.array([self.data.oMi[j].homogeneous for j in self._sph_joints])[self._sph_k]
        return np.einsum("nij,nj->ni", H[:, :3, :3], self._sph_c) + H[:, :3, 3]

    def arm_spheres(self, side, arm_q, hand_q, with_object=False):
        """Sphere model of one forearm + hand (+ the held object): (centres (n, 3) pelvis frame, radii (n,)).
        arm_q, hand_q: full (14,) vectors."""
        i = self._sph_rows[side][int(bool(with_object))]
        return self._sphere_centres(arm_q, hand_q)[i], self._sph_r[i]

    def arm_arm_distance(self, arm_q, hand_q, with_object=(False, False)):
        """Smallest surface distance (m) between the left and the right forearm/hand sphere models (< 0 = overlap).
        Conservative: never more than the URDF collision meshes' true distance. with_object: (left, right) held."""
        c = self._sphere_centres(arm_q, hand_q)
        wl, wr = (int(bool(w)) for w in with_object)
        il, ir = self._sph_rows["left"][wl], self._sph_rows["right"][wr]
        return float((cdist(c[il], c[ir]) - self._sph_rsum[wl, wr]).min())

    def arm_arm_closest(self, arm_q, hand_q, with_object=(False, False)):
        """arm_arm_distance + which spheres are closest: (distance m, left label, right label); labels are the
        ARM_SPHERES joint names without '_joint' (e.g. 'wrist_yaw' = wrist + palm) or 'object'."""
        c = self._sphere_centres(arm_q, hand_q)
        wl, wr = (int(bool(w)) for w in with_object)
        il, ir = self._sph_rows["left"][wl], self._sph_rows["right"][wr]
        D = cdist(c[il], c[ir]) - self._sph_rsum[wl, wr]
        i, j = np.unravel_index(int(np.argmin(D)), D.shape)
        names = [j_.replace("_joint", "") for j_, s in ARM_SPHERES.items() for _ in s] + ["object"]
        return float(D[i, j]), names[i], names[j]

    def min_arm_arm_distance(self, arm, hand, step=2, with_object=None, stop_below=None):
        """Lowest arm_arm_distance along a merged trajectory (arm, hand: (T, 14)), checked every `step` ticks + the
        last one. with_object: None or (left, right), each a bool or a (T,) bool array (object held at that tick).
        stop_below: return at the first checked tick below it. Returns (distance m, tick)."""
        T = len(arm)
        wo = [np.broadcast_to(np.asarray(w, bool), (T,)) for w in (with_object or (False, False))]
        ticks = list(range(0, T, max(1, int(step))))
        if T and ticks[-1] != T - 1:
            ticks.append(T - 1)
        best, tick = np.inf, -1
        for k in ticks:
            d = self.arm_arm_distance(arm[k], hand[k], (wo[0][k], wo[1][k]))
            if d < best:
                best, tick = d, k
                if stop_below is not None and d < stop_below:
                    break
        return best, tick

    def side_spheres(self, side, arm, hand):
        """One arm's sphere model along a trajectory (arm, hand: (T, 14); only `side`'s columns matter): centres
        (T, n + 1, 3) pelvis frame and radii (n + 1,); the last sphere is the held object (at the grasp point)."""
        i = self._sph_rows[side][1]
        c = np.array([self._sphere_centres(a, h)[i] for a, h in zip(arm, hand)]).reshape(-1, len(i), 3)
        return c, self._sph_r[i]

    def hand_spheres(self, side, arm, hand):
        """side_spheres without the held object, computed from that side's joints only (~3x faster: the neighbour
        check runs on every tick of every candidate plan): centres (T, n, 3) pelvis frame, radii (n,)."""
        if not hasattr(self, "_sph_side"):
            self._sph_side = {}
        if side not in self._sph_side:
            i = self._sph_rows[side][0]
            js, k = np.unique(np.asarray(self._sph_joints)[self._sph_k[i]], return_inverse=True)
            self._sph_side[side] = ([int(j) for j in js], k, self._sph_c[i], self._sph_r[i])
        js, k, cen, rad = self._sph_side[side]
        arm, hand = np.asarray(arm, float).reshape(-1, 14), np.asarray(hand, float).reshape(-1, 14)
        q = np.repeat(self.q0[None], len(arm), 0)
        q[:, self.arm_iq], q[:, self.hand_iq] = arm, hand
        out = np.empty((len(arm), len(k), 3))
        for t in range(len(arm)):
            self.pin.forwardKinematics(self.model, self.data, q[t])
            H = np.array([self.data.oMi[j].homogeneous for j in js])[k]
            out[t] = np.einsum("nij,nj->ni", H[:, :3, :3], cen) + H[:, :3, 3]
        return out, rad

    def sphere_distance_matrix(self, ca, ra, cb, rb, wa=None, wb=None, below=np.inf, chunk=2000):
        """Model distance (m) between every tick pair of two side_spheres() sets: (Ta, Tb). wa (Ta,) / wb (Tb,) bool:
        the object sphere counts at that tick (default: never). Exact where the result is < `below`; elsewhere a
        lower bound >= below (per link: a bounding sphere of its spheres), enough for a test against `below`."""
        def groups(c, r, w):
            held = np.zeros(len(c), bool) if w is None else np.asarray(w, bool)
            out = []
            for s, e in self._sph_groups:
                m = c[:, s:e].mean(1)
                b = (np.linalg.norm(c[:, s:e] - m[:, None], axis=-1) + r[s:e]).max(1)
                if e == c.shape[1]:                                 # object sphere: only where held
                    b = np.where(held, b, -np.inf)
                out.append((s, e, m, b))
            return out
        ra, rb = np.asarray(ra, float), np.asarray(rb, float)
        D = np.full((len(ca), len(cb)), np.inf)
        for sa, ea, ma, ba in groups(ca, ra, wa):
            for sb, eb, mb, bb in groups(cb, rb, wb):
                L = cdist(ma, mb) - ba[:, None] - bb[None]          # (+inf where an object is not held)
                near = L < below
                np.minimum(D, np.where(near, np.inf, L), out=D)
                I, J = np.nonzero(near)
                for k in range(0, len(I), chunk):
                    i, j = I[k:k + chunk], J[k:k + chunk]
                    d = (np.linalg.norm(ca[i, sa:ea][:, :, None] - cb[j, sb:eb][:, None], axis=-1)
                         - ra[sa:ea][None, :, None] - rb[sb:eb][None, None])
                    D[i, j] = np.minimum(D[i, j], d.reshape(len(i), -1).min(1))
        return D

    # ------------------------------------------------------------------ IK
    def solve(self, side, target, arm_q_seed, top_down=True, extra_seeds=(), restarts=True):
        """IK for one arm. target: grasp-point position (pelvis). arm_q_seed: (14,) full arm vector; the other
        arm's values are kept. Returns (arm_q (14,), ok, pos_err m, tilt rad)."""
        ik = self.cfg["ik"]
        seeds = [np.asarray(arm_q_seed, float)[self.sl(side)]] + [np.asarray(s, float) for s in extra_seeds]
        rng = np.random.default_rng(0)
        if restarts:
            seeds += [rng.uniform(self.lo[self.sl(side)], self.hi[self.sl(side)]) for _ in range(int(ik["random_restarts"]))]
        weights = ik["orient_weight"] if isinstance(ik["orient_weight"], (list, tuple)) else [ik["orient_weight"]]
        best = None
        for w in (weights if top_down else [0.0]):
            for k, s in enumerate(seeds):
                q7, perr, tilt = self._lsq(side, np.asarray(target, float), s, arm_q_seed, w)
                ok = perr <= ik["pos_tol"] and (not top_down or tilt <= np.radians(ik["max_tilt_deg"]))
                cost = perr + 0.01 * tilt + (0.0 if ok else 10.0)
                cost += 0.001 * np.abs(q7 - seeds[0]).sum()  # prefer solutions near the seed (continuity)
                if best is None or cost < best[0]:
                    best = (cost, q7, ok, perr, tilt)
                if ok and k == 0:
                    break  # the continuity seed converged: keep it
            if best[2]:
                break
        out = np.asarray(arm_q_seed, float).copy()
        out[self.sl(side)] = best[1]
        return out, best[2], best[3], best[4]

    def _lsq(self, side, target, q7, arm_q_full, w_o):
        """Bounded least squares (scipy TRF) on the same residuals as _dls: much more reliable than clipped DLS near
        the joint limits (a top-down grasp low in front of the body needs wrist pitch/yaw close to their limits)."""
        from scipy.optimize import least_squares
        pin, ik = self.pin, self.cfg["ik"]
        sl = self.sl(side)
        lo, hi = self.lo[sl], self.hi[sl]
        arm = np.asarray(arm_q_full, float).copy()
        q_start = np.clip(np.asarray(q7, float), lo + 1e-6, hi - 1e-6)
        w_reg = float(ik.get("seed_weight", 0.0))
        iv, fid = self.arm_iv[sl], self.palm[side]
        cache = {}

        def evaluate(x):
            key = x.tobytes()
            if key not in cache:
                arm[sl] = x
                q = self._fk(arm)
                M = self.data.oMf[fid]
                r = M.rotation @ self.offset[side]
                a = M.rotation @ self.axis[side]
                f = M.rotation[:, 0]
                J6 = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, iv]
                Jv, Jw = J6[:3], J6[3:]
                res = [M.translation + r - target]
                jac = [Jv - _skew(r) @ Jw]
                if w_o > 0:
                    res += [w_o * (a - DOWN), self.w_hint * w_o * np.cross(f, self.hint)]
                    jac += [-w_o * _skew(a) @ Jw, self.w_hint * w_o * _skew(self.hint) @ _skew(f) @ Jw]
                if w_reg > 0:
                    res.append(w_reg * (x - q_start))
                    jac.append(w_reg * np.eye(7))
                cache.clear()
                cache[key] = (np.concatenate(res), np.vstack(jac))
            return cache[key]

        def run(x0):
            return least_squares(lambda x: evaluate(x)[0], x0, jac=lambda x: evaluate(x)[1], bounds=(lo, hi),
                                 method="trf", max_nfev=int(ik["max_iters"]), xtol=1e-8, ftol=1e-10, gtol=1e-10).x
        q7 = run(q_start)
        if w_reg > 0:  # the seed pull picked the branch; now polish without it biasing the pose (keep a tiny one)
            q_start, w_reg = q7.copy(), 1e-3
            cache.clear()
            q7 = run(q7)
        arm[sl] = q7
        p, a = self.grasp_pose(side, arm)
        tilt = float(np.arccos(np.clip(a @ DOWN, -1.0, 1.0)))
        return q7, float(np.linalg.norm(target - p)), tilt

    def _dls(self, side, target, q7, arm_q_full, w_o):
        pin, ik = self.pin, self.cfg["ik"]
        sl = self.sl(side)
        lo, hi = self.lo[sl], self.hi[sl]
        arm = np.asarray(arm_q_full, float).copy()
        q7 = np.clip(q7, lo, hi)
        q_start = q7.copy()
        w_reg = float(ik.get("seed_weight", 0.0))
        iv = self.arm_iv[sl]
        fid = self.palm[side]
        lam2 = float(ik["damping"]) ** 2
        for _ in range(int(ik["max_iters"])):
            arm[sl] = q7
            q = self._fk(arm)
            M = self.data.oMf[fid]
            r = M.rotation @ self.offset[side]
            p = M.translation + r
            a = M.rotation @ self.axis[side]
            e_p = target - p
            e_o = np.cross(a, DOWN)
            J6 = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, iv]
            Jv, Jw = J6[:3], J6[3:]
            Jp = Jv - _skew(r) @ Jw
            if w_o > 0:
                f = M.rotation[:, 0]
                e_h = np.cross(f, self.hint)  # weak: fingers towards the hint direction
                J = np.vstack([Jp, w_o * Jw, self.w_hint * w_o * Jw])
                e = np.r_[e_p, w_o * e_o, self.w_hint * w_o * e_h]
            else:
                J, e = Jp, e_p
            if w_reg > 0:  # stay near the seed: no null-space drift between neighbouring Cartesian waypoints
                J = np.vstack([J, w_reg * np.eye(7)])
                e = np.r_[e, w_reg * (q_start - q7)]
            if np.linalg.norm(e_p) < 0.2 * ik["pos_tol"] and (w_o == 0 or np.linalg.norm(e_o) < 0.01):
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(len(e)), e)
            n = np.abs(dq).max()
            if n > ik["max_step"]:
                dq *= ik["max_step"] / n
            q7 = np.clip(q7 + dq, lo, hi)
        arm[sl] = q7
        p, a = self.grasp_pose(side, arm)
        tilt = float(np.arccos(np.clip(a @ DOWN, -1.0, 1.0)))
        return q7, float(np.linalg.norm(target - p)), tilt


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def cylinder_distance(c, r, cyl, z0):
    """Surface distance (m) of spheres to vertical cylinders standing on the plane z = z0: c (..., n, 3) centres,
    r (n,) radii, cyl (m, 4) rows (x, y, radius, top z). Returns (..., m): the minimum over the n spheres
    (< 0 = overlap); +inf for m = 0."""
    cyl = np.asarray(cyl, float).reshape(-1, 4)
    if not len(cyl):
        return np.full(c.shape[:-2] + (0,), np.inf)
    dxy = np.linalg.norm(c[..., None, :2] - cyl[:, :2], axis=-1)                    # (..., n, m)
    dh = np.maximum(dxy - cyl[:, 2], 0.0)
    dz = np.maximum(np.maximum(z0 - c[..., None, 2], c[..., None, 2] - cyl[:, 3]), 0.0)
    return (np.sqrt(dh * dh + dz * dz) - np.asarray(r, float)[:, None]).min(-2)
