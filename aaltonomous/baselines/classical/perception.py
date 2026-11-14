"""Head RGB-D -> coloured objects in the pelvis frame: a coloured blob, the centre of its top face (grasp target)
and, when the top face says so clearly, its shape and yaw.

    det = Perception(cfg, scene, calibration).detect(obs["head_rgb"], obs["head_depth"])
    # -> [Detection(color, top (3,) pelvis, n_px, n_depth, method, yaw, uv, shape, shape_conf, fit), ...]

Self-levelling (perception.level, _level): every frame the counter plane is fitted to the depth image (RANSAC +
trimmed least squares over pixels near the expected counter, bins excluded) and the head extrinsics are corrected
(rotation about the camera centre + height: roll, pitch, z) so that plane lies at scene.counter_z, level. Yaw and
in-plane shift are not observable from the plane and stay as calibrated. Poor fit -> calibration as is
(self.level["used"] False, ["why"]). Bin registration (perception.register, _register; profile sim only: the bins
must be fixed in the pelvis frame): then x, y and yaw from the bin walls in the levelled depth (FFT coarse search +
trimmed point-to-line Gauss-Newton), guarded likewise (self.reg). Every pixel -> pelvis conversion below uses the
corrected pose (self.T).
Geometry first (perception.geometry_first, _detect_geometry; needs depth and, by default, a levelled pose): candidate
pixels = a 3D point in the height band over the mat, outside the bin footprints -> holes filled, opening ->
connected components. Colour only classifies (nearest reference hue per pixel, saturation-weighted vote over the
blob, perception.color): a component whose colours lie an object apart is split; pixels clearly above the
component's own-colour top face are something over it (a finger) and are dropped; a component without pixels near
the counter (a top-face piece cut off by a finger, something hovering), of implausible size, or next to a bin and
wall-high or shapeless is no object; no acceptable colour -> color "unknown". Without usable depth (or levelling):
colour first, as below. self.path says which path the last frame took.
Colour first: per colour HSV threshold (config.yaml perception.hsv) -> opening -> connected components.
Per blob: erode, take the pixels with valid depth, deproject them (calibration.yaml intrinsics + the levelled
T_pelvis_optical), keep the top face (points within `top_band` of a robust max height) and use the median of those
as the top centre. With too few valid depth pixels (holes, or no depth stream) the blob centroid ray is intersected
with the plane at the known object height instead (method "plane"). Blobs over the bins, off the mat, or at a wrong
height are rejected.
Shape (perception.shape, _fit_shape): the known footprints (cube / cylinder / triangle, scene.yaml size) are fitted
to the top-face outline on the top plane; yaw is in the oracle's convention (scene.shape_polygon rot: cube face
normals at yaw + k*90 deg, triangle apexes at yaw + 90 + k*120 deg). shape None = not sure (occluded, washed out).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aaltonomous.scene.scene import shape_polygon

SHAPES =("cube", "cylinder", "triangle")


@dataclass
class Detection:
    color: str
    top: np.ndarray            # top-face centre, pelvis frame (m)
    n_px: int
    n_depth: int
    method: str                # "depth" | "plane"
    yaw: float | None = None   # rad about pelvis z, as the oracle's (scene.shape_polygon rot): cube face normals at
                               # yaw + k*90 deg, triangle apexes at yaw + 90 + k*120 deg; None = cylinder / unknown
    uv: tuple = field(default=(0.0, 0.0))
    shape: str | None = None   # "cube" | "cylinder" | "triangle"; None = evidence too weak (shape-agnostic grasp)
    shape_conf: float = 0.0    # 1 - rms(best shape) / rms(runner-up), 0..1; see perception.shape in config.yaml
    fit: dict | None = None    # footprint fit diagnostics (rms_mm per shape, shrink_mm, n_hull, ...)
    color_frac: float = 1.0    # geometry first: winner's share of the saturation-weighted colour votes ...
    n_votes: int = 0           # ... of this many classified pixels (0 = colour-first detection)

    def __repr__(self):
        x, y, z = self.top
        s = "" if self.shape is None else f" {self.shape}"
        s += "" if self.yaw is None or self.shape is None else f" {np.degrees(self.yaw):+.0f}deg"
        return (f"{self.color:<6} top=({x:+.3f},{y:+.3f},{z:+.3f}) px={self.n_px} depth_px={self.n_depth} "
                f"[{self.method}]{s}")


class Perception:
    def __init__(self, cfg, scene, calibration):
        self.cfg = cfg["perception"]
        self.scene = scene
        self.cal = calibration
        self.cam = self.cfg["camera"]
        self.size = float(self.cfg.get("object_size") or scene.size)
        self.height = float(self.cfg.get("object_height") or scene.object_height())
        # pelvis-frame xy correction added to every detection (head extrinsics error; tuned on the robot)
        self.xy_offset = np.asarray(self.cfg.get("xy_offset") or [0.0, 0.0], float)
        self.sh = self.cfg.get("shape") or {"enabled": False}
        if self.sh.get("enabled"):
            self._init_shapes()
        self.T0 = np.array(calibration._T(self.cam), float)   # calibrated head pose (eval_perception perturbs it)
        self.T = self.T0.copy()                                # pose of the current frame (levelled)
        self.lv = self.cfg.get("level") or {"enabled": False}
        self.level = {"used": False, "why": "not run"}         # last levelling result (see _level)
        self.path = None                                       # last detect(): "geometry" | "colour" | ...
        self.geo = self.cfg.get("geometry") or {}
        self._init_colors()
        self.rg = self.cfg.get("register") or {"enabled": False}
        self.reg = {"used": False, "why": "not run"}           # last bin registration result (see _register)
        if self.rg.get("enabled"):
            self._init_register()

    # ---------------------------------------------------------------- bin registration (x, y, yaw)
    def _init_register(self):
        """Bin wall centre-lines (pelvis xy segments) and, on a grid around them, a soft wall mask (1 on a line, 0
        beyond coarse_r) in the Fourier domain for the coarse search."""
        sc, G = self.scene, self.rg
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 - float(b.get("wall_thickness") or 0.0) / 2
        A, B, own = [], [], []
        for k, (cx, cy) in enumerate(b["centers"].values()):
            cm = [[cx + sx * half[0], cy + sy * half[1], 0.0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            cp = sc.mat_to_pelvis(np.array(cm))[:, :2]
            A += [cp[i] for i in range(4)]
            B += [cp[(i + 1) % 4] for i in range(4)]
            own += [k] * 4
        A, B, own = np.array(A), np.array(B), np.array(own)
        d = B - A
        self._slen = np.linalg.norm(d, axis=1)
        self._su = d / self._slen[:, None]
        self._sn = np.c_[-self._su[:, 1], self._su[:, 0]]
        # the camera sees each wall's face towards it (front walls: outer face, back walls: inner face, ...): move
        # the line half a wall thickness that way, else x / y come out biased by it
        cam = self.T0[:2, 3]
        side = np.sign(np.einsum("kd,kd->k", cam - (A + B) / 2, self._sn))
        off = (side * float(b.get("wall_thickness") or 0.0) / 2)[:, None] * self._sn
        self._sa, self._sb, self._sown = A + off, B + off, own
        self._bin_ctr = np.array([sc.mat_to_pelvis([x, y, float(b["wall_height"])]) for x, y in b["centers"].values()])
        self._bin_names = list(b["centers"])
        cell, S = float(G["cell"]), int(np.ceil(G["max_shift"] / G["cell"]))
        lo = self._sa.min(0) - G["max_shift"] - 0.03
        n = np.ceil((self._sa.max(0) + G["max_shift"] + 0.03 - lo) / cell).astype(int)
        gx, gy = np.meshgrid(lo[0] + (np.arange(n[0]) + 0.5) * cell, lo[1] + (np.arange(n[1]) + 0.5) * cell,
                             indexing="ij")
        dist, k = self._seg_dist(np.c_[gx.ravel(), gy.ravel()])
        M = np.clip(1.0 - dist.reshape(n) / G["coarse_r"], 0.0, 1.0)
        self._g = {"lo": lo, "n": n, "S": S, "N": tuple(n + S), "FM": np.fft.rfft2(M, s=tuple(n + S)),
                   "dist": np.append(dist, np.inf).astype(np.float32), "k": np.append(k, 0)}

    def _seg_dist(self, Q):
        """Points (N, 2) -> distance to the nearest wall line and its index (exact; used once at init)."""
        t = np.clip(np.einsum("nkd,kd->nk", Q[:, None] - self._sa[None], self._su), 0.0, self._slen)
        dist = np.linalg.norm(Q[:, None] - (self._sa[None] + t[..., None] * self._su[None]), axis=2)
        k = dist.argmin(1)
        return dist[np.arange(len(Q)), k], k

    def _cell(self, Q):
        """Points (N, 2) -> flat index into the registration grid (the extra last cell = outside it)."""
        g = self._g
        ij = np.floor((Q - g["lo"]) / self.rg["cell"]).astype(int)
        inside = np.all((ij >= 0) & (ij < g["n"]), axis=1)
        return np.where(inside, ij[:, 0] * g["n"][1] + ij[:, 1], int(np.prod(g["n"])))

    def _register(self, depth):
        """Camera x, y and yaw from the bin walls (the bins are fixed in the pelvis frame in the sim; NOT on the
        robot). Wall points = levelled points between h_lo and the rim (+ h_hi_margin) within max_shift + 1 cm of a
        wall. Coarse: for each yaw on a grid (about the camera's vertical axis), the xy shift maximising the
        correlation of the points' xy histogram with the soft wall mask (FFT). Fine: trimmed Gauss-Newton on the
        point-to-wall-line distances. Applied to self.T as a camera rotation about its vertical axis + xy shift."""
        G, sc = self.rg, self.scene
        out = {"used": False, "why": None, "dx_mm": 0.0, "dy_mm": 0.0, "dyaw_deg": 0.0, "n_points": 0,
               "n_inliers": 0, "rms_mm": None}
        self.reg = out
        if not G.get("enabled"):
            out["why"] = "off"
            return
        if not self._depth_usable(depth):
            out["why"] = "no depth"
            return
        if self.lv.get("enabled") and not self.level["used"]:
            out["why"] = "not levelled"
            return
        s = int(G["subsample"])
        d = depth[s // 2::s, s // 2::s].astype(float)
        dlo, dhi = self.cfg["depth_range_mm"]
        vv, uu = np.nonzero((d >= dlo) & (d <= dhi))
        P = self._to_pelvis(uu * s + s // 2, vv * s + s // 2, d[vv, uu])
        h = P[:, 2] - sc.counter_z
        ok = (h >= G["h_lo"]) & (h <= float(sc.cfg["bins"]["wall_height"]) + G["h_hi_margin"])
        Q = P[ok, :2]
        Q = Q[self._g["dist"][self._cell(Q)] <= G["max_shift"] + 0.01]
        if len(Q) > G["max_points"]:                           # evenly thinned (deterministic)
            Q = Q[np.linspace(0, len(Q) - 1, int(G["max_points"])).astype(int)]
        out["n_points"] = len(Q)
        if len(Q) < G["min_points"]:
            out["why"] = f"{len(Q)} bin wall points"
            return
        c = self.T[:2, 3]                                      # camera xy: yaw turns about it
        g = self._g
        best = (-1.0, 0.0, np.zeros(2))
        for yaw in np.radians(np.arange(-G["max_yaw_deg"], G["max_yaw_deg"] + 1e-9, G["yaw_step_deg"])):
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            H = np.bincount(self._cell((Q - c) @ R.T + c), minlength=int(np.prod(g["n"])) + 1)[:-1].reshape(g["n"])
            C = np.fft.irfft2(np.conj(np.fft.rfft2(H, s=g["N"])) * g["FM"], s=g["N"])
            S = g["S"]
            C = np.roll(np.roll(C, S, 0), S, 1)[:2 * S + 1, :2 * S + 1]   # shifts -S..S
            a = np.unravel_index(int(C.argmax()), C.shape)
            if C[a] > best[0]:
                best = (float(C[a]), yaw, (np.array(a) - S) * G["cell"])
        _, th, dxy = best
        thr = float(G["inlier_thr"])
        for it in range(int(G["iters"])):                      # trimmed Gauss-Newton, point-to-line
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            X = (Q - c) @ R.T
            Y = X + c + dxy
            k = g["k"][self._cell(Y)]                          # nearest wall line (3-4 mm grid lookup)
            t = np.einsum("nd,nd->n", Y - self._sa[k], self._su[k])
            r = np.einsum("nd,nd->n", Y - self._sa[k], self._sn[k])
            inl = (np.abs(r) < thr) & (t > -0.005) & (t < self._slen[k] + 0.005)
            if inl.sum() < 10:
                break
            n = self._sn[k[inl]]
            J = np.c_[n, n[:, 0] * -X[inl, 1] + n[:, 1] * X[inl, 0]]
            step = np.linalg.lstsq(J, -r[inl], rcond=None)[0]
            dxy, th = dxy + step[:2], th + step[2]
            thr = max(G["trim_k"] * 1.4826 * np.median(np.abs(r[inl])), G["min_thr"])
        rms = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else np.inf
        out.update(dx_mm=round(1000 * float(dxy[0]), 2), dy_mm=round(1000 * float(dxy[1]), 2),
                   dyaw_deg=round(float(np.degrees(th)), 3), n_inliers=int(inl.sum()), rms_mm=round(1000 * rms, 2))
        xf = np.abs(self._sn[k[inl], 0]) > 0.7                # inliers on x-facing walls (front / back)
        per_dir = (int(xf.sum()), int((~xf).sum()))
        seen = self._project(self._bin_ctr)                    # bins whose rim centre is in the image must show
        W, H_ = self.cal.cams[self.cam].width, self.cal.cams[self.cam].height   # both wall directions
        own = self._sown[k[inl]]
        per_bin = {nm: (int((xf & (own == j)).sum()), int((~xf & (own == j)).sum()))
                   for j, nm in enumerate(self._bin_names)}
        out["per_bin"] = per_bin
        lost = [nm for j, nm in enumerate(self._bin_names) if seen[j, 2] > 0 and 0 <= seen[j, 0] < W
                and 0 <= seen[j, 1] < H_ and min(per_bin[nm]) < G["min_bin_points"]]
        if inl.sum() < G["min_points"] or inl.mean() < G["min_inlier_frac"]:
            out["why"] = f"{inl.sum()} inliers ({inl.mean():.2f})"
        elif lost:
            out["why"] = f"bin(s) {lost} not where expected"
        elif min(per_dir) < G["min_dir_points"]:
            out["why"] = f"walls of one direction only {per_dir}"
        elif rms > G["max_rms"]:
            out["why"] = f"rms {1000 * rms:.1f} mm"
        elif np.linalg.norm(dxy) > G["max_shift"] or abs(np.degrees(th)) > G["max_yaw_deg"]:
            out["why"] = f"correction too large ({1000 * np.linalg.norm(dxy):.0f} mm, {np.degrees(th):.2f} deg)"
        else:
            T = self.T.copy()
            Rz = np.eye(3)
            Rz[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
            T[:3, :3] = Rz @ self.T[:3, :3]
            T[:2, 3] += dxy
            self.T = T
            out["used"] = True

    def _init_colors(self):
        """Reference hues (OpenCV 0-180) of the object classes and of known non-objects (bins)."""
        import cv2

        def hue(rgb):
            return float(cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0, 0])
        c, sc = self.cfg.get("color") or {}, self.scene
        refs = c.get("refs") or {k: hue(v) for k, v in sc.colors.items()}
        rej = c.get("reject")
        rej = [hue(sc.cfg["bins"]["rgb"])] if rej is None else list(rej)
        self.col_names = list(refs)
        self._hues = np.array([float(refs[k]) for k in refs] + [float(h) for h in rej], np.float32)
        self._n_cls = len(refs)                                # classes >= _n_cls are "reject" (not an object)

    # ---------------------------------------------------------------- camera pose (levelled)
    def _to_pelvis(self, u, v, d):
        p = self.cal.deproject(self.cam, u, v, d)
        return p @ self.T[:3, :3].T + self.T[:3, 3]

    def _project(self, P):
        """Pelvis-frame points (..., 3) -> (u, v, z_optical) with the current pose."""
        K = self.cal.cams[self.cam].K
        p = (np.asarray(P, float) - self.T[:3, 3]) @ self.T[:3, :3]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack([K[0, 0] * p[..., 0] / p[..., 2] + K[0, 2], K[1, 1] * p[..., 1] / p[..., 2] + K[1, 2],
                             p[..., 2]], -1)

    def _level(self, depth):
        """Fit the counter plane on a subsampled depth grid: pixels whose calibrated 3D point is within level.band of
        the expected counter, over the mat, bins excluded. In the optical frame a plane n.p = d is linear in inverse
        depth, 1/z = (n / d) . (x', y', 1) with x', y' the normalised pixel coordinates, and depth noise lies along
        the ray: fitting 1/z (RANSAC, then trimmed least squares on the depth residual) has no errors-in-variables
        bias (a z(x, y) fit in the pelvis frame tilts by ~0.2 deg under D435-like noise). Correction: rotate the
        camera about its own centre so the plane normal is vertical (roll, pitch), shift it in z so the plane is at
        counter_z. Sets self.T and self.level."""
        L, sc = self.lv, self.scene
        self.T = self.T0
        out = {"used": False, "why": None, "roll_deg": 0.0, "pitch_deg": 0.0, "dz_mm": 0.0, "n_inliers": 0,
               "inlier_frac": 0.0, "rms_mm": None}
        self.level = out
        if not L.get("enabled"):
            out["why"] = "off"
            return
        if depth is None:
            out["why"] = "no depth"
            return
        s = int(L["subsample"])
        d = depth[s // 2::s, s // 2::s].astype(float)
        dlo, dhi = self.cfg["depth_range_mm"]
        vv, uu = np.nonzero((d >= dlo) & (d <= dhi))
        P = self._to_pelvis(uu * s + s // 2, vv * s + s // 2, d[vv, uu])
        # region (mat, bins) from where each pixel's ray meets the expected counter, not from its noisy 3D point:
        # selecting by noisy coordinates truncates the noise at the region edges and tilts the fit
        pm = sc.pelvis_to_mat(np.c_[self._on_plane(np.c_[uu * s + s // 2, vv * s + s // 2].astype(float),
                                                   sc.counter_z), np.zeros(len(uu))])
        W, D = sc.cfg["counter"]["mat_size"]
        m = float(L["region_margin"])
        sel = ((np.abs(P[:, 2] - sc.counter_z) <= L["band"]) & (pm[:, 0] >= m) & (pm[:, 0] <= W - m)
               & (pm[:, 1] >= m) & (pm[:, 1] <= D - m))
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + L["bin_margin"]
        for ctr in sc.cfg["bins"]["centers"].values():
            sel &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
        if sel.sum() < L["min_inliers"]:
            out["why"] = f"{sel.sum()} counter pixels"
            return
        K = self.cal.cams[self.cam].K
        z = d[vv[sel], uu[sel]] / 1000.0
        A = np.c_[(uu[sel] * s + s // 2 - K[0, 2]) / K[0, 0], (vv[sel] * s + s // 2 - K[1, 2]) / K[1, 1],
                  np.ones(len(z))]
        rng = np.random.default_rng(0)                         # deterministic: same frame, same answer
        tri = rng.integers(0, len(z), (int(L["ransac_iters"]), 3))
        M = A[tri]
        ok = np.abs(np.linalg.det(M)) > 1e-9                   # (no collinear triples)
        coef = np.linalg.solve(M[ok], 1.0 / z[tri[ok]][..., None])[..., 0]
        with np.errstate(divide="ignore"):
            res = np.abs(z[None] - 1.0 / (coef @ A.T))         # depth residuals (m)
        thr = float(L["inlier_thr"])
        best = coef[np.argmax((res < thr).sum(1))]
        inl = np.abs(z - 1.0 / (A @ best)) < thr
        for _ in range(3):                                     # trimmed LS, band from the inliers' own spread;
            w = 1.0 / (A[inl] @ best) ** 2                     # 1/z residual = depth residual / z^2 (z from the
            best = np.linalg.lstsq(A[inl] * w[:, None], w / z[inl], rcond=None)[0]   # model: noisy z would bias)
            r = z - 1.0 / (A @ best)
            sd = 1.4826 * np.median(np.abs(r[inl]))
            inl = np.abs(r) < max(L["trim_k"] * sd, L["min_thr"])
        n = self.T0[:3, :3] @ best                             # plane normal / d (optical) -> pelvis axes
        h_cam = 1.0 / np.linalg.norm(n)                        # camera distance to the plane
        n = -n * h_cam                                         # unit normal pointing up (the camera looks down)
        ax = np.cross(n, [0.0, 0.0, 1.0])
        ang = float(np.arctan2(np.linalg.norm(ax), n[2]))
        rv = ax / max(np.linalg.norm(ax), 1e-12) * ang                    # rotation taking n to +z
        dz = sc.counter_z - (self.T0[2, 3] - h_cam)
        out.update(roll_deg=round(float(np.degrees(rv[0])), 3), pitch_deg=round(float(np.degrees(rv[1])), 3),
                   dz_mm=round(1000 * float(dz), 2), n_inliers=int(inl.sum()),
                   inlier_frac=round(float(inl.mean()), 3), rms_mm=round(1000 * float(r[inl].std()), 2))
        if inl.sum() < L["min_inliers"] or inl.mean() < L["min_inlier_frac"]:
            out["why"] = f"{inl.sum()} inliers ({inl.mean():.2f})"
        elif np.degrees(ang) > L["max_tilt_deg"] or abs(dz) > L["max_dz"]:
            out["why"] = f"correction too large ({np.degrees(ang):.2f} deg, {1000 * dz:.0f} mm)"
        elif r[inl].std() > L["max_rms"]:
            out["why"] = f"rms {1000 * r[inl].std():.1f} mm"
        else:
            from scipy.spatial.transform import Rotation
            T = self.T0.copy()
            T[:3, :3] = Rotation.from_rotvec(rv).as_matrix() @ self.T0[:3, :3]
            T[2, 3] += dz
            self.T = T
            out["used"] = True

    def _init_shapes(self):
        """Support functions h(phi) = max over the footprint of p . (cos phi, sin phi) of every shape model: cube
        every 1 deg over 90, triangle over 120, cylinder once (fillets rounded: polygon at apothem - r, then + r)."""
        sc, s = self.scene, self.sh
        size = float(self.cfg.get("object_size") or sc.size)
        phi = np.radians(np.arange(360.0))
        self._U = np.stack([np.cos(phi), np.sin(phi)])                         # (2, 360)
        models, H, self._area_model, self._apothem = [], [], {}, {}
        for shape, period in (("cube", 90), ("triangle", 120), ("cylinder", 1)):
            dims = sc.object_dims(shape)
            a, r = dims["apothem"] * size / sc.size, min(dims["fillet"], 0.9 * dims["apothem"])
            self._apothem[shape] = a
            poly = shape_polygon(shape, a - r)          # rounded footprint area = inner area + perimeter * r + pi r^2
            per = np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1).sum()
            self._area_model[shape] = self._area(poly) + per * r + np.pi * r * r
            for deg in range(period):
                if shape == "cylinder":
                    H.append(np.full(360, a))
                else:
                    H.append((shape_polygon(shape, a - r, np.radians(deg)) @ self._U).max(0) + r)
                models.append((shape, np.radians(deg)))
        self._models, self._H = models, np.array(H)
        self._kind = np.array([m[0] for m in models])
        self._A = np.c_[self._U.T, -np.ones(360)]                              # h_obs - h_model = cx cos + cy sin - e
        self._AA = (self._A[:, :, None] * self._A[:, None, :]).reshape(360, 9)
        self._lam2 = (float(s["rms_ref"]) / float(s["shrink_tol"])) ** 2      # prior on e, per unit direction weight

    def _solve(self, R, W):
        """Weighted least squares for every model at once. R (models, 360) = observed - model support, W weights.
        Unknowns x = (cx, cy, e): R ~ cx cos + cy sin - e, plus the prior sum(W) * lam2 * (e - shrink)^2."""
        A, ws = self._A, W.sum(1)
        M = (W @ self._AA).reshape(-1, 3, 3)
        M[:, 2, 2] += ws * self._lam2
        b = (W * R) @ A
        b[:, 2] += ws * self._lam2 * float(self.sh["shrink"])
        X = np.linalg.solve(M, b[..., None])[..., 0]
        res = R - X @ A.T
        return X, np.sqrt((W * res ** 2).sum(1) / np.maximum(ws, 1e-9)), res

    def masks(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        out = {}
        for color, ranges in self.cfg["hsv"].items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            k = int(self.cfg["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            out[color] = m
        return out

    def detect(self, rgb, depth, debug=False):
        import cv2
        c = self.cfg
        dets, rejected = [], []
        self._level(depth)
        self._register(depth)
        # xy_offset is tuned for the pose it is added to: registration enabled but refused on this frame -> the offset
        # measured for the levelled-only pose (register.fallback_xy_offset), if given
        fb = self.rg.get("fallback_xy_offset") if self.rg.get("enabled") and not self.reg["used"] else None
        self.xy_offset = np.asarray(self.cfg.get("xy_offset") or [0.0, 0.0] if fb is None else fb, float)
        self.path = "colour"
        if c.get("geometry_first") and self._depth_usable(depth):
            if self.level["used"] or not (self.lv.get("enabled") and self.geo.get("need_level", True)):
                self.path = "geometry"
                self._detect_geometry(rgb, depth, dets, rejected)
                return (dets, rejected) if debug else dets
            self.path = "colour (not levelled)"
        keep = self._keep_mask(depth)
        for color, m in self.masks(rgb).items():
            m = m & keep  # drop bin / counter pixels BEFORE grouping, so an object touching a same-colour bin survives
            k = int(c["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= c["min_area_px"]:
                    self._measure(color, lab, i, stats[i], cent[i], depth, dets, rejected)
        return (dets, rejected) if debug else dets

    def _depth_usable(self, depth):
        if depth is None:
            return False
        dlo, dhi = self.cfg["depth_range_mm"]
        return float(((depth >= dlo) & (depth <= dhi)).mean()) >= self.cfg["min_valid_frac"]

    def _measure(self, color, lab, i, st, cent, depth, dets, rejected, extra=None, need_shape=False):
        """Blob i of `lab` -> Detection (top centre from depth, else ray / plane), sanity checks, shape fit
        (need_shape: no accepted shape -> rejected)."""
        import cv2
        c = self.cfg
        dlo, dhi = c["depth_range_mm"]
        area = int(st[cv2.CC_STAT_AREA])
        e = int(c["erode_px"])
        x, y, w, h = (int(t) for t in st[:4])          # erode in a window around the blob (same result, faster)
        y0, x0 = max(y - e - 1, 0), max(x - e - 1, 0)
        blob = (lab[y0:y + h + e + 1, x0:x + w + e + 1] == i).astype(np.uint8)
        core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
        if core.sum() < 0.25 * area:
            core = blob
        v, u = np.nonzero(core)
        v, u = v + y0, u + x0
        d = depth[v, u].astype(float) if depth is not None else np.zeros(len(u))
        ok = (d >= dlo) & (d <= dhi)
        det = None
        if ok.sum() >= c["min_valid_depth_px"]:
            det = self._from_depth(color, u[ok], v[ok], d[ok], area)
        if det is None:
            det = self._from_plane(color, cent, area, int(ok.sum()))
        for k, val in (extra or {}).items():
            setattr(det, k, val)
        why = self._reject(det)
        if why:
            rejected.append((det, why))
            return
        if det.method == "depth" and self.sh.get("enabled"):
            self._fit_shape(det, lab, i, st, depth)
            if need_shape and det.shape is None:
                rejected.append((det, "next to a bin and no object shape"))
                return
        dets.append(det)

    # ---------------------------------------------------------------- geometry first
    def _classify(self, hsv):
        """(N, 3) HSV pixels -> class index (N,): 0.. = colour class, >= _n_cls = reject hue, -1 = none (grey, dark,
        or no reference within max_dist)."""
        c = self.cfg["color"]
        hh = hsv[:, 0].astype(np.float32)
        dh = np.abs(hh[:, None] - self._hues[None])
        dh = np.minimum(dh, 180.0 - dh)
        dh[:, self._n_cls:] += c["reject_margin"]               # a reject hue must be clearly nearer than a class
        k = dh.argmin(1)
        ok = (hsv[:, 1] >= c["sat_min"]) & (hsv[:, 2] >= c["val_min"]) & (dh[np.arange(len(k)), k] <= c["max_dist"])
        return np.where(ok, k, -1)

    def _vote(self, cls, sat):
        """Class votes weighted by saturation (washed-out pixels count less) -> (colour name or 'unknown', winner's
        share of the vote, n classified pixels)."""
        ok = cls >= 0
        if not ok.any():
            return "unknown", 0.0, 0
        w = np.bincount(cls[ok], weights=sat[ok].astype(float), minlength=len(self._hues))
        k = int(w.argmax())
        name = self.col_names[k] if k < self._n_cls else "unknown"
        return name, float(w[k] / w.sum()), int(ok.sum())

    def _detect_geometry(self, rgb, depth, dets, rejected):
        import cv2
        c, g, cc, sc = self.cfg, self.geo, self.cfg["color"], self.scene
        Hh, Ww = depth.shape
        dlo, dhi = c["depth_range_mm"]
        df = depth.astype(np.float32)
        valid = (df >= dlo) & (df <= dhi)
        vv, uu = np.nonzero(valid)
        P = self._to_pelvis(uu, vv, df[vv, uu])
        hm = np.full((Hh, Ww), np.nan, np.float32)               # height above the counter (m), NaN = no depth
        hm[vv, uu] = P[:, 2] - sc.counter_z
        pm = sc.pelvis_to_mat(P)
        mxy = np.zeros((Hh, Ww, 2), np.float32)                   # mat xy per pixel (blob centres vs the bins)
        mxy[vv, uu] = pm[:, :2]
        lo, hi = np.asarray(g["height_band"], float) * self.height
        W, D = sc.cfg["counter"]["mat_size"]
        m = float(g["mat_margin"])
        ok = ((hm[vv, uu] >= lo) & (hm[vv, uu] <= hi) & (pm[:, 0] >= -m) & (pm[:, 0] <= W + m)
              & (pm[:, 1] >= -m) & (pm[:, 1] <= D + m))
        bins = np.array(list(sc.cfg["bins"]["centers"].values()), float)
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2
        for ctr in bins:
            ok &= ~np.all(np.abs(pm[:, :2] - ctr) <= half + g["bin_margin"], axis=1)
        # top higher than an object's, next to a bin: a piece of its wall (in-plane camera error moved it outside)
        z_wall = self.height + 0.5 * (float(sc.cfg["bins"]["wall_height"]) - self.height)
        cand = np.zeros((Hh, Ww), np.uint8)
        cand[vv[ok], uu[ok]] = 1
        k = int(g["fill_px"])
        if k > 0:                      # depth holes inside a candidate belong to it (only holes: no bridging)
            cand |= cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) & ~valid
        k = int(c["open_px"])
        if k > 0:
            cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        cv_, cu_ = np.nonzero(cand)
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        cls = np.full((Hh, Ww), -1, np.int16)
        cls[cv_, cu_] = self._classify(hsv[cv_, cu_])
        blobs = []                     # (window slice, bool mask in window, class hint, split off another colour)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < c["min_area_px"]:
                continue
            x, y, w, h = (int(t) for t in stats[i, :4])
            win = np.s_[y:y + h, x:x + w]
            comp = lab[win] == i
            cw = cls[win]
            cnt = np.bincount(cw[comp & (cw >= 0)], minlength=len(self._hues))[:self._n_cls]
            pres = list(np.flatnonzero(cnt >= max(cc["split_px"], cc["split_frac"] * cnt.sum())))
            if len(pres) >= 2:             # two objects: their colours' points lie ~ one size apart on the counter;
                mw, vw = mxy[win], valid[win]   # one object with a washed-out top: top vs side face, ~ half a size
                pts = [mw[comp & (cw == j) & vw] for j in pres]
                if all(len(p) for p in pts):
                    xy = [np.median(p, 0) for p in pts]
                    sep = min(np.linalg.norm(p - q) for a, p in enumerate(xy) for q in xy[a + 1:])
                    if sep < cc["split_dist"] * self.size:
                        pres = [pres[int(np.argmax(cnt[pres]))]]
            if len(pres) < 2:
                blobs.append((win, comp, pres[0] if pres else None, False))
                continue
            # touching objects of different colours: every pixel goes to the nearest pixel of a present colour
            dist = np.stack([cv2.distanceTransform((~(comp & (cw == j))).astype(np.uint8), cv2.DIST_L2, 3)
                             for j in pres])
            near = dist.argmin(0)
            for q, j in enumerate(pres):
                sub = (comp & (near == q)).astype(np.uint8)
                ns, sl = cv2.connectedComponents(sub, connectivity=8)
                for s in range(1, ns):
                    if (sl == s).sum() >= c["min_area_px"]:
                        blobs.append((win, sl == s, j, True))
        glab = np.zeros((Hh, Ww), np.int32)
        f2 = self.cal.cams[self.cam].K[0, 0] * self.cal.cams[self.cam].K[1, 1]
        for win, b, j, split in blobs:
            hw = hm[win]
            if j is not None:          # drop what is above this object's own-colour top face (finger, hand), and
                hc = hw[b & (cls[win] == j) & np.isfinite(hw)]   # dark pixels at top-face level (a black fingertip
                if len(hc) >= 10:                                 # resting at the top's height; a lit top is bright)
                    ztop = np.percentile(hc, c["top_percentile"])
                    b = b & ~(hw > ztop + g["occluder_dz"]) & ~((hw > ztop - c["top_band"]) &
                                                                 (hsv[win][..., 2] < cc["val_min"]))
                    nb, bl, bst, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
                    if nb < 2:
                        continue
                    b = bl == 1 + int(np.argmax(bst[1:, cv2.CC_STAT_AREA]))   # keep the largest part
            hb = hw[b]
            fin = np.isfinite(hb)
            dd = df[win][b][fin]
            if not fin.any():
                continue
            # fronto-parallel equivalent area in units of size^2 (one object ~1-1.5: top face + near side face)
            area = float(b.sum()) * float(np.median(dd)) ** 2 / 1e6 / f2 / self.size ** 2
            low = int((hb[fin] <= g["support_h"] * self.height).sum())
            ztop = np.percentile(hb[fin], c["top_percentile"])
            gap = (np.abs(np.median(mxy[win][b][fin], 0) - bins) - half).max(1).min()   # to the nearest bin, m
            why = (f"floating ({low} px near the counter)" if low < g["support_min_px"] and not split else
                   f"size {area:.2f} object areas" if not g["area"][0] <= area <= g["area"][1] else
                   f"bin wall ({1000 * ztop:.0f} mm high, {1000 * gap:.0f} mm from a bin)"
                   if ztop > z_wall and gap < g["bin_near"] else None)
            # colour: saturation-weighted votes of the whole blob (vote "all"), or of the top face (robust top -
            # top_band; vote "top") with the whole blob as fallback
            cb, sb = cls[win][b], hsv[win][..., 1][b]
            nv = 0
            if cc["vote"] == "top":
                topf = np.zeros(len(hb), bool)
                topf[fin] = hb[fin] >= ztop - c["top_band"]
                color, frac, nv = self._vote(cb[topf], sb[topf])
            if nv < cc["min_votes"]:
                color, frac, nv = self._vote(cb, sb)
            if nv < cc["min_votes"]:
                color, frac, nv = "unknown", 0.0, nv
            elif frac < cc["min_frac"]:
                color = "unknown"
            y0, x0 = win[0].start, win[1].start
            bv, bu = np.nonzero(b)
            if why:
                Pb = self._to_pelvis(bu[fin] + x0, bv[fin] + y0, dd)
                rejected.append((Detection(color, np.median(Pb, 0), int(b.sum()), int(fin.sum()), "geometry"), why))
                continue
            j2 = len(dets) + len(rejected) + 1
            glab[bv + y0, bu + x0] = j2
            st = np.array([bu.min() + x0, bv.min() + y0, bu.max() - bu.min() + 1, bv.max() - bv.min() + 1, b.sum()])
            self._measure(color, glab, j2, st, np.array([bu.mean() + x0, bv.mean() + y0]), depth, dets, rejected,
                          {"color_frac": round(frac, 3), "n_votes": nv}, need_shape=gap < g["bin_near"])

    def _keep_mask(self, depth):
        """uint8 mask of pixels that may belong to an object: no valid depth (unknown), or a 3D point at object
        height over the mat and outside every bin footprint."""
        c, sc = self.cfg, self.scene
        keep = self._bin_hulls_mask(depth.shape if depth is not None else (480, 640))
        if depth is None:
            return keep
        dlo, dhi = c["depth_range_mm"]
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi)
        if valid.mean() < c["min_valid_frac"]:
            return keep  # (almost) no depth stream: only the projected bin hulls can be removed
        keep[:] = 255    # with depth: holes stay "unknown" (kept); valid pixels are judged in 3D
        vv, uu = np.nonzero(valid)
        P = self._to_pelvis(uu, vv, d[vv, uu])
        h = (P[:, 2] - sc.counter_z) / self.height
        pm = sc.pelvis_to_mat(P)
        ok = (h >= 0.25) & (h <= c["height_range"][1])
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
        for ctr in sc.cfg["bins"]["centers"].values():
            ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
        keep[vv[~ok], uu[~ok]] = 0
        return keep

    def _bin_hulls_mask(self, shape):
        """255 everywhere except inside the image hull of each bin box (footprint + margin, counter to rim)."""
        import cv2
        sc = self.scene
        keep = np.full(shape, 255, np.uint8)
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"]
        for cx, cy in b["centers"].values():
            corners = np.array([[cx + sx * half[0], cy + sy * half[1], z] for sx in (-1, 1) for sy in (-1, 1)
                                for z in (0.0, float(b["wall_height"]))])
            uvz = self._project(sc.mat_to_pelvis(corners))
            if np.any(uvz[:, 2] <= 0):
                continue
            hull = cv2.convexHull(np.round(uvz[:, :2]).astype(np.int32))
            cv2.fillConvexPoly(keep, hull, 0)
        return keep

    def _from_depth(self, color, u, v, d, area):
        P = self._to_pelvis(u, v, d)
        z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
        top = P[P[:, 2] >= z_top - self.cfg["top_band"]]
        if len(top) < 5:
            return None
        # pixels sample the top face unevenly (perspective): weight each by its footprint ~ depth^2
        w = (d[P[:, 2] >= z_top - self.cfg["top_band"]] / 1000.0) ** 2
        xy = (top[:, :2] * w[:, None]).sum(0) / w.sum()
        z = float(np.median(top[:, 2]))
        xy = xy + self.xy_offset
        return Detection(color, np.array([xy[0], xy[1], z]), area, len(u), "depth", _yaw(top[:, :2]),
                         (float(np.median(u)), float(np.median(v))))

    def _fit_shape(self, det, lab, i, st, depth):
        """Shape, yaw and footprint centre of blob i (sets det.shape / yaw / shape_conf / fit; det.top xy too if
        shape.centre). Top face = blob pixels whose box-smoothed depth puts them within shape.band of the top plane;
        the convex hull of their outline is intersected with that plane (xy free of depth noise; the image outline
        would include the near side face) and its support function in 360 directions is least-squares fitted by
        every model footprint (free centre + a uniform shrink kept near shape.shrink). Best rms wins.
        Occlusion (sim home pose: fingertips over the far side objects): directions whose support point lies next to
        something higher than the top face are censored (the face may go on under it): they count only where the
        outline pokes out of the model."""
        import cv2
        s = self.sh
        det.yaw = None                 # the minAreaRect angle is not in the oracle's convention
        x, y, w, h = (int(t) for t in st[:4])
        k, ko = int(s["smooth_px"]) | 1, int(s["occluder_px"])
        p = k + ko + 1
        x0, y0 = max(x - p, 0), max(y - p, 0)
        win = np.s_[y0:y + h + p, x0:x + w + p]
        draw = depth[win].astype(np.float32)
        dlo, dhi = self.cfg["depth_range_mm"]
        valid = (draw >= dlo) & (draw <= dhi)
        blob = lab[win] == i
        m = blob & valid
        if m.sum() < s["min_px"]:
            return
        dd, mc = draw, m
        if k > 1:                      # mean over the blob's own valid pixels (no counter / holes mixed in);
            wm = m.astype(np.float32)  # depth holes inside the blob get their neighbours' mean
            den = cv2.blur(wm, (k, k))
            dd = cv2.blur(draw * wm, (k, k)) / np.maximum(den, 1e-6)
            mc = blob & (den >= 0.3)
        vv, uu = np.nonzero(mc)
        z = self._to_pelvis(uu + x0, vv + y0, dd[vv, uu])[:, 2]
        zt = np.percentile(z, self.cfg["top_percentile"])
        zp = float(np.median(z[z >= zt - self.cfg["top_band"]]))
        sel = z >= zp - s["band"]
        top = np.zeros(m.shape, np.uint8)
        top[vv[sel], uu[sel]] = 1
        top_raw = top > 0
        if s["open_px"] > 0:
            top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((s["open_px"], s["open_px"]), np.uint8))
        cs, _ = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return
        ct = max(cs, key=cv2.contourArea)[:, 0, :]
        hull = cv2.convexHull(ct)[:, 0, :]
        if len(hull) < 3:
            return
        occ = np.zeros(m.shape, np.uint8)  # occluders: anything outside the blob higher than its top face
        out = valid & ~blob
        ov, ou = np.nonzero(out)
        if len(ou):
            do = draw
            if k > 1:                  # smoothed like the blob (noise), over the outside pixels only
                wo = out.astype(np.float32)
                do = cv2.blur(draw * wo, (k, k)) / np.maximum(cv2.blur(wo, (k, k)), 1e-6)
            zo = self._to_pelvis(ou + x0, ov + y0, do[ov, ou])[:, 2]
            hi = zo > zp + s["occluder_dz"]
            occ[ov[hi], ou[hi]] = 1
        if ko > 0:
            occ = cv2.dilate(occ, np.ones((2 * ko + 1, 2 * ko + 1), np.uint8))
        Q = self._on_plane(hull.astype(float) + [x0, y0], zp)
        hq = Q @ self._U
        arg = hq.argmax(0)
        hmax = hq[arg, np.arange(360)]
        vfree = occ[hull[:, 1], hull[:, 0]] == 0
        cens = ~vfree[arg]                                                   # support point next to an occluder
        if cens.mean() > 1.0 - s["min_free"]:
            det.fit = {"free": round(float(1.0 - cens.mean()), 3), "best": None}
            return
        R = hmax[None, :] - self._H                                          # observed - model support, (models, 360)
        W = np.broadcast_to((~cens).astype(float), R.shape)
        X, rms, res = self._solve(R, W)
        if cens.any():                 # censored directions: count where the outline is outside the fitted model
            X, rms, res = self._solve(R, np.where(cens & (res > 0), 1.0, W))
        best = {kd: int(np.flatnonzero(self._kind == kd)[np.argmin(rms[self._kind == kd])]) for kd in SHAPES}
        order = sorted(SHAPES, key=lambda kd: rms[best[kd]])
        j, j2 = best[order[0]], best[order[1]]
        shape, th = self._models[j]
        xc = X[j]
        if shape != "cylinder":        # sub-degree yaw: parabola through the neighbouring grid angles (cyclic)
            idx = np.flatnonzero(self._kind == shape)
            n, jj = len(idx), j - idx[0]
            rm, r0, rp = rms[idx[(jj - 1) % n]], rms[j], rms[idx[(jj + 1) % n]]
            den = rm - 2 * r0 + rp
            dl = float(np.clip(0.5 * (rm - rp) / den, -0.5, 0.5)) if den > 1e-12 else 0.0
            xc = xc + abs(dl) * (X[idx[(jj + (1 if dl > 0 else -1)) % n]] - xc)
            th = th + np.radians(dl)
        area = self._area(self._on_plane(ct.astype(float) + [x0, y0], zp))
        fill = area / self._area_model[shape]
        conf = float(1.0 - rms[j] / max(rms[j2], 1e-9))
        free = float(1.0 - cens.mean())
        known = valid & ~top_raw & (occ == 0)                                # seen, and not top face
        bad = {kd: self._disagree(kd, self._models[best[kd]][1], X[best[kd]], zp, top, known, x0, y0) for kd in SHAPES}
        det.fit = {"rms_mm": {kd: round(1000 * float(rms[best[kd]]), 2) for kd in SHAPES},
                   "bad": {kd: round(v, 3) for kd, v in bad.items()},
                   "best": shape, "shrink_mm": round(1000 * float(xc[2]), 2), "fill": round(float(fill), 3),
                   "free": round(free, 3), "n_hull": len(hull), "xy": xc[:2].copy(), "z_plane": zp,
                   "hull_uv": hull + [x0, y0], "hull_free": vfree}
        det.shape_conf = conf
        if (rms[j] > s["max_rms"] or conf < s["min_conf"] or free < s["min_free"]
                or not s["fill"][0] <= fill <= s["fill"][1] or bad[shape] > s["max_bad"]
                or bad[shape] > min(v for kd, v in bad.items() if kd != shape)):   # the pixels prefer another shape
            return
        det.shape = shape
        if shape != "cylinder":
            per = np.pi / 2 if shape == "cube" else 2 * np.pi / 3
            det.yaw = float((th + per / 2) % per - per / 2)
        if s.get("centre"):
            det.top = np.array([xc[0] + self.xy_offset[0], xc[1] + self.xy_offset[1], det.top[2]])

    def _disagree(self, shape, th, x, zp, top, known, x0, y0):
        """Fraction of a model footprint's pixel area contradicted by the image: top-face pixels outside it (grown
        2 px) + pixels seen to be something else (side face, counter) inside it (shrunk 2 px). Occluded = no vote."""
        import cv2
        poly = shape_polygon(shape, self._apothem[shape] - x[2], th) + x[:2]
        uv = self._project(np.c_[poly, np.full(len(poly), zp)])[:, :2] - [x0, y0]
        M = np.zeros(top.shape, np.uint8)
        cv2.fillPoly(M, [np.round(uv).astype(np.int32)], 1)
        ker = np.ones((5, 5), np.uint8)
        out = ((top > 0) & (cv2.dilate(M, ker) == 0)).sum() + (known & (cv2.erode(M, ker) > 0)).sum()
        return float(out / max(int(M.sum()), 1))

    def _on_plane(self, uv, z):
        """Pixels (N, 2) -> pelvis xy (N, 2) where their rays cross the horizontal plane at height z."""
        T, K = self.T, self.cal.cams[self.cam].K
        ray = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))] @ T[:3, :3].T
        return T[:2, 3] + ((z - T[2, 3]) / ray[:, 2])[:, None] * ray[:, :2]

    @staticmethod
    def _area(p):
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))

    def _from_plane(self, color, uv, area, n_depth):
        """Centroid ray of the blob intersected with the horizontal plane at the object's top."""
        T = self.T
        K = self.cal.cams[self.cam].K
        ray_opt = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
        ray = T[:3, :3] @ ray_opt
        o = T[:3, 3]
        z_top = self.scene.counter_z + self.height
        # the blob centroid lies between top face and front face: aim a bit below the top
        z_aim = z_top - 0.25 * self.height
        if ray[2] >= -1e-6:
            return Detection(color, np.array([np.nan, np.nan, np.nan]), area, n_depth, "plane")
        s = (z_aim - o[2]) / ray[2]
        p = o + s * ray
        p[:2] += self.xy_offset
        return Detection(color, np.array([p[0], p[1], z_top]), area, n_depth, "plane", None, (float(uv[0]), float(uv[1])))

    def _reject(self, det):
        if det is None or not np.all(np.isfinite(det.top)):
            return "no_3d"
        c, sc = self.cfg, self.scene
        h = (det.top[2] - sc.counter_z) / self.height
        lo, hi = c["height_range"]
        if not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        pm = sc.pelvis_to_mat(det.top)
        W, D = sc.cfg["counter"]["mat_size"]
        m = c["counter_margin"]
        if not (-m <= pm[0] <= W + m and -m <= pm[1] <= D + m):
            return "off_mat"
        bm = c["bin_exclusion_margin"]
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + bm
        for name, ctr in sc.cfg["bins"]["centers"].items():
            if np.all(np.abs(pm[:2] - np.asarray(ctr, float)) <= half):
                return f"in_bin:{name}"
        return None


def _yaw(xy):
    """Principal direction of the top-face points, folded to [-pi/4, pi/4) (square symmetry). Info only."""
    if len(xy) < 10:
        return None
    import cv2
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    a = np.radians(ang)
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
