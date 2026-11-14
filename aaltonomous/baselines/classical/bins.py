"""The three boxes (bins) from the head RGB-D, every look, fused over the episode (config perception.boxes).

    det = BoxDetector(cfg["perception"]["boxes"], scene)
    res = det.detect(P, cam_xy, counter_z, spheres)     # P: levelled pelvis points (N, 3) of the depth image
    boxes = tracker.update(res)                          # BoxTracker: per-episode memory -> Boxes (planner, masks)

Why: scene.yaml's boxes (near edge at pelvis x 0.40, 6 cm walls, 5 cm gaps) are the published layout; the organisers'
robot data shows the row at x 0.47-0.62, yawed -7 ... +14 deg, ~2 cm gaps, ~10 cm walls. Released at scene.yaml's
points there, objects land on or in front of the near walls. Nobody tells us which layout the finals use.

Detection (depth geometry only; colour is not used: the walls' orange is the objects' orange): wall points = levelled
points between h_lo and wall_range[1] + h_margin above the counter, inside the search region, outside the arms' sphere
model (+ arm_margin: the self-mask; perception.arm_spheres, set by the policy) -> xy occupancy grid (`cell`, counts
capped at `cap` so the dense near faces do not drown the side walls). Coarse: correlation with a soft square outline
(side size0, soft_r wide; wall points in a band just inside it count against it, inner_neg) at every yaw_step over
+-max_yaw_deg (FFT; templates precomputed), greedy peaks (suppressed within nms sides). Fine, each candidate on its own
(fit points: at most fit_cap per cell): every point near it goes to its nearest wall face (the face the camera sees: the
outer face of a wall the camera is outside of, else the inner face; wall_thickness apart), trimmed Gauss-Newton on
centre, yaw and side (the side is pulled to size0 by size_sd: a box seen on one side only keeps ~size0). Per box: wall
coverage (fraction of 1 cm bins with inliers along the near wall / the near half of each side wall), rms, rim = the
rim_pct percentile of its inliers' heights, wall-high points in a line well inside it (a neighbour's wall: the box
sits a gap off; re-seeded once on that wall). Accepted if the near wall (min_front) and a side wall (min_side) are
seen, rms <= max_rms, no wall inside, and centre / yaw / side / rim are inside the search ranges; conf = front x side x
rms terms (0..1); overlapping accepted ones: the best conf.
What it needs to see: the near wall of a box over at least min_front of its width and the near part of one side wall
(rim or face): the far walls may be out of the image (robot data: the head camera sees about the near half).

Names by position along the row (robot's left = +y): three accepted boxes -> left / center / right. Two: ~two
spacings apart -> left / right; neighbours -> by the side where a box seen in part stands (walls in the right place,
too little of them to deliver into). Else, if centre_y_max is set (the row is centred on the robot, as published and in
the robot data), by pelvis y: within centre_y_max of 0 = center, a box spacing (0.19-0.30 m +- centre_y_max) to
either side = left / right. Otherwise (and > 3 boxes) the names are uncertain: such boxes are kept for the masks and
the clearance checks but nothing is delivered to them, unless an earlier look of this episode named them (BoxTracker
matches by position). A named box is "ok" (deliverable); a name never found keeps scene.yaml's pose, not ok.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

NAMES = ("left", "center", "right")        # robot's left = +y
_NORMALS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])   # faces in the box frame: near, far, right, left
_TAN = (1, 1, 0, 0)                        # box-frame coordinate along each face


@dataclass(frozen=True)
class Box:
    """One box in the pelvis frame. Box frame: u = (cos yaw, sin yaw) (depth, away from the robot), v = (-sin, cos)
    (along the row, +y side positive); origin = the outer footprint's centre."""
    name: str | None
    c: tuple                   # outer footprint centre, pelvis xy (m)
    yaw: float                 # rad about pelvis z
    hu: float                  # half outer depth (along u)
    hv: float                  # half outer width (along v)
    rim: float                 # rim height above the counter (m)
    wall: float                # wall thickness
    floor: float = 0.003       # floor thickness
    conf: float = 1.0
    ok: bool = True            # deliver here (scene prior / detected with a certain name); False: masks only
    src: str = "scene"         # scene | detected
    info: dict = field(default_factory=dict, compare=False)

    def local(self, xy):
        """Pelvis xy (..., 2) -> box frame (..., 2)."""
        d = np.asarray(xy, float)[..., :2] - np.asarray(self.c, float)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)

    def pelvis(self, uv):
        uv = np.asarray(uv, float)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.asarray(self.c, float) + np.stack([c * uv[..., 0] - s * uv[..., 1], s * uv[..., 0] + c * uv[..., 1]],
                                                    -1)

    def inside(self, xy, margin=0.0):
        """Pelvis xy (..., 2) over the outer footprint + margin."""
        L = self.local(xy)
        return (np.abs(L[..., 0]) <= self.hu + margin) & (np.abs(L[..., 1]) <= self.hv + margin)

    def gap(self, xy):
        """Distance (m, Chebyshev in the box frame) outside the outer footprint; <= 0 inside."""
        L = np.abs(self.local(xy))
        return np.maximum(L[..., 0] - self.hu, L[..., 1] - self.hv)

    def corners(self, margin=0.0):
        """Outer footprint (+ margin) corners, pelvis xy (4, 2), CCW."""
        a, b = self.hu + margin, self.hv + margin
        return self.pelvis(np.array([[-a, -b], [a, -b], [a, b], [-a, b]]))

    @property
    def inner(self):
        """Inner half extents (u, v)."""
        return np.array([self.hu - self.wall, self.hv - self.wall])

    @property
    def front(self):
        """Midpoint of the near wall's outer face, pelvis xy."""
        return self.pelvis(np.array([-self.hu, 0.0]))

    def moved(self, other, pos=0.005, yaw_deg=1.0, rim=0.005):
        """True if `other` differs from this box by more than the tolerances (centre, yaw, rim, half sizes)."""
        if other is None:
            return True
        return bool(np.linalg.norm(np.subtract(self.c, other.c)) > pos or abs(np.degrees(self.yaw - other.yaw)) > yaw_deg
                    or abs(self.rim - other.rim) > rim or abs(self.hu - other.hu) > pos or abs(self.hv - other.hv) > pos)

    def text(self):
        x, y = self.c
        return (f"({x:.3f},{y:+.3f}) {np.degrees(self.yaw):+.1f}deg rim {1000 * self.rim:.0f}mm "
                f"side {2000 * self.hv:.0f}mm conf {self.conf:.2f}")


class Boxes:
    """The boxes as used downstream: named ones (left / center / right; .ok = deliverable) + `more` (detected, name
    uncertain: masks and rim clearances only)."""

    def __init__(self, named, more=(), source="scene", why=None):
        self.named = dict(named)
        self.more = list(more)
        self.source, self.why = source, why

    @classmethod
    def from_scene(cls, scene):
        b = scene.cfg["bins"]
        out = {}
        for name, (cx, cy) in b["centers"].items():
            c = scene.mat_to_pelvis([cx, cy, 0.0])[:2]
            out[name] = Box(name, (float(c[0]), float(c[1])), 0.0, float(b["outer_size"][1]) / 2,   # mat y = pelvis x
                            float(b["outer_size"][0]) / 2, float(b["wall_height"]), float(b.get("wall_thickness") or 0.0),
                            float(b.get("floor_thickness") or 0.0))
        return cls(out)

    @property
    def names(self):
        return tuple(self.named)

    def __getitem__(self, name):
        return self.named[name]

    def __contains__(self, name):
        return name in self.named

    def ok(self, name):
        b = self.named.get(name)
        return b is not None and b.ok

    def all(self):
        return list(self.named.values()) + self.more

    def inside_any(self, xy, margin=0.0):
        xy = np.asarray(xy, float)
        out = np.zeros(xy.shape[:-1], bool)
        for b in self.all():
            out |= b.inside(xy, margin)
        return out

    def gap(self, xy):
        """Distance (m) outside the nearest box footprint (<= 0: inside one)."""
        return np.min(np.stack([b.gap(xy) for b in self.all()]), axis=0)

    def min_rim(self):
        return min(b.rim for b in self.all())

    def max_rim(self):
        return max(b.rim for b in self.all())

    def text(self):
        """One line for the look log: each name's box (or why not), + uncertain ones."""
        parts = []
        for n, b in self.named.items():
            parts.append(f"{n[0].upper()} {b.text()}" if b.src == "detected" and b.ok else f"{n[0].upper()} -")
        s = " | ".join(parts) + (f" | +{len(self.more)} unnamed" if self.more else "")
        return s + (f" ({self.why})" if self.why else "")


class BoxDetector:
    def __init__(self, bc, scene):
        self.c, self.sc = bc, scene
        b = scene.cfg["bins"]
        self.s0 = float(bc.get("size") or np.mean(b["outer_size"]))
        self.t = float(bc["wall_thickness"] if bc.get("wall_thickness") is not None else b.get("wall_thickness", 0.0))
        self.floor = float(b.get("floor_thickness") or 0.0)
        cell, smax = float(bc["cell"]), float(bc["size_range"][1])
        ym = np.radians(float(bc["max_yaw_deg"]))
        fx = bc["front_x"]
        self.lo = np.array([fx[0] - 0.03, bc["y_range"][0]], float)
        hi = np.array([fx[1] + smax * (np.cos(ym) + np.sin(ym)) + 0.03, bc["y_range"][1]], float)
        self.n = np.ceil((hi - self.lo) / cell).astype(int)
        self.yaws = np.radians(np.arange(-bc["max_yaw_deg"], bc["max_yaw_deg"] + 1e-9, bc["yaw_step_deg"]))
        r, a = float(bc["soft_r"]), self.s0 / 2
        K = int(np.ceil((a * np.sqrt(2) + r) / cell))
        from scipy.fft import next_fast_len               # (a prime FFT length costs 5-10x)
        self.K, self.N = K, tuple(next_fast_len(int(k), real=True) for k in self.n + 2 * K)
        q = np.arange(-K, K + 1) * cell
        qx, qy = np.meshgrid(q, q, indexing="ij")
        FT, tsum = [], []
        for yaw in self.yaws:                    # soft square outline at this yaw, centred at index (0, 0) (wrapped)
            c, s = np.cos(yaw), np.sin(yaw)
            u, v = np.abs(c * qx + s * qy), np.abs(-s * qx + c * qy)
            out = np.hypot(np.maximum(u - a, 0), np.maximum(v - a, 0))
            ins = (u <= a) & (v <= a)
            d = np.where(ins, a - np.maximum(u, v), out)
            T = np.clip(1.0 - d / r, 0.0, 1.0)
            tsum.append(T.sum())
            # walls just inside the outline count against it: a neighbour's wall (a gap away) must not pass for this
            # box's own when the box's far side is out of view
            T = T - float(bc["inner_neg"]) * (ins & (d >= r) & (d <= r + float(bc["inner_band"])))
            W = np.zeros(self.N)
            idx = np.arange(-K, K + 1)
            W[np.ix_(idx % self.N[0], idx % self.N[1])] = T
            FT.append(np.conj(np.fft.rfft2(W)))
        self.FT, self.tsum = np.array(FT), float(np.mean(tsum))
        ii, jj = np.meshgrid(np.arange(self.n[0]), np.arange(self.n[1]), indexing="ij")
        self._ij = (ii, jj)

    # ---------------------------------------------------------------- detection
    def detect(self, P, cam, counter_z, spheres=None):
        """P (N, 3) levelled pelvis points, cam = camera xy, spheres = [(centres (n, 3), radii (n,))] self-mask.
        Returns {"boxes": [Box (name set if certain)], "named": bool, "why", "cands": [diagnostics], "n_points"}."""
        bc = self.c
        h = P[:, 2] - counter_z
        wmax = float(bc["wall_range"][1]) + float(bc["h_margin"])
        sel = ((h >= bc["h_lo"]) & (h <= wmax) & (P[:, 0] >= self.lo[0]) & (P[:, 1] >= self.lo[1])
               & (P[:, 0] < self.lo[0] + self.n[0] * bc["cell"]) & (P[:, 1] < self.lo[1] + self.n[1] * bc["cell"]))
        out = {"boxes": [], "named": False, "why": None, "cands": [], "n_points": 0}
        if spheres:
            sel &= ~self._near_spheres(P, sel, spheres, float(bc["arm_margin"]))
        Q, hq = P[sel, :2], h[sel]
        out["n_points"] = len(Q)
        if len(Q) < bc["min_points"]:
            out["why"] = f"{len(Q)} wall points"
            return out
        cell = float(bc["cell"])
        ij = np.floor((Q - self.lo) / cell).astype(int)
        H = np.minimum(np.bincount(ij[:, 0] * self.n[1] + ij[:, 1], minlength=int(np.prod(self.n))),
                       bc["cap"]).reshape(self.n).astype(float)
        K = self.K
        Hb = np.zeros(self.N)
        Hb[K:K + self.n[0], K:K + self.n[1]] = H
        C = np.fft.irfft2(np.fft.rfft2(Hb)[None] * self.FT, s=self.N)[:, K:K + self.n[0], K:K + self.n[1]]
        S = C.max(0) / (bc["cap"] * self.tsum)            # ~ fraction of the outline seen
        ky = C.argmax(0)
        cands, rr = [], (float(bc["nms"]) * self.s0 / cell) ** 2
        ii, jj = self._ij
        S = S.copy()
        lim = max(float(bc["min_score"]), float(bc["min_rel_score"]) * float(S.max()))
        for _ in range(int(bc["max_cands"])):
            i, j = np.unravel_index(int(S.argmax()), S.shape)
            if S[i, j] < lim:
                break
            cands.append([self.lo[0] + (i + 0.5) * cell, self.lo[1] + (j + 0.5) * cell, self.yaws[ky[i, j]], self.s0,
                          float(S[i, j])])
            S[(ii - i) ** 2 + (jj - j) ** 2 <= rr] = -1.0
        if not cands:
            out["why"] = "no box-like outline"
            return out
        # fit points: at most fit_cap per grid cell (the dense near faces must not outweigh the sparse side walls, nor
        # thin them out), then evenly thinned to max_points (deterministic)
        flat = ij[:, 0] * self.n[1] + ij[:, 1]
        o = np.argsort(flat, kind="stable")
        fs = flat[o]
        rank = np.arange(len(fs)) - np.searchsorted(fs, fs, side="left")
        keep = np.sort(o[rank < int(bc["fit_cap"])])
        Q, hq = Q[keep], hq[keep]
        if len(Q) > bc["max_points"]:
            k = np.linspace(0, len(Q) - 1, int(bc["max_points"])).astype(int)
            Q, hq = Q[k], hq[k]
        X, st = [], []
        for c in cands:
            x, s = self._fit(Q, hq, np.array(c[:4], float), cam)
            X.append(x)
            st.append(dict(s, score=round(float(c[4]), 3)))
            # a wall running through it: it sits a gap off its box (the box's far side out of view, the neighbour's
            # wall passed for its own): re-seed it shifted so that wall is its side wall (the smaller shift)
            if not s["ok"] and s.get("inner_line"):
                x, s = self._fit(Q, hq, self._reseed(x, s), cam)
                X.append(x)
                st.append(dict(s, reseeded=True))
        out["cands"] = st
        boxes = self._dedupe([self._box(x, s) for x, s in zip(X, st) if s["ok"]])
        if not boxes:
            out["why"] = "no box passed: " + "; ".join(s["why"] for s in st if s["why"])[:120]
            return out
        # a box only partly seen (too little of its walls to deliver into, but walls in the right place): still a box
        # there, for naming its neighbours
        part = [x for x, s in zip(X, st) if not s["ok"] and s["n"] >= bc["min_points"] and not s["inner"]
                and str(s["why"]).startswith(("near wall ", "side walls")) and s["cover"]["front"] >= 0.25
                and abs(s["rim"] - np.median([b.rim for b in boxes])) <= 0.015]
        out["boxes"], out["named"], why = self._name(boxes, part)
        out["why"] = why
        return out

    @staticmethod
    def _near_spheres(P, sel, spheres, margin):
        bad = np.zeros(len(P), bool)
        idx = np.flatnonzero(sel)
        for C, R in spheres:
            C, R = np.asarray(C, float).reshape(-1, 3), np.asarray(R, float).reshape(-1)
            if not len(C):
                continue
            lo, hi = C.min(0) - R.max() - margin, C.max(0) + R.max() + margin
            k = idx[np.all((P[idx] >= lo) & (P[idx] <= hi), axis=1)]
            if not len(k):
                continue
            d2 = ((P[k, None, :] - C[None]) ** 2).sum(-1)
            bad[k[(d2 <= (R[None] + margin) ** 2).any(1)]] = True
        return bad

    def _faces(self, Q, X, cam):
        """Signed distance (J, N, 4) of every point to every candidate's visible wall faces; inf off the segment."""
        d = Q[None] - X[:, None, :2]
        cs, sn = np.cos(X[:, 2])[:, None], np.sin(X[:, 2])[:, None]
        L = np.stack([cs * d[..., 0] + sn * d[..., 1], -sn * d[..., 0] + cs * d[..., 1]], -1)   # (J, N, 2)
        a = X[:, 3] / 2
        dc = np.asarray(cam, float)[None] - X[:, :2]
        lc = np.stack([np.cos(X[:, 2]) * dc[:, 0] + np.sin(X[:, 2]) * dc[:, 1],
                       -np.sin(X[:, 2]) * dc[:, 0] + np.cos(X[:, 2]) * dc[:, 1]], -1)
        inner = (lc @ _NORMALS.T) <= a[:, None]                  # camera inside that wall's plane: sees its inner face
        off = a[:, None] - self.t * inner
        r = L @ _NORMALS.T - off[:, None, :]
        tan = np.abs(L[..., list(_TAN)])
        return np.where(tan <= a[:, None, None] + 0.01, r, np.inf), L

    def _assign(self, Q, x, cam):
        """Nearest visible face of candidate x for every point: (residual (N,), face (N,), box frame (N, 2), near:
        the point lies within 3 cm of the outer footprint)."""
        r, L = self._faces(Q, x[None], cam)
        r, L = r[0], L[0]
        k = np.abs(r).argmin(1)
        near = np.all(np.abs(L) <= x[3] / 2 + 0.03, axis=1)
        return r[np.arange(len(Q)), k], k, L, near

    def _fit(self, Q, hq, x, cam):
        """Trimmed Gauss-Newton of one candidate on the points near it (centre, yaw, side; side prior size0 +-
        size_sd), then its statistics and acceptance (_stats). Each candidate alone: a wrong one cannot take a right
        one's points."""
        bc = self.c
        x = np.asarray(x, float).copy()
        sub = ((Q - x[:2]) ** 2).sum(1) <= (0.75 * x[3] + 0.1) ** 2     # (it moves < 9 cm: the steps are clipped)
        Q, hq = Q[sub], hq[sub]
        thr, sd_s = float(bc["inlier_thr"]), float(bc["size_sd"])
        for it in range(int(bc["iters"])):
            res, k, L, near = self._assign(Q, x, cam)
            s = near & (np.abs(res) < thr)
            if s.sum() < (10 if it else bc["min_points"]):  # (hopeless candidates stop early)
                break
            sig = max(1.4826 * float(np.median(np.abs(res[s]))), 0.5 * float(bc["min_thr"]))
            n = _NORMALS[k[s]]
            c, sn = np.cos(x[2]), np.sin(x[2])
            Rn = np.c_[c * n[:, 0] - sn * n[:, 1], sn * n[:, 0] + c * n[:, 1]]
            u, v = L[s, 0], L[s, 1]
            A = np.vstack([np.c_[-Rn, n[:, 0] * v - n[:, 1] * u, np.full(s.sum(), -0.5)] / sig,
                           [0.0, 0.0, 0.0, 1.0 / sd_s]])
            b = np.r_[-res[s] / sig, -(x[3] - self.s0) / sd_s]
            step = np.linalg.lstsq(A, b, rcond=None)[0]
            x += np.clip(step, [-0.015, -0.015, -0.05, -0.01], [0.015, 0.015, 0.05, 0.01])
            thr = max(float(bc["trim_k"]) * sig, float(bc["min_thr"]))
        res, k, L, near = self._assign(Q, x, cam)
        return x, self._stats(x, L, hq, res, near & (np.abs(res) < thr), k)

    def _stats(self, x, L, hq, res, s, kk):
        bc = self.c
        a = x[3] / 2
        st = {"c": (round(float(x[0]), 4), round(float(x[1]), 4)), "yaw_deg": round(float(np.degrees(x[2])), 2),
              "side": round(float(x[3]), 4), "n": int(s.sum()), "ok": False, "why": None}
        if s.sum() < bc["min_points"]:
            st["why"] = f"{int(s.sum())} points"
            return st

        def cov(k, lo, hi):
            t = L[s & (kk == k), _TAN[k]]
            bins = np.arange(lo, hi + 1e-9, 0.01)
            if len(bins) < 2:
                return 0.0
            cnt = np.histogram(t, bins)[0]
            return float((cnt >= 2).mean())
        st["cover"] = {"front": round(cov(0, -a, a), 2), "back": round(cov(1, -a, a), 2),
                       "right": round(cov(2, -a, 0.0), 2), "left": round(cov(3, -a, 0.0), 2)}
        rms = float(np.sqrt(np.mean(res[s] ** 2)))
        st["rms_mm"] = round(1000 * rms, 2)
        rim = float(np.percentile(hq[s], bc["rim_pct"]))
        st["rim"] = round(rim, 4)
        m = a - self.t - float(bc["inner_clear"])          # wall-high points well inside, on a line parallel to a
        ins = (np.abs(L[:, 0]) <= m) & (np.abs(L[:, 1]) <= m) & (hq > rim - float(bc["inner_clear"]))   # wall, longer
        line = self._inner_line(L[ins])                    # than an object: a wall through it (a neighbour's: this
        st["inner"] = 0 if line is None else line[2]       # box is shifted)
        if line is not None:
            st["inner_line"] = line[:2]
        cv = st["cover"]
        side = max(cv["left"], cv["right"])
        front = (float(np.cos(x[2]) * -a + x[0]))           # x of the near wall's midpoint
        conf = (min(1.0, cv["front"] / 0.8) * min(1.0, side / 0.6) * max(0.0, 1.0 - (rms / bc["max_rms"]) ** 2))
        st["conf"] = round(conf, 3)
        fx, wr, sr = bc["front_x"], bc["wall_range"], bc["size_range"]
        ry = 0.26 * abs(np.sin(x[2]))                     # a yawed row's side boxes: one box spacing x sin(yaw)
        why = (f"near wall {cv['front']:.2f}" if cv["front"] < bc["min_front"] else
               f"side walls {side:.2f}" if side < bc["min_side"] else
               f"rms {1000 * rms:.1f} mm" if rms > bc["max_rms"] else
               f"a wall inside ({st['inner']} points)" if st["inner"] >= bc["inner_min"] else
               f"near wall at x {front:.3f}" if not fx[0] - ry <= front <= fx[1] + ry else
               f"yaw {np.degrees(x[2]):.1f} deg" if abs(np.degrees(x[2])) > bc["max_yaw_deg"] + 1.0 else
               f"side {1000 * x[3]:.0f} mm" if not sr[0] <= x[3] <= sr[1] else
               f"rim {1000 * rim:.0f} mm" if not wr[0] <= rim <= wr[1] else
               f"conf {conf:.2f}" if conf < bc["min_conf"] else None)
        st["why"], st["ok"] = why, why is None
        return st

    def _inner_line(self, Li):
        """Box-frame points (n, 2) -> (axis k, coordinate, count) of the best line at a constant L[:, k] (+-1 cm)
        that runs >= inner_len along the other axis with >= inner_min points; None if there is none."""
        bc, best = self.c, None
        if len(Li) < bc["inner_min"]:
            return None
        for k in (0, 1):
            b = np.floor(Li[:, k] / 0.01).astype(int)
            vals, cnt = np.unique(b, return_counts=True)
            for v in vals[np.argsort(-cnt)][:3]:
                sel = np.abs(b - v) <= 1
                t = Li[sel, 1 - k]
                if sel.sum() >= bc["inner_min"] and np.percentile(t, 95) - np.percentile(t, 5) >= bc["inner_len"]:
                    if best is None or sel.sum() > best[2]:
                        best = (k, float(np.median(Li[sel, k])), int(sel.sum()))
                    break
        return best

    def _reseed(self, x, st):
        """Candidate x (cx, cy, yaw, side) with a wall inside it (st inner_line: box-frame axis k and coordinate p
        of that wall): moved along k so that wall becomes the nearer of its two walls across k."""
        k, p = st["inner_line"]
        a = x[3] / 2
        d = min((p - a, p + a), key=abs)                   # its +a or -a wall onto it
        duv = np.zeros(2)
        duv[k] = d
        c, s = np.cos(x[2]), np.sin(x[2])
        return np.array([x[0] + c * duv[0] - s * duv[1], x[1] + s * duv[0] + c * duv[1], x[2], x[3]])

    def _box(self, x, st):
        return Box(None, (float(x[0]), float(x[1])), float(x[2]), float(x[3]) / 2, float(x[3]) / 2, st["rim"], self.t,
                   self.floor, st["conf"], False, "detected", st)

    @staticmethod
    def _dedupe(boxes):
        out = []
        for b in sorted(boxes, key=lambda b: (-b.conf, -b.info.get("n", 0))):
            if all(np.linalg.norm(np.subtract(b.c, o.c)) > 0.8 * min(b.hv, o.hv) * 2 for o in out):
                out.append(b)
        return out

    def _name(self, boxes, part=()):
        """Boxes in row order (+y first = left). part: candidates (cx, cy, yaw, side) of boxes seen too little to use
        but standing there. Returns (boxes with names where certain, all named?, why)."""
        bc = self.c
        if len(boxes) > 3:
            boxes = sorted(boxes, key=lambda b: -b.conf)
            if boxes[3].conf > 0.5 * boxes[2].conf:
                return boxes, False, f"{len(boxes)} boxes: count ambiguous"
            boxes = boxes[:3]
        yaw = float(np.mean([b.yaw for b in boxes]))
        vdir = np.array([-np.sin(yaw), np.cos(yaw)])
        boxes = sorted(boxes, key=lambda b: -float(np.dot(b.c, vdir)))
        if len(boxes) == 3:
            return [replace(b, name=n, ok=True) for b, n in zip(boxes, NAMES)], True, None
        if len(boxes) == 2:
            a, b = boxes
            d = float(np.dot(np.subtract(a.c, b.c), vdir))
            if d >= 1.8 * 2 * max(a.hv, b.hv):              # room for the middle one between them
                return [replace(a, name="left", ok=True), replace(b, name="right", ok=True)], True, "center not found"
            # neighbours: the third stands where a partly seen box is (on one side only)
            there = [any(np.linalg.norm(x[:2] - (np.asarray(o.c) + sgn * d * vdir)) < 0.05
                         and abs(np.degrees(x[2] - yaw)) < 5.0 for x in part) for o, sgn in ((a, 1), (b, -1))]
            if there[0] != there[1]:
                n = ("center", "right") if there[0] else ("left", "center")
                return ([replace(a, name=n[0], ok=True), replace(b, name=n[1], ok=True)], True,
                        f"{'left' if there[0] else 'right'} not found (seen in part)")
        named = self._by_position(boxes)
        if named:
            missing = [n for n in NAMES if n not in [b.name for b in named]]
            return named, True, f"{', '.join(missing)} not found; named by position (centred row)"
        return boxes, False, f"{len(boxes)} box{'es' if len(boxes) > 1 else ''}: names uncertain"

    def _by_position(self, boxes):
        """centre_y_max (m; null: off): with fewer than three boxes seen, a box within it of pelvis y 0 is the center
        one, one a box spacing (its side + a 0-6 cm gap, +- centre_y_max) to the left / right the left / right one
        (the row is centred on the robot: published layout, robot data). All boxes must get a different name, else
        None."""
        c = self.c.get("centre_y_max")
        if c is None:
            return None
        out = []
        for b in boxes:
            y, s = b.c[1], 2 * b.hv
            n = ("center" if abs(y) <= c else "left" if s - c <= y <= s + 0.06 + c else
                 "right" if s - c <= -y <= s + 0.06 + c else None)
            if n is None or n in [o.name for o in out]:
                return None
            out.append(replace(b, name=n, ok=True))
        return out


class BoxTracker:
    """Per-episode memory of the named boxes: a look that names them fuses into them (conf-weighted mean if within
    fuse_pos / fuse_yaw_deg, else the new estimate replaces: the box was moved); a look whose names are uncertain
    matches its boxes to the remembered ones by position. Names never found keep the prior (scene.yaml), not ok."""

    def __init__(self, bc, prior: Boxes):
        self.c, self.prior = bc, prior
        self.reset()

    def reset(self):
        self.mem = {}            # name -> (Box, weight)
        self.boxes = Boxes(self.prior.named, source="prior", why="not detected yet")
        for n, b in self.boxes.named.items():
            self.boxes.named[n] = replace(b, ok=False)
        self.events = []

    def update(self, res):
        bc = self.c
        new, more = {}, []
        if res["named"]:
            new = {b.name: b for b in res["boxes"] if b.name}
        else:
            for b in res["boxes"]:
                m = min(self.mem.items(), key=lambda kv: np.linalg.norm(np.subtract(kv[1][0].c, b.c)), default=None)
                if m is not None and np.linalg.norm(np.subtract(m[1][0].c, b.c)) <= bc["fuse_pos"] and m[0] not in new:
                    new[m[0]] = replace(b, name=m[0], ok=True)
                else:
                    more.append(b)
        self.events = []
        for n, b in new.items():
            old = self.mem.get(n)
            if (old is not None and np.linalg.norm(np.subtract(old[0].c, b.c)) <= bc["fuse_pos"]
                    and abs(np.degrees(old[0].yaw - b.yaw)) <= bc["fuse_yaw_deg"]):
                w0, w1 = old[1], max(b.conf, 1e-3)
                f = w1 / (w0 + w1)

                def mix(p, q):
                    return (1 - f) * p + f * q
                fused = replace(b, c=tuple(mix(np.asarray(old[0].c), np.asarray(b.c))), yaw=mix(old[0].yaw, b.yaw),
                                hu=mix(old[0].hu, b.hu), hv=mix(old[0].hv, b.hv), rim=mix(old[0].rim, b.rim),
                                conf=max(old[0].conf, b.conf))
                self.mem[n] = (fused, min(w0 + w1, float(bc.get("max_weight", 5.0))))
            else:
                if old is not None:
                    self.events.append(f"{n} box re-placed: {old[0].text()} -> {b.text()}")
                self.mem[n] = (b, max(b.conf, 1e-3))
        for n in [n for n in self.mem if n not in new]:    # a remembered box where this look put another name
            o = self.mem[n][0]
            if any(np.linalg.norm(np.subtract(o.c, b.c)) < 1.6 * min(o.hv, b.hv) for b in new.values()):
                self.events.append(f"{n} box forgotten: another name stands there now")
                del self.mem[n]
        named = {}
        for n, pb in self.prior.named.items():
            named[n] = self.mem[n][0] if n in self.mem else replace(pb, ok=False)
        missing = [n for n in self.prior.named if n not in self.mem]
        why = res["why"] if not new else (f"{', '.join(missing)} not found" if missing else None)
        if not new and self.mem:
            why = f"this look: {res['why']}; kept from earlier looks" if res["why"] else "kept from earlier looks"
        self.boxes = Boxes(named, more, source="detected" if self.mem else "prior", why=why)
        return self.boxes
