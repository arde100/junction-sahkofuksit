"""Stillness gate before a look (config `still`): the head camera moves with the robot's body (handoffs/shake: still
when the arms are still, within 0.5 px of rest 0.2 s (median) / 0.5 s (p90) after an arm stops), so a look is taken
only when two head images dt_s apart agree.

    gate = StillGate(cfg["still"], calibration)
    r = gate.step(obs, t, spheres)     # None: keep holding (call again with a newer obs) | result dict: look now

Measure (measure()): corners (goodFeaturesToTrack) of the first image outside the arms' projected silhouettes (the
policy's sphere model of forearms + hands, + arm_margin_px), tracked into the second (pyramidal Lucas-Kanade), a
similarity transform fitted with RANSAC; motion = median displacement the fitted transform gives at the tracked
corners (px). Still: motion <= thr_px -> look. Else the second image becomes the reference and the gate waits dt_s
again; after timeout_s the look is taken anyway (result timeout True, logged). Too few trackable corners: the motion
cannot be measured -> look now (result why). threaded: the comparison (10-20 ms) runs in a daemon thread, step()
returns None meanwhile (act() does not wait for it).
"""
from __future__ import annotations

import threading
import time

import numpy as np


class StillGate:
    def __init__(self, cfg, calibration, cam="head"):
        self.c, self.cal, self.cam = cfg, calibration, cam
        self.reset()

    def reset(self):
        """Before every look (the settle ended)."""
        self.ref = self.t_ref = self.t0 = None
        self.job = None
        self.checks = []          # (t, motion px) of this look's comparisons

    # ------------------------------------------------------------------ measurement
    def mask(self, shape, spheres):
        """uint8 mask (255 = usable) without the arms' spheres projected (+ arm_margin_px)."""
        import cv2
        m = np.full(shape[:2], 255, np.uint8)
        K = self.cal.cams[self.cam].K
        for C, R in spheres or ():
            C, R = np.asarray(C, float).reshape(-1, 3), np.asarray(R, float).reshape(-1)
            uvz = self.cal.project(self.cam, C)
            for (u, v, z), r in zip(uvz, R):
                if z > 0.02 and np.isfinite(u) and np.isfinite(v):
                    rp = int(np.ceil(K[0, 0] * r / z + float(self.c.get("arm_margin_px", 12))))
                    if -rp <= u < shape[1] + rp and -rp <= v < shape[0] + rp:
                        cv2.circle(m, (int(round(u)), int(round(v))), rp, 0, -1)
        return m

    def measure(self, a, b, mask=None):
        """Image motion between RGB images a -> b: {"motion_px", "n" tracked, "inliers"} or {"why"}."""
        import cv2
        c = self.c
        ga = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(np.ascontiguousarray(b), cv2.COLOR_RGB2GRAY)
        p0 = cv2.goodFeaturesToTrack(ga, int(c.get("max_corners", 200)), float(c.get("quality", 0.01)),
                                     float(c.get("min_dist_px", 8)), mask=mask)
        n_min = int(c.get("min_tracks", 20))
        if p0 is None or len(p0) < n_min:
            return {"why": f"{0 if p0 is None else len(p0)} corners"}
        p1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        pa, pb = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        if len(pa) < n_min:
            return {"why": f"{len(pa)} tracked corners"}
        M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if M is None:
            return {"why": "no transform"}
        q = pa[inl.ravel() == 1] if inl is not None and inl.sum() >= n_min // 2 else pa
        disp = q @ M[:, :2].T + M[:, 2] - q
        return {"motion_px": round(float(np.median(np.linalg.norm(disp, axis=1))), 3), "n": int(len(pa)),
                "inliers": int(inl.sum()) if inl is not None else 0}

    # ------------------------------------------------------------------ gate
    def step(self, obs, t, spheres=None, threaded=False):
        c = self.c
        img = obs.get("head_rgb") if isinstance(obs, dict) else None
        if img is None:
            return {"still": None, "why": "no image", "waited_s": 0.0}
        if self.ref is None:
            self.ref, self.t_ref, self.t0 = np.array(img, copy=True), t, t
            return None
        if self.job is None:
            if t - self.t_ref < float(c.get("dt_s", 0.17)) - 1e-6:
                return None
            self.job = {"t": t, "img": np.array(img, copy=True), "res": None}
            job, ref, m = self.job, self.ref, self.mask(np.shape(img), spheres)

            def run():
                t1 = time.perf_counter()
                job["res"] = dict(self.measure(ref, job["img"], m), ms=round(1000 * (time.perf_counter() - t1), 1))
            if threaded:
                threading.Thread(target=run, daemon=True).start()
            else:
                run()
        job = self.job
        if job["res"] is None:
            return None
        self.job, r = None, job["res"]
        waited = round(job["t"] - self.t0, 2)
        if "why" in r:                 # cannot measure: look now, say why
            return {"still": None, "why": r["why"], "waited_s": waited, "ms": r["ms"]}
        self.checks.append((job["t"], r["motion_px"]))
        if r["motion_px"] <= float(c.get("thr_px", 0.3)):
            return {"still": True, "motion_px": r["motion_px"], "waited_s": waited, "checks": len(self.checks),
                    "ms": r["ms"]}
        if job["t"] - self.t0 >= float(c.get("timeout_s", 2.0)):
            return {"still": False, "timeout": True, "motion_px": r["motion_px"], "waited_s": waited,
                    "checks": len(self.checks), "ms": r["ms"]}
        self.ref, self.t_ref = job["img"], job["t"]   # still moving: compare the next image with this one
        return None

    @staticmethod
    def text(r):
        if r.get("still") is None:
            return f"still gate: not measured ({r.get('why')}), look now"
        if r.get("timeout"):
            return (f"still gate: TIMEOUT after {r['waited_s']:.1f} s, image still moving {r['motion_px']:.2f} px: "
                    f"look anyway ({r['checks']} checks)")
        return (f"still gate: {r['motion_px']:.2f} px over {r['waited_s']:.2f} s ({r['checks']} check"
                f"{'s' if r['checks'] > 1 else ''}, {r['ms']:.0f} ms)")
