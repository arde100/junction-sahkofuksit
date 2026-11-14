"""Far lab: grasp one object with a config profile's own grasp (default `none` = the robot base values) and deliver it
into a chosen box with the far release (baselines/classical/far.py, grasp.far), then record where it left the hand,
its velocity then, where it landed and came to rest, and the outcome by the organisers' rule. PRIVILEGED measurement
tool: the object pose and the TRUE box poses come from the scene service / the running variant (planner.set_boxes
with the true boxes, marked detected). Nothing in a policy may import this module.

Built on grasp_lab.Lab (stepping through AaltoEnv + its SafetyFilter at 30 Hz, sag feed-forward with the profile's
fixed prior, settle and correct, planning in a thread, recovery). Per case:
  1. place ONE object at pelvis (x, y) (--spots), wait until it rests, read its pose,
  2. plan object -> --box with the profile's planner (Planner._plan_side_hints; a box with no reachable drop slot gets
     the far release, far.py; --set '{"grasp.far.near_clear": -0.02}' also tries the ones predicted short),
  3. execute; from the far phase (`open`) on, every tick: object pose, measured grasp point (FK of measured joints),
     commanded / measured hand; then until the object rests (<= 2.5 s),
  4. derive: held at the release start (object within 4 cm of the measured grasp point, lifted) else the case is a
     SKIPPED grasp; loss of contact = first tick after the release command at which the object is >= 12 mm from where
     it sat in the hand (hand frame) or falls faster than 0.4 m/s more than the hand; its velocity there (central
     difference of the object track); landing = first tick at which the centre is below release height - 90 % of the
     drop; rest pose; outcome: target box (= correct) / other box (wrong) / rim (over a box footprint, above the
     rim) / counter / off (beyond the table: -1); predicted (plan.far) vs measured landing in the box frame.

    aalto py -m aaltonomous.sim.far_lab --boxes left,center,right --repeats 3 --out /data/aaro/wt/t3_tmp/x.jsonl
    aalto py -m aaltonomous.sim.far_lab --set '{"grasp.far.near_clear": -0.02}' ...
(2nd sim: AALTO_* env vars as in evaluate.py / grasp_lab.py)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import sys
import time
import traceback

import numpy as np

from aaltonomous.sim import evaluate as ev
from aaltonomous.sim import grasp_lab as GL

HZ = 30.0
DEFAULT_SPOTS = {"left": [(0.30, 0.17)], "center": [(0.30, 0.07), (0.30, -0.07)], "right": [(0.30, -0.17)]}


def true_boxes(S):
    """bins.Boxes of the running variant's boxes (VariantScene poses; scene.yaml boxes for the plain Scene), marked
    'detected' (conf 1) so the planner treats them like the box detection's."""
    from aaltonomous.baselines.classical.bins import Box, Boxes
    from aaltonomous.sim.aalto_task import boxes as Bx
    D, W, t = Bx.box_dims(S.cfg)
    b = S.cfg["bins"]
    if hasattr(S, "poses"):
        poses = S.poses
    else:
        poses = {n: tuple(S.mat_to_pelvis([c[0], c[1], 0.0])[:2]) + (0.0,) for n, c in b["centers"].items()}
    return Boxes({n: Box(n, (float(x), float(y)), float(yaw), D / 2, W / 2, float(b["wall_height"]), float(t),
                         float(b.get("floor_thickness") or 0.0), 1.0, True, "detected")
                  for n, (x, y, yaw) in poses.items()}, source="detected")


class FarLab(GL.Lab):
    def __init__(self, a, cfg, env, sc, S, log=print):
        super().__init__(a, cfg, env, sc, log)
        from aaltonomous.baselines.classical.planner import Planner
        from aaltonomous.sim import oracle
        self.S = S
        self.scene = S
        self.H, self.zc = S.object_height(), S.counter_z
        safety = env.filter.cfg if env is not None else None
        self.boxes = true_boxes(S)
        self.planner = Planner(cfg, S, self.post, self.home_q, self.planner.kin, safety, self.spec)
        self.planner.ff = self.ff
        self.planner.set_boxes(self.boxes)
        if self.cplanner is not None:
            self.cplanner = Planner(cfg, S, self.post, self.home_q, self.mkin, safety, self.spec)
            self.cplanner.ff = self.ff
            self.cplanner.set_boxes(self.boxes)
        self.perc = oracle.PrivilegedPerception(S, sc) if sc is not None else None

    # ------------------------------------------------------------------ planning: object -> its box
    def _plan_job(self, case, det, start, prev):
        self._job_box = case["box"]             # (the planning thread: the case being planned, maybe the next one)
        return super()._plan_job(case, det, start, prev)

    def _sides(self, det):
        b = self._job_box
        if b != "center":
            return [b]
        return ["left" if det.top[1] > 0 else "right"]

    def _plan_side(self, det, side, arm, hand, nbs=None):
        P = self.planner
        P.drop_used = {}
        p = P._plan_side_hints(det, side, self._job_box, arm, hand)
        if p.ok and p.far is None:
            p.reason = (p.reason + "; " if p.reason else "") + "stock drop slot (no far release)"
        return p

    def _row0(self, i, c):
        row = super()._row0(i, c)
        row.update(box=c["box"], px=c["px"], py=c["py"])
        return row

    def _plan_info(self, p, det):
        info = super()._plan_info(p, det)
        info["far"] = p.far
        return info

    def run_case(self, i, case, nxt):
        self.cur_box = case["box"]
        return super().run_case(i, case, nxt)

    # ------------------------------------------------------------------ measurement
    def _measure(self, j, row, o0):
        A, Hh, side = j["A"], j["H"], j["plan"].side
        A0 = A
        ph = {n: (s, e) for n, s, e in j["phases"]}
        from aaltonomous.baselines.classical.planner import correct_ticks
        cs, _ = ph["close"]
        k_corr = (cs - correct_ticks(self.cfg["grasp"]["correct"], self.dt)[1]
                  if self.cplanner is not None and "settle" in ph else None)
        far = j["plan"].far
        k0 = ph["open"][0] if "open" in ph else len(A)
        k_rel = k0 + (far["k_open"] if far else 0)          # release posture first commanded
        rows, t0 = [], None
        for k in range(len(A)):
            if k == k_corr:
                A = self._correct(j, A, k, row)
            self._step(A[k], Hh[k])
            if k == ph["lift"][1] - 1:
                o = self._obj("lift", side)
                row["rise_lift_mm"] = GL._r(1000 * (o["p"][2] - o0["p"][2]))
            if k >= k0 - 1:
                t0 = self.now() if t0 is None else t0
                rows.append(self._sample(k, side, A0[min(k, len(A0) - 1)]))
        # after the body: until at rest
        t_end = self.now()
        calm = 0
        while self.now() - t_end < 2.5:
            self._hold(1)
            rows.append(self._sample(len(A), side, A0[-1]))
            calm = calm + 1 if rows[-1]["speed"] < self.rest_speed else 0
            if calm >= 6 and self.now() - t_end > 0.6:
                break
        self.prev = {"side": side, "tail": j["tail"]}
        self._derive(row, rows, k0, k_rel, far, o0)

    def _sample(self, k, side, a_cmd):
        o = self._obj()
        arm, _, hand = self.env.backend.joints()
        gm = self.mkin.grasp_pose(side, arm)[0]
        self.mkin._fk(arm)
        Rm = self.mkin.data.oMf[self.mkin.palm[side]].rotation.copy()
        return {"k": k, "t": self.now(), "p": o["p"], "speed": o["speed"], "tilt": o["tilt"], "gm": gm, "Rm": Rm,
                "gc": self.mkin.grasp_pose(side, a_cmd)[0]}

    def _derive(self, row, rows, k0, k_rel, far, o0):
        S, zc = self.S, self.zc
        dt = self.dt
        r0 = next((r for r in rows if r["k"] >= k0), rows[0])
        gap0 = float(np.linalg.norm(r0["p"] - r0["gm"]))
        rise = float(r0["p"][2] - o0["p"][2])
        row.update(gap_open_mm=GL._r(1000 * gap0), rise_open_mm=GL._r(1000 * rise))
        row["held"] = gap0 < 0.04 and rise > 0.05
        fin = rows[-1]
        pm = S.pelvis_to_mat(fin["p"])
        b = S.bin_of(pm)
        over = [n for n in S.bin_names if self.boxes[n].inside(fin["p"][:2], 0.0)]
        out = ("off" if not S.on_counter(pm) else "correct" if b == self.cur_box else "wrong" if b is not None
               else "rim" if over else "counter")
        row.update(outcome=out if row["held"] else "skipped", final=[GL._r(1000 * v) for v in fin["p"]],
                   final_box=b, final_tilt=GL._r(fin["tilt"]), final_speed=GL._r(fin["speed"], 3))
        bx = self.boxes[self.cur_box]
        row["final_uv"] = [GL._r(1000 * v) for v in bx.local(fin["p"][:2])]
        if far:
            row["pred"] = {k: far.get(k) for k in ("z", "v_lat", "s_env", "s_rel", "land_uv", "margin", "margin_far",
                                                   "margin_side", "release", "tilt_deg", "t_flight")}
        if not row["held"]:
            return
        # loss of contact: object position in the measured hand frame moves >= 12 mm from where it sat, or it falls
        # faster than the hand by 0.4 m/s
        rel0 = r0["Rm"].T @ (r0["p"] - r0["gm"])
        i_loss = None
        for i in range(1, len(rows)):
            r = rows[i]
            if r["k"] < k_rel:
                continue
            d = np.linalg.norm(r["Rm"].T @ (r["p"] - r["gm"]) - rel0)
            vz = (rows[i]["p"][2] - rows[i - 1]["p"][2]) / dt
            vh = (rows[i]["gm"][2] - rows[i - 1]["gm"][2]) / dt
            if d >= 0.012 or vz - vh < -0.4:
                i_loss = i
                break
        if i_loss is None:
            row["loss"] = None
            return
        # the moment it left: back up to the last tick still within 4 mm of its seat (the threshold lags)
        j = i_loss
        while j - 1 > 0 and rows[j - 1]["k"] >= k_rel and np.linalg.norm(
                rows[j - 1]["Rm"].T @ (rows[j - 1]["p"] - rows[j - 1]["gm"]) - rel0) >= 0.004:
            j -= 1
        rl = rows[j]
        a, b2 = rows[max(j - 1, 0)], rows[min(j + 1, len(rows) - 1)]
        vel = (b2["p"] - a["p"]) / ((b2["t"] - a["t"]) or dt)
        hv = (b2["gm"] - a["gm"]) / ((b2["t"] - a["t"]) or dt)
        t_rel_cmd = next((r["t"] for r in rows if r["k"] >= k_rel), rl["t"])
        row["loss"] = {"t_after_cmd_s": GL._r(rl["t"] - t_rel_cmd, 3), "p": [GL._r(1000 * v) for v in rl["p"]],
                       "uv": [GL._r(1000 * v) for v in bx.local(rl["p"][:2])], "vel": [GL._r(v, 3) for v in vel],
                       "hand_vel": [GL._r(v, 3) for v in hv], "gm_vs_cmd_mm": [GL._r(1000 * v) for v in rl["gm"] - rl["gc"]]}
        # landing: first sample whose centre has dropped >= 90 % of (release height - resting height in a box)
        z_rest = zc + bx.floor + self.H / 2
        z_hit = rl["p"][2] - 0.9 * (rl["p"][2] - z_rest)
        land = next((r for r in rows[j:] if r["p"][2] <= z_hit), None)
        if land is not None:
            row["land"] = {"p": [GL._r(1000 * v) for v in land["p"]],
                           "uv": [GL._r(1000 * v) for v in bx.local(land["p"][:2])],
                           "t_s": GL._r(land["t"] - rl["t"], 3)}
        if far:   # the prediction redone from the MEASURED release state (ballistics only)
            pr = self.planner.far.predict(bx, rl["p"], vel)
            row["ballistic_uv"] = [GL._r(1000 * v) for v in pr[1]]


def make_cases(a):
    spots = json.loads(a.spots) if a.spots else DEFAULT_SPOTS
    out = []
    shapes = itertools.cycle(a.shapes.split(","))
    colors = itertools.cycle(a.colors.split(","))
    for r in range(a.repeats):
        for b in a.boxes.split(","):
            for x, y in spots[b]:
                out.append({"box": b, "px": x, "py": y, "shape": next(shapes), "color": next(colors), "rep": r,
                            "yaw": float(a.yaw)})
    return out


def _line(r):
    head = f"{r['i']:3d} {r['box']:6s} {r['shape']:8s} ({r['px']:.2f},{r['py']:+.2f})"
    if r.get("error"):
        return f"{head} ERROR {r['error']}"
    if not r.get("plan_ok"):
        return f"{head} PLAN FAIL {str(r.get('plan_reason'))[:160]}"
    p = r.get("pred") or {}
    lo = r.get("loss") or {}
    pu = p.get("land_uv") or [np.nan, np.nan]
    return (f"{head} {r.get('side')} z={p.get('z')} lat={p.get('v_lat')} pred uv {1000 * pu[0]:.0f},{1000 * pu[1]:.0f} "
            f"margin {1000 * (p.get('margin') or np.nan):.0f} land {(r.get('land') or {}).get('uv')} "
            f"ball {r.get('ballistic_uv')} final {r.get('final_uv')} loss {lo.get('t_after_cmd_s')}s v {lo.get('vel')} "
            f"[{r['outcome']}] clips {r.get('clips')}")


def summary(rows, out=print):
    cs = [r for r in rows if r.get("type") == "case"]
    by = {}
    for r in cs:
        by.setdefault(r["box"], []).append(r)
    out("\n  box     n  plan skip correct rim counter wrong off   loss_s(med) land_u_mm(mean/sd) pred_err_u_mm")
    for b, g in sorted(by.items()):
        n = lambda o: sum(r.get("outcome") == o for r in g)
        ls = [r["loss"]["t_after_cmd_s"] for r in g if r.get("loss")]
        lu = [r["land"]["uv"][0] for r in g if r.get("land")]
        pe = [r["land"]["uv"][0] - 1000 * r["pred"]["land_uv"][0] for r in g if r.get("land") and r.get("pred")]
        out(f"  {b:6s} {len(g):2d} {sum(bool(r.get('plan_ok')) for r in g):4d} {n('skipped'):4d} {n('correct'):7d} "
            f"{n('rim'):3d} {n('counter'):7d} {n('wrong'):5d} {n('off'):3d}   "
            f"{np.median(ls) if ls else float('nan'):.3f}       {np.mean(lu) if lu else float('nan'):.0f}/"
            f"{np.std(lu) if lu else float('nan'):.0f}            {np.mean(pe) if pe else float('nan'):.0f}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", default="left,center,right")
    ap.add_argument("--spots", default=None, help='JSON {box: [[pelvis x, y], ...]} object start positions')
    ap.add_argument("--shapes", default="cube")
    ap.add_argument("--colors", default="red,yellow,blue")
    ap.add_argument("--yaw", default=0.0, help="deg, mat frame")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--profile", default="none")
    ap.add_argument("--set", default=None, help="JSON {dotted.key: value} config overrides")
    ap.add_argument("--correct", default=None, choices=["on", "off", "dwell"])
    ap.add_argument("--time-limit", type=float, default=40.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    # (grasp_lab.Lab reads these)
    ap.add_argument("--hold", type=float, default=0.0)
    ap.add_argument("--sample-hz", type=float, default=5.0)
    ap.add_argument("--xy-error", default="0,0")
    ap.add_argument("--place-offset", default="0,0")
    ap.add_argument("--arm", default="auto")
    ap.add_argument("--finger-yaw", type=float, default=None)
    ap.add_argument("--face-align", default=False)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--mock", action="store_true", help="offline mock world (control flow only, no physics)")
    ap.add_argument("--variant", default="real_venue", help="--mock: the variant whose boxes / scene to use")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.aalto_task import variant as Vm
    from aaltonomous.sim.scene_client import SceneClient
    user = json.loads(a.set) if a.set else {}
    ov = dict(user)
    if a.correct:
        ov["grasp.correct.enabled"] = a.correct != "off"
    cfg = load_config(profile=a.profile, overrides=ov)
    if a.mock:                      # offline control-flow check: mock world (no physics), the variant's boxes
        V = Vm.load(a.variant)
        env, sc = GL.make_mock(cfg, sag=True)
        info = {"variant": V}
        S = Vm.scene(V)
        env.scene = env.backend.scene = S
    else:
        ev.apply_instance_env()
        sc = SceneClient(a.host, timeout_ms=2000)
        info = sc.info()
        S = Vm.scene(info.get("variant"))
        env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=1e9, realtime=True, img_server_ip=a.host,
                       head_rgbd=False, cameras=False, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
    lab = None
    try:
        lab = FarLab(a, cfg, env, GL._Retry(sc), S)
        cases = make_cases(a)
        for c in cases:      # grasp_lab cases are in mat coordinates
            m = S.pelvis_to_mat([c["px"], c["py"], 0.0])
            c.update(x=float(m[0]), y=float(m[1]))
        bx = lab.boxes
        print("[far_lab] variant", (info.get("variant") or {}).get("name"), "boxes:",
              "; ".join(f"{n} {bx[n].text()} near {float(np.asarray(bx[n].front) @ [np.cos(bx[n].yaw), np.sin(bx[n].yaw)]):.3f}"
                        for n in bx.names), "object height", S.object_height(), flush=True)
        meta = {"label": a.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "set": user,
                "variant": info.get("variant"), "far": lab.planner.far.c,
                "grasp": {k: cfg["grasp"][k] for k in ("point_in_palm",)}, "hand": cfg["hand"]}
        lab.run(cases, a.out, meta)
    finally:
        env.close()
        sc.close()
        rows = getattr(lab, "rows", [])
        for r in rows:
            print(_line(r))
        summary(rows)
    return rows


GL._line = _line            # Lab.run logs each case with the module's _line


def _sigterm(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    code = 0
    try:
        main()
    except KeyboardInterrupt:
        code = 130
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)
