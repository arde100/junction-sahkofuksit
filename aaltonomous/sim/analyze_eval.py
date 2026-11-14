"""ANALYSIS TOOL (privileged sim state; never import this from a policy): merge one evaluation's eval.json with an
object trace (trace_objects.py) and the policy server's log into a per-object account of what happened.

    python -m aaltonomous.sim.analyze_eval runs/eval/<run> [--trace trace.jsonl] [--server-log srv.log] [--failures]

Per trial: score line, one line per object (start pose, outcome, each time it was lifted: when, how high, where it
came down), the policy's events (server log `[classical t=..]` lines, or eval.json `oracle_events`), and a time
budget (executing vs looking/planning/settling). Ends with outcome counts per shape and a failure list.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

LIFT = 0.03          # m above the start height that counts as lifted


def load_trace(path):
    """-> list of episodes {layout, objects: {name: array (n, 6) = t, x, y, z, tilt, speed}}."""
    eps = {}
    for line in open(path):
        r = json.loads(line)
        e = eps.setdefault(r["ep"], {"layout": None, "objects": {}})
        if "layout" in r:
            e["layout"] = r["layout"]
        for o in r.get("objects", []):
            e["objects"].setdefault(o["name"], []).append((r["t"], *o["pos"], o["tilt"], o["speed"]))
    out = []
    for k in sorted(eps):
        e = eps[k]
        e["objects"] = {n: np.array(v) for n, v in e["objects"].items()}
        out.append(e)
    return out


def match_trace(trial, traces, tol=0.015):
    """The trace episode whose layout is this trial's (same colours/shapes in order, positions within tol)."""
    lay = trial.get("layout") or []
    for e in traces:
        tl = e.get("layout") or []
        if len(tl) == len(lay) and all(a["color"] == b["color"] and a.get("shape") == b.get("shape")
                                       and abs(a["x"] - b["x"]) < tol and abs(a["y"] - b["y"]) < tol
                                       for a, b in zip(lay, tl)):
            return e
    return None


def lifts(a):
    """Lift episodes of one object trajectory a (n, 6): [(t_up, t_down, dz_max, (x, y) after coming down)]."""
    t, p = a[:, 0], a[:, 1:4]
    up = np.flatnonzero(p[:, 2] > p[0, 2] + LIFT)
    out = []
    if len(up):
        cuts = np.flatnonzero(np.diff(up) > 1)
        for s, e in zip(np.r_[0, cuts + 1], np.r_[cuts, len(up) - 1]):
            i, j = up[s], up[e]
            k = min(j + 3, len(p) - 1)
            out.append((float(t[i]), float(t[j]), float(p[i:j + 1, 2].max() - p[0, 2]), (float(p[k, 0]), float(p[k, 1]))))
    return out


class GraspAudit:
    """One row per hand-close event of an episode (from the per-tick log episode_NNN.npz): where the grasp point was
    when the hand closed, relative to the nearest traced object, the finger tracking error while closed (the
    candidate "object in hand" signal: fingers blocked by an object cannot reach the commanded closed posture),
    and whether the object was lifted and where it ended up."""

    def __init__(self):
        from aaltonomous.baselines.classical.config import hand_postures, load_config
        from aaltonomous.baselines.classical.kinematics import ArmKinematics
        self.cfg = load_config()
        self.kin = ArmKinematics(self.cfg)
        self.closed = hand_postures(self.cfg)["closed"]

    def rows(self, npz_path, trace_ep):
        d = np.load(npz_path)
        t, aq, ac, hq, hc = d["t"], d["arm_q"], d["arm_q_cmd"], d["hand_q"], d["hand_q_cmd"]
        out = []
        for side in ("left", "right"):
            sl = self.kin.sl(side)
            idx = np.flatnonzero(np.abs(hc[:, sl] - self.closed[sl]).max(1) < 1e-3)
            if not len(idx):
                continue
            cuts = np.flatnonzero(np.diff(idx) > 1)
            for s, e in zip(np.r_[0, cuts + 1], np.r_[cuts, len(idx) - 1]):
                i, j = idx[s], idx[e]                     # closed posture commanded over ticks i..j
                i0 = max(i - 12, 0)                       # ~0.4 s earlier: arm at the grasp pose, hand still open
                p_cmd, _ = self.kin.grasp_pose(side, ac[i])
                self.kin._fk(ac[i])
                fc = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0]
                p_pre, _ = self.kin.grasp_pose(side, aq[i0])
                p, axis = self.kin.grasp_pose(side, aq[i])
                tilt = float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))
                f = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0]
                fyaw = float(np.arctan2(f[1], f[0]))
                err = np.abs(hq[i:j + 1, sl] - hc[i:j + 1, sl]).max(1)       # per tick: worst finger joint
                n = len(err)
                r = {"side": side, "t0": float(t[i]), "t1": float(t[j]), "grasp_xyz": p, "cmd_err_mm": 1e3 * float(np.linalg.norm(p - p_cmd)),
                     "cmd_xyz": [float(v) for v in p_cmd], "pre_err_mm": 1e3 * float(np.linalg.norm(p_pre - p_cmd)),
                     "cmd_q": [float(v) for v in ac[i, sl]], "q_err_pre": [float(v) for v in (aq[i0, sl] - ac[i0, sl])],
                     "q_err_closed": [float(v) for v in (aq[i, sl] - ac[i, sl])],
                     "cmd_fyaw": float(np.degrees(np.arctan2(fc[1], fc[0]))), "lift_mm": 1e3 * float(p[2] - p_cmd[2]),
                     "tilt": tilt, "finger_yaw": float(np.degrees(fyaw)),
                     "sig_0.3s": float(err[min(9, n - 1)]), "sig_1s": float(err[min(30, n - 1)]),
                     "sig_mid": float(err[n // 2]), "sig_end": float(err[max(n - 3, 0)]), "obj": None}
                if trace_ep:
                    best = None
                    for name, a in trace_ep["objects"].items():
                        k0 = max(int(np.searchsorted(a[:, 0], t[i] - 2.0)) - 1, 0)      # before the hand arrived
                        dxy = a[k0, 1:3] - p_cmd[:2]            # object centre relative to the COMMANDED grasp point
                        if best is None or np.linalg.norm(dxy) < np.linalg.norm(best[1]):
                            best = (name, dxy, a, k0)
                    if best is not None and np.linalg.norm(best[1]) < 0.08:
                        name, dxy, a, k0 = best
                        fx = fc[:2] / np.linalg.norm(fc[:2])          # commanded finger direction
                        k1 = min(int(np.searchsorted(a[:, 0], t[j] - 0.3)), len(a) - 1)   # just before the hand opens
                        k2 = min(int(np.searchsorted(a[:, 0], t[j] + 2.0)), len(a) - 1)
                        r.update(obj=name, off_along_mm=1e3 * float(dxy @ fx), off_lat_mm=1e3 * float(dxy @ [-fx[1], fx[0]]),
                                 held_at_open=bool(a[k1, 3] > a[0, 3] + LIFT), z_max_cm=100 * float(a[k0:k1 + 1, 3].max() - a[0, 3]),
                                 after=(float(a[k2, 1]), float(a[k2, 2]), float(a[k2, 3])))
                out.append(r)
        return sorted(out, key=lambda r: r["t0"])

    @staticmethod
    def fmt(r):
        s = (f"      grasp {r['side'][0].upper()} t={r['t0']:5.1f}-{r['t1']:5.1f}s cmd ({r['cmd_xyz'][0]:+.3f},{r['cmd_xyz'][1]:+.3f},"
             f"{r['cmd_xyz'][2]:+.3f}) fyaw {r['cmd_fyaw']:+4.0f} trk pre/closed {r['pre_err_mm']:2.0f}/{r['cmd_err_mm']:2.0f}mm "
             f"(up {r['lift_mm']:+3.0f}) tilt {r['tilt']:2.0f}  "
             f"sig .3s/1s/mid/end {r['sig_0.3s']:.3f}/{r['sig_1s']:.3f}/{r['sig_mid']:.3f}/{r['sig_end']:.3f}")
        if r["obj"]:
            s += (f"  {r['obj']} off along/lat {r['off_along_mm']:+4.0f}/{r['off_lat_mm']:+4.0f}mm  zmax +{r['z_max_cm']:.0f}cm  "
                  f"{'HELD' if r['held_at_open'] else 'not held'} -> ({r['after'][0]:+.2f},{r['after'][1]:+.2f},{r['after'][2]:+.2f})")
        return s


def server_events(path):
    """{seed: [event lines]} from a classical policy server log."""
    out, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.search(r"reset task=\S+ seed=(\d+)", line)
        if m:
            cur = out.setdefault(int(m.group(1)), [])
            continue
        m = re.search(r"\[(?:classical|oracle)[^\]]*\].*", line)
        if m and cur is not None:
            cur.append(m.group(0).rstrip())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--failures", action="store_true", help="only print trials with an object that is not correct")
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--grasps", action="store_true", help="one line per hand-close event (needs pinocchio; see GraspAudit)")
    ap.add_argument("--grasps-json", default=None, help="also write every grasp row here (json list)")
    a = ap.parse_args(argv)
    ev = json.loads((Path(a.run) / "eval.json").read_text())
    traces = load_trace(a.trace) if a.trace else []
    sev = server_events(a.server_log) if a.server_log else {}
    audit = GraspAudit() if a.grasps or a.grasps_json else None
    all_grasps = []
    by_shape, fails = {}, []
    for tr in ev["trials"]:
        e = match_trace(tr, traces)
        names = [o["name"] for o in tr["outcomes"]]
        bad = [o for o in tr["outcomes"] if o["outcome"] != "correct"]
        for o in tr["outcomes"]:
            shape = o["name"].split("_")[2]
            by_shape.setdefault(shape, {}).setdefault(o["outcome"], 0)
            by_shape[shape][o["outcome"]] += 1
        events = sev.get(tr["seed"]) or [f"[t={t:6.2f}] {m}" for t, m in tr.get("oracle_events", [])]
        durs = [float(m.group(1)) for m in (re.search(r"(\d+\.\d)s  safety_clips", x) for x in events) if m]
        grasps = []
        if audit is not None:
            npz = Path(a.run) / f"episode_{tr['episode']:03d}.npz"
            if npz.exists():
                grasps = audit.rows(npz, e)
                all_grasps += [{**g, "seed": tr["seed"], "grasp_xyz": [float(v) for v in g["grasp_xyz"]]} for g in grasps]
        if a.failures and not bad:
            continue
        print(f"\n=== trial {tr['episode']} seed {tr['seed']}: {tr['n_correct']}/{tr['n_objects']} correct, score "
              f"{tr['score']:+d}, sim {tr.get('sim_s')} s, end={tr['end_reason']}, clips="
              f"{(tr.get('safety') or {}).get('clipped_steps')}, hz={tr.get('command_hz')}, rt_hz={tr.get('policy_hz_rt')}"
              f"{'' if e or not traces else '  (no trace)'}")
        if durs:
            print(f"    time: {len(durs)} plans executed, {sum(durs):.1f} s moving, "
                  f"{(tr.get('sim_s') or 0) - sum(durs):.1f} s looking / planning / settling / idle")
        fin = {o["name"]: o for o in tr.get("final_objects", [])}
        for lo, name, o in zip(tr["layout"], names, tr["outcomes"]):
            p0 = None
            desc = ""
            if e and name in e["objects"]:
                arr = e["objects"][name]
                p0 = arr[0, 1:3]
                ls = lifts(arr)
                desc = "; ".join(f"lift {u:.1f}-{d:.1f}s +{100 * dz:.0f}cm down@({x:+.2f},{y:+.2f})" for u, d, dz, (x, y) in ls)
                moved = np.flatnonzero(np.linalg.norm(arr[:, 1:3] - arr[0, 1:3], axis=1) > 0.01)
                if len(moved) and not ls:
                    desc = f"pushed from t={arr[moved[0], 0]:.1f}s, never lifted"
                elif not len(moved):
                    desc = "never moved"
            f = fin.get(name, {}).get("pos")
            print(f"    {name:22s} yaw={np.degrees(lo.get('yaw', 0)):+4.0f} "
                  f"start=({'%+.3f,%+.3f' % tuple(p0) if p0 is not None else '?'}) -> {o['outcome']:8s} "
                  f"final=({'%+.2f,%+.2f,%+.3f' % tuple(f) if f else '?'})  {desc}")
            if o["outcome"] != "correct":
                fails.append((tr["seed"], name, o["outcome"], desc))
        if a.grasps:
            for g in grasps:
                print(audit.fmt(g))
        if not a.no_events:
            for x in events:
                print("      " + x[:230])
    if a.grasps_json:
        Path(a.grasps_json).write_text(json.dumps(all_grasps, indent=1))
    if all_grasps and traces:
        known = [g for g in all_grasps if g["obj"]]
        held = [g for g in known if g["held_at_open"]]
        lost = [g for g in known if not g["held_at_open"]]
        print(f"\n##### grasps: {len(all_grasps)} hand closes, {len(known)} matched to an object, {len(held)} held until the "
              f"hand opened, {len(lost)} not")
        for key in ("sig_0.3s", "sig_1s", "sig_mid", "sig_end"):
            h, m = [g[key] for g in held], [g[key] for g in lost]
            if h and m:
                print(f"  {key:9s} held: min {min(h):.3f} med {np.median(h):.3f} max {max(h):.3f}   "
                      f"not held: min {min(m):.3f} med {np.median(m):.3f} max {max(m):.3f}")
    s = ev["summary"]
    print(f"\n##### {a.run}\n  variant {s.get('variant')}  objects {s.get('objects_correct')}/{s.get('objects')}  "
          f"score_frac {s.get('score_frac')}  outcomes {s.get('outcome_counts')}  ends {s.get('end_reasons')}  "
          f"gate {s.get('gate', {}).get('passed')}  clips {s.get('safety_clipped_steps_total')}  "
          f"hz_min {s.get('command_hz_min')}  rt_hz_min {s.get('policy_hz_rt_min')}")
    print(f"  sim_s per trial: {[round(t.get('sim_s') or 0) for t in ev['trials']]}")
    print("  by shape: " + "  ".join(f"{k}: {v}" for k, v in sorted(by_shape.items())))
    print("  not correct:")
    for seed, name, out, desc in fails:
        print(f"    seed {seed} {name:22s} {out:8s} {desc}")


if __name__ == "__main__":
    main()
