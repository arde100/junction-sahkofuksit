"""Offline tests of the grasp lab against the mock world (no sim): control flow, timing, bookkeeping, output format
and error handling -- not grasp quality (the mock has no physics). Run in the sim container:
    aalto.sh py -m unittest aaltonomous/sim/test_grasp_lab.py -v
"""
import json
import os
import tempfile
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene
from aaltonomous.sim import grasp_lab as gl

GOOD = (0.39, 0.155)     # mat xy -> pelvis (0.295, -0.06): right arm
FAR = (0.45, 0.50)       # pelvis (0.64, 0.00): beyond the arms' reach (behind the centre bin) -> no plan
HOLD = 1.0


class FlakyScene(gl.MockScene):
    """Scene-service faults: one TimeoutError on the first state() after the first placement (must be retried) and a
    RuntimeError as soon as a cylinder has left the counter (mid-lift, hand low: the case is recorded, the hand raised
    straight up, the arms homed)."""

    def __init__(self, env):
        super().__init__(env)
        self.resets, self.calls, self.kind, self.timeouts, self.errors = 0, 0, None, 0, 0

    def reset(self, layout):
        if layout:
            self.resets, self.calls, self.kind = self.resets + 1, 0, layout[0]["shape"]
        return super().reset(layout)

    def state(self):
        self.calls += 1
        if self.resets == 1 and self.calls == 1 and not self.timeouts:
            self.timeouts += 1
            raise TimeoutError("injected timeout")
        s = super().state()
        sc = self.env.scene
        if (self.kind == "cylinder" and not self.errors and s["objects"]
                and s["objects"][0]["pos"][2] > sc.counter_z + sc.object_height() / 2 + 0.005):
            self.errors += 1
            raise RuntimeError("injected scene error")
        return s


def case(shape, xy, **kw):
    return {"shape": shape, "color": "red", "yaw": 0.0, "x": xy[0], "y": xy[1], "rep": 0, **kw}


class TestGraspLabMock(unittest.TestCase):
    def test_cli_sweep_rows_and_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "lab.jsonl")
            argv = ["--mock", "--shapes", "cube,cylinder", "--yaws", "0", "--xs", str(GOOD[0]), "--ys", str(GOOD[1]),
                    "--hold", str(HOLD), "--label", "t", "--out", out]
            rows = gl.main(argv)
            with open(out) as f:
                lines = [json.loads(s) for s in f]
        self.assertEqual([r["type"] for r in lines], ["meta", "case", "case"])
        self.assertEqual(lines[0]["label"], "t")
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIsNone(r["error"], r.get("traceback"))
            self.assertEqual(r["outcome"], "ok", r)
            self.assertTrue(r["plan_ok"] and r["lifted"] and r["held"] and r["released"])
            self.assertEqual(r["side"], "right")
            self.assertEqual(r["clips"], 0)
            self.assertAlmostEqual(r["phase_s"]["hold"], HOLD, places=2)
            self.assertNotIn("home", r["phase_s"])                  # home move cut off
            self.assertLess(r["displaced_mm"], 5.0)
            self.assertAlmostEqual(r["sig_1s"], 0.036, delta=0.01)  # mock_world held_gap: fingers stop short
            self.assertGreater(r["rise_lift_mm"], 80)
            self.assertLess(abs(r["carried_mm"]), 1.0)
            self.assertTrue(3.0 < r["sim_s"] < 15.0, r["sim_s"])
            self.assertGreaterEqual(r["sim_s"], r["exec_plan_s"])
            self.assertTrue(r["trace"] and {t[1] for t in r["trace"]} >= {"settle", "lift", "hold", "open", "post"})
        self.assertEqual(rows[0]["how"], "home")
        self.assertEqual(rows[1]["how"], "direct")                  # no trip home between cases

    def test_plan_only(self):
        rows = gl.main(["--plan-only", "--shapes", "cube", "--yaws", "20", "--xs", str(GOOD[0]), "--ys", str(GOOD[1])])
        self.assertTrue(rows[0]["plan_ok"], rows[0]["plan_reason"])
        self.assertNotIn("lifted", rows[0])

    def test_face_align_flags_map_onto_the_config(self):
        """--face-align / --no-face-align / --tri-phi -> grasp.face_align.enabled / .triangle_phi_deg (--set last);
        $AALTO_TRI_PHI -> the oracle's default phi; --finger-yaw forces the direction; plan-only rows say so."""
        import importlib
        from aaltonomous.sim import oracle
        cfg, _ = gl.build_cfg(gl.parser().parse_args(["--plan-only"]))
        self.assertTrue(cfg["grasp"]["face_align"]["enabled"])
        self.assertEqual(cfg["grasp"]["face_align"]["triangle_phi_deg"], oracle.ORACLE_OVERRIDES[
            "grasp.face_align.triangle_phi_deg"])
        cfg, _ = gl.build_cfg(gl.parser().parse_args(["--no-face-align", "--tri-phi", "60"]))
        self.assertEqual((cfg["grasp"]["face_align"]["enabled"], cfg["grasp"]["face_align"]["triangle_phi_deg"]),
                         (False, 60.0))
        cfg, _ = gl.build_cfg(gl.parser().parse_args(["--tri-phi", "60", "--set",
                                                      '{"grasp.face_align.triangle_phi_deg": 90}']))
        self.assertEqual(cfg["grasp"]["face_align"]["triangle_phi_deg"], 90)
        old = os.environ.get("AALTO_TRI_PHI")
        try:
            os.environ["AALTO_TRI_PHI"] = "75"
            importlib.reload(oracle)
            self.assertEqual(oracle.ORACLE_OVERRIDES["grasp.face_align.triangle_phi_deg"], 75.0)
        finally:
            if old is None:
                os.environ.pop("AALTO_TRI_PHI", None)
            else:
                os.environ["AALTO_TRI_PHI"] = old
            importlib.reload(oracle)
        base = ["--plan-only", "--shapes", "triangle", "--yaws", "0", "--xs", str(GOOD[0]), "--ys", str(GOOD[1])]
        on, off = gl.main(base + ["--tri-phi", "105"])[0], gl.main(base + ["--no-face-align"])[0]
        self.assertTrue(on["plan_ok"] and off["plan_ok"], (on["plan_reason"], off["plan_reason"]))
        self.assertEqual((on["tri_phi"], on["face_align"], off["face_align"]), (105.0, True, False))
        self.assertTrue(on["plan_note"].startswith("face-aligned"), on["plan_note"])
        self.assertFalse(off["plan_note"].startswith("face-aligned"), off["plan_note"])
        forced = gl.main(base + ["--finger-yaw", "30"])[0]
        self.assertTrue(forced["plan_ok"], forced["plan_reason"])
        self.assertAlmostEqual(forced["finger_yaw_deg"], 30.0, delta=12.0)

    def test_profile_switch(self):
        """--profile oracle (default) = ORACLE_OVERRIDES as before; --profile base = the classical config's base values
        (the robot profile: base hand postures, grasp point, sag prior, face_align off) with the flags and --set on
        top; --profile <name> = that config profile; rows and meta say which."""
        from aaltonomous.baselines.classical.config import load_config
        from aaltonomous.sim import oracle
        cfg, _ = gl.build_cfg(gl.parser().parse_args([]))
        self.assertEqual(cfg["hand"]["closed"], oracle.ORACLE_OVERRIDES["hand.closed"])
        self.assertEqual(cfg["grasp"]["point_in_palm"], oracle.ORACLE_OVERRIDES["grasp.point_in_palm"])
        self.assertEqual(cfg["motion"]["arm_max_vel"], oracle.ORACLE_OVERRIDES["motion.arm_max_vel"])
        self.assertTrue(cfg["grasp"]["face_align"]["enabled"])
        self.assertEqual(cfg, gl.build_cfg(gl.parser().parse_args(["--profile", "oracle"]))[0])
        base = load_config(profile="none")
        cfg, _ = gl.build_cfg(gl.parser().parse_args(["--profile", "base"]))
        for k in ("hand", "grasp", "motion", "sag", "verify", "home_arm_q"):
            self.assertEqual(cfg[k], base[k], k)
        cfg, user = gl.build_cfg(gl.parser().parse_args(
            ["--profile", "base", "--face-align", "--ff", "off", "--correct", "off", "--set",
             '{"hand.closed": [0, 0.5, 1.0, -1.0, -1.4, -1.0, -1.4], "motion.arm_max_vel": 0.8}']))
        self.assertEqual((cfg["grasp"]["face_align"]["enabled"], cfg["sag"]["enabled"], cfg["grasp"]["correct"]["enabled"]),
                         (True, False, False))
        self.assertEqual((cfg["hand"]["closed"], cfg["motion"]["arm_max_vel"]), ([0, 0.5, 1.0, -1.0, -1.4, -1.0, -1.4], 0.8))
        self.assertEqual(cfg["hand"]["open"], base["hand"]["open"])
        self.assertEqual(user["motion.arm_max_vel"], 0.8)
        sim = load_config(profile="sim")
        self.assertEqual(gl.build_cfg(gl.parser().parse_args(["--profile", "sim"]))[0]["hand"], sim["hand"])
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "lab.jsonl")
            rows = gl.main(["--plan-only", "--profile", "base", "--shapes", "cube", "--yaws", "0", "--xs", str(GOOD[0]),
                            "--ys", str(GOOD[1]), "--out", out])
            with open(out) as f:
                meta = json.loads(f.readline())
        self.assertTrue(rows[0]["plan_ok"], rows[0]["plan_reason"])
        self.assertEqual((rows[0]["profile"], rows[0]["face_align"]), ("base", False))
        self.assertEqual((meta["profile"], meta["cfg"]["hand"]), ("base", base["hand"]))
        r = gl.main(["--mock", "--profile", "base", "--shapes", "cube", "--yaws", "0", "--xs", str(GOOD[0]), "--ys",
                     str(GOOD[1]), "--hold", str(HOLD), "--time-limit", "80"])[0]
        self.assertEqual((r["outcome"], r["clips"], r["profile"]), ("ok", 0, "base"), r.get("error"))
        self.assertGreater(r["sig_fs_1s"], 0.0)    # the robot's verification signal is recorded too
        # joints per phase (LEFT convention) inside the base profile's ik.joint_range (W1), measured at the close
        lo, hi = (np.degrees(np.asarray([v[i] for v in base["ik"]["joint_range"].values()], float)) for i in (0, 1))
        for ph, (mn, mx) in r["joints_deg"].items():
            if ph not in ("home", "to_pregrasp"):   # (moves from the start pose: the lab homes to the sim's ARM_HOME)
                self.assertTrue(np.all(np.asarray(mn) >= lo - 1) and np.all(np.asarray(mx) <= hi + 1), (ph, mn, mx))
        self.assertEqual(len(r["joints_close_meas_deg"]), 7)

    def test_neighbour(self):
        """--neighbour: a second object placed next to every case object; the case object is the one at the case
        position (followed by name), the other is reported (nb_disp_mm / nb_tilt_deg) and planned around when
        neighbours.enabled (a neighbour where the hand must go -> no plan, the case is recorded as such)."""
        base = ["--mock", "--profile", "base", "--shapes", "cube", "--yaws", "0", "--xs", str(GOOD[0]), "--ys",
                str(GOOD[1]), "--hold", str(HOLD), "--time-limit", "80"]
        r = gl.main(base + ["--neighbour", "110,0", "--set", '{"neighbours.enabled": true}'])[0]
        self.assertEqual((r["outcome"], r["clips"]), ("ok", 0), r.get("error") or r.get("plan_reason"))
        self.assertTrue(r["nb_planned"])
        self.assertLess(r["nb_disp_mm"], 1.0)                          # the mock moves nothing it does not grasp
        self.assertLess(r["displaced_mm"], 5.0)                        # (the case object, not the neighbour)
        off = gl.main(base + ["--neighbour", "110,0", "--set", '{"neighbours.enabled": false}'])[0]   # (base: on)
        self.assertEqual(off["outcome"], "ok", off.get("error"))
        self.assertFalse(off["nb_planned"])
        cases = gl.make_cases(gl.parser().parse_args(["--shapes", "cube", "--yaws", "0", "--xs", "0.4", "--ys", "0.15",
                                                      "--neighbour", "60,0;0,-60", "--neighbour-shape", "triangle"]),
                              Scene.load())
        self.assertEqual([(n["shape"], round(n["x"], 3), round(n["y"], 3)) for n in cases[0]["nb"]],
                         [("triangle", 0.46, 0.15), ("triangle", 0.4, 0.09)])
        # 7 cm beside it: the base cage's hand (fingers +-41.5 mm wide, sphere model ~13 mm more) comes too close
        blk = gl.main(["--plan-only"] + base[1:] + ["--neighbour", "70,0", "--set", '{"neighbours.enabled": true}'])[0]
        self.assertFalse(blk["plan_ok"])
        self.assertIn("mm from", blk["plan_reason"])

    def test_correct_switch_on_the_sagging_mock(self):
        """--correct on / dwell / off on the mock with the sim's arm sag (--mock-sag): on = shift applied, grasp
        point at the close within 1 mm of the planned one; dwell = settle phase and measurement, no shift, the sag
        error stays; off = no settle phase; all three grasps held (4 cm attach radius)."""
        base = ["--mock", "--mock-sag", "--shapes", "cube", "--yaws", "0", "--xs", str(GOOD[0]), "--ys", str(GOOD[1]),
                "--hold", str(HOLD), "--ff", "off"]   # (the correction alone; with the feed-forward: the test below)
        rows = {m: gl.main(base + ["--correct", m])[0] for m in ("on", "dwell", "off")}
        for m, r in rows.items():
            with self.subTest(m):
                self.assertEqual((r["outcome"], r["clips"]), ("ok", 0), r.get("error"))
        on, dwell, off = rows["on"], rows["dwell"], rows["off"]
        self.assertTrue(on["corr"]["applied"], on["corr"])
        self.assertGreater(on["corr"]["err_norm_mm"], 4.0)
        self.assertLess(on["err_close_norm_mm"], 1.0)
        self.assertEqual((dwell["corr"]["applied"], dwell["corr"]["why"][:5]), (False, "dwell"))
        self.assertGreater(dwell["err_close_norm_mm"], 4.0)
        self.assertNotIn("corr", off)
        self.assertNotIn("settle", off["phase_s"])
        self.assertAlmostEqual(on["phase_s"]["settle"], 0.4, places=2)
        self.assertGreater(off["err_close_norm_mm"], 4.0)
        self.assertEqual(len(on["err_close_along_across_up_mm"]), 3)
        self.assertEqual(on["err_pre_corr_norm_mm"], on["corr"]["err_norm_mm"])
        self.assertNotIn("err_pre_corr_norm_mm", off)

    def test_ff_switch_on_the_sagging_mock(self):
        """--ff on (the oracle's default: profile sim's prior): the sagging mock arm is at the planned grasp point
        before any correction (err_pre_corr < 0.5 mm, the correction refused below min_err), 0 clips, held; --ff
        off: the sag (> 4 mm) is what the correction removes. --ff on with --correct off: < 0.5 mm at the close."""
        base = ["--mock", "--mock-sag", "--shapes", "cube", "--yaws", "0", "--xs", str(GOOD[0]), "--ys", str(GOOD[1]),
                "--hold", str(HOLD)]
        on, off = gl.main(base + ["--ff", "on", "--correct", "dwell"])[0], gl.main(base + ["--ff", "off"])[0]
        alone = gl.main(base + ["--ff", "on", "--correct", "off"])[0]
        for r in (on, off, alone):
            self.assertEqual((r["outcome"], r["clips"]), ("ok", 0), r.get("error"))
        self.assertEqual((on["ff"], off["ff"]), (True, False))
        self.assertLess(on["err_pre_corr_norm_mm"], 0.5)
        self.assertLess(on["err_close_norm_mm"], 0.5)
        self.assertGreater(off["err_pre_corr_norm_mm"], 4.0)
        self.assertLess(alone["err_close_norm_mm"], 0.5)
        self.assertEqual(gl.build_cfg(gl.parser().parse_args(["--plan-only"]))[0]["sag"]["enabled"], True)
        self.assertEqual(gl.build_cfg(gl.parser().parse_args(["--plan-only", "--ff", "off"]))[0]["sag"]["enabled"],
                         False)

    def test_plan_failure_miss_and_errors(self):
        a = gl.parser().parse_args(["--mock", "--hold", str(HOLD)])
        cfg, _ = gl.build_cfg(a)
        env, _ = gl.make_mock(cfg)
        sc = FlakyScene(env)
        lab = gl.Lab(a, cfg, env, gl._Retry(sc, log=lambda *x: None), log=lambda *x: None)
        cases = [case("cube", GOOD), case("cube", FAR), case("cube", GOOD, xy_error=(0.0, 60.0)),
                 case("cylinder", GOOD), case("triangle", GOOD)]
        rows = lab.run(cases)
        ok, far, miss, err, after = rows
        self.assertEqual(sc.timeouts, 1)                            # retried, case unaffected
        self.assertEqual(ok["outcome"], "ok", ok.get("error"))
        self.assertFalse(far["plan_ok"])                            # plan failure: recorded, no sim time, no reset
        self.assertEqual(far["outcome"], "plan")
        self.assertTrue(far["plan_reason"])
        self.assertLess(far["sim_s"], 1e-6)
        self.assertEqual(sc.resets, 4)                              # the FAR case never placed its object
        self.assertTrue(miss["plan_ok"])                            # 6 cm perception error: grasp closes on air
        self.assertFalse(miss["lifted"])
        self.assertEqual(miss["outcome"], "lift")
        self.assertTrue(miss["released"])
        self.assertLess(miss["displaced_mm"], 1.0)
        self.assertIn("injected scene error", err["error"])         # exception: recorded, arms homed, run goes on
        self.assertEqual(err["outcome"], "error")
        self.assertEqual(err["trace"][-1][1], "lift")
        self.assertEqual(err["recovery"], "raised right, homed")
        self.assertEqual(after["outcome"], "ok", after.get("error"))
        self.assertEqual(after["how"], "home")
        self.assertEqual(env.filter.mode, "ok")
        self.assertEqual(env.backend.objects, [])                   # parked at the end


if __name__ == "__main__":
    unittest.main()
