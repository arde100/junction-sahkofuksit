"""Offline tests of the grasp lab against the mock world (no sim): control flow, timing, bookkeeping, output format
and error handling -- not grasp quality (the mock has no physics). Run in the sim container:
    aalto.sh py -m unittest aaltonomous/sim/test_grasp_lab.py -v
"""
import json
import os
import tempfile
import unittest

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

    def test_correct_switch_on_the_sagging_mock(self):
        """--correct on / dwell / off on the mock with the sim's arm sag (--mock-sag): on = shift applied, grasp
        point at the close within 1 mm of the planned one; dwell = settle phase and measurement, no shift, the sag
        error stays; off = no settle phase; all three grasps held (4 cm attach radius)."""
        base = ["--mock", "--mock-sag", "--shapes", "cube", "--yaws", "0", "--xs", str(GOOD[0]), "--ys", str(GOOD[1]),
                "--hold", str(HOLD)]
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
