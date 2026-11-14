"""Runtime side of the Aalto task, called once per env step from an observation term (main thread, so it may
write sim state safely).

1. Head RGB-D: the head pinhole camera's rgb + distance_to_image_plane are published on tcp :55560 in the exact
   wire format of interface/pc2/head_rgbd_server.py (uint16 mm depth, 0 = invalid), so AaltoEnv reads sim and
   real depth through the same HeadRGBDClient (DDSBackend head_rgbd=True).
2. Scene service: zmq REP on tcp :55580, JSON in / JSON out (client: aaltonomous/sim/scene_client.py):
     {"cmd": "reset", "layout": [{color, shape, x, y, yaw}, ...]}   place a layout (mat coords), park the rest;
                                                                    an entry may say lying: true (+ ridge_yaw rad,
                                                                    pelvis frame; side 0-2) for a triangle / cylinder
                                                                    on its side; reply "layout" = what was placed
     {"cmd": "reset_random", "seed": int|null}                      sample + place (scene.yaml layout knobs)
                                                                    reset_random applies the variant's `layout:`
                                                                    (messy.py: lying prisms, clusters); `reset` places
                                                                    exactly what it is given (the evaluator applies
                                                                    `layout:` itself before it sends)
     {"cmd": "state"}                                               objects: name, color, shape, active, pos
                                                                    (pelvis frame), quat_xyzw, speed (m/s), tilt_deg,
                                                                    lying, ridge_yaw (from the pose); "layout" placed
     {"cmd": "evaluate", "task": "sort_fixed"|"sort_language"|"insert_shapes", "mapping": {...}|null}
                                                                    spec §3 outcomes + score (Scene checker)
     {"cmd": "ping"}
     {"cmd": "info"}                                                DR config (null = off), ports, hand_gains,
                                                                    variant (AALTO_VARIANT, variant.py; null = off)
     {"cmd": "randomize", "seed": int|null}                         DR: sample + apply one parameter set (dr.py);
                                                                    no-op returning params null when DR is off
     {"cmd": "robot", "joints": regex|null}                         live articulation per joint (by NAME): pos,
                                                                    pos_target, vel, stiffness, damping, armature,
                                                                    effort_limit, friction, pos_limits, soft_limits
                                                                    (probe_variant.py checks gains / wiring / sag),
                                                                    joint_limit_overrides (the variant's request);
                                                                    + ground truth the POLICY never gets: waist angles,
                                                                    links (pelvis / torso / d435 / wrists / palms) and
                                                                    the head camera in the pelvis frame, the pelvis in
                                                                    the table frame, body_yaw_deg, sim_t (body_info)
     {"cmd": "geometry"}                                            what the STAGE holds (pelvis frame): bin part /
                                                                    counter / lid prim poses, yaw, sizes; object mesh
                                                                    sizes + masses; counter roughness, colours; head
                                                                    camera pose; lights (probe_variant.py scene)
   Ports via env AALTO_RGBD_PORT / AALTO_SCENE_PORT; 0 disables.
   ping / info also return the sim clock: sim_t (s), rtf (sim/wall over the last ~10 s window), rtf_mean.
4. Sim clock (real-time governor): zmq PUB on tcp :AALTO_CLOCK_PORT (default scene port + 1 = 55581), one
   b"<sim_t float64 LE><wall float64 LE>" message per env step. The sim runs slower than real time (rendering);
   AaltoEnv's sim backend paces its 30 Hz ticks on this clock instead of wall time (interface/backends.py SimClock),
   so the physics sees commands at the intended rate whatever the machine load. Logs a warning when RTF < 0.95.
3. Domain randomization (dr.py), only if AALTO_DR names a config: lights, mat/counter/bin/object colours,
   object mass/friction, layout pose jitter, head camera pose + RGB-D noise (noise on :55560 only; the teleimager
   mono head stream and the wrist cameras get the lighting/colour DR but no pixel noise).
"""
from __future__ import annotations

import json
import os
import re
import struct
import time

import numpy as np
import torch

from . import dr, messy, variant
from .layout import PELVIS_W, assign, evaluate, object_names, parse_name, random_layout

_RT = None


def _t(x):
    return x.torch if hasattr(x, "torch") else x


class AaltoRuntime:
    def __init__(self, env):
        import zmq
        self.variant = variant.active()
        self.sc = variant.scene(self.variant)     # scene.yaml (+ the variant's overlay): placement and scoring
        self.body_yaw_deg = variant.body_yaw_base_deg(self.variant)   # None = no body_yaw (robot at the init pose)
        # head camera follower (camera_follow): a camera prim under a robot link does NOT move with the link in this
        # sim (R3 live: torso mount and yawed pelvis both kept the spawn pose), so with a torso / d435 mount or a body
        # yaw its world pose is written every step from the parent link's live pose x the mount (+ DR jitter).
        # Organisers' setup (pelvis mount, no body_yaw): never touched.
        self.cam_parent, self.cam_mount = variant.camera_mount(self.variant, head_camera_pelvis_T(None))
        self.cam_follow = "front_camera" in env.scene.keys() and camera_follows(self.variant)
        self.cam_jitter, self._cam_body, self._cam_err, self._cam_usd, self._cam_last = None, None, None, None, None
        self.cam_image_check = None      # depth self-check: does the rendered image follow (camera_image_check)
        if self.cam_follow:
            print(f"[aalto] head camera follows {self.cam_parent} every step (mount t "
                  f"{np.round(self.cam_mount[:3, 3], 6).tolist()})")
        self.names = object_names(self.sc)
        self.layout, self.active = [], []
        self.ctx = zmq.Context.instance()
        self.rep = self.rgbd = None
        sp = int(os.environ.get("AALTO_SCENE_PORT", "55580"))
        if sp:
            self.rep = self.ctx.socket(zmq.REP)
            self.rep.setsockopt(zmq.LINGER, 0)
            self.rep.bind(f"tcp://*:{sp}")
            print(f"[aalto] scene service on tcp :{sp}")
        rp = int(os.environ.get("AALTO_RGBD_PORT", "55560"))
        if rp and "front_camera" in env.scene.keys():
            from aaltonomous.interface.pc2.head_rgbd_server import HeadRGBDServer
            self._latest = None
            self.rgbd = HeadRGBDServer(self, port=rp, ctx=self.ctx)
            print(f"[aalto] head RGB-D on tcp :{rp}")
        self.last_frame = None
        self.zmq = zmq
        self.dr_cfg = dr.load()
        self.dr_params, self._dr_base, self._cam_rng = None, None, None
        cp = int(os.environ.get("AALTO_CLOCK_PORT", str(sp + 1 if sp else 0)))
        self.clock = None
        if cp:
            self.clock = self.ctx.socket(zmq.PUB)
            self.clock.setsockopt(zmq.LINGER, 0)
            self.clock.setsockopt(zmq.SNDHWM, 10)
            self.clock.bind(f"tcp://*:{cp}")
            print(f"[aalto] sim clock on tcp :{cp}")
        self.step_dt = float(env.step_dt)
        self.sim_t = 0.0
        self.rtf = self.rtf_mean = None
        self._win = self._start = None            # (wall, sim_t) at window / clock start
        self.ports = {"scene": sp, "rgbd": rp, "clock": cp}
        if self.dr_cfg:
            print(f"[aalto] domain randomization ON: {self.dr_cfg['_path']}")

    # HeadRGBDServer source interface
    def read(self):
        return self._latest

    # ------------------------------------------------------------------ scene
    def place(self, env, layout, sampled=False):
        """sampled: a layout this service drew (reset_random) -> apply the variant's `layout:` (lying, clusters).
        An explicit `reset` is placed as given (check_scene drops, grasp labs; the evaluator applies it itself)."""
        if sampled:
            layout = messy.messy_layout(self.variant, self.sc, layout)
        if self.dr_params:      # DR pose jitter, not for cluster members (it would push them into each other)
            jit = dr.jitter_layout(layout, self.dr_params)
            layout = [dict(o) if o.get("cluster") else j for o, j in zip(layout, jit)]
        poses, active = assign(self.sc, layout)
        origin = _t(env.scene.env_origins)[0]
        dev = origin.device
        for n in self.names:
            p, q = poses[n]
            pose = torch.tensor([[*(np.asarray(p) + PELVIS_W), *q]], dtype=torch.float32, device=dev)
            pose[:, :3] += origin
            obj = env.scene[n]
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=dev))
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev),
                                                 env_ids=torch.tensor([0], device=dev))
        yaw = variant.body_yaw(self.variant, layout)    # None: no body_yaw in the variant, or a parking reset
        if yaw is not None:
            self.set_body_yaw(env, yaw)
        self.layout, self.active = list(layout), active
        return active

    def set_body_yaw(self, env, yaw):
        """Variant body_yaw: turn the whole (fixed-base) robot about the vertical through the pelvis, relative to the
        table. Objects, boxes and scoring stay in the table frame; the cameras turn with the robot."""
        origin = _t(env.scene.env_origins)[0]
        dev = origin.device
        q = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))           # xyzw
        pose = torch.tensor([[*PELVIS_W, *q]], dtype=torch.float32, device=dev)
        pose[:, :3] += origin
        env.scene["robot"].write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=dev))
        self.body_yaw_deg = round(float(np.degrees(yaw)), 4)
        if self.cam_follow:     # the head camera follows the turned pelvis from now on (camera_follow, every step)
            self.drive_camera(env)

    def drive_camera(self, env):
        """Write the head camera's world pose = parent link's live pose (PhysX) x mount, then DR's camera jitter
        (world frame, as _apply_dr defines it) on top."""
        robot = env.scene["robot"]
        if self._cam_body is None:
            self._cam_body = list(robot.data.body_names).index(self.cam_parent)
        i = self._cam_body
        pos = _t(robot.data.body_link_pos_w)[0, i].cpu().numpy().astype(float)
        q = _t(robot.data.body_link_quat_w)[0, i].cpu().numpy().astype(float)
        p, R = camera_world_pose(pos, q, self.cam_mount, self.cam_jitter)
        cam = env.scene["front_camera"]
        cam.set_world_poses(torch.tensor([p], dtype=torch.float32),
                            torch.tensor([R_quat_xyzw(R)], dtype=torch.float32), convention="ros")
        # R3c live: with only the pose above (Fabric world matrix + the sensor's data) the IMAGE did not move: write
        # the camera prim's USD transform too. The prim is world-level (scene_cfg), so USD and Fabric agree.
        self._write_camera_usd(cam, p, R)
        self._cam_last = (p, R)

    def _write_camera_usd(self, cam, p, R):
        from pxr import Gf, Usd, UsdGeom
        if self._cam_usd is None:
            prim = _stage().GetPrimAtPath(_env0(cam.cfg.prim_path))
            parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            op = UsdGeom.Xformable(prim).MakeMatrixXform()             # replaces the spawn's translate / orient ops
            self._cam_usd = (op, np.linalg.inv(np.array(parent, float).T))
        op, inv_parent = self._cam_usd
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R @ np.diag([1.0, -1.0, -1.0]), p      # ROS optical -> USD camera (looks -z, y up)
        L = inv_parent @ T
        op.Set(Gf.Matrix4d(*L.T.flatten().tolist()))                  # Gf: row-vector convention

    def state(self, env):
        origin = _t(env.scene.env_origins)[0].cpu().numpy()
        out = []
        for n in self.names:
            d = env.scene[n].data
            p = _t(d.root_pos_w)[0].cpu().numpy() - origin - PELVIS_W
            q = _t(d.root_quat_w)[0].cpu().numpy()
            v = _t(d.root_lin_vel_w)[0].cpu().numpy()
            color, shape, _ = parse_name(n)
            out.append({"name": n, "color": color, "shape": shape, "active": n in self.active,
                        "pos": [round(float(x), 5) for x in p], "quat_xyzw": [round(float(x), 5) for x in q],
                        "speed": round(float(np.linalg.norm(v)), 5), **messy.pose_truth(q)})
        return out

    def handle(self, env, req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "t": time.time(), **self.clock_info()}
        if cmd == "reset":
            return {"ok": True, "active": self.place(env, req["layout"]), "layout": self.layout,
                    "body_yaw_deg": self.body_yaw_deg}
        if cmd == "info":
            d = None if not self.dr_cfg else {"name": self.dr_cfg.get("name"), "path": self.dr_cfg["_path"]}
            return {"ok": True, "dr": d, "ports": self.ports, "n_objects": len(self.names),
                    "hand_gains": os.environ.get("AALTO_HAND_GAINS") or None, "task": os.environ.get("TASK") or None,
                    "variant": variant.summary(self.variant), **self.clock_info()}
        if cmd == "robot":
            try:
                body = body_info(env)
            except Exception as e:   # never lose the joint table over the extras
                body = {"body_errors": {"body_info": f"{type(e).__name__}: {e}"}}
            return {"ok": True, **robot_info(env, req.get("joints")), **body, "body_yaw_deg": self.body_yaw_deg,
                    "camera_follow": self.cam_follow, "camera_image_check": self.cam_image_check,
                    "joint_limit_overrides": variant.joint_limits(self.variant), **self.clock_info()}
        if cmd == "geometry":
            return {"ok": True, **geometry_info(env, self)}
        if cmd == "randomize":
            return {"ok": True, "params": self.randomize(env, req.get("seed"))}
        if cmd == "reset_random":
            if self.dr_cfg:
                self.randomize(env, req.get("seed"))
            lay = random_layout(self.sc, req.get("seed"))
            return {"ok": True, "active": self.place(env, lay, sampled=True), "layout": self.layout,
                    "body_yaw_deg": self.body_yaw_deg}
        if cmd == "state":
            return {"ok": True, "objects": self.state(env), "layout": self.layout}
        if cmd == "evaluate":
            r = evaluate(self.sc, req.get("task", "sort_fixed"), self.state(env), req.get("mapping"))
            return {"ok": True, **r}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    # ------------------------------------------------------------------ sim clock
    def clock_info(self):
        r = lambda x: None if x is None else round(x, 3)
        return {"sim_t": round(self.sim_t, 4), "step_dt": self.step_dt, "rtf": r(self.rtf), "rtf_mean": r(self.rtf_mean)}

    def _tick_clock(self, env):
        self.sim_t = int(env.common_step_counter) * self.step_dt
        now = time.monotonic()
        if self.clock is not None:
            try:
                self.clock.send(struct.pack("<dd", self.sim_t, time.time()), self.zmq.NOBLOCK)
            except self.zmq.Again:
                pass
        if self._win is None:
            self._win = self._start = (now, self.sim_t)
            return
        if now - self._win[0] >= 10.0:
            self.rtf = (self.sim_t - self._win[1]) / (now - self._win[0])
            self.rtf_mean = (self.sim_t - self._start[1]) / (now - self._start[0])
            self._win = (now, self.sim_t)
            if self.rtf < 0.95:
                print(f"[aalto] WARNING real-time factor {self.rtf:.2f} (< 0.95; mean {self.rtf_mean:.2f}): the sim runs "
                      f"slower than real time; AaltoEnv paces on the sim clock", flush=True)

    # ------------------------------------------------------------------ per step
    def step(self, env):
        self._tick_clock(env)
        if self.cam_follow:
            try:
                self.drive_camera(env)
            except Exception as e:      # report once; never kill the sim over the follower
                if self._cam_err is None:
                    self._cam_err = f"{type(e).__name__}: {e}"
                    print(f"[aalto] WARNING head camera follower failed: {self._cam_err}", flush=True)
        if self.rep is not None:
            for _ in range(4):
                try:
                    msg = self.rep.recv(self.zmq.NOBLOCK)
                except self.zmq.Again:
                    break
                try:
                    resp = self.handle(env, json.loads(msg))
                except Exception as e:  # never kill the sim over a bad request
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.rep.send(json.dumps(resp).encode())
        if self.rgbd is not None:
            self._publish_rgbd(env)
            if self.cam_follow and self.cam_image_check is None and self.sim_t > 3.0 and self._latest is not None:
                try:
                    self.cam_image_check = camera_image_check(self, env)
                except Exception as e:
                    self.cam_image_check = {"verdict": f"error {type(e).__name__}: {e}"}
                print(f"[aalto] head camera image check (depth of the counter): {self.cam_image_check}", flush=True)

    def _publish_rgbd(self, env):
        from tasks.common_observations.camera_state import _camera_frame
        cam = env.scene["front_camera"]
        out = cam.data.output
        frame = _camera_frame(cam)
        if frame is not None and frame == self.last_frame:
            return
        self.last_frame = frame
        rgb = _t(out["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
        dist = _t(out["distance_to_image_plane"])[0, ..., 0].cpu().numpy()
        dist = np.where(np.isfinite(dist), dist, 0.0)
        depth = np.clip(np.round(dist * 1000.0), 0, 65535).astype(np.uint16)
        depth[(dist < 0.1) | (dist > 10.0)] = 0     # D435 has no valid depth this close / far
        if self.dr_params and self._cam_rng is not None:
            rgb, depth = dr.corrupt(rgb, depth, self.dr_params.get("camera"), self._cam_rng)
        self._latest = (rgb, depth, time.time())
        self.rgbd.publish_one()


    # ------------------------------------------------------------------ domain randomization
    def randomize(self, env, seed):
        if not self.dr_cfg:
            return None
        params = dr.sample(self.dr_cfg, seed, self.names, self.sc.bin_names)
        if self._dr_base is None:
            self._dr_base = _capture_base(env, self)
        params["applied"] = _apply_dr(env, self, params)
        self.dr_params = params
        self._cam_rng = np.random.default_rng(params["camera"]["stream_seed"])
        return params


def _env0(path):
    return path.replace("env_.*", "env_0")


def _stage():
    import omni.usd
    return omni.usd.get_context().get_stage()


def _shaders(stage, root, name):
    """UsdPreviewSurface shaders under root with input `name`."""
    from pxr import Usd, UsdShade
    out = []
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        return out
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdShade.Shader):
            sh = UsdShade.Shader(p)
            if sh.GetInput(name):
                out.append(sh)
    return out


def _capture_base(env, rt):
    """Nominal values, captured once before the first randomization (DR is always relative to these)."""
    stage = _stage()
    b = {"lights": {}, "colors": {}, "mass": {}, "material": {}, "cam": None}
    for key, path in (("key", "/World/KeyLight"), ("dome", "/World/DomeLight")):
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            t = p.GetAttribute("xformOp:translate")
            b["lights"][key] = {"intensity": p.GetAttribute("inputs:intensity").Get(),
                                "color": tuple(p.GetAttribute("inputs:color").Get() or (1, 1, 1)),
                                "translate": tuple(t.Get()) if t and t.Get() is not None else None}
    roots = {"Counter": "/World/envs/env_0/Counter"}
    for bn in rt.sc.bin_names:
        for part in ("floor", "wall_xlo", "wall_xhi", "wall_ylo", "wall_yhi"):
            roots[f"bin:{bn}:{part}"] = f"/World/envs/env_0/Bin_{bn}_{part}"
    for n in rt.names:
        roots[f"obj:{n}"] = _env0(env.scene[n].cfg.prim_path)
    for k, r in roots.items():
        b["colors"][k] = [(sh, tuple(sh.GetInput("diffuseColor").Get())) for sh in _shaders(stage, r, "diffuseColor")]
    for n in rt.names:
        obj = env.scene[n]
        try:
            b["mass"][n] = _t(obj.data.body_mass).clone()
        except Exception as e:
            print(f"[aalto] DR: no mass access for {n}: {e}")
        try:
            import warp as wp
            b["material"][n] = wp.to_torch(obj.root_view.get_material_properties()).clone()
        except Exception as e:
            print(f"[aalto] DR: no material access for {n}: {e}")
    if "front_camera" in env.scene.keys():
        cam = env.scene["front_camera"]
        try:
            b["cam"] = (_t(cam.data.pos_w).clone(), _t(cam.data.quat_w_ros).clone())
        except Exception as e:
            print(f"[aalto] DR: no camera pose access: {e}")
    return b


def _apply_dr(env, rt, P):
    """Apply one parameter set. Returns which groups were applied (False = API missing, logged)."""
    from pxr import Gf, Sdf
    stage, base, ok = _stage(), rt._dr_base, {}
    try:
        L = P["lighting"]
        for key in ("key", "dome"):
            lb = base["lights"].get(key)
            if not lb:
                continue
            p = stage.GetPrimAtPath("/World/KeyLight" if key == "key" else "/World/DomeLight")
            p.GetAttribute("inputs:intensity").Set(float(lb["intensity"]) * L[f"{key}_intensity_scale"])
            p.GetAttribute("inputs:color").Set(Gf.Vec3f(*[float(c) * j for c, j in zip(lb["color"], L[f"{key}_rgb"])]))
            if key == "key" and lb["translate"] is not None:
                p.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*[float(a) + d for a, d in zip(lb["translate"], L["key_pos_offset"])]))
        ok["lighting"] = True
    except Exception as e:
        ok["lighting"] = f"{type(e).__name__}: {e}"
    try:
        T, O = P["textures"], P["objects"]
        for k, lst in base["colors"].items():
            for sh, rgb in lst:
                if k == "Counter":
                    new = [c * j for c, j in zip(rgb, T["counter_rgb"])]
                elif k.startswith("bin:"):
                    new = [c * j for c, j in zip(rgb, T["bin_rgb"][k.split(":")[1]])]
                else:
                    o = O[k[4:]]
                    new = dr.jitter_color(rgb, o["dh"], o["s"], o["v"])
                sh.GetInput("diffuseColor").Set(Gf.Vec3f(*[float(min(max(c, 0.0), 1.0)) for c in new]))
        tex = stage.GetPrimAtPath("/World/envs/env_0/Mat/material/texture")
        if tex.IsValid():
            from pxr import UsdShade
            sh = UsdShade.Shader(tex)
            s, bias = T["mat_scale"], T["mat_bias"]
            tint = variant.mat_tint(rt.variant)          # the variant's mat colour: DR scales around it
            if tint is not None:
                s = [a * b for a, b in zip(s, tint)]
            (sh.GetInput("scale") or sh.CreateInput("scale", Sdf.ValueTypeNames.Float4)).Set(Gf.Vec4f(*s, 1.0))
            (sh.GetInput("bias") or sh.CreateInput("bias", Sdf.ValueTypeNames.Float4)).Set(Gf.Vec4f(bias, bias, bias, 0.0))
        ok["colors"] = True
    except Exception as e:
        ok["colors"] = f"{type(e).__name__}: {e}"
    try:
        ids = torch.tensor([0], dtype=torch.int32)
        for n in rt.names:
            o, obj = P["objects"][n], env.scene[n]
            if n in base["mass"]:
                m = base["mass"][n] * o["mass_scale"]
                obj.set_masses_index(masses=m.to(m.device), env_ids=ids.to(m.device))
            if n in base["material"]:
                import warp as wp
                mat = base["material"][n].clone()
                mat[..., 0:2] *= o["friction_scale"]
                obj.root_view.set_material_properties(wp.from_torch(mat.contiguous(), dtype=wp.float32),
                                                      wp.from_torch(ids, dtype=wp.int32))
        ok["physics"] = True
    except Exception as e:
        ok["physics"] = f"{type(e).__name__}: {e}"
    try:
        if base["cam"] is not None:
            from scipy.spatial.transform import Rotation as R
            pos0, q0 = base["cam"]
            C = P["camera"]
            q = (R.from_rotvec(C["rot_offset_rad"]) * R.from_quat(q0[0].cpu().numpy())).as_quat()   # xyzw
            pos = pos0[0].cpu().numpy() + np.asarray(C["pos_offset"])
            if rt.cam_follow:   # the follower places the camera every step: hand it the jitter instead
                rt.cam_jitter = (R.from_rotvec(C["rot_offset_rad"]).as_matrix(), np.asarray(C["pos_offset"], float))
            else:
                env.scene["front_camera"].set_world_poses(torch.tensor([pos], dtype=torch.float32),
                                                          torch.tensor([q], dtype=torch.float32), convention="ros")
            ok["camera_pose"] = True
    except Exception as e:
        ok["camera_pose"] = f"{type(e).__name__}: {e}"
    return ok


ROBOT_FIELDS = {"pos": "joint_pos", "pos_target": "joint_pos_target", "vel": "joint_vel",
                "stiffness": "joint_stiffness", "damping": "joint_damping", "armature": "joint_armature",
                "effort_limit": "joint_effort_limits", "friction": "joint_friction_coeff",
                "pos_limits": "joint_pos_limits", "soft_limits": "soft_joint_pos_limits", "torque": "applied_torque"}


def robot_info(env, pattern=None):
    """Live articulation values by joint NAME (what the solver really uses: gains after every override, limits,
    PD target vs position = tracking error / sag). pattern: regex on joint names (default: every joint)."""
    robot = env.scene["robot"]
    names = list(robot.data.joint_names)
    sel = [i for i, n in enumerate(names) if not pattern or re.search(pattern, n)]
    cols, errors = {}, {}
    for key, attr in ROBOT_FIELDS.items():
        try:
            cols[key] = _t(getattr(robot.data, attr))[0].detach().cpu().numpy()
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {e}"

    def js(x):
        x = np.asarray(x, float)
        return [js(v) for v in x] if x.ndim else (round(float(x), 6) if np.isfinite(x) else None)
    joints = {names[i]: {k: js(c[i]) for k, c in cols.items()} for i in sel}
    acts = {}
    for g, a in getattr(robot, "actuators", {}).items():
        acts[g] = {"class": type(a).__name__, "joints": list(getattr(a, "joint_names", []))}
    return {"joints": joints, "joint_order": names, "actuators": acts, "errors": errors,
            "soft_joint_pos_limit_factor": robot.cfg.soft_joint_pos_limit_factor}


def R_quat_xyzw(R):
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return (q / np.linalg.norm(q)).tolist()


def counter_depth(T, K, u, v, z):
    """Distance to the image plane (m) of pixel (u, v) on the horizontal plane at height z, camera T (world, ROS)."""
    d = T[:3, :3] @ np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
    return None if d[2] >= -1e-6 else (z - T[2, 3]) / d[2]


def camera_image_check(rt, env, x=0.42, ys=np.linspace(-0.15, 0.15, 13)):
    """Does the RENDERED head image follow the camera pose the follower writes? Counter points between the object band
    and the boxes are projected with the written pose; at those pixels the rendered depth is compared with the
    counter's depth seen from (a) the written pose and (b) the spawn pose (parent link at its zero pose x mount).
    A soft waist sags ~1.7 deg at home: ~10-15 mm of depth apart, far above the render noise. A pure body yaw barely
    changes the depth of a horizontal plane: then 'undecided' (probe_variant camera --compare decides by pixels)."""
    import yaml
    with open(variant.REPO / "aaltonomous/interface/calibration.yaml") as f:
        K = yaml.safe_load(f)["cameras"]["head"]["intrinsics"]
    origin = _t(env.scene.env_origins)[0].cpu().numpy()
    p, R = rt._cam_last
    Tw = np.eye(4)
    Tw[:3, :3], Tw[:3, 3] = R, p - origin
    Ts = np.eye(4)
    Ts[:3, :3], Ts[:3, 3] = (variant.link_T_zero(rt.cam_parent) @ rt.cam_mount)[:3, :3], \
        (variant.link_T_zero(rt.cam_parent) @ rt.cam_mount)[:3, 3] + PELVIS_W
    z = rt.sc.counter_z + PELVIS_W[2]
    depth = rt._latest[1]
    res_w, res_s = [], []
    for y in ys:
        pt = np.array([x, y, rt.sc.counter_z]) + PELVIS_W
        c = Tw[:3, :3].T @ (pt - Tw[:3, 3])
        u, v = K["fx"] * c[0] / c[2] + K["cx"], K["fy"] * c[1] / c[2] + K["cy"]
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            continue
        d = depth[int(round(v)), int(round(u))] / 1000.0
        dw, ds = counter_depth(Tw, K, u, v, z), counter_depth(Ts, K, u, v, z)
        if d <= 0 or dw is None or ds is None:
            continue
        res_w.append(abs(d - dw))
        res_s.append(abs(d - ds))
    if len(res_w) < 3:
        return {"verdict": "undecided (counter not visible)"}
    mw, ms = float(np.median(res_w)), float(np.median(res_s))
    sep = float(np.median(np.abs(np.array(res_w) - np.array(res_s))))
    if abs(ms - mw) < 0.004:
        verdict = "undecided (written and spawn pose too close)"
    else:
        verdict = "follows" if mw < ms else "does NOT follow (the image stays at the spawn pose)"
    return {"verdict": verdict, "depth_resid_written_mm": round(mw * 1000, 1),
            "depth_resid_spawn_mm": round(ms * 1000, 1), "n": len(res_w), "sep_mm": round(sep * 1000, 1)}


camera_follows = variant.camera_follows

def camera_world_pose(link_pos, link_quat_xyzw, mount, jitter=None):
    """(position, rotation) of the head camera (ROS optical axes) in the world: link pose x mount (4x4 T_link_optical),
    then an optional world-frame jitter (R, dpos) on top."""
    Rl = quat_xyzw_R(link_quat_xyzw)
    R = Rl @ np.asarray(mount, float)[:3, :3]
    p = np.asarray(link_pos, float) + Rl @ np.asarray(mount, float)[:3, 3]
    if jitter is not None:
        R, p = jitter[0] @ R, p + jitter[1]
    return p, R


def head_camera_pelvis_T(V):
    """The head camera's mount pose in the pelvis frame as scene_cfg builds it (calibration.yaml + the variant)."""
    import yaml
    with open(variant.REPO / "aaltonomous/interface/calibration.yaml") as f:
        T = np.asarray(yaml.safe_load(f)["cameras"]["head"]["T_pelvis_optical"], float)
    return variant.camera_T(V, T)


BODY_LINKS = ("pelvis", "torso_link", "d435_link", "left_wrist_yaw_link", "left_hand_palm_link", "right_wrist_yaw_link",
              "right_hand_palm_link")


def quat_xyzw_R(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def body_info(env):
    """Ground truth the POLICY never gets (tests / probes only): waist joint angles, link poses in the PELVIS frame
    (pos m, R 3x3; pelvis = the articulation's own pelvis body), the head camera's pose in the pelvis frame, and the
    pelvis in the table frame (= world - env origin - PELVIS_W, the frame objects / scoring use; differs from the
    pelvis only under the variant's body_yaw). Quaternions are IsaacLab's (x, y, z, w)."""
    robot, out, err = env.scene["robot"], {}, {}
    origin = _t(env.scene.env_origins)[0].cpu().numpy()
    names = list(robot.data.body_names)
    pos = _t(robot.data.body_link_pos_w)[0].cpu().numpy() - origin - PELVIS_W
    quat = _t(robot.data.body_link_quat_w)[0].cpu().numpy()
    ip = names.index("pelvis")
    Rp, tp = quat_xyzw_R(quat[ip]), pos[ip]
    r = lambda a: [[round(float(v), 6) for v in row] for row in a] if np.ndim(a) == 2 else [round(float(v), 6) for v in a]
    links = {}
    for n in BODY_LINKS:
        if n in names:
            i = names.index(n)
            links[n] = {"pos": r(Rp.T @ (pos[i] - tp)), "R": r(Rp.T @ quat_xyzw_R(quat[i]))}
    out["links"] = links
    out["pelvis_in_table"] = {"pos": r(tp), "yaw_deg": round(float(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0]))), 4)}
    jn = list(robot.data.joint_names)
    jp = _t(robot.data.joint_pos)[0].cpu().numpy()
    out["waist"] = {n: round(float(jp[jn.index(n)]), 6) for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
                    if n in jn}
    if "front_camera" in env.scene.keys():
        try:
            cam = env.scene["front_camera"]
            cp = _t(cam.data.pos_w)[0].cpu().numpy() - origin - PELVIS_W
            out["camera"] = {"pos": r(Rp.T @ (cp - tp)),
                             "R": r(Rp.T @ quat_xyzw_R(_t(cam.data.quat_w_ros)[0].cpu().numpy())),
                             "parent": variant.camera_parent(variant.active()),
                             "follower": variant.camera_follows(variant.active()), "prim": cam.cfg.prim_path}
        except Exception as e:
            err["camera"] = f"{type(e).__name__}: {e}"
    out["usd_path"] = str(getattr(robot.cfg.spawn, "usd_path", ""))     # probes: zero-pose link transforms
    out["body_errors"] = err
    return out


def _world_points(prim):
    """World-space points spanning the geometry under prim: cube corners (size x the full local-to-world transform)
    and mesh vertices; other gprims: the corners of their world bound box (with its own matrix, not axis-aligned)."""
    from pxr import Usd, UsdGeom
    tc, out = Usd.TimeCode.Default(), []
    cache = UsdGeom.BBoxCache(tc, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for p in Usd.PrimRange(prim):
        if not p.IsA(UsdGeom.Gprim):
            continue
        M = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(tc), float)    # row-vector convention
        if p.IsA(UsdGeom.Cube):
            h = float(UsdGeom.Cube(p).GetSizeAttr().Get()) / 2
            local = h * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        elif p.IsA(UsdGeom.Mesh):
            local = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
        else:
            b = cache.ComputeWorldBound(p)
            lo, hi = np.array(b.GetRange().GetMin(), float), np.array(b.GetRange().GetMax(), float)
            local = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            M = np.array(b.GetMatrix(), float)
        out.append(np.c_[local, np.ones(len(local))] @ M)
    return np.vstack(out)[:, :3] if out else np.zeros((0, 3))


def _prim_box(stage, path, origin):
    """Static prim as authored on the stage: pelvis-frame position, yaw (deg, about z), and the size of its geometry
    along the prim's OWN axes (geometry expressed in the prim frame, so a yawed box reads its true size; an
    axis-aligned bound of a yawed box would read ~W (cos + sin)). None if the prim does not exist."""
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)
    t = m[3, :3] - origin - PELVIS_W
    yaw = float(np.degrees(np.arctan2(m[0, 1], m[0, 0])))           # Gf rows = transformed basis vectors
    P = _world_points(prim)
    size = np.zeros(3)
    if len(P):
        Rn = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=1, keepdims=True)   # rows = the prim's axes (scale kept)
        local = (P - m[3, :3]) @ Rn.T                                       # back into the prim's own axes
        size = local.max(axis=0) - local.min(axis=0)
    return {"pos": [round(float(v), 5) for v in t], "yaw_deg": round(yaw, 3), "size": [round(float(v), 5) for v in size]}


def geometry_info(env, rt):
    """What the stage holds (pelvis frame), for probe_variant.py scene: independent of the Scene the runtime scores
    with, so a mismatch between the built geometry and the scoring geometry shows up."""
    stage = _stage()
    origin = _t(env.scene.env_origins)[0].cpu().numpy()
    root = "/World/envs/env_0"
    out = {"bins": {}, "counter": _prim_box(stage, f"{root}/Counter", origin),
           "lid": _prim_box(stage, f"{root}/Lid", origin), "mat": stage.GetPrimAtPath(f"{root}/Mat").IsValid(),
           "objects": {}, "colors": {}, "errors": {}}
    for bn in rt.sc.bin_names:
        out["bins"][bn] = {p: _prim_box(stage, f"{root}/Bin_{bn}_{p}", origin)
                           for p in ("floor", "wall_xlo", "wall_xhi", "wall_ylo", "wall_yhi")}
    for k, path in (("counter", f"{root}/Counter"), ("bin_wall", f"{root}/Bin_center_wall_yhi"),
                    ("bin_floor", f"{root}/Bin_center_floor")):
        sh = _shaders(stage, path, "diffuseColor")
        if sh:
            r = sh[0].GetInput("roughness")
            out["colors"][k] = {"diffuse": [round(float(c), 4) for c in sh[0].GetInput("diffuseColor").Get()],
                                "roughness": None if not r or r.Get() is None else round(float(r.Get()), 3)}
    for n in rt.names:
        color, shape, _ = parse_name(n)
        sh = _shaders(stage, _env0(env.scene[n].cfg.prim_path), "diffuseColor")
        if sh and color not in out["colors"]:
            out["colors"][color] = {"diffuse": [round(float(c), 4) for c in sh[0].GetInput("diffuseColor").Get()]}
        if shape in out["objects"]:
            continue
        b = _prim_box(stage, _env0(env.scene[n].cfg.prim_path), origin) or {}
        try:
            b["mass"] = round(float(_t(env.scene[n].data.body_mass).flatten()[0]), 5)
        except Exception as e:
            out["errors"][f"mass {n}"] = f"{type(e).__name__}: {e}"
        out["objects"][shape] = b
    out["object_height"] = rt.sc.object_height()
    if "front_camera" in env.scene.keys():
        try:
            cam = env.scene["front_camera"]
            out["camera"] = {"pos": [round(float(v), 5) for v in _t(cam.data.pos_w)[0].cpu().numpy() - origin - PELVIS_W],
                             "quat_xyzw_ros": [round(float(v), 6) for v in _t(cam.data.quat_w_ros)[0].cpu().numpy()]}
        except Exception as e:
            out["errors"]["camera"] = f"{type(e).__name__}: {e}"
    out["lights"] = {}
    for key, path in (("key", "/World/KeyLight"), ("dome", "/World/DomeLight")):
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            t = p.GetAttribute("xformOp:translate")
            out["lights"][key] = {"intensity": p.GetAttribute("inputs:intensity").Get(),
                                  "pos": None if not t or t.Get() is None else [round(float(v), 3) for v in t.Get()]}
    return out


def runtime(env):
    global _RT
    if _RT is None:
        _RT = AaltoRuntime(env)
    return _RT


# ---------------------------------------------------------------------- Isaac Lab hooks
_placeholder = None


def aalto_step_obs(env) -> torch.Tensor:
    """ObsTerm: runs the runtime once per step; returns a dummy (num_envs, 1) tensor."""
    global _placeholder
    runtime(env).step(env)
    if _placeholder is None:
        _placeholder = torch.zeros((env.num_envs, 1), device=env.device)
    return _placeholder


def reset_random_layout(env, env_ids, seed=None):
    """Event (startup / DDS reset category 1): a fresh random layout from scene.yaml."""
    rt = runtime(env)
    rt.place(env, random_layout(rt.sc, seed))
