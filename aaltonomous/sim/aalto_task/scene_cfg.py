"""Isaac Lab scene for the Aaltonomous challenge, generated from aaltonomous/scene/scene.yaml + interface/calibration.yaml.

World = pelvis frame + PELVIS_W (robot at the origin facing +x). Contents:
  robot            G1 29-DoF + Dex3, fixed base (upstream preset), pelvis at PELVIS_W
  counter          static slab, top at scene.counter_z, footprint counter_size centred on the mat
  mat              visual quad with scene/assets/mat_texture.png (the printed mat), 0.5 mm above the counter
  bin_<name>_*     static floor + 4 walls per bin (fixed + known), colour from scene.yaml bins.rgb (all identical)
  obj_<c>_<s>_<k>  18 rigid objects from the printed STL meshes (scene/assets/object_<shape>.stl), convex hulls
  lid              rung 3 only (Aalto-Insert-*): static triangle-mesh plate with the 3 holes on the lid bin
  front_camera     head RealSense colour, ONE pinhole cam at calibration.yaml head.T_pelvis_optical, rgb + depth
  left/right_wrist_camera  upstream presets (no measured mount yet, M7)
AALTO_VARIANT (variant.py; unset = the organisers' sim): robot gains / limits / joint limits, render colours, lights,
mat, table footprint + gloss, box row (moved / yawed, boxes.py), object mesh scale + physics, head camera offset and
a scene.yaml overlay, applied below where each piece is built.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from tasks.common_config import CameraPresets, G1RobotPresets  # upstream (unitree_sim_isaaclab on sys.path)

from aaltonomous.scene.scene import Scene
from . import variant
from .boxes import bin_parts, counter_box, yaw_quat_xyzw
from .layout import ARM_HOME, PELVIS_W, assign, object_names, parse_name

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "aaltonomous/scene/assets"
CALIB = REPO / "aaltonomous/interface/calibration.yaml"
V = variant.active()
SC = variant.scene(V)          # = Scene.load() without a variant
ENV = "/World/envs/env_.*"

# Physics of the printed PLA parts (DR knobs later, spec §6). Friction is on the high side: Dex3 fingertips
# are rubber; the sim tends to need more grip than reality for the same grasp.
OBJ_MASS = variant.object_physics(V, "mass", 0.05)       # kg, 45 mm PLA part at ~20 % infill (variant: foam)
OBJ_STATIC_FRICTION = variant.object_physics(V, "static_friction", 1.2)
OBJ_DYNAMIC_FRICTION = variant.object_physics(V, "dynamic_friction", 1.0)
OBJ_MESH_SCALE = variant.object_mesh_scale(SC)           # (1, 1, 1) unless the variant changes objects.size
COUNTER_THICKNESS = 0.04


def _w(p_pelvis):
    return tuple(float(v) for v in np.asarray(p_pelvis, float) + PELVIS_W)


def _rgb(name):
    return variant.object_rgb(V, SC, name)   # render only: names / task logic keep the internal colour


def rotmat_to_quat_xyzw(R):
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return tuple(float(v) for v in q / np.linalg.norm(q))


# ------------------------------------------------------------------ custom spawners
@configclass
class StlMeshCfg(sim_utils.MeshCfg):
    """Rigid/static body from an STL (mm), recentred so the origin is the object centroid (bbox centre in z).
    approximation: PhysX MeshCollisionAPI approximation ("convexHull" for dynamic, "none" = exact for static)."""
    func: object = None
    path: str = ""
    center_z: bool = True
    approximation: str = "convexHull"
    swap_xy: bool = False      # mesh authored in MAT axes -> pelvis axes (x<->y mirror, faces re-wound)
    scale: tuple = (1.0, 1.0, 1.0)   # per-axis, after mm -> m (variant objects.size / height_ratio)

    def __post_init__(self):
        self.func = spawn_stl_mesh


@clone
def spawn_stl_mesh(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import trimesh
    from pxr import UsdPhysics
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh

    mesh = trimesh.load(cfg.path, force="mesh")
    mesh.apply_scale(0.001)
    if tuple(cfg.scale) != (1.0, 1.0, 1.0):
        mesh.apply_transform(np.diag([*(float(s) for s in cfg.scale), 1.0]))
    if cfg.swap_xy:
        mesh = trimesh.Trimesh(mesh.vertices[:, [1, 0, 2]], mesh.faces[:, ::-1], process=False)
    if cfg.center_z:
        lo, hi = mesh.bounds
        mesh.apply_translation([0.0, 0.0, -(lo[2] + hi[2]) / 2])
    prim = _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, **kwargs)
    if cfg.collision_props is not None and cfg.approximation != "convexHull":
        mp = prim.GetStage().GetPrimAtPath(prim_path + "/geometry/mesh")
        UsdPhysics.MeshCollisionAPI.Apply(mp).GetApproximationAttr().Set(cfg.approximation)
    return prim


@configclass
class TexturedQuadCfg(sim_utils.SpawnerCfg):
    """Visual-only quad (size x, y) in the xy plane, UV-mapped 0..1 with a texture image."""
    func: object = None
    size: tuple = (1.0, 1.0)
    texture: str = ""
    tint: tuple = None         # texture multiplier (UsdUVTexture scale); None = not authored (DR may author it)
    uvs: tuple = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))   # for vertices (-x,-y) (x,-y) (x,y) (-x,y)

    def __post_init__(self):
        self.func = spawn_textured_quad


@clone
def spawn_textured_quad(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    from isaaclab.sim.utils import create_prim, get_current_stage

    stage = get_current_stage()
    prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    hx, hy = cfg.size[0] / 2, cfg.size[1] / 2
    mesh = UsdGeom.Mesh.Define(stage, prim_path + "/quad")
    mesh.CreatePointsAttr([Gf.Vec3f(-hx, -hy, 0), Gf.Vec3f(hx, -hy, 0), Gf.Vec3f(hx, hy, 0), Gf.Vec3f(-hx, hy, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in cfg.uvs]))

    mat = UsdShade.Material.Define(stage, prim_path + "/material")
    sh = UsdShade.Shader.Define(stage, prim_path + "/material/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, prim_path + "/material/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, prim_path + "/material/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(cfg.texture)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    if cfg.tint is not None:
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*[float(c) for c in cfg.tint], 1.0))
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return prim


# ------------------------------------------------------------------ scene pieces
def _static_box(name, center_pelvis, size, rgb, roughness=0.8, yaw=0.0):
    rot = {} if yaw == 0.0 else {"rot": yaw_quat_xyzw(yaw)}     # yaw about pelvis z (variant boxes); xyzw
    return AssetBaseCfg(
        prim_path=f"{ENV}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_w(center_pelvis), **rot),
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(s) for s in size),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005,
                                                             rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=roughness, metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6,
                                                            restitution=0.0),
        ),
    )


def counter_cfg():
    # counter_size centred on the mat (mat x runs along pelvis y), or the variant's table footprint (boxes.py)
    centre, size = counter_box(SC, COUNTER_THICKNESS)
    return _static_box("Counter", centre, size, variant.color_override(V, "counter", (0.55, 0.45, 0.35)),
                       roughness=variant.table_roughness(V, 0.8))


def mat_cfg():
    """Printed mat texture on the counter. export.mat_image: row 0 = far edge (pelvis +x), column 0 = robot's LEFT
    (pelvis +y). USD UVs: v = 0 at the image bottom, u = 0 at the left column."""
    c = SC.cfg["counter"]
    mat = np.asarray(c["mat_size"], float)
    centre = SC.mat_to_pelvis([mat[0] / 2, mat[1] / 2, 0.0005])
    return AssetBaseCfg(
        prim_path=f"{ENV}/Mat",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_w(centre)),
        spawn=TexturedQuadCfg(size=(float(mat[1]), float(mat[0])),
                              texture=str(ASSETS / (V or {}).get("mat_texture", c["mat_texture"])),
                              tint=variant.mat_tint(V),
                              uvs=((1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))),
    )


def bin_cfgs():
    out = {}
    rgb = variant.color_override(V, "bins", tuple(c / 255.0 for c in SC.bin_rgb))
    floor_rgb = variant.color_override(V, "bin_floor", rgb)
    for name in SC.cfg["bins"]["centers"]:
        # floor + 4 walls, pelvis centre / size / yaw: scene.yaml's axis-aligned boxes, or the variant's row
        for part, c, s, yaw in bin_parts(SC, name):
            out[f"bin_{name}_{part}"] = _static_box(f"Bin_{name}_{part}", c, s, floor_rgb if part == "floor" else rgb,
                                                    yaw=yaw)
    return out


def object_cfgs():
    poses, _ = assign(SC, [])  # everything parked; the reset event places the layout
    out = {}
    for n in object_names(SC):
        color, shape, _ = parse_name(n)
        p, q = poses[n]
        out[n] = RigidObjectCfg(
            prim_path=f"{ENV}/{n}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_w(p), rot=tuple(float(v) for v in q)),
            spawn=StlMeshCfg(
                path=str(ASSETS / f"object_{shape}.stl"), scale=OBJ_MESH_SCALE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=OBJ_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.004,
                                                                 rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_rgb(color), roughness=0.7, metallic=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max", restitution_combine_mode="min",
                    static_friction=OBJ_STATIC_FRICTION, dynamic_friction=OBJ_DYNAMIC_FRICTION, restitution=0.0),
            ),
        )
    return out


def lid_cfg():
    """Rung-3 lid plate on the lid bin rim. The STL is in mat axes around the lid centre -> swap_xy. A yawed variant
    box turns it with the box (its holes stay sized for scene.yaml's objects)."""
    cx, cy = SC.cfg["bins"]["centers"][SC.lid_bin]
    yaw = SC.poses[SC.lid_bin][2] if variant.is_variant_scene(SC) else 0.0
    rot = {} if yaw == 0.0 else {"rot": yaw_quat_xyzw(yaw)}
    return AssetBaseCfg(
        prim_path=f"{ENV}/Lid",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_w(SC.mat_to_pelvis([cx, cy, SC.lid_z[0]])), **rot),
        spawn=StlMeshCfg(path=str(ASSETS / "lid_shapes.stl"), center_z=False, swap_xy=True, approximation="none",
                         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002,
                                                                          rest_offset=0.0),
                         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.85), roughness=0.8)),
    )


def head_camera_cfg():
    cal = yaml.safe_load(open(CALIB))["cameras"]["head"]
    K = cal["intrinsics"]
    # + the variant's head_camera offset; parent pelvis (organisers' mount, default) or torso_link / d435_link with
    # the same pose at waist 0 (variant.LINK_POSE_ZERO, from the robot USD; NOT read here: this runs before Kit),
    # so the camera moves with a bending waist as on the robot
    parent, T = variant.camera_mount(V, np.asarray(cal["T_pelvis_optical"], float))
    path = f"{ENV}/Robot/{parent}/front_cam"
    if variant.camera_follows(V):
        # R3 live: a camera prim under a robot link neither moves with the link nor renders where its pose is written
        # (link transforms only reach Fabric, not the camera prim's USD). So the following camera is a WORLD-level prim
        # (no parent transform to fight), spawned at its zero-pose world pose and driven every step by the runtime
        # (hooks.drive_camera: Fabric + USD), mount = parent link x T below.
        Tw = variant.link_T_zero(parent) @ T
        yaw = np.radians(variant.body_yaw_base_deg(V) or 0.0)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rz @ Tw[:3, :3], Rz @ Tw[:3, 3] + PELVIS_W
        path = f"{ENV}/HeadCam_{parent}"
        print(f"[aalto] head camera follows {parent} (world-level prim {path}, driven every step)")
    return CameraCfg(
        prim_path=path,
        update_period=1.0 / 30,
        height=int(K["height"]), width=int(K["width"]),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[K["fx"], 0, K["cx"], 0, K["fy"], K["cy"], 0, 0, 1],
            width=int(K["width"]), height=int(K["height"]), clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in T[:3, 3]), rot=rotmat_to_quat_xyzw(T[:3, :3]),
                                   convention="ros"),
    )


def _robot_cfg():
    yaw = np.radians(variant.body_yaw_base_deg(V) or 0.0)           # variant body_yaw (constant part); 0 = upstream
    rot = (0.0, 0.0, 0.0, 1.0) if yaw == 0.0 else (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
    cfg = G1RobotPresets.g1_29dof_dex3_base_fix(init_pos=tuple(float(v) for v in PELVIS_W), init_rot=rot)
    cfg.init_state.joint_pos = {**cfg.init_state.joint_pos, **ARM_HOME}
    # OUR sim only (unset = the upstream preset the organisers' sim runs: arms stiffness 300 / 400, damping 3 / 2.5;
    # hands 100 / 10, armature 0.1, effort 300; soft_joint_pos_limit_factor 0.9). The sim ignores the DDS kp / kd
    # (only q is read), so these actuator gains ARE the simulated arm / finger stiffness.
    #   AALTO_VARIANT: robot.soft_joint_pos_limit_factor, robot.actuators.<arms|hands>, robot.joint_limits (variant.py)
    #   AALTO_HAND_GAINS="kp,kd[,armature[,effort_limit]]" (hands; applied last = wins over the variant), e.g.
    #   "1.5,0.2,0.002,1.4" (real runner: kp 1.5 / kd 0.2, organisers 2026-11-14). Both reported by `info`.
    cfg.soft_joint_pos_limit_factor = variant.soft_limit_factor(V, cfg.soft_joint_pos_limit_factor)
    for group, params in variant.actuator_overrides(V, os.environ.get("AALTO_HAND_GAINS")):
        variant.apply_actuator(cfg.actuators[group], group, params)
        print(f"[aalto] {group} actuators overridden: {params}")
    lim = variant.install_joint_limits(cfg, V)     # robot.joint_limits: authored on the USD joints at spawn
    if lim:
        print(f"[aalto] variant joint limits requested (rad, None = USD): {lim}")
    if V:
        print(f"[aalto] variant {V['name']} ({V['_path']}): soft_joint_pos_limit_factor "
              f"{cfg.soft_joint_pos_limit_factor}, right_hand_wiring {variant.wiring(V)}")
    return cfg


def _fixed_cfgs():
    robot = _robot_cfg()
    return {
        "robot": robot,
        "floor": AssetBaseCfg(
            prim_path=f"{ENV}/Floor",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            spawn=sim_utils.CuboidCfg(size=(8.0, 8.0, 0.1),
                                      collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                                      visual_material=sim_utils.PreviewSurfaceCfg(
                                          diffuse_color=variant.color_override(V, "floor", (0.35, 0.35, 0.38))))),
        "dome_light": AssetBaseCfg(prim_path="/World/DomeLight",
                                   spawn=sim_utils.DomeLightCfg(color=variant.light(V, "dome", "rgb", (0.9, 0.9, 0.9)),
                                                                intensity=variant.light(V, "dome", "intensity", 800.0))),
        # "our own lights at the counter": one overhead area-ish light slightly in front of the robot
        "key_light": AssetBaseCfg(prim_path="/World/KeyLight",
                                  init_state=AssetBaseCfg.InitialStateCfg(pos=variant.light(V, "key", "pos",
                                                                                             (0.5, 0.0, 2.2))),
                                  spawn=sim_utils.SphereLightCfg(color=variant.light(V, "key", "rgb", (1.0, 0.97, 0.92)),
                                                                 intensity=variant.light(V, "key", "intensity",
                                                                                         60000.0),
                                                                 radius=0.15)),
        "counter": counter_cfg(),
        **({"mat": mat_cfg()} if (V or {}).get("mat", True) else {}),     # variant mat: false = bare counter
        "front_camera": head_camera_cfg(),
        "left_wrist_camera": CameraPresets.left_dex3_wrist_camera(),
        "right_wrist_camera": CameraPresets.right_dex3_wrist_camera(),
    }


def make_scene_cfg_class(name="AaltoSceneCfg", lid=False):
    """InteractiveSceneCfg subclass with every entity as a dataclass field (InteractiveScene reads fields only)."""
    attrs = {**_fixed_cfgs(), **bin_cfgs(), **object_cfgs()}
    if lid:
        attrs["lid"] = lid_cfg()
    ns = dict(attrs)
    ns["__annotations__"] = {k: type(v) for k, v in attrs.items()}
    ns["__module__"] = __name__
    return configclass(type(name, (InteractiveSceneCfg,), ns))


AaltoSceneCfg = make_scene_cfg_class()
AaltoLidSceneCfg = make_scene_cfg_class("AaltoLidSceneCfg", lid=True)
