"""Simulator variants: OUR edits of the organisers' sim (real2sim), selected by one environment variable.

    AALTO_VARIANT=real_like  aalto sim        # aaltonomous/sim/variants/real_like.yaml
    AALTO_VARIANT=/path/to/my.yaml ...        # any file, same schema
Unset / '' / 'off' = the organisers' sim exactly (every function below returns its input or the upstream default).
The scene service `info` reports the active variant (`variant`, null when off); evaluate.py reads `color_words`
from it, so the one switch on the sim also renames the colour words the policy is told.

Schema (every key optional; colours 0-255 like scene.yaml; unknown keys are an error, so typos can't pass silently):
  name: str                                      (default: file stem; never inherited, nor are notes)
  inherit: <variant name | path>                 start from that variant (recursively), then deep-merge this file
                                                 on top: dicts merge key by key, anything else (lists too) replaces
  robot:
    soft_joint_pos_limit_factor: float           upstream 0.9: the action clamp (layeredcontrol RobotController) keeps
                                                 every PD target in the middle 90 % of the USD range; 1.0 = full range
    actuators:                                   per upstream group ("arms", "hands", "waist"); each key replaces
                                                 upstream's (waist upstream: 10000 / 10000, effort 1000 = rigid; the
                                                 legs hang from the FIXED pelvis and move nothing above it):
      <group>: {stiffness, damping, armature, effort_limit, friction}
                                                 a number (all joints of the group) or {joint regex: number} that
                                                 matches every joint of the group exactly once (checked here)
    right_hand_wiring: upstream | spec           spec = rt/dex3/right cmd + state slots as interface/joint_spec.yaml
                                                 (thumb 0-2, index 3-4, middle 5-6; upstream: middle 3-4, index 5-6)
    joint_limits: {<joint regex>: [lo, hi]}      rad, null = keep the USD value; authored on the robot USD's joints
                                                 at spawn = a HARD stop in PhysX (soft limits follow via the factor);
                                                 every regex must match a joint_spec joint, no joint twice
  colors:                                        RENDER only (names, task logic, scoring keep scene.yaml's colours)
    objects: {<internal colour>: [r, g, b]}      e.g. red: [240, 120, 20] renders the "red" objects orange
    bins: [r, g, b]    bin_floor: [r, g, b]      bin_floor: the inside floor (cardboard) instead of the bin colour
    counter: [r, g, b]    floor: [r, g, b]
    mat_tint: [sr, sg, sb]                       multiplies the mat texture (DR's mat_scale multiplies on top)
  color_words: {<internal colour>: word}         evaluator: words in the instruction handed to the policy
  lighting: {dome: {intensity, rgb}, key: {intensity, rgb, pos}}   rgb 0..1 (light colour), pos m (world)
  head_camera:                                   the SIMULATED camera only; the policy keeps calibration.yaml, so
                                                 this is the calibration error the policy has to live with
    pos_offset: [dx, dy, dz]                     m, pelvis frame, added to calibration.yaml's T_pelvis_optical
    rot_offset_deg: [rx, ry, rz]                 rotation vector (deg) in the PELVIS frame, applied on the left
    rot_offset_optical_deg: [rx, ry, rz]         rotation vector (deg) in the OPTICAL frame (x right, y down, z
                                                 forward), applied on the right: [-2.8, 0, 0] = looks 2.8 deg further
                                                 down (the convention of the drop-2 camera fit)
    parent: pelvis | torso_link | d435_link      link the camera prim hangs on (default pelvis, as the organisers'
                                                 task); same pose at waist 0 (offset from the robot USD), but on the
                                                 torso it moves when a soft waist bends, as on the robot
  layout:                                        objects as they really lie (messy.py), applied to every layout the
                                                 sim places and the evaluator logs (seeded by the layout itself):
    lying_prism_prob: 0..1                       each triangular prism lies on a rectangular side (ridge on top)
    lying_cylinder_prob: 0..1                    the same for cylinders (they lie on their curved side; may roll)
    cluster_prob: 0..1                           per episode: 2-4 objects form a touching cluster in the object band
    cluster_spacing: [lo, hi]                    centre distances in a cluster, x objects.size (default [1.05, 1.25])
    cluster_n: [lo, hi]                          cluster size (default [2, 4])
  body_yaw: {deg, jitter_deg}                    turn the whole fixed-base robot about the vertical through the pelvis
                                                 relative to the table: deg constant, + a per-episode N(0, jitter_deg)
                                                 (clipped at 3 sigma, seeded by the layout, so seeds repeat)
  mat: bool                                      false = no printed mat on the counter
  mat_texture: path                              another mat image (relative: aaltonomous/scene/assets/)
  table: {x: [x0, x1], y: [y0, y1], roughness}   counter top footprint, pelvis frame (default: scene.yaml counter_size
                                                 centred on the mat; scoring's "off" follows it); roughness of its
                                                 PreviewSurface (default 0.8; ~0.3 = glossy laminate, key-light glare)
  boxes:                                         box row, PELVIS frame (boxes.py; outer_size / wall_height /
                                                 wall_thickness stay in scene.bins via the overlay below):
    front_x: m                                   x of the centre box's near outer wall at its middle (default: cfg)
    y: m                                         y of the centre box's centre (default: cfg, 0)
    gap: m                                       between neighbouring outer walls (default: cfg spacing - width)
    yaw_deg: deg                                 row (and box) yaw about pelvis +z, CCW from above: + = the LEFT end
                                                 comes nearer (the drop-2 report's yaw has the OPPOSITE sign)
    centers: {left: [x, y], center: .., right: ..}   explicit box centres instead of the row (yaw_deg still applies)
    per_box: {<bin>: {dx, dy, yaw_deg}}          extra offset / yaw of one box on top
                                                 Geometry, placement, scoring (yawed inner walls), the evaluator's
                                                 oracle and aalto check all follow; validate() refuses boxes that
                                                 overlap each other, the object band (+ circumradius) or leave the table
  object_physics: {mass, static_friction, dynamic_friction}   every object (kg; default 0.05 / 1.2 / 1.0, PLA)
  scene: {...}                                   deep-merged onto scene.yaml for the SIM side (geometry it builds,
                                                 placement, scoring); keys must exist in scene.yaml. objects.size /
                                                 height_ratio also scale the object meshes (exported at scene.yaml's)

Precedence: AALTO_HAND_GAINS (scene_cfg.py) is applied AFTER the variant, i.e. it wins over robot.actuators.hands
(a quick sweep knob on top of a variant). AALTO_DR jitters around the variant's values (its base colours / lights
are read from the stage, which the variant already set).
Pure python (yaml + numpy) at import; the Isaac-side patches at the bottom import upstream modules lazily.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
VARIANT_DIR = REPO / "aaltonomous/sim/variants"
SPEC = REPO / "aaltonomous/interface/joint_spec.yaml"
COLOR_KEYS = ("objects", "bins", "bin_floor", "counter", "floor", "mat_tint")
TOP_KEYS = ("name", "notes", "inherit", "robot", "colors", "color_words", "lighting", "head_camera", "mat",
            "mat_texture", "table", "boxes", "object_physics", "body_yaw", "layout", "scene")
ROBOT_KEYS = ("soft_joint_pos_limit_factor", "actuators", "right_hand_wiring", "joint_limits")
CAMERA_KEYS = ("pos_offset", "rot_offset_deg", "rot_offset_optical_deg", "parent")
CAMERA_PARENTS = ("pelvis", "torso_link", "d435_link")   # d435_link: fixed to torso_link (upstream's G1 head camera)
BODY_YAW_KEYS = ("deg", "jitter_deg")
OBJECT_PHYSICS_KEYS = ("mass", "static_friction", "dynamic_friction")
ACT_KEYS = ("stiffness", "damping", "armature", "effort_limit", "friction")
WIRINGS = ("upstream", "spec")

# Upstream (unitree_sim_isaaclab @ the sim image) tables this module patches; test_variant.py checks them against
# joint_spec.yaml. Command: action_provider_dds.py {left,right}_hand_joint_mapping (joint -> slot in the 7-motor
# HandCmd). State: dex3_state.py publishes get_robot_girl_joint_names() order, gathered with hard-coded articulation
# indices (UPSTREAM_STATE_IDX; the runtime check in dex3_state_func compares them with the names).
_F = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
UPSTREAM_SLOTS = {s: {f"{s}_hand_{f}_joint": i for i, f in enumerate(_F)} for s in ("left", "right")}
UPSTREAM_STATE_IDX = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
UPSTREAM_GROUP_EXPR = {"arms": [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
                       "hands": [".*_hand_index_.*_joint", ".*_hand_middle_.*_joint", ".*_hand_thumb_.*_joint"],
                       "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]}   # upstream 10000 / 10000


# ---------------------------------------------------------------------------------------------- loading
def load(ref=None):
    """ref: None -> $AALTO_VARIANT; '' / 'off' -> None; a name in aaltonomous/sim/variants/ or a yaml path."""
    ref = os.environ.get("AALTO_VARIANT", "") if ref is None else ref
    if not ref or ref == "off":
        return None
    V, p, chain = _read(ref)
    V.setdefault("name", p.stem)
    V["_path"] = str(p)
    if chain:
        V["inherit"] = chain                      # resolved: parent, grandparent, ... (info / summary)
    return validate(V)


def _resolve(ref, rel=None):
    p = Path(ref)
    if not p.suffix:
        return VARIANT_DIR / f"{ref}.yaml"
    return p if p.is_absolute() or rel is None else rel / p


def _read(ref, rel=None, seen=()):
    """-> (merged dict, path, [parent names]) with `inherit` resolved recursively (cycles refused)."""
    p = _resolve(ref, rel)
    key = str(p.resolve())
    if key in seen:
        raise ValueError(f"variant: inherit cycle through {p}")
    V = yaml.safe_load(p.read_text()) or {}
    parent = V.pop("inherit", None)
    if not parent:
        return V, p, []
    P, pp, chain = _read(parent, p.parent, seen + (key,))
    P = {k: v for k, v in P.items() if k not in ("name", "notes")}
    return deep_merge(P, V), p, [pp.stem] + chain


def deep_merge(base, over):
    """Dicts merge key by key (recursively); anything else in `over` replaces."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


_ACTIVE = {}


def active():
    """The variant of this process ($AALTO_VARIANT), loaded once (scene_cfg, env_cfg, hooks share it)."""
    if "v" not in _ACTIVE:
        _ACTIVE["v"] = load()
    return _ACTIVE["v"]


def _unknown(d, allowed, where):
    bad = sorted(set(d) - set(allowed))
    if bad:
        raise ValueError(f"variant: unknown key(s) {bad} in {where} (allowed: {list(allowed)})")


def spec_names(group):
    """Joint names of an upstream actuator group (joint_spec.yaml: arm / hand / the waist part of locked)."""
    spec = yaml.safe_load(SPEC.read_text())
    if group == "waist":
        return [j["name"] for j in spec["locked"] if j["name"].startswith("waist")]
    return [j["name"] for j in spec["arm" if group == "arms" else "hand"]]


def spec_hand_slots():
    """{side: {joint: slot}} from joint_spec.yaml (dds_index within that side's 7-motor Dex3 message)."""
    out = {"left": {}, "right": {}}
    for j in yaml.safe_load(SPEC.read_text())["hand"]:
        out[j["name"].split("_")[0]][j["name"]] = int(j["dds_index"])
    return out


def group_values(group, v):
    """A number or {regex: number} -> {regex: float}; a dict must match every joint of the group exactly once
    (IsaacLab leaves unmatched joints at their USD value and rejects double matches, both silent / late in a sim)."""
    if not isinstance(v, dict):
        return {".*": float(v)}
    d = {str(k): float(x) for k, x in v.items()}
    for n in spec_names(group):
        hits = [k for k in d if re.fullmatch(k, n)]
        if len(hits) != 1:
            raise ValueError(f"variant: {group} joint {n} matched by {hits or 'no'} key(s) of {d}")
    return d


def validate(V):
    _unknown(V, TOP_KEYS + ("_path",), "the variant")
    R = V.get("robot") or {}
    _unknown(R, ROBOT_KEYS, "robot")
    if R.get("right_hand_wiring", "upstream") not in WIRINGS:
        raise ValueError(f"variant: robot.right_hand_wiring must be one of {WIRINGS}")
    A = R.get("actuators") or {}
    _unknown(A, UPSTREAM_GROUP_EXPR, "robot.actuators")
    for g, params in A.items():
        _unknown(params or {}, ACT_KEYS, f"robot.actuators.{g}")
        for k, v in (params or {}).items():
            group_values(g, v)
    check_joint_limits(R.get("joint_limits") or {})
    C = V.get("colors") or {}
    _unknown(C, COLOR_KEYS, "colors")
    from aaltonomous.scene.scene import Scene
    internal = list(Scene.load().colors)
    _unknown(C.get("objects") or {}, internal, "colors.objects (keys are the INTERNAL colour names)")
    for k in ("bins", "bin_floor", "counter", "floor", "mat_tint"):
        if C.get(k) is not None and len(C[k]) != 3:
            raise ValueError(f"variant: colors.{k} needs 3 values")
    check_words(V.get("color_words") or {}, internal)
    _unknown(V.get("lighting") or {}, ("dome", "key"), "lighting")
    for k, allowed in (("dome", ("intensity", "rgb")), ("key", ("intensity", "rgb", "pos"))):
        _unknown((V.get("lighting") or {}).get(k) or {}, allowed, f"lighting.{k}")
    H = V.get("head_camera") or {}
    _unknown(H, CAMERA_KEYS, "head_camera")
    for k, v in H.items():
        if k == "parent":
            if v not in CAMERA_PARENTS:
                raise ValueError(f"variant: head_camera.parent must be one of {CAMERA_PARENTS}")
        elif v is not None and len(v) != 3:
            raise ValueError(f"variant: head_camera.{k} needs 3 values")
    Y = V.get("body_yaw") or {}
    _unknown(Y, BODY_YAW_KEYS, "body_yaw")
    if float(Y.get("jitter_deg") or 0.0) < 0 or abs(float(Y.get("deg") or 0.0)) > 30:
        raise ValueError(f"variant: body_yaw needs jitter_deg >= 0 and |deg| <= 30: {Y}")
    O = V.get("object_physics") or {}
    _unknown(O, OBJECT_PHYSICS_KEYS, "object_physics")
    if any(float(v) <= 0 for v in O.values() if v is not None):
        raise ValueError(f"variant: object_physics values must be > 0: {O}")
    from .boxes import BOX_KEYS, PER_BOX_KEYS, TABLE_KEYS, report
    T = V.get("table") or {}
    _unknown(T, TABLE_KEYS, "table")
    for k in ("x", "y"):
        if T.get(k) is not None and (len(T[k]) != 2 or float(T[k][0]) >= float(T[k][1])):
            raise ValueError(f"variant: table.{k} must be [lo, hi] with lo < hi")
    if T.get("roughness") is not None and not 0.0 <= float(T["roughness"]) <= 1.0:
        raise ValueError("variant: table.roughness must be in [0, 1]")
    B = V.get("boxes")
    if B is not None:
        _unknown(B, BOX_KEYS, "boxes")
        for n, d in (B.get("per_box") or {}).items():
            _unknown(d or {}, PER_BOX_KEYS, f"boxes.per_box.{n}")
    if V.get("scene"):
        merge_scene(Scene.load().cfg, V["scene"])     # raises on keys scene.yaml doesn't have
    sc = scene(V)
    if sc.size <= 0 or sc.object_height() <= 0:
        raise ValueError("variant: scene.objects.size / height_ratio must be > 0")
    if is_variant_scene(sc):
        bad = report(sc)["problems"]
        if bad:
            raise ValueError("variant: box / table geometry: " + "; ".join(bad))
    from . import messy
    _unknown(V.get("layout") or {}, messy.LAYOUT_KEYS, "layout (the scene.yaml layout knobs go under scene.layout)")
    messy.validate(V.get("layout") or {})
    if messy.active(V):
        c = messy.band_clearance(V, sc)
        if min(c.values()) <= 0.0:
            raise ValueError(f"variant: lying objects in the object band could touch / spawn under a box: {c}")
    return V


def summary(V):
    """JSON-able description for the scene service `info` (null when off)."""
    if not V:
        return None
    out = {k: copy.deepcopy(v) for k, v in V.items() if k != "_path"}
    out["path"] = V.get("_path")
    out["right_hand_wiring"] = wiring(V)
    return out


# ---------------------------------------------------------------------------------------------- robot
def wiring(V):
    return ((V or {}).get("robot") or {}).get("right_hand_wiring", "upstream")


def soft_limit_factor(V, default):
    f = ((V or {}).get("robot") or {}).get("soft_joint_pos_limit_factor")
    return default if f is None else float(f)


def hand_gains_env(s):
    """AALTO_HAND_GAINS="kp,kd[,armature[,effort_limit]]" -> actuator params for the hands group."""
    v = [float(x) for x in s.split(",")]
    return dict(zip(("stiffness", "damping", "armature", "effort_limit"), v))


def actuator_overrides(V, hand_gains=None):
    """[(group, params)] in application order: the variant's groups, then AALTO_HAND_GAINS (wins)."""
    out = [(g, dict(p or {})) for g, p in (((V or {}).get("robot") or {}).get("actuators") or {}).items()]
    if hand_gains:
        out.append(("hands", hand_gains_env(hand_gains)))
    return out


def apply_actuator(act, group, params):
    """Write params onto an IsaacLab actuator cfg (ImplicitActuatorCfg; duck-typed so it is testable offline).
    effort_limit -> the implicit actuator's SOLVER limit: IsaacLab 3 field joint_effort_limit (the deprecated alias
    effort_limit, which upstream's hands use (300), maps to the same for implicit actuators: actuator_compat.py;
    clearing it avoids the 'conflicting values' error and the rated limit then tracks the solver limit)."""
    for k, v in params.items():
        d = group_values(group, v)
        if k == "effort_limit" and hasattr(act, "joint_effort_limit"):
            act.joint_effort_limit, act.effort_limit = d, None
        else:
            setattr(act, k, d)


def all_joint_names():
    """Every joint joint_spec.yaml names (arm, hand, locked = legs + waist)."""
    spec = yaml.safe_load(SPEC.read_text())
    return [j["name"] for k in ("arm", "hand", "locked") for j in spec[k]]


def check_joint_limits(L):
    """robot.joint_limits: every regex matches >= 1 joint_spec joint, no joint is matched twice, [lo, hi] with
    lo < hi when both are set (null = keep the USD value)."""
    names = all_joint_names()
    for k, v in L.items():
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"variant: robot.joint_limits.{k} must be [lo, hi] (rad, null = keep)")
        if v[0] is not None and v[1] is not None and float(v[0]) >= float(v[1]):
            raise ValueError(f"variant: robot.joint_limits.{k}: lo {v[0]} >= hi {v[1]}")
        if not any(re.fullmatch(k, n) for n in names):
            raise ValueError(f"variant: robot.joint_limits.{k} matches no joint of joint_spec.yaml")
    spec = {j["name"]: j for k in ("arm", "hand", "locked") for j in yaml.safe_load(SPEC.read_text())[k]}
    for n, (lo, hi) in joint_limits({"robot": {"joint_limits": L}}, names).items():   # raises on double matches
        lo = spec[n].get("lower", -np.inf) if lo is None else lo     # a kept side is the URDF one (= USD)
        hi = spec[n].get("upper", np.inf) if hi is None else hi
        if lo >= hi:
            raise ValueError(f"variant: robot.joint_limits leaves {n} with an empty range [{lo}, {hi}]")
    return L


def joint_limits(V, names=None):
    """robot.joint_limits -> {joint name: (lo | None, hi | None)} rad for the matched joints ({} = none)."""
    L = ((V or {}).get("robot") or {}).get("joint_limits") or {}
    if not L:
        return {}
    out = {}
    for n in all_joint_names() if names is None else names:
        hits = [k for k in L if re.fullmatch(k, n)]
        if len(hits) > 1:
            raise ValueError(f"variant: joint {n} matched by robot.joint_limits keys {hits}")
        if hits:
            lo, hi = L[hits[0]]
            out[n] = (None if lo is None else float(lo), None if hi is None else float(hi))
    return out


def resolve(d, name, default=None):
    """Value of {regex: v} (or a number) for one joint name."""
    if not isinstance(d, dict):
        return default if d is None else float(d)
    for k, v in d.items():
        if re.fullmatch(k, name):
            return float(v)
    return default


# ---------------------------------------------------------------------------------------------- wiring
def hand_slots(w):
    """{side: {joint: slot}} the sim uses for rt/dex3/<side>/cmd AND /state under wiring w."""
    if w == "upstream":
        return copy.deepcopy(UPSTREAM_SLOTS)
    if w == "spec":
        return spec_hand_slots()
    raise ValueError(w)


def state_order(w):
    """14 joint names in the published state order (left slots 0-6, then right slots 0-6)."""
    s = hand_slots(w)
    return [n for side in ("left", "right") for n, _ in sorted(s[side].items(), key=lambda kv: kv[1])]


# ---------------------------------------------------------------------------------------------- render
def rgb255(c):
    return tuple(float(x) / 255.0 for x in c)


def object_rgb(V, scene, color):
    """Render colour (0..1) of the objects of internal colour `color` (scene.yaml unless the variant overrides)."""
    o = ((V or {}).get("colors") or {}).get("objects") or {}
    return rgb255(o.get(color, scene.colors[color]))


def color_override(V, key, default):
    """colors.bins / counter / floor (0-255 in the file) as 0..1, else default (already 0..1)."""
    c = ((V or {}).get("colors") or {}).get(key)
    return default if c is None else rgb255(c)


def mat_tint(V):
    t = ((V or {}).get("colors") or {}).get("mat_tint")
    return None if t is None else tuple(float(x) for x in t)


def light(V, which, key, default):
    v = (((V or {}).get("lighting") or {}).get(which) or {}).get(key)
    if v is None:
        return default
    return tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else float(v)


def rotvec(deg):
    """Rotation matrix of a rotation vector given in degrees (Rodrigues)."""
    r = np.radians(np.asarray(deg if deg is not None else [0, 0, 0], float))
    a = float(np.linalg.norm(r))
    if a == 0:
        return np.eye(3)
    k = r / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def camera_T(V, T):
    """calibration.yaml T_pelvis_optical (4x4) with the variant's head_camera offset (unchanged when none):
    R = rotvec(rot_offset_deg) @ R_cal @ rotvec(rot_offset_optical_deg), t = t_cal + pos_offset."""
    H = (V or {}).get("head_camera") or {}
    if not any(H.get(k) for k in ("pos_offset", "rot_offset_deg", "rot_offset_optical_deg")):   # parent alone: T
        return T
    T = np.array(T, float)
    T[:3, :3] = rotvec(H.get("rot_offset_deg")) @ T[:3, :3] @ rotvec(H.get("rot_offset_optical_deg"))
    T[:3, 3] += np.asarray(H.get("pos_offset") or [0, 0, 0], float)
    return T


# ---------------------------------------------------------------------------------------------- objects / table
def object_physics(V, key, default):
    v = ((V or {}).get("object_physics") or {}).get(key)
    return default if v is None else float(v)


def object_mesh_scale(sc):
    """(sx, sy, sz) for the object STLs, which export.py wrote at scene.yaml's size / height_ratio: (1, 1, 1)
    unless the variant's scene overlay changes them."""
    from aaltonomous.scene.scene import Scene
    s0 = Scene.load()
    s, h = sc.size / s0.size, sc.object_height() / s0.object_height()
    return (s, s, h)


def camera_parent(V):
    return ((V or {}).get("head_camera") or {}).get("parent") or "pelvis"


# Pose of the camera parent links in the pelvis frame at the zero joint pose, from the sim's robot USD
# (g1_29dof_with_dex3_base_fix.usd; test_variant.py checks these against the file with link_T_pelvis). Constants, NOT
# read at run time: scene_cfg is imported before Kit starts, and importing pxr then breaks Kit's own USD (R3 live).
_D435_R = [[0.6743023922073234, 0.0, 0.7384553364039569], [0.0, 1.0, 0.0], [-0.7384553364039569, 0.0, 0.6743023922073234]]
LINK_POSE_ZERO = {"pelvis": ([0.0, 0.0, 0.0], np.eye(3).tolist()),
                  "torso_link": ([-0.003963499795645475, 0.0, 0.04399999976158142], np.eye(3).tolist()),
                  "d435_link": ([0.05365999788045883, 0.01752999983727932, 0.47387000918388367], _D435_R)}


def link_T_zero(link):
    """4x4 pose of a camera parent link in the pelvis frame at the zero joint pose (LINK_POSE_ZERO)."""
    t, R = LINK_POSE_ZERO[link]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R, float), np.asarray(t, float)
    return T


def link_T_pelvis(usd_path, link):
    """4x4 pose of `link` in the pelvis frame at the zero joint pose, read from the robot USD (its authored link
    transforms are the zero configuration). OFFLINE / tests only: imports pxr, which must not happen in the sim
    process before Kit is up (use link_T_zero there)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(usd_path))
    tc = Usd.TimeCode.Default()
    found = {}
    for prim in Usd.PrimRange(st.GetDefaultPrim()):
        if prim.GetName() in ("pelvis", link) and prim.GetName() not in found and prim.IsA(UsdGeom.Xformable):
            found[prim.GetName()] = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc), float).T
    if set(found) != {"pelvis", link}:
        raise ValueError(f"variant: link {link!r} / pelvis not found in {usd_path}")
    return np.linalg.inv(found["pelvis"]) @ found[link]


def camera_mount(V, T_cal):
    """(parent link, T_parent_optical) for the head camera prim: calibration.yaml's T_pelvis_optical with the
    variant's head_camera offsets (camera_T), re-expressed in head_camera.parent's frame at waist 0 (LINK_POSE_ZERO;
    pure numpy, safe before Kit). Default parent pelvis: exactly camera_T(V, T_cal) (the organisers' mount)."""
    T = camera_T(V, T_cal)
    parent = camera_parent(V)
    if parent == "pelvis":
        return parent, T
    return parent, np.linalg.inv(link_T_zero(parent)) @ np.asarray(T, float)


def camera_follows(V):
    """The head camera needs the runtime follower (hooks.drive_camera): mounted on a moving link, or the robot is
    turned (body_yaw). The organisers' setup (pelvis mount, no body_yaw) never does."""
    return camera_parent(V) != "pelvis" or body_yaw_base_deg(V) is not None


def body_yaw_base_deg(V):
    """Constant part of body_yaw (deg), None without a body_yaw section."""
    Y = (V or {}).get("body_yaw")
    return None if Y is None else float(Y.get("deg") or 0.0)


def body_yaw(V, layout):
    """Body yaw (rad) for an episode with this layout: deg + N(0, jitter_deg) clipped at 3 sigma, the noise seeded by
    the layout (the evaluator's layouts are seeded, so a seed repeats its yaw). None: no body_yaw in the variant, or
    an empty layout (homing / parking keeps the current yaw)."""
    base = body_yaw_base_deg(V)
    if base is None or not layout:
        return None
    import json
    import zlib
    sig = float(V["body_yaw"].get("jitter_deg") or 0.0)
    rng = np.random.default_rng(zlib.crc32(json.dumps(layout, sort_keys=True, default=str).encode()))
    return float(np.radians(base + sig * float(np.clip(rng.standard_normal(), -3.0, 3.0))))


def table_roughness(V, default):
    v = ((V or {}).get("table") or {}).get("roughness")
    return default if v is None else float(v)


# ---------------------------------------------------------------------------------------------- scene
def merge_scene(base, over, where="scene"):
    """Deep merge (dicts merge, everything else replaces); every key must exist in base."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise ValueError(f"variant: {where}.{k} is not in scene.yaml")
        out[k] = merge_scene(out[k], v, f"{where}.{k}") if isinstance(v, dict) and isinstance(out[k], dict) else v
    return out


def scene(V):
    """The Scene the SIM builds / places / scores with: scene.yaml + the variant's `scene` overlay, and, when the
    variant has `boxes` or a table footprint, a boxes.VariantScene (moved / yawed boxes, larger table). V may be a
    loaded variant or its `info` summary (evaluator, probe, check_scene rebuild the sim's scene from it)."""
    from aaltonomous.scene.scene import Scene
    sc = Scene.load()
    if V and V.get("scene"):
        sc = Scene(merge_scene(sc.cfg, V["scene"]))
    T = (V or {}).get("table") or {}
    if V and (V.get("boxes") is not None or T.get("x") is not None or T.get("y") is not None):
        from .boxes import VariantScene, counter_extent, row_poses
        table = None
        if T.get("x") is not None or T.get("y") is not None:
            dx, dy = counter_extent(sc)
            table = (tuple(float(v) for v in T.get("x") or dx), tuple(float(v) for v in T.get("y") or dy))
        poses = row_poses(sc, V["boxes"]) if V.get("boxes") is not None else None
        sc = VariantScene(copy.deepcopy(sc.cfg), poses, table)
    return sc


def is_variant_scene(sc):
    from .boxes import VariantScene
    return isinstance(sc, VariantScene)


# ---------------------------------------------------------------------------------------------- colour words
def check_words(m, internal):
    _unknown(m, internal, "color_words (keys are the INTERNAL colour names)")
    for k, w in m.items():
        if not re.fullmatch(r"[a-z]+", str(w)):
            raise ValueError(f"variant: color_words.{k} = {w!r}: one lower-case word")
    told = [m.get(c, c) for c in internal]
    if len(set(told)) != len(told):
        raise ValueError(f"variant: color_words {m} makes two colours the same word: {told}")
    return m


def parse_words(s):
    """CLI form 'red=orange,blue=purple' -> dict ('' / 'off' / 'none' -> {})."""
    if not s or s in ("off", "none"):
        return {}
    return dict(kv.split("=", 1) for kv in s.split(","))


def rename(text, m):
    """Replace whole colour words (case kept: red / Red / RED) simultaneously, so swaps work. {} = unchanged."""
    m = {k: v for k, v in (m or {}).items() if k != v}
    if not m or not text:
        return text

    def sub(mt):
        w = mt.group(0)
        r = m[w.lower()]
        return r.upper() if w.isupper() and len(w) > 1 else r.capitalize() if w[0].isupper() else r
    return re.sub(r"\b(" + "|".join(map(re.escape, m)) + r")\b", sub, text, flags=re.IGNORECASE)


def inverse(m):
    return {v: k for k, v in (m or {}).items()}


def told_mapping(mapping, m):
    """Internal colour -> bin mapping as the policy reads it from the renamed instruction (logs only; scoring stays
    on the internal mapping, which is the same assignment of objects to bins)."""
    if not mapping:
        return mapping
    out = {}
    for k, b in mapping.items():
        out[tuple(m.get(x, x) for x in k) if isinstance(k, tuple) else m.get(k, k)] = b
    return out


# ---------------------------------------------------------------------------------------------- Isaac side
def set_usd_joint_limits(root, limits):
    """Author lower / upper limits on the revolute joints under `root` (a Usd.Prim) whose prim NAME is a key of
    limits {name: (lo | None, hi | None)} rad (USD stores degrees). Returns {name: (lo, hi)} rad as now authored."""
    from pxr import Usd, UsdPhysics
    done = {}
    for prim in Usd.PrimRange(root):          # instance proxies can't be edited: such a joint counts as not found
        n = prim.GetName()
        if n not in limits or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        j = UsdPhysics.RevoluteJoint(prim)
        for v, attr in zip(limits[n], (j.GetLowerLimitAttr(), j.GetUpperLimitAttr())):
            if v is not None:
                attr.Set(float(np.degrees(v)))
        done[n] = (float(np.radians(j.GetLowerLimitAttr().Get())), float(np.radians(j.GetUpperLimitAttr().Get())))
    return done


def install_joint_limits(robot_cfg, V):
    """robot.joint_limits: wrap the robot's spawn function so that, right after the USD is referenced (before
    PhysX parses it), the matched joints get the variant's limits. Upstream / no limits: cfg untouched. Raises
    at spawn if a joint is not found (a silent no-op would fake a hard stop)."""
    lim = joint_limits(V)
    if not lim:
        return {}
    orig = robot_cfg.spawn.func

    def spawn_with_limits(prim_path, cfg, translation=None, orientation=None, **kwargs):
        prim = orig(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
        roots = [prim]
        try:
            import isaaclab.sim as sim_utils
            roots += [p for p in sim_utils.find_matching_prims(prim_path) if p.GetPath() != prim.GetPath()]
        except Exception as e:     # regex lookup is a bonus (num_envs 1): the spawned prim is what counts
            print(f"[aalto] joint limits: no prim lookup for {prim_path}: {e}")
        for r in roots:
            done = set_usd_joint_limits(r, lim)
            if set(done) != set(lim):
                raise RuntimeError(f"variant joint_limits: joints {sorted(set(lim) - set(done))} not found under "
                                   f"{r.GetPath()}")
            print(f"[aalto] variant joint limits on {r.GetPath()} (rad): "
                  + ", ".join(f"{n} [{lo:+.3f}, {hi:+.3f}]" for n, (lo, hi) in done.items()))
        return prim
    robot_cfg.spawn.func = spawn_with_limits
    return lim


def install_wiring(V):
    """Command side of right_hand_wiring: spec. Wraps upstream DDSActionProvider._setup_joint_mapping so that the
    per-side joint -> HandCmd slot tables (and the precomputed source index tensors get_action uses) follow
    joint_spec.yaml. The target joints and their order are kept (aalto_sim_main's absolute-target wrapper uses them).
    No-op for upstream wiring. Call before the provider is created (register.py)."""
    if wiring(V) != "spec":
        return False
    import torch
    from action_provider.action_provider_dds import DDSActionProvider as P
    if getattr(P, "_aalto_wiring", False):
        return True
    orig, slots = P._setup_joint_mapping, spec_hand_slots()

    def setup(self):
        orig(self)
        if not getattr(self, "enable_dex3", False):
            return
        for side in ("left", "right"):
            mp = getattr(self, f"{side}_hand_joint_mapping")
            new = {n: slots[side][n] for n in mp}                       # same keys, same order = same targets
            setattr(self, f"{side}_hand_joint_mapping", new)
            setattr(self, f"_{side}_hand_source_indices", list(new.values()))
            old = getattr(self, f"_{side}_hand_source_idx_t")
            setattr(self, f"_{side}_hand_source_idx_t", torch.tensor(list(new.values()), dtype=torch.long,
                                                                     device=old.device))
        print(f"[aalto] variant: Dex3 command wiring = joint_spec.yaml: right {self.right_hand_joint_mapping}")
    P._setup_joint_mapping = setup
    P._aalto_wiring = True
    return True


def dex3_state_func(V):
    """State side of right_hand_wiring: the observation term that publishes rt/dex3/*/state. Upstream's function
    unchanged for upstream wiring; for spec a wrapper that points upstream's gather cache (dex3_state._obs_cache,
    hard-coded articulation indices) at the joints in joint_spec order, by NAME, then calls upstream."""
    from tasks.common_observations import dex3_state as ds
    if wiring(V) != "spec":
        return ds.get_robot_dex3_joint_states
    import torch
    order = state_order("spec")

    def get_robot_dex3_joint_states_spec(env, enable_dds: bool = True):
        c = ds._obs_cache
        jp = env.scene["robot"].data.joint_pos
        dev = (jp.torch if hasattr(jp, "torch") else jp).device
        if c.get("_aalto_t") is None or c["hand_idx_t"] is not c["_aalto_t"] or c["device"] != dev:
            names = list(env.scene["robot"].data.joint_names)
            up = [names.index(n) for n in ds.get_robot_girl_joint_names()]
            if up != UPSTREAM_STATE_IDX:   # upstream's indices are not what its names say: report, still go by name
                print(f"[aalto] WARNING dex3 state: upstream indices {UPSTREAM_STATE_IDX} != by-name {up}")
            idx = [names.index(n) for n in order]
            t = torch.tensor(idx, dtype=torch.long, device=dev)
            c.update(hand_idx_t=t, _aalto_t=t, device=dev, batch=None)     # batch None: upstream re-expands idx
            print(f"[aalto] variant: Dex3 state wiring = joint_spec.yaml: {order} -> articulation {idx}")
        return ds.get_robot_dex3_joint_states(env, enable_dds)
    return get_robot_dex3_joint_states_spec
