"""Load config.yaml, build the 14-vectors of the Dex3 postures (left/right mirrored) and check them against
joint_spec.yaml so a bad posture fails at load time, not on the robot."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import yaml

from aaltonomous.interface.safety import JointSpec

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
POSTURES = ("open", "preshape", "closed")
AUTO = "auto"   # AALTO_PROFILE value: detect sim / real robot (policy_server.make_policy -> autoprofile.py)


def mirror_left_to_right(left7):
    """Left DDS order (thumb0,1,2, middle0,1, index0,1) -> right DDS order (thumb0,1,2, index0,1, middle0,1),
    signs flipped (joint_spec.yaml: the hands are mirrored, the same motion has opposite signs) EXCEPT thumb_0: its
    axis is not mirrored between the hands (URDF forward kinematics: the same value swings both thumbs towards
    their index finger; the sim behaves so, and in the first real teleop data the right thumb_0 is negative like
    the left one). Its own inverse. (Until 2026-11-14 thumb_0 was negated too: harmless, every posture had 0.)"""
    l = np.asarray(left7, float)
    return np.array([l[0], -l[1], -l[2], -l[5], -l[6], -l[3], -l[4]])


def _set(cfg, k, v):  # "motion.arm_max_vel": 0.3
    d = cfg
    keys = k.split(".")
    for kk in keys[:-1]:
        d = d[kk]
    d[keys[-1]] = v


def profile_items(profiles, name, _chain=()):
    """(dotted key, value) pairs of profiles.<name>: `inherit: <parent>` first applies the parent's (recursively),
    then the profile's own keys (later wins). Unknown profiles and inheritance cycles are errors."""
    if name in _chain:
        raise ValueError(f"config profile inheritance cycle: {' -> '.join(_chain + (name,))}")
    if name not in profiles:
        raise ValueError(f"unknown config profile {name!r} (have {sorted(profiles)})")
    own = dict(profiles[name] or {})
    parent = own.pop("inherit", None)
    return (profile_items(profiles, parent, _chain + (name,)) if parent else []) + list(own.items())


def load_config(path=None, overrides=None, profile=None):
    """config.yaml + the selected profile's overrides (profiles.<name>, dotted keys; `inherit:` = a parent profile's
    first) + `overrides` + $AALTO_SET. Profile: argument > $AALTO_PROFILE > the file's `profile:` key; "none" / "" =
    base values only. $AALTO_SET: experiment hook, applied last: JSON {dotted.key: value} or @/path/to/file.json (e.g.
    to run the policy server or the oracle with one feature off, or with other hand postures, without another code
    tree). Profile "auto" (the image's AALTO_PROFILE: the policy server picks sim / robot itself, autoprofile.py,
    and passes the profile explicitly) = the file's default here."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    if profile is None:
        profile = os.environ.get("AALTO_PROFILE", cfg.get("profile"))
    if profile == AUTO:
        profile = cfg.get("profile")
    profiles = cfg.pop("profiles", None) or {}
    if profile and profile != "none":
        for k, v in profile_items(profiles, profile):
            _set(cfg, k, v)
    cfg["profile"] = profile or "none"
    for k, v in (overrides or {}).items():
        _set(cfg, k, v)
    env = os.environ.get("AALTO_SET")
    if env:
        for k, v in json.loads(Path(env[1:]).read_text() if env.startswith("@") else env).items():
            _set(cfg, k, v)
        cfg["aalto_set"] = env
    return cfg


def hand_postures(cfg, spec: JointSpec | None = None):
    """{name: (14,) hand vector} for open / preshape / closed (+ extra postures), validated against the URDF limits."""
    spec = spec or JointSpec()
    h = cfg["hand"]
    missing = [n for n in POSTURES if n not in h]
    if missing:
        raise ValueError(f"hand postures missing: {missing}")
    right = h.get("right") or {}
    out = {}
    for name in [k for k in h if k != "right"]:  # open, preshape, closed (required) + any extra (e.g. release)
        left = np.asarray(h[name], float)
        r = np.asarray(right[name], float) if name in right else mirror_left_to_right(left)
        v = np.r_[left, r]
        bad = [spec.hand_names[i] for i in np.flatnonzero((v < spec.hand_lower - 1e-9) | (v > spec.hand_upper + 1e-9))]
        if bad:
            raise ValueError(f"hand posture {name!r} outside joint_spec limits: {bad}")
        out[name] = v
    return out


def home_arm(cfg, spec: JointSpec | None = None):
    spec = spec or JointSpec()
    q = np.asarray(cfg["home_arm_q"], float)
    if q.shape != (14,):
        raise ValueError("home_arm_q must have 14 values")
    m = float(cfg["ik"]["limit_margin"])
    bad = [spec.arm_names[i] for i in np.flatnonzero((q < spec.arm_lower + m) | (q > spec.arm_upper - m))]
    if bad:
        raise ValueError(f"home_arm_q too close to the joint limits: {bad}")
    return q

