"""Registers the Aalto-* gym ids on import (imported by aaltonomous/sim/aalto_sim_main.py after
upstream `tasks`). Run: TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh"""
import gymnasium as gym

from . import env_cfg, variant

# AALTO_VARIANT right_hand_wiring: spec -> patch upstream's DDS action provider class before sim_main creates it
# (no-op otherwise; the state side is in env_cfg)
variant.install_wiring(variant.active())

for _id, _cfg in (("Aalto-Sort-G129-Dex3-Joint", env_cfg.AaltoSortEnvCfg),
                  ("Aalto-Insert-G129-Dex3-Joint", env_cfg.AaltoInsertEnvCfg)):
    gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv",
                 kwargs={"env_cfg_entry_point": _cfg}, disable_env_checker=True)
