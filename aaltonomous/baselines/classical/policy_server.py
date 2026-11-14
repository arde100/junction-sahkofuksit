"""Reference submission: the classical sorter (ClassicalPolicy, rungs 1-2) behind the policy-server protocol.

    python -m aaltonomous.baselines.classical.policy_server            # :8000, what the Docker image runs
    docker build -f aaltonomous/docker/baseline.Dockerfile -t aalto/baseline-classical:v1 .

Same planner/perception as run.py / sim_eval.py (no forked logic). It runs threaded: while it looks and plans
(IK, ~0.1-few s) act() returns 1-tick holds at once, so it never blocks the runner (command rate stays ~30 Hz);
motion comes back in `motion.chunk`-tick chunks. It returns "done" when nothing sortable is left.
Uses only head_rgb, head_depth, arm_q, hand_q, t from obs; camera intrinsics/extrinsics come from the
calibration.yaml baked into the image (spec §11), so rebuild after a calibration change.
Target bins come from the instruction text of each reset (instruction.py; no text: the fixed rung-1 mapping); what
was understood (and any problem) is logged once per reset. Objects whose bin was not understood stay on the counter.
Rung 3 (insert_shapes: the reset's task id, or a lid / hole instruction; insert.py) sorts nothing: the holes are out
of the arms' reach, and the lid lies on the left bin (a drop there scores -1).
Env: AALTO_TIME_LIMIT (s, default 120; must match the runner's --time-limit), AALTO_ARM_VEL (rad/s override),
AALTO_PROFILE (config.yaml profile; unset = the file's `sim`, `none` = robot base values; `auto` = the image's
default: detect sim / real robot per episode and use `sim` / `none`, autoprofile.py).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from aaltonomous.deploy.policy_server import Policy, serve

from .config import AUTO, load_config
from .policy import ClassicalPolicy

log = logging.getLogger("aaltonomous.baseline")


class ClassicalServerPolicy(Policy):
    name = "classical-rung1"

    def __init__(self, time_limit=None, arm_vel=None, profile=None, **kw):
        tl = float(time_limit or os.environ.get("AALTO_TIME_LIMIT", 120.0))
        ov = {}
        av = arm_vel or os.environ.get("AALTO_ARM_VEL")
        if av:
            ov["motion.arm_max_vel"] = float(av)
        if profile is not None:  # explicit profile (autoprofile: one server policy per world), else $AALTO_PROFILE
            kw["config"] = load_config(overrides=ov, profile=profile)
        self.pol = ClassicalPolicy(threaded=True, require_confirm=False, time_limit=tl, overrides=ov,
                                   log=lambda m: log.info(m), **kw)
        self.action_horizon = int(self.pol.cfg["motion"]["chunk"])

    STALE_MAX_S = 90.0   # a planning thread of the previous episode still alive after this long: reset anyway

    def reset(self, instruction, task=None, seed=None):
        # An episode can end mid-look (time limit) with its planning thread still running, and the kinematics /
        # planner state is not thread-safe. NEVER wait for it here: the runner's watchdog safe-stops an episode
        # after 1 s without a command, counted from its own reset (a blocked reset = a dead episode = a failed sim
        # gate). The reset is deferred instead: act() holds until that thread has ended, then resets and goes on.
        job = self.pol._job
        self._pending, self._probe = (instruction, task, seed), None
        self._stale = job if job is not None and job["result"] is None else None
        self._stale_t0, self._hold = time.monotonic(), None
        if self._stale is None:
            self._reset_now()
        else:
            log.info("reset task=%s seed=%s deferred: the previous episode's look is still running", task, seed)

    def reset_waiting(self):
        """True while a deferred reset waits for the previous episode's planning thread (autoprofile then holds)."""
        if getattr(self, "_pending", None) is None:
            return False
        st = self._stale
        if st is not None and st["result"] is None and time.monotonic() - self._stale_t0 < self.STALE_MAX_S:
            return True
        return False

    def _reset_now(self):
        instruction, task, seed = self._pending
        self._pending = self._stale = None
        log.info("reset task=%s seed=%s instruction=%r", task, seed, instruction)
        # logs the mapping it understood (ClassicalPolicy._mapping); task insert_shapes -> rung-3 mode (insert.py)
        self.pol.reset({"instruction": instruction, "task": task})
        log.info("mapping (%s): %s", self.pol.mapping_source, self.pol.mapping)
        if self._probe is not None:
            probe, self._probe = self._probe, None
            self.feed_probe(probe)

    def feed_probe(self, result):
        """autoprofile hand-over: the world probe's still arm = the first sample of the sag feed-forward estimate."""
        if getattr(self, "_pending", None) is not None:   # reset deferred: after it
            self._probe = result
            return
        r = self.pol.feed_probe(result)
        if r is not None:
            log.info("sag feed-forward: probe sample (%d obs) %s", r[3], "used" if r[1].any() else "refused")

    def act(self, obs):
        if getattr(self, "_pending", None) is not None:
            if self.reset_waiting():   # hold the first pose seen (a constant target: no step for the safety filter)
                if self._hold is None:
                    sp = self.pol.spec
                    self._hold = (np.clip(np.asarray(obs["arm_q"], np.float64), sp.arm_lower, sp.arm_upper),
                                  np.clip(np.asarray(obs["hand_q"], np.float64), sp.hand_lower, sp.hand_upper))
                return {"arm_q_target": self._hold[0][None].copy(), "hand_q_target": self._hold[1][None].copy(),
                        "done": False}
            self._reset_now()
        a = self.pol.act(obs)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


def make_policy():
    """AALTO_PROFILE=auto: AutoProfilePolicy (one ClassicalServerPolicy per world, detection per episode); else
    ClassicalServerPolicy on the configured profile (forced: no detection)."""
    if os.environ.get("AALTO_PROFILE", "").strip() == AUTO:
        from .autoprofile import AutoProfilePolicy
        return AutoProfilePolicy(lambda p: ClassicalServerPolicy(profile=p), log=log.info)
    return ClassicalServerPolicy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = make_policy()
    if hasattr(pol, "wait_ready"):  # built in the background; serve once both exist (the runner waits for the port)
        pol.wait_ready(300.0)
    log.info("classical baseline loaded in %.1f s", time.monotonic() - t0)
    serve(pol, port=int(os.environ.get("PORT", 8000)))
