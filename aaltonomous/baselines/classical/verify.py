"""Grasp verification from the measured finger joints alone (no force or tactile input).

The closed posture squeezes ~9 mm into a 45 mm object, so fingers that hold an object stop short of the commanded
posture, while an empty hand reaches it. Signal per hand: sig = max over its 7 joints of |hand_q - commanded target|,
read while the closed posture is commanded. Sim, 54 grasps with ground truth: ambiguous 0.3 s after the closed
posture is first fully commanded (transient contact while a missed object is squeezed out), cleanly separable from
1 s on (held 0.016-0.076, not held 0.000-0.004). A real hand has a nonzero no-load error: thresholds are per profile
(config `verify`, off in the base config).

    mon = GraspMonitor(after_s=1.0, empty_below=0.008, confirm=2)
    v = mon.update(hand_q7, target7, closed_s)    # once per observation: "unknown" | "held" | "empty"

Pure numpy, no robot / planner state: the policy feeds it one observation per act() call per hand.
"""
from __future__ import annotations

import numpy as np

UNKNOWN, HELD, EMPTY = "unknown", "held", "empty"


FINGERS = slice(3, 7)   # the two fingers' joints in either hand's DDS order (0-2 = thumb)


def grasp_signal(hand_q, target, kind="abs7"):
    """How far one hand's fingers are from the commanded posture (rad).
    "abs7": max |hand_q - target| over all 7 joints. Validated in the sim (259 grasps: held >= 0.007, typically
            0.037; empty <= 0.0074 one second after the close, ~0.0002 once settled).
    "finger_shortfall": largest closing-direction shortfall of the 4 finger joints, max(0, sign(target) * (target -
            hand_q)); fingers close away from 0 (left negative, right positive). For the real hand (kp 1.5): its thumb
            has a constant no-load error (thumb_0 +0.075 rad) and every joint a no-load error of 0.04-0.12 rad, so
            only fingers held back by an object separate (first real data drop: held 0.13-0.37, empty 0.013-0.043)."""
    q, t = np.asarray(hand_q, float), np.asarray(target, float)
    if kind == "finger_shortfall":
        return float(max(0.0, (np.sign(t[FINGERS]) * (t[FINGERS] - q[FINGERS])).max()))
    return float(np.abs(q - t).max())


def grasp_window(phases, hand, sl):
    """(k_closed, k_release, target) of a plan for the hand columns `sl` of `hand` (T, 14): the first tick of the
    `close` phase at which the closed posture is fully commanded, the first tick of the `open` phase (release starts)
    and the closed posture (7,). None if the plan has no close or open phase."""
    ph = {name: (s, e) for name, s, e in phases}
    if "close" not in ph or "open" not in ph:
        return None
    s, e = ph["close"]
    target = np.asarray(hand[e - 1, sl], float).copy()
    at = np.abs(np.asarray(hand[s:e, sl], float) - target).max(axis=1) < 1e-9
    return s + int(np.argmax(at)), ph["open"][0], target


class GraspMonitor:
    """Verdict for one grasp of one hand. `empty` only after `confirm` consecutive observations with
    sig < empty_below, none earlier than `after_s` after the closed posture was first fully commanded; `held` while
    sig >= empty_below. `empty` latches (an empty hand cannot pick the object up again); `slipped` = it was `held`
    before it became `empty` (object lost during the carry)."""

    def __init__(self, after_s=1.0, empty_below=0.008, confirm=2, signal="abs7"):
        self.after_s, self.empty_below, self.confirm = float(after_s), float(empty_below), max(1, int(confirm))
        self.signal = signal or "abs7"
        self.reset()

    def reset(self):
        self.verdict, self.sig, self.n_low, self.held_seen = UNKNOWN, None, 0, False

    @property
    def slipped(self):
        return self.verdict == EMPTY and self.held_seen

    def update(self, hand_q, target, closed_s):
        """hand_q, target: (7,) measured / commanded joints of this hand. closed_s: seconds since the closed posture
        was first fully commanded (None: not yet). Feed it only until the release starts."""
        if self.verdict == EMPTY or closed_s is None or closed_s < self.after_s:
            return self.verdict  # (unknown until after_s)
        self.sig = grasp_signal(hand_q, target, self.signal)
        if self.sig >= self.empty_below:
            self.n_low, self.held_seen, self.verdict = 0, True, HELD
        else:
            self.n_low += 1
            if self.n_low >= self.confirm:
                self.verdict = EMPTY
        return self.verdict
