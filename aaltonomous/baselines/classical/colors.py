"""Episode colour classes: the colours this episode's mapping names, matched to the colour classes of the world, so
that the words need not be the configuration's class names (sim: red / yellow / blue; robot: orange / yellow /
purple) and the configuration need not know what the instruction will call the objects. Pure Python, ~0.1 ms.

    ec = match_colors(mapping_colors(mapping), world, reject, cfg["perception"]["color"]["match"], words)
    ec.refs       # {class: [hues]}: what perception classifies into (Perception.set_classes(ec.refs, ec.reject, ...))
    ec.names      # class names (object_bin's `classes`); a combined class "blue/purple" = could be either
    ec.summary()  # one log line

world = {class: [hues]} (config perception.color.refs: MEASURED hues of the objects that exist in this world), reject
= hues that are no object (the bins). Hues are OpenCV H 0-180, distances circular. Parameters (perception.color.match):
  1. A named colour that is a world class: that class with its measured hues (never a lexicon hue).
  2. A named colour without a hue (black, white, grey, brown, ...): no class (warning); its key never applies.
  3. A named colour that is no world class: its lexicon hue (instruction.HUE). Merged with an UNNAMED world class (the
     same objects under another name: "red" for the robot's orange objects, "purple" for the sim's blue ones) iff
     that class is the colour's nearest within merge_max, the colour is that class's nearest named claimant, and
     neither has a runner-up within `ambiguous` of that distance (else: ambiguous, no merge, a warning). Merged: the
     class takes the colour's name, keeps its measured hues and gets the lexicon hue as well unless that lies within
     extra_sep of another class's hues (the measured hue is the anchor; the lexicon hue helps where the world's
     configuration is stale, e.g. the sim configuration with orange objects).
  4. World classes neither named nor merged stay classes of their own (a "default" key gives them its bin).
  5. A class that has only a lexicon hue (3., unmerged) nearer than min_sep to another class: the two cannot be told
     apart reliably; they become one combined class "a/b" (warning) whose objects get a bin only where every name
     agrees (object_bin), never a guess between them.
  6. Reject hues within reject_drop of any class hue are dropped for the episode (the instruction says objects of
     that colour exist: e.g. "orange" objects with the sim's orange bins); the bins are then excluded by geometry only.
These are the prior. The scene then calibrates it (Calibrator, after every look; perception.color.calib): the blobs'
own hues form clusters, the clusters are assigned to the named colours / unnamed classes / other lexicon colours, and a
clear assignment moves that class to where its objects are, for the rest of the episode (see Calibrator). Per object,
perception keeps a class only with a margin (Perception._decide, color.obj_margin); a cluster two named colours could
equally be gets both names ("orange/yellow": object_bin gives it a bin only where they agree). Never a guess.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .instruction import HUE

MATCH = {"merge_max": 25.0, "ambiguous": 8.0, "extra_sep": 10.0, "min_sep": 15.0, "reject_drop": 6.0, "absorb": 7.5}


def hue_dist(a, b):
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _dist(A, B):
    return min((hue_dist(a, b) for a in A for b in B), default=float("inf"))


def _hues(v):
    return [float(h) for h in (v if isinstance(v, (list, tuple)) else [v])]


@dataclass
class EpisodeColors:
    refs: dict                                       # class -> [hues]
    reject: list                                     # reject hues kept for this episode
    alias: dict = field(default_factory=dict)        # world class -> episode class (colour-first detection names)
    notes: dict = field(default_factory=dict)        # class -> how it was made
    problems: list = field(default_factory=list)     # [{"kind", "severity", "msg"}]
    dropped: list = field(default_factory=list)      # reject hues dropped (why)

    @property
    def names(self):
        return list(self.refs)

    def summary(self):
        cl = "; ".join(f"{k} {'/'.join(f'{h:g}' for h in v)} ({self.notes.get(k, '')})" for k, v in self.refs.items())
        rj = ("reject " + "/".join(f"{h:g}" for h in self.reject)) if self.reject else "no reject"
        rj += "".join(f"; {d}" for d in self.dropped)
        return f"colour classes: {cl}; {rj}" + "".join(f"; {p['severity']}: {p['msg']}" for p in self.problems)


def match_colors(said, world, reject=(), params=None, words=None, observed=None, partners=None,
                 keep_measured=False) -> EpisodeColors:
    """said: canonical colours the mapping names (instruction.mapping_colors), world: {class: hue | [hues]}, reject:
    [hues], params: perception.color.match (defaults MATCH), words: Parse.words (log only). observed: {colour or
    world class: hue} seen in this episode's scene (Calibrator): that hue replaces the prior one (keep_measured: a
    world class keeps its measured hues as well); partners {world class: named colour} the prior merged: an observed
    named colour keeps its partner if that lies within `absorb` of what was seen (the same objects), else the partner
    is a class of its own again. See the module docstring."""
    p = {**MATCH, **(params or {})}
    world = {k: _hues(v) for k, v in world.items()}
    obs = {k: float(v) for k, v in (observed or {}).items()}

    def seen(k):
        return [obs[k]] + (list(world[k]) if keep_measured and k in world else [])
    said = [c for c in dict.fromkeys(said)]
    refs, notes, probs, lex_only, alias = {}, {}, [], set(), {}

    def name(c):
        w = [x for x in (words or {}).get(c, []) if x != c]
        return f"{c} (said {'/'.join(w)})" if w else c
    pend = []
    for c in said:
        if c in obs:
            refs[c], notes[c] = seen(c), f"seen H{obs[c]:g}"
            if c in world:
                alias[c] = c
        elif c in world:
            refs[c], notes[c], alias[c] = list(world[c]), "measured", c
        elif HUE.get(c) is None:
            probs.append({"kind": "no_hue", "severity": "warning",
                          "msg": f"{name(c)}: no hue to recognise it by (objects of that colour get no bin)"})
        else:
            pend.append(c)
    free = [w for w in world if w not in refs]
    whue = {w: seen(w) if w in obs else list(world[w]) for w in free}
    for w in free:                                   # the prior's partner, seen where it was expected: same objects
        c = (partners or {}).get(w)
        if c in obs and w not in obs and _dist([obs[c]], world[w]) <= p["absorb"]:
            alias[w] = c
    free = [w for w in free if w not in alias]
    near = {(c, w): _dist([HUE[c]], whue[w]) for c in pend for w in free}
    near = {k: d for k, d in near.items() if d <= p["merge_max"]}
    merged = {}
    for c in pend:
        cand = sorted((d, w) for (c2, w), d in near.items() if c2 == c)
        if not cand:
            continue
        d, w = cand[0]
        claim = sorted((d2, c2) for (c2, w2), d2 in near.items() if w2 == w)
        rivals = [x for d2, x in cand[1:] if d2 - d < p["ambiguous"]] + \
                 [x for d2, x in claim if x != c and d2 - d < p["ambiguous"]]
        if claim[0][1] != c and claim[0][0] < d:
            continue                                 # another named colour is nearer to w: it may take it
        if rivals:
            probs.append({"kind": "ambiguous", "severity": "warning",
                          "msg": f"{name(c)}: {w} ({d:g}) vs {', '.join(rivals)} too close to call: not merged"})
            continue
        merged[c] = (w, d)
    for c in pend:
        if c in merged:
            w, d = merged[c]
            refs[c], alias[w] = list(whue[w]), c
            notes[c] = f"= unnamed {w}, {d:g} from lexicon {HUE[c]:g}"
        else:
            refs[c], notes[c] = [float(HUE[c])], "lexicon only"
            lex_only.add(c)
    refs = {k: refs[k] for k in said if k in refs}  # said order, then the unnamed world classes, then seen others
    for w in free:
        if w not in alias:
            refs[w], alias[w] = list(whue[w]), w
            notes[w] = f"seen H{obs[w]:g}, not named" if w in obs else "measured, not named"
    for k, h in obs.items():                         # seen colours of neither (a distractor): a class of their own
        if k not in refs and k not in world and k not in said:
            refs[k], notes[k] = [h], f"seen H{h:g}, not named"
    for c in merged:                                 # the lexicon hue as an extra reference where it is clear
        h = float(HUE[c])
        others = [x for k, v in refs.items() if k != c for x in v]
        if _dist([h], others) >= p["extra_sep"] and _dist([h], refs[c]) >= 1.0:
            refs[c].append(h)
            notes[c] += " + lexicon"
        elif _dist([h], refs[c]) >= 1.0:
            notes[c] += f" (lexicon {h:g} too near another class)"
    # lexicon-only classes too close to another class: combined (union of the close pairs)
    group = {k: {k} for k in refs}
    ks = list(refs)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if (a in lex_only or b in lex_only) and _dist(refs[a], refs[b]) < p["min_sep"]:
                g = group[a] | group[b]
                for k in g:
                    group[k] = g
    out, done = {}, set()
    for k in ks:
        if k in done:
            continue
        g = [x for x in ks if x in group[k]]
        done |= set(g)
        if len(g) == 1:
            out[k] = refs[k]
            continue
        nm = "/".join(g)
        out[nm] = [h for x in g for h in refs[x]]
        notes[nm] = " + ".join(f"{x}: {notes[x]}" for x in g)
        for w, e in list(alias.items()):
            if e in g:
                alias[w] = nm
        gap = min(_dist(refs[a], refs[b]) for i, a in enumerate(g) for b in g[i + 1:])
        probs.append({"kind": "inseparable", "severity": "warning",
                      "msg": f"{' / '.join(name(x) for x in g)}: hues {gap:g} apart and not all measured: one class "
                             f"{nm}, a bin only where they agree"})
    allh = [h for v in out.values() for h in v]
    kept, dropped = [], []
    for r in _hues(list(reject)):
        if _dist([r], allh) < p["reject_drop"]:
            dropped.append(f"reject {r:g} dropped ({_dist([r], allh):g} from a class)")
        else:
            kept.append(r)
    return EpisodeColors(out, kept, alias, notes, probs, dropped)


# ---------------------------------------------------------------------------------------------------- calibration
CALIB = {"link": 7.5, "max_sd": 8.0, "sigma_measured": 5.0, "sigma_lexicon": 6.0, "tol": 9.0, "unexplained": 13.0,
         "margin": 4.0, "unnamed_penalty": 4.0, "max_shift": 10.0, "shift_step": 2.0, "sigma_shift": 4.0,
         "shift_min": 4.0, "sep_floor": 8.0, "approach": 2.0, "near": 20.0, "max_clusters": 6, "max_per_colour": 2,
         "outlier": 6.0, "keep_measured": True, "pair_max": 12.0, "pair_cost": 0.0, "own_tie": 1.0}


def calibrate_look(cal, perception, dets):
    """One look's detections (perception.Detection: hue, hue_sd, hue_hist) -> Calibrator.update; changed classes go
    to perception (set_classes) and this look's detections are re-voted (relabel); clusters two labels could equally
    be get both names ("orange/yellow": a bin only where they agree). Returns (changed, log line or None)."""
    blobs = [(getattr(d, "hue", None), 0.0 if getattr(d, "hue_hist", None) is None else float(d.hue_hist[1].sum()),
              getattr(d, "hue_sd", None)) for d in dets]
    changed, line, over = cal.update(blobs)
    if changed:
        perception.set_classes(cal.ec.refs, cal.ec.reject, cal.ec.alias)
        perception.relabel(dets)
    for i, name in over.items():
        dets[i].color = name
    # count check (an episode has at most max_per_colour objects of one colour; a sanity check, not a rule): more in
    # one class = its boundaries are wrong: logged, and the members `outlier` or more from their median hue are unknown
    k, notes = int(cal.p.get("max_per_colour") or 0), []
    for name in sorted({d.color for d in dets if d.color != "unknown" and "/" not in d.color}) if k > 0 else ():
        ds = [d for d in dets if d.color == name and getattr(d, "hue", None) is not None]
        if len(ds) <= k:
            continue
        med = min((d.hue for d in ds), key=lambda h: sum(hue_dist(h, e.hue) for e in ds))   # medoid hue
        far = [d for d in ds if hue_dist(d.hue, med) >= cal.p["outlier"]]
        for d in far:
            d.color = "unknown"
        notes.append(f"{len(ds)} objects called {name} (at most {k} of a colour): H "
                     f"{'/'.join(f'{d.hue:g}' for d in ds)}" + (f"; H {'/'.join(f'{d.hue:g}' for d in far)} -> unknown"
                                                                 if far else ""))
    if notes:
        line = "; ".join(([line] if line else []) + notes)
    return changed, line


def _circ_mean(h, w):
    a = np.asarray(h, float) * (np.pi / 90.0)
    w = np.asarray(w, float)
    return round(float(np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 2.0) % 180.0, 1)


def clusters(blobs, link, max_sd=None):
    """[(hue, weight, sd)] -> [(centre, weight, member indices)]: single linkage on the hue circle (neighbours at most
    `link` apart), centre = weighted circular mean. Blobs without a hue or with a spread above max_sd (two colours in
    one blob) are left out."""
    pts = sorted((float(h) % 180.0, i) for i, (h, w, sd) in enumerate(blobs)
                 if h is not None and w > 0 and (max_sd is None or sd is None or sd <= max_sd))
    if not pts:
        return []
    gaps = [((pts[(j + 1) % len(pts)][0] - pts[j][0]) % 180.0, j) for j in range(len(pts))]
    start = (max(gaps)[1] + 1) % len(pts)            # cut the circle at its widest gap
    order = pts[start:] + pts[:start]
    groups, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if hue_dist(a[0], b[0]) <= link:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    return [(_circ_mean([blobs[i][0] for _, i in g], [blobs[i][1] for _, i in g]),
             float(sum(blobs[i][1] for _, i in g)), [i for _, i in g]) for g in groups]


class Calibrator:
    """The episode's colour classes calibrated from what the scene shows.

        cal = Calibrator(said, world, reject, match_params, calib_params, words)
        cal.ec                                   # EpisodeColors now (= match_colors until something is seen)
        changed, line = cal.update([(hue, weight, sd), ...])   # after each look: its geometry blobs

    Each look's blob hues are clustered (single linkage, `link`) and the clusters assigned one-to-one to labels: the
    named colours (anchors: their measured class hue(s); a merged class's lexicon hue as well; else the lexicon hue),
    the unnamed world classes (measured) and the other lexicon colours (+ unnamed_penalty: they explain only what the
    others do not: a distractor, or an unnamed yellow next to "orange"). A cluster's cost under a label is (distance
    / sigma)^2 to its nearest anchor (sigma_measured / sigma_lexicon; candidates within `tol`), a cluster nobody
    explains costs `unexplained`; the lowest total wins. A cluster's label counts only if every assignment labelling
    it otherwise costs >= `margin` more; else it is ambiguous and nothing changes for it (its objects keep the prior's
    verdict, which is "no bin" where the prior was ambiguous).
    With max_shift > 0 a common hue shift of the whole scene (white balance; cost (shift / sigma_shift)^2, only with
    two or more clusters) is part of the fit. Special cases: a merged colour whose two hypotheses (its partner's
    measured hue, its word's hue) are both seen as separate clusters is decided for neither ("orange/unknown"); a
    cluster no named colour can be (beyond `tol` of all) gets the nearest other colour's name (a distractor: no bin
    unless "everything else").
    A clear assignment becomes an observation (the label's class takes the cluster's hue, keep_measured: next to its
    measured hues; match_colors(observed=)) only if:
      - the label was not observed before (observations persist for the episode and never flip: a later cluster
        within `link` of an observed hue, with no other label's hue within own_tie as near, IS that label before any
        assignment and refines it; a farther one is ignored);
      - a label with a measured prior: the cluster lies >= shift_min from its anchors (else the prior stands) and is
        not nearer, by more than `approach`, to a measured label this look does not see than the prior was;
      - a label without one: the cluster lies >= sep_floor from every measured label this look does not see.
    """

    def __init__(self, said, world, reject=(), match_params=None, calib_params=None, words=None):
        self.said = [c for c in dict.fromkeys(said)]
        self.world = {k: _hues(v) for k, v in world.items()}
        self.reject, self.mp, self.words = list(reject), match_params, words
        self.p = {**CALIB, **(calib_params or {})}
        self.prior = match_colors(self.said, self.world, self.reject, match_params, words)
        self.ec = self.prior
        self.partners = {w: e for w, e in self.prior.alias.items() if e != w and e in self.said}   # merged w -> c
        self.obs = {}                                # label -> [hue, weight]
        self.last = None

    def _labels(self):
        """{label: (anchors [(hue, sigma, measured)], penalty)}."""
        p, pr = self.p, self.prior
        sm, sl = p["sigma_measured"], p["sigma_lexicon"]
        out = {}
        keep = p["keep_measured"]

        def seen(k):
            return [(self.obs[k][0], sm, True)] + ([(h, sm, True) for h in self.world[k]]
                                                   if keep and k in self.world else [])
        for c in self.said:
            if HUE.get(c) is None:
                continue
            if c in self.obs:
                out[c] = (seen(c), 0.0)
            elif c in self.world:
                out[c] = ([(h, sm, True) for h in self.world[c]], 0.0)
            else:
                part = [w for w, e in pr.alias.items() if e == c and w != c]
                out[c] = ([(h, sm, True) for w in part for h in self.world[w]] + [(float(HUE[c]), sl, False)], 0.0)
        for w, hs in self.world.items():
            e = pr.alias.get(w, w)
            if w in self.said or (e != w and "/" not in e):
                continue                             # named, or merged into a named colour
            out[w] = (seen(w) if w in self.obs else [(h, sm, True) for h in hs], 0.0)
        for x, h in HUE.items():
            if h is None or x in out or x in self.world:
                continue
            out[x] = ([(self.obs[x][0], sm, True)] if x in self.obs else [(float(h), sl, False)],
                      p["unnamed_penalty"])
        # order on the hue circle (assignments must keep it): what was seen, else the measured hue of a world class,
        # else the word's lexicon hue
        self._key = {k: (self.obs[k][0] if k in self.obs else self.world[k][0] if k in self.world else float(HUE[k]))
                     for k in out}
        return out

    def _ordered(self, cl, combo):
        """No two nearby clusters (< 45 apart, labels' hues too) assigned to different labels in the opposite order of
        the labels' hues: of two clusters in the orange-yellow range the lower one is the orange."""
        sd = [(cl[k][0], self._key[nm]) for k, (_, nm) in enumerate(combo) if nm is not None]
        for i, (c1, k1) in enumerate(sd):
            for c2, k2 in sd[i + 1:]:
                dc, dk = (c2 - c1 + 90.0) % 180.0 - 90.0, (k2 - k1 + 90.0) % 180.0 - 90.0
                if k1 != k2 and 0.5 < abs(dc) < 45 and abs(dk) < 45 and dc * dk < 0:
                    return False
        return True

    @staticmethod
    def _cost(h, anchors):
        return min(((hue_dist(h, a) / s) ** 2, a, m) for a, s, m in anchors)

    def update(self, blobs):
        """blobs: [(hue, weight, sd)] of one look. Returns (classes changed, log line or None: only when it differs
        from the previous look's)."""
        p = self.p
        cl = sorted(clusters(blobs, p["link"], p["max_sd"]), key=lambda c: -c[1])[:int(p["max_clusters"])]
        if not cl:
            return False, None, {}
        labels = self._labels()
        # a cluster within `link` of a class calibrated earlier this episode IS that class, unless another label's
        # hue (calibrated or prior) is about as near: persistence, whatever the priors would say
        owned = {}
        for k, (c, _, _) in enumerate(cl):
            d = sorted((hue_dist(c, h), nm) for nm, (h, _) in self.obs.items())
            if not d or d[0][0] > p["link"]:
                continue
            others = [hue_dist(c, a) for nm, (anc, _) in labels.items() if nm != d[0][1] for a, _, _ in anc]
            if min(others, default=float("inf")) - d[0][0] >= p["own_tie"]:
                owned[k] = d[0][1]
        tots = []                                    # (cost, labels per cluster, common hue shift)
        ms = float(p["max_shift"]) if len(cl) >= 2 else 0.0     # one cluster cannot tell a common shift
        for dl in np.arange(-ms, ms + 1e-9, float(p["shift_step"])) if ms > 0 else [0.0]:
            cand = []
            for k, (c, _, _) in enumerate(cl):
                if k in owned:
                    cand.append([(self._cost(c - dl, labels[owned[k]][0])[0], owned[k])])
                    continue
                opts = []
                for nm, (anc, pen) in labels.items():
                    q = self._cost(c - dl, anc)[0]
                    if q <= p["tol"]:
                        opts.append((q + pen, nm))
                cand.append(sorted(opts)[:2] + [(float(p["unexplained"]), None)])
            q0 = (dl / p["sigma_shift"]) ** 2
            for combo in itertools.product(*cand):
                ks = {}
                for k, (_, nm) in enumerate(combo):
                    if nm is not None:
                        ks.setdefault(nm, []).append(k)
                # one cluster per label, or two within pair_max (one colour's objects split by its own spread, at
                # pair_cost), in the labels' order on the hue circle
                if all(len(v) == 1 or (len(v) == 2 and hue_dist(cl[v[0]][0], cl[v[1]][0]) <= p["pair_max"])
                       for v in ks.values()) and self._ordered(cl, combo):
                    tots.append((q0 + sum(q for q, _ in combo) + p["pair_cost"] * sum(len(v) == 2 for v in ks.values()),
                                 tuple(nm for _, nm in combo), float(dl)))
        tots.sort(key=lambda t: (t[0], str(t[1]), abs(t[2])))
        best, lab, shift = tots[0]
        seen = {nm for nm in lab if nm is not None}
        notes, changed, over, took, confirmed = [], False, {}, set(), []
        # a merged named colour whose two hypotheses (the partner's measured hue, the word's lexicon hue) are both in
        # the scene as different clusters: which one it is cannot be told: none of them gets it
        contested = set()
        for nm, (anc, pen) in labels.items():
            if nm in self.obs or not any(m for _, _, m in anc) or all(m for _, _, m in anc):
                continue
            lex, mea = set(), set()
            for k, (c, _, _) in enumerate(cl):
                if k in owned:
                    continue
                q, a, m = self._cost(c - shift, anc)
                other = lab[k] not in (None, nm) and self._cost(c - shift, labels[lab[k]][0])[0] + \
                    labels[lab[k]][1] < q                    # clearly another label's cluster
                if q <= p["tol"] and not other:
                    (mea if m else lex).add(k)
            if lex and mea:
                contested.add(nm)
                for k in (lex | mea):
                    for i in cl[k][2]:
                        over[i] = f"{nm}/unknown"
                notes.append(f"{nm}: H{'/'.join(f'{cl[k][0]:g}' for k in sorted(lex | mea))} (its word's hue and "
                             f"its measured partner's both seen): not decided")
        for k, (c, w, idx) in enumerate(cl):
            nm = lab[k]
            if k in owned:                           # calibrated before: refine within link, never re-opened
                h0, w0 = self.obs[nm]
                hn = _circ_mean([h0, c], [w0, w])
                if hue_dist(hn, h0) >= 0.5:
                    changed = True
                    notes.append(f"{nm} H{h0:g} -> H{hn:g} (refined)")
                self.obs[nm] = [hn, w0 + w]
                for i in idx:
                    over.pop(i, None)
                continue
            if any(i in over for i in idx):
                continue
            if not any(self._cost(c - shift, labels[s][0])[0] <= p["tol"] for s in self.said if s in labels):
                other = nm if nm is not None else min((x for x, h in HUE.items() if h is not None),
                                                      key=lambda x: hue_dist(c - shift, HUE[x]))
                for i in idx:                        # no named colour can be it: an unnamed colour (its own name)
                    over[i] = other
                if nm is None:
                    continue
            if nm is None or nm in contested:
                continue
            alt = next(((t, ls[k]) for t, ls, _ in tots if ls[k] != nm), (float("inf"), None))
            if alt[0] - best < p["margin"]:
                notes.append(f"H{c:g} ({len(idx)}): {nm} or {alt[1] or 'nothing'}, not clear")
                weak = {labels[x][1] > 0.0 for x in (nm, alt[1])} if alt[1] is not None else None
                if weak is not None and len(weak) == 1:   # both real labels, or both other lexicon colours: a bin
                    for i in idx:                         # only where both agree (vs. a real one: its prior verdict)
                        over[i] = f"{nm}/{alt[1]}"
                continue
            anc, pen = labels[nm]
            q, a, am = self._cost(c - shift, anc)
            if nm in self.obs:
                h0, w0 = self.obs[nm]
                if hue_dist(c, h0) <= p["link"] or (nm in took and hue_dist(c, h0) <= p["pair_max"]):
                    hn = _circ_mean([h0, c], [w0, w])
                    if hue_dist(hn, h0) >= 0.5:
                        changed = True
                        notes.append(f"{nm} H{h0:g} -> H{hn:g} (refined)")
                    self.obs[nm] = [hn, w0 + w]
                else:
                    notes.append(f"H{c:g}: far from {nm}'s H{h0:g} seen before, kept")
                continue
            meas = [x for x, s, m in anc if m]
            unseen = [(o, [x for x, s, m in labels[o][0] if m]) for o in labels
                      if o != nm and o not in seen and labels[o][1] == 0.0]
            unseen = [(o, xs) for o, xs in unseen if xs]
            if meas:
                if _dist([c], [x for x, s, m in anc]) < p["shift_min"]:
                    confirmed.append((nm, c, w))     # where the prior expects it (taken only if others move)
                    continue
                bad = [o for o, xs in unseen if _dist([c], xs) < min(p["near"], _dist([x for x, s, m in anc], xs)
                                                                    - p["approach"])]
            else:
                bad = [o for o, xs in unseen if _dist([c], xs) < p["sep_floor"]]
            if bad:
                notes.append(f"H{c:g} ({len(idx)}) = {nm}? towards unseen {', '.join(bad)}: not taken")
                continue
            self.obs[nm] = [c, w]
            took.add(nm)
            changed = True
            notes.append(f"{nm} <- H{c:g} ({len(idx)} obj, cost {q:.1f}, next {alt[0] - best:.1f} more; prior "
                         f"{'measured' if am else 'lexicon'} H{a:g}{f', scene shift {shift:+g}' if shift else ''})")
        if any(labels[nm][1] == 0.0 for nm in took):    # a class moved: the clear ones the prior expected take their
            for nm, c, w in confirmed:                  # hue too, so the classes' boundaries come from one scene
                if nm not in self.obs:
                    self.obs[nm] = [c, w]
                    notes.append(f"{nm} <- H{c:g} (as expected)")
        if changed:
            self.ec = match_colors(self.said, self.world, self.reject, self.mp, self.words,
                                   {k: v[0] for k, v in self.obs.items()}, self.partners, p["keep_measured"])
        line = ("colour calibration: " + "; ".join(notes)) if notes else None
        if changed:
            line += " => " + ", ".join(f"{k} {'/'.join(f'{h:g}' for h in v)}" for k, v in self.ec.refs.items())
        if line == self.last and not changed:
            return False, None, over
        self.last = line
        return changed, line, over
