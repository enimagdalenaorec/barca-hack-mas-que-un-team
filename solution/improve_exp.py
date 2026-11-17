"""Overnight improvement search on top of Version D (external matches only).

    python -m solution.improve_exp cache ARM[,ARM] FOLD [FOLD ...]   # held-out probabilities

Leave one game out on the 10 external matches, same harness as solution/robust_exp.py. Each arm's
held-out P(home) / P(dead) are cached per fold under external/cache/improve/, so decoders are
tuned afterwards without refitting (solution/improve_dec.py). Nothing here reads the competition
match. Verification code, not part of the submission.

An arm:
    tw      trailing-mean width of the TRAINING ball (1 = clean); a tuple = one width per training
            match, cycled (mixed-width training)
    shift   de-lag: the ball is moved `shift` frames earlier (ball'[t] = ball[t + shift]) inside
            each run, in training and on the held-out match alike. A trailing mean over w frames
            shifted by (w - 1) / 2 is exactly the centred mean.
    touch   features.TOUCH_FEATURES
    sw      spell weights (ref seconds, lo, hi) or None
    rs      GBM random_state
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import dead_ball, dead_features, features  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import logo2 as L2  # noqa: E402
from solution import model as M  # noqa: E402
from solution import robust_exp as R  # noqa: E402
from solution import smooth_train_exp as S  # noqa: E402
from solution import touch_exp as T  # noqa: E402  (sets augment.TRACKER_SMOOTH_W = 1)

OUT_DIR = os.path.join(os.path.dirname(L.CACHE_DIR), "improve")
BASES = ("default", "harsh", "lag")
KINDS = ("clean", "trail21", "gauss6")
EXTRA_KINDS = ("trail13", "trail29")      # default base only: tracker width mismatch
EV_SHIFTS = (0, 5, 10, 15)                # flip evidence against a de-lagged ball
D_SW = (8.0, 0.25, 4.0)
ARMS = {
    "D": dict(tw=21, shift=0, touch=True, sw=D_SW, rs=0),
    "S": dict(tw=21, shift=10, touch=True, sw=D_SW, rs=0),
    # clean-trained with touches and weights, for ensembling / "is smooth training still needed"
    "C": dict(tw=1, shift=0, touch=True, sw=D_SW, rs=0),
    # spell-weight variants on the de-lagged arm
    "S_w4": dict(tw=21, shift=10, touch=True, sw=(4.0, 0.25, 4.0), rs=0),
    "S_w16": dict(tw=21, shift=10, touch=True, sw=(16.0, 0.25, 4.0), rs=0),
    "S_w8c8": dict(tw=21, shift=10, touch=True, sw=(8.0, 0.125, 8.0), rs=0),
    "S_w8c2": dict(tw=21, shift=10, touch=True, sw=(8.0, 0.5, 2.0), rs=0),
    "S_now": dict(tw=21, shift=10, touch=True, sw=None, rs=0),
    # team-model variants of Version D (P(dead) is Version D's: dead=False)
    "D_w4": dict(tw=21, shift=0, touch=True, sw=(4.0, 0.25, 4.0), rs=0, dead=False),
    "D_w16": dict(tw=21, shift=0, touch=True, sw=(16.0, 0.25, 4.0), rs=0, dead=False),
    "D_w8c8": dict(tw=21, shift=0, touch=True, sw=(8.0, 0.125, 8.0), rs=0, dead=False),
    "D_w8c2": dict(tw=21, shift=0, touch=True, sw=(8.0, 0.5, 2.0), rs=0, dead=False),
    "D_w4c8": dict(tw=21, shift=0, touch=True, sw=(4.0, 0.125, 8.0), rs=0, dead=False),
    "D7": dict(tw=21, shift=0, touch=True, sw=D_SW, rs=0, dead=False, idsse="7"),
    "D_mix": dict(tw=(13, 21, 29), shift=0, touch=True, sw=D_SW, rs=0),
    # mixed-width training (robust to the unknown tracker width), de-lag by 10
    "S_mix": dict(tw=(13, 21, 29), shift=10, touch=True, sw=D_SW, rs=0),
    # other de-lags with w=21 training
    "S5": dict(tw=21, shift=5, touch=True, sw=D_SW, rs=0, dead=False),
    "S15": dict(tw=21, shift=15, touch=True, sw=D_SW, rs=0, dead=False),
}


def delag(ball, runs, shift):
    """ball'[t] = ball[t + shift] inside each run; the last `shift` frames of a run are missing."""
    if not shift:
        return ball
    out = np.full_like(ball, np.nan)
    for a, b in runs:
        if b - a > shift:
            out[a:b - shift] = ball[a + shift:b]
    return out


def kind_ball(ball, kind):
    if kind.startswith("trail"):
        return T.tracker_ball(ball.astype(np.float64), int(kind[5:])).astype(np.float32)
    return R.smooth_ball(ball, kind)


def train_width(arm, name, train):
    tw = arm["tw"]
    return tw if isinstance(tw, int) else tw[sorted(train).index(name) % len(tw)]


_AUG = {}


def training_copy(name, w, shift):
    key = (name, w, shift)
    if key not in _AUG:
        aug = dict(L.degrade(name, ("default",), L.train_seed(name, 0)))
        ball = aug["ball_xy"].astype(np.float64)
        if w > 1:
            ball = T.tracker_ball(ball, w)
        aug["ball_xy"] = delag(ball, L.data()[name]["runs"], shift).astype(np.float32)
        _AUG[key] = aug
    return _AUG[key]


_TEAM, _DEAD = {}, {}


def team_rows(name, w, shift, touch):
    key = (name, w, shift, touch)
    if key not in _TEAM:
        arrays = L.data()[name]
        aug = training_copy(name, w, shift)
        owner = arrays["owner"]
        idx = np.flatnonzero((owner >= 0) & ~arrays["dead"])[::M.SUBSAMPLE]
        features.TOUCH_FEATURES = touch
        xs, ys = [], []
        for swap in (False, True):
            _, X = features.compute(aug, arrays["runs"], swap=swap)
            xs.append(X[idx])
            ys.append((owner[idx] == (1 if swap else 0)).astype(np.int8))
        _TEAM[key] = np.concatenate(xs), np.concatenate(ys)
    return _TEAM[key]


def dead_rows(name, w, shift):
    key = (name, w, shift)
    if key not in _DEAD:
        arrays = L.data()[name]
        idx = np.flatnonzero(L.inside_mask(arrays))[::M.SUBSAMPLE]
        _, X = dead_features.compute(training_copy(name, w, shift), arrays["runs"])
        _DEAD[key] = X[idx], arrays["dead"][idx].astype(np.int8)
    return _DEAD[key]


def team_weights(name, sw):
    arrays = L.data()[name]
    idx = np.flatnonzero((arrays["owner"] >= 0) & ~arrays["dead"])[::M.SUBSAMPLE]
    saved = S.SPELL_REF_FRAMES, S.SPELL_W
    S.SPELL_REF_FRAMES, S.SPELL_W = sw[0] * 25, (sw[1], sw[2])
    try:
        w = S.spell_weights(arrays)[idx]
    finally:
        S.SPELL_REF_FRAMES, S.SPELL_W = saved
    return np.concatenate([w, w])


def fit(parts, weights=None, rs=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    gbm = dict(M.GBM, max_features=L.REF.max_features, random_state=rs)
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    if weights is None:
        return HistGradientBoostingClassifier(**gbm).fit(X, y)
    return HistGradientBoostingClassifier(**gbm).fit(X, y, sample_weight=np.concatenate(weights))


ALL_REGIMES = [(b, k) for b in BASES for k in KINDS] + [("default", k) for k in EXTRA_KINDS]
REPORT_REGIMES = [("default", k) for k in KINDS + EXTRA_KINDS] + [("harsh", "trail21"),
                                                                   ("lag", "trail21")]


def regimes(need_ev=True):
    """Every regime for the evidence file and the first two arms; the reported ones after."""
    return ALL_REGIMES if need_ev or os.environ.get("IMPROVE_ALL") else REPORT_REGIMES


def path(name, tag):
    return os.path.join(OUT_DIR, "{}__{}.npz".format(name, tag))


def cache_fold(i, arm_names):
    names = L.names()
    name = names[i]
    arm_names = [a for a in arm_names if not os.path.isfile(path(name, a))]
    need_ev = not os.path.isfile(path(name, "ev"))
    if not arm_names and not need_ev:
        return
    t0 = time.time()
    runs = L.data()[name]["runs"]
    team_train, dead_train = L.train_names(name, "4"), L.train_names(name, "7")
    models = {}
    for a in arm_names:
        c = ARMS[a]
        tt = L.train_names(name, c.get("idsse", "4"))
        parts = [team_rows(n, train_width(c, n, tt), c["shift"], c["touch"]) for n in tt]
        weights = None if c["sw"] is None else [team_weights(n, c["sw"]) for n in tt]
        team = fit(parts, weights, c["rs"])
        dead = None
        if c.get("dead", True):
            dead = fit([dead_rows(n, train_width(c, n, dead_train), c["shift"])
                        for n in dead_train], rs=c["rs"])
        models[a] = team, dead
        print(name, "fitted", a, "{:.0f}s".format(time.time() - t0), flush=True)
    out = {a: {} for a in arm_names}
    ev = {}
    for base, kind in regimes(need_ev):
        for k in T.SEEDS:
            held0 = dict(L2.held_arrays(name, base, k))
            ball = kind_ball(held0["ball_xy"], kind)
            tag = "{}/{}/{}/".format(base, kind, k)
            by_shift = {}
            for sh in sorted(set(EV_SHIFTS) | {ARMS[a]["shift"] for a in arm_names}):
                held = dict(held0)
                held["ball_xy"] = delag(ball, runs, sh)
                by_shift[sh] = held
                if need_ev and sh in EV_SHIFTS:
                    d0, d1 = M.nearest_distances(held, runs)
                    ev[tag + "d0_{}".format(sh)] = d0.astype(np.float32)
                    ev[tag + "d1_{}".format(sh)] = d1.astype(np.float32)
                    ev[tag + "forced_{}".format(sh)] = dead_ball.pre_kickoff_mask(held, runs)
            feats = {}
            for a in arm_names:
                c = ARMS[a]
                key = (c["shift"], c["touch"])
                if key not in feats:
                    held = by_shift[c["shift"]]
                    features.TOUCH_FEATURES = c["touch"]
                    _, X0 = features.compute(held, runs, swap=False)
                    _, X1 = features.compute(held, runs, swap=True)
                    _, XD = dead_features.compute(held, runs)
                    feats[key] = X0, X1, XD
                X0, X1, XD = feats[key]
                team, dead = models[a]
                inside = ~np.isnan(X0).all(axis=1)
                p = np.full(len(X0), np.nan)
                p[inside] = 0.5 * (team.predict_proba(X0[inside])[:, 1]
                                   + 1 - team.predict_proba(X1[inside])[:, 1])
                out[a][tag + "ph"] = p.astype(np.float32)
                if dead is not None:
                    out[a][tag + "pd"] = L2.dead_proba(dead, XD).astype(np.float32)
        print(name, base, kind, "{:.0f}s".format(time.time() - t0), flush=True)
    features.TOUCH_FEATURES = True
    os.makedirs(OUT_DIR, exist_ok=True)
    for a in arm_names:
        np.savez_compressed(path(name, a) + ".tmp.npz", **out[a])
        os.replace(path(name, a) + ".tmp.npz", path(name, a))
    if need_ev:
        np.savez_compressed(path(name, "ev") + ".tmp.npz", **ev)
        os.replace(path(name, "ev") + ".tmp.npz", path(name, "ev"))
    print(name, "done", "{:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        for arg in sys.argv[3:]:
            cache_fold(int(arg), sys.argv[2].split(","))
