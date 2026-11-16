"""Train on a tracker-like smoothed ball? Leave one game out on the 10 external matches.

    python -m solution.smooth_train_exp

The competition tracker smooths its ball (solution/touch_diag.py, aggregate numbers). On held-out
external matches whose ball is smoothed the same way (touch_exp.tracker_ball: a 21-frame trailing
average), the submitted models drop from 0.664 to 0.515 spellF1, because they were trained on
clean balls. Here the training matches get the same smoothing on top of the usual augmentation,
for the team model and the dead-ball model alike, and three arms are compared:

    A  trained on clean balls, no touches     (the fallback submission)
    B  trained on smoothed balls, no touches
    C  trained on smoothed balls, touches on  (features.TOUCH_* as configured)

each scored on held-out matches with the tracker-like ball ("tracker", the regime that resembles
our match) and with the usual clean ball ("default"), two seeds, decoded exactly as submitted
(viterbi3, flip rule, P(dead) on 7 IDSSE, temperature 2). Nothing here reads the competition match.
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import dead_ball, dead_features, features  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import logo2 as L2  # noqa: E402
from solution import model as M  # noqa: E402
from solution import touch_exp as T  # noqa: E402  (also sets augment.TRACKER_SMOOTH_W = 1)

# Params
SPECS = ("tracker", "default")
ARMS = {"A": dict(smooth=False, touch=False), "B": dict(smooth=True, touch=False),
        "C": dict(smooth=True, touch=True)}
OUT = os.path.join(os.path.dirname(L.CACHE_DIR), "smooth_train_exp.json")

_AUG, _TEAM, _DEAD = {}, {}, {}


def training_copy(name, smooth):
    """The training augmentation of logo (default regime, train seed), plus the tracker-like
    smoothing of the ball when `smooth`."""
    key = (name, smooth)
    if key not in _AUG:
        aug = dict(L.degrade(name, ("default",), L.train_seed(name, 0)))
        if smooth:
            aug["ball_xy"] = T.tracker_ball(aug["ball_xy"].astype(np.float64)).astype(np.float32)
        _AUG[key] = aug
    return _AUG[key]


def team_rows(name, smooth, touch):
    key = (name, smooth, touch)
    if key not in _TEAM:
        arrays = L.data()[name]
        aug = training_copy(name, smooth)
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


def dead_rows(name, smooth):
    key = (name, smooth)
    if key not in _DEAD:
        arrays = L.data()[name]
        idx = np.flatnonzero(L.inside_mask(arrays))[::M.SUBSAMPLE]
        _, X = dead_features.compute(training_copy(name, smooth), arrays["runs"])
        _DEAD[key] = X[idx], arrays["dead"][idx].astype(np.int8)
    return _DEAD[key]


def fit(parts):
    gbm = dict(M.GBM, max_features=L.REF.max_features, random_state=L.REF.rs)
    return HistGradientBoostingClassifier(**gbm).fit(
        np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))


def run():
    names = L.names()
    scores = {a: {s: np.zeros((len(names), len(T.SEEDS), 4)) for s in SPECS} for a in ARMS}
    for i, name in enumerate(names):
        runs = L.data()[name]["runs"]
        team_train, dead_train = L.train_names(name, "4"), L.train_names(name, "7")
        team = {a: fit([team_rows(n, c["smooth"], c["touch"]) for n in team_train])
                for a, c in ARMS.items()}
        dead = {sm: fit([dead_rows(n, sm) for n in dead_train]) for sm in (False, True)}
        for spec in SPECS:
            for j, k in enumerate(T.SEEDS):
                held = T.held_arrays(name, spec, k)
                _, XD = dead_features.compute(held, runs)
                forced = dead_ball.pre_kickoff_mask(held, runs)
                pdead = {sm: L2.dead_proba(m, XD) for sm, m in dead.items()}
                for a, c in ARMS.items():
                    ph, d_us, d_them = T.held_inputs(name, held, runs, team[a], c["touch"])
                    h = dict(ph=ph, pd7=pdead[c["smooth"]], forced=forced, d1_us=d_us,
                             d1_them=d_them)
                    scores[a][spec][i, j] = L2.score(name, L2.flip_codes(name, h, T.FINAL),
                                                     full=True)
        print("done", name, "  ".join("{} {:.3f}".format(a, scores[a]["tracker"][i, :, 0].mean())
                                      for a in ARMS), flush=True)
    features.TOUCH_FEATURES = True
    with open(OUT, "w") as handle:
        json.dump({a: {s: v.tolist() for s, v in d.items()} for a, d in scores.items()}, handle)
    report(names, scores)


def report(names, scores):
    met = np.array([n in L.METRICA for n in names])
    print("\n| arm | held-out ball | spellF1 | Metrica | IDSSE | macroF1 | possessions pred / true |")
    print("|---|---|---|---|---|---|---|")
    for spec in SPECS:
        for a in ARMS:
            s = L2.per_match(scores[a][spec])
            print("| {} | {} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} / {:.0f} |".format(
                a, spec, s.mean(), s[met].mean(), s[~met].mean(),
                scores[a][spec][:, :, 1].mean(), scores[a][spec][:, :, 2].mean(),
                scores[a][spec][:, :, 3].mean()))
    for spec in SPECS:
        ref = L2.per_match(scores["A"][spec])
        for a in ("B", "C"):
            s = L2.per_match(scores[a][spec])
            print("{} vs A on {}: {:+.4f}, wins {}/10".format(
                a, spec, s.mean() - ref.mean(), int((s > ref + 1e-9).sum())))
    print("\n| match | " + " | ".join("{} tracker".format(a) for a in ARMS) + " |")
    print("|---|" + "---|" * len(ARMS))
    for i, n in enumerate(names):
        print("| {} | {} |".format(n, " | ".join(
            "{:.3f}".format(L2.per_match(scores[a]["tracker"])[i]) for a in ARMS)))


if __name__ == "__main__":
    run()
