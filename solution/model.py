"""Train the "which team has the ball" model and measure it honestly, one held-out match at a time.

    python -m solution.model --evaluate     # leave-one-match-out scores + switch-cost tuning
    python -m solution.model --train        # fit on every training match, save data/cache/model.pkl
    python -m solution.model --learning-curve   # does more training data still help?

The model: gradient-boosted trees (scikit-learn HistGradientBoostingClassifier) on the per-team
features of solution/features.py, predicting P(our team has the ball) on live frames. Training
matches are first made to look like our automatic tracking (solution/augment.py), and every frame
is used twice -- as seen by each team -- so the model has no home/away bias.

Honest testing: for each match with a full H/A/D label (Metrica and IDSSE), the model is trained
on all the OTHER matches and scored on that one with the official metric library, through the
whole pipeline -- our dead-ball rule on the augmented ball, the model, then Viterbi smoothing.
The same is done for the published baseline rule (nearest player, +/-50 vote) so the two can be
compared on matches neither has seen. SkillCorner matches only ever train (no dead-ball label).

The Viterbi switch cost is chosen here, as the value with the best mean held-out spellF1.
"""

import argparse
import os
import pickle
import sys
import zlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import DATA_DIR  # noqa: E402
from solution import augment, dead_ball, decode, download_external, features  # noqa: E402
from solution import external_games, sample_games  # noqa: E402

# Params
MODEL_PATH = os.path.join(DATA_DIR, "cache", "model.pkl")
SUBSAMPLE = 5          # train on every 5th live frame; neighbours are nearly identical
SWITCH_COSTS = (2, 4, 6, 8, 10, 14, 20, 30)  # the chosen one lives in decode.SWITCH_COST
BASELINE_WINDOW = 50
# Chosen in solution/experiments.py (combo): held-out spellF1 0.645 against 0.626 for
# 300 trees / 31 leaves / lr 0.05 on the same training matches, better on 7 of 7.
GBM = dict(max_iter=600, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, random_state=0)
# SkillCorner trains nothing: leaving it out raised held-out spellF1 from 0.578 to 0.626, better
# on 7 of 7 (solution/experiments.py, mix). Its labels come from SkillCorner's own software and
# its tracking from a broadcast feed at 10 fps; both are further from our target than the rest.
SOURCES = ("idsse",)
# How the training matches are degraded before training (solution/augment.py REGIMES).
AUG_REGIME = "default"


def training_matches():
    """[(name, arrays)] for every match we train on: Metrica + the sources in SOURCES."""
    out = [("metrica{}".format(n), sample_games.load(n)) for n in sample_games.GAMES]
    for source, match in download_external.converted():
        if source in SOURCES:
            out.append(("{}_{}".format(source, match), external_games.load(source, match)))
    return out


def seed_of(name):
    return zlib.crc32(name.encode()) % (2 ** 31)


_ROWS = {}


def params_key(params):
    """Hashable form of an augment params dict, for memo keys."""
    return tuple(sorted((params or {}).items()))


def rows(name, arrays, params=None):
    """(X, y) training rows of one match: augmented, both team views, live frames, subsampled.

    Memoised per match and augment params, because leave-one-out refits on the same matches many
    times.
    """
    key = (name, params_key(params))
    if key not in _ROWS:
        _ROWS[key] = _rows(name, arrays, params)
    return _ROWS[key]


def _rows(name, arrays, params=None):
    aug = augment.augment(arrays, seed_of(name), params)
    owner = arrays["owner"]
    keep = owner >= 0
    if int(arrays["has_dead"]):
        keep &= ~arrays["dead"]
    idx = np.flatnonzero(keep)[::SUBSAMPLE]
    xs, ys = [], []
    for swap in (False, True):
        _, X = features.compute(aug, arrays["runs"], swap=swap)
        xs.append(X[idx])
        ys.append((owner[idx] == (1 if swap else 0)).astype(np.int8))
    return np.concatenate(xs), np.concatenate(ys)


def fit(matches, params=None):
    parts = [rows(name, arrays, params) for name, arrays in matches]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return HistGradientBoostingClassifier(**GBM).fit(X, y)


def p_home(model, arrays, runs):
    """P(home has the ball) per frame: the mean of the two team views."""
    _, X0 = features.compute(arrays, runs, swap=False)
    _, X1 = features.compute(arrays, runs, swap=True)
    p = np.full(len(X0), np.nan)
    inside = ~np.isnan(X0).all(axis=1)
    p[inside] = 0.5 * (model.predict_proba(X0[inside])[:, 1]
                       + 1 - model.predict_proba(X1[inside])[:, 1])
    return p


def labels_from(p, arrays, runs, switch_cost):
    """Codes H/A/D/X for a whole match: dead rule, then Viterbi on each live stretch."""
    dead = dead_ball.dead_mask(arrays, runs)
    codes = np.full(len(p), pm.X, dtype=np.int8)
    live = np.zeros(len(p), dtype=bool)
    for start, stop in runs:
        codes[start:stop][dead[start:stop]] = pm.D
        live[start:stop] = ~dead[start:stop]
    for start, stop in dead_ball.stretches(live, runs):
        path = decode.viterbi(p[start:stop], switch_cost)
        codes[start:stop] = np.where(path == 0, pm.H, pm.A)
    return codes


def labels_from3(p_home, p_dead, arrays, runs, costs):
    """Codes H/A/D/X for a whole match: one three-state Viterbi per run (decode.viterbi3), frames
    before the detected kick-off forced dead. costs = (switch, stop, restart)."""
    forced = dead_ball.pre_kickoff_mask(arrays, runs)
    codes = np.full(len(p_home), pm.X, dtype=np.int8)
    letters = np.array([pm.H, pm.A, pm.D], dtype=np.int8)
    for start, stop in runs:
        path = decode.viterbi3(p_dead[start:stop], p_home[start:stop], forced[start:stop], *costs)
        codes[start:stop] = letters[path]
    return codes


def baseline_codes(arrays, runs):
    """The published baseline's team rule, applied after our dead-ball rule (pipeline.build)."""
    from solution import pipeline
    letters = pipeline.build(arrays, runs, BASELINE_WINDOW)
    return pm.label_codes(letters)


def truth_codes(arrays):
    codes = np.full(len(arrays["owner"]), pm.X, dtype=np.int8)
    for start, stop in arrays["runs"]:
        seg = slice(start, stop)
        codes[seg] = np.where(arrays["dead"][seg], pm.D,
                              np.where(arrays["owner"][seg] == 0, pm.H,
                                       np.where(arrays["owner"][seg] == 1, pm.A, pm.X)))
    return codes


def score(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def evaluate(matches):
    labelled = [m for m in matches if int(m[1]["has_dead"])]
    results = {"baseline": [], "argmax": []}
    results.update({c: [] for c in SWITCH_COSTS})
    for name, arrays in labelled:
        model = fit([m for m in matches if m[0] != name])
        held = augment.augment(arrays, seed_of(name) + 1)
        runs = arrays["runs"]
        gt = truth_codes(arrays)
        p = p_home(model, held, runs)
        results["baseline"].append(score(gt, baseline_codes(held, runs)))
        results["argmax"].append(score(gt, labels_from(p, held, runs, 0.0)))
        for cost in SWITCH_COSTS:
            results[cost].append(score(gt, labels_from(p, held, runs, cost)))
        print("  {:<14} baseline spellF1 {:.3f}  model+viterbi(10) {:.3f}  gt possessions {}".format(
            name, results["baseline"][-1][0], results[10][-1][0], results[10][-1][3]), flush=True)

    print("\nheld-out mean over {} matches        spellF1  macroF1  possessions (truth)".format(
        len(labelled)))
    for key, rows_ in results.items():
        r = np.array(rows_, dtype=float)
        label = key if isinstance(key, str) else "viterbi cost {}".format(key)
        print("  {:<28} {:.3f}    {:.3f}    {:>5.0f} ({:.0f})".format(
            label, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean()))
    best = max(SWITCH_COSTS, key=lambda c: np.mean([r[0] for r in results[c]]))
    print("\nbest switch cost: {}".format(best))
    return results, best


def learning_curve(matches, sizes=(3, 6, 9, 11), seed=0):
    """Held-out spellF1 when training on only `size` of the other matches (random subsets).

    If the score is still climbing at the largest size, more training matches would help.
    """
    from solution.decode import SWITCH_COST
    rng = np.random.default_rng(seed)
    labelled = [m for m in matches if int(m[1]["has_dead"])]
    curve = {size: [] for size in sizes}
    for name, arrays in labelled:
        others = [m for m in matches if m[0] != name]
        held = augment.augment(arrays, seed_of(name) + 1)
        gt = truth_codes(arrays)
        for size in sizes:
            subset = [others[i] for i in rng.choice(len(others), size, replace=False)]
            p = p_home(fit(subset), held, arrays["runs"])
            curve[size].append(score(gt, labels_from(p, held, arrays["runs"], SWITCH_COST))[0])
    print("training matches   held-out spellF1 (mean over {} matches)".format(len(labelled)))
    for size in sizes:
        print("  {:>2}               {:.3f}".format(size, np.mean(curve[size])))
    return curve


def fingerprint(matches, params=None):
    """Everything the saved model depends on. A saved model with another fingerprint is stale."""
    names = features.compute(matches[0][1], matches[0][1]["runs"][:1])[0]
    return {"features": names, "gbm": GBM, "matches": [name for name, _ in matches],
            "augment": augment.settings(params), "subsample": SUBSAMPLE}


def train_and_save(matches=None, params=None):
    matches = matches or training_matches()
    params = augment.regime(AUG_REGIME) if params is None else params
    model = fit(matches, params)
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump({"model": model, "fingerprint": fingerprint(matches, params)}, handle)
    return model


def load_or_train(params=None):
    """The saved model if it was trained exactly as the code now would, else a fresh one.
    Training is seeded, so a fresh clone with the same data rebuilds the same model."""
    matches = training_matches()
    params = augment.regime(AUG_REGIME) if params is None else params
    if os.path.isfile(MODEL_PATH):
        with open(MODEL_PATH, "rb") as handle:
            saved = pickle.load(handle)
        if isinstance(saved, dict) and saved.get("fingerprint") == fingerprint(matches, params):
            return saved["model"]
    return train_and_save(matches, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--learning-curve", action="store_true")
    args = parser.parse_args()
    matches = training_matches()
    print("training matches: {}".format(", ".join(name for name, _ in matches)))
    if args.evaluate:
        evaluate(matches)
    if args.learning_curve:
        learning_curve(matches)
    if args.train:
        train_and_save(matches)
        print("saved {}".format(os.path.relpath(MODEL_PATH, ROOT)))
