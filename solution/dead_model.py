"""P(ball out of play) per frame, learned from the labelled training matches.

    python -m solution.dead_model --train    # fit on every labelled match, save data/cache/dead_model.pkl

The alternative to solution/dead_ball.py's "ball missing for 2 s" rule. Same recipe as the team
model (solution/model.py): gradient-boosted trees, the same GBM settings, every graded frame of
every match with a dead-ball label, augmented with the same seed so the ball gaps look like ours,
subsampled. Features: solution/dead_features.py. Decoded together with P(home) by
decode.viterbi3 (model.labels_from3).
"""

import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402
from solution import augment, dead_features, model  # noqa: E402

# Params
MODEL_PATH = os.path.join(DATA_DIR, "cache", "dead_model.pkl")

_ROWS = {}


def rows(name, arrays, params=None):
    """(X, y) of one labelled match: augmented, every graded frame, subsampled. Memoised."""
    key = (name, model.params_key(params))
    if key not in _ROWS:
        aug = augment.augment(arrays, model.seed_of(name), params)
        inside = np.zeros(len(arrays["dead"]), dtype=bool)
        for start, stop in arrays["runs"]:
            inside[start:stop] = True
        idx = np.flatnonzero(inside)[::model.SUBSAMPLE]
        _, X = dead_features.compute(aug, arrays["runs"])
        _ROWS[key] = X[idx], arrays["dead"][idx].astype(np.int8)
    return _ROWS[key]


def fit(matches, params=None):
    parts = [rows(name, arrays, params) for name, arrays in matches if int(arrays["has_dead"])]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return HistGradientBoostingClassifier(**model.GBM).fit(X, y)


def p_dead(clf, arrays, runs):
    """P(dead) per frame; NaN outside the runs."""
    _, X = dead_features.compute(arrays, runs)
    p = np.full(len(X), np.nan)
    inside = ~np.isnan(X).all(axis=1)
    p[inside] = clf.predict_proba(X[inside])[:, 1]
    return p


def fingerprint(matches, params):
    """Everything the saved model depends on. A saved model with another fingerprint is stale."""
    return {"features": dead_features.NAMES, "gbm": model.GBM, "subsample": model.SUBSAMPLE,
            "matches": [name for name, arrays in matches if int(arrays["has_dead"])],
            "augment": augment.settings(params)}


def load_or_train(params=None):
    """The saved model if it was trained exactly as the code now would, else a fresh one."""
    matches = model.training_matches()
    params = augment.regime(model.AUG_REGIME) if params is None else params
    if os.path.isfile(MODEL_PATH):
        with open(MODEL_PATH, "rb") as handle:
            saved = pickle.load(handle)
        if isinstance(saved, dict) and saved.get("fingerprint") == fingerprint(matches, params):
            return saved["model"]
    clf = fit(matches, params)
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump({"model": clf, "fingerprint": fingerprint(matches, params)}, handle)
    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", action="store_true")
    if parser.parse_args().train:
        load_or_train()
        print("saved {}".format(os.path.relpath(MODEL_PATH, ROOT)))
