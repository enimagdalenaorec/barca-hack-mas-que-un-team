"""Tracking in, submission out. The one command that regenerates our file.

    python -m solution.pipeline                     # -> evaluation/results/model_viterbi.csv
    python -m solution.pipeline --method rule       # -> evaluation/results/deadball_smooth50.csv

Two decisions, made in this order:

  1. Alive or dead, per frame -- solution/dead_ball.py.
  2. Which team, on alive frames only, never looking across a stoppage:
       model  (default) the trained model's P(home) per frame (solution/model.py), smoothed into
              possessions by Viterbi with a switching cost (solution/decode.py). Needs the
              training matches: python -m solution.sample_games, then
              python -m solution.download_external. Trains on first use, then reuses
              data/cache/model.pkl.
       rule   the published baseline's rule -- the team of the tracked player nearest the ball
              -- with a +/-50-frame majority vote. Kept for comparison.

Why the order matters. The published baseline smooths everything at once, so near the edge of a
stoppage the visible-ball frames outvote the dead ones: every stoppage shrinks by the window at
both ends and any stoppage shorter than twice the window disappears. Deciding dead first, then
smoothing inside each live stretch, keeps stoppages exactly where the tracking puts them.

Frames outside the graded halves are written X.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME  # noqa: E402
from solution import dead_ball, tracking  # noqa: E402
from solution.diagnostics import graded_runs  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
WINDOW = 50  # +/- frames; the published floor's value, kept so the comparison changes one thing
HOME, AWAY, NOBODY = 0, 1, -1
DEAD = "rule"  # dead-ball method of the model path: "rule" or "model" (solution/experiments.py, dead)


def nearest_team(arrays):
    """Team code of the tracked player nearest the ball, NOBODY where there is no ball.

    The same rule as starter/baseline.py, vectorised; it reproduces baseline_raw exactly.
    """
    offset = arrays["players_xy"] - arrays["ball_xy"][:, None, :]
    distance = np.where(np.isnan(offset[..., 0]), np.inf, np.square(offset).sum(axis=-1))
    nearest = np.argmin(distance, axis=1)
    found = np.isfinite(distance[np.arange(len(nearest)), nearest])
    return np.where(found, arrays["slot_team"][nearest], NOBODY).astype(np.int8)


def vote(codes, start, stop, window):
    """Majority team over +/- window inside [start, stop) only. NOBODY where no frame votes.

    Cumulative sums, as in baseline.py, so the window costs nothing; ties go to home, likewise.
    """
    segment = codes[start:stop]
    home = np.concatenate(([0], np.cumsum(segment == HOME)))
    away = np.concatenate(([0], np.cumsum(segment == AWAY)))
    index = np.arange(len(segment))
    low = np.maximum(0, index - window)
    high = np.minimum(len(segment), index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    return np.where(h + a == 0, NOBODY, np.where(h >= a, HOME, AWAY)).astype(np.int8)


def build(arrays, runs, window=WINDOW):
    """Label array 'H'/'A'/'D'/'X' for every frame."""
    dead = dead_ball.dead_mask(arrays, runs)
    codes = nearest_team(arrays)
    labels = np.full(len(codes), "X", dtype="<U1")
    for start, stop in runs:
        labels[start:stop][dead[start:stop]] = "D"
    live = np.zeros(len(codes), dtype=bool)
    for start, stop in runs:
        live[start:stop] = ~dead[start:stop]
    for start, stop in dead_ball.stretches(live, runs):
        team = vote(codes, start, stop, window)
        labels[start:stop] = np.where(team == HOME, "H", np.where(team == AWAY, "A", "X"))
    return labels


def build_model(arrays, runs, dead=DEAD):
    """Label array from the trained model + Viterbi. Imported lazily: needs scikit-learn.

    dead="rule": dead_ball.dead_mask, then a two-state Viterbi on each live stretch.
    dead="model": the learned P(dead) (solution/dead_model.py) and one H/A/D Viterbi per half.
    """
    from solution import decode, model
    import possession_metrics as pm
    p = model.p_home(model.load_or_train(), arrays, runs)
    if dead == "model":
        from solution import dead_model
        p_dead = dead_model.p_dead(dead_model.load_or_train(), arrays, runs)
        codes = model.labels_from3(p, p_dead, arrays, runs,
                                   (decode.SWITCH_COST, decode.STOP_COST, decode.RESTART_COST))
    else:
        codes = model.labels_from(p, arrays, runs, decode.SWITCH_COST)
    return np.array(pm.LABELS, dtype="<U1")[codes]


def write(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=("model", "rule"), default="model")
    parser.add_argument("--window", type=int, default=WINDOW, help="rule method only")
    parser.add_argument("--dead", choices=("rule", "model"), default=DEAD,
                        help="model method only: dead-ball rule or learned P(dead)")
    parser.add_argument("--name", default=None, help="file name on the board, without .csv")
    args = parser.parse_args()

    arrays, runs = tracking.load(GAME), graded_runs(GAME)
    if args.method == "model":
        labels = build_model(arrays, runs, args.dead)
        name = args.name or "model_viterbi"
    else:
        labels = build(arrays, runs, args.window)
        name = args.name or "deadball_smooth{}".format(args.window)
    path = write(os.path.join(RESULTS_DIR, name + ".csv"), labels)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
