"""Calibrate the training augmentation to our match's ball gaps, from aggregates only.

    python -m solution.gap_stats     # prints the aggregates and the "calibrated" regime

solution/augment.py hides the ball across 95% of training stoppages and adds 5% live-play gaps.
How many of OUR long ball gaps are live play is unknown (no labels), but the players tell: they
move at stoppage pace in a dead gap and at live pace in a live one. So, with the training
matches' own dead / live player speeds as the two references:

    f              = share of our gap frames that are live, from the mean player speed in gaps
    dead           = (1 - f) * m_long / HIDE_DEAD_P      (m_long: share of graded frames in gaps)
    LIVE_GAP_SHARE = f * m_long / (1 - dead)             (augment counts it over live frames)
    LIVE_GAP_S     = p25-p75 length of the gaps that look live (default if fewer than 5)

Nothing of our match is labelled or printed frame by frame; only these aggregates.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import augment, dead_ball, dead_features  # noqa: E402

# Params
FPS = 25
MIN_FAST_GAPS = 5


def frame_speed(arrays, runs):
    """Mean player speed per frame inside the runs; NaN elsewhere."""
    out = np.full(len(arrays["ball_xy"]), np.nan)
    for start, stop in runs:
        spd = dead_features.player_speeds(arrays["players_xy"][start:stop].astype(np.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
            out[start:stop] = np.nanmean(spd, axis=1)
    return out


def calibrated(ours, our_runs, labelled, report=None):
    """{"LIVE_GAP_SHARE", "LIVE_GAP_S"} matching our ball gaps. `labelled`: training dicts with
    `dead` and `runs`. `report`, if a dict, receives the intermediate aggregates."""
    dead_v, live_v = [], []
    for arrays in labelled:
        v = frame_speed(arrays, arrays["runs"])
        inside = ~np.isnan(v)
        dead_v.append(v[inside & arrays["dead"]])
        live_v.append(v[inside & ~arrays["dead"]])
    v_dead, v_live = float(np.mean(np.concatenate(dead_v))), float(np.mean(np.concatenate(live_v)))

    v = frame_speed(ours, our_runs)
    missing = np.isnan(ours["ball_xy"][:, 0])
    gaps = [(a, b) for a, b in dead_ball.stretches(missing, our_runs)
            if b - a >= dead_ball.MIN_DEAD_S * FPS]
    graded = sum(int(b) - int(a) for a, b in our_runs)
    gap_frames = np.concatenate([np.arange(a, b) for a, b in gaps]) if gaps else np.array([], int)
    m_long = len(gap_frames) / graded
    v_gap = float(np.nanmean(v[gap_frames])) if len(gap_frames) else v_dead
    f = float(np.clip((v_gap - v_dead) / (v_live - v_dead), 0, 1))
    hide = augment.HIDE_DEAD_P
    dead = (1 - f) * m_long / hide
    share = f * m_long / (1 - dead)

    middle = (v_dead + v_live) / 2.0
    fast = [(b - a) / FPS for a, b in gaps if np.nanmean(v[a:b]) > middle]
    if len(fast) >= MIN_FAST_GAPS:
        lengths = (round(float(np.percentile(fast, 25)), 1), round(float(np.percentile(fast, 75)), 1))
    else:
        lengths = augment.LIVE_GAP_S
    if report is not None:
        report.update(v_dead=v_dead, v_live=v_live, v_gap=v_gap, m_long=m_long, f=f,
                      n_gaps=len(gaps), n_fast_gaps=len(fast))
    return {"LIVE_GAP_SHARE": round(share, 4), "LIVE_GAP_S": lengths}


def from_disk():
    """The calibrated regime for our match, printing the aggregates it came from."""
    from games import GAME
    from solution import model, tracking
    from solution.diagnostics import graded_runs
    labelled = [arrays for _, arrays in model.training_matches() if int(arrays["has_dead"])]
    report = {}
    out = calibrated(tracking.load(GAME), graded_runs(GAME), labelled, report)
    print("gap_stats: player speed dead {v_dead:.2f} / live {v_live:.2f} / in our gaps {v_gap:.2f} "
          "m/s; our gaps cover {m_long:.3f} of graded frames ({n_gaps} gaps, {n_fast_gaps} at live "
          "pace); live share of gap frames f = {f:.2f}".format(**report))
    print("gap_stats: calibrated regime {}".format(out))
    return out


if __name__ == "__main__":
    from_disk()
