"""Per-frame evidence that the ball is out of play, mostly from how the players move.

The training matches' ball gaps are synthetic (solution/augment.py), but their players are real,
so the strongest honest signal is the players: they slow down and stand still at a stoppage,
mass in a box for a corner, line up in their own halves for a kick-off. Ball context adds where
the ball was last seen (near a line?) and whether a visible ball is lying still. Everything is
team-agnostic: no team views, one row per frame, NaN outside the runs.
"""

import warnings

import numpy as np

from solution import dead_ball
from solution.features import FPS, SPEED_SMOOTH, VEL_LAG, _rolling_mean, _time_since_seen

# Params
STILL_MS = 1.0         # a player under this speed counts as standing
BALL_STILL_MS = 0.5    # a visible ball under this speed over BALL_STILL_S is lying still
BALL_STILL_S = 0.4
BOX_DEPTH_M, BOX_WIDTH_M = 16.5, 40.32  # penalty area

NAMES = ["spd_mean", "spd_p75", "spd_mean_past1", "spd_mean_next1", "spd_mean_past3",
         "spd_mean_next3", "spd_drop3", "still_share", "since_seen_s", "until_seen_s",
         "last_ball_touch_d", "last_ball_goal_d", "last_ball_speed", "ball_still",
         "box_count_max", "own_half_share"]


def compute(arrays, runs):
    """(names, matrix float32 [n_frames, 16]); rows outside runs are NaN."""
    out = np.full((len(arrays["ball_xy"]), len(NAMES)), np.nan, dtype=np.float32)
    for start, stop in runs:
        if stop > start:
            cols = _run(arrays, int(start), int(stop))
            out[start:stop] = np.stack([cols[name] for name in NAMES], axis=1)
    return list(NAMES), out


def player_speeds(xy):
    """Speed (m/s) of every player slot per frame of one run, over VEL_LAG frames; NaN at the
    run's first VEL_LAG frames and wherever a player is missing."""
    spd = np.full(xy.shape[:2], np.nan)
    if len(xy) > VEL_LAG:
        spd[VEL_LAG:] = np.linalg.norm(xy[VEL_LAG:] - xy[:-VEL_LAG], axis=-1) * FPS / VEL_LAG
    return spd


def _run(arrays, start, stop):
    length, width = (float(v) for v in arrays["pitch"])
    xy = arrays["players_xy"][start:stop].astype(np.float64)
    ball = arrays["ball_xy"][start:stop].astype(np.float64)
    m = len(ball)
    cols = {}

    spd = player_speeds(xy)
    tracked = (~np.isnan(spd)).sum(1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        cols["spd_mean"] = np.nanmean(spd, axis=1)
        cols["spd_p75"] = np.nanpercentile(spd, 75, axis=1)
        cols["still_share"] = np.where(tracked > 0, (spd < STILL_MS).sum(1) / np.maximum(tracked, 1),
                                       np.nan)
    for w in (1, 3):
        for direction, tag in ((-1, "past"), (1, "next")):
            cols["spd_mean_{}{}".format(tag, w)] = _rolling_mean(cols["spd_mean"], w * FPS, direction)
    cols["spd_drop3"] = cols["spd_mean_past3"] - cols["spd_mean_next3"]

    seen = ~np.isnan(ball[:, 0])
    cols["since_seen_s"] = _time_since_seen(seen) / FPS
    cols["until_seen_s"] = _time_since_seen(seen[::-1])[::-1] / FPS
    last = np.maximum.accumulate(np.where(seen, np.arange(m), -1))
    has_last = last >= 0
    last_xy = np.where(has_last[:, None], ball[np.maximum(last, 0)], np.nan)
    cols["last_ball_touch_d"] = np.minimum(last_xy[:, 1], width - last_xy[:, 1])
    cols["last_ball_goal_d"] = np.minimum(last_xy[:, 0], length - last_xy[:, 0])
    step = np.concatenate([[np.nan], np.linalg.norm(np.diff(ball, axis=0), axis=-1) * FPS])
    speed = _rolling_mean(step, SPEED_SMOOTH, -1)
    cols["last_ball_speed"] = np.where(has_last, speed[np.maximum(last, 0)], np.nan)
    slow = _rolling_mean(step, int(BALL_STILL_S * FPS), -1)
    with np.errstate(invalid="ignore"):
        cols["ball_still"] = (seen & (slow < BALL_STILL_MS)).astype(float)

    x, y = xy[..., 0], xy[..., 1]
    with np.errstate(invalid="ignore"):
        in_band = np.abs(y - width / 2.0) < BOX_WIDTH_M / 2.0
        left = (in_band & (x < BOX_DEPTH_M)).sum(1)
        right = (in_band & (x > length - BOX_DEPTH_M)).sum(1)
    cols["box_count_max"] = np.maximum(left, right).astype(float)
    cols["own_half_share"] = dead_ball.own_half_share(arrays, start, stop)
    return cols
