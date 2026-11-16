"""Why does touch detection fire on the external matches but not the competition match?

    python -m solution.touch_diag        # run by a person: it reads the competition match

Prints aggregate statistics only -- percentiles, rates and shares over whole matches, never a frame,
label or coordinate -- side by side for the external matches (degraded as in training) and the
competition match, on live frames:

  ball step     how far the ball moves between frames; how often it does not move at all (a ball
                position updated at a lower rate, or held, shows up here)
  velocity jump |v_after - v_before| over lags of 3 and 8 frames, raw and smoothed over 5 frames:
                percentiles, and how often it exceeds 3 / 5 / 8 m/s (per minute of live play)
  near a player at the jumps: distance from the ball to the nearest player, and the share credited
                by the detector's rule (nearest within 2.5 m and 0.5 m clearer than the opponent)
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import dead_ball, tracking  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import model as M  # noqa: E402
from solution.diagnostics import graded_runs  # noqa: E402

FPS = 25
warnings.simplefilter("ignore", RuntimeWarning)


def smooth(ball, k):
    if k <= 1:
        return ball
    n = len(ball)
    cs = np.concatenate([np.zeros((1, 2)), np.cumsum(np.nan_to_num(ball), axis=0)])
    cn = np.concatenate([[0], np.cumsum(~np.isnan(ball[:, 0]))])
    lo = np.clip(np.arange(n) - k // 2, 0, n)
    hi = np.clip(np.arange(n) + k - k // 2, 0, n)
    return np.where(((cn[hi] - cn[lo]) == k)[:, None], (cs[hi] - cs[lo]) / k, np.nan)


def jump(ball, lag):
    before = np.full_like(ball, np.nan)
    after = np.full_like(ball, np.nan)
    before[lag:] = (ball[lag:] - ball[:-lag]) * FPS / lag
    after[:-lag] = (ball[lag:] - ball[:-lag]) * FPS / lag
    return np.linalg.norm(after - before, axis=1)


def stats(arrays, runs):
    ball = arrays["ball_xy"].astype(np.float64)
    players = arrays["players_xy"].astype(np.float64)
    team = arrays["slot_team"]
    live = np.zeros(len(ball), dtype=bool)
    for start, stop in runs:
        live[start:stop] = True
    live &= ~dead_ball.dead_mask(arrays, runs)
    seen = live & ~np.isnan(ball[:, 0])
    minutes = live.sum() / FPS / 60.0
    out = {"live minutes": minutes, "ball seen on live frames": seen.mean() / max(live.mean(), 1e-9)}

    step = np.concatenate([[np.nan], np.linalg.norm(np.diff(ball, axis=0), axis=1)])
    both = seen & ~np.isnan(step)
    out["ball step m p10/p50/p90"] = np.nanpercentile(step[both], [10, 50, 90])
    out["ball step exactly 0 (share)"] = (step[both] == 0).mean()
    out["ball step < 1 cm (share)"] = (step[both] < 0.01).mean()

    dist = np.sqrt(np.square(players - ball[:, None, :]).sum(-1))
    d0 = np.nanmin(np.where(np.isnan(dist[:, team == 0]), np.inf, dist[:, team == 0]), axis=1)
    d1 = np.nanmin(np.where(np.isnan(dist[:, team == 1]), np.inf, dist[:, team == 1]), axis=1)
    near = np.minimum(d0, d1)
    out["nearest player m p10/p50/p90"] = np.percentile(near[seen & np.isfinite(near)], [10, 50, 90])
    for k in (1, 5):
        for lag in (3, 8):
            dv = jump(smooth(ball, k), lag)
            ok = seen & ~np.isnan(dv)
            tag = "smooth{} lag{}".format(k, lag)
            out["jump m/s p50/p90/p99 " + tag] = np.percentile(dv[ok], [50, 90, 99])
            for thr in (3, 5, 8):
                out["jumps >= {} m/s per min {}".format(thr, tag)] = (ok & (dv >= thr)).sum() / minutes
            big = ok & (dv >= 5)
            credited = big & (((d0 <= 2.5) & (d0 + 0.5 <= d1)) | ((d1 <= 2.5) & (d1 + 0.5 <= d0)))
            out["share of >=5 jumps credited " + tag] = credited.sum() / max(big.sum(), 1)
            if big.any():
                out["nearest player at >=5 jumps p50 " + tag] = np.median(near[big & np.isfinite(near)])
    return out


def fmt(v):
    if isinstance(v, np.ndarray):
        return " / ".join("{:.2f}".format(x) for x in v)
    return "{:.3f}".format(v)


if __name__ == "__main__":
    ext = []
    for name in L.names():
        a = L.data()[name]
        ext.append(stats(L.degrade(name, ("default",), M.seed_of(name) + 1), a["runs"]))
    ours = stats(tracking.load(), graded_runs())
    print("{:<48} {:>26} {:>26}".format("", "external (mean of 10)", "competition match"))
    for key in ours:
        vals = [e[key] for e in ext if key in e]
        mean = np.mean(vals, axis=0) if vals else np.nan
        print("{:<48} {:>26} {:>26}".format(key, fmt(mean), fmt(ours[key])))
