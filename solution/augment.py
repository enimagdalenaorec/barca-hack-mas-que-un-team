"""Make clean training tracking look like our automatic tracking before features are computed.

A model trained on perfect data learns to rely on things automatic tracking does not have. What
the organisers document about the tactical ATD (DATA.md): the ball is missing on 41% of graded
frames, and ~20.5 of 22 players are tracked on an average graded frame. Automatic trackers also
lose the ball across most stoppages and bridge short losses themselves, as Metrica's own sample
tracking does (solution/sample_games.py).

So, per training match, with a fixed seed so the training set is reproducible:

  1. short ball gaps (< BRIDGE_S) are bridged by linear interpolation, as our tracker appears to;
  2. on sources with a dead-ball label, the ball is hidden across each stoppage with probability
     HIDE_DEAD_P, the way ours vanishes when play stops;
  3. extra live-play ball gaps are cut until LIVE_GAP_SHARE of live frames have no ball, with
     lengths drawn from LIVE_GAP_S -- our long, rare live losses;
  4. each player is dropped for random blocks covering PLAYER_DROP_SHARE of frames;
  5. the ball is smoothed the way the competition tracker smooths it: a trailing average over
     TRACKER_SMOOTH_W frames. Aggregate statistics of the competition ball (python -m
     solution.touch_diag, run by a person) show its velocity changes are a fifth of the external
     ones (99th percentile of the 3-frame change 3.8 against 16.8 m/s) and that it sits further
     from its nearest player (median 1.88 against 1.25 m). 21 frames bring the external balls to
     ~3.8 m/s and ~1.6 m. One number for the whole match, from aggregates, no frame-level
     information; agreed by the team on 2026-11-17. Leave one game out on held-out matches
     with that ball (python -m solution.smooth_train_exp): spellF1 0.515 -> 0.627 with touch
     features, 10 of 10 matches. On clean held-out balls it is the other way round (0.664 ->
     0.584): the model is now trained for the competition ball.

Not added by default: per-frame position noise. Independent jitter on every frame turns into
fake ball speed, because ball speed is computed from positions.

Player identity is never used by the features (everything is per team), so breaking players into
tracklets the way our 46 slots do would change nothing and is not simulated.
"""

import numpy as np

from solution.dead_ball import stretches

# Params
FPS = 25
BRIDGE_S = 1.0
HIDE_DEAD_P = 0.95
LIVE_GAP_SHARE = 0.05
LIVE_GAP_S = (2.0, 15.0)
PLAYER_DROP_SHARE = 0.03
PLAYER_DROP_S = (1.0, 10.0)
TRACKER_SMOOTH_W = 21  # frames of trailing average on the ball; 1 = off (see step 5 above)
PARAMS = ("BRIDGE_S", "HIDE_DEAD_P", "LIVE_GAP_SHARE", "LIVE_GAP_S", "PLAYER_DROP_SHARE",
          "PLAYER_DROP_S", "TRACKER_SMOOTH_W")
# Named overrides of the settings above, for training and for degrading held-out matches.
# "harsh": more and longer live-play ball gaps, fewer hidden stoppages.
REGIMES = {"default": {},
           "harsh": {"LIVE_GAP_SHARE": 0.15, "LIVE_GAP_S": (3.0, 30.0), "HIDE_DEAD_P": 0.85}}


def regime(name):
    """The overrides of a named regime, as a fresh dict."""
    return dict(REGIMES[name])


def settings(params=None):
    """Every setting augment() uses: the module defaults, overridden by params."""
    out = {name: globals()[name] for name in PARAMS}
    unknown = set(params or {}) - set(PARAMS)
    if unknown:
        raise KeyError("unknown augment settings {}".format(sorted(unknown)))
    out.update(params or {})
    return out


def _bridge(xy, runs, max_frames):
    """Linear interpolation across NaN gaps up to max_frames long, inside each run."""
    out = xy.copy()
    missing = np.isnan(xy[:, 0])
    for start, stop in stretches(missing, runs):
        if stop - start > max_frames or start == 0 or stop >= len(xy):
            continue
        if np.isnan(xy[start - 1, 0]) or np.isnan(xy[stop, 0]):
            continue
        w = (np.arange(start, stop) - (start - 1)) / float(stop - start + 1)
        out[start:stop] = xy[start - 1] * (1 - w[:, None]) + xy[stop] * w[:, None]
    return out


def augment(arrays, seed, params=None):
    """Augmented copy of arrays; params overrides the module settings for this call only."""
    p = settings(params)
    rng = np.random.default_rng(seed)
    runs = arrays["runs"]
    out = dict(arrays)
    ball = _bridge(arrays["ball_xy"].astype(np.float64), runs, int(p["BRIDGE_S"] * FPS))
    players = arrays["players_xy"].astype(np.float64).copy()

    if int(arrays.get("has_dead", 0)):
        for start, stop in stretches(arrays["dead"], runs):
            if rng.random() < p["HIDE_DEAD_P"]:
                ball[start:stop] = np.nan

    live = np.zeros(len(ball), dtype=bool)
    for start, stop in runs:
        live[start:stop] = True
    if int(arrays.get("has_dead", 0)):
        live &= ~arrays["dead"]
    live_idx = np.flatnonzero(live)
    budget = int(p["LIVE_GAP_SHARE"] * len(live_idx)) - int(np.isnan(ball[live_idx, 0]).sum())
    while budget > 0 and len(live_idx):
        length = int(rng.uniform(*p["LIVE_GAP_S"]) * FPS)
        start = int(rng.choice(live_idx))
        ball[start:start + length] = np.nan
        budget -= length

    n, slots = players.shape[:2]
    for slot in range(slots):
        budget = int(p["PLAYER_DROP_SHARE"] * n)
        while budget > 0:
            length = int(rng.uniform(*p["PLAYER_DROP_S"]) * FPS)
            start = int(rng.integers(0, n))
            players[start:start + length, slot] = np.nan
            budget -= length

    ball = ball.astype(np.float32).astype(np.float64)
    if p["TRACKER_SMOOTH_W"] > 1:
        ball = trailing_mean(ball, int(p["TRACKER_SMOOTH_W"]))
    out["ball_xy"] = ball.astype(np.float32)
    out["players_xy"] = players.astype(np.float32)
    return out


def trailing_mean(ball, w):
    """Mean of the ball over the last w frames; missing unless all w frames (and this one) are
    seen, so a gap never smears into the path."""
    n = len(ball)
    cs = np.concatenate([np.zeros((1, 2)), np.cumsum(np.nan_to_num(ball), axis=0)])
    cn = np.concatenate([[0], np.cumsum(~np.isnan(ball[:, 0]))])
    hi = np.arange(1, n + 1)
    lo = np.maximum(0, hi - w)
    full = ((cn[hi] - cn[lo]) == hi - lo) & ~np.isnan(ball[:, 0])
    return np.where(full[:, None], (cs[hi] - cs[lo]) / (hi - lo)[:, None], np.nan)
