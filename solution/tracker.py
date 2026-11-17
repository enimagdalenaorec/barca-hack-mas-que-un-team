"""Which kind of tracker made the ball path: causal (lagging) or not. Decides the configuration.

    python -m solution.tracker            # the external matches, three ball kinds (no match data)

The competition tracker smooths its ball. If it smooths causally (a trailing filter, as a live
tracker does), the ball sits where its player WAS a few frames ago; the models must then be
trained on a ball smoothed the same way (augment.TRACKER_SMOOTH_W), with the touch features and
per-possession weights that only pay off on that ball ("trailing", branch eni's Version D). If it
does not lag, the configuration trained on the clean ball is better ("clean", branch karlo-v2).
Leave one game out on the 10 external matches (solution/robust_exp.py), spellF1:

    held-out ball           clean config   trailing config
    clean                   0.664          0.605
    trailing mean, w 21     0.515          0.649
    Gaussian, sigma 6       0.637          0.604

So the pipeline measures the lag on the tracking it is given: for each shift `lag`, the median
distance from the ball at frame t to the nearest player position at frame t - lag; the shift with
the smallest median is the lag. On the 10 external matches (python -m solution.tracker) the best
lag is -6..0 frames for the clean ball and -6..-2 for the Gaussian-smoothed one, against +4..+8 for
the trailing mean. LAG_THRESHOLD sits between the two groups. No label is read, nothing is tied
to a frame: it is one whole-match statistic of the input, computed by the same code on every
match, and the threshold was fixed on external matches only.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LAGS = tuple(range(-20, 21, 2))   # frames
LAG_THRESHOLD = 2                 # best lag >= this: causal tracker -> "trailing"
CONFIGS = {
    # DELAGS / DEAD_DELAGS: de-lagged views of the ball (model.DELAGS); the decoder costs that go
    # with them (decode.py, "Krug 6"). "clean" keeps one view and the costs chosen on clean balls.
    "trailing": {"TRACKER_SMOOTH_W": 21, "TOUCH_FEATURES": True, "SPELL_WEIGHTS": True,
                 "DELAGS": (0, 10, 15), "DEAD_DELAGS": (0, 10),
                 "SWITCH_COST": 12.0, "FLIP_FACTOR": 0.8, "DEAD_TEMPERATURE": 3.0},
    # "clean": karlo-v2 plus the per-possession weights of round 5. Round 7 (solution/clean_exp.py,
    # docs/logo_external.md "Krug 7"): clean held-out ball 0.6641 -> 0.6769 (8/10, harsh +0.017,
    # lag +0.014), Gaussian-smoothed ball 0.6367 -> 0.6487 (7/10).
    "clean": {"TRACKER_SMOOTH_W": 1, "TOUCH_FEATURES": False, "SPELL_WEIGHTS": True,
              "DELAGS": (0,), "DEAD_DELAGS": (0,),
              "SWITCH_COST": 20.0, "FLIP_FACTOR": 0.4, "DEAD_TEMPERATURE": 2.0},
}


def lag_curve(arrays, runs):
    """Median distance from the ball at frame t to the nearest player at frame t - lag, per lag,
    on frames inside the runs with a ball. One aggregate curve per match."""
    ball = arrays["ball_xy"].astype(np.float64)
    players = arrays["players_xy"].astype(np.float64)
    inside = np.zeros(len(ball), dtype=bool)
    for start, stop in runs:
        inside[start:stop] = True
    out = []
    for lag in LAGS:
        shifted = np.full_like(players, np.nan)
        if lag > 0:
            shifted[lag:] = players[:-lag]
        elif lag < 0:
            shifted[:lag] = players[-lag:]
        else:
            shifted = players
        d = np.sqrt(np.square(shifted - ball[:, None, :]).sum(-1))
        near = np.where(np.isnan(d), np.inf, d).min(axis=1)
        out.append(np.median(near[inside & np.isfinite(near)]))
    return np.array(out)


def best_lag(arrays, runs):
    return LAGS[int(np.argmin(lag_curve(arrays, runs)))]


def detect(arrays, runs):
    """"trailing" or "clean" for the tracking in `arrays`."""
    return "trailing" if best_lag(arrays, runs) >= LAG_THRESHOLD else "clean"


def apply(kind):
    """Set the module constants of configuration `kind` (before any model is loaded)."""
    from solution import augment, decode, features, model
    settings = CONFIGS[kind]
    augment.TRACKER_SMOOTH_W = settings["TRACKER_SMOOTH_W"]
    features.TOUCH_FEATURES = settings["TOUCH_FEATURES"]
    model.SPELL_WEIGHTS = settings["SPELL_WEIGHTS"]
    model.DELAGS, model.DEAD_DELAGS = settings["DELAGS"], settings["DEAD_DELAGS"]
    decode.SWITCH_COST = settings["SWITCH_COST"]
    decode.FLIP_FACTOR = settings["FLIP_FACTOR"]
    decode.DEAD_TEMPERATURE = settings["DEAD_TEMPERATURE"]
    return kind


if __name__ == "__main__":
    from solution import logo as L
    from solution import robust_exp as R
    print("| ball | best lag per match | detected |\n|---|---|---|")
    for kind in R.BALLS:
        lags = [best_lag(R.held_arrays(n, "default", kind, 1), L.data()[n]["runs"]) for n in L.names()]
        print("| {} | {} | {} |".format(kind, " ".join("{:+d}".format(x) for x in lags),
                                        ", ".join(sorted({("trailing" if x >= LAG_THRESHOLD else "clean")
                                                          for x in lags}))))
