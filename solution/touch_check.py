"""Is the competition tracking clean enough for touch detection? Run by a person, not an agent.

    python -m solution.touch_check          # the detector as configured in features.py
    python -m solution.touch_check heavy    # the 1 s smoothed detector of solution/touch_exp.py

Prints aggregate numbers only (touches per minute of live play, and the same on the external
matches for comparison) -- no frame, label or coordinate. On the external matches the detector
finds roughly 25-30 touches per minute of live play; a much higher rate on the competition match
would mean tracking jitter is faking touches.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import dead_ball, features, tracking  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import model as M  # noqa: E402
from solution.diagnostics import graded_runs  # noqa: E402

FPS = 25


def touches_per_minute(arrays, runs, live):
    features.TOUCH_FEATURES = True
    names, X = features.compute(arrays, runs)
    features.TOUCH_FEATURES = False
    n = sum(int((X[live, names.index(c)] == 0).sum()) for c in ("touch_since_us", "touch_since_them"))
    return n / (live.sum() / FPS / 60.0)


if __name__ == "__main__":
    if "heavy" in sys.argv[1:]:   # the heavily smoothed detector of solution/touch_exp.py
        from solution import touch_exp
        for key, value in touch_exp.HEAVY.items():
            setattr(features, key, value)
        print("detector: heavy smoothing {}".format(touch_exp.HEAVY))
    rates = []
    for name in L.names():
        a = L.data()[name]
        held = L.degrade(name, ("default",), M.seed_of(name) + 1)
        live = L.inside_mask(a) & ~dead_ball.dead_mask(held, a["runs"])
        rates.append(touches_per_minute(held, a["runs"], live))
    print("external matches: {:.1f} touches per minute of live play (min {:.1f}, max {:.1f})".format(
        np.mean(rates), np.min(rates), np.max(rates)))
    d, runs = tracking.load(), graded_runs()
    live = np.zeros(len(d["ball_xy"]), dtype=bool)
    for start, stop in runs:
        live[start:stop] = True
    live &= ~dead_ball.dead_mask(d, runs)
    print("competition match: {:.1f} touches per minute of live play".format(
        touches_per_minute(d, runs, live)))
