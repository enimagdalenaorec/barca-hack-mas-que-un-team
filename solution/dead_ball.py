"""Is the ball in play? One boolean per frame, before anyone asks which team has it.

The rule: **the ball is dead wherever the tracker has lost it for at least MIN_DEAD_S seconds.**

Why something this simple. On Metrica's three public sample games, where the events give the
true stoppages (solution/sample_games.py), the tracker's ball is missing on 86-99% of dead frames
and on only 2-6% of live ones, and every gap is almost entirely one or the other (466 of 476).
Short gaps are live play -- a ball hidden behind a player for a moment -- so a minimum length
separates the two:

    dead F1 on sample games 1 / 2 / 3       stoppages found (true: 94 / 94 / 99)
    any gap        0.895 / 0.939 / 0.969    153 / 166 / 157
    gap >= 1 s     0.897 / 0.942 / 0.971     98 / 105 / 104
    gap >= 2 s     0.898 / 0.943 / 0.972     93 /  97 /  98    <- chosen
    gap >= 3 s     0.898 / 0.944 / 0.970     88 /  94 /  96

Tried and dropped, also on the sample games: adding frames where the ball is seen outside the
lines (dead F1 down on two games of three, ~15 extra fake stoppages each).

On the competition match the same evidence points the same way, without any ground truth: the
ball is missing in 150 long blocks rather than in flickers, players move at half speed during
them (1.2 vs 2.2 m/s), and 45 of 59 Smart Tagging set pieces are taken just as a gap ends.

What this does not catch, and could be the next improvement: the dead frames where the ball IS
visible (a ball placed for a free kick and tracked), and long live-play gaps.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Params
FPS = 25
MIN_DEAD_S = 2.0  # chosen on the sample games, see the table above


def stretches(mask, runs):
    """[(start, stop)] maximal True spans of mask, never crossing the edge of a run."""
    out = []
    for a, b in runs:
        edges = np.diff(np.concatenate(([0], mask[a:b].astype(np.int8), [0])))
        out += [(a + int(s), a + int(t))
                for s, t in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    return out


def dead_mask(arrays, runs, min_dead_s=MIN_DEAD_S):
    """True where the ball is out of play, inside the runs. Frames outside the runs are False."""
    missing = np.isnan(arrays["ball_xy"][:, 0])
    dead = np.zeros(len(missing), dtype=bool)
    for start, stop in stretches(missing, runs):
        if stop - start >= min_dead_s * FPS:
            dead[start:stop] = True
    return dead
