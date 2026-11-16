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

Second rule: **nothing is live before a half's kick-off.** The competition grades the second half
from frame 110,808 but it kicks off at 113,742 (FAQ.md); the ball is tracked while it is placed on
the centre spot, so the first rule alone called 82% of those frames live. The kick-off is found in
the tracking, not copied from the docs, so the rule carries to any match: before a kick-off every
player stands in his own half; the kick-off is the moment that formation breaks -- the first frame,
after the set formation has held for KICKOFF_HOLD_S, where the share of players in their own half
(over one second) drops below KICKOFF_SHARE, with the ball inside the centre circle. Searched only
in the first KICKOFF_SEARCH_S of a half. A share rather than "everyone" because the tracker
occasionally puts a slot in the wrong place. And only when the players were calm before it: a
later kick-off (after an early goal) is a real kick-off too, but the play before it was live.
On the training halves it changes nothing except before one retaken kick-off.

What this does not catch: the dead frames where the ball IS visible mid-half (a ball placed for a
free kick and tracked), and long live-play gaps.
"""

import os
import warnings
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Params
FPS = 25
MIN_DEAD_S = 2.0  # chosen on the sample games, see the table above
KICKOFF_SHARE = 0.85    # share of tracked players in their own half that counts as "set"
KICKOFF_HOLD_S = 3.0    # the set formation must hold this long before the kick-off
KICKOFF_SEARCH_S = 300  # only the first five minutes of a half are searched
OWN_HALF_TOL_M = 1.0    # kickers stand on the line; allow a metre
KICKOFF_BALL_M = 9.15   # ...and the ball is inside the centre circle when the formation breaks
KICKOFF_BALL_S = 2.0    # looked for within this many seconds either side
# Before a half's first kick-off the players walk out and line up; nobody sprints. A Metrica
# kick-off that followed minutes of real play (an early goal) is well above this, so the rule
# stays out of it.
KICKOFF_CALM_MS = 2.5


def stretches(mask, runs):
    """[(start, stop)] maximal True spans of mask, never crossing the edge of a run."""
    out = []
    for a, b in runs:
        edges = np.diff(np.concatenate(([0], mask[a:b].astype(np.int8), [0])))
        out += [(a + int(s), a + int(t))
                for s, t in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    return out


def own_half_share(arrays, start, stop):
    """Per frame of [start, stop): share of tracked players standing in their own half."""
    xy, team = arrays["players_xy"][start:stop], arrays["slot_team"]
    middle = float(arrays["pitch"][0]) / 2.0
    x0, x1 = xy[:, team == 0, 0], xy[:, team == 1, 0]
    team0_left = np.nanmean(x0) < np.nanmean(x1)
    with np.errstate(invalid="ignore"):
        own0 = (x0 < middle + OWN_HALF_TOL_M) if team0_left else (x0 > middle - OWN_HALF_TOL_M)
        own1 = (x1 > middle - OWN_HALF_TOL_M) if team0_left else (x1 < middle + OWN_HALF_TOL_M)
    tracked = (~np.isnan(x0)).sum(1) + (~np.isnan(x1)).sum(1)
    inside = (own0 & ~np.isnan(x0)).sum(1) + (own1 & ~np.isnan(x1)).sum(1)
    return np.where(tracked > 0, inside / np.maximum(tracked, 1), np.nan)


def kickoff_frame(arrays, start, stop):
    """First live frame of a half, or `start` when no set formation is found before play."""
    stop = min(stop, start + int(KICKOFF_SEARCH_S * FPS))
    share = own_half_share(arrays, start, stop)
    valid = ~np.isnan(share)
    cs = np.concatenate(([0.0], np.cumsum(np.where(valid, share, 0.0))))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(share))
    hi = np.minimum(len(share), idx + FPS)
    smooth = np.where(cn[hi] - cn[idx] > 0, (cs[hi] - cs[idx]) / np.maximum(cn[hi] - cn[idx], 1),
                      np.nan)
    hold = int(KICKOFF_HOLD_S * FPS)
    centre = arrays["pitch"] / 2.0
    near = int(KICKOFF_BALL_S * FPS)
    held = 0
    for i, value in enumerate(smooth):
        if value >= KICKOFF_SHARE:
            held += 1
            continue
        if held >= hold:
            ball = arrays["ball_xy"][max(start, start + i - near):start + i + near]
            dist = np.linalg.norm(ball - centre, axis=1)
            if np.any(dist < KICKOFF_BALL_M) and calm(arrays, start, start + i):
                return start + i
        held = 0
    return start


def calm(arrays, start, stop):
    """No real play in [start, stop): the busiest moments are still a walk or a jog."""
    xy = arrays["players_xy"][start:stop]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        per_frame = np.nanmean(np.linalg.norm(np.diff(xy, axis=0), axis=-1) * FPS, axis=1)
    return bool(np.nanpercentile(per_frame, 90) < KICKOFF_CALM_MS)


def dead_mask(arrays, runs, min_dead_s=MIN_DEAD_S):
    """True where the ball is out of play, inside the runs. Frames outside the runs are False."""
    missing = np.isnan(arrays["ball_xy"][:, 0])
    dead = np.zeros(len(missing), dtype=bool)
    for start, stop in stretches(missing, runs):
        if stop - start >= min_dead_s * FPS:
            dead[start:stop] = True
    return dead | pre_kickoff_mask(arrays, runs)


def pre_kickoff_mask(arrays, runs):
    """True on [start, kick-off) of each run: nothing is live before a half's kick-off."""
    mask = np.zeros(len(arrays["ball_xy"]), dtype=bool)
    for start, stop in runs:
        mask[start:kickoff_frame(arrays, start, stop)] = True
    return mask
