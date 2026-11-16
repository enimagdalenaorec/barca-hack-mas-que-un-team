"""Turn per-frame P(home has the ball) into clean possessions: Viterbi with a switching cost.

Inside one live stretch there are two states, home and away. Each frame scores
log P(state) from the model; changing state costs `switch_cost` (in the same log units). The
Viterbi algorithm finds the sequence with the best total score, so the team only changes when the
evidence for the other side outweighs the cost -- a one-frame blip never pays for a switch, while
a clear, sustained change of control does.

This is a two-state hidden Markov model with a symmetric transition matrix, and `switch_cost` is
-log of the switch probability per frame. It is the one setting of this step, tuned on the
training matches with the official spellF1 (solution/model.py), never on our match.
"""

import numpy as np

# Params
EPS = 1e-4  # keeps log() finite for a model that is (over)confident
# Chosen on the 10 external matches only (3 Metrica + 7 IDSSE), leave one game out
# (python -m solution.logo switch; docs/logo_external.md, e): 14 has the best held-out mean over
# the grid 6-60, and a nested choice (pick on 9 matches, score the 10th) picks 14 in 10 of 10 folds.
# With the flip rule below, the joint nested choice moves it to 20 ("Krug 2").
SWITCH_COST = 20.0
# Three-state decoder (viterbi3, the default dead-ball path of the submission): leaving play
# for a stoppage, and coming back. Leave one game out on the 10 external matches
# (python -m solution.logo dead; docs/logo_external.md, b): the stop x restart grid is flat
# (within 0.002), and a nested choice (pick on 9, score the 10th) scores the same as 20 / 8.
STOP_COST = 20.0
RESTART_COST = 8.0
# Evidence-dependent switch cost (None = off). On by default since "Krug 2": with the flip
# rule, SWITCH_COST 20, P(dead) on all 7 IDSSE matches and DEAD_TEMPERATURE 2 the nested
# held-out spellF1 rises +0.0097, 10/10 matches, and holds under the harsh and lag regimes. When set, switching team costs
# SWITCH_COST * FLIP_FACTOR on the frames within FLIP_WINDOW of a "flip" -- the team nearest the
# ball changes and the new nearest player is within FLIP_RADIUS_M -- and SWITCH_COST elsewhere, so
# a short possession that starts and ends on such frames no longer has to outweigh two full
# costs. Leave one game out on the 10 external matches, two held-out seeds, three ball-loss
# regimes (python -m solution.logo2 flip final; docs/logo_external.md, "Krug 2").
FLIP_FACTOR = 0.4
FLIP_RADIUS_M = 2.0
FLIP_WINDOW = 10  # frames either side of the flip
# Temperature on P(dead) before decoding: sigmoid(logit(p) / T). None = off. Same study.
DEAD_TEMPERATURE = 2.0


def viterbi(p_home, switch_cost):
    """0 = home / 1 = away for each frame of one live stretch. NaN probabilities count as 0.5.

    switch_cost is a number, or one number per frame (the cost of switching INTO that frame).
    Plain Python floats in the loop: with two states, numpy's per-call overhead would dominate.
    """
    p = np.clip(np.nan_to_num(p_home, nan=0.5), EPS, 1 - EPS)
    log_h, log_a = np.log(p).tolist(), np.log(1 - p).tolist()
    costs = (np.broadcast_to(np.asarray(switch_cost, dtype=float), p.shape)).tolist()
    n = len(log_h)
    back = bytearray(2 * n)  # back[2t + s] = previous state on the best path into state s at t
    h, a = log_h[0], log_a[0]
    for t in range(1, n):
        c = costs[t]
        if h >= a - c:
            nh, bh = h, 0
        else:
            nh, bh = a - c, 1
        if a >= h - c:
            na, ba = a, 1
        else:
            na, ba = h - c, 0
        h, a = nh + log_h[t], na + log_a[t]
        back[2 * t], back[2 * t + 1] = bh, ba
    path = np.empty(n, dtype=np.int8)
    state = 0 if h >= a else 1
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[2 * t + state]
    return path


def viterbi3(p_dead, p_home, forced_dead, switch_cost, stop_cost, restart_cost):
    """0 = home / 1 = away / 2 = dead for each frame of one run. NaN probabilities count as 0.5.

    Emissions: dead log p_dead; home log(1 - p_dead) + log p_home; away log(1 - p_dead) +
    log(1 - p_home). Frames in forced_dead can only be dead. Costs: home <-> away switch_cost,
    home/away -> dead stop_cost, dead -> home/away restart_cost; staying is free.
    switch_cost is a number, or one number per frame (the cost of switching INTO that frame).
    """
    n = len(p_dead)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    if np.ndim(switch_cost):
        return _viterbi3_framewise(p_dead, p_home, forced_dead, switch_cost, stop_cost,
                                   restart_cost)
    pd = np.clip(np.nan_to_num(np.asarray(p_dead, dtype=float), nan=0.5), EPS, 1 - EPS)
    ph = np.clip(np.nan_to_num(np.asarray(p_home, dtype=float), nan=0.5), EPS, 1 - EPS)
    live = np.where(np.asarray(forced_dead, dtype=bool), -np.inf, np.log(1 - pd))
    e_h, e_a = (live + np.log(ph)).tolist(), (live + np.log(1 - ph)).tolist()
    e_d = np.log(pd).tolist()
    sw, st, rs = float(switch_cost), float(stop_cost), float(restart_cost)
    back = bytearray(3 * n)  # back[3t + s] = previous state on the best path into state s at t
    h, a, d = e_h[0], e_a[0], e_d[0]
    for t in range(1, n):
        nh, bh = h, 0
        if a - sw > nh:
            nh, bh = a - sw, 1
        if d - rs > nh:
            nh, bh = d - rs, 2
        na, ba = a, 1
        if h - sw > na:
            na, ba = h - sw, 0
        if d - rs > na:
            na, ba = d - rs, 2
        nd, bd = d, 2
        if h - st > nd:
            nd, bd = h - st, 0
        if a - st > nd:
            nd, bd = a - st, 1
        h, a, d = nh + e_h[t], na + e_a[t], nd + e_d[t]
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = bh, ba, bd
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax([h, a, d]))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path


def _viterbi3_framewise(p_dead, p_home, forced_dead, switch_cost, stop_cost, restart_cost):
    """viterbi3 with one switch cost per frame; the same recursion, sw read per frame."""
    n = len(p_dead)
    pd = np.clip(np.nan_to_num(np.asarray(p_dead, dtype=float), nan=0.5), EPS, 1 - EPS)
    ph = np.clip(np.nan_to_num(np.asarray(p_home, dtype=float), nan=0.5), EPS, 1 - EPS)
    live = np.where(np.asarray(forced_dead, dtype=bool), -np.inf, np.log(1 - pd))
    e_h, e_a = (live + np.log(ph)).tolist(), (live + np.log(1 - ph)).tolist()
    e_d = np.log(pd).tolist()
    costs = np.broadcast_to(np.asarray(switch_cost, dtype=float), (n,)).tolist()
    st, rs = float(stop_cost), float(restart_cost)
    back = bytearray(3 * n)
    h, a, d = e_h[0], e_a[0], e_d[0]
    for t in range(1, n):
        sw = costs[t]
        nh, bh = h, 0
        if a - sw > nh:
            nh, bh = a - sw, 1
        if d - rs > nh:
            nh, bh = d - rs, 2
        na, ba = a, 1
        if h - sw > na:
            na, ba = h - sw, 0
        if d - rs > na:
            na, ba = d - rs, 2
        nd, bd = d, 2
        if h - st > nd:
            nd, bd = h - st, 0
        if a - st > nd:
            nd, bd = a - st, 1
        h, a, d = nh + e_h[t], na + e_a[t], nd + e_d[t]
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = bh, ba, bd
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax([h, a, d]))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path


def flip_frames(d_home, d_away, radius_m, window):
    """True on the frames within `window` of a flip: the team nearest the ball changes from one
    frame to the next (both distances known on both frames) and the new nearest player is within
    radius_m. d_home / d_away: distance from the ball to each team's nearest player, NaN when
    the ball or the whole team is missing."""
    d_home, d_away = np.asarray(d_home, dtype=float), np.asarray(d_away, dtype=float)
    n = len(d_home)
    with np.errstate(invalid="ignore"):
        side = np.where(np.isnan(d_home) | np.isnan(d_away), -1, (d_away > d_home).astype(int))
        change = (side[1:] != side[:-1]) & (side[1:] >= 0) & (side[:-1] >= 0)
        change &= np.minimum(d_home, d_away)[1:] <= radius_m
    idx = np.flatnonzero(change) + 1
    mark = np.zeros(n + 1)
    np.add.at(mark, np.maximum(idx - window, 0), 1)
    np.add.at(mark, np.minimum(idx + window + 1, n), -1)
    return np.cumsum(mark[:-1]) > 0


def tempered(p, temperature):
    """sigmoid(logit(p) / temperature); NaN stays NaN. p is clipped as the decoders clip it."""
    q = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return 1.0 / (1.0 + np.exp(-np.log(q / (1 - q)) / temperature))
