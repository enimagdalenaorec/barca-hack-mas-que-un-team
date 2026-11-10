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
# Held-out mean spellF1 over 7 labelled training matches, final model
# (python -m solution.model --evaluate):
#   cost 2: 0.592   4: 0.630   6: 0.642   8: 0.643   10: 0.645   14: 0.647   20: 0.640   30: 0.624
# Chosen on the 3 Metrica matches only, whose labels are closest to the public ground truth's
# granularity, by a rule fixed before the run (python -m solution.experiments switch;
# docs/experiments.md, 14): 14 beats 10 on 3 of 3 (0.744 vs 0.739) and on the all-7 mean.
SWITCH_COST = 14.0
# Three-state decoder (viterbi3, pipeline --dead model): leaving play for a stoppage, and coming
# back. Best mean held-out spellF1 under the calibrated regime (python -m solution.experiments
# dead); the grid over {4, 8, 12, 20}^2 is flat, 0.657-0.659.
STOP_COST = 20.0
RESTART_COST = 8.0


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
    """
    n = len(p_dead)
    if n == 0:
        return np.empty(0, dtype=np.int8)
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
