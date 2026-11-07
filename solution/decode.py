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
#   cost 2: 0.592   4: 0.630   6: 0.642   8: 0.643   10: 0.645   14: 0.648   20: 0.641   30: 0.625
# A plateau from 6 to 20; 10 sits in its middle rather than on its noisiest point.
SWITCH_COST = 10.0


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
