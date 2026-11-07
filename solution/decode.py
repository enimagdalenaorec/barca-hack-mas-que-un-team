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
# Held-out mean spellF1 over 7 labelled training matches (python -m solution.model --evaluate):
#   cost  2: 0.544   4: 0.568   6: 0.568   8: 0.568   10: 0.567   14: 0.559   20: 0.548
# Flat from 4 to 10; 6 sits in the middle of the plateau.
SWITCH_COST = 6.0


def viterbi(p_home, switch_cost):
    """0 = home / 1 = away for each frame of one live stretch. NaN probabilities count as 0.5."""
    p = np.clip(np.nan_to_num(p_home, nan=0.5), EPS, 1 - EPS)
    emit = np.stack([np.log(p), np.log(1 - p)], axis=1)
    n = len(p)
    back = np.zeros((n, 2), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, n):
        stay, cross = score, score[::-1] - switch_cost
        back[t] = np.where(stay >= cross, [0, 1], [1, 0])
        score = np.maximum(stay, cross) + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
