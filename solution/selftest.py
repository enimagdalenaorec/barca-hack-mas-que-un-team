"""Assert-based self-checks on synthetic data (no match data needed).

    python -m solution.selftest      # runs every test_* function here, prints `ok N`
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import dead_ball, decode, model  # noqa: E402

FPS = 25
LIVE_BLOCK, DEAD_BLOCK = 400, 150


def synthetic_match(n, moving=True, with_labels=False, seed=0):
    """A fake match dict shaped like the training arrays: 11 v 11 on a 105 x 68 pitch, players
    random-walking at ~2 m/s (or standing still), the ball on player 0. With labels: one run,
    400-frame live blocks alternating with 150-frame dead blocks where players freeze and the
    ball is gone, and the owner alternating per live block."""
    rng = np.random.default_rng(seed)
    pitch = np.array([105.0, 68.0])
    xy = np.empty((n, 22, 2))
    xy[0] = rng.uniform(0, 1, (22, 2)) * pitch
    vel = np.zeros((22, 2))
    dead = np.zeros(n, dtype=bool)
    if with_labels:
        dead = (np.arange(n) % (LIVE_BLOCK + DEAD_BLOCK)) >= LIVE_BLOCK
    for t in range(1, n):
        vel = 0.9 * vel + rng.normal(0, 0.7, (22, 2))  # AR(1) heading: ~2 m/s on average
        step = vel / FPS if moving and not dead[t] else 0.0
        xy[t] = np.clip(xy[t - 1] + step, 0, pitch)
    out = {"pitch": pitch.astype(np.float32), "slot_team": np.r_[np.zeros(11), np.ones(11)].astype(int),
           "players_xy": xy.astype(np.float32), "ball_xy": xy[:, 0].astype(np.float32).copy(),
           "runs": np.array([[0, n]])}
    if with_labels:
        block = np.arange(n) // (LIVE_BLOCK + DEAD_BLOCK)
        out["ball_xy"][dead] = np.nan
        out["dead"] = dead
        out["owner"] = np.where(dead, -1, block % 2).astype(np.int8)
        out["has_dead"] = np.array(1)
    return out


def test_viterbi3_follows_clear_evidence():
    p_dead = np.r_[np.full(50, .02), np.full(50, .98), np.full(50, .02)]
    p_home = np.r_[np.full(50, .95), np.full(50, .5), np.full(50, .05)]
    path = decode.viterbi3(p_dead, p_home, np.zeros(150, bool), 10, 8, 8)
    assert (path[:50] == 0).all() and (path[50:100] == 2).all() and (path[100:] == 1).all()


def test_viterbi3_ignores_one_frame_blip():
    p_home = np.full(100, .9)
    p_home[40] = .01
    path = decode.viterbi3(np.full(100, .02), p_home, np.zeros(100, bool), 10, 8, 8)
    assert (path == 0).all()


def test_viterbi3_forced_dead_wins_over_evidence():
    forced = np.zeros(60, bool)
    forced[:20] = True
    path = decode.viterbi3(np.full(60, .01), np.full(60, .9), forced, 10, 8, 8)
    assert (path[:20] == 2).all() and (path[20:] == 0).all()


def test_viterbi3_nan_and_tiny_inputs():
    path = decode.viterbi3(np.full(30, np.nan), np.full(30, np.nan), np.zeros(30, bool), 10, 8, 8)
    assert path.shape == (30,) and set(path.tolist()) <= {0, 1, 2}
    assert decode.viterbi3(np.r_[.9], np.r_[.5], np.r_[False], 10, 8, 8).tolist() == [2]
    assert decode.viterbi3(np.array([]), np.array([]), np.array([], bool), 10, 8, 8).shape == (0,)


def test_pre_kickoff_mask_no_formation_is_all_false():
    arrays = synthetic_match(200)  # players scattered: no kick-off formation
    assert dead_ball.pre_kickoff_mask(arrays, [(0, 200)]).sum() == 0


def test_labels_from3_reproduces_clear_truth():
    arrays = synthetic_match(2000, with_labels=True)
    p_home = np.where(arrays["dead"], .5, np.where(arrays["owner"] == 0, .95, .05))
    p_dead = np.where(arrays["dead"], .95, .05)
    codes = model.labels_from3(p_home, p_dead, arrays, arrays["runs"], (10, 8, 8))
    assert (codes == model.truth_codes(arrays)).all()


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
    print("ok {}".format(len(tests)))


if __name__ == "__main__":
    main()
