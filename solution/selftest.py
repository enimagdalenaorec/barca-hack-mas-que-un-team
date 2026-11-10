"""Assert-based self-checks on synthetic data (no match data needed).

    python -m solution.selftest      # runs every test_* function here, prints `ok N`
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import augment, dead_ball, dead_features, dead_model, decode, gap_stats, model  # noqa: E402

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


def test_dead_features_shape_and_nan_outside_runs():
    arrays = synthetic_match(300)
    names, X = dead_features.compute(arrays, [(50, 250)])
    assert X.shape == (300, len(names)) and len(names) == 16
    assert np.isnan(X[:50]).all() and np.isnan(X[250:]).all()


def test_dead_features_still_players_read_as_still():
    arrays = synthetic_match(300, moving=False)
    names, X = dead_features.compute(arrays, [(0, 300)])
    assert np.nanmedian(X[:, names.index("still_share")]) > 0.9


def test_augment_params_do_not_leak():
    arrays = synthetic_match(5500, with_labels=True)  # long enough that 0.5 needs many gaps
    a = augment.augment(arrays, 1, {"LIVE_GAP_SHARE": 0.5})
    b = augment.augment(arrays, 1)
    assert augment.LIVE_GAP_SHARE == 0.05
    assert np.isnan(a["ball_xy"][:, 0]).sum() > np.isnan(b["ball_xy"][:, 0]).sum()


def test_dead_model_learns_synthetic_stoppages():
    m = [("s{}".format(i), synthetic_match(3000, with_labels=True, seed=i)) for i in range(3)]
    clf = dead_model.fit(m[:2])
    p = dead_model.p_dead(clf, m[2][1], m[2][1]["runs"])
    dead = m[2][1]["dead"]
    assert np.nanmean(p[dead]) > np.nanmean(p[~dead]) + 0.3


def test_calibrated_all_gaps_dead_gives_no_live_gaps():
    train = [synthetic_match(5500, with_labels=True, seed=s) for s in range(2)]
    ours = dict(train[0])
    ours["ball_xy"] = ours["ball_xy"].copy()  # gaps only on dead blocks
    out = gap_stats.calibrated(ours, ours["runs"], train)
    assert out["LIVE_GAP_SHARE"] < 0.01 and out["LIVE_GAP_S"] == (2.0, 15.0)  # < 5 fast gaps


def test_calibrated_live_gaps_raise_share():
    train = [synthetic_match(5500, with_labels=True, seed=s) for s in range(2)]
    ours = synthetic_match(5500, seed=7)  # moving players everywhere
    for a in range(0, 5500, 500):
        ours["ball_xy"][a:a + 150] = np.nan  # 11 fast 6 s gaps
    out = gap_stats.calibrated(ours, [(0, 5500)], train)
    assert out["LIVE_GAP_SHARE"] > 0.2 and 5.0 <= out["LIVE_GAP_S"][0] <= out["LIVE_GAP_S"][1] <= 7.0


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
    print("ok {}".format(len(tests)))


if __name__ == "__main__":
    main()
