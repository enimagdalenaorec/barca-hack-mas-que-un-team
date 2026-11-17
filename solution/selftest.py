"""Assert-based self-checks on synthetic data (no match data needed).

    python -m solution.selftest      # runs every test_* function here, prints `ok N`
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import augment, dead_ball, dead_features, dead_model, decode, model  # noqa: E402

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


def test_viterbi3_per_frame_cost_equals_scalar_cost():
    rng = np.random.default_rng(0)
    p_dead, p_home = rng.random(500), rng.random(500)
    forced = np.zeros(500, bool)
    forced[:30] = True
    a = decode.viterbi3(p_dead, p_home, forced, 3.0, 2.0, 1.0)
    b = decode.viterbi3(p_dead, p_home, forced, np.full(500, 3.0), 2.0, 1.0)
    assert (a == b).all()


def test_flip_frames_marks_window_around_close_change_only():
    d_home = np.r_[np.full(50, 1.0), np.full(50, 5.0), np.full(50, 9.0)]
    d_away = np.r_[np.full(50, 5.0), np.full(50, 1.0), np.full(50, 20.0)]
    mark = decode.flip_frames(d_home, d_away, 2.0, 3)   # flip at 50 (close), at 100 (9 m: no)
    assert np.flatnonzero(mark).tolist() == list(range(47, 54))
    d_away[50:60] = np.nan                               # ball lost at the change: no flip
    assert not decode.flip_frames(d_home, d_away, 2.0, 3)[40:70].any()


def test_labels_from3_options_off_and_flip_allows_short_spell():
    arrays = synthetic_match(300)
    arrays["players_xy"][:] = arrays["players_xy"][0]          # everyone stands still
    home, away = arrays["players_xy"][0, 0], arrays["players_xy"][0, 11]
    ball = np.tile(home, (300, 1))
    ball[150:162] = away                                        # 12 frames at an away player
    arrays["ball_xy"] = ball
    p_home = np.full(300, .9)
    p_home[150:162] = .2
    p_dead = np.full(300, .02)
    runs = [(0, 300)]
    plain = model.labels_from3(p_home, p_dead, arrays, runs, (14, 20, 8), flip=None,
                               dead_temperature=None)
    assert (plain == model.pm.H).all()
    flip = model.labels_from3(p_home, p_dead, arrays, runs, (14, 20, 8), flip=(0.3, 2.0, 5))
    assert (flip[150:162] == model.pm.A).all() and (flip[:150] == model.pm.H).all()


def test_tempered_identity_and_flattening():
    p = np.r_[.1, .5, .9, np.nan]
    assert np.allclose(decode.tempered(p, 1.0)[:3], p[:3]) and np.isnan(decode.tempered(p, 2.0)[3])
    assert decode.tempered(p, 2.0)[0] > .1 and decode.tempered(p, 2.0)[2] < .9


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


def test_controlled_merges_short_opposing_spell():
    owner = np.r_[np.zeros(100), np.ones(30), np.zeros(100), np.ones(80)].astype(np.int8)
    out = model.controlled(owner, np.zeros(310, bool), [(0, 310)], 50)
    assert (out[:230] == 0).all() and (out[230:] == 1).all()


def test_controlled_keeps_restart_spell_and_merges_short_last_spell():
    # dead | A 10 | B 100 | A 20 | dead
    owner = np.r_[np.full(20, -1), np.zeros(10), np.ones(100), np.zeros(20), np.full(20, -1)]
    owner = owner.astype(np.int8)
    dead = owner < 0
    out = model.controlled(owner, dead, [(0, 170)], 50)
    assert (out[20:30] == 0).all() and (out[30:150] == 1).all() and (out[dead] == -1).all()


def test_controlled_contested_phase_stays_with_last_team():
    owner = np.r_[np.zeros(100), np.tile(np.r_[np.ones(20), np.zeros(20)], 4)].astype(np.int8)
    out = model.controlled(owner, np.zeros(len(owner), bool), [(0, len(owner))], 50)
    assert (out == 0).all()


def test_tracker_detects_lagging_ball_and_restores_configs():
    from solution import augment, features, model, tracker
    rng = np.random.default_rng(0)
    n = 3000
    walk = np.cumsum(rng.normal(0, 0.3, size=(n, 2)), axis=0) + 50.0
    players = np.full((n, 4, 2), np.nan)
    players[:, 0] = walk
    players[:, 1:] = walk[:, None, :] + rng.normal(0, 10, size=(1, 3, 2))
    runs = [(0, n)]
    on_time = {"ball_xy": walk.copy(), "players_xy": players}
    lagged = {"ball_xy": np.vstack([np.full((10, 2), np.nan), walk[:-10]]), "players_xy": players}
    assert tracker.best_lag(on_time, runs) == 0
    assert tracker.best_lag(lagged, runs) == 10
    assert tracker.detect(on_time, runs) == "clean" and tracker.detect(lagged, runs) == "trailing"
    saved = (augment.TRACKER_SMOOTH_W, features.TOUCH_FEATURES, model.SPELL_WEIGHTS)
    try:
        tracker.apply("clean")
        assert (augment.TRACKER_SMOOTH_W, features.TOUCH_FEATURES, model.SPELL_WEIGHTS) == (1, False, True)
        tracker.apply("trailing")
        assert (augment.TRACKER_SMOOTH_W, features.TOUCH_FEATURES, model.SPELL_WEIGHTS) == (21, True, True)
    finally:
        augment.TRACKER_SMOOTH_W, features.TOUCH_FEATURES, model.SPELL_WEIGHTS = saved


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
    print("ok {}".format(len(tests)))


if __name__ == "__main__":
    main()
