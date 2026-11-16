"""Touch features (features.TOUCH_FEATURES), leave one game out on the 10 external matches.

    python -m solution.touch_exp sanity     # how touches line up with the true owner, one match
    python -m solution.touch_exp run        # the experiment, judged by logo2's adoption rule

Same harness as solution/logo2.py: for each external match the team model is trained on the
other matches (Metrica + the 4 IDSSE of the team model), P(dead) on the other 9 (all 7 IDSSE);
the held-out match is degraded under the regimes default / harsh / lag with two seeds, and decoded
with exactly the submitted configuration (viterbi3, flip rule, P(dead) on 7 IDSSE, temperature 2).
Both arms -- touch features off (the reference, which must reproduce the submitted 0.664) and on
-- share everything else. Adoption, fixed in logo2 before any measurement: default mean gain
>= 0.008, wins on >= 7 of 10 matches, harsh and lag not below -0.005.

Nothing here reads the competition match.
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import augment, dead_ball, dead_features, features  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import logo2 as L2  # noqa: E402
from solution import model as M  # noqa: E402

pm = M.pm
# These experiments were run before the tracker-like smoothing became part of augment.py (and add
# it themselves where they need it), so they switch it off to keep reproducing their numbers.
augment.TRACKER_SMOOTH_W = 1

# Params
SPECS = ("default", "harsh", "lag")
SEEDS = L2.SEEDS
FINAL = L2.BEST_FLIP + ("pd7", 2.0)   # the submitted decoding
OUT = os.path.join(os.path.dirname(L.CACHE_DIR), "touch_exp_{}.json")

_ROWS = {}


def team_rows(name, touch):
    """As logo.train_rows with the reference settings, touch features on or off."""
    key = (name, touch)
    if key not in _ROWS:
        arrays = L.data()[name]
        aug = L.degrade(name, ("default",), L.train_seed(name, 0))
        owner = arrays["owner"]
        idx = np.flatnonzero((owner >= 0) & ~arrays["dead"])[::M.SUBSAMPLE]
        features.TOUCH_FEATURES = touch
        xs, ys = [], []
        for swap in (False, True):
            _, X = features.compute(aug, arrays["runs"], swap=swap)
            xs.append(X[idx])
            ys.append((owner[idx] == (1 if swap else 0)).astype(np.int8))
        _ROWS[key] = np.concatenate(xs), np.concatenate(ys)
    return _ROWS[key]


def fit_team(train, touch):
    parts = [team_rows(n, touch) for n in train]
    gbm = dict(M.GBM, max_features=L.REF.max_features, random_state=L.REF.rs)
    return HistGradientBoostingClassifier(**gbm).fit(
        np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))


def held_inputs(name, held, runs, clf, touch):
    """The decoder's inputs from one degraded held-out copy: P(home) and the flip evidence."""
    features.TOUCH_FEATURES = touch
    ph, X0, cols = L2.team_proba(clf, held, runs)
    return ph, X0[:, cols.index("d1_us")], X0[:, cols.index("d1_them")]


JITTER_M = {"jitter10": 0.10, "jitter20": 0.20}


TRACKER_W = 21  # frames of trailing average; see tracker_ball()


def tracker_ball(ball, w=TRACKER_W):
    """A ball smoothed the way the competition tracker appears to smooth it: a trailing (causal)
    moving average, which also makes the ball lag its player. w = 21 brings the external balls'
    99th-percentile 3-frame velocity change from 16.8 to ~3.8 m/s and their median ball-to-nearest
    -player distance from 1.25 to ~1.6 m, against 3.8 m/s and 1.88 m on the competition match
    (aggregate numbers from python -m solution.touch_diag, run by a person). A stress regime for
    validation only; no model setting is chosen from it."""
    n = len(ball)
    cs = np.concatenate([np.zeros((1, 2)), np.cumsum(np.nan_to_num(ball), axis=0)])
    cn = np.concatenate([[0], np.cumsum(~np.isnan(ball[:, 0]))])
    hi = np.arange(1, n + 1)
    lo = np.maximum(0, hi - w)
    full = ((cn[hi] - cn[lo]) == hi - lo) & ~np.isnan(ball[:, 0])
    return np.where(full[:, None], (cs[hi] - cs[lo]) / (hi - lo)[:, None], np.nan)


def held_arrays(name, spec, k):
    if spec == "tracker":
        held = dict(L2.held_arrays(name, "default", k))
        held["ball_xy"] = tracker_ball(held["ball_xy"].astype(np.float64)).astype(np.float32)
        return held
    """logo2's degraded held-out copy; "jitterNN" adds independent per-frame ball noise of NN cm
    on top of the default regime -- the noise that could fake a touch, which logo's smooth
    position noise (1 s correlation time) cannot."""
    if spec not in JITTER_M:
        return L2.held_arrays(name, spec, k)
    held = dict(L2.held_arrays(name, "default", k))
    rng = np.random.default_rng(M.seed_of(name) + 100 * k)
    held["ball_xy"] = (held["ball_xy"] + rng.normal(0, JITTER_M[spec], held["ball_xy"].shape)
                       ).astype(np.float32)
    return held


def run(specs=SPECS, tag="main"):
    global SPECS
    SPECS = specs
    names = L.names()
    scores = {t: {s: np.zeros((len(names), len(SEEDS), 4)) for s in SPECS} for t in (False, True)}
    for i, name in enumerate(names):
        runs = L.data()[name]["runs"]
        team_train = L.train_names(name, "4")
        models = {t: fit_team(team_train, t) for t in (False, True)}
        dead7 = L.fitted("dead", L.train_names(name, "7"), L2.SEVEN)
        for spec in SPECS:
            for j, k in enumerate(SEEDS):
                held = held_arrays(name, spec, k)
                _, XD = dead_features.compute(held, runs)
                common = dict(pd7=L2.dead_proba(dead7, XD),
                              forced=dead_ball.pre_kickoff_mask(held, runs))
                for t in (False, True):
                    ph, d_us, d_them = held_inputs(name, held, runs, models[t], t)
                    h = dict(common, ph=ph, d1_us=d_us, d1_them=d_them)
                    scores[t][spec][i, j] = L2.score(name, L2.flip_codes(name, h, FINAL),
                                                     full=True)
        L._MODELS.clear()
        print("done", name, " off {:.3f}  on {:.3f}".format(
            scores[False][SPECS[0]][i, :, 0].mean(), scores[True][SPECS[0]][i, :, 0].mean()),
            flush=True)
    features.TOUCH_FEATURES = False
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT.format(tag), "w") as handle:
        json.dump({str(t): {s: v.tolist() for s, v in d.items()} for t, d in scores.items()},
                  handle)
    if tag == "main":
        report(names, scores)
    else:
        report_extra(names, scores)


HEAVY = dict(TOUCH_SMOOTH=25, TOUCH_LAG=8, TOUCH_MIN_DV=3.0, TOUCH_RADIUS_M=3.5)


def heavy():
    """Touches on a ball smoothed over 1 s, so that the detector no longer depends on how much
    the tracker itself smooths the ball (touch_diag: the competition ball changes velocity far
    more gradually than the external balls). Settings chosen on the external matches only:
    after this smoothing, real kicks are 3-8 m/s jumps over 8 frames with the nearest player
    at a median 1.9 m."""
    for key, value in HEAVY.items():
        setattr(features, key, value)
    run(("default", "harsh", "lag", "jitter20"), "heavy")


def smoothed(k=5):
    """Touches on a ball path smoothed over k frames, under the adoption regimes and jitter."""
    features.TOUCH_SMOOTH = k
    run(("default", "harsh", "lag", "jitter10", "jitter20"), "smooth{}".format(k))


def report_extra(names, scores):
    """Safety checks outside the adoption rule (logo2's position-noise regimes)."""
    print("\n| regime | touch off | touch on | gain | wins |\n|---|---|---|---|---|")
    for s in SPECS:
        off, on = L2.per_match(scores[False][s]), L2.per_match(scores[True][s])
        print("| {} | {:.4f} | {:.4f} | {:+.4f} | {}/10 |".format(
            s, off.mean(), on.mean(), on.mean() - off.mean(), int((on > off + 1e-9).sum())))


def report(names, scores):
    met = np.array([n in L.METRICA for n in names])
    ref, new = scores[False], scores[True]
    print("\n| arm | spellF1 default | Metrica | IDSSE | macroF1 | harsh | lag | possessions pred / true |")
    print("|---|---|---|---|---|---|---|---|")
    for label, sc in (("touch off (reference)", ref), ("touch on", new)):
        d = L2.per_match(sc["default"])
        print("| {} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.0f} / {:.0f} |".format(
            label, d.mean(), d[met].mean(), d[~met].mean(), sc["default"][:, :, 1].mean(),
            L2.per_match(sc["harsh"]).mean(), L2.per_match(sc["lag"]).mean(),
            sc["default"][:, :, 2].mean(), sc["default"][:, :, 3].mean()))
    gain = L2.per_match(new["default"]).mean() - L2.per_match(ref["default"]).mean()
    wins = int((L2.per_match(new["default"]) > L2.per_match(ref["default"]) + 1e-9).sum())
    robust = {s: L2.per_match(new[s]).mean() - L2.per_match(ref[s]).mean() for s in ("harsh", "lag")}
    print("\ngain {:+.4f}, wins {}/10, harsh {:+.4f}, lag {:+.4f} -> {}".format(
        gain, wins, robust["harsh"], robust["lag"], L2.verdict(gain, wins, robust)))
    print("\n| match | off | on |\n|---|---|---|")
    for i, n in enumerate(names):
        print("| {} | {:.3f} | {:.3f} |".format(n, L2.per_match(ref["default"])[i],
                                               L2.per_match(new["default"])[i]))


def sanity(name="metrica1"):
    a = L.data()[name]
    held = L.degrade(name, ("default",), M.seed_of(name) + 1)
    features.TOUCH_FEATURES = True
    names, X = features.compute(held, a["runs"])
    features.TOUCH_FEATURES = False
    col = {c: X[:, names.index(c)] for c in names}
    live = (a["owner"] >= 0) & ~a["dead"]
    touches = int((col["touch_since_us"] == 0).sum() + (col["touch_since_them"] == 0).sum())
    print("{} features; {} touches detected on {} ({:.1f} per minute of live play)".format(
        len(names), touches, name, touches / (live.sum() / 25 / 60)))
    for c, test in (("touch_last", lambda v: v != 0), ("near_us_past1", lambda v: ~np.isnan(v))):
        m = live & test(col[c])
        side = col[c][m] > (0 if c == "touch_last" else 0.5)
        print("  {:<14} agrees with the true owner on {:.3f} of the {:.0%} of live frames it covers"
              .format(c, (side == (a["owner"][m] == 0)).mean(), m.sum() / live.sum()))


if __name__ == "__main__":
    {"sanity": sanity, "run": run,
     "noise": lambda: run(("noise", "noisehigh", "jitter10", "jitter20"), "noise"),
     "smooth": lambda: smoothed(),
     "heavy": heavy,
     "tracker": lambda: run(("tracker",), "tracker"),
     }[(sys.argv[1:] or ["sanity"])[0]]()
