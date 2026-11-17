"""Independent cross-check of Version D against karlo-v2 when the tracker's smoothing is unknown.

    python -m solution.robust_exp stats           # aggregate ball statistics of the smoothings
    python -m solution.robust_exp fold 0 1 2      # score those folds (writes one json per fold)
    python -m solution.robust_exp report          # tables + the adoption rule

Same harness as solution/smooth_train_exp.py (leave one game out on the 10 external matches, two
held-out seeds, submitted decoding). Held-out regimes = base degradation (default / harsh / lag)
x ball (clean / trailing mean of 21 frames / symmetric Gaussian sigma 6 frames per visible
stretch). Arms:

    K   karlo-v2: clean training balls, no touch features, no spell weights, clean dead model
    T   round 3: clean training balls, touch features, no weights, clean dead model
    D   Version D: trailing-21 training balls, touches, spell weights, dead model on smoothed balls
    Dc  Version D's team model with the dead model trained on clean balls

Nothing here reads the competition match. Verification code, not part of the submission.
"""

import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import dead_ball, dead_features, features  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import logo2 as L2  # noqa: E402
from solution import model as M  # noqa: E402
from solution import smooth_train_exp as S  # noqa: E402
from solution import touch_exp as T  # noqa: E402  (sets augment.TRACKER_SMOOTH_W = 1)

BASES = ("default", "harsh", "lag")
BALLS = ("clean", "trail21", "gauss6")
ARMS = {"K": dict(smooth=False, touch=False, weights=False, dead=False),
        "T": dict(smooth=False, touch=True, weights=False, dead=False),
        "D": dict(smooth=True, touch=True, weights=True, dead=True),
        "Dc": dict(smooth=True, touch=True, weights=True, dead=False)}
GAUSS_SIGMA = 6.0
OUT_DIR = os.path.join(os.path.dirname(L.CACHE_DIR), "robust_exp")
FPS = 25


def gauss_ball(ball, sigma=GAUSS_SIGMA):
    """Symmetric Gaussian along each visible stretch of the ball; missing frames stay missing."""
    out = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    for s, e in dead_ball.stretches(seen, [(0, len(ball))]):
        out[s:e] = gaussian_filter1d(ball[s:e], sigma, axis=0, mode="nearest")
    return out


def smooth_ball(ball, kind):
    ball = ball.astype(np.float64)
    if kind == "trail21":
        ball = T.tracker_ball(ball)
    elif kind == "gauss6":
        ball = gauss_ball(ball)
    elif kind != "clean":
        raise KeyError(kind)
    return ball.astype(np.float32)


def held_arrays(name, base, kind, k):
    held = dict(L2.held_arrays(name, base, k))
    held["ball_xy"] = smooth_ball(held["ball_xy"], kind)
    return held


def run_fold(i):
    names = L.names()
    name = names[i]
    runs = L.data()[name]["runs"]
    team_train, dead_train = L.train_names(name, "4"), L.train_names(name, "7")
    team = {}
    for a, c in ARMS.items():
        key = (c["smooth"], c["touch"], c["weights"])
        if key not in team:
            parts = [S.team_rows(n, c["smooth"], c["touch"]) for n in team_train]
            team[key] = S.fit(parts, [S.team_weights(n) for n in team_train] if c["weights"]
                              else None)
    dead = {sm: S.fit([S.dead_rows(n, sm) for n in dead_train]) for sm in (False, True)}
    out = {}
    for base in BASES:
        for kind in BALLS:
            sc = {a: [] for a in ARMS}
            for k in T.SEEDS:
                held = held_arrays(name, base, kind, k)
                _, XD = dead_features.compute(held, runs)
                forced = dead_ball.pre_kickoff_mask(held, runs)
                pdead = {sm: L2.dead_proba(m, XD) for sm, m in dead.items()}
                inputs = {}
                for a, c in ARMS.items():
                    key = (c["smooth"], c["touch"], c["weights"])
                    if key not in inputs:
                        inputs[key] = T.held_inputs(name, held, runs, team[key], c["touch"])
                    ph, d_us, d_them = inputs[key]
                    h = dict(ph=ph, pd7=pdead[c["dead"]], forced=forced, d1_us=d_us,
                             d1_them=d_them)
                    sc[a].append([float(v) for v in L2.score(
                        name, L2.flip_codes(name, h, T.FINAL), full=True)])
            out["{}/{}".format(base, kind)] = sc
            print(name, base, kind, "  ".join("{} {:.3f}".format(
                a, np.mean([r[0] for r in sc[a]])) for a in ARMS), flush=True)
    features.TOUCH_FEATURES = True
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "fold_{}.json".format(name)), "w") as handle:
        json.dump(out, handle)


def load():
    names = L.names()
    res = {}
    for i, name in enumerate(names):
        with open(os.path.join(OUT_DIR, "fold_{}.json".format(name))) as handle:
            fold = json.load(handle)
        for regime, sc in fold.items():
            for a, rows in sc.items():
                res.setdefault(regime, {}).setdefault(
                    a, np.zeros((len(names), len(T.SEEDS), 4)))[i] = rows
    return names, res


def report():
    names, res = load()
    met = np.array([n in L.METRICA for n in names])
    print("| held-out ball | base | arm | spellF1 | Metrica | IDSSE | macroF1 | poss pred/true |")
    print("|---|---|---|---|---|---|---|---|")
    for kind in BALLS:
        for base in BASES:
            for a in ARMS:
                r = res["{}/{}".format(base, kind)][a]
                s = L2.per_match(r)
                print("| {} | {} | {} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} / {:.0f} |".format(
                    kind, base, a, s.mean(), s[met].mean(), s[~met].mean(), r[:, :, 1].mean(),
                    r[:, :, 2].mean(), r[:, :, 3].mean()))
    for new, ref in (("D", "K"), ("T", "K"), ("D", "T"), ("Dc", "D")):
        print("\n{} vs {} (adoption rule: gain >= 0.008, wins >= 7, harsh / lag >= -0.005)".format(
            new, ref))
        print("| held-out ball | gain default | wins | Metrica | IDSSE | harsh | wins | lag | wins "
              "| verdict |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for kind in BALLS:
            d = {b: L2.per_match(res["{}/{}".format(b, kind)][new])
                 - L2.per_match(res["{}/{}".format(b, kind)][ref]) for b in BASES}
            wins = {b: int((v > 1e-9).sum()) for b, v in d.items()}
            print("| {} | {:+.4f} | {}/10 | {:+.3f} | {:+.3f} | {:+.4f} | {}/10 | {:+.4f} | {}/10 "
                  "| {} |".format(
                      kind, d["default"].mean(), wins["default"], d["default"][met].mean(),
                      d["default"][~met].mean(), d["harsh"].mean(), wins["harsh"],
                      d["lag"].mean(), wins["lag"],
                      L2.verdict(d["default"].mean(), wins["default"],
                                 {"harsh": d["harsh"].mean(), "lag": d["lag"].mean()})))
    print("\n| match | " + " | ".join("{} {}".format(a, kind) for kind in BALLS
                                      for a in ("K", "D")) + " |")
    print("|---|" + "---|" * (2 * len(BALLS)))
    for i, n in enumerate(names):
        print("| {} | {} |".format(n, " | ".join(
            "{:.3f}".format(L2.per_match(res["default/{}".format(kind)][a])[i])
            for kind in BALLS for a in ("K", "D"))))


def _jump(ball, lag):
    before = np.full_like(ball, np.nan)
    after = np.full_like(ball, np.nan)
    before[lag:] = (ball[lag:] - ball[:-lag]) * FPS / lag
    after[:-lag] = (ball[lag:] - ball[:-lag]) * FPS / lag
    return np.linalg.norm(after - before, axis=1)


def stats():
    """Aggregates of touch_diag.stats / touch_check on the external matches, per ball kind:
    99th percentile of the 3-frame velocity change, median ball-to-nearest-player distance, share
    of live frames with the ball, touches per minute of the submitted detector. Documented
    competition aggregates (docs/logo_external.md): 3.8 m/s, 1.88 m, 14.7 touches per minute."""
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    print("| ball | p99 3-frame dv m/s | p99 8-frame dv | nearest player p50 m | ball seen (live) "
          "| touches / min (min-max) |\n|---|---|---|---|---|---|")
    for kind in BALLS:
        rows = []
        for name in L.names():
            a = L.data()[name]
            runs = a["runs"]
            held = held_arrays(name, "default", kind, 1)
            ball = held["ball_xy"].astype(np.float64)
            live = L.inside_mask(a) & ~dead_ball.dead_mask(held, runs)
            seen = live & ~np.isnan(ball[:, 0])
            d0, d1 = M.nearest_distances(held, runs)
            near = np.fmin(d0, d1)
            dv3, dv8 = _jump(ball, 3), _jump(ball, 8)
            features.TOUCH_FEATURES = True
            cols, X = features.compute(held, runs)
            n = sum(int((X[live, cols.index(c)] == 0).sum())
                    for c in ("touch_since_us", "touch_since_them"))
            rows.append((np.percentile(dv3[seen & ~np.isnan(dv3)], 99),
                         np.percentile(dv8[seen & ~np.isnan(dv8)], 99),
                         np.median(near[seen & np.isfinite(near)]),
                         seen.sum() / live.sum(), n / (live.sum() / FPS / 60.0)))
        r = np.array(rows)
        print("| {} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.1f} ({:.1f}-{:.1f}) |".format(
            kind, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean(), r[:, 4].mean(),
            r[:, 4].min(), r[:, 4].max()))


LAGS = tuple(range(-20, 21, 2))


def lag_curve(arrays, runs):
    """Median distance from the ball at frame t to the nearest player position at frame t - lag,
    for each lag in LAGS, on graded frames with the ball. A causal (trailing) smoother leaves the
    ball where its player WAS, so the minimum sits at a positive lag (~ (w - 1) / 2 frames); a
    symmetric smoother keeps it at 0. One aggregate curve per match, no frame-level output."""
    ball = arrays["ball_xy"].astype(np.float64)
    players = arrays["players_xy"].astype(np.float64)
    n = len(ball)
    inside = np.zeros(n, dtype=bool)
    for start, stop in runs:
        inside[start:stop] = True
    out = []
    for lag in LAGS:
        shifted = np.full_like(players, np.nan)
        if lag > 0:
            shifted[lag:] = players[:-lag]
        elif lag < 0:
            shifted[:lag] = players[-lag:]
        else:
            shifted = players
        d = np.sqrt(np.square(shifted - ball[:, None, :]).sum(-1))
        near = np.where(np.isnan(d), np.inf, d).min(axis=1)
        out.append(np.median(near[inside & np.isfinite(near)]))
    return np.array(out)


def lagstat(match=False):
    """python -m solution.robust_exp lagstat          external matches, three ball kinds
    python -m solution.robust_exp lagstat match    RUN BY A PERSON: reads the competition match
    and prints the same aggregate curve for it."""
    print("| ball | best lag (frames) | " + " | ".join("{:+d}".format(l) for l in LAGS) + " |")
    print("|---|---|" + "---|" * len(LAGS))
    if match:
        from solution import tracking
        from solution.diagnostics import graded_runs
        c = lag_curve(tracking.load(), graded_runs())
        print("| competition | {:+d} | {} |".format(LAGS[int(np.argmin(c))],
                                                 " | ".join("{:.2f}".format(v) for v in c)))
        return
    for kind in BALLS:
        curves = np.array([lag_curve(held_arrays(name, "default", kind, 1),
                                     L.data()[name]["runs"]) for name in L.names()])
        best = [LAGS[int(i)] for i in curves.argmin(1)]
        print("| {} | {:+.1f} ({:+d}..{:+d}) | {} |".format(
            kind, np.mean(best), min(best), max(best),
            " | ".join("{:.2f}".format(v) for v in curves.mean(0))))


if __name__ == "__main__":
    cmd = (sys.argv[1:] or ["report"])[0]
    if cmd == "lagstat":
        lagstat("match" in sys.argv[2:])
        sys.exit()
    if cmd == "fold":
        for arg in sys.argv[2:]:
            run_fold(int(arg))
    elif cmd == "stats":
        stats()
    else:
        report()
