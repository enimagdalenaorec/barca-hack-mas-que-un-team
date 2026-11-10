"""Experiments on the possession model, all scored the same honest way: leave one match out.

    python -m solution.experiments timing      # 1: are our possession changes late?
    python -m solution.experiments decoding    # 2: smoothing variants
    python -m solution.experiments features    # 3: which feature groups matter
    python -m solution.experiments mix         # 4: which training matches
    python -m solution.experiments models      # 5: other models and settings
    python -m solution.experiments augment     # 6: sensitivity to the augmentation guesses
    python -m solution.experiments stacking    # 7: a second model over the first one's guesses
    python -m solution.experiments budget      # where are the remaining points?
    python -m solution.experiments labels      # 8: training-label granularity vs the public window

Each held-out match is one of the 7 with a full H/A/D label (Metrica, IDSSE); the model is
trained on the other 11 matches every time. Results are the mean over the 7 and, against the
reference configuration, on how many of the 7 a variant is better -- with 7 matches a difference
under ~0.01-0.02 spellF1 is noise unless it wins on most of them.

Nothing here touches the competition match.
"""

import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import augment, dead_ball, decode, features  # noqa: E402
from solution import model as M  # noqa: E402

pm = M.pm

# Params
TOL = 50  # frames: how far apart a predicted and a true team switch may be to count as a pair


# -- cached building blocks --------------------------------------------------------------------

MATCHES = None
_HELD = {}
_ROWS = {}


def matches():
    global MATCHES
    if MATCHES is None:
        MATCHES = M.training_matches()
    return MATCHES


def labelled():
    return [m for m in matches() if int(m[1]["has_dead"])]


def held_out(name, arrays):
    """Augmented held-out copy, truth codes and both feature views. Model-independent."""
    if name not in _HELD:
        held = augment.augment(arrays, M.seed_of(name) + 1)
        names, X0 = features.compute(held, arrays["runs"], swap=False)
        _, X1 = features.compute(held, arrays["runs"], swap=True)
        _HELD[name] = dict(held=held, gt=M.truth_codes(arrays), X0=X0, X1=X1, names=names)
    return _HELD[name]


def train_rows(name, arrays, aug=None, offset=0):
    """(X, y, names) training rows; `aug` overrides augment.py params for this call; `offset`
    picks which of the SUBSAMPLE interleaved frame sets to use."""
    key = (name, tuple(sorted((aug or {}).items())), offset)
    if key not in _ROWS:
        saved = {k: getattr(augment, k) for k in (aug or {})}
        for k, v in (aug or {}).items():
            setattr(augment, k, v)
        try:
            aug_arrays = augment.augment(arrays, M.seed_of(name))
        finally:
            for k, v in saved.items():
                setattr(augment, k, v)
        owner = arrays["owner"]
        keep = owner >= 0
        if int(arrays["has_dead"]):
            keep &= ~arrays["dead"]
        idx = np.flatnonzero(keep)[offset::M.SUBSAMPLE]
        xs, ys = [], []
        for swap in (False, True):
            names, X = features.compute(aug_arrays, arrays["runs"], swap=swap)
            xs.append(X[idx])
            ys.append((owner[idx] == (1 if swap else 0)).astype(np.int8))
        _ROWS[key] = (np.concatenate(xs), np.concatenate(ys), names)
    return _ROWS[key]


def default_model():
    return HistGradientBoostingClassifier(**M.GBM)


def predictions(columns=None, make_model=default_model, use=lambda name: True,
                weight=lambda name: 1.0, aug=None, offsets=(0,)):
    """{held-out name: P(home) per frame} for one configuration. Several `offsets` train one
    model per interleaved frame set and average them (an ensemble over all frames)."""
    out = {}
    for name, arrays in labelled():
        train = [(n, a) for n, a in matches() if n != name and use(n)]
        h = held_out(name, arrays)
        inside = ~np.isnan(h["X0"]).all(axis=1)
        p = np.zeros(len(h["X0"]))
        for offset in offsets:
            parts = [train_rows(n, a, aug, offset) for n, a in train]
            names = parts[0][2]
            cols = [names.index(c) for c in (columns or names)]
            X = np.concatenate([q[0][:, cols] for q in parts])
            y = np.concatenate([q[1] for q in parts])
            w = np.concatenate([np.full(len(q[1]), weight(n)) for q, (n, _) in zip(parts, train)])
            clf = make_model()
            clf = clf.fit(X, y) if np.all(w == 1) else clf.fit(X, y, sample_weight=w)
            p[inside] += 0.5 * (clf.predict_proba(h["X0"][inside][:, cols])[:, 1]
                                + 1 - clf.predict_proba(h["X1"][inside][:, cols])[:, 1])
        p /= len(offsets)
        p[~inside] = np.nan
        out[name] = p
    return out


# -- decoding ----------------------------------------------------------------------------------

def decode_match(name, p, cost=decode.SWITCH_COST, cost_missing=None, shift=0, min_len=0):
    """Codes for one held-out match under a decoding variant."""
    arrays = dict(labelled())[name]
    h = held_out(name, arrays)
    held, runs = h["held"], arrays["runs"]
    dead = dead_ball.dead_mask(held, runs)
    codes = np.full(len(p), pm.X, dtype=np.int8)
    live = np.zeros(len(p), dtype=bool)
    for start, stop in runs:
        codes[start:stop][dead[start:stop]] = pm.D
        live[start:stop] = ~dead[start:stop]
    seen = ~np.isnan(held["ball_xy"][:, 0])
    for start, stop in dead_ball.stretches(live, runs):
        c = cost if cost_missing is None else np.where(seen[start:stop], cost, cost_missing)
        path = decode.viterbi(p[start:stop], c)
        if min_len:
            path = drop_short(path, min_len)
        if shift:
            path = shift_switches(path, shift)
        codes[start:stop] = np.where(path == 0, pm.H, pm.A)
    return codes


def drop_short(path, min_len):
    """Merge possessions shorter than min_len frames into the previous one (or the next, first)."""
    path = path.copy()
    changed = True
    while changed:
        changed = False
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(path)) + 1, [len(path)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < min_len and len(bounds) > 2:
                path[a:b] = path[a - 1] if a > 0 else path[b]
                changed = True
                break
    return path


def shift_switches(path, k):
    """Move every team switch k frames earlier (later if k < 0), never past its neighbours."""
    bounds = list(np.flatnonzero(np.diff(path)) + 1)
    if not bounds:
        return path
    out = path.copy()
    edges = [0] + bounds + [len(path)]
    for i in range(1, len(edges) - 1):
        t = edges[i]
        new = int(np.clip(t - k, edges[i - 1] + 1, edges[i + 1] - 1))
        out[min(t, new):max(t, new)] = path[t] if new < t else path[t - 1]
    return out


# -- scoring -----------------------------------------------------------------------------------

def switch_latency(gt, pred, runs):
    """Signed latency (pred - truth, frames) of matched H<->A switches."""
    def team_switches(codes):
        return [t for t in pm.transitions(codes, runs)
                if t[1] in (pm.H, pm.A) and t[2] in (pm.H, pm.A)]
    pairs, _, _ = pm.match_transitions(team_switches(gt), team_switches(pred), TOL)
    return np.array([p[0] - g[0] for g, p in pairs], dtype=float)


def score_all(codes_by_name):
    """{name: (spellF1, macroF1, n_pred_spells)} and the mean row."""
    per = {}
    for name, codes in codes_by_name.items():
        r = pm.evaluate(held_out(name, dict(labelled())[name])["gt"], codes)
        per[name] = (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"])
    return per


def table(title, variants, reference):
    """Print mean spellF1 / macroF1 / possessions and wins vs the reference variant."""
    print("\n" + title)
    print("  {:<34} {:>8} {:>8} {:>6} {:>10}".format("variant", "spellF1", "macroF1", "poss",
                                                     "wins vs ref"))
    ref = variants[reference]
    for label, per in variants.items():
        s = np.array([per[n][0] for n in per])
        m = np.array([per[n][1] for n in per])
        poss = np.array([per[n][2] for n in per])
        wins = sum(per[n][0] > ref[n][0] + 1e-9 for n in per)
        mark = " (ref)" if label == reference else "{}/{}".format(wins, len(per))
        by_source = "  ".join("{} {:.3f}".format(src, np.mean([per[n][0] for n in per
                                                               if n.startswith(src)]))
                              for src in ("metrica", "idsse"))
        print("  {:<34} {:>8.3f} {:>8.3f} {:>6.0f} {:>10}   {}".format(
            label, s.mean(), m.mean(), poss.mean(), mark, by_source))


# -- the experiments ---------------------------------------------------------------------------

def exp_timing(p):
    lat = []
    for name, probs in p.items():
        codes = decode_match(name, probs)
        lat.append(switch_latency(held_out(name, dict(labelled())[name])["gt"], codes,
                                  dict(labelled())[name]["runs"]))
    lat = np.concatenate(lat)
    print("\nexp 1 -- team-switch latency (predicted minus true, frames; + means late)")
    print("  matched switches {}  mean {:+.1f}  median {:+.1f}  p25 {:+.1f}  p75 {:+.1f}".format(
        len(lat), lat.mean(), np.median(lat), *np.percentile(lat, [25, 75])))
    variants = {}
    for k in (-10, 0, 5, 10, 15, 20, 25, 35):
        variants["shift switches {:+d} frames earlier".format(k)] = score_all(
            {n: decode_match(n, q, shift=k) for n, q in p.items()})
    table("exp 1 -- shifting every team switch earlier", variants,
          "shift switches +0 frames earlier")


def exp_decoding(p):
    variants = {"reference (cost 6)": score_all({n: decode_match(n, q) for n, q in p.items()})}
    for c_missing in (3, 10, 20):
        variants["cost 6 seen / {} ball missing".format(c_missing)] = score_all(
            {n: decode_match(n, q, cost_missing=c_missing) for n, q in p.items()})
    for c_seen, c_missing in ((4, 12), (8, 20)):
        variants["cost {} seen / {} ball missing".format(c_seen, c_missing)] = score_all(
            {n: decode_match(n, q, cost=c_seen, cost_missing=c_missing) for n, q in p.items()})
    for m in (13, 25, 50):
        variants["min possession {:.1f} s".format(m / 25)] = score_all(
            {n: decode_match(n, q, min_len=m) for n, q in p.items()})
    table("exp 2 -- smoothing variants", variants, "reference (cost 6)")


GROUPS = {
    "around the ball": ("d1_", "d2_", "n3_", "n7_", "ball_speed", "ball_x", "ball_y_off"),
    "over time (1-3 s)": ("near_us_past1", "near_us_next1", "near_us_past3", "near_us_next3",
                          "d1_diff_"),
    "ball visibility": ("since_seen_s", "until_seen_s", "last_ball_x"),
    "team shape": ("cx_", "sx_", "sy_", "count_", "vx_", "spread_ratio", "ball_vs_cx_"),
    "ball flight (new)": ("flight",),
    "control (new)": ("comove_", "pressed_"),
    "long context (new)": ("near_us_past5", "near_us_next5", "near_us_past10", "near_us_next10"),
    "restart": ("since_restart_s", "until_stop_s", "stop_len_s", "before_stop_", "restart_"),
}


def exp_restart():
    names = held_out(*labelled()[0])["names"]
    restart = [f for f in names if group_of(f) == "restart"]
    without = [f for f in names if group_of(f) != "restart"]
    no_press = [f for f in names if not f.startswith("pressed_")]
    variants = {
        "without restart features": score_predictions(predictions(columns=without)),
        "with restart features ({})".format(len(restart)): score_predictions(predictions()),
        "with restart, without pressing": score_predictions(predictions(columns=no_press)),
    }
    table("restart features (football rules of who restarts)", variants,
          "without restart features")


def group_of(feature):
    for group, prefixes in GROUPS.items():
        if any(feature == p or feature.startswith(p) for p in prefixes):
            return group
    raise KeyError(feature)


def score_predictions(p):
    return score_all({n: decode_match(n, q) for n, q in p.items()})


def exp_features():
    names = held_out(*labelled()[0])["names"]
    new = {"ball flight (new)", "control (new)", "long context (new)"}
    old = [f for f in names if group_of(f) not in new]
    variants = {"all {} features".format(len(names)): score_predictions(predictions())}
    variants["the original {} features".format(len(old))] = score_predictions(
        predictions(columns=old))
    for group in GROUPS:
        keep = [f for f in names if group_of(f) != group]
        variants["without " + group] = score_predictions(predictions(columns=keep))
    table("exp 3 -- feature groups", variants, "all {} features".format(len(names)))


def exp_mix():
    variants = {"all 12 matches": score_predictions(predictions())}
    variants["without SkillCorner"] = score_predictions(
        predictions(use=lambda n: not n.startswith("sc")))
    variants["SkillCorner at half weight"] = score_predictions(
        predictions(weight=lambda n: 0.5 if n.startswith("sc") else 1.0))
    variants["without Metrica"] = score_predictions(
        predictions(use=lambda n: not n.startswith("metrica")))
    variants["without IDSSE"] = score_predictions(
        predictions(use=lambda n: not n.startswith("idsse")))
    table("exp 4 -- training mix (each held-out match always excluded from its own training)",
          variants, "all 12 matches")


def exp_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def gbm(**kw):
        return lambda: HistGradientBoostingClassifier(**{**M.GBM, **kw})

    variants = {
        "boosting: 300 trees, 31 leaves": score_predictions(predictions()),
        "boosting: 150 trees, 15 leaves": score_predictions(predictions(
            make_model=gbm(max_iter=150, max_leaf_nodes=15))),
        "boosting: 600 trees, 63 leaves": score_predictions(predictions(
            make_model=gbm(max_iter=600, max_leaf_nodes=63))),
        "boosting: 300 trees, lr 0.1": score_predictions(predictions(
            make_model=gbm(learning_rate=0.1))),
        "logistic regression": score_predictions(predictions(make_model=lambda: make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0)))),
        "random forest (200 trees)": score_predictions(predictions(
            make_model=lambda: RandomForestClassifier(
                n_estimators=200, min_samples_leaf=50, max_features="sqrt", n_jobs=-1,
                random_state=0))),
    }
    table("exp 5 -- models", variants, "boosting: 300 trees, 31 leaves")


def exp_augment():
    variants = {"reference augmentation": score_predictions(predictions())}
    for label, aug in (("no live-play ball gaps", {"LIVE_GAP_SHARE": 0.0}),
                       ("3x live-play ball gaps (15%)", {"LIVE_GAP_SHARE": 0.15}),
                       ("hide 80% of stoppages", {"HIDE_DEAD_P": 0.8}),
                       ("hide every stoppage", {"HIDE_DEAD_P": 1.0}),
                       ("no dropped players", {"PLAYER_DROP_SHARE": 0.0}),
                       ("drop players 10% of time", {"PLAYER_DROP_SHARE": 0.10}),
                       ("no augmentation at all", {"LIVE_GAP_SHARE": 0.0, "HIDE_DEAD_P": 0.0,
                                                   "PLAYER_DROP_SHARE": 0.0, "BRIDGE_S": 0.0})):
        variants[label] = score_predictions(predictions(aug=aug))
    table("exp 6 -- training augmentation (held-out matches keep the reference augmentation)",
          variants, "reference augmentation")


STACK_WINDOWS_S = (0.2, 0.5, 1.0, 2.0, 5.0)


def stack_features(p, runs):
    """Second-stage inputs: the first model's P(home) and its mean over past/next windows."""
    cols = [p]
    for w in STACK_WINDOWS_S:
        frames = max(1, int(w * 25))
        for direction in (-1, 1):
            cols.append(features._rolling_mean(p, frames, direction))
    X = np.stack(cols, axis=1).astype(np.float32)
    inside = np.zeros(len(p), dtype=bool)
    for start, stop in runs:
        inside[start:stop] = True
    X[~inside] = np.nan
    return X


def first_stage_p(train, target_arrays_X0X1, make_model=default_model):
    """Fit the first stage on `train` matches, return P(home) for given (X0, X1) feature views."""
    parts = [train_rows(n, a) for n, a in train]
    X = np.concatenate([q[0] for q in parts])
    y = np.concatenate([q[1] for q in parts])
    clf = make_model().fit(X, y)
    out = []
    for X0, X1 in target_arrays_X0X1:
        p = np.full(len(X0), np.nan)
        inside = ~np.isnan(X0).all(axis=1)
        p[inside] = 0.5 * (clf.predict_proba(X0[inside])[:, 1]
                           + 1 - clf.predict_proba(X1[inside])[:, 1])
        out.append(p)
    return out


def _train_views(name, arrays):
    """Feature views of a TRAINING match as the model would see it (augmented, seed of name)."""
    aug = augment.augment(arrays, M.seed_of(name))
    _, X0 = features.compute(aug, arrays["runs"], swap=False)
    _, X1 = features.compute(aug, arrays["runs"], swap=True)
    return X0, X1


def exp_stacking(use=lambda name: True, make_model=default_model):
    """Nested leave-one-out, so the second stage only ever sees honest first-stage guesses."""
    views = {n: _train_views(n, a) for n, a in matches() if use(n)}
    first, stacked = {}, {}
    for name, arrays in labelled():
        train = [(n, a) for n, a in matches() if n != name and use(n)]
        # Out-of-fold first-stage guesses on every training match (inner leave-one-out).
        xs, ys = [], []
        for inner, inner_arrays in train:
            p_inner = first_stage_p([m for m in train if m[0] != inner], [views[inner]],
                                    make_model)[0]
            owner = inner_arrays["owner"]
            keep = owner >= 0
            if int(inner_arrays["has_dead"]):
                keep &= ~inner_arrays["dead"]
            idx = np.flatnonzero(keep)[::M.SUBSAMPLE]
            xs.append(stack_features(p_inner, inner_arrays["runs"])[idx])
            ys.append((owner[idx] == 0).astype(np.int8))
        second = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                max_leaf_nodes=15, min_samples_leaf=200,
                                                random_state=0).fit(np.concatenate(xs),
                                                                    np.concatenate(ys))
        h = held_out(name, arrays)
        p1 = first_stage_p(train, [(h["X0"], h["X1"])], make_model)[0]
        p2 = np.full(len(p1), np.nan)
        Xs = stack_features(p1, arrays["runs"])
        ok = ~np.isnan(Xs[:, 0])
        p2[ok] = second.predict_proba(Xs[ok])[:, 1]
        first[name], stacked[name] = p1, p2
        print("  stacking: {} done".format(name), flush=True)
    variants = {"first stage only (cost 6)": score_predictions(first)}
    for cost in (2, 4, 6, 10):
        variants["stacked, cost {}".format(cost)] = score_all(
            {n: decode_match(n, q, cost=cost) for n, q in stacked.items()})
    table("exp 7 -- second model over the first model's guesses", variants,
          "first stage only (cost 6)")


def no_sc(name):
    return not name.startswith("sc")


def exp_combo():
    """The winners of experiments 4 and 5 together, with the switch cost re-tuned."""
    def gbm(**kw):
        return lambda: HistGradientBoostingClassifier(**{**M.GBM, **kw})

    configs = {
        "all 12, reference model": dict(),
        "no SC, reference model": dict(use=no_sc),
        "no SC, 600 trees / 63 leaves": dict(use=no_sc, make_model=gbm(max_iter=600,
                                                                       max_leaf_nodes=63)),
        "no SC, 600 / 63, lr 0.1": dict(use=no_sc, make_model=gbm(
            max_iter=600, max_leaf_nodes=63, learning_rate=0.1)),
        "no SC, 1000 / 63": dict(use=no_sc, make_model=gbm(max_iter=1000, max_leaf_nodes=63)),
        "no SC, 600 / 127": dict(use=no_sc, make_model=gbm(max_iter=600, max_leaf_nodes=127)),
    }
    variants = {}
    for label, cfg in configs.items():
        p = predictions(**cfg)
        for cost in (4, 6, 10):
            variants["{}, cost {}".format(label, cost)] = score_all(
                {n: decode_match(n, q, cost=cost) for n, q in p.items()})
    table("exp 4+5 combined", variants, "all 12, reference model, cost 6")


def exp_curve_no_sc():
    """Learning curve on Metrica + IDSSE only: would the 3 IDSSE matches we skipped help?"""
    rng = np.random.default_rng(0)
    pool = [m for m in matches() if no_sc(m[0])]
    print("\nlearning curve without SkillCorner (held-out spellF1, mean of 7, 2 random draws)")
    for size in (2, 4, 6):
        scores = []
        for draw in range(2):
            p = {}
            for name, arrays in labelled():
                others = [m for m in pool if m[0] != name]
                pick = {others[i][0] for i in rng.choice(len(others), size, replace=False)}
                p.update(predictions_for(name, arrays, [m for m in others if m[0] in pick]))
            scores.append(np.mean([v[0] for v in score_predictions(p).values()]))
        print("  {} training matches   {:.3f}".format(size, np.mean(scores)))


def predictions_for(name, arrays, train):
    parts = [train_rows(n, a) for n, a in train]
    clf = default_model().fit(np.concatenate([q[0] for q in parts]),
                              np.concatenate([q[1] for q in parts]))
    h = held_out(name, arrays)
    p = np.full(len(h["X0"]), np.nan)
    inside = ~np.isnan(h["X0"]).all(axis=1)
    p[inside] = 0.5 * (clf.predict_proba(h["X0"][inside])[:, 1]
                       + 1 - clf.predict_proba(h["X1"][inside])[:, 1])
    return {name: p}


def restart_accuracy(p):
    """Share of live stretches whose first possession has the right team (vs the truth)."""
    hits = total = 0
    for name, probs in p.items():
        arrays = dict(labelled())[name]
        gt = held_out(name, arrays)["gt"]
        codes = decode_match(name, probs)
        live = np.isin(codes, (pm.H, pm.A))
        for a, b in dead_ball.stretches(live, arrays["runs"]):
            truth = gt[a:min(b, a + 50)]
            truth = truth[np.isin(truth, (pm.H, pm.A))]
            if len(truth):
                total += 1
                hits += int(codes[a] == np.bincount(truth).argmax())
    return hits / max(total, 1), total


def exp_round2():
    names = held_out(*labelled()[0])["names"]
    without = [f for f in names if group_of(f) != "restart"]
    p_with, p_without = predictions(), predictions(columns=without)
    for label, p in (("without restart features", p_without), ("with restart features", p_with)):
        acc, n = restart_accuracy(p)
        print("  first possession after a stoppage right: {:.3f} of {} ({})".format(acc, n, label))
    variants = {"single model (ref)": score_predictions(p_with),
                "ensemble of 5 (all frames)": score_predictions(
                    predictions(offsets=(0, 1, 2, 3, 4))),
                "IDSSE counted double": score_predictions(
                    predictions(weight=lambda n: 2.0 if n.startswith("idsse") else 1.0)),
                "Metrica counted double": score_predictions(
                    predictions(weight=lambda n: 2.0 if n.startswith("metrica") else 1.0))}
    table("round 2 -- ensembles and source weights", variants, "single model (ref)")


def exp_selftrain(conf_levels=(0.7, 0.85), weights=(0.5, 1.0), rounds=2):
    """Self-training, simulated honestly: each held-out match is treated like our match --
    unlabelled -- and the model retrains on its own confident, smoothed guesses about it."""
    base = predictions()
    variants = {"no self-training (ref)": score_predictions(base)}
    for conf in conf_levels:
        for w in weights:
            out = {}
            for name, arrays in labelled():
                train = [(n, a) for n, a in matches() if n != name]
                parts = [train_rows(n, a) for n, a in train]
                X = np.concatenate([q[0] for q in parts])
                y = np.concatenate([q[1] for q in parts])
                h = held_out(name, arrays)
                inside = ~np.isnan(h["X0"]).all(axis=1)
                p = base[name]
                for _ in range(rounds):
                    codes = decode_match(name, p)
                    side = np.where(codes == pm.H, 0, np.where(codes == pm.A, 1, -1))
                    sure = (side >= 0) & inside & (np.maximum(p, 1 - p) >= conf) & \
                           ((p >= 0.5) == (side == 0))
                    idx = np.flatnonzero(sure)[::M.SUBSAMPLE]
                    Xp = np.concatenate([h["X0"][idx], h["X1"][idx]])
                    yp = np.concatenate([(side[idx] == 0), (side[idx] == 1)]).astype(np.int8)
                    clf = default_model().fit(
                        np.concatenate([X, Xp]), np.concatenate([y, yp]),
                        sample_weight=np.concatenate([np.ones(len(y)), np.full(len(yp), w)]))
                    p = np.full(len(h["X0"]), np.nan)
                    p[inside] = 0.5 * (clf.predict_proba(h["X0"][inside])[:, 1]
                                       + 1 - clf.predict_proba(h["X1"][inside])[:, 1])
                out[name] = p
            variants["self-train: confidence {} weight {}, {} rounds".format(conf, w, rounds)] = \
                score_predictions(out)
            print("  self-training conf {} w {} done".format(conf, w), flush=True)
    table("4.4 -- self-training on the (unlabelled) held-out match", variants,
          "no self-training (ref)")


def exp_budget(p):
    """Where are the remaining points? Replace one stage at a time with the true answer."""
    ours, oracle_dead, oracle_team, oracle_both = {}, {}, {}, {}
    for name, probs in p.items():
        arrays = dict(labelled())[name]
        gt = held_out(name, arrays)["gt"]
        runs = arrays["runs"]
        codes = decode_match(name, probs)
        ours[name] = codes
        # Perfect dead/alive, our team decision (Viterbi on each true live stretch).
        c = gt.copy()
        live = np.isin(gt, (pm.H, pm.A))
        for start, stop in dead_ball.stretches(live, runs):
            path = decode.viterbi(probs[start:stop], decode.SWITCH_COST)
            c[start:stop] = np.where(path == 0, pm.H, pm.A)
        oracle_dead[name] = c
        # Our dead/alive, perfect team wherever the truth has one.
        c = codes.copy()
        mask = np.isin(c, (pm.H, pm.A)) & np.isin(gt, (pm.H, pm.A))
        c[mask] = gt[mask]
        oracle_team[name] = c
        oracle_both[name] = gt
    table("error budget -- each stage replaced by the truth", {
        "ours": score_all(ours),
        "perfect dead-ball, our team": score_all(oracle_dead),
        "our dead-ball, perfect team": score_all(oracle_team),
    }, "ours")


# -- label granularity -------------------------------------------------------------------------

def granularity(codes):
    """Possession statistics of one label sequence, scored the way the metric sees it (runs from
    the non-X frames): possessions and direct H<->A turnovers per live minute, median possession
    length (s), share of possessions under 1 s and under 3 s."""
    runs = pm.valid_runs(codes)
    segs = pm.segments(codes, runs)
    poss = np.array([(b - a) / M.dead_ball.FPS for a, b, k in segs if k in (pm.H, pm.A)])
    live_min = poss.sum() / 60.0
    direct = sum(1 for (_, b1, k1), (a2, _, k2) in zip(segs, segs[1:])
                 if b1 == a2 and {k1, k2} == {pm.H, pm.A})
    return (len(poss) / live_min, float(np.median(poss)), float(np.mean(poss < 1)),
            float(np.mean(poss < 3)), direct / live_min, len(poss))


def exp_labels():
    """Do the training labels cut possessions as finely as the public ground truth does?"""
    rows = [(name.rstrip("0123456789").split("_")[0], name, granularity(M.truth_codes(arrays)))
            for name, arrays in labelled()]
    public = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    rows.append(("public", "BAR-ATH 5 min", granularity(pm.label_codes(pm.read_label_csv(public)))))
    print("{:<8} {:<16} {:>9} {:>9} {:>7} {:>7} {:>10} {:>6}".format(
        "source", "match", "poss/min", "median s", "<1 s", "<3 s", "direct/min", "poss"))
    for source in ("metrica", "idsse", "public"):
        group = [r for r in rows if r[0] == source]
        for _, name, g in group:
            print("{:<8} {:<16} {:>9.2f} {:>9.1f} {:>7.2f} {:>7.2f} {:>10.2f} {:>6d}".format(
                source, name, *g))
        if len(group) > 1:
            mean = np.mean([g[:5] for _, _, g in group], axis=0)
            print("{:<8} {:<16} {:>9.2f} {:>9.1f} {:>7.2f} {:>7.2f} {:>10.2f}".format(
                source, "mean", *mean))


if __name__ == "__main__":
    which = sys.argv[1:] or ["timing", "decoding"]
    if "labels" in which:
        exp_labels()
    if "features" in which:
        exp_features()
    if "mix" in which:
        exp_mix()
    if "models" in which:
        exp_models()
    if "augment" in which:
        exp_augment()
    if "restart" in which:
        exp_restart()
    if "round2" in which:
        exp_round2()
    if "selftrain" in which:
        exp_selftrain()
    if "combo" in which:
        exp_combo()
    if "curve" in which:
        exp_curve_no_sc()
    if "stacking" in which:
        exp_stacking(use=no_sc)
    base = predictions() if set(which) & {"timing", "decoding", "budget"} else None
    if "timing" in which:
        exp_timing(base)
    if "decoding" in which:
        exp_decoding(base)
    if "budget" in which:
        exp_budget(base)
