"""Round 2 of leave-one-game-out on the 10 external matches (3 Metrica + 7 IDSSE).

    python -m solution.logo2 prep       # fit the fold models once, cache held-out probabilities
    python -m solution.logo2 ref        # the reference under the three held-out regimes
    python -m solution.logo2 costs      # 1: joint SWITCH x STOP x RESTART grid
    python -m solution.logo2 evidence   # 2: switch cost that depends on the evidence per frame
    python -m solution.logo2 flip       # 2b: the flip family on a wider grid, with dead7 and T_dead
    python -m solution.logo2 final      # the candidates side by side, plus position-noise checks
    python -m solution.logo2 temp       # 3: temperature on P(home) and P(dead)
    python -m solution.logo2 dead7      # 4: P(dead) trained on 3 Metrica + 7 IDSSE
    python -m solution.logo2 stack      # 5: second stage over the temporal context of the probabilities
    python -m solution.logo2 deadgbm    # 6: GBM settings of the dead model
    python -m solution.logo2 smalldead  # 6b: the small dead model on top of the proposal
    python -m solution.logo2 mpm        # 7: posterior (max-marginal) decoding instead of Viterbi

Same rules as solution/logo.py: nothing here reads the competition match, its tracking or any
file outside data/external/cache; every held-out match is scored with the official library by
models that never saw it. Differences from round 1 (docs/logo_external.md, "Krug 2"):

  * the reference is the submission default: team model + learned P(dead) + decode.viterbi3;
  * every variant and the reference are scored on the held-out match degraded with TWO seeds
    (seed_of + 1, seed_of + 2); means are over the 20 (match, seed) pairs, wins are counted per
    match on the mean of its two seeds;
  * three held-out regimes: "default", "harsh" and "lag" (logo.lagged, 3 s);
  * adoption: default gain >= ADOPT_GAIN and >= ADOPT_WINS wins, harsh and lag not below
    -ROBUST_DROP; a hyperparameter chosen on held-out scores is judged on the nested estimate.

All decoding experiments work on the probabilities cached by `prep`, so they take seconds.
No default of the submission is changed by this module.
"""

import collections
import itertools
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import dead_ball, dead_features, decode, features  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import model as M  # noqa: E402

pm = M.pm

# Params
CACHE_DIR = os.path.join(os.path.dirname(L.CACHE_DIR), "logo2")
FPS = 25
SEEDS = (1, 2)                       # held-out degradation seeds: seed_of(name) + k
SPECS = collections.OrderedDict((("default", ("default",)), ("harsh", ("harsh",)),
                                 ("lag", ("lag", 3.0))))
# Extra safety checks outside the adoption rule: smooth position noise of round 1 (logo.NOISE).
ALL_SPECS = collections.OrderedDict(SPECS, noise=("noise", "mid", ("default",)),
                                    noisehigh=("noise", "high", ("default",)))
ADOPT_GAIN, ADOPT_WINS, ROBUST_DROP = 0.008, 7, 0.005   # fixed before any measurement
REF_COSTS = (decode.SWITCH_COST, decode.STOP_COST, decode.RESTART_COST)
EVIDENCE = ("d1_us", "d1_them", "ball_speed", "since_seen_s", "until_seen_s")
SEVEN = L.REF._replace(idsse="7")
POOL_WORKERS = 6


# -- cached held-out probabilities --------------------------------------------------------------

def held_path(name, spec, k):
    return os.path.join(CACHE_DIR, "held_{}_{}_{}.npz".format(name, spec, k))


def held_arrays(name, spec, k):
    """Degraded held-out copy of a match: regime SPECS[spec], seed seed_of(name) + k."""
    return L.degrade(name, ALL_SPECS[spec], M.seed_of(name) + k)


def team_proba(clf, held, runs):
    """(P(home), home-view feature matrix, its column names)."""
    names_, X0 = features.compute(held, runs, swap=False)
    _, X1 = features.compute(held, runs, swap=True)
    inside = ~np.isnan(X0).all(axis=1)
    p = np.full(len(X0), np.nan)
    p[inside] = 0.5 * (clf.predict_proba(X0[inside])[:, 1]
                       + 1 - clf.predict_proba(X1[inside])[:, 1])
    return p, X0, names_


def dead_proba(clf, X):
    p = np.full(len(X), np.nan)
    inside = ~np.isnan(X).all(axis=1)
    p[inside] = clf.predict_proba(X[inside])[:, 1]
    return p


def prep():
    """One team model and two dead models (4 / 7 IDSSE) per fold; their probabilities on the
    held-out match under every regime and seed, the forced-dead mask and a few evidence columns."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name in L.names():
        todo = [(s, k) for s in ALL_SPECS for k in SEEDS
                if not os.path.isfile(held_path(name, s, k))]
        if not todo:
            continue
        runs = L.data()[name]["runs"]
        team = L.fitted("team", L.train_names(name, "4"), L.REF)
        dead4 = L.fitted("dead", L.train_names(name, "4"), L.REF)
        dead7 = L.fitted("dead", L.train_names(name, "7"), SEVEN)
        for spec, k in todo:
            held = held_arrays(name, spec, k)
            ph, X0, cols = team_proba(team, held, runs)
            _, XD = dead_features.compute(held, runs)
            out = dict(ph=ph, pd=dead_proba(dead4, XD), pd7=dead_proba(dead7, XD),
                       forced=dead_ball.pre_kickoff_mask(held, runs))
            out.update({c: X0[:, cols.index(c)] for c in EVIDENCE})
            np.savez_compressed(held_path(name, spec, k), **{
                key: (v if v.dtype == bool else v.astype(np.float32)) for key, v in out.items()})
        L._MODELS.clear()
        print("prep", name, flush=True)


_HELD = {}


def held(name, spec, k):
    """Cached arrays of one (match, regime, seed) as float64 (forced: bool)."""
    key = (name, spec, k)
    if key not in _HELD:
        with np.load(held_path(name, spec, k)) as z:
            _HELD[key] = {f: (z[f] if z[f].dtype == bool else z[f].astype(np.float64))
                          for f in z.files}
    return _HELD[key]


# -- decoding and scoring -----------------------------------------------------------------------

_GT = {}


def truth(name):
    if name not in _GT:
        gt = L.truth(name)
        _GT[name] = gt, pm.valid_runs(gt)
    return _GT[name]


def codes_from(name, path_fn):
    """Codes for a whole match from path_fn(start, stop) -> 0/1/2 per frame of one run."""
    n = len(L.data()[name]["owner"])
    codes = np.full(n, pm.X, dtype=np.int8)
    letters = np.array([pm.H, pm.A, pm.D], dtype=np.int8)
    for start, stop in L.data()[name]["runs"]:
        codes[start:stop] = letters[path_fn(int(start), int(stop))]
    return codes


def decode3(name, ph, pdead, forced, costs=REF_COSTS):
    """Exactly model.labels_from3, on cached probabilities."""
    return codes_from(name, lambda a, b: decode.viterbi3(pdead[a:b], ph[a:b], forced[a:b], *costs))


def score(name, codes, full=False):
    """(spellF1, macroF1, predicted possessions, true possessions); macroF1 only when full."""
    gt, runs = truth(name)
    if full:
        r = pm.evaluate(gt, codes)
        return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]
    r = pm.spell_scores(gt, codes, runs)   # what pm.evaluate calls (no abstentions here)
    return r["mean_f1"], np.nan, r["n_pred"], r["n_gt"]


def run(make_codes, spec="default", full=False):
    """[10, 2 seeds, 4] scores; make_codes(name, h) -> codes, h = held(name, spec, k)."""
    return np.array([[score(n, make_codes(n, held(n, spec, k)), full) for k in SEEDS]
                     for n in L.names()], dtype=float)


def reference(spec="default", full=False):
    return run(lambda n, h: decode3(n, h["ph"], h["pd"], h["forced"]), spec, full)


def per_match(r):
    """[10] spellF1 per match, mean of its seeds."""
    return r[:, :, 0].mean(1)


def nested(options):
    """options {label: [10, seeds, 4]} on the default regime: for each match the label with the
    best mean spellF1 on the other 9. Returns the list of 10 picks."""
    labels = list(options)
    S = np.array([per_match(options[k]) for k in labels])
    return [labels[int(np.argmax(np.delete(S, i, axis=1).mean(1)))] for i in range(S.shape[1])]


def apply_picks(options, picks):
    return np.array([options[k][i] for i, k in enumerate(picks)])


def verdict(gain, wins, robust):
    """robust: {regime: gain} for harsh and lag (None when not measured)."""
    if gain <= 0:
        return "odbaci"
    if gain >= ADOPT_GAIN and wins >= ADOPT_WINS:
        if robust is None:
            return "možda (robusnost nije mjerena)"
        if all(robust[k] >= -ROBUST_DROP for k in ("harsh", "lag")):
            return "usvoji"
    return "možda"


def row(label, r, ref, robust=None, is_ref=False):
    met = np.array([n in L.METRICA for n in L.names()])
    s = per_match(r)
    head = "| {} | {:.4f} | {:.3f} | {:.3f} | {:.0f} / {:.0f} ".format(
        label, s.mean(), s[met].mean(), s[~met].mean(), r[:, :, 2].mean(), r[:, :, 3].mean())
    if is_ref:
        return head + "| referenca | | | | |"
    gain = s.mean() - per_match(ref).mean()
    wins = int((s > per_match(ref) + 1e-9).sum())
    rob = ["{:+.4f}".format(robust[k]) if robust and k in robust else "–" for k in ("harsh", "lag")]
    return head + "| {:+.4f} | {}/10 | {} | {} | {} |".format(gain, wins, rob[0], rob[1],
                                                           verdict(gain, wins, robust))


HEADER = ("| varijanta | spellF1 default (20) | Metrica | IDSSE | posjeda pred./istina | Δ | pobjede "
          "| Δ harsh | Δ lag | odluka |\n|---|---|---|---|---|---|---|---|---|---|")


def table(title, rows_):
    print("\n### {}\n\n{}".format(title, HEADER))
    for line in rows_:
        print(line)
    sys.stdout.flush()


def robust_gain(make_codes, refs):
    return {s: per_match(run(make_codes, s)).mean() - per_match(refs[s]).mean()
            for s in ("harsh", "lag")}


_POOL = {}


def _pool_run(i):
    return run(_POOL["make"](_POOL["grid"][i]))


def pooled(grid, make):
    """{setting: default-regime scores}; forked workers when the grid is large (same numbers)."""
    if len(grid) < 150:
        return {g: run(make(g)) for g in grid}
    import multiprocessing
    for n in L.names():
        truth(n)
        for k in SEEDS:
            held(n, "default", k)
    _POOL.update(make=make, grid=list(grid))
    with multiprocessing.get_context("fork").Pool(POOL_WORKERS) as pool:
        return dict(zip(grid, pool.map(_pool_run, range(len(grid)), chunksize=4)))


def study(title, grid, make, top=6, extra=()):
    """grid: list of hashable settings; make(setting) -> make_codes. Scores every setting on
    default, prints the best `top` (with harsh / lag) and the nested choice. Returns the scores."""
    refs = {s: reference(s) for s in SPECS}
    scores = pooled(grid, make)
    order = sorted(grid, key=lambda g: -per_match(scores[g]).mean())
    rows_ = [row("referenca", refs["default"], refs["default"], is_ref=True)]
    shown = order[:top] + [g for g in extra if g in scores and g not in order[:top]]
    for g in shown:
        rows_.append(row(str(g), scores[g], refs["default"], robust_gain(make(g), refs)))
    picks = nested(scores)
    nest = apply_picks(scores, picks)
    rob = {}
    for s in ("harsh", "lag"):
        cache = {}
        vals = []
        for i, (n, g) in enumerate(zip(L.names(), picks)):
            if g not in cache:
                cache[g] = run(make(g), s)
            vals.append(cache[g][i, :, 0].mean())
        rob[s] = float(np.mean(vals)) - per_match(refs[s]).mean()
    rows_.append(row("ugniježđeni izbor", nest, refs["default"], rob))
    table(title, rows_)
    print("\nugniježđeni izbori:", dict(collections.Counter(str(p) for p in picks)))
    print("mreža: {} postavki; default spellF1 min {:.4f} / medijan {:.4f} / max {:.4f}".format(
        len(grid), *np.percentile([per_match(scores[g]).mean() for g in grid], [0, 50, 100])))
    sys.stdout.flush()
    return scores


# -- experiments --------------------------------------------------------------------------------

def exp_ref():
    rows_ = []
    for s in SPECS:
        r = reference(s, full=True)
        met = np.array([n in L.METRICA for n in L.names()])
        sp = per_match(r)
        rows_.append("| {} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} / {:.0f} | {:.4f} / {:.4f} |".format(
            s, sp.mean(), sp[met].mean(), sp[~met].mean(), r[:, :, 1].mean(), r[:, :, 2].mean(),
            r[:, :, 3].mean(), r[:, 0, 0].mean(), r[:, 1, 0].mean()))
    print("\n### Referenca (2 seeda)\n\n| regim | spellF1 (20) | Metrica | IDSSE | macroF1 | posjeda "
          "pred./istina | seed +1 / seed +2 |\n|---|---|---|---|---|---|---|")
    print("\n".join(rows_))
    r = reference("default")
    print("\n| utakmica | seed +1 | seed +2 |\n|---|---|---|")
    for i, n in enumerate(L.names()):
        print("| {} | {:.3f} | {:.3f} |".format(n, r[i, 0, 0], r[i, 1, 0]))


COST_GRID = [(sw, st, rs) for sw in (8, 10, 12, 14, 17, 20) for st in (8, 12, 20, 30)
             for rs in (2, 4, 8, 12, 20)]


def exp_costs():
    study("1. Zajednička mreža SWITCH × STOP × RESTART", COST_GRID,
          lambda c: (lambda n, h: decode3(n, h["ph"], h["pd"], h["forced"], c)), top=8)


# -- 2: switch cost that depends on the evidence ------------------------------------------------

def viterbi3_fw(p_dead, p_home, forced_dead, switch_cost, stop_cost, restart_cost):
    """decode.viterbi3 with one switch cost per frame (the cost of switching INTO that frame)."""
    n = len(p_dead)
    pd_ = np.clip(np.nan_to_num(np.asarray(p_dead, dtype=float), nan=0.5), decode.EPS, 1 - decode.EPS)
    ph = np.clip(np.nan_to_num(np.asarray(p_home, dtype=float), nan=0.5), decode.EPS, 1 - decode.EPS)
    live = np.where(np.asarray(forced_dead, dtype=bool), -np.inf, np.log(1 - pd_))
    e_h, e_a = (live + np.log(ph)).tolist(), (live + np.log(1 - ph)).tolist()
    e_d = np.log(pd_).tolist()
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


def side_means(p, w):
    """(mean of p over the w frames before t, mean over the w frames from t), edges shortened."""
    cs = np.concatenate(([0.0], np.cumsum(p)))
    idx = np.arange(len(p))
    lo, hi = np.maximum(idx - w, 0), np.minimum(idx + w, len(p))
    past = (cs[idx] - cs[lo]) / np.maximum(idx - lo, 1)
    nxt = (cs[hi] - cs[idx]) / np.maximum(hi - idx, 1)
    return past, nxt


def jump(ph, w):
    """|mean P(home) over the next w frames - mean over the last w|: 1 at a clean change."""
    past, nxt = side_means(np.nan_to_num(ph, nan=0.5), w)
    return np.abs(nxt - past)


def flips(h, radius_m, w):
    """Frames within w of a change of the team nearest the ball, the new one within radius_m."""
    us, them = h["d1_us"], h["d1_them"]
    with np.errstate(invalid="ignore"):
        side = np.where(np.isnan(us) | np.isnan(them), -1, (them > us).astype(int))
        change = (side[1:] != side[:-1]) & (side[1:] >= 0) & (side[:-1] >= 0)
        change &= np.minimum(us, them)[1:] <= radius_m
    idx = np.flatnonzero(change) + 1
    mark = np.zeros(len(us) + 1)
    np.add.at(mark, np.maximum(idx - w, 0), 1)
    np.add.at(mark, np.minimum(idx + w + 1, len(us)), -1)
    return np.cumsum(mark[:-1]) > 0


def frame_cost(h, setting):
    """Per-frame switch cost. setting = (kind, base, *params):
    ("jump", S, a, w): S * max(1 - a * jump, 0.1); ("flip", S, r, radius, w): S * r near a flip;
    ("gap", S, g): S * g where the ball is missing; ("jumpgap", S, a, w, g): both."""
    kind, base = setting[0], float(setting[1])
    cost = np.full(len(h["ph"]), base)
    if kind in ("jump", "jumpgap"):
        cost *= np.maximum(1 - setting[2] * jump(h["ph"], setting[3]), 0.1)
    if kind == "flip":
        cost[flips(h, setting[3], setting[4])] *= setting[2]
    if kind in ("gap", "jumpgap"):
        cost[np.isnan(h["d1_us"]) & np.isnan(h["d1_them"])] *= setting[-1]
    return cost


def decode_fw(name, h, setting, ph=None, pdead=None):
    ph = h["ph"] if ph is None else ph
    pdead = h["pd"] if pdead is None else pdead
    cost = frame_cost(h, setting)
    return codes_from(name, lambda a, b: viterbi3_fw(pdead[a:b], ph[a:b], h["forced"][a:b],
                                                     cost[a:b], *REF_COSTS[1:]))


EVIDENCE_GRID = (
    [("jump", S, a, w) for S in (14, 17, 20, 25) for a in (0.4, 0.6, 0.8, 1.0) for w in (6, 12, 25)]
    + [("flip", S, r, R, w) for S in (14, 17, 20, 25) for r in (0.3, 0.5, 0.7) for R in (1.5, 3.0)
       for w in (2, 6)]
    + [("gap", S, g) for S in (12, 14, 17) for g in (0.5, 1.5, 2.0, 4.0)])


def exp_evidence():
    scores = study("2. Cijena prijelaza ovisna o dokazu", EVIDENCE_GRID,
                   lambda g: (lambda n, h: decode_fw(n, h, g)), top=8)
    for kind in ("jump", "flip", "gap"):
        sub = {g: v for g, v in scores.items() if g[0] == kind}
        best = max(sub, key=lambda g: per_match(sub[g]).mean())
        print("najbolje u obitelji {}: {} {:.4f}".format(kind, best, per_match(sub[best]).mean()))


FLIP_GRID = [("flip", S, r, R, w, pdkey, T) for S in (17, 20, 25, 30) for r in (0.1, 0.2, 0.3, 0.4)
             for R in (1.0, 1.5, 2.0) for w in (6, 10, 15) for pdkey in ("pd", "pd7")
             for T in (1.0, 2.0)]


def flip_codes(n, h, g):
    """g = ("flip", S, r, radius, w, which P(dead), T_dead)."""
    pdead = h[g[5]] if g[6] == 1.0 else tempered(h[g[5]], g[6])
    return decode_fw(n, h, g[:5], pdead=pdead)


def exp_flip():
    """The flip family on a wider grid, jointly with the dead model's training set and T_dead."""
    scores = study("2b. Cijena prijelaza niža oko promjene najbliže momčadi: šira mreža, zajedno s "
                   "P(dead) na 7 IDSSE i T_dead", FLIP_GRID,
                   lambda g: (lambda n, h: flip_codes(n, h, g)), top=10)
    for pdkey in ("pd", "pd7"):
        for T in (1.0, 2.0):
            sub = {g: v for g, v in scores.items() if g[5] == pdkey and g[6] == T}
            best = max(sub, key=lambda g: per_match(sub[g]).mean())
            picks = nested(sub)
            print("{} T_dead {:g}: najbolje {} {:.4f}; ugniježđeno {:.4f}, izbori {}".format(
                pdkey, T, best, per_match(sub[best]).mean(),
                per_match(apply_picks(sub, picks)).mean(),
                dict(collections.Counter(map(str, picks)))))
    with open(os.path.join(CACHE_DIR, "flip_grid.json"), "w") as handle:
        json.dump({repr(g): per_match(v).tolist() for g, v in scores.items()}, handle)


TOP_FLIP = ("flip", 20, 0.3, 2.0, 10)    # the best default mean on all 10
BEST_FLIP = ("flip", 20, 0.4, 2.0, 10)   # second by 0.0001, a weaker discount: the proposal


def exp_final():
    """The candidates side by side: each piece alone and stacked, fixed settings (the best on
    all 10, optimistic) and the nested choice inside the same sub-grid; then macroF1 and the two
    position-noise checks, which are outside the adoption rule."""
    import ast
    with open(os.path.join(CACHE_DIR, "flip_grid.json")) as handle:
        grid = {ast.literal_eval(k): np.array(v) for k, v in json.load(handle).items()}
    refs = {s: reference(s, full=True) for s in ALL_SPECS}
    names_ = L.names()

    def nested_picks(pdkey, T):
        sub = [g for g in grid if g[5] == pdkey and g[6] == T]
        S = np.array([grid[g] for g in sub])
        return [sub[int(np.argmax(np.delete(S, i, axis=1).mean(1)))] for i in range(len(names_))]

    off = ("flip", decode.SWITCH_COST, 1.0, 2.0, 10)
    variants = collections.OrderedDict((
        ("samo P(dead) na 7 IDSSE", [off + ("pd7", 1.0)] * 10),
        ("samo T_dead 2", [off + ("pd", 2.0)] * 10),
        ("flip {} (fiksno)".format(BEST_FLIP[1:]), [BEST_FLIP + ("pd", 1.0)] * 10),
        ("flip, ugniježđeni izbor", nested_picks("pd", 1.0)),
        ("flip + dead7 (fiksno)", [BEST_FLIP + ("pd7", 1.0)] * 10),
        ("flip + dead7, ugniježđeni izbor", nested_picks("pd7", 1.0)),
        ("flip + dead7 + T_dead 2 (fiksno, prijedlog)", [BEST_FLIP + ("pd7", 2.0)] * 10),
        ("flip {} + dead7 + T_dead 2 (fiksno, najbolji na svih 10)".format(TOP_FLIP[1:]),
         [TOP_FLIP + ("pd7", 2.0)] * 10),
        ("flip + dead7 + T_dead 2, ugniježđeni izbor", nested_picks("pd7", 2.0)),
    ))
    memo = {}

    def scores(picks, spec):
        for g in set(picks):
            if (g, spec) not in memo:
                memo[(g, spec)] = run(lambda n, h: flip_codes(n, h, g), spec, full=True)
        return np.array([memo[(g, spec)][i] for i, g in enumerate(picks)])

    rows_ = [row("referenca", refs["default"], refs["default"], is_ref=True)]
    extra = ["| referenca | {:.4f} | | {:.4f} | {:.4f} | | | | |".format(
        refs["default"][:, :, 1].mean(), per_match(refs["noise"]).mean(),
        per_match(refs["noisehigh"]).mean())]
    for label, picks in variants.items():
        r = {s: scores(picks, s) for s in ALL_SPECS}
        gain = {s: per_match(r[s]).mean() - per_match(refs[s]).mean() for s in ALL_SPECS}
        rows_.append(row(label, r["default"], refs["default"], gain))
        wins = {s: int((per_match(r[s]) > per_match(refs[s]) + 1e-9).sum()) for s in ALL_SPECS}
        extra.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:.4f} | {:+.4f} ({}/10) | {:+.4f} ({}/10) "
                     "| {}/10 | {}/10 |".format(
                         label, r["default"][:, :, 1].mean(),
                         r["default"][:, :, 1].mean() - refs["default"][:, :, 1].mean(),
                         per_match(r["noise"]).mean(), per_match(r["noisehigh"]).mean(),
                         gain["noise"], wins["noise"], gain["noisehigh"], wins["noisehigh"],
                         wins["harsh"], wins["lag"]))
    table("Krug 2: kandidati", rows_)
    print("\n| varijanta | macroF1 default | Δ macroF1 | spellF1 šum mid | šum high | Δ šum mid "
          "(pobjede) | Δ šum high (pobjede) | pobjede harsh | pobjede lag |\n"
          "|---|---|---|---|---|---|---|---|---|")
    print("\n".join(extra))
    best = [BEST_FLIP + ("pd7", 2.0)] * 10
    r = scores(best, "default")
    print("\n| utakmica | referenca | flip + dead7 + T_dead 2 (fiksno) |\n|---|---|---|")
    for i, n in enumerate(names_):
        print("| {} | {:.3f} | {:.3f} |".format(n, per_match(refs["default"])[i], per_match(r)[i]))


# -- 3: temperature -----------------------------------------------------------------------------

def tempered(p, T):
    """sigmoid(logit(p) / T), NaN stays NaN; clipped as the decoder clips."""
    q = np.clip(p, decode.EPS, 1 - decode.EPS)
    return 1.0 / (1.0 + np.exp(-np.log(q / (1 - q)) / T))


TEMPS = (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0)


def exp_temp():
    study("3. Temperatura (T_home, T_dead)", list(itertools.product(TEMPS, TEMPS)),
          lambda g: (lambda n, h: decode3(n, tempered(h["ph"], g[0]), tempered(h["pd"], g[1]),
                                          h["forced"])), top=6)


# -- 4: dead model on more matches --------------------------------------------------------------

def exp_dead7():
    refs = {s: reference(s) for s in SPECS}
    mk = lambda n, h: decode3(n, h["ph"], h["pd7"], h["forced"])
    r = run(mk)
    table("4. P(dead) treniran na 3 Metrica + 7 IDSSE (team model ostaje na 4)", [
        row("referenca (dead na 4 IDSSE)", refs["default"], refs["default"], is_ref=True),
        row("dead na 7 IDSSE", r, refs["default"], robust_gain(mk, refs))])
    print("\n| utakmica | ref | dead7 |\n|---|---|---|")
    for i, n in enumerate(L.names()):
        print("| {} | {:.3f} | {:.3f} |".format(n, per_match(refs["default"])[i], per_match(r)[i]))


# -- 7: posterior decoding ----------------------------------------------------------------------

def mpm3(p_dead, p_home, forced_dead, switch_cost, stop_cost, restart_cost):
    """Max-posterior-marginal state per frame of the same chain as decode.viterbi3 (potentials
    exp(emission), exp(-cost)); scaled forward-backward in plain floats."""
    n = len(p_dead)
    pd_ = np.clip(np.nan_to_num(np.asarray(p_dead, dtype=float), nan=0.5), decode.EPS, 1 - decode.EPS)
    ph = np.clip(np.nan_to_num(np.asarray(p_home, dtype=float), nan=0.5), decode.EPS, 1 - decode.EPS)
    live = np.where(np.asarray(forced_dead, dtype=bool), 0.0, 1 - pd_)
    eh, ea, ed = (live * ph).tolist(), (live * (1 - ph)).tolist(), pd_.tolist()
    sw, st, rs = (float(np.exp(-c)) for c in (switch_cost, stop_cost, restart_cost))
    fh, fa, fd = [0.0] * n, [0.0] * n, [0.0] * n
    h, a, d = eh[0], ea[0], ed[0]
    z = h + a + d
    fh[0], fa[0], fd[0] = h / z, a / z, d / z
    for t in range(1, n):
        ph_, pa_, pd__ = fh[t - 1], fa[t - 1], fd[t - 1]
        h = (ph_ + pa_ * sw + pd__ * rs) * eh[t]
        a = (pa_ + ph_ * sw + pd__ * rs) * ea[t]
        d = (pd__ + (ph_ + pa_) * st) * ed[t]
        z = h + a + d
        fh[t], fa[t], fd[t] = h / z, a / z, d / z
    path = np.empty(n, dtype=np.int8)
    bh = ba = bd = 1.0
    for t in range(n - 1, -1, -1):
        gh, ga, gd = fh[t] * bh, fa[t] * ba, fd[t] * bd
        path[t] = 0 if (gh >= ga and gh >= gd) else (1 if ga >= gd else 2)
        xh, xa, xd = bh * eh[t], ba * ea[t], bd * ed[t]
        bh, ba, bd = xh + sw * xa + st * xd, xa + sw * xh + st * xd, xd + rs * (xh + xa)
        z = bh + ba + bd
        bh, ba, bd = bh / z, ba / z, bd / z
    return path


MPM_GRID = [(sw, st, rs) for sw in (6, 10, 14, 20) for st in (12, 20) for rs in (4, 8)]


def exp_mpm():
    study("7. Posteriorno dekodiranje (najveća marginalna vjerojatnost po frameu)", MPM_GRID,
          lambda c: (lambda n, h: codes_from(n, lambda a, b: mpm3(
              h["pd"][a:b], h["ph"][a:b], h["forced"][a:b], *c))), top=4)


# -- 6: GBM settings of the dead model ----------------------------------------------------------

DEAD_GBMS = collections.OrderedDict((
    ("200 stabala", dict(max_iter=200)),
    ("lr 0.05", dict(learning_rate=0.05)),
    ("min_samples_leaf 50", dict(min_samples_leaf=50)),
    ("min_samples_leaf 1000", dict(min_samples_leaf=1000)),
    ("15 listova", dict(max_leaf_nodes=15)),
    ("100 stabala, 15 listova", dict(max_iter=100, max_leaf_nodes=15)),
))
_XD = {}


def dead_matrix(name, spec, k):
    key = (name, spec, k)
    if key not in _XD:
        _XD[key] = dead_features.compute(held_arrays(name, spec, k), L.data()[name]["runs"])[1]
    return _XD[key]


def dead_variant(label, specs, idsse="4"):
    """{(name, spec, k): P(dead)} from a dead model with DEAD_GBMS[label], cached on disk."""
    path = os.path.join(CACHE_DIR, "deadgbm_{}{}.npz".format(
        label.replace(" ", "_").replace(",", ""), "" if idsse == "4" else "_idsse" + idsse))
    out = {}
    if os.path.isfile(path):
        with np.load(path) as z:
            out = {tuple(f.split("|")): z[f].astype(np.float64) for f in z.files}
    want = [(n, s, str(k)) for n in L.names() for s in specs for k in SEEDS]
    if any(w not in out for w in want):
        for n in L.names():
            if all((n, s, str(k)) in out for s in specs for k in SEEDS):
                continue
            parts = [L.dead_rows(t, L.REF) for t in L.train_names(n, idsse)]
            clf = HistGradientBoostingClassifier(**dict(M.GBM, **DEAD_GBMS[label])).fit(
                np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
            for s in specs:
                for k in SEEDS:
                    if (n, s, str(k)) not in out:
                        out[(n, s, str(k))] = dead_proba(clf, dead_matrix(n, s, k))
        np.savez_compressed(path, **{"|".join(key): v.astype(np.float32) for key, v in out.items()})
        with np.load(path) as z:
            out = {tuple(f.split("|")): z[f].astype(np.float64) for f in z.files}
    return out


def exp_deadgbm():
    refs = {s: reference(s) for s in SPECS}
    rows_ = [row("referenca ({})".format(", ".join("{} {}".format(k, v) for k, v in M.GBM.items()
                                                    if k != "random_state")),
                 refs["default"], refs["default"], is_ref=True)]
    for label in DEAD_GBMS:
        p = dead_variant(label, ["default"])
        r = np.array([[score(n, decode3(n, held(n, "default", k)["ph"], p[(n, "default", str(k))],
                                        held(n, "default", k)["forced"])) for k in SEEDS]
                      for n in L.names()], dtype=float)
        robust = None
        if per_match(r).mean() - per_match(refs["default"]).mean() > 0:
            p = dead_variant(label, list(SPECS))
            robust = {s: np.mean([[score(n, decode3(n, held(n, s, k)["ph"], p[(n, s, str(k))],
                                                    held(n, s, k)["forced"]))[0] for k in SEEDS]
                                  for n in L.names()]) - per_match(refs[s]).mean()
                      for s in ("harsh", "lag")}
        rows_.append(row(label, r, refs["default"], robust))
        print(rows_[-1], flush=True)
    table("6. GBM postavke dead modela (jedna po jedna promjena prema zadanima)", rows_)


# -- 5: second stage over the temporal context of the probabilities -----------------------------

STACK_WINDOWS_S = (0.4, 1.0, 3.0, 10.0)
STACK_GBM = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                 l2_regularization=1.0, random_state=0)
STACK_COSTS = [(sw, st, rs) for sw in (6, 10, 14, 20) for st in (12, 20) for rs in (4, 8)]


def oof_path(t, h):
    return os.path.join(CACHE_DIR, "oof_{}_without_{}.npz".format(t, h))


def stack_prep():
    """First-stage P(home), P(dead) on the TRAINING copy of every reference training match t,
    from models that saw neither t nor the fold's held-out match h (h = None: only t left out,
    for the folds whose held-out match is not a reference training match)."""
    ref7 = [n for n in L.names() if n in L.METRICA or n in L.REF_IDSSE]
    feats = {}

    def copy_of(t):
        if t not in feats:
            runs = L.data()[t]["runs"]
            aug = L.degrade(t, ("default",), M.seed_of(t))
            names_, X0 = features.compute(aug, runs, swap=False)
            _, X1 = features.compute(aug, runs, swap=True)
            feats[t] = X0, X1, dead_features.compute(aug, runs)[1], names_
        return feats[t]

    jobs = [((t,), None) for t in ref7] + [(pair, pair) for pair in itertools.combinations(ref7, 2)]
    for out_of, pair in jobs:
        targets = [(t, None) for t in out_of] if pair is None else [(pair[0], pair[1]),
                                                                    (pair[1], pair[0])]
        if all(os.path.isfile(oof_path(t, h)) for t, h in targets):
            continue
        train = tuple(n for n in ref7 if n not in out_of)
        team, dead = L.fitted("team", train, L.REF), L.fitted("dead", train, L.REF)
        for t, h in targets:
            X0, X1, XD, names_ = copy_of(t)
            inside = ~np.isnan(X0).all(axis=1)
            ph = np.full(len(X0), np.nan)
            ph[inside] = 0.5 * (team.predict_proba(X0[inside])[:, 1]
                                + 1 - team.predict_proba(X1[inside])[:, 1])
            np.savez_compressed(oof_path(t, h), ph=ph.astype(np.float32),
                                pd=dead_proba(dead, XD).astype(np.float32),
                                since=X0[:, names_.index("since_seen_s")],
                                until=X0[:, names_.index("until_seen_s")])
        L._MODELS.clear()
        print("stack prep", out_of, flush=True)


def stack_matrix(ph, pdead, since, until, kind, swap=False):
    """Second-stage features of one match. kind "team": P(us) = P(home) or 1 - P(home);
    kind "dead": the team-free certainty |2 P(home) - 1| in its place."""
    p = (1 - ph if swap else ph) if kind == "team" else np.abs(2 * ph - 1)
    cols = [p, pdead, since, until]
    for w in STACK_WINDOWS_S:
        frames = int(w * FPS)
        for direction in (-1, 1):
            cols.append(features._rolling_mean(p, frames, direction))
            cols.append(features._rolling_mean(pdead, frames, direction))
    return np.stack(cols, axis=1).astype(np.float32)


def stack_models(h):
    """(team, dead) second-stage models of the fold whose held-out match is h."""
    ref7 = [n for n in L.names() if n in L.METRICA or n in L.REF_IDSSE]
    xt, yt, xd, yd = [], [], [], []
    for t in (n for n in ref7 if n != h):
        with np.load(oof_path(t, h if h in ref7 else None)) as z:
            o = {f: z[f].astype(np.float64) for f in z.files}
        a = L.data()[t]
        inside = L.inside_mask(a)
        idx = np.flatnonzero(inside)[::M.SUBSAMPLE]
        xd.append(stack_matrix(o["ph"], o["pd"], o["since"], o["until"], "dead")[idx])
        yd.append(a["dead"][idx].astype(np.int8))
        idx = np.flatnonzero((a["owner"] >= 0) & ~a["dead"] & inside)[::M.SUBSAMPLE]
        for swap in (False, True):
            xt.append(stack_matrix(o["ph"], o["pd"], o["since"], o["until"], "team", swap)[idx])
            yt.append((a["owner"][idx] == (1 if swap else 0)).astype(np.int8))
    fit = lambda X, y: HistGradientBoostingClassifier(**STACK_GBM).fit(np.concatenate(X),
                                                                       np.concatenate(y))
    return fit(xt, yt), fit(xd, yd)


def stack_proba(spec):
    """{(name, k): (P(home), P(dead))} of the second stage on the held-out copies."""
    path = os.path.join(CACHE_DIR, "stack_{}.npz".format(spec))
    if not os.path.isfile(path):
        out = {}
        for n in L.names():
            team, dead = stack_models(n)
            for k in SEEDS:
                h = held(n, spec, k)
                args = (h["ph"], h["pd"], h["since_seen_s"], h["until_seen_s"])
                inside = ~np.isnan(h["pd"])
                ph, pdead = np.full(len(inside), np.nan), np.full(len(inside), np.nan)
                ph[inside] = 0.5 * (
                    team.predict_proba(stack_matrix(*args, "team", False)[inside])[:, 1]
                    + 1 - team.predict_proba(stack_matrix(*args, "team", True)[inside])[:, 1])
                pdead[inside] = dead.predict_proba(stack_matrix(*args, "dead")[inside])[:, 1]
                out["{}|{}|h".format(n, k)] = ph.astype(np.float32)
                out["{}|{}|d".format(n, k)] = pdead.astype(np.float32)
        np.savez_compressed(path, **out)
    with np.load(path) as z:
        return {(n, k): (z["{}|{}|h".format(n, k)].astype(np.float64),
                         z["{}|{}|d".format(n, k)].astype(np.float64))
                for n in L.names() for k in SEEDS}


def exp_stack():
    stack_prep()
    P = {s: stack_proba(s) for s in SPECS}
    refs = {s: reference(s) for s in SPECS}

    def codes(n, s, k, use, costs):
        h = held(n, s, k)
        ph = P[s][(n, k)][0] if "team" in use else h["ph"]
        pdead = P[s][(n, k)][1] if "dead" in use else h["pd"]
        return decode3(n, ph, pdead, h["forced"], costs)

    def scores(s, use, costs):
        return np.array([[score(n, codes(n, s, k, use, costs)) for k in SEEDS] for n in L.names()],
                        dtype=float)

    rows_ = [row("referenca", refs["default"], refs["default"], is_ref=True)]
    for use in (("team",), ("dead",), ("team", "dead")):
        tag = "drugi stupanj: " + " + ".join(use)
        r = scores("default", use, REF_COSTS)
        rob = {s: per_match(scores(s, use, REF_COSTS)).mean() - per_match(refs[s]).mean()
               for s in ("harsh", "lag")}
        rows_.append(row(tag + ", zadane cijene", r, refs["default"], rob))
        grid = {c: scores("default", use, c) for c in STACK_COSTS}
        best = max(grid, key=lambda c: per_match(grid[c]).mean())
        rob = {s: per_match(scores(s, use, best)).mean() - per_match(refs[s]).mean()
               for s in ("harsh", "lag")}
        rows_.append(row("{}, najbolje cijene {} (optimistično)".format(tag, best), grid[best],
                         refs["default"], rob))
        picks = nested(grid)
        rob = {}
        for s in ("harsh", "lag"):
            memo = {c: scores(s, use, c) for c in set(picks)}
            rob[s] = np.mean([memo[c][i, :, 0].mean() for i, c in enumerate(picks)]) \
                - per_match(refs[s]).mean()
        rows_.append(row(tag + ", ugniježđeni izbor cijena", apply_picks(grid, picks),
                         refs["default"], rob))
        print("ugniježđeni izbori ({}):".format(tag), dict(collections.Counter(map(str, picks))))
    table("5. Drugi stupanj nad vremenskim kontekstom vjerojatnosti", rows_)


def exp_smalldead():
    """Does the small dead model of experiment 6 add anything on top of the proposal?"""
    label = "100 stabala, 15 listova"
    specs = list(SPECS)
    refs = {s: reference(s) for s in specs}
    small = {i: dead_variant(label, specs, i) for i in ("4", "7")}

    def scores(s, flip, pdead, T):
        """pdead: "pd" / "pd7" (cached, zadani GBM) or ("small", idsse)."""
        out = []
        for n in L.names():
            out.append([])
            for k in SEEDS:
                h = held(n, s, k)
                p = h[pdead] if isinstance(pdead, str) else small[pdead[1]][(n, s, str(k))]
                p = p if T == 1.0 else tempered(p, T)
                out[-1].append(score(n, decode_fw(n, h, flip, pdead=p)))
        return np.array(out, dtype=float)

    off = ("flip", decode.SWITCH_COST, 1.0, 2.0, 10)
    variants = collections.OrderedDict((
        ("mali dead (4 IDSSE)", (off, ("small", "4"), 1.0)),
        ("mali dead (7 IDSSE)", (off, ("small", "7"), 1.0)),
        ("prijedlog: flip + dead7 + T_dead 2", (BEST_FLIP, "pd7", 2.0)),
        ("flip + mali dead (4 IDSSE)", (BEST_FLIP, ("small", "4"), 1.0)),
        ("flip + mali dead (7 IDSSE)", (BEST_FLIP, ("small", "7"), 1.0)),
        ("flip + mali dead (7 IDSSE) + T_dead 2", (BEST_FLIP, ("small", "7"), 2.0)),
    ))
    rows_ = [row("referenca", refs["default"], refs["default"], is_ref=True)]
    res = {}
    for name_, v in variants.items():
        res[name_] = {s: scores(s, *v) for s in specs}
        rows_.append(row(name_, res[name_]["default"], refs["default"],
                         {s: per_match(res[name_][s]).mean() - per_match(refs[s]).mean()
                          for s in ("harsh", "lag")}))
    table("6b. Mali dead model ({}) povrh prijedloga".format(label), rows_)
    base = res["prijedlog: flip + dead7 + T_dead 2"]
    for name_ in list(variants)[3:]:
        print("{} prema prijedlogu: ".format(name_) + ", ".join(
            "{} {:+.4f} ({}/10)".format(s, per_match(res[name_][s]).mean() - per_match(base[s]).mean(),
                                        int((per_match(res[name_][s]) > per_match(base[s]) + 1e-9).sum()))
            for s in specs))


EXPERIMENTS = collections.OrderedDict(prep=prep, ref=exp_ref, costs=exp_costs,
                                      evidence=exp_evidence, flip=exp_flip, final=exp_final, temp=exp_temp, dead7=exp_dead7,
                                      stack=exp_stack, deadgbm=exp_deadgbm, smalldead=exp_smalldead, mpm=exp_mpm)

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["ref"]
    unknown = [w for w in wanted if w not in EXPERIMENTS]
    if unknown:
        sys.exit("unknown experiment {}; choose from {}".format(unknown, ", ".join(EXPERIMENTS)))
    for w in wanted:
        EXPERIMENTS[w]()
