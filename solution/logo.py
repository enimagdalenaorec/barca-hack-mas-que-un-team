"""Leave-one-game-out on the 10 external matches only (3 Metrica + 7 IDSSE).

    python -m solution.logo ref        # the submission's default configuration
    python -m solution.logo idsse      # a: train on 4 vs 7 IDSSE matches
    python -m solution.logo dead       # b: dead-ball rule vs learned P(dead), realistic ball loss
    python -m solution.logo deadlag    # b2: the same when ball gaps lag the stoppages (stress test)
    python -m solution.logo lagtrain   # b3: training under that stress regime, normal held-out
    python -m solution.logo noise      # c: smooth position noise and more player drop-outs
    python -m solution.logo control    # d: controlled-possession training labels, minimum length
    python -m solution.logo switch     # e: Viterbi switch cost, nested choice
    python -m solution.logo ensemble   # f: average of models over augment seeds / frame offsets
    python -m solution.logo kickoff    # g: kick-off detection and the first possession
    python -m solution.logo combo      # the changes that passed, together
    python -m solution.logo all

Every held-out match is scored with the official library (evaluation/possession_metrics.py) by a
model that never saw it. Nothing here reads the competition match, its tracking or any file under
data/cache: the only inputs are data/external/cache/*.npz, and predictions are cached under
data/external/cache/logo/. The adoption rule and the results are in docs/logo_external.md.

The reference is the submission default: train on Metrica + the 4 IDSSE matches of
model.IDSSE_TRAIN (minus the held-out one), default augmentation, dead-ball rule, SWITCH_COST.
No default of the submission is changed by this module.
"""

import collections
import hashlib
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import augment, dead_ball, dead_features, decode, external_games, features  # noqa: E402
from solution import model as M  # noqa: E402

pm = M.pm

# Params
CACHE_DIR = os.path.join(external_games.CACHE_DIR, "logo")
FPS = 25
METRICA = ("metrica1", "metrica2", "metrica3")
REF_IDSSE = tuple("idsse_" + m for m in M.IDSSE_TRAIN)
ADOPT_GAIN, ADOPT_WINS = 0.01, 7     # docs/logo_external.md, fixed before any measurement
BALL_MISSING_TARGET = 0.41           # DATA.md: share of graded frames without the ball
TRACKED_TARGET = 20.5                # DATA.md: mean tracked players on a graded frame
SWITCH_GRID = (6, 8, 10, 12, 14, 17, 20, 25, 30, 40, 50, 60)
STOP_GRID = RESTART_GRID = (4, 8, 12, 20)
MIN_LEN_GRID = (0, 8, 12, 25, 38, 50, 75)
GAP_BUCKETS_S = (5.0, 15.0, 40.0)    # stoppages borrow a Metrica stoppage of similar length
NOISE_SMOOTH_S = 1.0                 # correlation time of the position noise
NOISE = {"mid": (0.25, 0.5, TRACKED_TARGET), "high": (0.5, 1.0, TRACKED_TARGET - 1.0)}

# How a model is trained. idsse: "4" (reference) or "7"; spec: degradation of the training matches;
# control_s: controlled-possession relabelling of TRAINING labels; offset: which interleaved frame
# set; shift: augment seed shift; max_features/rs: GBM column subsampling and its seed.
Train = collections.namedtuple("Train", "idsse spec control_s offset shift max_features rs",
                               defaults=("4", ("default",), None, 0, 0, 1.0, 0))
REF = Train()


# -- data --------------------------------------------------------------------------------------

_DATA = None


def data():
    """{name: arrays} of the 10 external matches, in a fixed order."""
    global _DATA
    if _DATA is None:
        saved, M.IDSSE_TRAIN = M.IDSSE_TRAIN, None
        try:
            _DATA = dict(M.training_matches(control_s=0))
        finally:
            M.IDSSE_TRAIN = saved
        assert all(int(a["has_dead"]) for a in _DATA.values())
    return _DATA


def names():
    return list(data())


def train_names(held, idsse):
    keep = [n for n in names() if n in METRICA or idsse == "7" or n in REF_IDSSE]
    return tuple(n for n in keep if n != held)


def inside_mask(arrays):
    mask = np.zeros(len(arrays["owner"]), dtype=bool)
    for start, stop in arrays["runs"]:
        mask[start:stop] = True
    return mask


# -- degradations ------------------------------------------------------------------------------

_GAPS = None


def gap_stats():
    """How Metrica's own tracker loses the ball, measured on the 3 sample games after the short
    gaps are bridged: per stoppage (length, frames before the ball vanishes, frames it is back
    before the restart, frames the gap starts before / ends after the stoppage), and the lengths
    of the gaps that lie wholly in live play."""
    global _GAPS
    if _GAPS is None:
        recs, live_gaps, summary = [], [], []
        for name in METRICA:
            a = data()[name]
            runs, dead = a["runs"], a["dead"]
            ball = augment._bridge(a["ball_xy"].astype(np.float64), runs,
                                   int(augment.BRIDGE_S * FPS))
            miss = np.isnan(ball[:, 0])
            for a0, b0 in runs:
                for s, e in dead_ball.stretches(dead, [(a0, b0)]):
                    idx = np.flatnonzero(miss[s:e])
                    if not len(idx):
                        recs.append((e - s, e - s, 0, 0, 0))
                        continue
                    lead, trail = int(idx[0]), int(e - s - 1 - idx[-1])
                    pre = post = 0
                    while lead == 0 and s - pre - 1 >= a0 and miss[s - pre - 1]:
                        pre += 1
                    while trail == 0 and e + post < b0 and miss[e + post]:
                        post += 1
                    recs.append((e - s, lead, trail, pre, post))
                for s, e in dead_ball.stretches(miss, [(a0, b0)]):
                    if not dead[max(s - 1, a0):min(e + 1, b0)].any():
                        live_gaps.append(e - s)
            ins = inside_mask(a)
            summary.append((miss[ins & dead].mean(), miss[ins & ~dead].mean(), miss[ins].mean()))
        _GAPS = dict(recs=np.array(recs), live=np.array(live_gaps), summary=np.array(summary))
    return _GAPS


def realistic(arrays, seed, natural):
    """Held-out or training copy whose ball is lost the way Metrica's tracker loses it.

    natural (Metrica): the tracker's own gaps stay. Otherwise (IDSSE, perfect ball): every
    stoppage borrows the gap of a random Metrica stoppage of similar length -- the ball stays
    visible for a while after play stops, comes back before the restart, sometimes never goes.
    Then live-play gaps with Metrica's live gap lengths are added until BALL_MISSING_TARGET of
    the graded frames have no ball. Players drop as in the default augmentation.
    """
    stats = gap_stats()
    rng = np.random.default_rng(seed)
    runs, dead = arrays["runs"], arrays["dead"]
    out = dict(arrays)
    ball = augment._bridge(arrays["ball_xy"].astype(np.float64), runs, int(augment.BRIDGE_S * FPS))
    if not natural:
        recs = stats["recs"]
        edges = np.array(GAP_BUCKETS_S) * FPS
        bucket = np.searchsorted(edges, recs[:, 0])
        for a0, b0 in runs:
            for s, e in dead_ball.stretches(dead, [(a0, b0)]):
                pool = recs[bucket == np.searchsorted(edges, e - s)]
                _, lead, trail, pre, post = pool[rng.integers(len(pool))]
                lo, hi = s + lead - pre, e - trail + post
                if s + lead < e - trail:
                    ball[max(lo, a0):min(hi, b0)] = np.nan
    ins = inside_mask(arrays)
    live_idx = np.flatnonzero(ins & ~dead)
    budget = int(BALL_MISSING_TARGET * ins.sum()) - int(np.isnan(ball[ins, 0]).sum())
    while budget > 0:
        length = int(rng.choice(stats["live"]))
        start = int(rng.choice(live_idx))
        budget -= int((~np.isnan(ball[start:start + length, 0])).sum())
        ball[start:start + length] = np.nan
    players = arrays["players_xy"].astype(np.float64).copy()
    n, slots = players.shape[:2]
    for slot in range(slots):
        left = int(augment.PLAYER_DROP_SHARE * n)
        while left > 0:
            length = int(rng.uniform(*augment.PLAYER_DROP_S) * FPS)
            start = int(rng.integers(0, n))
            players[start:start + length, slot] = np.nan
            left -= length
    out["ball_xy"], out["players_xy"] = ball.astype(np.float32), players.astype(np.float32)
    return out


def smooth_noise(rng, shape, sigma_m):
    """Time-correlated Gaussian noise (metres), standard deviation sigma_m, along axis 0."""
    noise = gaussian_filter1d(rng.standard_normal(shape), NOISE_SMOOTH_S * FPS, axis=0)
    return noise * (sigma_m / noise.std())


def noisy(arrays, seed, level):
    """Smooth position error on every player and the ball, and player drop-outs until the mean
    number tracked on graded frames is down to the level's target. NaN stays NaN."""
    sigma_players, sigma_ball, tracked_target = NOISE[level]
    rng = np.random.default_rng(seed)
    out = dict(arrays)
    players = arrays["players_xy"].astype(np.float64)
    players = players + smooth_noise(rng, players.shape, sigma_players)
    ball = arrays["ball_xy"].astype(np.float64)
    ball = ball + smooth_noise(rng, ball.shape, sigma_ball)
    graded = inside_mask(arrays)
    ins = np.flatnonzero(graded)
    seen = ~np.isnan(players[:, :, 0])
    excess = int((seen[ins].sum(1).mean() - tracked_target) * len(ins))
    slots = np.flatnonzero(seen[ins].any(0))
    while excess > 0:
        length = int(rng.uniform(*augment.PLAYER_DROP_S) * FPS)
        start, slot = int(rng.choice(ins)), int(rng.choice(slots))
        excess -= int((seen[start:start + length, slot] & graded[start:start + length]).sum())
        seen[start:start + length, slot] = False
    players[~seen] = np.nan
    out["ball_xy"], out["players_xy"] = ball.astype(np.float32), players.astype(np.float32)
    return out


def lagged(arrays, seed, lag_s):
    """Stress test, an ASSUMPTION and not a measurement: the default augmentation, except that
    the ball stays visible for up to lag_s after play stops and is back up to lag_s before the
    restart (uniform), so ball gaps no longer line up with the stoppages frame for frame."""
    out = augment.augment(arrays, seed, {"HIDE_DEAD_P": 0.0})
    rng = np.random.default_rng(seed + 13)
    ball = out["ball_xy"].copy()
    for s, e in dead_ball.stretches(arrays["dead"], arrays["runs"]):
        lead, trail = (int(rng.uniform(0, lag_s) * FPS) for _ in range(2))
        if rng.random() < augment.HIDE_DEAD_P and s + lead < e - trail:
            ball[s + lead:e - trail] = np.nan
    out["ball_xy"] = ball
    return out


def degrade(name, spec, seed):
    """Degraded copy of a match. spec: ("default",) / ("harsh",) are augment.py regimes,
    ("real",) is realistic(), ("lag", seconds) is lagged(), ("noise", level, base_spec) adds
    noisy() on top of base_spec."""
    arrays = data()[name]
    if spec[0] == "lag":
        return lagged(arrays, seed, spec[1])
    if spec[0] in ("default", "harsh"):
        return augment.augment(arrays, seed, augment.regime(spec[0]))
    if spec[0] == "real":
        return realistic(arrays, seed, natural=name in METRICA)
    if spec[0] == "noise":
        return noisy(degrade(name, spec[2], seed), seed + 7, spec[1])
    raise KeyError(spec)


_HELD = collections.OrderedDict()


def held_out(name, spec):
    """Degraded held-out copy (seed_of + 1, as model.evaluate) and its two team views."""
    key = (name, spec)
    if key not in _HELD:
        held = degrade(name, spec, M.seed_of(name) + 1)
        runs = data()[name]["runs"]
        _, X0 = features.compute(held, runs, swap=False)
        _, X1 = features.compute(held, runs, swap=True)
        _HELD[key] = dict(held=held, X0=X0, X1=X1)
        while len(_HELD) > 12:
            _HELD.popitem(last=False)
    _HELD.move_to_end(key)
    return _HELD[key]


_GT = {}


def truth(name):
    if name not in _GT:
        _GT[name] = M.truth_codes(data()[name])
    return _GT[name]


# -- models, cached ----------------------------------------------------------------------------

_ROWS, _DROWS, _MODELS = {}, {}, {}


def train_seed(name, shift):
    return M.seed_of(name) + 1000 * shift


def train_rows(name, tr):
    """As model.rows, plus the frame-set offset, the augment seed shift and other degradations."""
    key = (name, tr.spec, tr.control_s, tr.offset, tr.shift)
    if key not in _ROWS:
        arrays = data()[name]
        aug = degrade(name, tr.spec, train_seed(name, tr.shift))
        owner = arrays["owner"]
        if tr.control_s:
            owner = M.controlled(owner, arrays["dead"], arrays["runs"],
                                 int(round(tr.control_s * FPS)))
        idx = np.flatnonzero((owner >= 0) & ~arrays["dead"])[tr.offset::M.SUBSAMPLE]
        xs, ys = [], []
        for swap in (False, True):
            _, X = features.compute(aug, arrays["runs"], swap=swap)
            xs.append(X[idx])
            ys.append((owner[idx] == (1 if swap else 0)).astype(np.int8))
        _ROWS[key] = np.concatenate(xs), np.concatenate(ys)
    return _ROWS[key]


def dead_rows(name, tr):
    key = (name, tr.spec, tr.offset, tr.shift)
    if key not in _DROWS:
        arrays = data()[name]
        aug = degrade(name, tr.spec, train_seed(name, tr.shift))
        idx = np.flatnonzero(inside_mask(arrays))[tr.offset::M.SUBSAMPLE]
        _, X = dead_features.compute(aug, arrays["runs"])
        _DROWS[key] = X[idx], arrays["dead"][idx].astype(np.int8)
    return _DROWS[key]


def fitted(kind, train, tr):
    key = (kind, train, tr._replace(idsse=""))
    if key not in _MODELS:
        parts = [(train_rows if kind == "team" else dead_rows)(n, tr) for n in train]
        gbm = dict(M.GBM, max_features=tr.max_features, random_state=tr.rs)
        _MODELS[key] = HistGradientBoostingClassifier(**gbm).fit(
            np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    return _MODELS[key]


def cached(key, compute):
    """compute() as float32, stored under data/external/cache/logo/ by the hash of key."""
    key = repr((key, sorted(M.GBM.items()), M.SUBSAMPLE))
    path = os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest()[:20] + ".npy")
    if os.path.isfile(path):
        return np.load(path).astype(np.float64)
    out = compute().astype(np.float32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, out)
    return out.astype(np.float64)


def p_home(name, tr=REF, held_spec=("default",)):
    """P(home has the ball) on the held-out match from a model that never saw it."""
    train = train_names(name, tr.idsse)

    def compute():
        h = held_out(name, held_spec)
        clf = fitted("team", train, tr)
        inside = ~np.isnan(h["X0"]).all(axis=1)
        p = np.full(len(h["X0"]), np.nan)
        p[inside] = 0.5 * (clf.predict_proba(h["X0"][inside])[:, 1]
                           + 1 - clf.predict_proba(h["X1"][inside])[:, 1])
        return p
    return cached(("team", name, train, tuple(tr), held_spec), compute)


def p_home_mean(name, trs, held_spec=("default",)):
    return np.mean([p_home(name, tr, held_spec) for tr in trs], axis=0)


def p_dead(name, tr=REF, held_spec=("default",)):
    train = train_names(name, tr.idsse)

    def compute():
        held = held_out(name, held_spec)["held"]
        clf = fitted("dead", train, tr)
        _, X = dead_features.compute(held, data()[name]["runs"])
        p = np.full(len(X), np.nan)
        inside = ~np.isnan(X).all(axis=1)
        p[inside] = clf.predict_proba(X[inside])[:, 1]
        return p
    return cached(("dead", name, train, tuple(tr), held_spec), compute)


# -- decoding and scoring ----------------------------------------------------------------------

def drop_short(path, min_len):
    """Merge possessions shorter than min_len frames into the previous one (the next, if first)."""
    path = path.copy()
    while True:
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(path)) + 1, [len(path)]))
        short = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a < min_len]
        if not short or len(bounds) <= 2:
            return path
        a, b = short[0]
        path[a:b] = path[a - 1] if a > 0 else path[b]


def decode2(name, p, held_spec=("default",), cost=None, min_dead_s=dead_ball.MIN_DEAD_S,
            kickoff=True, min_len=0):
    """Codes from the dead-ball rule + two-state Viterbi (model.labels_from with options)."""
    cost = decode.SWITCH_COST if cost is None else cost
    held, runs = held_out(name, held_spec)["held"], data()[name]["runs"]
    missing = np.isnan(held["ball_xy"][:, 0])
    dead = np.zeros(len(p), dtype=bool)
    for start, stop in dead_ball.stretches(missing, runs):
        if stop - start >= min_dead_s * FPS:
            dead[start:stop] = True
    if kickoff:
        dead |= dead_ball.pre_kickoff_mask(held, runs)
    codes = np.full(len(p), pm.X, dtype=np.int8)
    live = np.zeros(len(p), dtype=bool)
    for start, stop in runs:
        codes[start:stop][dead[start:stop]] = pm.D
        live[start:stop] = ~dead[start:stop]
    for start, stop in dead_ball.stretches(live, runs):
        path = decode.viterbi(p[start:stop], cost)
        if min_len:
            path = drop_short(path, min_len)
        codes[start:stop] = np.where(path == 0, pm.H, pm.A)
    return codes


def decode3(name, p, pd, held_spec=("default",), costs=None):
    costs = costs or (decode.SWITCH_COST, decode.STOP_COST, decode.RESTART_COST)
    return M.labels_from3(p, pd, held_out(name, held_spec)["held"], data()[name]["runs"], costs)


def score(name, codes):
    r = pm.evaluate(truth(name), codes)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def run(make_codes):
    """[10, 4] scores (spellF1, macroF1, predicted possessions, true possessions)."""
    return np.array([score(n, make_codes(n)) for n in names()], dtype=float)


def nested(options):
    """options {label: [10, 4]}: for each match the option with the best mean spellF1 on the
    other 9 is applied to it. Returns ([10, 4], picks)."""
    labels = list(options)
    S = np.array([options[k][:, 0] for k in labels])
    picks = [labels[int(np.argmax(np.delete(S, i, axis=1).mean(1)))] for i in range(S.shape[1])]
    return np.array([options[k][i] for i, k in enumerate(picks)]), picks


def decision(rows_, ref):
    gain = rows_[:, 0].mean() - ref[:, 0].mean()
    wins = int((rows_[:, 0] > ref[:, 0] + 1e-9).sum())
    if gain >= ADOPT_GAIN and wins >= ADOPT_WINS:
        return gain, wins, "usvoji"
    return gain, wins, "možda" if gain > 0 else "odbaci"


def table(title, variants, reference, per_match=()):
    """Markdown table of {label: [10, 4]} against variants[reference]; per_match lists labels
    whose per-match spellF1 is printed too."""
    ns = names()
    met = np.array([n in METRICA for n in ns])
    ref = variants[reference]
    print("\n### {}\n".format(title))
    print("| varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | posjeda pred./istina | Δ | pobjede | odluka |")
    print("|---|---|---|---|---|---|---|---|---|")
    out = {}
    for label, r in variants.items():
        gain, wins, verdict = decision(r, ref)
        tail = "| referenca | | |" if label == reference else "| {:+.3f} | {}/10 | {} |".format(
            gain, wins, verdict)
        print("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} / {:.0f} {}".format(
            label, r[:, 0].mean(), r[met, 0].mean(), r[~met, 0].mean(), r[:, 1].mean(),
            r[:, 2].mean(), r[:, 3].mean(), tail))
        out[label] = dict(spell=r[:, 0].tolist(), macro=r[:, 1].tolist(), pred=r[:, 2].tolist())
    if per_match:
        print("\n| utakmica | " + " | ".join(per_match) + " |")
        print("|---|" + "---|" * len(per_match))
        for i, n in enumerate(ns):
            print("| {} | ".format(n) + " | ".join("{:.3f}".format(variants[k][i, 0])
                                                   for k in per_match) + " |")
    sys.stdout.flush()
    os.makedirs(CACHE_DIR, exist_ok=True)
    slug = hashlib.sha1(title.encode()).hexdigest()[:8]
    with open(os.path.join(CACHE_DIR, "table_{}.json".format(slug)), "w") as handle:
        json.dump(dict(title=title, reference=reference, matches=ns, variants=out), handle)
    return out


def reference(held_spec=("default",)):
    return run(lambda n: decode2(n, p_home(n, REF, held_spec), held_spec))


# -- experiments -------------------------------------------------------------------------------

def exp_ref():
    table("Referenca", {"referenca": reference()}, "referenca", per_match=["referenca"])


def exp_idsse():
    seven = REF._replace(idsse="7")
    variants = {"3 Metrica + 4 IDSSE (ref)": reference(),
                "3 Metrica + 7 IDSSE": run(lambda n: decode2(n, p_home(n, seven)))}
    table("a. IDSSE 4 naspram 7", variants, "3 Metrica + 4 IDSSE (ref)", per_match=list(variants))
    # The three matches outside the reference set are not affected by "held-out is in training"
    # in the 4-match arm; the seven that are in it lose one training match in both arms.


def describe_regimes():
    g = gap_stats()
    recs = g["recs"]
    hidden = recs[:, 1] < recs[:, 0]
    q = lambda v: "/".join("{:.1f}".format(x / FPS) for x in np.percentile(v, [25, 50, 75, 90]))
    print("\nMetrica (premošteno < {:g} s), po utakmici: udio mrtvih frameova bez lopte {}; "
          "živih {}; ukupno {}".format(augment.BRIDGE_S, *(np.round(g["summary"][:, i], 3).tolist()
                                                         for i in range(3))))
    print("prekida {}; lopta nikad ne nestane u {:.1%}; lopta vidljiva nakon prekida igre "
          "(s, 25/50/75/90 pct) {}; vidljiva prije nastavka {}; rupa počinje prije prekida u "
          "{:.1%}, završava nakon nastavka u {:.1%}".format(
              len(recs), 1 - hidden.mean(), q(recs[hidden, 1]), q(recs[hidden, 2]),
              (recs[:, 3] > 0).mean(), (recs[:, 4] > 0).mean()))
    print("rupe u živoj igri: {} ukupno, duljina (s, 25/50/75/90 pct) {}".format(
        len(g["live"]), q(g["live"])))
    print("\n| regim (held-out) | bez lopte: mrtvi | živi | ukupno | praćeno igrača |")
    print("|---|---|---|---|---|")
    for spec in (("default",), ("harsh",), ("real",)):
        rows_ = []
        for n in names():
            a, held = data()[n], held_out(n, spec)["held"]
            ins, miss = inside_mask(a), np.isnan(held["ball_xy"][:, 0])
            rows_.append((miss[ins & a["dead"]].mean(), miss[ins & ~a["dead"]].mean(),
                          miss[ins].mean(), (~np.isnan(held["players_xy"][ins, :, 0])).sum(1).mean()))
        r = np.array(rows_)
        idsse = np.array([n not in METRICA for n in names()])
        for tag, sel in (("Metrica", ~idsse), ("IDSSE", idsse)):
            print("| {} {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(spec[0], tag,
                                                                       *r[sel].mean(0)))
    sys.stdout.flush()


def dead_variants(held_spec, tr, tag):
    """Rule at three minimum lengths and the learned model, for one training recipe."""
    out = {}
    for s in (1.0, 2.0, 3.0):
        out["{}pravilo {:g} s".format(tag, s)] = run(
            lambda n: decode2(n, p_home(n, tr, held_spec), held_spec, min_dead_s=s))
    grid = {(st, rs): run(lambda n: decode3(n, p_home(n, tr, held_spec), p_dead(n, tr, held_spec),
                                            held_spec, (decode.SWITCH_COST, st, rs)))
            for st in STOP_GRID for rs in RESTART_GRID}
    out["{}naučeni P(dead), STOP {:g} / RESTART {:g} (zadano)".format(
        tag, decode.STOP_COST, decode.RESTART_COST)] = grid[(decode.STOP_COST, decode.RESTART_COST)]
    best = max(grid, key=lambda k: grid[k][:, 0].mean())
    out["{}naučeni, najbolji na svih 10: STOP {} / RESTART {} (optimistično)".format(
        tag, *best)] = grid[best]
    nest, picks = nested(grid)
    out["{}naučeni, ugniježđeni izbor STOP/RESTART".format(tag)] = nest
    print("ugniježđeni izbori ({} {}): {}".format(held_spec[0], tag, collections.Counter(picks)))
    return out


def exp_dead():
    describe_regimes()
    for spec in (("default",), ("harsh",), ("real",)):
        variants = dead_variants(spec, REF, "")
        if spec == ("real",):
            variants.update(dead_variants(spec, REF._replace(spec=("real",)), "trening real: "))
            variants.update(dead_variants(spec, REF._replace(spec=("harsh",)), "trening harsh: "))
        table("b. Mrtva lopta, held-out regim '{}'".format(spec[0]), variants, "pravilo 2 s",
              per_match=["pravilo 2 s", "naučeni, ugniježđeni izbor STOP/RESTART"])


def exp_deadlag():
    spec = ("lag", 3.0)
    variants = dead_variants(spec, REF, "")
    variants.update(dead_variants(spec, REF._replace(spec=spec), "trening lag: "))
    table("b2. Mrtva lopta, stres-test: lopta vidljiva do 3 s nakon prekida i prije nastavka "
          "(pretpostavka)", variants, "pravilo 2 s",
          per_match=["pravilo 2 s", "naučeni, ugniježđeni izbor STOP/RESTART",
                     "trening lag: naučeni, ugniježđeni izbor STOP/RESTART"])


def exp_lagtrain():
    """Does training under the lag stress regime cost anything when the held-out match has
    gaps that line up with the stoppages?"""
    lag = REF._replace(spec=("lag", 3.0))
    for spec in (("default",), ("harsh",), ("real",)):
        variants = {"pravilo 2 s, zadani trening (ref)": reference(spec)}
        for tag, tr in (("zadani trening", REF), ("trening lag", lag)):
            if tr is not REF:
                variants["pravilo 2 s, " + tag] = run(
                    lambda n: decode2(n, p_home(n, tr, spec), spec))
            variants["naučeni P(dead), " + tag] = run(
                lambda n: decode3(n, p_home(n, tr, spec), p_dead(n, tr, spec), spec))
            variants["naučeni P(dead), {}, samo P(dead) iz tog treninga".format(tag)] = run(
                lambda n: decode3(n, p_home(n, REF, spec), p_dead(n, tr, spec), spec))
        table("b3. Trening pod lag regimom, held-out regim '{}'".format(spec[0]), variants,
              "pravilo 2 s, zadani trening (ref)")


def exp_noise():
    for level in ("mid", "high"):
        spec = ("noise", level, ("default",))
        sp, sb, tr = NOISE[level]
        variants = {"čisti held-out, zadani trening": reference(),
                    "šum, zadani trening": reference(spec)}
        for train_level in ("mid", "high"):
            t = REF._replace(spec=("noise", train_level, ("default",)))
            variants["šum, trening sa šumom '{}'".format(train_level)] = run(
                lambda n: decode2(n, p_home(n, t, spec), spec))
            if level == "mid":
                variants["čisti held-out, trening sa šumom '{}'".format(train_level)] = run(
                    lambda n: decode2(n, p_home(n, t)))
        table("c. Šum '{}' (igrači σ {} m, lopta σ {} m, praćeno {} igrača)".format(
            level, sp, sb, tr), variants, "šum, zadani trening")


def exp_control():
    variants = {"izvorne oznake (ref)": reference()}
    for s in (0.5, 1.0, 2.0):
        t = REF._replace(control_s=s)
        variants["trening CONTROL_S {:g} s".format(s)] = run(lambda n: decode2(n, p_home(n, t)))
    nest, picks = nested({k: v for k, v in variants.items()})
    variants["CONTROL_S ugniježđeni izbor (uklj. bez)"] = nest
    print("ugniježđeni izbori CONTROL_S:", collections.Counter(picks))
    grid = {m: run(lambda n: decode2(n, p_home(n), min_len=m)) for m in MIN_LEN_GRID}
    for m in MIN_LEN_GRID[1:]:
        variants["min. duljina posjeda {} frameova".format(m)] = grid[m]
    nest, picks = nested(grid)
    variants["min. duljina, ugniježđeni izbor"] = nest
    print("ugniježđeni izbori min. duljine:", collections.Counter(picks))
    table("d. Kontrolirani posjed (samo trening oznake) i minimalna duljina posjeda", variants,
          "izvorne oznake (ref)")


def switch_grid(trs=(REF,), title="e. Cijena prijelaza (SWITCH_COST)"):
    grid = {c: run(lambda n: decode2(n, p_home_mean(n, trs), cost=c)) for c in SWITCH_GRID}
    variants = {"cijena {}".format(c): grid[c] for c in SWITCH_GRID}
    nest, picks = nested(grid)
    variants["ugniježđeni izbor"] = nest
    print("ugniježđeni izbori cijene:", collections.Counter(picks))
    met = np.array([n in METRICA for n in names()])
    for tag, sel in (("Metrica", met), ("IDSSE", ~met), ("svih 10", met | ~met)):
        means = {c: grid[c][sel, 0].mean() for c in SWITCH_GRID}
        print("najbolja cijena, {}: {}  ({})".format(tag, max(means, key=means.get), ", ".join(
            "{}: {:.3f}".format(c, v) for c, v in means.items())))
    print("najbolja cijena po utakmici:", {n: SWITCH_GRID[int(np.argmax(
        [grid[c][i, 0] for c in SWITCH_GRID]))] for i, n in enumerate(names())})
    table(title, variants, "cijena {:g}".format(decode.SWITCH_COST))
    return grid


def exp_switch():
    switch_grid()


ENSEMBLES = {
    "3 pomaka subsamplea": [REF._replace(offset=k) for k in range(3)],
    "5 pomaka subsamplea": [REF._replace(offset=k) for k in range(5)],
    "3 seeda augmentacije": [REF._replace(shift=k) for k in range(3)],
    "5 seedova augmentacije": [REF._replace(shift=k) for k in range(5)],
}
# Not run (machine-time budget): both at once, and column subsampling with different GBM seeds.
ENSEMBLES_SKIPPED = {
    "5 × (pomak + seed)": [REF._replace(offset=k, shift=k) for k in range(5)],
    "5 × max_features 0.5": [REF._replace(max_features=0.5, rs=k) for k in range(5)],
}


def exp_ensemble():
    variants = {"jedan model (ref)": reference()}
    for label, trs in ENSEMBLES.items():
        variants[label] = run(lambda n: decode2(n, p_home_mean(n, trs)))
    for label in ("5 pomaka subsamplea", "5 seedova augmentacije"):
        singles = [run(lambda n: decode2(n, p_home(n, tr)))[:, 0].mean() for tr in ENSEMBLES[label]]
        print("pojedinačni članovi ({}), prosjek spellF1: {}".format(
            label, np.round(singles, 4).tolist()))
    table("f. Ansambl", variants, "jedan model (ref)")


def first_spell(codes, start, stop):
    """(a, b, team code) of the first H/A possession in [start, stop), or None."""
    seg = codes[start:stop]
    idx = np.flatnonzero((seg == pm.H) | (seg == pm.A))
    if not len(idx):
        return None
    a = int(idx[0])
    change = np.flatnonzero(seg[a:] != seg[a])
    b = a + int(change[0]) if len(change) else len(seg)
    return start + a, start + b, int(seg[a])


def nearest_team(held, frame, look=2 * FPS):
    """Team code of the player nearest the ball on the first frame from `frame` with a ball."""
    for t in range(frame, min(frame + look, len(held["ball_xy"]))):
        if not np.isnan(held["ball_xy"][t, 0]):
            d = np.linalg.norm(held["players_xy"][t] - held["ball_xy"][t], axis=1)
            if np.isfinite(np.nanmin(d)):
                return pm.H if held["slot_team"][int(np.nanargmin(d))] == 0 else pm.A
    return None


def exp_kickoff():
    saved = {k: getattr(dead_ball, k) for k in ("KICKOFF_SHARE", "KICKOFF_HOLD_S",
                                                 "KICKOFF_BALL_M", "KICKOFF_CALM_MS")}
    settings = [("zadano", {}), ("SHARE 0.80", {"KICKOFF_SHARE": 0.80}),
                ("SHARE 0.90", {"KICKOFF_SHARE": 0.90}), ("HOLD 5 s", {"KICKOFF_HOLD_S": 5.0}),
                ("HOLD 1.5 s", {"KICKOFF_HOLD_S": 1.5}), ("CALM 2.0", {"KICKOFF_CALM_MS": 2.0})]
    for spec in (("default",), ("real",)):
        print("\nheld-out regim '{}': pogreška kickoff_frame prema prvom živom frameu oznaka "
              "(frameovi, 20 poluvremena)".format(spec[0]))
        variants = {"bez pre-kickoff maske": run(
            lambda n: decode2(n, p_home(n, REF, spec), spec, kickoff=False))}
        for label, over in settings:
            for k, v in over.items():
                setattr(dead_ball, k, v)
            try:
                errs = []
                for n in names():
                    a, held = data()[n], held_out(n, spec)["held"]
                    for start, stop in a["runs"]:
                        first_live = start + int(np.flatnonzero(~a["dead"][start:stop])[0])
                        errs.append(dead_ball.kickoff_frame(held, start, stop) - first_live)
                variants["maska, " + label] = run(
                    lambda n: decode2(n, p_home(n, REF, spec), spec))
            finally:
                for k, v in saved.items():
                    setattr(dead_ball, k, v)
            errs = np.array(errs)
            print("  {:<11} točno {}/20, unutar 1 s {}/20, najveća |pogreška| {} frameova, "
                  "pogreške ≠ 0: {}".format(label, int((errs == 0).sum()),
                                           int((np.abs(errs) <= FPS).sum()),
                                           int(np.abs(errs).max()), errs[errs != 0].tolist()))
        # First possession of each half: who gets it, and does "nearest to the ball" do better?
        hits = near_hits = both = 0
        lead_err = []
        override = {}
        for n in names():
            a, held = data()[n], held_out(n, spec)["held"]
            codes = decode2(n, p_home(n, REF, spec), spec)
            fixed = codes.copy()
            for start, stop in a["runs"]:
                gt_first, pred_first = first_spell(truth(n), start, stop), first_spell(codes, start, stop)
                guess = nearest_team(held, pred_first[0])
                hits += pred_first[2] == gt_first[2]
                near_hits += guess == gt_first[2]
                both += 1
                lead_err.append(pred_first[0] - gt_first[0])
                if guess is not None:
                    fixed[pred_first[0]:pred_first[1]] = guess
            override[n] = fixed
        variants["prvi posjed = momčad najbliža lopti"] = run(lambda n: override[n])
        print("  prvi posjed poluvremena: model+Viterbi pogađa momčad {}/{}, pravilo najbližeg "
              "igrača {}/{}; početak prvog predviđenog posjeda − istina (frameovi): {}".format(
                  hits, both, near_hits, both, lead_err))
        table("g. Početak poluvremena, held-out regim '{}'".format(spec[0]), variants,
              "maska, zadano")


def exp_combo():
    """The single changes that helped, stacked, each against the reference."""
    ens = ENSEMBLES["5 pomaka subsamplea"]
    costs = (decode.SWITCH_COST, decode.STOP_COST, decode.RESTART_COST)
    variants = {
        "referenca": reference(),
        "ansambl 5 pomaka": run(lambda n: decode2(n, p_home_mean(n, ens))),
        "naučeni P(dead)": run(lambda n: decode3(n, p_home(n), p_dead(n))),
        "naučeni P(dead) + ansambl P(home)": run(
            lambda n: decode3(n, p_home_mean(n, ens), p_dead(n))),
    }
    table("Kombinacija (held-out regim 'default')", variants, "referenca",
          per_match=list(variants))
    for spec in (("default",), ("harsh",), ("real",)):
        grid = {c: run(lambda n: decode3(n, p_home(n, REF, spec), p_dead(n, REF, spec), spec,
                                         (c,) + costs[1:])) for c in SWITCH_GRID[:9]}
        v = {"pravilo 2 s, cijena {:g} (ref)".format(costs[0]): reference(spec)}
        v.update({"naučeni P(dead), cijena {}".format(c): grid[c] for c in grid})
        nest, picks = nested(grid)
        v["naučeni P(dead), ugniježđeni izbor cijene"] = nest
        print("ugniježđeni izbori cijene uz naučeni P(dead) ({}):".format(spec[0]),
              collections.Counter(picks))
        table("Cijena prijelaza uz naučeni P(dead), held-out regim '{}'".format(spec[0]), v,
              "pravilo 2 s, cijena {:g} (ref)".format(costs[0]))


EXPERIMENTS = collections.OrderedDict(
    ref=exp_ref, idsse=exp_idsse, dead=exp_dead, deadlag=exp_deadlag, lagtrain=exp_lagtrain,
    noise=exp_noise, control=exp_control,
    switch=exp_switch, ensemble=exp_ensemble, kickoff=exp_kickoff, combo=exp_combo)

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["ref"]
    if wanted == ["all"]:
        wanted = list(EXPERIMENTS)
    unknown = [w for w in wanted if w not in EXPERIMENTS]
    if unknown:
        sys.exit("unknown experiment {}; choose from {}".format(unknown, ", ".join(EXPERIMENTS)))
    for w in wanted:
        EXPERIMENTS[w]()
