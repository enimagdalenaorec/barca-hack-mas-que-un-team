"""Round 7: can the clean-ball configuration (karlo-v2) still improve? External matches only.

    python -m solution.clean_exp fold <i...>   # fit and score fold i (0-9), cache probabilities
    python -m solution.clean_exp report        # arms on clean and Gaussian held-out balls
    python -m solution.clean_exp decoder       # nested decoder grid for the best arm

The pipeline uses the clean configuration when the tracker's ball does not lag its players
(solution/tracker.py), so this round scores on the two ball kinds that configuration is for: the
clean held-out ball and the Gaussian-smoothed one (sigma 6), each under default / harsh / lag.
Arms, all trained on clean balls with one view and the dead model on clean balls:
  K   the clean configuration as submitted
  W   K + per-possession sample weights (model.SPELL_WEIGHTS, round 5)
  TW  K + touch features + weights (round 3's touches, round 5's weights)
Adoption rule (as every round): mean spellF1 gain >= 0.008 on the clean ball, wins >= 7 of 10,
harsh / lag not below -0.005; and not below -0.005 on the Gaussian ball.
"""

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import dead_ball, dead_features, features  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import logo2 as L2  # noqa: E402
from solution import robust_exp as R  # noqa: E402
from solution import smooth_train_exp as S  # noqa: E402
from solution import touch_exp as T  # noqa: E402

BASES = ("default", "harsh", "lag")
BALLS = ("clean", "gauss6")
ARMS = {"K": dict(touch=False, weights=False), "W": dict(touch=False, weights=True),
        "TW": dict(touch=True, weights=True)}
OUT_DIR = os.path.join(os.path.dirname(L.CACHE_DIR), "clean_exp")


def run_fold(i):
    name = L.names()[i]
    runs = L.data()[name]["runs"]
    team_train, dead_train = L.train_names(name, "4"), L.train_names(name, "7")
    team = {}
    for a, c in ARMS.items():
        parts = [S.team_rows(n, False, c["touch"]) for n in team_train]
        team[a] = S.fit(parts, [S.team_weights(n) for n in team_train] if c["weights"] else None)
    dead = S.fit([S.dead_rows(n, False) for n in dead_train])
    os.makedirs(OUT_DIR, exist_ok=True)
    for base in BASES:
        for kind in BALLS:
            for k in T.SEEDS:
                held = R.held_arrays(name, base, kind, k)
                _, XD = dead_features.compute(held, runs)
                save = dict(pd7=L2.dead_proba(dead, XD),
                            forced=dead_ball.pre_kickoff_mask(held, runs))
                for a, c in ARMS.items():
                    ph, d_us, d_them = T.held_inputs(name, held, runs, team[a], c["touch"])
                    save["ph_" + a] = ph
                save["d1_us"], save["d1_them"] = d_us, d_them
                np.savez_compressed(os.path.join(OUT_DIR, "{}_{}_{}_{}.npz".format(
                    name, base, kind, k)), **save)
            print(name, base, kind, flush=True)
    features.TOUCH_FEATURES = True


_H = {}


def held(name, base, kind, k):
    key = (name, base, kind, k)
    if key not in _H:
        with np.load(os.path.join(OUT_DIR, "{}_{}_{}_{}.npz".format(name, base, kind, k))) as z:
            _H[key] = {f: z[f] for f in z.files}
    return _H[key]


def scores(arm, base, kind, g=T.FINAL, full=False):
    """[10, seeds, 4]; g = decoder setting as logo2.flip_codes takes it."""
    out = []
    for n in L.names():
        row = []
        for k in T.SEEDS:
            h = held(n, base, kind, k)
            h2 = dict(ph=h["ph_" + arm].astype(np.float64), pd7=h["pd7"].astype(np.float64),
                      forced=h["forced"], d1_us=h["d1_us"].astype(np.float64),
                      d1_them=h["d1_them"].astype(np.float64))
            row.append(L2.score(n, L2.flip_codes(n, h2, g), full))
        out.append(row)
    return np.array(out, dtype=float)


def report():
    print("| arm | ball | default spellF1 | gain | wins | harsh gain | lag gain | macroF1 |")
    print("|---|---|---|---|---|---|---|---|")
    for kind in BALLS:
        ref = {b: L2.per_match(scores("K", b, kind)) for b in BASES}
        for a in ARMS:
            r = {b: scores(a, b, kind, full=(b == "default")) for b in BASES}
            s = L2.per_match(r["default"])
            print("| {} | {} | {:.4f} | {:+.4f} | {}/10 | {:+.4f} | {:+.4f} | {:.4f} |".format(
                a, kind, s.mean(), s.mean() - ref["default"].mean(),
                int((s > ref["default"] + 1e-9).sum()),
                L2.per_match(r["harsh"]).mean() - ref["harsh"].mean(),
                L2.per_match(r["lag"]).mean() - ref["lag"].mean(), r["default"][:, :, 1].mean()))


GRID = [("flip", sw, r, 2.0, 10, "pd7", td) for sw in (12, 16, 20, 25) for r in (0.4, 0.6, 0.8)
        for td in (1.0, 2.0, 3.0)]


def decoder(arm):
    """Nested choice over GRID on the clean ball (pick on 9, score the 10th), compared with the
    submitted decoding; also reported on the Gaussian ball and the harsh / lag regimes."""
    res = {g: L2.per_match(scores(arm, "default", "clean", g)) for g in GRID}
    S_ = np.array([res[g] for g in GRID])
    picks = [GRID[int(np.argmax(np.delete(S_, i, axis=1).mean(1)))] for i in range(S_.shape[1])]
    nested = np.array([res[g][i] for i, g in enumerate(picks)])
    ref = L2.per_match(scores("K", "default", "clean"))
    best = GRID[int(np.argmax(S_.mean(1)))]
    print("arm", arm, "clean: submitted {:.4f}, best-on-10 {} {:.4f}, nested {:.4f} (wins {}/10)"
          .format(ref.mean(), best[1:5] + best[6:], res[best].mean(), nested.mean(),
                  int((nested > ref + 1e-9).sum())))
    print("nested picks:", sorted({(g[1], g[2], g[6]) for g in picks}))
    for b, kind in itertools.product(BASES, BALLS):
        r0 = L2.per_match(scores("K", b, kind)).mean()
        r1 = L2.per_match(scores(arm, b, kind, best)).mean()
        print("  best-on-10 setting on {}/{}: {:.4f} vs K submitted {:.4f} ({:+.4f})".format(
            b, kind, r1, r0, r1 - r0))


if __name__ == "__main__":
    cmd = (sys.argv[1:] or ["report"])[0]
    if cmd == "fold":
        for arg in sys.argv[2:]:
            run_fold(int(arg))
    elif cmd == "report":
        report()
    elif cmd == "decoder":
        decoder(sys.argv[2] if len(sys.argv) > 2 else "K")
