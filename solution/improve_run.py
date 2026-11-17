"""Drivers of the overnight studies (see improve_exp.py / improve_dec.py). External matches only.

    python -m solution.improve_run ref            # D and S, submitted decoding, every regime
    python -m solution.improve_run dec VAR        # nested decoder re-tuning of a variant
"""

import itertools
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import improve_dec as DD  # noqa: E402
from solution import improve_exp as IE  # noqa: E402
from solution import logo as L  # noqa: E402

Cfg, Var, REF = DD.Cfg, DD.Var, DD.REF
LOG = os.path.join(ROOT, "logs")
REPORT = [("default", "trail21"), ("harsh", "trail21"), ("lag", "trail21"), ("default", "clean"),
          ("default", "gauss6"), ("default", "trail13"), ("default", "trail29")]


def full(var, cfgs):
    """cfgs: one Cfg, or a list of 10 (one per held-out match: a nested choice).
    -> {regime: [10, 2, 4]} with macroF1."""
    if isinstance(cfgs, Cfg):
        res = DD.scores([(var, cfgs, b, k, True) for b, k in REPORT])
        return {"{}/{}".format(b, k): r for (b, k), r in zip(REPORT, res)}
    uniq = sorted(set(cfgs))
    res = DD.scores([(var, c, b, k, True) for c in uniq for b, k in REPORT])
    out = {}
    for j, (b, k) in enumerate(REPORT):
        out["{}/{}".format(b, k)] = np.array(
            [res[uniq.index(c) * len(REPORT) + j][i] for i, c in enumerate(cfgs)])
    return out


def line(label, r, ref):
    """One report row against the reference dict."""
    def d(key):
        return DD.pmatch(r[key]) - DD.pmatch(ref[key])
    g = d("default/trail21")
    wins = int((g > 1e-9).sum())
    ok = g.mean() >= 0.008 and wins >= 7 and d("harsh/trail21").mean() >= -0.005 \
        and d("lag/trail21").mean() >= -0.005
    return ("| {} | {:.4f} ({:+.4f}) | {}/10 | {:+.4f} | {:+.4f} | {:.4f} ({:+.4f}) | {:.4f} ({:+.4f}) "
            "| {:+.4f} / {:+.4f} | {:.3f} | {:.0f}/{:.0f} | {} |").format(
        label, DD.pmatch(r["default/trail21"]).mean(), g.mean(), wins,
        d("harsh/trail21").mean(), d("lag/trail21").mean(),
        DD.pmatch(r["default/clean"]).mean(), d("default/clean").mean(),
        DD.pmatch(r["default/gauss6"]).mean(), d("default/gauss6").mean(),
        d("default/trail13").mean(), d("default/trail29").mean(),
        r["default/trail21"][:, :, 1].mean(), r["default/trail21"][:, :, 2].mean(),
        r["default/trail21"][:, :, 3].mean(), "ADOPT" if ok else ("ref" if r is ref else "no"))


HEAD = ("| variant | trail21 (gain) | wins | harsh | lag | clean | gauss6 | trail13 / trail29 "
        "| macroF1 | poss pred/true | rule |\n|---|---|---|---|---|---|---|---|---|---|---|")


def save(name, obj):
    with open(os.path.join(LOG, name + ".pkl"), "wb") as handle:
        pickle.dump(obj, handle)


def load(name):
    with open(os.path.join(LOG, name + ".pkl"), "rb") as handle:
        return pickle.load(handle)


def ref_d():
    path = os.path.join(LOG, "ref_D.pkl")
    if not os.path.isfile(path):
        save("ref_D", full(DD.V["D"], REF))
    return load("ref_D")


def per_match_table(rows):
    names = L.names()
    print("| match | " + " | ".join(k for k, _ in rows) + " |\n|---|" + "---|" * len(rows))
    for i, n in enumerate(names):
        print("| {} | {} |".format(n, " | ".join(
            "{:.3f}".format(DD.pmatch(r["default/trail21"])[i]) for _, r in rows)))


def variants(spec):
    """{label: Var} from "label=ph1+ph2:pd1+pd2:sh,..." or names of DD.V / single arms."""
    out = {}
    for item in spec.split(","):
        if "=" in item:
            label, body = item.split("=")
            ph, pd_, sh = body.split(":")
            out[label] = Var(tuple(ph.split("+")), tuple(pd_.split("+")), int(sh))
        elif item in DD.V:
            out[item] = DD.V[item]
        else:
            c = IE.ARMS[item]
            pd_arm = item if c.get("dead", True) else ("S" if c["shift"] == 10 else "D")
            out[item] = Var((item,), (pd_arm,), c["shift"])
    return out


def cmd_ref(spec="D,S"):
    ref = ref_d()
    print(HEAD)
    rows = []
    for label, var in variants(spec).items():
        r = ref if label == "D" else full(var, REF)
        save("full_" + label, r)
        rows.append((label, r))
        print(line(label, r, ref), flush=True)
    per_match_table(rows)


GRID1 = [Cfg(S=S, r=r, Td=Td, Th=Th) for S in (8, 10, 12, 14, 17, 20, 25, 30)
         for r in (0.2, 0.4, 0.6, 0.8, 1.0) for Td in (1.0, 1.5, 2.0, 3.0) for Th in (0.7, 1.0, 1.4)]


def grid2(anchors, evs):
    out = []
    for a in anchors:
        out += [a._replace(R=R, w=w, ev=ev) for R in (1.0, 2.0, 3.0, 4.0) for w in (5, 10, 15, 25)
                for ev in evs]
        out += [a._replace(st=st, rs=rs) for st in (10, 14, 20, 30) for rs in (4, 8, 12, 16)]
    return out


def cmd_dec(label):
    """Nested decoder re-tuning on default/trail21: grid 1 (switch, flip factor, temperatures),
    then flip geometry / evidence de-lag and stop / restart around its 3 best; the nested choice
    is over the union."""
    label, var = next(iter(variants(label).items()))
    ref = ref_d()
    S1, _, _ = DD.study(var, GRID1, label=label + " grid1")
    top = [GRID1[i] for i in np.argsort(-S1.mean(1))[:3]]
    evs = (None,) if var.sh else (0, 5, 10, 15)
    g2 = [g for g in dict.fromkeys(grid2(top, evs)) if g not in set(GRID1)]
    S2, _, _ = DD.study(var, g2, label=label + " grid2")
    grid, S = GRID1 + g2, np.concatenate([S1, S2])
    picks = DD.nested_picks(S)
    best = grid[int(np.argmax(S.mean(1)))]
    order = np.argsort(-S.mean(1))[:8]
    for i in order:
        print("   {:.4f} {}".format(S.mean(1)[i], grid[i]))
    save("dec_" + label, dict(grid=grid, S=S, picks=picks))
    print(HEAD)
    r_same = full(var, REF)
    print(line(label + " submitted decoding", r_same, ref))
    r_best = full(var, best)
    print(line(label + " best-on-10 " + str(tuple(best)), r_best, ref))
    r_nest = full(var, [grid[p] for p in picks])
    print(line(label + " NESTED", r_nest, ref), flush=True)
    save("decfull_" + label, dict(best=best, r_best=r_best, r_nest=r_nest, r_same=r_same))
    per_match_table([("D", ref), (label, r_same), ("best", r_best), ("nested", r_nest)])


if __name__ == "__main__":
    {"ref": cmd_ref, "dec": cmd_dec}[sys.argv[1]](*sys.argv[2:])
