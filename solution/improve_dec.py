"""Decoder studies on the cached held-out probabilities of solution/improve_exp.py.

Library: variants, configs, pooled scoring, nested choice. Driven by small scripts / -c snippets.
External matches only. Verification code, not part of the submission.
"""

import collections
import ctypes
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decode  # noqa: E402
from solution import improve_exp as IE  # noqa: E402
from solution import logo as L  # noqa: E402
from solution import logo2 as L2  # noqa: E402

pm = L2.pm
SEEDS = (1, 2)
# The three-state Viterbi in C (solution/cvit/vit.c, the same recursion as decode.viterbi3, checked
# to reproduce its numbers), built on first use: the decoder grid has hundreds of settings.
_SRC = os.path.join(ROOT, "solution", "cvit", "vit.c")
_SO = os.path.join(ROOT, "logs", "cvit", "vit.so")
if not os.path.isfile(_SO):
    import subprocess
    os.makedirs(os.path.dirname(_SO), exist_ok=True)
    subprocess.check_call(["cc", "-O2", "-shared", "-fPIC", "-o", _SO, _SRC])
_LIB = ctypes.CDLL(_SO)
_DP = ctypes.POINTER(ctypes.c_double)
_LIB.vit3.argtypes = [ctypes.c_int, _DP, _DP, _DP, _DP, ctypes.c_double, ctypes.c_double,
                      ctypes.POINTER(ctypes.c_int8)]

# S switch, r flip factor, R flip radius, w flip window, st stop, rs restart, Td dead temperature,
# Th home temperature, ev flip-evidence de-lag (None = the variant's own shift)
Cfg = collections.namedtuple("Cfg", "S r R w st rs Td Th ev", defaults=(20.0, 0.4, 2.0, 10, 20.0,
                                                                        8.0, 2.0, 1.0, None))
# ph: arms whose P(home) are averaged; pd: arms whose P(dead) are averaged; sh: de-lag the
# pipeline applies (selects forced / default flip evidence)
Var = collections.namedtuple("Var", "ph pd sh")
REF = Cfg()
V = {"D": Var(("D",), ("D",), 0), "S": Var(("S",), ("S",), 10)}

_NPZ = {}


def npz(name, tag):
    key = (name, tag)
    if key not in _NPZ:
        with np.load(IE.path(name, tag)) as z:
            _NPZ[key] = {f: z[f] for f in z.files}
    return _NPZ[key]


def vit3(pdead, ph, forced, cost, st, rs):
    pd_ = np.clip(np.nan_to_num(pdead, nan=0.5), decode.EPS, 1 - decode.EPS)
    p = np.clip(np.nan_to_num(ph, nan=0.5), decode.EPS, 1 - decode.EPS)
    live = np.where(forced, -np.inf, np.log(1 - pd_))
    eh = np.ascontiguousarray(live + np.log(p))
    ea = np.ascontiguousarray(live + np.log(1 - p))
    ed = np.ascontiguousarray(np.log(pd_))
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    path = np.empty(len(p), dtype=np.int8)
    _LIB.vit3(len(p), eh.ctypes.data_as(_DP), ea.ctypes.data_as(_DP), ed.ctypes.data_as(_DP),
              cost.ctypes.data_as(_DP), float(st), float(rs),
              path.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
    return path


_FLIP = {}


def flip_mask(name, tag, ev, R, w):
    key = (name, tag, ev, R, w)
    if key not in _FLIP:
        e = npz(name, "ev")
        if len(_FLIP) > 4000:
            _FLIP.clear()
        _FLIP[key] = decode.flip_frames(e[tag + "d0_{}".format(ev)].astype(np.float64),
                                        e[tag + "d1_{}".format(ev)].astype(np.float64), R, w)
    return _FLIP[key]


def mean_p(name, arms, tag, field):
    ps = [npz(name, a)[tag + field].astype(np.float64) for a in arms]
    return ps[0] if len(ps) == 1 else np.mean(ps, axis=0)


def codes(name, var, cfg, base, kind, k):
    tag = "{}/{}/{}/".format(base, kind, k)
    ph = mean_p(name, var.ph, tag, "ph")
    pdead = mean_p(name, var.pd, tag, "pd")
    if cfg.Td != 1.0:
        pdead = decode.tempered(pdead, cfg.Td)
    if cfg.Th != 1.0:
        ph = decode.tempered(ph, cfg.Th)
    forced = npz(name, "ev")[tag + "forced_{}".format(var.sh)]
    cost = np.full(len(ph), float(cfg.S))
    if cfg.r != 1.0:
        ev = var.sh if cfg.ev is None else cfg.ev
        cost[flip_mask(name, tag, ev, cfg.R, cfg.w)] *= cfg.r
    out = np.full(len(ph), pm.X, dtype=np.int8)
    letters = np.array([pm.H, pm.A, pm.D], dtype=np.int8)
    for a, b in L.data()[name]["runs"]:
        a, b = int(a), int(b)
        out[a:b] = letters[vit3(pdead[a:b], ph[a:b], forced[a:b], cost[a:b], cfg.st, cfg.rs)]
    return out


def score_one(args):
    var, cfg, base, kind, full = args
    return np.array([[L2.score(n, codes(n, var, cfg, base, kind, k), full) for k in SEEDS]
                     for n in L.names()], dtype=float)


_POOL = None


def pool():
    global _POOL
    if _POOL is None:
        L.data()
        for n in L.names():
            L2.truth(n)
        _POOL = mp.get_context("fork").Pool(int(os.environ.get("IMPROVE_WORKERS", "7")))
    return _POOL


def scores(tasks):
    """[(var, cfg, base, kind, full)] -> list of [10, 2, 4]."""
    return pool().map(score_one, tasks, chunksize=1)


def pmatch(r):
    return r[:, :, 0].mean(1)


def nested_picks(S):
    """S [n_options, 10] -> for each match the option best on the other 9."""
    return [int(np.argmax(np.delete(S, i, axis=1).mean(1))) for i in range(S.shape[1])]


def study(var, grid, base="default", kind="trail21", label=""):
    """Scores every cfg of grid; prints best-on-10, the nested estimate and the picks."""
    res = scores([(var, g, base, kind, False) for g in grid])
    S = np.array([pmatch(r) for r in res])
    picks = nested_picks(S)
    nest = np.array([S[p, i] for i, p in enumerate(picks)])
    best = int(np.argmax(S.mean(1)))
    print("{}: {} options, best-on-10 {:.4f} {}  nested {:.4f}  picks {}".format(
        label, len(grid), S.mean(1)[best], grid[best], nest.mean(),
        collections.Counter(picks).most_common(4)), flush=True)
    return S, picks, best
