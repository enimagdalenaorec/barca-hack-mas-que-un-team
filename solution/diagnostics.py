"""Describe submissions without any ground truth: how many possessions, how long, how jumpy.

This is how we judge a change on the competition match. The public 5 minutes may only confirm
that the pipeline is correct (RULES.md), so nothing here reads a ground truth file. It reports
two kinds of number:

  shape         possession counts and durations, flicker, dead-ball stretches. Compare these
                with what real matches look like (measured on other games), never with a target
                tuned to this one.
  consistency   agreement with the Smart Tagging. The tagging is automatic and noisy, so this is
                evidence, not accuracy -- a big drop means something broke, a small change means
                nothing.

    python -m solution.diagnostics evaluation/results/baseline_smooth50.csv  [more.csv ...]

Several files print side by side, one column each. Only graded frames (the two halves) count,
and X is forward-filled exactly as the scorer does before anything temporal is measured.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
SHORT_POSSESSIONS_S = (1, 3)  # report how many possessions are shorter than these
# Smart Tagging team names -> our codes. "N/A" (every throw-in, among others) maps to nothing.
TAG_TEAM = {"barcelona": pm.H, "atletic club": pm.A}
# Phases tagged with the team that HAS the ball, and phases tagged with the team that does not.
ATTACKING_CODES = {"build up", "progression", "creating chances", "attacking transition",
                   "ball in final third", "ball in the box"}
DEFENDING_CODES = {"defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition"}
SET_PIECE_CODE = "set pieces"
# A set-piece tag starts around the restart and then runs ~35 s to cover the play that follows,
# so only its START says anything about the restart. We look for the stoppage just before it and
# for the first team in possession shortly after it.
SET_PIECE_LOOKBACK_S = 3
SET_PIECE_LOOKAHEAD_S = 10


def graded_runs(game=GAME):
    """Half-open [start, stop) spans of the two halves, the same spans the scorer grades."""
    return [(first, last + 1) for _, first, last in half_video_blocks(game)]


def read_codes(path, game=GAME):
    """Label codes for a submission, or a ValueError with the validator's reason."""
    frame = pm.read_label_csv(path)
    reason = pm.validate_submission(frame, source_frames("tactical", game))
    if reason:
        raise ValueError("{}: {}".format(path, reason))
    return pm.label_codes(frame)


def shape(codes, runs):
    """Possession and stoppage statistics over graded frames."""
    graded = np.concatenate([codes[a:b] for a, b in runs])
    filled = pm.fill_abstentions(codes, runs)
    segs = pm.segments(filled, runs)
    spells = [(a, b, k) for a, b, k in segs if k in pm.POSSESSION_LABELS]
    dead = [(a, b) for a, b, k in segs if k == pm.D]
    durations = np.array([b - a for a, b, _ in spells], dtype=float) / FPS
    dead_durations = np.array([b - a for a, b in dead], dtype=float) / FPS

    # A direct hand-over is H then A (or A then H) with no stoppage between: the flicker that
    # creates fake possessions. Counted on consecutive segments inside a half.
    direct = sum(1 for (_, b1, k1), (a2, _, k2) in zip(segs, segs[1:])
                 if b1 == a2 and {k1, k2} == {pm.H, pm.A})

    out = {
        "H share": np.mean(graded == pm.H),
        "A share": np.mean(graded == pm.A),
        "D share": np.mean(graded == pm.D),
        "X share": np.mean(graded == pm.X),
        "possessions": len(spells),
        "  median s": np.median(durations) if len(durations) else np.nan,
        "  mean s": durations.mean() if len(durations) else np.nan,
        "  p90 s": np.percentile(durations, 90) if len(durations) else np.nan,
    }
    for limit in SHORT_POSSESSIONS_S:
        out["  under {} s".format(limit)] = int(np.sum(durations < limit))
    out["direct H<->A flips"] = direct
    out["dead-ball stretches"] = len(dead)
    out["  median s "] = np.median(dead_durations) if len(dead_durations) else np.nan
    return out


def _in_runs(frame, runs):
    return any(a <= frame < b for a, b in runs)


def consistency(codes, runs, events):
    """Agreement with the Smart Tagging. Evidence, not accuracy -- see the module docstring."""
    filled = pm.fill_abstentions(codes, runs)
    lookback = SET_PIECE_LOOKBACK_S * FPS
    lookahead = SET_PIECE_LOOKAHEAD_S * FPS

    restarts = [e for e in events if (e["code"] or "").strip().lower() == SET_PIECE_CODE
                and _in_runs(e["start_frame"], runs)]
    dead_before = sum(1 for e in restarts
                      if np.any(filled[max(0, e["start_frame"] - lookback):
                                       e["start_frame"] + lookahead] == pm.D))
    # Who has the ball once the set piece is taken: the first alive frame after the tag starts.
    team_hits = team_total = 0
    for e in restarts:
        team = TAG_TEAM.get((e["tags"].get("Team") or "").strip().lower())
        if team is None:
            continue
        after = filled[e["start_frame"]:e["start_frame"] + lookahead]
        alive = after[np.isin(after, pm.POSSESSION_LABELS)]
        team_total += 1
        team_hits += int(len(alive) > 0 and alive[0] == team)

    agree = total = 0
    for e in events:
        code = (e["code"] or "").strip().lower()
        team = TAG_TEAM.get((e["tags"].get("Team") or "").strip().lower())
        if team is None or code not in ATTACKING_CODES | DEFENDING_CODES:
            continue
        expected = team if code in ATTACKING_CODES else (pm.A if team == pm.H else pm.H)
        window = np.array([filled[f] for f in range(e["start_frame"], e["end_frame"])
                           if _in_runs(f, runs)], dtype=np.int8)
        window = window[np.isin(window, pm.POSSESSION_LABELS)]
        agree += int(np.sum(window == expected))
        total += len(window)

    return {
        "set pieces w/ D before": "{}/{}".format(dead_before, len(restarts)),
        "set-piece team right": "{}/{}".format(team_hits, team_total),
        "phase team agreement": agree / total if total else np.nan,
    }


def _fmt(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(value)
    if np.isnan(value):
        return "-"
    return "{:.3f}".format(value) if abs(value) < 1 else "{:.1f}".format(value)


def report(paths, game=GAME):
    import load_smart_tagging  # starter/, standard library only
    events = load_smart_tagging.load(game)
    runs = graded_runs(game)

    columns, names = [], []
    for path in paths:
        codes = read_codes(path, game)
        columns.append({**shape(codes, runs), **consistency(codes, runs, events)})
        names.append(os.path.splitext(os.path.basename(path))[0])

    width = max(12, max(len(n) for n in names) + 2)
    print("{:<26}".format("graded frames only") + "".join(n.rjust(width) for n in names))
    print("-" * (26 + width * len(names)))
    for i, key in enumerate(columns[0]):
        if key == "set pieces w/ D before":
            print("-- consistency with Smart Tagging (noisy, not ground truth)")
        print("{:<26}".format(key) + "".join(_fmt(c[key]).rjust(width) for c in columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="submission CSVs")
    report(parser.parse_args().paths)
