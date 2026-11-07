"""Metrica's three public sample games, as practice data with a known answer for dead balls.

These are OTHER games, so RULES.md allows tuning on them freely. They come from
https://github.com/metrica-sports/sample-data (no licence file; the README asks for responsible use
and acknowledgement, which the technical document gives). Download them into data/sample/:

    data/sample/Sample_Game_1/Sample_Game_1_RawEventsData.csv
    data/sample/Sample_Game_1/Sample_Game_1_RawTrackingData_{Home,Away}_Team.csv
    data/sample/Sample_Game_2/...                                   (same layout)
    data/sample/Sample_Game_3/Sample_Game_3_{events.json,metadata.xml,tracking.txt}

load(n) returns the same keys as solution.tracking.load(), so every signal runs unchanged on
either, plus the answer key we build from the events:

    runs   [(start, stop)]   half-open spans from each kickoff to the last event of that half,
                             the equivalent of the competition's graded halves
    dead   (N,) bool         True where the ball is out of play

How "dead" is derived: every restart -- a SET PIECE (kick-off, throw-in, corner, free kick,
penalty) or a goal kick -- ends a stoppage, and the stoppage began when the last event before the
restart ended: the BALL OUT, the foul, the goal. So the ball is dead from the end of that event
to the frame the restart is taken. This is a mechanical rule on human-produced events, accurate
to the event timing, which is good to a few frames.

Unlike our match, positions here are per player rather than per tracklet, and there is no
exported speed, so speeds are computed from positions -- which is how we compute them for our
match too, because its exported ball speed is not usable (see dead_ball.py).
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

# Params
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CACHE_VERSION = 2
FPS = 25
PITCH = (105.0, 68.0)  # README: both CSV games are 105 x 68 m; game 3's metadata says the same
GAMES = (1, 2, 3)


def _folder(n):
    return os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(n))


def _speeds(xy):
    """m/s from consecutive positions, NaN where either frame is missing. Same length as xy."""
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * FPS
    return np.concatenate([step[:1] * np.nan, step]).astype(np.float32)


# -- reading the two formats -------------------------------------------------------------------

def _read_csv_game(n):
    """Games 1 and 2: one wide CSV per team, three header rows, then x,y pairs per player."""
    teams, ball, frames, periods = [], None, None, None
    for side in ("Home", "Away"):
        path = os.path.join(_folder(n), "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(n, side))
        table = pd.read_csv(path, skiprows=2)
        # Columns: Period, Frame, Time [s], then (x, y) per player, then Ball x, y. The y column of
        # each pair is unnamed, so position is what identifies it.
        values = table.iloc[:, 3:].to_numpy(dtype=np.float64)
        teams.append(values[:, :-2].reshape(len(table), -1, 2))
        ball = values[:, -2:]
        frames = table["Frame"].to_numpy()
        periods = table["Period"].to_numpy()

    events = pd.read_csv(os.path.join(_folder(n), "Sample_Game_{}_RawEventsData.csv".format(n)))
    events = [{"type": row["Type"], "subtype": str(row["Subtype"]) if pd.notna(row["Subtype"]) else "",
               "period": int(row["Period"]),
               "start": int(row["Start Frame"]), "end": int(row["End Frame"])}
              for _, row in events.iterrows()]
    players = np.concatenate(teams, axis=1)
    slot_team = np.array([0] * teams[0].shape[1] + [1] * teams[1].shape[1], dtype=np.int8)
    return frames, periods, players, ball, slot_team, events


def _read_epts_game(n):
    """Game 3: FIFA/EPTS like our match, with x,y only (no speed channel)."""
    folder = _folder(n)
    root = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of_channel = {ch.get("id"): ch.get("playerId") for ch in root.iter("PlayerChannel")}
    # One column per player of either squad, substitutes included.
    roster = [p.get("id") for p in root.iter("Player")]
    slot_team = np.array([team_ids.index(team_of_player[p]) for p in roster], dtype=np.int8)

    # Each line holds the 22 players on the pitch, and which player sits in which group changes at
    # every substitution: the metadata carries one DataFormatSpecification per frame range, each
    # listing its groups in order. Lines are parsed per range and scattered into roster columns.
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        order = [player_of_channel[r] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      [roster.index(p) for p in order]))

    with open(os.path.join(folder, "Sample_Game_3_tracking.txt")) as handle:
        lines = handle.read().split("\n")
    lines = [line for line in lines if line]
    frames = np.array([int(line[:line.index(":")]) for line in lines], dtype=np.int64)
    players = np.full((len(lines), len(roster), 2), np.nan)
    ball = np.full((len(lines), 2), np.nan)
    for start, end, columns in specs:
        rows = np.flatnonzero((frames >= start) & (frames <= end))
        if not len(rows):
            continue
        block = "\n".join(lines[r] for r in rows).replace(":", ",").replace(";", ",")
        import io
        table = pd.read_csv(io.StringIO(block), header=None, dtype=np.float64).to_numpy()
        players[rows[:, None], np.array(columns)[None, :]] = (
            table[:, 1:1 + 2 * len(columns)].reshape(len(rows), len(columns), 2))
        ball[rows] = table[:, 1 + 2 * len(columns):3 + 2 * len(columns)]

    with open(os.path.join(folder, "Sample_Game_3_events.json")) as handle:
        raw = json.load(handle)["data"]
    def subtype(e):
        s = e["subtypes"]
        if s is None:
            return ""
        return s["name"] if isinstance(s, dict) else "-".join(t["name"] for t in s)

    events = [{"type": e["type"]["name"], "subtype": subtype(e), "period": int(e["period"]),
               "start": int(e["start"]["frame"]), "end": int(e["end"]["frame"] or 0)}
              for e in raw]
    # No period column in the tracking: a frame belongs to the half whose events surround it.
    periods = np.ones(len(frames), dtype=np.int64)
    second = min(e["start"] for e in events if e["period"] == 2)
    periods[frames >= second] = 2
    return frames, periods, players, ball, slot_team, events


# -- the answer key ----------------------------------------------------------------------------

def is_restart(event):
    """A SET PIECE, or a goal kick -- which Metrica records as the PASS (or BALL LOST, if it was
    intercepted) that takes it, never as a SET PIECE. Missing those mislabels ~20 stoppages a game
    as live play."""
    if event["type"] == "SET PIECE":
        return True
    return event["type"] in ("PASS", "BALL LOST") and event["subtype"].startswith("GOAL KICK")


def dead_ball(n_frames, events, runs, index_of):
    """Dead from the end of the event before each restart until that restart is taken."""
    dead = np.zeros(n_frames, dtype=bool)
    ordered = sorted(events, key=lambda e: (e["start"], e["end"]))
    for i, event in enumerate(ordered):
        if not is_restart(event):
            continue
        taken = index_of(event["start"])
        run = next(((a, b) for a, b in runs if a <= taken < b), None)
        if run is None:
            continue
        # Strictly earlier: game 3 logs the keeper's RECOVERY on the same frame as the goal kick.
        previous = [e for e in ordered[:i]
                    if e["period"] == event["period"] and e["start"] < event["start"]]
        stopped = (index_of(max(previous[-1]["start"], previous[-1]["end"])) + 1
                   if previous else run[0])
        dead[max(run[0], stopped):taken] = True
    return dead


def build(n):
    frames, periods, players, ball, slot_team, events = (
        _read_csv_game(n) if n in (1, 2) else _read_epts_game(n))
    if not np.array_equal(np.diff(frames), np.ones(len(frames) - 1)):
        raise ValueError("sample game {}: tracking frames are not contiguous".format(n))
    first = int(frames[0])

    def index_of(frame):
        return int(frame) - first

    # Each half runs from its kickoff to the end of its last event.
    runs = []
    for period in sorted(set(e["period"] for e in events)):
        in_period = [e for e in events if e["period"] == period]
        kickoff = min(e["start"] for e in in_period)
        last = max(max(e["start"], e["end"]) for e in in_period)
        runs.append((index_of(kickoff), min(len(frames), index_of(last) + 1)))

    scale = np.array(PITCH)
    players_xy = (players * scale).astype(np.float32)
    ball_xy = (ball * scale).astype(np.float32)
    return {
        "ball_xy": ball_xy,
        "ball_speed": _speeds(ball_xy),
        "players_xy": players_xy,
        "players_speed": np.stack([_speeds(players_xy[:, s]) for s in range(players_xy.shape[1])],
                                  axis=1),
        "slot_team": slot_team,
        "pitch": scale.astype(np.float32),
        "runs": np.array(runs, dtype=np.int64),
        "dead": dead_ball(len(frames), events, runs, index_of),
        "version": np.array(CACHE_VERSION),
    }


def load(n, force=False):
    """One sample game, from the cache if it exists."""
    path = os.path.join(CACHE_DIR, "sample_game_{}_v{}.npz".format(n, CACHE_VERSION))
    if not force and os.path.isfile(path):
        with np.load(path) as cached:
            arrays = {key: cached[key] for key in cached.files}
    else:
        arrays = build(n)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(path, **arrays)
    arrays["runs"] = [tuple(int(v) for v in run) for run in arrays["runs"]]
    return arrays


if __name__ == "__main__":
    for n in GAMES:
        g = load(n, force="--force" in sys.argv)
        graded = np.zeros(len(g["dead"]), dtype=bool)
        for a, b in g["runs"]:
            graded[a:b] = True
        seen = ~np.isnan(g["ball_xy"][:, 0])
        dead = g["dead"]
        print("game {}: {} frames, {} graded, dead {:.3f}, ball seen {:.3f}".format(
            n, len(dead), graded.sum(), dead[graded].mean(), seen[graded].mean()))
        print("  ball missing when dead {:.3f}, ball missing when alive {:.3f}".format(
            (~seen)[graded & dead].mean(), (~seen)[graded & ~dead].mean()))
