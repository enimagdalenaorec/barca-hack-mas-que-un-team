"""The tactical ATD as numpy arrays, parsed once and cached.

Parsing the 187 MB raw file takes a few seconds; loading the cache takes well under one. Every
later step of the pipeline starts from load(), so nothing else ever reads the raw text.

    python -m solution.tracking            # build the cache (or rebuild it with --force)

What you get, all indexed by VIDEO frame (row i == submission row i, so frame_id - 1):

    ball_xy       (N, 2)     float32  metres, NaN where the tracker has no ball
    ball_speed    (N,)       float32  m/s as exported, NaN where no ball
    players_xy    (N, S, 2)  float32  metres, NaN where the slot is empty on that frame
    players_speed (N, S)     float32  m/s
    slot_team     (S,)       int8     0 = home (Barcelona), 1 = away (Athletic)
    pitch         (2,)       float32  length, width in metres

S is the number of track slots (46 here). A slot is a tracklet, not a person -- see DATA.md.

Coordinates: the ATD is normalised 0-1, scaled here by games.py's measured pitch so distances are
real metres on both axes. The origin is a corner of the pitch, as in the raw file.

The cache lives in data/cache/, which is gitignored with the rest of data/. It is derived match
data and must never be committed or shared.
"""

import argparse
import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CACHE_VERSION = 1  # bump when the array layout changes, so stale caches are rebuilt
HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """0/1 per track slot, home first -- the FIFA/EPTS convention baseline.py also relies on."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    return np.array([HOME if player.get("teamId") == team_ids[0] else AWAY
                     for player in root.iter("Player")], dtype=np.int8)


def parse(game=GAME, data_dir=None):
    """Read the raw ATD into the arrays described above. Slow path; use load()."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    slot_team = slot_teams(meta_path)
    n_slots = len(slot_team)
    n_frames = source_frames("tactical", game)

    # "<frame>:<x,y,s>;<x,y,s>;...:<x,y,s>" -- every separator becomes a comma, and the line is
    # one flat CSV row: frame id, 3 values per slot, 3 for the ball.
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        engine="c").to_numpy()
    del text

    expected_cols = 1 + 3 * n_slots + 3
    if table.shape != (n_frames, expected_cols):
        raise ValueError("ATD has shape {}, expected ({}, {})".format(
            table.shape, n_frames, expected_cols))
    frame_ids = table[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("ATD frame ids are not 1..{} in order".format(n_frames))

    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width], dtype=np.float64)

    players = table[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)
    ball = table[:, 1 + 3 * n_slots:]
    return {
        "ball_xy": (ball[:, :2] * scale).astype(np.float32),
        "ball_speed": ball[:, 2].astype(np.float32),
        "players_xy": (players[:, :, :2] * scale).astype(np.float32),
        "players_speed": players[:, :, 2].astype(np.float32),
        "slot_team": slot_team,
        "pitch": scale.astype(np.float32),
        "version": np.array(CACHE_VERSION),
    }


def cache_path(game=GAME):
    return os.path.join(CACHE_DIR, "{}_tactical_atd_v{}.npz".format(game, CACHE_VERSION))


def load(game=GAME, data_dir=None, force=False):
    """The arrays, from the cache if it exists, building it otherwise."""
    path = cache_path(game)
    if not force and os.path.isfile(path):
        with np.load(path) as cached:
            return {key: cached[key] for key in cached.files}

    arrays = parse(game, data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    return arrays


def summary(arrays):
    """Aggregate shape of the data -- counts only, nothing frame-level."""
    n_frames = len(arrays["ball_xy"])
    ball_seen = ~np.isnan(arrays["ball_xy"][:, 0])
    slots_seen = ~np.isnan(arrays["players_xy"][:, :, 0])
    per_team = [slots_seen[:, arrays["slot_team"] == team].sum(axis=1) for team in (HOME, AWAY)]
    return {
        "frames": n_frames,
        "slots": len(arrays["slot_team"]),
        "slots_per_team": [int(np.sum(arrays["slot_team"] == t)) for t in (HOME, AWAY)],
        "ball_seen_share": float(ball_seen.mean()),
        "mean_tracked_home": float(per_team[0].mean()),
        "mean_tracked_away": float(per_team[1].mean()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild the cache")
    args = parser.parse_args()

    arrays = load(force=args.force)
    print("cache: {}".format(os.path.relpath(cache_path(), ROOT)))
    for key, value in summary(arrays).items():
        print("  {:<18} {}".format(key, round(value, 4) if isinstance(value, float) else value))
