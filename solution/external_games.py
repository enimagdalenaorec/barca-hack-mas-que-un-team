"""Training matches from other public datasets, converted to the shape of our own tracking.

Every match here is from ANOTHER game, so RULES.md allows training and tuning on it freely. The
technical document must list each source, who produced its labels and how much we used:

  IDSSE  Bassek, Rein, Weber & Memmert, "An integrated dataset of spatiotemporal and event data
         in elite soccer", Scientific Data 12, 195 (2025). Bundesliga / 2. Bundesliga 2022/23,
         official DFL tracking at 25 fps. CC-BY 4.0. https://huggingface.co/datasets/pysport/idsse-data
         Labels: the DFL feed's own per-frame BallStatus (alive/dead) and BallPossession
         (home/away) -- corrected official tracking, the same kind of source our ground truth
         comes from.
  SC     SkillCorner open data, A-League 2024/25, broadcast tracking at 10 fps. MIT licence.
         https://github.com/SkillCorner/opendata
         Labels: SkillCorner's per-frame possession group (home/away), produced by their own
         software. No ball-status channel, so SC matches teach "which team", never "dead or not".

load(source, match_id) returns the keys of solution.tracking.load() plus:

    runs      [(start, stop)]  half-open spans of each half
    dead      (N,) bool        True where the ball is out of play (all False when has_dead is 0)
    owner     (N,) int8        0 home, 1 away, -1 unknown
    has_dead  ()   int         1 if `dead` is a real label for this source

Converted files go to data/external/cache/ (gitignored). Raw files already in
data/external/idsse/ are read in place and kept; raw downloads are deleted after conversion by
solution/download_external.py, so only these compact arrays stay on disk.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

# Params
RAW_DIR = os.path.join(DATA_DIR, "external", "raw")
# IDSSE files already on disk under their original names; read in place of a download.
LOCAL_IDSSE_DIR = os.path.join(DATA_DIR, "external", "idsse")
# Converted training arrays live apart from anything computed from the competition match.
CACHE_DIR = os.path.join(DATA_DIR, "external", "cache")
CACHE_VERSION = 1
FPS = 25
UNKNOWN = -1


def local_idsse(competition, match_id):
    """(info, positions) paths of an IDSSE match in LOCAL_IDSSE_DIR, or None if either is absent."""
    stem = "{}_DFL-MAT-{}.xml".format(competition, match_id)
    paths = (os.path.join(LOCAL_IDSSE_DIR, "DFL_02_01_matchinformation_" + stem),
             os.path.join(LOCAL_IDSSE_DIR, "DFL_04_03_positions_raw_observed_" + stem))
    return paths if all(os.path.isfile(p) for p in paths) else None


def cache_path(source, match_id):
    return os.path.join(CACHE_DIR, "ext_{}_{}_v{}.npz".format(source, match_id, CACHE_VERSION))


def _speeds(xy):
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * FPS
    return np.concatenate([step[:1] * np.nan, step]).astype(np.float32)


def _finish(ball_xy, players_xy, slot_team, pitch, runs, dead, owner, has_dead):
    """Common tail of every converter: speeds, dtypes, one dict layout."""
    players_xy = players_xy.astype(np.float32)
    ball_xy = ball_xy.astype(np.float32)
    return {
        "ball_xy": ball_xy,
        "ball_speed": _speeds(ball_xy),
        "players_xy": players_xy,
        "players_speed": np.stack([_speeds(players_xy[:, s]) for s in range(players_xy.shape[1])],
                                  axis=1),
        "slot_team": np.asarray(slot_team, dtype=np.int8),
        "pitch": np.asarray(pitch, dtype=np.float32),
        "runs": np.asarray(runs, dtype=np.int64),
        "dead": np.asarray(dead, dtype=bool),
        "owner": np.asarray(owner, dtype=np.int8),
        "has_dead": np.array(int(has_dead)),
        "version": np.array(CACHE_VERSION),
    }


# -- IDSSE (DFL XML) ---------------------------------------------------------------------------

def convert_idsse(info_path, positions_path):
    """DFL match information + raw observed positions -> arrays.

    DFL positions are one FrameSet per person (and one for the ball) per half, each a list of
    <Frame N=... X=... Y=.../>, with the origin at the centre spot. The ball's frames also carry
    BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 away). Referees are dropped.
    """
    info = ET.parse(info_path).getroot()
    roles = {t.get("TeamId"): t.get("Role") for t in info.iter("Team")}
    env = next(info.iter("Environment"))
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))

    sets = {}  # (section, team, person) -> dict of lists
    current = None
    for event, elem in ET.iterparse(positions_path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            key = (elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId"))
            current = sets.setdefault(key, {"n": [], "x": [], "y": [], "status": [], "poss": []})
        elif event == "end" and elem.tag == "Frame" and current is not None:
            current["n"].append(int(elem.get("N")))
            current["x"].append(float(elem.get("X")))
            current["y"].append(float(elem.get("Y")))
            if "BallStatus" in elem.attrib:
                current["status"].append(int(elem.get("BallStatus")))
                current["poss"].append(int(elem.get("BallPossession") or 0))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            current = None
            elem.clear()

    sections = sorted({k[0] for k in sets}, key=lambda s: ("first" not in s, s))
    persons = sorted({(k[1], k[2]) for k in sets if roles.get(k[1]) in ("home", "guest")})
    slot_team = [0 if roles[team] == "home" else 1 for team, _ in persons]
    is_ball = [k for k in sets if k[1].lower() == "ball"]

    blocks, runs, offset = [], [], 0
    for section in sections:
        keys = [k for k in sets if k[0] == section]
        first = min(min(sets[k]["n"]) for k in keys)
        last = max(max(sets[k]["n"]) for k in keys)
        n = last - first + 1
        players = np.full((n, len(persons), 2), np.nan)
        for slot, (team, person) in enumerate(persons):
            data = sets.get((section, team, person))
            if data:
                idx = np.array(data["n"]) - first
                players[idx, slot, 0] = data["x"]
                players[idx, slot, 1] = data["y"]
        ball = np.full((n, 2), np.nan)
        dead = np.ones(n, dtype=bool)
        owner = np.full(n, UNKNOWN, dtype=np.int8)
        for key in is_ball:
            if key[0] != section:
                continue
            data = sets[key]
            idx = np.array(data["n"]) - first
            ball[idx, 0], ball[idx, 1] = data["x"], data["y"]
            status = np.array(data["status"])
            poss = np.array(data["poss"])
            dead[idx] = status == 0
            owner[idx] = np.where(poss == 1, 0, np.where(poss == 2, 1, UNKNOWN))
        blocks.append((players, ball, dead, owner))
        runs.append((offset, offset + n))
        offset += n

    # Origin moved from the centre spot to a corner, the convention of our own tracking.
    shift = np.array(pitch) / 2.0
    players = np.concatenate([b[0] for b in blocks]) + shift
    ball = np.concatenate([b[1] for b in blocks]) + shift
    dead = np.concatenate([b[2] for b in blocks])
    owner = np.concatenate([b[3] for b in blocks])
    owner[dead] = UNKNOWN
    return _finish(ball, players, slot_team, pitch, runs, dead, owner, has_dead=True)


# -- SkillCorner (JSON lines, 10 fps) ----------------------------------------------------------

def convert_skillcorner(match_path, tracking_path):
    """SkillCorner match JSON + tracking JSONL -> arrays at 25 fps.

    Positions are interpolated from 10 to 25 fps where both neighbouring source frames have the
    object; labels take the nearest source frame. Frames the broadcast did not track (replays,
    close-ups: ~27% of a match) stay NaN with an unknown owner.
    """
    with open(match_path) as handle:
        match = json.load(handle)
    pitch = (float(match["pitch_length"]), float(match["pitch_width"]))
    home_id = match["home_team"]["id"]
    team_of = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in match["players"]}
    players_ids = sorted(team_of)
    slot_of = {pid: i for i, pid in enumerate(players_ids)}

    frames = []
    with open(tracking_path) as handle:
        for line in handle:
            frames.append(json.loads(line))
    periods = sorted({f["period"] for f in frames if f["period"] is not None})

    blocks, runs, offset = [], [], 0
    for period in periods:
        rows = [f for f in frames if f["period"] == period]
        n10 = len(rows)
        p10 = np.full((n10, len(players_ids), 2), np.nan)
        b10 = np.full((n10, 2), np.nan)
        o10 = np.full(n10, UNKNOWN, dtype=np.int8)
        for i, f in enumerate(rows):
            ball = f["ball_data"] or {}
            if ball.get("x") is not None:
                b10[i] = (ball["x"], ball["y"])
            for p in f["player_data"]:
                slot = slot_of.get(p["player_id"])
                if slot is not None:
                    p10[i, slot] = (p["x"], p["y"])
            group = (f["possession"] or {}).get("group")
            o10[i] = 0 if group == "home team" else 1 if group == "away team" else UNKNOWN

        # 10 -> 25 fps: target frame k sits at source position k * 10 / 25.
        n = int(np.floor((n10 - 1) * FPS / 10.0)) + 1
        pos = np.arange(n) * 10.0 / FPS
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n10 - 1)
        w = (pos - lo)[:, None]
        players = p10[lo] * (1 - w[:, :, None]) + p10[hi] * w[:, :, None]
        ball = b10[lo] * (1 - w) + b10[hi] * w
        owner = o10[np.rint(pos).astype(int).clip(0, n10 - 1)]
        blocks.append((players, ball, owner))
        runs.append((offset, offset + n))
        offset += n

    shift = np.array(pitch) / 2.0
    players = np.concatenate([b[0] for b in blocks]) + shift
    ball = np.concatenate([b[1] for b in blocks]) + shift
    owner = np.concatenate([b[2] for b in blocks])
    slot_team = [team_of[pid] for pid in players_ids]
    return _finish(ball, players, slot_team, pitch, runs, np.zeros(len(owner), bool), owner,
                   has_dead=False)


def load(source, match_id):
    path = cache_path(source, match_id)
    if not os.path.isfile(path):
        raise IOError("{} not converted yet -- run python -m solution.download_external".format(
            path))
    with np.load(path) as cached:
        arrays = {key: cached[key] for key in cached.files}
    arrays["runs"] = [tuple(int(v) for v in run) for run in arrays["runs"]]
    return arrays
