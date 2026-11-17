"""Download and adapt a public Metrica Sports sample game, then run our possession logic.

    python -m solution.external_games --game Sample_Game_1
    python -m solution.external_games --game Sample_Game_1 --force
    python -m solution.external_games --game Sample_Game_2 --skip-download

The adapter supports the two CSV games in https://github.com/metrica-sports/sample-data:
Sample_Game_1 and Sample_Game_2. Sample_Game_3 is the newer EPTS/JSON release and is deliberately
not parsed here; kloppy would be the right reader for it.

Nothing is downloaded until this module is run, and the files land in data/external/metrica/,
which is covered by the gated-data ignore rules. The script then converts the home/away tracking
CSVs and the event CSV into the same arrays and event table that solution/tracking.py consumes,
calls the same build_possession_frame() state machine, and writes a frame,label CSV plus shape
diagnostics. No ground truth is read or required.

Event mapping is deliberately conservative. PASS, RECOVERY, SET PIECE and SHOT name the team in
possession and become anchors directly. BALL LOST and CHALLENGE are ignored because the possessing
side has to be inferred or the two rows can overlap; the telemetry passes handle those transitions.
Goals are not converted into dead-ball windows because the sample events do not carry a reliable
stoppage interval; out-of-pitch and set-piece lead-ins still produce D.

Metrica numbers frames from 1 inside each period, so both tracking files are joined on
(period, frame) and re-indexed to one continuous 0-based row per match. That keeps the state
machine's frame indexing valid even when the second half restarts its own frame counter. Events
are translated through the same (period, frame) -> row map.
"""

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from solution import tracking as tr  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25.0
PITCH = np.array([105.0, 68.0], dtype=np.float32)
RAW_BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
SUPPORTED_GAMES = ("Sample_Game_1", "Sample_Game_2")
FILES = {
    "tracking_home": "{game}_RawTrackingData_Home_Team.csv",
    "tracking_away": "{game}_RawTrackingData_Away_Team.csv",
    "events": "{game}_RawEventsData.csv",
}
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "external", "metrica")
TEAM_CODES = {"home": tr.HOME, "away": tr.AWAY}
MAX_LOCAL_FRAME = 10_000_000  # keeps the (period, frame) key inside int64


def download(url, path, force=False):
    """Fetch `url` to `path`, skipping an existing non-empty file unless forced."""
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0 and not force:
        print("  cached    {}".format(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "barca-hack-mas-que-un-team/1.0"})
    print("  download  {}".format(url))
    with urllib.request.urlopen(request, timeout=180) as response:
        with open(temporary, "wb") as handle:
            shutil.copyfileobj(response, handle)
    os.replace(temporary, path)
    print("  saved     {} ({:.1f} MB)".format(path, path.stat().st_size / 1e6))
    return path


def _number(value, default=np.nan):
    """Best-effort float from a CSV cell; missing or malformed both become `default`."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _frame_key(period, frame):
    """Stable integer key for a Metrica (period, frame) pair."""
    return period.astype(np.int64) * MAX_LOCAL_FRAME + frame.astype(np.int64)


def read_tracking_csv(path):
    """One Metrica tracking CSV -> period, local frame, player positions and ball position.

    Metrica's CSV has three header rows (team, shirt number, column name). Player coordinates are
    alternating x/y columns from column 3 up to the `Ball` column. Everything stays normalised
    here; pitch scaling happens in build_arrays().
    """
    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    if len(raw) < 4:
        raise ValueError("{}: too few rows for a tracking CSV".format(path))
    header = [str(value).strip() for value in raw.iloc[2].tolist()]
    if "Ball" not in header:
        raise ValueError("{}: no Ball column in the third header row".format(path))
    ball_col = header.index("Ball")
    x_cols = list(range(3, ball_col, 2))
    y_cols = [column + 1 for column in x_cols]
    if not x_cols or max(y_cols) >= len(header):
        raise ValueError("{}: malformed player columns".format(path))

    data = raw.iloc[3:, :ball_col + 2].copy()
    frame = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    period = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(frame) & np.isfinite(period)
    frame = frame[keep].astype(np.int64)
    period = period[keep].astype(np.int8)
    data = data.iloc[keep]

    x = data.iloc[:, x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = data.iloc[:, y_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    players = np.stack([x, y], axis=2)
    ball = np.stack([
        pd.to_numeric(data.iloc[:, ball_col], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(data.iloc[:, ball_col + 1], errors="coerce").to_numpy(dtype=np.float64),
    ], axis=1)
    order = np.lexsort((frame, period))
    if not np.array_equal(order, np.arange(len(frame))):
        frame, period = frame[order], period[order]
        players, ball = players[order], ball[order]
    player_ids = [header[column] for column in x_cols]
    return {"frame": frame, "period": period, "players": players, "ball": ball,
            "player_ids": player_ids}


def _speeds(xy, fps):
    """Speed over the last axis from consecutive positions; first frame is unknown."""
    speed = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if len(xy) > 1:
        step = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
        speed[1:] = step.astype(np.float32) * fps
    return speed


def build_arrays(home_path, away_path, fps=FPS):
    """Combine both tracking CSVs into the dict shape solution/tracking.py expects.

    Returns `(arrays, period, frame_map)`. `frame_map` translates a Metrica (period, local frame)
    pair to the continuous 0-based row index used by the state machine.
    """
    home = read_tracking_csv(home_path)
    away = read_tracking_csv(away_path)
    home_key = _frame_key(home["period"], home["frame"])
    away_key = _frame_key(away["period"], away["frame"])
    common = np.intersect1d(home_key, away_key)
    if len(common) == 0:
        raise ValueError("home and away tracking files have no frames in common")
    if len(np.unique(common)) != len(common):
        raise ValueError("duplicate (period, frame) keys in the tracking files")

    home_index = np.searchsorted(home_key, common)
    away_index = np.searchsorted(away_key, common)
    period = home["period"][home_index]
    players = np.concatenate([home["players"][home_index], away["players"][away_index]],
                             axis=1).astype(np.float32) * PITCH
    home_ball = home["ball"][home_index].astype(np.float32) * PITCH
    away_ball = away["ball"][away_index].astype(np.float32) * PITCH
    ball = np.where(np.isfinite(home_ball), home_ball, away_ball)
    slot_team = np.concatenate([
        np.full(home["players"].shape[1], tr.HOME, dtype=np.int8),
        np.full(away["players"].shape[1], tr.AWAY, dtype=np.int8),
    ])

    frame_map = {
        (int(p), int(f)): index
        for index, (p, f) in enumerate(zip(home["period"][home_index],
                                           home["frame"][home_index]))
    }
    arrays = {
        "ball_xy": ball.astype(np.float32),
        "ball_speed": _speeds(ball, fps),
        "players_xy": players.astype(np.float32),
        "players_speed": _speeds(players, fps),
        "slot_team": slot_team,
        "pitch": PITCH.copy(),
        "version": np.array(tr.CACHE_VERSION),
    }
    return arrays, period, frame_map


def read_events_csv(path, frame_map, fps=FPS):
    """Metrica event CSV -> the columns solution/tracking.py uses for anchors.

    Only events that name the team in possession are mapped. Event frames are translated from
    Metrica's per-period 1-based numbering through `frame_map`, and a missing end frame gets a
    short five-second search window for the set-piece restart detector. The window is capped at
    the end of the event's period, so a throw-in late in the first half can never search into the
    second-half kickoff.
    """
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    required = {"Team", "Type", "Subtype", "Period", "Start Frame", "End Frame",
                "Start Time [s]", "End Time [s]"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("{}: missing event columns {}".format(path, sorted(missing)))
    events = raw.rename(columns={
        "Team": "team", "Type": "kind", "Subtype": "subtype", "Period": "period",
        "Start Frame": "start_frame", "End Frame": "end_frame",
        "Start Time [s]": "start_s", "End Time [s]": "end_s",
    })

    n_rows = len(frame_map)
    period_stop = {}
    for (period, _frame), index in frame_map.items():
        period_stop[period] = max(period_stop.get(period, 0), index + 1)
    rows = []
    for row in events.itertuples(index=False):
        team = str(row.team).strip().lower()
        if team not in TEAM_CODES:
            continue
        kind = str(row.kind).strip().upper()
        code_key = {
            "PASS": "pass",
            "RECOVERY": "interception",
            "SET PIECE": tr.SET_PIECE_CODE,
            "SHOT": "creating chances",
        }.get(kind)
        if code_key is None:
            continue

        period = _number(row.period)
        start_local = _number(row.start_frame)
        if not np.isfinite(period) or not np.isfinite(start_local):
            continue
        period = int(period)
        start_local = int(start_local)
        start_frame = frame_map.get((period, start_local))
        if start_frame is None:
            continue

        end_local = _number(row.end_frame, default=np.nan)
        end_frame = frame_map.get((period, int(end_local))) if np.isfinite(end_local) else None
        if end_frame is None or end_frame <= start_frame:
            end_frame = min(start_frame + int(5 * fps), period_stop.get(period, n_rows) - 1)
        start_s = _number(row.start_s, default=start_frame / fps)
        end_s = _number(row.end_s, default=start_s + 5.0)
        if end_s <= start_s:
            end_s = start_s + 5.0

        rows.append({
            "code": kind,
            "code_key": code_key,
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "team": row.team,
            "team_code": TEAM_CODES[team],
            "type": row.subtype,
            "half": "1st Half" if period == 1 else "2nd Half",
        })
    columns = ["code", "code_key", "start_s", "end_s", "start_frame", "end_frame",
               "team", "team_code", "type", "half"]
    return pd.DataFrame(rows, columns=columns)


def period_runs(period):
    """Half-open [start, stop) spans for each period, for shape diagnostics."""
    if len(period) == 0:
        return []
    breaks = np.flatnonzero(np.diff(period) != 0) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, len(period)]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def run_game(game, data_dir, force=False, skip_download=False,
             use_physics=True, use_contact=True, out=None):
    """Download, convert and run one sample game. Returns the output path."""
    if game not in SUPPORTED_GAMES:
        raise ValueError("this CSV adapter supports {}; {} is EPTS/JSON".format(
            list(SUPPORTED_GAMES), game))

    folder = Path(data_dir) / game
    paths = {}
    for key, pattern in FILES.items():
        filename = pattern.format(game=game)
        destination = folder / filename
        if skip_download:
            if not destination.is_file():
                raise FileNotFoundError("--skip-download but {} is missing".format(destination))
            paths[key] = destination
        else:
            url = "{}/{}/{}".format(RAW_BASE, game, filename)
            paths[key] = download(url, destination, force=force)

    arrays, period, frame_map = build_arrays(paths["tracking_home"], paths["tracking_away"])
    events = read_events_csv(paths["events"], frame_map)
    table = tr.build_possession_frame(
        arrays=arrays, events=events, fps=FPS,
        use_physics=use_physics, use_contact=use_contact)

    output = Path(out) if out else folder / "{}_possession.csv".format(game)
    output.parent.mkdir(parents=True, exist_ok=True)
    table[["frame", "state"]].rename(columns={"state": "label"}).to_csv(
        output, index=False, lineterminator="\n")

    codes = pm.label_codes(table["state"].to_numpy())
    runs = period_runs(period)
    filled = pm.fill_abstentions(codes, runs)
    spells = pm.possession_spells(filled, runs)
    durations = np.array([stop - start for start, stop, _ in spells], dtype=float) / FPS
    counts = table["state"].value_counts().reindex(tr.LABELS, fill_value=0)
    stats = pm.possession_stats(filled, runs)

    print("\n{}  {} frames  {} periods".format(game, len(table), len(runs)))
    print("  " + "  ".join("{} {:>7}".format(label, counts[label]) for label in tr.LABELS))
    print("  possessions {}  mean {:.1f} s  median {:.1f} s  home share {:.3f}".format(
        len(spells),
        float(durations.mean()) if len(durations) else 0.0,
        float(np.median(durations)) if len(durations) else 0.0,
        stats["home_share"]))
    reasons = table.loc[table["anchor_reason"] != "", "anchor_reason"].value_counts()
    print("  anchors: " + (", ".join("{}={}".format(k, v) for k, v in reasons.items())
                          if len(reasons) else "(none)"))
    print("  wrote {}".format(output))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="Sample_Game_1", choices=SUPPORTED_GAMES)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="download/cache root (default: data/external/metrica)")
    parser.add_argument("--out", default=None, help="output CSV (default: <data-dir>/<game>/)")
    parser.add_argument("--force", action="store_true", help="redownload cached files")
    parser.add_argument("--skip-download", action="store_true",
                        help="use existing local files and never touch the network")
    parser.add_argument("--no-physics", action="store_true",
                        help="skip the sharp-trajectory anchors")
    parser.add_argument("--no-contact", action="store_true",
                        help="skip the sustained-control anchors")
    args = parser.parse_args()

    run_game(args.game, args.data_dir, force=args.force, skip_download=args.skip_download,
             use_physics=not args.no_physics, use_contact=not args.no_contact, out=args.out)
