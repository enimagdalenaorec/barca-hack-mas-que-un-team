"""Download, convert, run and score one IDSSE match against official DFL possession labels.

    python -m solution.test_idsse
    python -m solution.test_idsse --skip-download

This is the independent held-out check the Metrica test could not provide. IDSSE ships the DFL
tracking feed with two official per-frame fields on the ball:

    BallStatus     0 = ball not in play, 1 = ball in play
    BallPossession 1 = home team, 2 = away team

Those fields are the ground truth. They are not derived from the event stream and the possession
state machine never sees them. The only thing fed to `build_possession_frame()` is the converted
tracking data plus a Smart Tagging XML built from the separate IDSSE event feed.

What the adapter does:

* downloads one match (default J03WOH) from the public `pysport/idsse-data` Hugging Face dataset;
* writes the exact ATD text/metadata layout `solution/tracking.py` reads;
* maps IDSSE events into the Smart Tagging instance/label layout;
* writes the official `possession_gt.csv` (frame,label) and an audit CSV with the raw DFL fields;
* passes the formatted match through `build_possession_frame()`;
* sweeps three run-length smoothing thresholds and scores macro-F1 and possession-spell F1
  against the official labels.

Event mapping, kept deliberately conservative:

    Play, OtherBallAction, Nutmeg                  -> PASS             (acting team)
    BallClaiming                                   -> INTERCEPTION     (claiming team)
    TacklingGame with PossessionChange=true        -> TACKLE           (WinnerTeam)
    ShotAtGoal / Penalty / ChanceWithoutShot       -> CREATING CHANCES (shooting team)
    ThrowIn, FreeKick, GoalKick, CornerKick,
    KickOff, Penalty                               -> SET PIECES
    SuccessfulShot                                 -> GOALS window from the shot to the next kickoff
    Foul, Offside                                  -> GOALS stoppage window to the next free kick,
                                                      penalty or kickoff

Cards, substitutions, deleted events and VAR reviews are not used as independent stoppage starts:
a card is administered inside the foul stoppage, and a VAR review is either already inside a goal
window or did not stop play. This avoids manufacturing dead frames from events that do not stop the
game.

The IDSSE kickoff/free-kick/corner records carry `StartFrame`/`EndFrame`; those are used verbatim.
Throw-ins and goal kicks do not, so their dead window starts at `DecisionTimestamp` (capped at 12 s
before the action, which keeps the real restart inside tracking.py's 15 s restart search) and ends
five seconds after the action.

The state machine now also uses a deterministic kinematic influence field: player velocity vectors
and a constant-velocity time-to-intercept (TTI) rank who can reach the projected ball first. That
field resolves Team=N/A set pieces, gates physics touches on kinematic agreement, and vetoes
anchors that contradict the projected interception of a fast loose ball.

The final state gets a deterministic run-length smoothing pass
(`tracking.smooth_micro_possessions`): contiguous H/A runs shorter than the threshold are absorbed
by the previous stable state, while D runs are preserved and short physics-backed runs are
protected. The active evaluation uses 18 frames (0.72 s); the sweep also reports 25, 37 and 50
frames for comparison.

All generated files land under data/external/idsse/<match>/, which is gated by `.gitignore`
(`data/**`, `*.csv`, `*.mp4`). No commit or push is performed.
"""

import argparse
import io
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
from solution import tracking as tr  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25.0
HOME, AWAY = tr.HOME, tr.AWAY
MATCH_ID = "J03WOH"
BENCHMARK = 0.645
SMOOTH_THRESHOLDS = (18, 25, 37, 50)
DEFAULT_SMOOTH = 18
HF_BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
DEFAULT_DATA_DIR = ROOT / "data" / "external" / "idsse"
TEAM_NAMES = {HOME: "Home", AWAY: "Away", -1: "N/A"}
TEAM_CODES = {"home": HOME, "away": AWAY}
FILE_PATTERNS = {
    "meta": "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-{match}.xml",
    "events": "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-{match}.xml",
    "positions": "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-{match}.xml",
}
MIN_BYTES = {"meta": 5_000, "events": 100_000, "positions": 300_000_000}

# Same-frame anchor precedence: higher priority is written later, so the more specific event wins.
ANCHOR_PRIORITY = {
    "PASS": 10,
    "CREATING CHANCES": 20,
    "TACKLE": 30,
    "INTERCEPTION": 40,
    "SET PIECES": 50,
    "GOALS": 60,
}
SET_PIECE_TAGS = {"ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff", "Penalty"}
STOPPAGE_TAGS = {"Foul", "Offside"}
DEAD_SET_PIECE_TAGS = {"FreeKick", "Penalty", "KickOff"}
KICKOFF_TAG = "KickOff"

# A set piece without StartFrame is searched from at most 12 s before the action, so the restart is
# inside SET_PIECE_MAX_WAIT_S (15 s). The tag end runs five seconds past the action.
NO_FRAME_LEAD_IN_S = 12.0
NO_FRAME_TAIL_S = 5.0
KICKOFF_LEAD_IN_S = 1.0


def _log(message):
    print(message, flush=True)


def download_file(url, destination, minimum, force=False):
    """Download one file atomically, reusing a complete cached copy."""
    destination = Path(destination)
    if not force and destination.is_file() and destination.stat().st_size >= minimum:
        _log("cached  {}  ({:.1f} MB)".format(destination.name,
                                               destination.stat().st_size / 1e6))
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    _log("downloading {} ...".format(destination.name))
    request = urllib.request.Request(url, headers={"User-Agent": "opencode-idsse-check/1.0"})
    downloaded = 0
    milestone = 0
    with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if downloaded >= milestone + 50 * (1 << 20):
                milestone = downloaded
                if total:
                    _log("  {:.1f} / {:.1f} MB".format(downloaded / 1e6, total / 1e6))
                else:
                    _log("  {:.1f} MB".format(downloaded / 1e6))
    if temporary.stat().st_size < minimum:
        raise RuntimeError("{}: downloaded {} bytes, expected at least {}".format(
            destination.name, temporary.stat().st_size, minimum))
    temporary.replace(destination)
    _log("wrote   {}  ({:.1f} MB)".format(destination.name,
                                          destination.stat().st_size / 1e6))
    return destination


def fetch_match(match_id, data_dir, skip_download=False, force=False):
    """Return the three local IDSSE XML paths, downloading them unless told not to."""
    folder = Path(data_dir) / match_id
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, pattern in FILE_PATTERNS.items():
        filename = pattern.format(match=match_id)
        destination = folder / filename
        if skip_download:
            if not destination.is_file():
                raise FileNotFoundError("--skip-download but {} is missing".format(destination))
            if destination.stat().st_size < MIN_BYTES[key]:
                raise RuntimeError("{} is truncated ({} bytes)".format(
                    destination, destination.stat().st_size))
            _log("cached  {}  ({:.1f} MB)".format(destination.name,
                                                   destination.stat().st_size / 1e6))
            paths[key] = destination
        else:
            paths[key] = download_file("{}/{}".format(HF_BASE, filename), destination,
                                       MIN_BYTES[key], force=force)
    return paths


def read_match_info(path):
    """Team ids/names and pitch dimensions from the official match-information XML."""
    root = ET.parse(str(path)).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    if general is None or environment is None:
        raise ValueError("{}: missing General/Environment".format(path))
    info = {
        "match_id": general.get("MatchId"),
        "home_id": general.get("HomeTeamId"),
        "away_id": general.get("GuestTeamId"),
        "home_name": general.get("HomeTeamName"),
        "away_name": general.get("GuestTeamName"),
        "result": general.get("Result"),
        "pitch": (float(environment.get("PitchX")), float(environment.get("PitchY"))),
        "stadium": environment.get("StadiumName"),
    }
    if info["home_id"] is None or info["away_id"] is None:
        raise ValueError("{}: missing team ids".format(path))
    return info


def _number(value):
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_positions(path, home_id, away_id, pitch):
    """Read the player-major DFL position XML into normalised per-frame arrays.

    The ball FrameSet carries the official `BallStatus` and `BallPossession` fields. The player
    FrameSets are one tracklet each; a player who is not on the pitch is simply absent from that
    frame, which becomes NaN in the slot array, exactly like the ATD representation.
    """
    pitch_x, pitch_y = float(pitch[0]), float(pitch[1])
    ball_parts = []
    player_sets = []
    frame_set_count = 0
    started = time.perf_counter()

    for _event, elem in etree.iterparse(str(path), tag="FrameSet", huge_tree=True):
        frame_set_count += 1
        team = elem.get("TeamId")
        person = elem.get("PersonId")
        section = elem.get("GameSection")
        frames = elem.findall("Frame")
        if not frames:
            elem.clear()
            continue

        count = len(frames)
        ns = np.empty(count, dtype=np.int64)
        xs = np.empty(count, dtype=np.float64)
        ys = np.empty(count, dtype=np.float64)
        speeds = np.empty(count, dtype=np.float64)

        if team == "BALL":
            status = np.empty(count, dtype=np.int8)
            possession = np.empty(count, dtype=np.int8)
            times = []
            for index, frame in enumerate(frames):
                ns[index] = int(frame.get("N"))
                xs[index] = _number(frame.get("X"))
                ys[index] = _number(frame.get("Y"))
                speeds[index] = _number(frame.get("S"))
                status[index] = int(frame.get("BallStatus"))
                possession[index] = int(frame.get("BallPossession"))
                times.append(frame.get("T"))
            ball_parts.append((section, ns, xs, ys, speeds, status, possession, times))
        else:
            if team not in (home_id, away_id):
                raise ValueError("{}: unknown TeamId {!r}".format(path, team))
            for index, frame in enumerate(frames):
                ns[index] = int(frame.get("N"))
                xs[index] = _number(frame.get("X"))
                ys[index] = _number(frame.get("Y"))
                speeds[index] = _number(frame.get("S"))
            player_sets.append((team, person, section, ns, xs, ys, speeds))

        elem.clear()
        if frame_set_count % 10 == 0:
            _log("  parsed {} FrameSets ({:.0f}s)".format(
                frame_set_count, time.perf_counter() - started))

    if len(ball_parts) != 2:
        raise ValueError("{}: expected 2 ball FrameSets, found {}".format(
            path, len(ball_parts)))

    ball_n = np.concatenate([part[1] for part in ball_parts])
    ball_x = np.concatenate([part[2] for part in ball_parts])
    ball_y = np.concatenate([part[3] for part in ball_parts])
    ball_speed = np.concatenate([part[4] for part in ball_parts])
    ball_status = np.concatenate([part[5] for part in ball_parts])
    ball_possession = np.concatenate([part[6] for part in ball_parts])
    ball_times = [stamp for part in ball_parts for stamp in part[7]]
    ball_section = np.concatenate([
        np.full(len(part[1]), part[0], dtype=object) for part in ball_parts
    ])

    n_frames = len(ball_n)
    if not np.all(np.diff(ball_n) > 0):
        raise ValueError("{}: ball frames are not strictly increasing".format(path))
    if len(ball_times) != n_frames:
        raise ValueError("{}: timestamp count {} != frames {}".format(
            path, len(ball_times), n_frames))

    ball_time = pd.to_datetime(ball_times, utc=True)
    # Normalise explicitly to microseconds. Depending on the input precision pandas may infer
    # datetime64[us] or datetime64[ns]; `.astype("int64")` then changes units underneath us.
    ball_time_us = (ball_time.tz_convert("UTC").tz_localize(None)
                    .astype("datetime64[us]").astype("int64").to_numpy())
    if not np.all(np.diff(ball_time_us) > 0):
        raise ValueError("{}: ball timestamps are not strictly increasing".format(path))

    half1_stop = int(np.sum(ball_section == "firstHalf"))
    if half1_stop <= 0 or half1_stop >= n_frames:
        raise ValueError("{}: could not split the two halves".format(path))
    if not np.all(ball_section[:half1_stop] == "firstHalf") \
            or not np.all(ball_section[half1_stop:] == "secondHalf"):
        raise ValueError("{}: half sections are not contiguous".format(path))

    ball_norm = np.empty((n_frames, 2), dtype=np.float32)
    ball_norm[:, 0] = (ball_x + pitch_x / 2.0) / pitch_x
    ball_norm[:, 1] = (ball_y + pitch_y / 2.0) / pitch_y

    home_people = []
    away_people = []
    for team, person, _section, _n, _x, _y, _s in player_sets:
        bucket = home_people if team == home_id else away_people
        if person not in bucket:
            bucket.append(person)
    slot_people = home_people + away_people
    n_slots = len(slot_people)
    slot_team = np.array([HOME] * len(home_people) + [AWAY] * len(away_people),
                         dtype=np.int8)
    person_index = {person: index for index, person in enumerate(slot_people)}

    players_norm = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    players_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    for _team, person, _section, ns, xs, ys, speeds in player_sets:
        slot = person_index[person]
        rows = np.searchsorted(ball_n, ns)
        if np.any(rows >= n_frames) or not np.array_equal(ball_n[rows], ns):
            raise ValueError("player {} has frames outside the ball timeline".format(person))
        players_norm[rows, slot, 0] = (xs + pitch_x / 2.0) / pitch_x
        players_norm[rows, slot, 1] = (ys + pitch_y / 2.0) / pitch_y
        players_speed[rows, slot] = speeds

    _log("tracking: {} frames  ({} first half, {} second half)".format(
        n_frames, half1_stop, n_frames - half1_stop))
    _log("tracking: {} slots  ({} home, {} away)".format(
        n_slots, len(home_people), len(away_people)))
    _log("tracking: ball seen on {:.1f}% of frames".format(
        100.0 * np.mean(np.isfinite(ball_norm[:, 0]))))

    return {
        "n_frames": n_frames,
        "half1_stop": half1_stop,
        "ball_raw_n": ball_n,
        "ball_times": ball_times,
        "ball_time_us": ball_time_us,
        "ball_norm": ball_norm,
        "ball_speed": ball_speed.astype(np.float32),
        "ball_status": ball_status,
        "ball_possession": ball_possession,
        "players_norm": players_norm,
        "players_speed": players_speed,
        "slot_people": slot_people,
        "slot_team": slot_team,
        "pitch": np.array([pitch_x, pitch_y], dtype=np.float32),
        "raw_index": {int(n): index for index, n in enumerate(ball_n)},
    }


def write_atd(raw_path, meta_path, tracking, home_name, away_name):
    """Write the exact `frame:player;player:ball` text and `Team`/`Player` metadata layout."""
    players = tracking["players_norm"]
    ball = tracking["ball_norm"]
    player_speed = tracking["players_speed"]
    ball_speed = tracking["ball_speed"]
    slot_team = tracking["slot_team"]
    slot_people = tracking["slot_people"]
    n_frames, n_slots = players.shape[0], players.shape[1]

    def fmt(value):
        return "NaN" if not math.isfinite(float(value)) else "{:.8f}".format(float(value))

    started = time.perf_counter()
    with open(raw_path, "w", newline="\n") as handle:
        chunk = []
        for frame in range(n_frames):
            parts = []
            for slot in range(n_slots):
                x, y = players[frame, slot]
                parts.append("{},{},{}".format(fmt(x), fmt(y),
                                               fmt(player_speed[frame, slot])))
            bx, by = ball[frame]
            parts.append("{},{},{}".format(fmt(bx), fmt(by), fmt(ball_speed[frame])))
            chunk.append("{}:{}".format(frame + 1, ";".join(parts)))
            if len(chunk) >= 5000:
                handle.write("\n".join(chunk) + "\n")
                chunk = []
        if chunk:
            handle.write("\n".join(chunk) + "\n")

    root = ET.Element("main")
    teams = ET.SubElement(root, "Teams")
    ET.SubElement(teams, "Team", {"id": "home"}).text = home_name
    ET.SubElement(teams, "Team", {"id": "away"}).text = away_name
    players_el = ET.SubElement(root, "Players")
    for slot, person in enumerate(slot_people):
        team_id = "home" if slot_team[slot] == HOME else "away"
        ET.SubElement(players_el, "Player", {"id": person, "teamId": team_id}).text = person
    ET.ElementTree(root).write(str(meta_path), encoding="utf-8", xml_declaration=True)
    _log("ATD: wrote {} rows, {} slots in {:.1f}s".format(
        n_frames, n_slots, time.perf_counter() - started))


def read_atd(raw_path, meta_path, pitch, n_frames):
    """Read the generated ATD exactly as `tracking.parse()` would, without a GAMES entry."""
    slot_team = tr.slot_teams(str(meta_path))
    n_slots = len(slot_team)
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        engine="c").to_numpy()
    del text

    expected_cols = 1 + 3 * n_slots + 3
    if table.shape != (n_frames, expected_cols):
        raise ValueError("generated ATD has shape {}, expected ({}, {})".format(
            table.shape, n_frames, expected_cols))
    frame_ids = table[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("generated ATD frame ids are not 1..{} in order".format(n_frames))

    scale = np.asarray(pitch, dtype=np.float64)
    players = table[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)
    ball = table[:, 1 + 3 * n_slots:]
    return {
        "ball_xy": (ball[:, :2] * scale).astype(np.float32),
        "ball_speed": ball[:, 2].astype(np.float32),
        "players_xy": (players[:, :, :2] * scale).astype(np.float32),
        "players_speed": players[:, :, 2].astype(np.float32),
        "slot_team": slot_team,
        "pitch": scale.astype(np.float32),
        "version": np.array(tr.CACHE_VERSION),
    }


EVENT_COLUMNS = ["code", "code_key", "start_s", "end_s", "start_frame", "end_frame",
                 "team", "team_code", "type", "half"]


def load_events(path):
    """All top-level IDSSE actions with the fields the adapter needs."""
    root = ET.parse(str(path)).getroot()
    rows = []
    for event in root.iter("Event"):
        actions = list(event)
        if not actions:
            continue
        action = actions[0]
        tag = action.tag
        event_time = event.get("EventTime")
        if not event_time:
            continue

        team_id = action.get("Team")
        decision_time = action.get("DecisionTimestamp")
        calculated_frame = event.get("CalculatedFrame") or action.get("CalculatedFrame")
        possession_change = action.get("PossessionChange")

        if tag == "TacklingGame":
            team_id = action.get("WinnerTeam")
        elif tag == "KickOff":
            play = action.find("Play")
            if play is not None:
                team_id = play.get("Team") or team_id
        elif tag in ("ShotAtGoal", "Penalty"):
            shot = action if tag == "ShotAtGoal" else action.find("ShotAtGoal")
            if shot is not None:
                team_id = shot.get("Team") or team_id
                calculated_frame = calculated_frame or shot.get("CalculatedFrame")
                decision_time = decision_time or shot.get("DecisionTimestamp")
        elif tag == "Foul":
            team_id = action.get("TeamFouled") or action.get("TeamFouler") or team_id

        event_stamp = pd.Timestamp(event_time)
        decision_stamp = pd.Timestamp(decision_time) if decision_time else None
        rows.append({
            "tag": tag,
            "event_id": event.get("EventId"),
            "event_time": event_time,
            "event_us": event_stamp.value // 1000,
            "decision_us": decision_stamp.value // 1000 if decision_stamp else None,
            "team_id": team_id,
            "start_frame": int(event.get("StartFrame"))
            if event.get("StartFrame") is not None else None,
            "end_frame": int(event.get("EndFrame"))
            if event.get("EndFrame") is not None else None,
            "calculated_frame": int(calculated_frame)
            if calculated_frame is not None else None,
            "possession_change": (possession_change or "").strip().lower() == "true",
            "is_goal": next(event.iter("SuccessfulShot"), None) is not None,
        })
    rows.sort(key=lambda row: (row["event_us"], row["event_id"] or ""))
    return rows


def _nearest_row(time_us, tracking):
    times = tracking["ball_time_us"]
    index = int(np.searchsorted(times, time_us))
    if index <= 0:
        return 0
    if index >= len(times):
        return len(times) - 1
    before, after = index - 1, index
    if time_us - times[before] <= times[after] - time_us:
        return before
    return after


def _row_for_raw(raw_frame, event_us, tracking):
    row = tracking["raw_index"].get(int(raw_frame))
    if row is not None:
        return row
    return _nearest_row(event_us, tracking)


def _event_row(event, tracking):
    if event["calculated_frame"] is not None:
        return _row_for_raw(event["calculated_frame"], event["event_us"], tracking)
    return _nearest_row(event["event_us"], tracking)


def _set_piece_interval(event, tracking):
    """[start, stop) rows for a SET PIECES instance, clamped to the current half."""
    n_frames = tracking["n_frames"]
    half1_stop = tracking["half1_stop"]
    event_row = _nearest_row(event["event_us"], tracking)
    period_start = 0 if event_row < half1_stop else half1_stop

    if event["start_frame"] is not None:
        start = _row_for_raw(event["start_frame"], event["event_us"], tracking)
        if event["end_frame"] is not None:
            end = _row_for_raw(event["end_frame"], event["event_us"], tracking)
        else:
            end = start + int(round(15.0 * FPS))
    else:
        if event["decision_us"] is not None:
            decision_row = _nearest_row(event["decision_us"], tracking)
            start = max(decision_row, event_row - int(round(NO_FRAME_LEAD_IN_S * FPS)))
        else:
            start = event_row - int(round(KICKOFF_LEAD_IN_S * FPS))
        end = event_row + int(round(NO_FRAME_TAIL_S * FPS))

    start = max(period_start, min(n_frames - 1, start))
    end = max(start + 1, min(n_frames, end))
    return start, end


def build_instances(events, tracking, home_id, away_id):
    """IDSSE actions expressed as Smart Tagging rows for the unchanged anchor parser."""
    n_frames = tracking["n_frames"]
    half1_stop = tracking["half1_stop"]
    team_codes = {home_id: HOME, away_id: AWAY}
    instances = []
    set_pieces = []

    def add(code, team_code, start, end, kind):
        start = max(0, min(n_frames - 1, int(start)))
        end = max(start + 1, min(n_frames, int(end)))
        if end <= start:
            return
        instances.append({
            "code": code,
            "code_key": code.lower(),
            "start_s": start / FPS,
            "end_s": end / FPS,
            "start_frame": start,
            "end_frame": end,
            "team": TEAM_NAMES.get(int(team_code), "N/A"),
            "team_code": int(team_code),
            "type": kind,
            "half": "1st Half" if start < half1_stop else "2nd Half",
        })

    for event in events:
        tag = event["tag"]
        team_code = team_codes.get(event["team_id"], -1)
        actionable = tag in ("Play", "OtherBallAction", "Nutmeg", "BallClaiming",
                             "TacklingGame", "ShotAtGoal", "Penalty", "ChanceWithoutShot")
        if actionable:
            row = _event_row(event, tracking)
            if tag in ("Play", "OtherBallAction", "Nutmeg") and team_code >= 0:
                add("PASS", team_code, row, row + 1, tag)
            elif tag == "BallClaiming" and team_code >= 0:
                add("INTERCEPTION", team_code, row, row + 1, tag)
            elif tag == "TacklingGame" and event["possession_change"] and team_code >= 0:
                add("TACKLE", team_code, row, row + 1, tag)
            elif tag in ("ShotAtGoal", "Penalty", "ChanceWithoutShot") and team_code >= 0:
                add("CREATING CHANCES", team_code, row, row + 1, tag)

        if tag in SET_PIECE_TAGS:
            start, end = _set_piece_interval(event, tracking)
            set_pieces.append({"start": start, "end": end, "tag": tag})
            add("SET PIECES", team_code, start, end, tag)

    set_pieces.sort(key=lambda item: (item["start"], item["tag"]))
    set_starts = np.array([item["start"] for item in set_pieces], dtype=np.int64)
    kickoffs = [item for item in set_pieces if item["tag"] == KICKOFF_TAG]
    kickoff_starts = np.array([item["start"] for item in kickoffs], dtype=np.int64)

    # A foul or offside stops play until the next free kick/penalty/kickoff. If the next set piece
    # is a throw-in, corner or goal kick instead, the foul was probably played on (advantage), so
    # do not manufacture a dead window.
    for event in events:
        tag = event["tag"]
        if tag not in STOPPAGE_TAGS:
            continue
        row = _nearest_row(event["event_us"], tracking)
        index = int(np.searchsorted(set_starts, row, side="right"))
        if index >= len(set_starts):
            continue
        next_start = int(set_starts[index])
        next_tag = set_pieces[index]["tag"]
        allowed = DEAD_SET_PIECE_TAGS if tag == "Foul" else {"FreeKick", "Penalty"}
        if next_tag in allowed and next_start > row:
            add("GOALS", -1, row, next_start, "Stoppage")

    for event in events:
        if not event["is_goal"]:
            continue
        start = _event_row(event, tracking)
        index = int(np.searchsorted(kickoff_starts, start, side="right"))
        end = int(kickoff_starts[index]) if index < len(kickoff_starts) else n_frames
        if end > start:
            add("GOALS", -1, start, end, "Goal")

    instances.sort(key=lambda item: (item["start_frame"],
                                     ANCHOR_PRIORITY.get(item["code"], 0)))
    return pd.DataFrame(instances, columns=EVENT_COLUMNS)


def write_smart_tagging(path, events):
    root = ET.Element("file")
    all_instances = ET.SubElement(root, "ALL_INSTANCES")
    for index, row in enumerate(events.to_dict("records"), start=1):
        node = ET.SubElement(all_instances, "instance")
        ET.SubElement(node, "ID").text = str(index)
        ET.SubElement(node, "start").text = "{:.3f}".format(float(row["start_s"]))
        ET.SubElement(node, "end").text = "{:.3f}".format(float(row["end_s"]))
        ET.SubElement(node, "code").text = str(row["code"])
        labels = (
            ("Team", TEAM_NAMES.get(int(row["team_code"]), "N/A")),
            ("Type", str(row["type"])),
            ("Half", str(row["half"])),
        )
        for group, text in labels:
            label = ET.SubElement(node, "label")
            ET.SubElement(label, "group").text = group
            ET.SubElement(label, "text").text = text
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def read_smart_tagging(path):
    """The same DataFrame shape `tracking.load_events()` produces, for the generated XML."""
    rows = []
    for instance in ET.parse(str(path)).getroot().iter("instance"):
        code = (instance.findtext("code") or "").strip()
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        team = tags.get("Team")
        rows.append({
            "code": code,
            "code_key": code.lower(),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * FPS)),
            "end_frame": int(round(end_s * FPS)),
            "team": team,
            "team_code": TEAM_CODES.get((team or "").strip().lower(), -1),
            "type": tags.get("Type"),
            "half": tags.get("Half"),
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def check_ignore_rules():
    """Fail loudly if the generated match data would no longer be gitignored."""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required = ("data/**", "*.csv", "*.mp4")
    missing = [pattern for pattern in required if pattern not in text]
    if missing:
        raise RuntimeError(".gitignore no longer covers: {}".format(", ".join(missing)))


def official_ground_truth(tracking):
    """Official DFL labels: BallStatus=0 -> D, else BallPossession 1 -> H, 2 -> A."""
    status = tracking["ball_status"].astype(np.int16)
    possession = tracking["ball_possession"].astype(np.int16)
    if not set(np.unique(status)).issubset({0, 1}):
        raise ValueError("unexpected BallStatus values: {}".format(sorted(set(status))))
    active = status == 1
    if np.any(active & ~np.isin(possession, [1, 2])):
        raise ValueError("active ball frames without BallPossession 1/2")

    labels = np.full(len(status), "D", dtype=object)
    labels[active & (possession == 1)] = "H"
    labels[active & (possession == 2)] = "A"
    counts = Counter(labels.tolist())
    return labels, {
        "H": int(counts.get("H", 0)),
        "A": int(counts.get("A", 0)),
        "D": int(counts.get("D", 0)),
        "active_home": int(np.sum(active & (possession == 1))),
        "active_away": int(np.sum(active & (possession == 2))),
        "inactive_home": int(np.sum(~active & (possession == 1))),
        "inactive_away": int(np.sum(~active & (possession == 2))),
    }


def score(gt_labels, pred_labels, runs):
    gt_codes = pm.label_codes(gt_labels)
    pred_codes = pm.label_codes(pred_labels)
    cm = pm.confusion(gt_codes, pred_codes)
    cm3 = pm.scored_confusion(cm)
    filled = pm.fill_abstentions(pred_codes, runs)
    spells = pm.spell_scores(gt_codes, filled, runs)
    gt_stats = pm.possession_stats(gt_codes, runs)
    pred_stats = pm.possession_stats(filled, runs)

    per_class = {}
    for index, label in enumerate(("H", "A", "D")):
        precision, recall, f1 = pm.per_class_prf(cm3, index)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm3[index, :].sum()),
        }

    transitions = {}
    for tolerance in pm.TAUS:
        detail = pm.transition_scores(
            pm.transitions(gt_codes, runs), pm.transitions(filled, runs), tolerance)
        transitions[str(tolerance)] = {
            key: value for key, value in detail.items()
            if key not in ("pairs", "missed", "spurious")
        }

    return {
        "macro_f1": pm.macro_f1(cm3),
        "spell_f1_mean": spells["mean_f1"],
        "spell_f1_at_0.5": spells["per_iou"][0.5]["f1"],
        "spell_iou": {str(key): value["f1"] for key, value in spells["per_iou"].items()},
        "n_gt_spells": spells["n_gt"],
        "n_pred_spells": spells["n_pred"],
        "gt_mean_spell_duration_s": gt_stats["mean_duration_s"],
        "pred_mean_spell_duration_s": pred_stats["mean_duration_s"],
        "gt_home_share": gt_stats["home_share"],
        "pred_home_share": pred_stats["home_share"],
        "accuracy": pm.accuracy(cm3),
        "alive_dead_f1": pm.alive_dead_f1(cm3),
        "per_class": per_class,
        "transitions": transitions,
        "confusion": cm.tolist(),
        "n_frames": int(len(gt_codes)),
        "n_graded": int(cm3.sum()),
    }


def print_scoreboard(result, match_id, info, benchmark, smooth_min_frames=None):
    if smooth_min_frames is None:
        smooth_min_frames = result.get("smooth_min_frames")
    print("\n=== IDSSE {} independent evaluation ===".format(match_id))
    print("{} vs {}  ({})".format(info["home_name"], info["away_name"], info["result"]))
    if smooth_min_frames is not None:
        print("smoothing  min_frames={}  ({:.2f} s)".format(
            smooth_min_frames, float(smooth_min_frames) / FPS))
    print("official DFL BallStatus/BallPossession ground truth")
    print("frames {}  graded {}".format(result["n_frames"], result["n_graded"]))
    print("official labels  H {}  A {}  D {}".format(
        result["gt_counts"]["H"], result["gt_counts"]["A"], result["gt_counts"]["D"]))
    print("possessions  GT {}  pred {}  (mean {:.1f}s vs {:.1f}s)".format(
        result["n_gt_spells"], result["n_pred_spells"],
        result["gt_mean_spell_duration_s"], result["pred_mean_spell_duration_s"]))
    print("anchors  {}".format("  ".join(
        "{}={}".format(code, count) for code, count in result["anchor_counts"].items())))

    print("\n{:<12} {:>10}".format("metric", "score"))
    print("-" * 24)
    print("{:<12} {:>10.4f}".format("macroF1", result["macro_f1"]))
    print("{:<12} {:>10.4f}".format("spellF1", result["spell_f1_mean"]))
    print("{:<12} {:>10.4f}".format("spell@0.5", result["spell_f1_at_0.5"]))
    print("{:<12} {:>10.4f}".format("alive/dead", result["alive_dead_f1"]))
    print("{:<12} {:>10.4f}".format("accuracy", result["accuracy"]))
    print("{:<12} {:>10.4f}".format("home share", result["pred_home_share"]))
    print("{:<12} {:>10.4f}".format("GT home share", result["gt_home_share"]))

    print("\n{:<4} {:>9} {:>9} {:>9} {:>9}".format(
        "cls", "precision", "recall", "f1", "support"))
    print("-" * 46)
    for label in ("H", "A", "D"):
        row = result["per_class"][label]
        print("{:<4} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}".format(
            label, row["precision"], row["recall"], row["f1"], row["support"]))

    print("\nspellF1 by IoU")
    print("  " + "  ".join("{}:{:.4f}".format(key, value)
                           for key, value in result["spell_iou"].items()))

    print("\ntransition F1  " + "  ".join(
        "{}f:{:.4f}".format(tol, row["f1"])
        for tol, row in result["transitions"].items()))

    print("\nbenchmark {:.4f}  macro delta {:+.4f}  spell delta {:+.4f}".format(
        benchmark, result["macro_f1"] - benchmark, result["spell_f1_mean"] - benchmark))


def run(match_id, data_dir, skip_download=False, force=False, benchmark=BENCHMARK):
    started = time.perf_counter()
    check_ignore_rules()
    print("IDSSE check: all generated files stay under {}".format(Path(data_dir) / match_id))
    print("IDSSE check: .gitignore covers data/**, *.csv and *.mp4; no commit or push is performed")

    paths = fetch_match(match_id, data_dir, skip_download=skip_download, force=force)
    info = read_match_info(paths["meta"])
    print("match: {} vs {}  pitch {:.1f}x{:.1f}".format(
        info["home_name"], info["away_name"], info["pitch"][0], info["pitch"][1]))

    tracking = parse_positions(paths["positions"], info["home_id"], info["away_id"],
                               info["pitch"])
    folder = Path(data_dir) / match_id
    atd_raw = folder / "{}_atd.txt".format(match_id)
    atd_meta = folder / "{}_atd.xml".format(match_id)
    smart_tagging = folder / "{}_smart_tagging.xml".format(match_id)
    gt_csv = folder / "possession_gt.csv"
    official_csv = folder / "official_ball_labels.csv"
    pred_csv = folder / "possession_pred.csv"
    report_json = folder / "{}_idsse_report.json".format(match_id)

    write_atd(atd_raw, atd_meta, tracking, info["home_name"], info["away_name"])
    arrays = read_atd(atd_raw, atd_meta, tracking["pitch"], tracking["n_frames"])

    expected_players = (tracking["players_norm"].astype(np.float64)
                        * tracking["pitch"].astype(np.float64))
    expected_ball = (tracking["ball_norm"].astype(np.float64)
                     * tracking["pitch"].astype(np.float64))
    if not np.allclose(arrays["players_xy"], expected_players, equal_nan=True, atol=1e-4):
        raise ValueError("ATD round-trip changed player positions")
    if not np.allclose(arrays["ball_xy"], expected_ball, equal_nan=True, atol=1e-4):
        raise ValueError("ATD round-trip changed ball positions")
    if not np.array_equal(arrays["slot_team"], tracking["slot_team"]):
        raise ValueError("ATD metadata changed the slot-to-team mapping")

    events = load_events(paths["events"])
    events_smart = build_instances(events, tracking, info["home_id"], info["away_id"])
    write_smart_tagging(smart_tagging, events_smart)
    events_read = read_smart_tagging(smart_tagging)
    for column in ("code", "start_frame", "end_frame", "team_code"):
        if not np.array_equal(events_smart[column].to_numpy(), events_read[column].to_numpy()):
            raise ValueError("Smart Tagging round-trip changed {}".format(column))
    print("events: {} IDSSE actions -> {} Smart Tagging instances".format(
        len(events), len(events_read)))

    gt_labels, gt_counts = official_ground_truth(tracking)
    pd.DataFrame({"frame": np.arange(tracking["n_frames"], dtype=np.int32),
                  "label": gt_labels}).to_csv(gt_csv, index=False, lineterminator="\n")
    pd.DataFrame({
        "frame": np.arange(tracking["n_frames"], dtype=np.int32),
        "source_frame": tracking["ball_raw_n"],
        "timestamp": tracking["ball_times"],
        "ball_status": tracking["ball_status"],
        "ball_possession": tracking["ball_possession"],
        "label": gt_labels,
    }).to_csv(official_csv, index=False, lineterminator="\n")

    runs = [(0, tracking["half1_stop"]),
            (tracking["half1_stop"], tracking["n_frames"])]
    runs = [(start, stop) for start, stop in runs if stop > start]
    anchor_counts = {
        str(code): int(count) for code, count in events_read["code"].value_counts().items()
    }

    variants = {}
    for smooth_min_frames in SMOOTH_THRESHOLDS:
        print("\nrunning build_possession_frame(smooth_min_frames={}) ...".format(
            smooth_min_frames))
        table = tr.build_possession_frame(
            arrays=arrays, events=events_read, fps=FPS,
            smooth_min_frames=smooth_min_frames)
        pred_labels = table["state"].to_numpy()
        variant_csv = folder / "possession_pred_smooth{}.csv".format(smooth_min_frames)
        pd.DataFrame({"frame": table["frame"], "label": pred_labels}).to_csv(
            variant_csv, index=False, lineterminator="\n")
        if smooth_min_frames == DEFAULT_SMOOTH:
            pd.DataFrame({"frame": table["frame"], "label": pred_labels}).to_csv(
                pred_csv, index=False, lineterminator="\n")

        result = score(gt_labels, pred_labels, runs)
        result["match"] = match_id
        result["source"] = "IDSSE / DFL official BallStatus+BallPossession"
        result["home_team"] = info["home_name"]
        result["away_team"] = info["away_name"]
        result["final_score"] = info["result"]
        result["half1_stop"] = int(tracking["half1_stop"])
        result["runs"] = [[int(start), int(stop)] for start, stop in runs]
        result["gt_counts"] = gt_counts
        result["anchor_counts"] = anchor_counts
        result["smooth_min_frames"] = int(smooth_min_frames)
        result["smooth_min_seconds"] = float(smooth_min_frames) / FPS
        result["benchmark"] = float(benchmark)
        result["benchmark_macro_delta"] = result["macro_f1"] - benchmark
        result["benchmark_spell_delta"] = result["spell_f1_mean"] - benchmark
        variants[str(smooth_min_frames)] = result
        print_scoreboard(result, match_id, info, benchmark, smooth_min_frames)

    print("\n=== IDSSE smoothing sweep ===")
    print("{:>10} {:>8} {:>10} {:>10} {:>12}".format(
        "min_frames", "seconds", "macroF1", "spellF1", "pred/GT"))
    print("-" * 55)
    for smooth_min_frames in SMOOTH_THRESHOLDS:
        result = variants[str(smooth_min_frames)]
        print("{:>10d} {:>8.2f} {:>10.4f} {:>10.4f} {:>7d}/{:<4d}".format(
            smooth_min_frames, result["smooth_min_seconds"], result["macro_f1"],
            result["spell_f1_mean"], result["n_pred_spells"], result["n_gt_spells"]))

    best_threshold = max(
        variants, key=lambda key: (variants[key]["spell_f1_mean"],
                                   variants[key]["macro_f1"]))
    primary = dict(variants[str(DEFAULT_SMOOTH)])
    primary["smoothing_sweep"] = variants
    primary["smoothing_thresholds"] = list(SMOOTH_THRESHOLDS)
    primary["default_smooth_min_frames"] = DEFAULT_SMOOTH
    primary["best_smooth_min_frames"] = int(best_threshold)
    primary["best_spell_f1"] = variants[best_threshold]["spell_f1_mean"]
    primary["load_seconds"] = round(time.perf_counter() - started, 2)
    report_json.write_text(json.dumps(primary, indent=2), encoding="utf-8")

    print("\nwrote {}".format(atd_raw))
    print("wrote {}".format(atd_meta))
    print("wrote {}".format(smart_tagging))
    print("wrote {}".format(gt_csv))
    print("wrote {}".format(official_csv))
    print("wrote {}".format(pred_csv))
    for smooth_min_frames in SMOOTH_THRESHOLDS:
        print("wrote {}".format(folder / "possession_pred_smooth{}.csv".format(
            smooth_min_frames)))
    print("wrote {}".format(report_json))
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match-id", default=MATCH_ID,
                        help="IDSSE match id (default: J03WOH)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="download/cache root (default: data/external/idsse)")
    parser.add_argument("--skip-download", action="store_true",
                        help="use existing local files and never touch the network")
    parser.add_argument("--force", action="store_true", help="redownload cached files")
    parser.add_argument("--benchmark", type=float, default=BENCHMARK,
                        help="comparison score (default: 0.645)")
    args = parser.parse_args()
    if args.skip_download and args.force:
        parser.error("--skip-download and --force are mutually exclusive")
    run(args.match_id, args.data_dir, skip_download=args.skip_download,
        force=args.force, benchmark=args.benchmark)


if __name__ == "__main__":
    main()



