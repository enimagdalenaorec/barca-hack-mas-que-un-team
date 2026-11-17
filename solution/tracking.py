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

# --- Possession state machine -----------------------------------------------------------------
# The submission alphabet, plus X for "no evidence yet". X is not a football state; it is what
# the file says before the first anchor and after a dead ball until a new anchor arrives.
LABELS = np.array(["H", "A", "D", "X"])
UNKNOWN = len(LABELS) - 1
DEAD_CODE = len(LABELS) - 2

# Smart Tagging writes the club names, not sides. The XML spells Athletic the Spanish way.
TEAM_TAGS = {
    "barcelona": HOME,
    "fc barcelona": HOME,
    "atletic club": AWAY,
    "athletic club": AWAY,
    "athletic": AWAY,
}

# Codes whose Team tag names the side IN possession. The last four are aliases for the event
# names a passing/dribbling/tackling export might use, so a richer tagging file needs no edit.
ATTACKING_CODES = {
    "build up", "progression", "creating chances", "attacking transition",
    "ball in final third", "ball in the box", "long ball", "players in the box",
    "pass", "passing", "dribble", "carry", "tackle", "interception",
}
# Codes whose Team tag names the side OUT of possession, so the anchor is the other side.
DEFENDING_CODES = {
    "defending in attacking third", "defending in middle third",
    "defending in defensive third", "defensive transition",
}
SET_PIECE_CODE = "set pieces"
GOAL_CODE = "goals"
STOPPAGE_CODES = {SET_PIECE_CODE, GOAL_CODE}

# A set-piece tag starts a few seconds before the restart and then runs ~35 s over the live play
# that follows, so only the lead-in is dead. Both tagged kickoffs put the real kickoff 82 frames
# (3.28 s) after the tag start; that is the fallback when the ball is not visible enough to find
# the first moving frame. See starter/load_smart_tagging.py for the kickoff time-base check.
SET_PIECE_FALLBACK_S = 3.28
SET_PIECE_MAX_WAIT_S = 15.0
RESTART_SPEED_MPS = 2.0
RESTART_CONSECUTIVE = 3
# The restart detector waits for three moving frames, so the anchor lands a couple of frames after
# the official restart. Pulling it back closes that deterministic latency.
SET_PIECE_ANCHOR_LEAD_FRAMES = 2
# Event timestamps mark the strike, not the end of control. A short tail keeps the possession
# alive through the touch and the first frames of the following dead ball or challenge.
POSSESSION_TAIL_FRAMES = 4

# --- Physical possession evidence -------------------------------------------------------------
# The exporter's ball-speed column is NaN for this file, so every speed here is measured from the
# ball positions. A player has to be within this radius to be treated as touching the ball; 1.5 m
# is roughly a stride and is also the loose-ball radius in the brief.
CONTACT_RADIUS_M = 1.5
CONTROL_CLOSE_M = 0.8         # a player this close is touching the ball even if it is moving fast
CONTROL_MAX_SPEED_MPS = 4.0   # a slower ball can be trapped/controlled (about 14 km/h)
CONTROL_MIN_FRAMES = 6        # 0.24 s of sustained proximity; rejects a chase-by/pass-by
CONTROL_DOMINANCE_M = 0.25    # nearest player must be clearly closer than the runner-up
CONTACT_GUARD_FRAMES = 3      # do not create two contact anchors within 0.12 s of each other
LOOSE_BALL_SPEED_MPS = 4.0    # above this, a ball with nobody within CONTACT_RADIUS_M is loose

# A live event anchor that changes possession is only credible once the named side is actually
# within control range. Otherwise it is a pass/challenge timestamp, not a completed turnover.
TURNOVER_CONFIRM_REASONS = ("pass", "tackle", "interception", "creating chances")
TURNOVER_CONFIRM_WINDOW_FRAMES = 20
TURNOVER_CONFIRM_RADIUS_M = CONTACT_RADIUS_M

# Velocity is measured over a moving window either side of the candidate frame. One frame of
# tracking noise moves a direction estimate a long way; three frames is 0.12 s and is stable
# enough for a real touch while still localising a hard change to within a few frames.
PHYSICS_WINDOW_FRAMES = 3
PHYSICS_MIN_SPEED_MPS = 4.0
PHYSICS_TURN_DEG = 45.0
PHYSICS_SPEED_CHANGE_MPS = 1.5
PHYSICS_MIN_SEPARATION_FRAMES = 10
SPEED_WINDOW_FRAMES = 5       # centred median of frame-to-frame speed, for control/loose gating

# --- Kinematic pitch control ------------------------------------------------------------------
# Player velocity is a finite difference over this many frames either side. Five frames is 0.2 s
# at 25 fps: long enough to reject single-frame tracking noise, short enough to preserve a sprint.
PLAYER_VELOCITY_WINDOW_FRAMES = 5
MAX_PLAYER_SPEED_MPS = 12.0   # cap a tracking jump so it cannot create an impossible TTI
TTI_REACH_M = CONTACT_RADIUS_M
TTI_HORIZON_S = 1.2           # only players who can reach the projected ball inside this count
TTI_MAX_DISTANCE_M = 35.0     # ignore players too far away to influence the next touch
TTI_KAPPA_S = 0.35            # softmax temperature for the two-team control probability
TTI_MIN_PROBABILITY = 0.50    # below this a kinematic team is too ambiguous to override distance
LOOSE_TTI_HORIZON_S = 1.0
LOOSE_TTI_MIN_PROBABILITY = 0.60
LOOSE_ANCHOR_MIN_RUN_FRAMES = 5
LOOSE_GATE_MIN_PROBABILITY = 0.75
LOOSE_GATE_HORIZON_S = 0.40
KINEMATIC_SEPARATION_FRAMES = 10

# --- Inactivity stoppage detector -------------------------------------------------------------
# A live possession always has somebody moving towards or with the ball. When the whole tracked
# player cloud goes quiet for a sustained window, the ball is not in play even if no tag says so
# and even if the ball is briefly missing from the tracking feed. This is event-free and uses
# only positions, so it generalises across matches and tag vocabularies.
INACTIVITY_WINDOW_FRAMES = 75        # 3 s centred rolling window
INACTIVITY_PLAYER_SPEED_MPS = 1.2    # mean across tracked players
INACTIVITY_BALL_SPEED_MPS = 2.5      # rolling median ball speed
INACTIVITY_BALL_DISTANCE_M = 3.0     # rolling median ball-to-nearest-player distance
INACTIVITY_MISSING_FRACTION = 0.5    # share of the window with no ball observation
INACTIVITY_MIN_FRAMES = 25           # 1 s minimum, so a jog to fetch the ball is not a stoppage

# Short possessions supported by a physical touch are evidence, not tracking blips, so the
# run-length smoother is not allowed to absorb them. Event tags are deliberately excluded: on
# IDSSE they protected more false micro-runs than genuine ones.
SMOOTH_PROTECT_REASONS = ("physics_interception", "physics_recovery")


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


# --- Smart Tagging ---------------------------------------------------------------------------
def load_events(game=GAME, data_dir=None, fps=None):
    """Smart Tagging instances as a DataFrame, one row per tagged interval.

    Columns: code, code_key, start_s, end_s, start_frame, end_frame, team, team_code, type, half.
    `code_key` is the lower-cased code and `team_code` is HOME/AWAY/-1, so downstream code never
    has to parse the XML or compare club names again. Frames are half-open on the right, the same
    convention starter/load_smart_tagging.py uses.
    """
    fps = fps or GAMES[game]["fps"]
    path = require(["smart_tagging"], game, data_dir)[0]
    rows = []
    for instance in ET.parse(path).getroot().iter("instance"):
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
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "team": team,
            "team_code": TEAM_TAGS.get((team or "").strip().lower(), -1),
            "type": tags.get("Type"),
            "half": tags.get("Half"),
        })
    columns = ["code", "code_key", "start_s", "end_s", "start_frame", "end_frame",
               "team", "team_code", "type", "half"]
    return pd.DataFrame(rows, columns=columns)


def _restart_frame(ball_xy, start, stop, pitch, fps,
                   speed_mps=RESTART_SPEED_MPS, consecutive=RESTART_CONSECUTIVE):
    """First frame in [start, stop) where the ball is in play and moving, or None.

    A set-piece tag begins while the ball is still being placed, so the restart is not the tag
    start. Requiring `consecutive` moving frames above `speed_mps` rejects a single tracking jump
    when the ball is re-acquired. The search is per event, so the per-frame work stays vectorized.
    """
    n_frames = len(ball_xy)
    low = max(0, min(n_frames, int(start)))
    high = max(low, min(n_frames, int(stop)))
    if high - low < consecutive + 1:
        return None

    window = np.asarray(ball_xy[low:high], dtype=np.float64)
    length, width = pitch
    finite = np.isfinite(window[:, 0]) & np.isfinite(window[:, 1])
    inside = (finite
              & (window[:, 0] >= 0.0) & (window[:, 0] <= length)
              & (window[:, 1] >= 0.0) & (window[:, 1] <= width))
    step = np.linalg.norm(np.diff(window, axis=0), axis=1) * fps
    moving = inside[1:] & inside[:-1] & (step >= speed_mps)
    if len(moving) < consecutive:
        return None

    run = np.convolve(moving.astype(np.int8), np.ones(consecutive, dtype=np.int8), mode="valid")
    hit = np.flatnonzero(run >= consecutive)
    if len(hit) == 0:
        return None
    return low + int(hit[0]) + 1  # moving[i] is the step from frame i to i+1


def set_piece_restart_frames(events, ball_xy, pitch, fps=25,
                             max_wait_s=SET_PIECE_MAX_WAIT_S):
    """{tag start frame: restart frame} for every SET PIECES instance.

    Falls back to SET_PIECE_FALLBACK_S after the tag start when the ball is not visible enough to
    locate the first moving frame inside the search window.
    """
    out = {}
    if events is None or len(events) == 0:
        return out
    wait = int(round(max_wait_s * fps))
    fallback = int(round(SET_PIECE_FALLBACK_S * fps))
    for event in events.itertuples(index=False):
        if event.code_key != SET_PIECE_CODE:
            continue
        start = int(event.start_frame)
        stop = min(int(event.end_frame), start + wait)
        restart = _restart_frame(ball_xy, start, stop, pitch, fps)
        if restart is None:
            restart = min(int(event.end_frame), start + fallback)
        out[start] = restart
    return out


def stoppage_windows(events, restarts):
    """[(first_dead_frame, first_live_frame, reason)] for GOALS and SET PIECES.

    GOALS intervals are genuine stoppages -- celebration, VAR, restart -- so the whole interval is
    dead. A SET PIECES interval is not: it runs ~35 s over live play, so only [tag start, restart)
    is dead. The restart comes from set_piece_restart_frames().
    """
    windows = []
    if events is None or len(events) == 0:
        return windows
    for event in events.itertuples(index=False):
        if event.code_key not in STOPPAGE_CODES:
            continue
        start, end = int(event.start_frame), int(event.end_frame)
        if event.code_key == GOAL_CODE:
            windows.append((start, end, GOAL_CODE))
        elif event.code_key == SET_PIECE_CODE:
            stop = int(restarts.get(start, end))
            if stop > start:
                windows.append((start, stop, SET_PIECE_CODE))
    return windows


def _rolling(values, window, how="mean", center=True):
    """NaN-tolerant centred rolling `how` over `window` frames."""
    series = pd.Series(np.asarray(values, dtype=np.float64))
    min_periods = max(5, int(window) // 3)
    rolled = series.rolling(int(window), center=center, min_periods=min_periods)
    return getattr(rolled, how)().to_numpy()


def player_activity_series(players_xy, fps=25.0):
    """Mean speed of the tracked player cloud, in m/s, one value per frame.

    Finite differences of consecutive positions. The step from frame i to i+1 is assigned to
    frame i+1 so the value is aligned with the interval it measures; the first frame copies the
    second. Slots with no observation are skipped, so a partially visible frame does not bias the
    cloud downwards.
    """
    players = np.asarray(players_xy[:, :, :2], dtype=np.float64)
    n_frames = len(players)
    if n_frames == 0 or players.shape[1] == 0:
        return np.full(n_frames, np.nan, dtype=np.float64)

    step = np.linalg.norm(np.diff(players, axis=0), axis=2) * fps
    valid = np.isfinite(step)
    count = valid.sum(axis=1)
    total = np.where(valid, step, 0.0).sum(axis=1)
    speed = np.full(n_frames, np.nan, dtype=np.float64)
    speed[1:] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    if n_frames > 1:
        speed[0] = speed[1]
    return speed


def inactivity_mask(ball_xy, players_xy, fps=25.0, telemetry=None,
                    window=INACTIVITY_WINDOW_FRAMES,
                    player_speed_mps=INACTIVITY_PLAYER_SPEED_MPS,
                    ball_speed_mps=INACTIVITY_BALL_SPEED_MPS,
                    ball_distance_m=INACTIVITY_BALL_DISTANCE_M,
                    missing_fraction=INACTIVITY_MISSING_FRACTION,
                    min_frames=INACTIVITY_MIN_FRAMES):
    """Frames whose surrounding play is too quiet to be live possession.

    Three independent physical witnesses of a stoppage, each over the same centred window:

      * the tracked player cloud barely moves (mean speed below `player_speed_mps`);
      * the ball is stationary or slow (rolling median speed below `ball_speed_mps`);
      * the ball sits far from every player (rolling median nearest distance above
        `ball_distance_m`) or is absent for most of the window (`missing_fraction`).

    A frame is quiet when the player witness AND at least one ball witness agree, and only runs
    of at least `min_frames` survive. That conjunction is what separates a genuine stoppage from
    a live possession in which the ball happens to be missing: live players keep moving.
    """
    n_frames = len(ball_xy)
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    if telemetry is None:
        blank_teams = np.zeros(np.asarray(players_xy).shape[1], dtype=np.int8)
        telemetry = ball_telemetry(ball_xy, players_xy, blank_teams, fps)

    played = _rolling(player_activity_series(players_xy, fps), window)
    quiet_players = np.isfinite(played) & (played < player_speed_mps)

    seen = telemetry["ball_seen"]
    missing_share = _rolling((~seen).astype(np.float64), window)
    ball_speed = _rolling(np.where(seen, telemetry["speed"], np.nan), window, "median")
    distance = _rolling(np.where(seen, telemetry["nearest_distance"], np.nan),
                        window, "median")
    quiet_ball = ((missing_share >= missing_fraction)
                  | (ball_speed < ball_speed_mps)
                  | (distance > ball_distance_m))

    raw = quiet_players & quiet_ball
    keep = np.zeros(n_frames, dtype=bool)
    starts, stops = _boolean_runs(raw, int(min_frames))
    for start, stop in zip(starts, stops):
        keep[start:stop] = True
    return keep


def _dead_ball_components(ball_xy, pitch, events=None, fps=25, pitch_margin_m=0.0,
                          nan_is_dead=False, max_wait_s=SET_PIECE_MAX_WAIT_S,
                          inactivity=None):
    """(dead, dead_reason, outside_pitch) arrays.

    Dead is deliberately strict: the ball must be seen OUTSIDE the pitch, or an event-defined
    stoppage must cover the frame, or a sustained inactivity run must be supplied. A missing ball
    alone is not outside the pitch -- it is an unknown observation, and forward-filling the last
    possession through it is the point of the state machine. Pass nan_is_dead=True to opt into
    treating it as dead instead.
    """
    n_frames = len(ball_xy)
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    length, width = pitch
    seen = np.isfinite(x) & np.isfinite(y)
    outside = (seen
               & ((x < -pitch_margin_m) | (x > length + pitch_margin_m)
                  | (y < -pitch_margin_m) | (y > width + pitch_margin_m)))

    dead = outside.copy()
    reason = np.full(n_frames, "", dtype=object)
    reason[outside] = "out_of_pitch"

    if nan_is_dead:
        missing = ~seen
        dead |= missing
        reason[missing & (reason == "")] = "ball_missing"

    if events is not None and len(events) > 0:
        restarts = set_piece_restart_frames(events, ball_xy, pitch, fps, max_wait_s)
        for start, stop, code in stoppage_windows(events, restarts):
            low, high = max(0, start), min(n_frames, stop)
            if high <= low:
                continue
            new = ~dead[low:high]
            dead[low:high] = True
            reason[low + np.flatnonzero(new)] = code

    if inactivity is not None:
        inactive = np.asarray(inactivity, dtype=bool) & ~dead
        dead |= inactive
        reason[inactive & (reason == "")] = "inactivity"

    return dead, reason, outside


def dead_ball_mask(ball_xy, pitch, events=None, fps=25, pitch_margin_m=0.0,
                   nan_is_dead=False, max_wait_s=SET_PIECE_MAX_WAIT_S):
    """Boolean frame mask: ball outside the pitch, or a goal/set-piece stoppage.

    Ball coordinates are metres, the same convention as load(). `pitch_margin_m` is a tolerance
    around the touchlines; 0.0 is the strict reading of "outside the pitch dimensions".
    """
    return _dead_ball_components(ball_xy, pitch, events, fps, pitch_margin_m,
                                 nan_is_dead, max_wait_s)[0]


def nearest_player_at_frames(frames, ball_xy, players_xy, slot_team):
    """(team, distance) of the closest tracked player to the ball at the requested frames.

    Used to resolve Team=N/A set pieces at the exact restart frame. The distance is Euclidean in
    metres, not squared, so callers can compare it with a radius. Frames must be valid indexes;
    a missing ball or an empty track slot yields distance inf and team -1.
    """
    frames = np.asarray(frames, dtype=np.int64)
    if frames.size == 0:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float64)

    ball = np.asarray(ball_xy[frames, :2], dtype=np.float64)
    players = np.asarray(players_xy[frames, :, :2], dtype=np.float64)
    if players.shape[1] == 0:
        return np.full(len(frames), -1, dtype=np.int8), np.full(len(frames), np.inf)

    dx = players[:, :, 0] - ball[:, 0, None]
    dy = players[:, :, 1] - ball[:, 1, None]
    d2 = dx * dx + dy * dy
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    nearest = np.argmin(d2, axis=1)
    distance = np.sqrt(d2[np.arange(len(frames)), nearest])
    team = np.where(np.isfinite(distance), slot_team[nearest], -1).astype(np.int8)
    return team, distance


def nearest_player_arrays(ball_xy, players_xy, slot_team):
    """Per-frame nearest and second-nearest player distance, plus the nearest player's team.

    One vectorized pass over (frames, slots). The second distance is used for dominance: a contact
    anchor is only credible when the nearest player is not tied with an opponent beside him.
    """
    n_frames = len(ball_xy)
    n_slots = players_xy.shape[1]
    if n_slots == 0:
        empty = np.full(n_frames, np.inf, dtype=np.float64)
        return {"distance": empty, "team": np.full(n_frames, -1, dtype=np.int8),
                "second_distance": empty.copy()}

    ball = np.asarray(ball_xy[:, :2], dtype=np.float64)
    players = np.asarray(players_xy[:, :, :2], dtype=np.float64)
    dx = players[:, :, 0] - ball[:, 0, None]
    dy = players[:, :, 1] - ball[:, 1, None]
    d2 = dx * dx + dy * dy
    d2 = np.where(np.isfinite(d2), d2, np.inf)

    nearest = np.argmin(d2, axis=1)
    distance = np.sqrt(d2[np.arange(n_frames), nearest])
    team = np.where(np.isfinite(distance), slot_team[nearest], -1).astype(np.int8)

    d2[np.arange(n_frames), nearest] = np.inf
    second = np.sqrt(np.min(d2, axis=1))
    return {"distance": distance, "team": team, "second_distance": second}


def player_velocities(players_xy, fps=25.0, window=PLAYER_VELOCITY_WINDOW_FRAMES):
    """Player velocity vectors in m/s from finite differences of the tracking positions.

    A central difference over `window` frames is averaged with the one-sided differences that are
    available at the sequence edges. If one side of a tracklet is missing, the other side is used;
    a slot with no usable displacement is treated as stationary. The norm is capped at
    `MAX_PLAYER_SPEED_MPS` so a single tracking jump cannot dominate the TTI calculation.

    Returns a float64 array shaped (frames, slots, 2). It is intentionally independent of the
    exporter's speed column, which is NaN for some feeds and carries no direction in any case.
    """
    players = np.asarray(players_xy[:, :, :2], dtype=np.float64)
    n_frames, n_slots, _ = players.shape
    velocity = np.zeros((n_frames, n_slots, 2), dtype=np.float64)
    if n_frames < 2 or n_slots == 0:
        return velocity

    window = max(1, int(window))
    forward = np.full_like(players, np.nan)
    backward = np.full_like(players, np.nan)
    if n_frames > window:
        delta = (players[window:] - players[:-window]) * (fps / float(window))
        forward[:-window] = delta
        backward[window:] = delta

    stack = np.stack((forward, backward), axis=0)
    count = np.sum(np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    velocity = np.divide(total, count, out=np.zeros_like(total), where=count > 0)

    speed = np.linalg.norm(velocity, axis=2)
    scale = np.divide(MAX_PLAYER_SPEED_MPS, speed,
                      out=np.ones_like(speed), where=speed > MAX_PLAYER_SPEED_MPS)
    return velocity * scale[:, :, None]


def ball_kinematics(ball_xy, fps=25.0, window=PHYSICS_WINDOW_FRAMES,
                    speed_window=SPEED_WINDOW_FRAMES):
    """Ball velocity, speed, trajectory angle and turn angle over a moving window.

    `velocity_before` is the average velocity over [t-window, t] and `velocity_after` over
    [t, t+window]. A sharp touch changes the direction and/or magnitude between the two. Speeds
    are in m/s and angles in degrees. `speed` is a centred median of frame-to-frame speed, which
    is the stable signal used for control and loose-ball gating.
    """
    ball = np.asarray(ball_xy[:, :2], dtype=np.float64)
    n_frames = len(ball)
    window = max(1, int(window))
    before = np.full_like(ball, np.nan)
    after = np.full_like(ball, np.nan)
    if n_frames > window:
        delta = (ball[window:] - ball[:-window]) * (fps / float(window))
        before[window:] = delta
        after[:-window] = delta

    speed_before = np.linalg.norm(before, axis=1)
    speed_after = np.linalg.norm(after, axis=1)
    dot = before[:, 0] * after[:, 0] + before[:, 1] * after[:, 1]
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    turn = np.degrees(np.arctan2(np.abs(cross), dot))
    turn[~np.isfinite(turn) | (speed_before < 0.1) | (speed_after < 0.1)] = np.nan

    step = np.full(n_frames, np.nan)
    if n_frames > 1:
        step[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    speed = pd.Series(step).rolling(max(1, int(speed_window)), center=True,
                                    min_periods=1).median().to_numpy()

    return {
        "velocity_before": before,
        "velocity_after": after,
        "speed_before": speed_before,
        "speed_after": speed_after,
        "speed_change": speed_after - speed_before,
        "speed": speed,
        "angle": np.degrees(np.arctan2(after[:, 1], after[:, 0])),
        "turn": turn,
    }


def time_to_intercept(ball_xy, ball_velocity, players_xy, players_velocity,
                      reach=TTI_REACH_M, horizon=TTI_HORIZON_S,
                      max_distance=TTI_MAX_DISTANCE_M):
    """Per-player time until the player can reach the ball's projected path.

    Constant-velocity approximation. For relative position ``r = p - b`` and relative velocity
    ``v = v_p - v_b`` the first time inside `reach` solves ``|r + v t| = reach``. That is the
    earliest positive root of ``a t^2 + 2 b t + c = 0``. A player already inside `reach` has
    ``tti = 0``. No solution, an intercept outside `horizon`, or a player farther than
    `max_distance` gives ``inf``.

    This is deliberately cheaper than a full pitch-control surface: it only has to rank the
    players who can plausibly be the next body on the ball, not estimate a continuous field over
    the whole pitch.
    """
    ball = np.asarray(ball_xy[:, :2], dtype=np.float64)
    players = np.asarray(players_xy[:, :, :2], dtype=np.float64)
    n_frames, n_slots, _ = players.shape
    tti = np.full((n_frames, n_slots), np.inf, dtype=np.float64)
    distance = np.full((n_frames, n_slots), np.inf, dtype=np.float64)
    if n_frames == 0 or n_slots == 0:
        return tti, distance

    ball_v = np.asarray(ball_velocity, dtype=np.float64)
    ball_v = np.where(np.isfinite(ball_v), ball_v, 0.0)
    player_v = np.asarray(players_velocity, dtype=np.float64)
    player_v = np.where(np.isfinite(player_v), player_v, 0.0)

    rx = players[:, :, 0] - ball[:, 0, None]
    ry = players[:, :, 1] - ball[:, 1, None]
    vx = player_v[:, :, 0] - ball_v[:, 0, None]
    vy = player_v[:, :, 1] - ball_v[:, 1, None]
    distance = np.sqrt(rx * rx + ry * ry)
    a = vx * vx + vy * vy
    b = rx * vx + ry * vy
    c = distance * distance - reach * reach
    discriminant = b * b - a * c

    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt(np.maximum(discriminant, 0.0))
        first = (-b - root) / a

    valid = (np.isfinite(first) & (first >= 0.0) & (a > 1e-8)
             & (discriminant >= 0.0) & (distance <= max_distance))
    tti = np.where(valid, first, np.inf)
    tti = np.where(np.isfinite(distance) & (distance <= reach), 0.0, tti)
    tti = np.where(tti <= horizon, tti, np.inf)
    return tti, distance


def reachability_field(ball_xy, players_xy, slot_team, fps=25.0, kinematics=None,
                       reach=TTI_REACH_M, horizon=TTI_HORIZON_S, kappa=TTI_KAPPA_S):
    """Per-frame team influence from time-to-intercept and a two-team softmax.

    The minimum TTI for each side is passed through a softmax over the two teams. The resulting
    probability is the confidence that the side can reach the projected ball first. A team is
    reported only when at least one player has a finite TTI inside the horizon.
    """
    kinematics = ball_kinematics(ball_xy, fps) if kinematics is None else kinematics
    player_v = player_velocities(players_xy, fps)
    ball_v = np.where(np.isfinite(kinematics["velocity_after"]),
                      kinematics["velocity_after"], kinematics["velocity_before"])
    ball_v = np.where(np.isfinite(ball_v), ball_v, 0.0)
    player_tti, _distance = time_to_intercept(
        ball_xy, ball_v, players_xy, player_v, reach, horizon)

    n_frames = len(ball_xy)
    team_tti = np.full((n_frames, 2), np.inf, dtype=np.float64)
    for side in (HOME, AWAY):
        slots = slot_team == side
        if np.any(slots):
            team_tti[:, side] = np.min(player_tti[:, slots], axis=1)

    finite = np.isfinite(team_tti)
    any_finite = np.any(finite, axis=1)
    logits = np.full_like(team_tti, -np.inf)
    logits[finite] = -team_tti[finite] / max(float(kappa), 1e-6)
    safe_max = np.where(any_finite, np.max(logits, axis=1), 0.0)
    weights = np.exp(logits - safe_max[:, None])
    denominator = np.sum(weights, axis=1)
    probability = np.divide(weights, denominator[:, None],
                            out=np.zeros_like(weights),
                            where=denominator[:, None] > 0)

    team = np.full(n_frames, -1, dtype=np.int8)
    if np.any(any_finite):
        team[any_finite] = np.argmax(probability[any_finite], axis=1).astype(np.int8)
    best_probability = np.where(any_finite, np.max(probability, axis=1), 0.0)
    best_tti = np.min(team_tti, axis=1)
    best_slot = np.argmin(player_tti, axis=1).astype(np.int16)

    return {
        "player_velocity": player_v,
        "player_tti": player_tti,
        "kinematic_team_tti": team_tti,
        "kinematic_tti": best_tti,
        "kinematic_team": team,
        "kinematic_probability": best_probability,
        "kinematic_slot": best_slot,
    }


def ball_telemetry(ball_xy, players_xy, slot_team, fps=25.0):
    """Everything the possession passes need from the ball and the player cloud.

    Returns a dict of per-frame arrays: kinematics, nearest-player geometry, the loose-ball mask,
    the contact mask (within CONTACT_RADIUS_M) and the sharp-turn physics mask.
    """
    kinematics = ball_kinematics(ball_xy, fps)
    nearest = nearest_player_arrays(ball_xy, players_xy, slot_team)
    kinematic = reachability_field(ball_xy, players_xy, slot_team, fps,
                                   kinematics=kinematics)
    ball_seen = np.isfinite(ball_xy[:, 0]) & np.isfinite(ball_xy[:, 1])
    distance = nearest["distance"]

    loose = ball_seen & (kinematics["speed"] > LOOSE_BALL_SPEED_MPS) \
        & (distance > CONTACT_RADIUS_M)
    contact = ball_seen & (distance <= CONTACT_RADIUS_M) & (nearest["team"] >= 0)
    physics = (contact
               & (kinematics["turn"] >= PHYSICS_TURN_DEG)
               & (kinematics["speed_before"] >= PHYSICS_MIN_SPEED_MPS)
               & (np.abs(kinematics["speed_change"]) >= PHYSICS_SPEED_CHANGE_MPS))
    return {
        **kinematics,
        **kinematic,
        "nearest_distance": distance,
        "nearest_team": nearest["team"],
        "second_distance": nearest["second_distance"],
        "ball_seen": ball_seen,
        "ball_loose": loose,
        "contact": contact,
        "physics": physics,
    }


def _boolean_runs(mask, min_length=1):
    """(starts, stops) of maximal True runs of at least `min_length`; stops are half-open."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    keep = (stops - starts) >= max(1, int(min_length))
    return starts[keep], stops[keep]


def _state_codes(team, dead):
    """Forward-filled state codes: H=0, A=1, D=2, X=3.

    This is the same pandas ffill() the final table uses, exposed so the physical passes can ask
    "what would the state be at this frame before I add an anchor?" without rebuilding a table.
    """
    state = pd.Series(np.nan, index=np.arange(len(dead)), dtype="float64")
    state[dead] = 2
    anchored = (team >= 0) & ~dead
    state[anchored] = team[anchored]
    return state.ffill().fillna(UNKNOWN).astype(np.int8).to_numpy().copy()


def smooth_micro_possessions(state_series, min_frames=18, anchor_reason=None,
                             protect_reasons=()):
    """Absorb H/A possession blips shorter than `min_frames` into the previous stable state.

    Contiguous identical states are grouped with the standard pandas run-length idiom. Short
    ``H``/``A`` groups are blanked to NaN and forward-filled, so the previous state absorbs the
    blip. ``D`` groups are never removed: dead-ball windows are evidence, not noise, and shortening
    them would fabricate live possession. A short leading ``H``/``A`` run has no previous stable
    state to inherit, so it is left untouched instead of becoming X.

    A short run that contains an anchor whose reason is in `protect_reasons` is kept. Physics and
    set-piece anchors are direct evidence for that possession; without this exemption the smoother
    would erase a genuine one-touch spell just because it is shorter than the threshold.

    The input can be any pandas Series (the returned Series keeps its index) or an array-like,
    which is converted to a default-index Series. `anchor_reason` must align with the state series.
    """
    if not isinstance(state_series, pd.Series):
        state_series = pd.Series(state_series)
    min_frames = int(min_frames)
    if min_frames <= 1 or state_series.empty:
        return state_series.copy()

    state = state_series.astype(object)
    groups = (state != state.shift()).cumsum()
    lengths = groups.map(groups.value_counts())
    micro = state.isin((LABELS[HOME], LABELS[AWAY])) & (lengths < min_frames)
    if anchor_reason is not None and len(protect_reasons) > 0:
        reasons = pd.Series(np.asarray(anchor_reason, dtype=object), index=state.index)
        protected = reasons.isin(tuple(protect_reasons)).groupby(groups).transform("any")
        micro &= ~protected.astype(bool)
    if not micro.any():
        return state_series.copy()

    smoothed = state.mask(micro).ffill()
    return smoothed.fillna(state)


def physics_anchors(team, reason, dead, telemetry,
                    separation=PHYSICS_MIN_SEPARATION_FRAMES, use_kinematic=True,
                    kinematic_mode="agree"):
    """Add anchors at hard touches, optionally gated by kinematic influence.

    A candidate must be a sharp turn (PHYSICS_TURN_DEG), a moving ball (PHYSICS_MIN_SPEED_MPS),
    a real speed change, and within contact range of a tracked player. `kinematic_mode` decides
    what the time-to-intercept field does with that candidate:

    * ``override``  -- the kinematic team may replace the Euclidean nearest player;
    * ``agree``     -- the touch only counts when the kinematic field is confident and names the
                       same side as the nearest player, which filters ambiguous contacts;
    * ``confidence`` -- keep the nearest-player side but require a confident kinematic field;
    * ``off``       -- the original nearest-only rule.

    Nearby candidates are one touch, not several, so they are thinned by `separation`; Smart
    Tagging anchors win if they are within that guard. Mutates `team` and `reason`; returns the
    number of anchors added.
    """
    if not use_kinematic:
        kinematic_mode = "off"
    state = _state_codes(team, dead)
    nearest_team = telemetry["nearest_team"]
    kinematic_team = telemetry["kinematic_team"]
    kinematic_probability = telemetry["kinematic_probability"]
    nearest_turnover = (nearest_team >= 0) & (nearest_team != state)
    confident = kinematic_probability >= TTI_MIN_PROBABILITY
    if kinematic_mode == "override":
        kinematic_turnover = ((kinematic_team >= 0) & (kinematic_team != state)
                              & confident)
    elif kinematic_mode == "agree":
        kinematic_turnover = (nearest_turnover & (kinematic_team == nearest_team)
                              & confident)
    elif kinematic_mode == "confidence":
        kinematic_turnover = nearest_turnover & confident
    else:
        kinematic_turnover = np.zeros_like(nearest_turnover)
    candidate = telemetry["physics"] & ~dead & (nearest_turnover | kinematic_turnover)
    starts, stops = _boolean_runs(candidate, 1)
    if len(starts) == 0:
        return 0

    anchor_frames = np.sort(np.flatnonzero(team >= 0))
    added = 0
    for start, stop in zip(starts, stops):
        score = np.nan_to_num(telemetry["turn"][start:stop], nan=-1.0)
        local_nearest = nearest_team[start:stop]
        local_kinematic = kinematic_team[start:stop]
        local_state = state[start:stop]
        local_turnover = ((local_nearest >= 0) & (local_nearest != local_state))
        local_confident = kinematic_probability[start:stop] >= TTI_MIN_PROBABILITY
        if kinematic_mode == "override":
            kinematic_ok = ((local_kinematic >= 0) & local_confident
                            & (local_kinematic != local_state))
        elif kinematic_mode == "agree":
            kinematic_ok = (local_turnover & (local_kinematic == local_nearest)
                            & local_confident)
        elif kinematic_mode == "confidence":
            kinematic_ok = local_turnover & local_confident
        else:
            kinematic_ok = np.zeros(stop - start, dtype=bool)
        usable = local_turnover | kinematic_ok
        score = np.where(usable, score, -1.0)
        frame = int(start + np.argmax(score))

        position = int(np.searchsorted(anchor_frames, frame))
        near = ((position > 0
                 and frame - anchor_frames[position - 1] <= separation)
                or (position < len(anchor_frames)
                    and anchor_frames[position] - frame <= separation))
        if near or team[frame] >= 0:
            continue

        nearest_side = int(nearest_team[frame])
        kinematic_side = int(kinematic_team[frame])
        if (kinematic_mode == "override" and kinematic_side >= 0
                and kinematic_side != state[frame]
                and kinematic_probability[frame] >= TTI_MIN_PROBABILITY):
            side = kinematic_side
        else:
            side = nearest_side
        if side < 0 or side == state[frame]:
            continue
        team[frame] = side
        reason[frame] = ("physics_interception"
                         if telemetry["speed_after"][frame]
                         >= telemetry["speed_before"][frame] else "physics_recovery")
        anchor_frames = np.insert(anchor_frames, position, frame)
        added += 1
    return added


def contact_anchors(team, reason, dead, telemetry,
                    min_frames=CONTROL_MIN_FRAMES, radius=CONTACT_RADIUS_M,
                    dominance=CONTROL_DOMINANCE_M, max_speed=CONTROL_MAX_SPEED_MPS,
                    close=CONTROL_CLOSE_M, guard=CONTACT_GUARD_FRAMES):
    """Add anchors when an opponent has sustained control of the ball.

    The candidate mask is a player inside `radius`, clearly nearer than the runner-up, and either
    a slow/controlled ball or a player almost touching it. Only candidate runs of at least
    `min_frames` count. A run that starts while the current state already names that team is a
    continuation, not a turn-over. Runs are processed chronologically and the local state is
    updated after each accepted anchor, so a pass followed by a recovery can both be represented.

    A loose ball -- faster than LOOSE_BALL_SPEED_MPS with nobody inside `radius` -- can never
    satisfy this mask, so it leaves the forward-filled state untouched until a player arrives.

    Mutates `team` and `reason`; returns the number of anchors added.
    """
    nearest = telemetry["nearest_distance"]
    nearest_team = telemetry["nearest_team"]
    state = _state_codes(team, dead)
    controlled = (telemetry["ball_seen"] & ~dead
                  & (nearest <= radius) & (nearest_team >= 0)
                  & (telemetry["second_distance"] >= nearest + dominance)
                  & ((telemetry["speed"] <= max_speed) | (nearest <= close)))

    runs = []
    for side in (HOME, AWAY):
        starts, stops = _boolean_runs(controlled & (nearest_team == side), min_frames)
        runs.extend(zip(starts.tolist(), stops.tolist()))
    runs.sort(key=lambda item: item[0])

    anchor_frames = np.sort(np.flatnonzero(team >= 0))
    added = 0
    last_added = -10 ** 9
    for start, _stop in runs:
        side = int(nearest_team[start])
        if state[start] == side or team[start] >= 0 or start - last_added < guard:
            continue
        position = int(np.searchsorted(anchor_frames, start))
        if position < len(anchor_frames) and anchor_frames[position] == start:
            continue
        team[start] = side
        reason[start] = "contact_control"
        next_anchor = (int(anchor_frames[position])
                       if position < len(anchor_frames) else len(team))
        segment = np.full(next_anchor - start, side, dtype=np.int8)
        segment[dead[start:next_anchor]] = 2
        state[start:next_anchor] = segment
        anchor_frames = np.insert(anchor_frames, position, start)
        last_added = start
        added += 1
    return added


def loose_ball_anchors(team, reason, dead, telemetry, fps=25.0,
                       min_probability=LOOSE_TTI_MIN_PROBABILITY,
                       horizon=LOOSE_TTI_HORIZON_S,
                       min_run=LOOSE_ANCHOR_MIN_RUN_FRAMES,
                       separation=KINEMATIC_SEPARATION_FRAMES,
                       min_speed=LOOSE_BALL_SPEED_MPS,
                       require_turnover=True, require_inside_run=True):
    """Add predicted interception anchors for fast loose balls.

    The existing state machine only reacts when a player is already within Euclidean contact
    range. A pass into space therefore keeps the old possession until the receiver arrives. For
    each fast loose-ball run this pass projects the ball and asks which team can reach it first
    inside the TTI horizon. The earliest predicted reach frame becomes an anchor for that team,
    so possession can flip at the projected interception rather than at the first frame that
    happens to be within 1.5 m. A run is skipped when the two-team control probability is too
    close to 0.5 or when no player can reach the projected path inside the horizon, which is the
    advanced loose-ball gate: ambiguous flights keep the previous state.

    Mutates `team` and `reason`; returns the number of anchors added.
    """
    loose = (telemetry["ball_loose"] & telemetry["ball_seen"] & ~dead
             & (telemetry["speed"] > min_speed))
    starts, stops = _boolean_runs(loose, min_run)
    if len(starts) == 0:
        return 0

    kinematic_team = telemetry["kinematic_team"]
    probability = telemetry["kinematic_probability"]
    tti = telemetry["kinematic_tti"]
    state = _state_codes(team, dead) if require_turnover else None
    anchor_frames = np.sort(np.flatnonzero(team >= 0))
    added = 0
    for start, stop in zip(starts, stops):
        window = slice(start, stop)
        valid = ((kinematic_team[window] >= 0)
                 & (probability[window] >= min_probability)
                 & np.isfinite(tti[window])
                 & (tti[window] <= horizon))
        if not valid.any():
            continue

        candidates = np.flatnonzero(valid) + start
        predicted = candidates + np.rint(tti[candidates] * fps).astype(np.int64)
        predicted = np.clip(predicted, start + 1, stop - 1)
        usable = np.ones(len(candidates), dtype=bool)
        if require_inside_run:
            usable &= ((predicted > start) & (predicted < stop - 1))
        if require_turnover:
            usable &= kinematic_team[candidates] != state[predicted]
        if not usable.any():
            continue

        candidates = candidates[usable]
        predicted = predicted[usable]
        # Earliest predicted touch wins; ties go to the stronger control probability.
        order = np.lexsort((-probability[candidates], predicted))
        best = int(candidates[order[0]])
        frame = int(predicted[order[0]])
        side = int(kinematic_team[best])
        if side < 0:
            continue

        position = int(np.searchsorted(anchor_frames, frame))
        near = ((position > 0
                 and frame - anchor_frames[position - 1] <= separation)
                or (position < len(anchor_frames)
                    and anchor_frames[position] - frame <= separation))
        if near or team[frame] >= 0:
            continue

        team[frame] = side
        reason[frame] = "loose_ball_kinematic"
        anchor_frames = np.insert(anchor_frames, position, frame)
        added += 1
    return added


def extend_possession_tails(state_series, frames=POSSESSION_TAIL_FRAMES):
    """Extend each H/A spell by `frames` frames into the following live/dead state.

    Event timestamps mark the strike or the challenge, not the instant control is lost. On IDSSE
    the matched possession ends were 4-7 frames early on average, and the largest bias was at the
    live-ball -> dead-ball edge. This deterministic tail correction keeps the previous side for a
    short grace period. It never writes into ``X`` and never extends past the end of the series.

    The input is the final state series (``H``/``A``/``D``/``X`` strings). The returned Series
    keeps the original index.
    """
    if not isinstance(state_series, pd.Series):
        state_series = pd.Series(state_series)
    frames = int(frames)
    if frames <= 0 or state_series.empty:
        return state_series.copy()

    state = state_series.astype(object)
    labels = state.to_numpy(copy=True)
    if not np.isin(labels, (LABELS[HOME], LABELS[AWAY])).any():
        return state_series.copy()

    # Split on every state change, including H -> A. Two adjacent possessions must not be
    # mistaken for one long run, or a direct turnover would never receive its tail.
    bounds = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    starts, stops = bounds[:-1], bounds[1:]
    sides = labels[starts].copy()  # read before any mutation of a shared boundary

    for start, stop, side in zip(starts, stops, sides):
        if side not in (LABELS[HOME], LABELS[AWAY]):
            continue
        high = min(int(stop) + frames, len(labels))
        segment = labels[stop:high]
        extend = segment != LABELS[UNKNOWN]
        segment[extend] = side
        labels[stop:high] = segment
    return pd.Series(labels, index=state.index)


def team_distances(ball_xy, players_xy, slot_team):
    """Minimum distance from the ball to each side at every frame, shape (frames, 2)."""
    ball = np.asarray(ball_xy[:, :2], dtype=np.float64)
    players = np.asarray(players_xy[:, :, :2], dtype=np.float64)
    n_frames = len(ball)
    distance = np.full((n_frames, 2), np.inf, dtype=np.float64)
    if n_frames == 0:
        return distance
    dx = players[:, :, 0] - ball[:, 0, None]
    dy = players[:, :, 1] - ball[:, 1, None]
    player_distance = np.hypot(dx, dy)
    player_distance[~np.isfinite(player_distance)] = np.inf
    for side in (HOME, AWAY):
        slots = slot_team == side
        if np.any(slots):
            distance[:, side] = np.min(player_distance[:, slots], axis=1)
    return distance


def confirm_live_turnovers(team, reason, dead, ball_xy, players_xy, slot_team,
                           window_frames=TURNOVER_CONFIRM_WINDOW_FRAMES,
                           radius=TURNOVER_CONFIRM_RADIUS_M,
                           confirm_reasons=TURNOVER_CONFIRM_REASONS,
                           drop_unconfirmed=True,
                           move_to_support=True):
    """Gate live turnover anchors on the new side actually reaching the ball.

    An event tag names the player involved, but a possession change is only real once that side
    controls the ball. For every event anchor that flips the forward-filled state, look
    `window_frames` ahead for a live frame where the named side has a player within `radius`.

    By default the anchor stays at the event frame when supported (the tag is the best boundary
    estimate; the look-ahead only confirms it). With ``move_to_support=True`` the anchor moves to
    the first supported frame instead, but never across a dead-ball gap and never onto another
    side's anchor. When the first supported frame already carries a same-side anchor, the event
    anchor is redundant and is dropped. Unsupported anchors are dropped by default so the
    previous stable possession keeps filling.

    Set-piece and physical anchors are left alone: they already name a restart or a ball touch.
    Mutates `team` and `reason`; returns ``(moved, dropped)``.

    The scan keeps a running forward-filled state instead of freezing the pre-pass state, so an
    anchor that becomes a turnover *because* an earlier anchor was dropped is gated too. A later
    anchor by the same side can therefore still re-establish possession once it has support.
    """
    team = np.asarray(team)
    n_frames = len(team)
    if n_frames == 0 or not np.any(team >= 0):
        return 0, 0
    side_distance = team_distances(ball_xy, players_xy, slot_team)
    near = (side_distance <= float(radius)) & ~dead[:, None]
    confirm = set(confirm_reasons)
    window = max(1, int(window_frames))
    dead_frames = np.flatnonzero(dead)

    moved = 0
    dropped = 0
    current = UNKNOWN
    for frame in range(n_frames):
        if dead[frame]:
            current = DEAD_CODE  # dead takes priority, exactly like _state_codes
            continue
        side = int(team[frame])
        if side < 0:
            continue
        if side != current and reason[frame] in confirm:
            position = int(np.searchsorted(dead_frames, frame))
            next_dead = int(dead_frames[position]) if position < len(dead_frames) else n_frames
            high = min(frame + window + 1, n_frames, next_dead)
            candidates = frame + np.flatnonzero(near[frame:high, side])
            if len(candidates) == 0:
                if drop_unconfirmed:
                    team[frame] = -1
                    reason[frame] = ""
                    dropped += 1
                    continue  # `current` keeps the previous stable side
            elif move_to_support and int(candidates[0]) != frame:
                target = int(candidates[0])
                if team[target] == side:
                    # a later anchor already establishes this possession; this one is redundant
                    team[frame] = -1
                    reason[frame] = ""
                    dropped += 1
                    continue
                if team[target] >= 0:
                    # another side owns the first supported frame; never overwrite it
                    if drop_unconfirmed:
                        team[frame] = -1
                        reason[frame] = ""
                        dropped += 1
                    continue
                label = str(reason[frame])
                team[frame] = -1
                reason[frame] = ""
                team[target] = side
                reason[target] = "confirmed_" + label
                moved += 1
                continue  # the moved anchor sets `current` when the scan reaches it
        current = side
    return moved, dropped


def loose_ball_gate(team, reason, dead, telemetry, fps=25.0,
                    min_probability=LOOSE_GATE_MIN_PROBABILITY,
                    horizon=LOOSE_GATE_HORIZON_S,
                    min_speed=LOOSE_BALL_SPEED_MPS):
    """Veto anchors that contradict the projected interception of a fast loose ball.

    A loose ball has no owner, so an event anchor written while the ball is travelling must not
    be trusted blindly. When the time-to-intercept field is confident that a different team wins
    the projected race, the old anchor is removed and the previous stable state keeps filling
    until a player actually controls the ball. If the field is ambiguous, or the ball is slow, or
    the projected winner matches the anchor, nothing changes.

    This is a gate, not a predictor: it never creates possession. Mutates `team` and `reason`;
    returns the number of anchors removed.
    """
    loose = (telemetry["ball_loose"] & telemetry["ball_seen"] & ~dead
             & (telemetry["speed"] > min_speed))
    kinematic_team = telemetry["kinematic_team"]
    probability = telemetry["kinematic_probability"]
    tti = telemetry["kinematic_tti"]
    veto = (loose & (team >= 0)
            & (kinematic_team >= 0) & (kinematic_team != team)
            & (probability >= min_probability)
            & np.isfinite(tti) & (tti <= horizon))
    removed = int(np.sum(veto))
    if removed:
        team[veto] = -1
        reason[veto] = ""
    return removed


def possession_anchors(events, n_frames, dead, ball_xy=None, pitch=None, fps=25,
                       max_wait_s=SET_PIECE_MAX_WAIT_S, players_xy=None, slot_team=None,
                       telemetry=None, use_kinematic=True,
                       restart_lead_frames=SET_PIECE_ANCHOR_LEAD_FRAMES):
    """(team_code, reason) arrays with one anchor per clear Smart Tagging possession event.

    Attacking phases name the side in possession. Defending phases name the side out of it, so
    their anchor is the other side. SET PIECES anchor at the detected restart, because the tag
    start is still the dead-ball lead-in. GOALS are not anchors: a goal is followed by a kickoff,
    which is itself a set piece.

    A SET PIECES tag with Team=N/A (every throw-in, among others) is resolved from the kinematic
    influence field when it is confident, then from the Euclidean nearest player as a fallback.
    That needs `players_xy` and `slot_team`; without them the tag still produces no anchor. A
    known team tag is never overridden by either resolver.

    Defending codes are written first and attacking codes second, so if an export tags the same
    phase both ways at the same frame the attacking tag wins. Set pieces are written last.
    """
    team = np.full(n_frames, -1, dtype=np.int8)
    reason = np.full(n_frames, "", dtype=object)
    if events is None or len(events) == 0:
        return team, reason

    for codes, invert in ((DEFENDING_CODES, True), (ATTACKING_CODES, False)):
        for event in events.itertuples(index=False):
            if event.code_key not in codes:
                continue
            side = int(event.team_code)
            if side < 0:
                continue
            if invert:
                side = AWAY if side == HOME else HOME
            frame = int(event.start_frame)
            if 0 <= frame < n_frames and not dead[frame]:
                team[frame] = side
                reason[frame] = event.code_key

    if ball_xy is not None and pitch is not None:
        restarts = set_piece_restart_frames(events, ball_xy, pitch, fps, max_wait_s)
        resolve_nearest = players_xy is not None and slot_team is not None
        for event in events.itertuples(index=False):
            if event.code_key != SET_PIECE_CODE:
                continue
            frame = int(restarts.get(int(event.start_frame), int(event.start_frame)))
            if restart_lead_frames > 0:
                earlier = max(0, frame - int(restart_lead_frames))
                if not dead[earlier]:
                    frame = earlier
            if not (0 <= frame < n_frames) or dead[frame]:
                continue

            side = int(event.team_code)
            anchor_reason = "set_pieces_restart"
            if side < 0:
                if use_kinematic and telemetry is not None:
                    kinematic_side = int(telemetry["kinematic_team"][frame])
                    kinematic_probability = float(
                        telemetry["kinematic_probability"][frame])
                    if (kinematic_side >= 0
                            and kinematic_probability >= TTI_MIN_PROBABILITY):
                        side = kinematic_side
                        anchor_reason = "set_pieces_kinematic"
                if side < 0:
                    if not resolve_nearest:
                        continue
                    nearest_team, _distance = nearest_player_at_frames(
                        np.array([frame]), ball_xy, players_xy, slot_team)
                    if nearest_team[0] < 0:
                        continue
                    side = int(nearest_team[0])
                    anchor_reason = "set_pieces_nearest"

            team[frame] = side
            reason[frame] = anchor_reason

    return team, reason


def build_possession_frame(arrays=None, events=None, game=GAME, data_dir=None, fps=None,
                           pitch_margin_m=0.0, nan_is_dead=False,
                           max_wait_s=SET_PIECE_MAX_WAIT_S,
                           use_physics=True, use_contact=True, use_kinematic=True,
                           use_set_piece_kinematic=None, use_physics_kinematic=None,
                           use_loose_kinematic=False, use_loose_gate=True,
                           physics_kinematic_mode="agree",
                           loose_gate_min_probability=LOOSE_GATE_MIN_PROBABILITY,
                           loose_gate_horizon=LOOSE_GATE_HORIZON_S,
                           loose_gate_min_speed=LOOSE_BALL_SPEED_MPS,
                           loose_min_probability=LOOSE_TTI_MIN_PROBABILITY,
                           loose_horizon=LOOSE_TTI_HORIZON_S,
                           loose_min_run=LOOSE_ANCHOR_MIN_RUN_FRAMES,
                           loose_min_speed=LOOSE_BALL_SPEED_MPS,
                           set_piece_lead_frames=SET_PIECE_ANCHOR_LEAD_FRAMES,
                           use_turnover_confirmation=True,
                           turnover_confirm_window=TURNOVER_CONFIRM_WINDOW_FRAMES,
                           turnover_confirm_radius=TURNOVER_CONFIRM_RADIUS_M,
                           turnover_confirm_drop=True,
                           turnover_confirm_move=True,
                           smooth_min_frames=18,
                           smooth_protect_reasons=SMOOTH_PROTECT_REASONS,
                           tail_frames=POSSESSION_TAIL_FRAMES,
                           use_inactivity=True,
                           inactivity_window=INACTIVITY_WINDOW_FRAMES,
                           inactivity_player_speed=INACTIVITY_PLAYER_SPEED_MPS,
                           inactivity_ball_speed=INACTIVITY_BALL_SPEED_MPS,
                           inactivity_ball_distance=INACTIVITY_BALL_DISTANCE_M,
                           inactivity_missing_fraction=INACTIVITY_MISSING_FRACTION,
                           inactivity_min_frames=INACTIVITY_MIN_FRAMES):
    """One row per video frame with telemetry, anchors and the final possession state.

    The state machine is still the same vectorized steps:

      1. write D on every dead frame and the anchor side on every anchor frame;
      2. pandas ffill() from the previous row;
      3. absorb H/A runs shorter than `smooth_min_frames` into the previous stable state;
      4. extend the surviving H/A spells by `tail_frames` frames to correct event latency.

    Phase 2 adds evidence before step 2. `confirm_live_turnovers` first holds back live event
    anchors that change possession until the named side actually reaches the ball, because a
    tackle or pass tag is written at the challenge, not when the new side controls it.
    `physics_anchors` catches a hard touch from a sharp change in the ball's trajectory and
    speed; its time-to-intercept field acts as an agreement gate, so the nearest player keeps
    naming the side unless the kinematics clearly support it.
    `contact_anchors` catches sustained control by a player who is clearly nearest to the ball.
    Team=N/A set pieces are resolved from the same kinematic field. Finally `loose_ball_gate`
    vetoes anchors that contradict the projected interception of a fast loose ball, because a
    ball in flight has no owner: the previous stable state keeps filling until a player actually
    controls it.

    Dead-ball evidence is event-free in its last layer: `inactivity_mask` marks a sustained collapse
    in player movement, combined with a slow, distant or missing ball, as a stoppage. That catches
    the untagged dead time between a goal and the next kickoff, and the dead time inside a long
    set-piece tag, without requiring the tag stream to name it.

    `loose_ball_anchors` is the optional predictive counterpart. It is off by default because on
    IDSSE its early anchors were less reliable than the event stream; pass
    ``use_loose_kinematic=True`` to enable it.

    Missing ball observations are unknown, not dead, so ffill crosses them. Frames before the
    first anchor stay X.

    The final smoothing pass groups contiguous identical states and removes only short H/A runs;
    D runs are preserved, except short runs backed by a physics anchor (SMOOTH_PROTECT_REASONS).
    Pass ``smooth_min_frames=None`` (or a value <= 1) to disable it.

    Returns columns: frame, ball_x, ball_y, ball_seen, outside_pitch, dead, dead_reason,
    ball_speed_mps, ball_vx, ball_vy, ball_angle_deg, trajectory_turn_deg, speed_change_mps,
    nearest_distance_m, nearest_team, second_distance_m, ball_loose, contact, physics_candidate,
    anchor, anchor_reason, state.
    """
    fps = fps or GAMES[game]["fps"]
    arrays = load(game, data_dir) if arrays is None else arrays
    events = load_events(game, data_dir, fps) if events is None else events

    ball_xy = arrays["ball_xy"]
    players_xy = arrays["players_xy"]
    slot_team = arrays["slot_team"]
    pitch = arrays["pitch"]
    n_frames = len(ball_xy)

    telemetry = ball_telemetry(ball_xy, players_xy, slot_team, fps)
    inactivity = None
    if use_inactivity:
        inactivity = inactivity_mask(
            ball_xy, players_xy, fps, telemetry=telemetry,
            window=inactivity_window,
            player_speed_mps=inactivity_player_speed,
            ball_speed_mps=inactivity_ball_speed,
            ball_distance_m=inactivity_ball_distance,
            missing_fraction=inactivity_missing_fraction,
            min_frames=inactivity_min_frames)
    dead, dead_reason, outside = _dead_ball_components(
        ball_xy, pitch, events, fps, pitch_margin_m, nan_is_dead, max_wait_s,
        inactivity=inactivity)
    set_piece_kinematic = (use_kinematic if use_set_piece_kinematic is None
                           else bool(use_set_piece_kinematic))
    physics_kinematic = (use_kinematic if use_physics_kinematic is None
                         else bool(use_physics_kinematic))
    loose_kinematic = bool(use_loose_kinematic)
    team, anchor_reason = possession_anchors(
        events, n_frames, dead, ball_xy, pitch, fps, max_wait_s,
        players_xy=players_xy, slot_team=slot_team, telemetry=telemetry,
        use_kinematic=set_piece_kinematic,
        restart_lead_frames=set_piece_lead_frames)

    if use_turnover_confirmation:
        confirm_live_turnovers(
            team, anchor_reason, dead, ball_xy, players_xy, slot_team,
            window_frames=turnover_confirm_window,
            radius=turnover_confirm_radius,
            drop_unconfirmed=turnover_confirm_drop,
            move_to_support=turnover_confirm_move)

    if use_physics:
        physics_anchors(team, anchor_reason, dead, telemetry,
                        use_kinematic=physics_kinematic,
                        kinematic_mode=physics_kinematic_mode)
    if use_contact:
        contact_anchors(team, anchor_reason, dead, telemetry)
    if use_loose_gate:
        loose_ball_gate(team, anchor_reason, dead, telemetry, fps,
                        min_probability=loose_gate_min_probability,
                        horizon=loose_gate_horizon,
                        min_speed=loose_gate_min_speed)
    if loose_kinematic:
        loose_ball_anchors(team, anchor_reason, dead, telemetry, fps,
                           min_probability=loose_min_probability,
                           horizon=loose_horizon, min_run=loose_min_run,
                           min_speed=loose_min_speed)

    state = _state_codes(team, dead)
    anchored = (team >= 0) & ~dead
    anchor = np.full(n_frames, "", dtype=object)
    anchor[anchored] = LABELS[team[anchored]]

    nearest_team = telemetry["nearest_team"]
    nearest_label = np.full(n_frames, "", dtype=object)
    known = nearest_team >= 0
    nearest_label[known] = LABELS[nearest_team[known]]
    nearest_distance = np.where(np.isfinite(telemetry["nearest_distance"]),
                                telemetry["nearest_distance"], np.nan)
    second_distance = np.where(np.isfinite(telemetry["second_distance"]),
                               telemetry["second_distance"], np.nan)

    x, y = ball_xy[:, 0], ball_xy[:, 1]
    table = pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int32),
        "ball_x": x.astype(np.float32),
        "ball_y": y.astype(np.float32),
        "ball_seen": telemetry["ball_seen"],
        "outside_pitch": outside,
        "dead": dead,
        "dead_reason": dead_reason,
        "ball_speed_mps": telemetry["speed"].astype(np.float32),
        "ball_vx": telemetry["velocity_after"][:, 0].astype(np.float32),
        "ball_vy": telemetry["velocity_after"][:, 1].astype(np.float32),
        "ball_angle_deg": telemetry["angle"].astype(np.float32),
        "trajectory_turn_deg": telemetry["turn"].astype(np.float32),
        "speed_change_mps": telemetry["speed_change"].astype(np.float32),
        "nearest_distance_m": nearest_distance.astype(np.float32),
        "nearest_team": nearest_label,
        "second_distance_m": second_distance.astype(np.float32),
        "ball_loose": telemetry["ball_loose"],
        "contact": telemetry["contact"],
        "physics_candidate": telemetry["physics"],
        "anchor": anchor,
        "anchor_reason": anchor_reason,
        "state": LABELS[state],
    })
    if smooth_min_frames is not None and int(smooth_min_frames) > 1:
        table["state"] = smooth_micro_possessions(
            table["state"], min_frames=int(smooth_min_frames),
            anchor_reason=table["anchor_reason"],
            protect_reasons=smooth_protect_reasons or ())
    if tail_frames is not None and int(tail_frames) > 0:
        table["state"] = extend_possession_tails(
            table["state"], frames=int(tail_frames))
    return table


def possession_changes(table):
    """Rows where the state differs from the previous frame, with the cause visible.

    The first row is not a change. `prev_frame` is the last frame of the previous state, so a
    boundary at frame f is the half-open interval [prev_frame + 1, f) of the new state.
    """
    changed = table["state"].ne(table["state"].shift())
    if len(changed) > 0:
        changed.iloc[0] = False
    previous = table.shift()
    out = pd.DataFrame({
        "frame": table.loc[changed, "frame"].to_numpy(dtype=np.int64),
        "prev_frame": previous.loc[changed, "frame"].to_numpy(dtype=np.int64),
        "prev_state": previous.loc[changed, "state"].to_numpy(),
        "state": table.loc[changed, "state"].to_numpy(),
        "dead": table.loc[changed, "dead"].to_numpy(),
        "anchor": table.loc[changed, "anchor"].to_numpy(),
        "anchor_reason": table.loc[changed, "anchor_reason"].to_numpy(),
        "dead_reason": table.loc[changed, "dead_reason"].to_numpy(),
    })
    out.insert(4, "change", out["prev_state"].astype(str) + "->" + out["state"].astype(str))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild the cache")
    args = parser.parse_args()

    arrays = load(force=args.force)
    print("cache: {}".format(os.path.relpath(cache_path(), ROOT)))
    for key, value in summary(arrays).items():
        print("  {:<18} {}".format(key, round(value, 4) if isinstance(value, float) else value))
