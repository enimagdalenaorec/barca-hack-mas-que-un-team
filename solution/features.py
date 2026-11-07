"""Per-frame features for "which team has the ball", computed per team and never per player.

Everything is from the point of view of one team, "us" (slot_team == 0 unless swapped), with the
pitch flipped so that we always attack towards +x. Computing the same features with the teams
swapped gives the mirror image of every frame, which doubles the training set and removes any
home/away bias; at prediction time both views are averaged (solution/model.py).

Groups:

  around the ball   distance from the ball to our nearest and second-nearest player and theirs,
                    how many of each within 3 m and 7 m, ball speed (from positions -- our ATD's
                    exported ball speed is unusable), where the ball is on the pitch
  over time         the same "who is closer" evidence averaged over the last and next 1 s and
                    3 s, so a single touch weighs less than control held for a while
  ball visibility   seconds since the ball was last seen and until it is next seen
  team shape        each team's centroid along the attacking direction, spread, speed and
                    forward velocity -- the team in possession spreads out and pushes up, which
                    still works while the ball is missing
  ball flight       where the ball will be in 0.4 s and 1 s if it keeps its velocity, and how
                    close each team is to that point -- the intended receiver of a pass
  control           how differently the ball and each team's nearest player move (a dribbled ball
                    moves with its player), and how many opponents press that player
  long context      the "who is closer" evidence over the last and next 5 s and 10 s

NaN stays NaN: gradient boosting handles missing values natively, which matters because the ball
is absent on a large share of frames.
"""

import numpy as np

# Params
FPS = 25
RADII_M = (3.0, 7.0)
WINDOWS_S = (1.0, 3.0)
LONG_WINDOWS_S = (5.0, 10.0)
FLIGHT_S = (0.4, 1.0)
PRESS_M = 5.0
VEL_LAG = 2  # frames, for velocities from positions
GAP_CAP_S = 10.0
SPEED_SMOOTH = 5  # frames, for ball speed from positions


def _rolling_mean(values, window, direction):
    """Mean of non-NaN values over the past (direction -1) or next (+1) `window` frames."""
    valid = ~np.isnan(values)
    v = np.where(valid, values, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(values))
    if direction < 0:
        lo, hi = np.maximum(0, idx - window + 1), idx + 1
    else:
        lo, hi = idx, np.minimum(len(values), idx + window)
    count = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / count, np.nan)


def _time_since_seen(seen):
    """Frames since the last True (0 on a True frame), capped; inf-like cap before the first."""
    idx = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, idx, -10 ** 9))
    return np.minimum(idx - last, int(GAP_CAP_S * FPS)).astype(np.float32)


def attack_signs(arrays, runs):
    """Per run, +1 if team 0 attacks towards +x, else -1: team 0 sits deeper on its own half."""
    signs = []
    xy, team = arrays["players_xy"], arrays["slot_team"]
    for start, stop in runs:
        x0 = np.nanmean(xy[start:stop][:, team == 0, 0])
        x1 = np.nanmean(xy[start:stop][:, team == 1, 0])
        signs.append(1.0 if x0 < x1 else -1.0)
    return signs


def compute(arrays, runs, swap=False):
    """(names, matrix float32 [n_frames, n_features]) for every frame; rows outside runs are NaN."""
    team = arrays["slot_team"].astype(int)
    if swap:
        team = 1 - team
    length = float(arrays["pitch"][0])
    ball = arrays["ball_xy"].astype(np.float64).copy()
    players = arrays["players_xy"].astype(np.float64).copy()
    n = len(ball)

    # Orient every run so that "us" attack towards +x.
    for (start, stop), sign in zip(runs, attack_signs({**arrays, "slot_team": team}, runs)):
        if sign < 0:
            ball[start:stop, 0] = length - ball[start:stop, 0]
            players[start:stop, :, 0] = length - players[start:stop, :, 0]

    cols = {}
    offset = players - ball[:, None, :]
    dist = np.sqrt(np.square(offset).sum(-1))  # NaN where ball or player missing
    for side, name in ((0, "us"), (1, "them")):
        d = np.where(np.isnan(dist[:, team == side]), np.inf, dist[:, team == side])
        part = np.sort(d, axis=1)[:, :2]
        part[np.isinf(part)] = np.nan
        cols["d1_" + name] = part[:, 0]
        cols["d2_" + name] = part[:, 1]
        for r in RADII_M:
            cols["n{:g}_{}".format(r, name)] = np.where(
                np.isnan(ball[:, 0]), np.nan, (d < r).sum(axis=1))
    cols["d1_diff"] = cols["d1_them"] - cols["d1_us"]
    near_us = np.where(np.isnan(cols["d1_diff"]), np.nan, (cols["d1_diff"] > 0).astype(float))

    step = np.linalg.norm(np.diff(ball, axis=0), axis=-1) * FPS
    step = np.concatenate([[np.nan], step])
    cols["ball_speed"] = _rolling_mean(step, SPEED_SMOOTH, -1)
    cols["ball_x"] = ball[:, 0]
    cols["ball_y_off"] = np.abs(ball[:, 1] - arrays["pitch"][1] / 2.0)

    for w in WINDOWS_S:
        frames = int(w * FPS)
        for direction, tag in ((-1, "past"), (1, "next")):
            cols["near_us_{}{:g}".format(tag, w)] = _rolling_mean(near_us, frames, direction)
            cols["d1_diff_{}{:g}".format(tag, w)] = _rolling_mean(cols["d1_diff"], frames, direction)

    seen = ~np.isnan(ball[:, 0])
    cols["since_seen_s"] = _time_since_seen(seen) / FPS
    cols["until_seen_s"] = _time_since_seen(seen[::-1])[::-1] / FPS
    last_x = ball[:, 0].copy()
    idx = np.where(seen, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    last_x = np.where(idx >= 0, ball[np.maximum(idx, 0), 0], np.nan)
    cols["last_ball_x"] = last_x

    for side, name in ((0, "us"), (1, "them")):
        xs, ys = players[:, team == side, 0], players[:, team == side, 1]
        with np.errstate(all="ignore"):
            cx = np.nanmean(xs, axis=1)
            cols["cx_" + name] = cx
            cols["sx_" + name] = np.nanstd(xs, axis=1)
            cols["sy_" + name] = np.nanstd(ys, axis=1)
            cols["count_" + name] = (~np.isnan(xs)).sum(axis=1).astype(float)
            vx = np.concatenate([[np.nan], np.diff(cx)]) * FPS
            cols["vx_" + name] = _rolling_mean(vx, FPS, -1)
    cols["cx_gap"] = cols["cx_us"] - cols["cx_them"]
    cols["spread_ratio"] = (cols["sx_us"] * cols["sy_us"]) / (cols["sx_them"] * cols["sy_them"])
    cols["ball_vs_cx_us"] = ball[:, 0] - cols["cx_us"]
    cols["ball_vs_cx_them"] = ball[:, 0] - cols["cx_them"]

    # Ball flight: distance from each team to where the ball is heading.
    ball_v = np.full_like(ball, np.nan)
    ball_v[VEL_LAG:] = (ball[VEL_LAG:] - ball[:-VEL_LAG]) * FPS / VEL_LAG
    for horizon in FLIGHT_S:
        target = ball + ball_v * horizon
        to_target = np.sqrt(np.square(players - target[:, None, :]).sum(-1))
        for side, name in ((0, "us"), (1, "them")):
            d = to_target[:, team == side]
            with np.errstate(all="ignore"):
                best = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
            cols["flight{:g}_{}".format(horizon, name)] = np.where(np.isinf(best), np.nan, best)
        cols["flight{:g}_diff".format(horizon)] = (cols["flight{:g}_them".format(horizon)]
                                                   - cols["flight{:g}_us".format(horizon)])

    # Control: does the ball move with each team's nearest player, and is that player pressed?
    players_v = np.full_like(players, np.nan)
    players_v[VEL_LAG:] = (players[VEL_LAG:] - players[:-VEL_LAG]) * FPS / VEL_LAG
    rows_ = np.arange(n)
    for side, name in ((0, "us"), (1, "them")):
        cols_side = np.flatnonzero(team == side)
        d = np.where(np.isnan(dist[:, cols_side]), np.inf, dist[:, cols_side])
        nearest = cols_side[np.argmin(d, axis=1)]
        found = np.isfinite(d.min(axis=1))
        rel = np.linalg.norm(ball_v - players_v[rows_, nearest], axis=1)
        cols["comove_" + name] = np.where(found, rel, np.nan)
        opp = np.flatnonzero(team != side)
        gap = np.sqrt(np.square(players[:, opp] - players[rows_, nearest][:, None, :]).sum(-1))
        cols["pressed_" + name] = np.where(found, (gap < PRESS_M).sum(axis=1), np.nan)

    for w in LONG_WINDOWS_S:
        frames = int(w * FPS)
        for direction, tag in ((-1, "past"), (1, "next")):
            cols["near_us_{}{:g}".format(tag, w)] = _rolling_mean(near_us, frames, direction)

    names = sorted(cols)
    matrix = np.full((n, len(names)), np.nan, dtype=np.float32)
    inside = np.zeros(n, dtype=bool)
    for start, stop in runs:
        inside[start:stop] = True
    for j, name in enumerate(names):
        matrix[inside, j] = np.asarray(cols[name], dtype=np.float64)[inside]
    return names, matrix
