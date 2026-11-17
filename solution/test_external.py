"""Download, convert, run and score one Metrica Sports sample game.

    python -m solution.test_external --game Sample_Game_1
    python -m solution.test_external --game Sample_Game_2 --skip-download

This is an external held-out check, not a replacement for the BAR-ATH scorer. It downloads the
three public Metrica CSVs (home tracking, away tracking, events), converts them to the same ATD
raw/metadata layout solution/tracking.py reads, converts the event stream to a Smart Tagging XML
with the same instance/label structure, then feeds both through the existing
build_possession_frame() without changing any core possession logic.

The ground truth is derived independently from the Metrica events:

* PASS / RECOVERY / SET PIECE / SHOT name the team in possession at that touch.
* A CHALLENGE ending in -WON gives possession to that team; -LOST gives it to the opponent.
* BALL LOST does not name the next possessor, so it does not create a possession marker; the next
  RECOVERY/PASS/CHALLENGE does. This is a last-touch definition.
* BALL OUT, FAULT RECEIVED, CARD, OFFSIDE, END HALF and goals create dead-ball intervals that run
  until the next SET PIECE in the same period (or the period end).

Every generated file lands under data/external/metrica/, which is gated by .gitignore. The script
prints macro-F1, spell-F1 at the headline IoU and the supporting scoreboard.
"""

import argparse
import io
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from solution import tracking as tr  # noqa: E402
from solution import external_games as eg  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25.0
TEAM_CODES = {"home": tr.HOME, "away": tr.AWAY}
TEAM_NAMES = {tr.HOME: "Home", tr.AWAY: "Away"}
GT_TEAM_LABELS = {tr.HOME: "H", tr.AWAY: "A"}
DEAD_LOST_SUBTYPES = {"END HALF", "OFFSIDE", "REFEREE HIT"}
BENCHMARK = 0.645
SET_PIECE_WINDOW_S = 5.0

# Same marker order as the GT generator. Higher priority is applied later when two events share a
# start frame, so a recovery beats a ball-loss row and a goal/ball-out beats the touch that caused it.
POSSESSION_PRIORITY = {
    "CHALLENGE": 30,
    "SHOT": 40,
    "PASS": 50,
    "RECOVERY": 60,
    "SET PIECE": 70,
}


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _upper(value):
    return _text(value).upper()


def _opponent(team):
    if team == tr.HOME:
        return tr.AWAY
    if team == tr.AWAY:
        return tr.HOME
    return -1


def tracking_slot_ids(path):
    """Player ids from a Metrica tracking CSV's third header row, in slot order."""
    raw = pd.read_csv(path, header=None, dtype=str, nrows=3, low_memory=False)
    header = [str(value).strip() for value in raw.iloc[2].tolist()]
    if "Ball" not in header:
        raise ValueError("{}: no Ball column in the header".format(path))
    ball_col = header.index("Ball")
    return [header[column] for column in range(3, ball_col, 2)]


def period_stop_map(period):
    """{period number: first global row after that period}."""
    return {int(period[start]): stop for start, stop in eg.period_runs(period)}


def _period_speeds(xy, runs, fps=FPS):
    """Speed per frame, computed inside each period so halftime never becomes a velocity jump."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    for start, stop in runs:
        if stop - start > 1:
            step = np.linalg.norm(np.diff(xy[start:stop], axis=0), axis=-1) * fps
            out[start + 1:stop] = step.astype(np.float32)
    return out


def write_atd(arrays, home_ids, away_ids, raw_path, meta_path, period):
    """Write the exact ATD text/metadata layout tracking.parse() consumes.

    Metrica's frame counters restart or jump between periods, so the generated rows are compact
    and continuous. The first frame of every period after the first is blanked for ball and
    players: that prevents a false velocity/turn/contact anchor from firing across the halftime
    boundary while leaving the event-driven set-piece dead ball intact.
    """
    n_frames = len(arrays["ball_xy"])
    n_slots = arrays["players_xy"].shape[1]
    if len(home_ids) + len(away_ids) != n_slots:
        raise ValueError("header slots {} != array slots {}".format(
            len(home_ids) + len(away_ids), n_slots))

    runs = eg.period_runs(period)
    players = arrays["players_xy"].copy()
    ball = arrays["ball_xy"].copy()
    for start, _stop in runs[1:]:
        players[start] = np.nan
        ball[start] = np.nan

    player_speed = _period_speeds(players, runs)
    ball_speed = _period_speeds(ball, runs)
    pitch = np.asarray(arrays["pitch"], dtype=np.float64)
    players_norm = players.astype(np.float64) / pitch
    ball_norm = ball.astype(np.float64) / pitch

    def fmt(value):
        return "NaN" if not np.isfinite(value) else "{:.8f}".format(float(value))

    with open(raw_path, "w", newline="\n") as handle:
        for frame in range(n_frames):
            slots = []
            for slot in range(n_slots):
                x, y = players_norm[frame, slot]
                speed = player_speed[frame, slot]
                slots.append("{},{},{}".format(fmt(x), fmt(y), fmt(speed)))
            bx, by = ball_norm[frame]
            ball_text = "{},{},{}".format(fmt(bx), fmt(by), fmt(ball_speed[frame]))
            handle.write("{}:{}:{}\n".format(frame + 1, ";".join(slots), ball_text))

    root = ET.Element("main")
    teams = ET.SubElement(root, "Teams")
    ET.SubElement(teams, "Team", {"id": "home"}).text = "Home"
    ET.SubElement(teams, "Team", {"id": "away"}).text = "Away"
    players_el = ET.SubElement(root, "Players")
    for team_id, ids in (("home", home_ids), ("away", away_ids)):
        for slot, player_id in enumerate(ids):
            ET.SubElement(players_el, "Player", {
                "id": "{}_{}".format(team_id, slot),
                "teamId": team_id,
            }).text = str(player_id)
    ET.ElementTree(root).write(meta_path, encoding="utf-8", xml_declaration=True)
    return players, ball


def read_atd(raw_path, meta_path, pitch, n_frames):
    """Read the generated ATD exactly as tracking.parse() would, without needing a GAMES entry."""
    slot_team = tr.slot_teams(meta_path)
    n_slots = len(slot_team)
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        engine="c").to_numpy()
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


def load_metrica_events(path, frame_map):
    """All Metrica events with global row indexes and normalised type/subtype fields."""
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    required = {"Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("{}: missing event columns {}".format(path, sorted(missing)))

    events = raw.rename(columns={
        "Team": "team", "Type": "kind", "Subtype": "subtype", "Period": "period",
        "Start Frame": "start_local", "End Frame": "end_local",
        "Start Time [s]": "start_s", "End Time [s]": "end_s",
    })
    rows = []
    for row in events.to_dict("records"):
        period_i = int(row["period"]) if pd.notna(row["period"]) else -1
        start_local = int(row["start_local"]) if pd.notna(row["start_local"]) else None
        if period_i < 0 or start_local is None:
            continue
        start_frame = frame_map.get((period_i, start_local))
        if start_frame is None:
            continue
        end_local = int(row["end_local"]) if pd.notna(row["end_local"]) else start_local
        end_frame = frame_map.get((period_i, end_local), start_frame)
        if end_frame <= start_frame:
            end_frame = start_frame
        rows.append({
            "period": period_i,
            "team": _text(row["team"]),
            "team_code": TEAM_CODES.get(_text(row["team"]).lower(), -1),
            "kind": _upper(row["kind"]),
            "subtype": _upper(row["subtype"]),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_local": start_local,
            "end_local": end_local,
        })
    return pd.DataFrame(rows).sort_values(
        ["period", "start_frame", "kind"], kind="stable").reset_index(drop=True)


def _set_piece_starts(events):
    starts = {}
    for row in events.to_dict("records"):
        if row["kind"] == "SET PIECE":
            starts.setdefault(int(row["period"]), []).append(int(row["start_frame"]))
    return {period: sorted(set(frames)) for period, frames in starts.items()}


def _next_set_piece(starts, stop_map, period, frame):
    candidates = starts.get(period, [])
    index = int(np.searchsorted(candidates, frame, side="right"))
    if index < len(candidates):
        return candidates[index]
    return stop_map.get(period, frame)


def dead_intervals(events, stop_map):
    """Merged [start, stop) dead-ball spans derived from Metrica's stoppage events."""
    set_pieces = _set_piece_starts(events)
    spans = []
    for row in events.to_dict("records"):
        kind, subtype = row["kind"], row["subtype"]
        start = int(row["start_frame"])
        dead = False
        if kind == "BALL OUT":
            dead = True
        elif kind in ("FAULT RECEIVED", "CARD"):
            dead = True
        elif kind == "BALL LOST" and subtype in DEAD_LOST_SUBTYPES:
            dead = True
        elif kind == "SHOT" and "GOAL" in subtype:
            dead = True
            start = int(row["end_frame"]) if row["end_frame"] > start else start
        if not dead:
            continue

        stop = _next_set_piece(set_pieces, stop_map, int(row["period"]), start)
        if stop <= start:
            continue
        spans.append((start, stop))

    spans.sort()
    merged = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _period_for_frame(frame, stop_map):
    for period, stop in sorted(stop_map.items()):
        if frame < stop:
            return period
    return max(stop_map) if stop_map else -1


def derive_gt(events, n_frames, stop_map):
    """Event-derived last-touch possession labels with stoppage spans overwritten as D."""
    gt = np.full(n_frames, "D", dtype=object)

    # Build per-period marker lists directly from the event rows. Markers at the same frame are
    # ordered by priority so the higher-priority row writes the interval that follows.
    by_period = {}
    for row in events.to_dict("records"):
        kind = row["kind"]
        team = int(row["team_code"])
        subtype = row["subtype"]
        if team < 0:
            continue
        if kind in ("PASS", "RECOVERY", "SHOT", "SET PIECE"):
            label = team
        elif kind == "CHALLENGE":
            if subtype.endswith("-WON"):
                label = team
            elif subtype.endswith("-LOST"):
                label = _opponent(team)
            else:
                continue
        else:
            continue
        by_period.setdefault(int(row["period"]), []).append(
            (int(row["start_frame"]), POSSESSION_PRIORITY.get(kind, 20), label))

    for period, period_markers in by_period.items():
        period_markers.sort()
        stop_period = stop_map.get(period, n_frames)
        for index, (start, _priority, label) in enumerate(period_markers):
            stop = period_markers[index + 1][0] if index + 1 < len(period_markers) \
                else stop_period
            if stop > start:
                gt[start:stop] = GT_TEAM_LABELS.get(label, "D")

    for start, stop in dead_intervals(events, stop_map):
        gt[start:stop] = "D"
    return gt


def smart_tagging_instances(events, stop_map):
    """Metrica events expressed as Smart Tagging instances for the existing anchor parser."""
    instances = []
    for row in events.to_dict("records"):
        kind = row["kind"]
        subtype = row["subtype"]
        team = int(row["team_code"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        period = int(row["period"])
        if kind == "PASS":
            instances.append(("PASS", team, start, max(end, start + 1), subtype, period))
        elif kind == "RECOVERY":
            instances.append(("INTERCEPTION", team, start, max(end, start + 1),
                              subtype, period))
        elif kind == "SHOT":
            instances.append(("CREATING CHANCES", team, start, max(end, start + 1),
                              subtype, period))
        elif kind == "CHALLENGE" and team >= 0:
            if subtype.endswith("-WON"):
                instances.append(("TACKLE", team, start, max(end, start + 1),
                                  subtype, period))
            elif subtype.endswith("-LOST"):
                instances.append(("TACKLE", _opponent(team), start,
                                  max(end, start + 1), subtype, period))
        elif kind == "SET PIECE" and team >= 0:
            window = start + int(round(SET_PIECE_WINDOW_S * FPS))
            instances.append(("SET PIECES", team, start, max(end, window), subtype, period))

    for start, stop in dead_intervals(events, stop_map):
        instances.append(("GOALS", -1, start, stop, "Stoppage",
                          _period_for_frame(start, stop_map)))

    instances.sort(key=lambda item: (item[2], item[0]))
    return instances


def write_smart_tagging(path, instances, fps=FPS):
    root = ET.Element("file")
    all_instances = ET.SubElement(root, "ALL_INSTANCES")
    for index, (code, team, start, end, kind, period) in enumerate(instances, start=1):
        node = ET.SubElement(all_instances, "instance")
        ET.SubElement(node, "ID").text = str(index)
        ET.SubElement(node, "start").text = "{:.3f}".format(start / fps)
        ET.SubElement(node, "end").text = "{:.3f}".format(end / fps)
        ET.SubElement(node, "code").text = code
        labels = (
            ("Team", TEAM_NAMES.get(team, "N/A")),
            ("Type", kind or "Metrica"),
            ("Half", "1st Half" if period == 1 else "2nd Half"),
        )
        for group, text in labels:
            label = ET.SubElement(node, "label")
            ET.SubElement(label, "group").text = group
            ET.SubElement(label, "text").text = text
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_smart_tagging(path, fps=FPS):
    """The same DataFrame shape tracking.load_events() produces, for the generated XML."""
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
            "team_code": TEAM_CODES.get((team or "").strip().lower(), -1),
            "type": tags.get("Type"),
            "half": tags.get("Half"),
        })
    return pd.DataFrame(rows)


def score(gt_labels, pred_labels, period):
    gt_codes = pm.label_codes(gt_labels)
    pred_codes = pm.label_codes(pred_labels)
    runs = eg.period_runs(period)
    cm3 = pm.scored_confusion(pm.confusion(gt_codes, pred_codes))
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
    return {
        "macro_f1": pm.macro_f1(cm3),
        "spell_f1_mean": spells["mean_f1"],
        "spell_f1_at_0.5": spells["per_iou"][0.5]["f1"],
        "n_gt_spells": spells["n_gt"],
        "n_pred_spells": spells["n_pred"],
        "gt_mean_spell_duration_s": gt_stats["mean_duration_s"],
        "pred_mean_spell_duration_s": pred_stats["mean_duration_s"],
        "accuracy": pm.accuracy(cm3),
        "alive_dead_f1": pm.alive_dead_f1(cm3),
        "per_class": per_class,
        "confusion": cm3.tolist(),
        "n_frames": int(len(gt_codes)),
        "n_graded": int(cm3.sum()),
    }


def print_scoreboard(game, result):
    print("\n=== {} external evaluation ===".format(game))
    print("frames {}  graded {}".format(result["n_frames"], result["n_graded"]))
    print("possessions  GT {}  pred {}  (mean {:.1f}s vs {:.1f}s)".format(
        result["n_gt_spells"], result["n_pred_spells"],
        result["gt_mean_spell_duration_s"], result["pred_mean_spell_duration_s"]))
    print("\n{:<10} {:>10}".format("metric", "score"))
    print("-" * 22)
    print("{:<10} {:>10.4f}".format("macroF1", result["macro_f1"]))
    print("{:<10} {:>10.4f}".format("spellF1", result["spell_f1_mean"]))
    print("{:<10} {:>10.4f}".format("spell@0.5", result["spell_f1_at_0.5"]))
    print("{:<10} {:>10.4f}".format("alive/dead", result["alive_dead_f1"]))
    print("{:<10} {:>10.4f}".format("accuracy", result["accuracy"]))
    print("\n{:<4} {:>9} {:>9} {:>9} {:>9}".format("cls", "precision", "recall", "f1", "support"))
    print("-" * 46)
    for label in ("H", "A", "D"):
        row = result["per_class"][label]
        print("{:<4} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}".format(
            label, row["precision"], row["recall"], row["f1"], row["support"]))
    print("\nbenchmark {:.4f}  macro delta {:+.4f}  spell delta {:+.4f}".format(
        BENCHMARK, result["macro_f1"] - BENCHMARK,
        result["spell_f1_mean"] - BENCHMARK))


def run_game(game, data_dir, force=False, skip_download=False):
    folder = Path(data_dir) / game
    paths = {}
    for key, pattern in eg.FILES.items():
        filename = pattern.format(game=game)
        destination = folder / filename
        if skip_download:
            if not destination.is_file():
                raise FileNotFoundError("--skip-download but {} is missing".format(destination))
            paths[key] = destination
        else:
            url = "{}/{}/{}".format(eg.RAW_BASE, game, filename)
            paths[key] = eg.download(url, destination, force=force)

    arrays, period, frame_map = eg.build_arrays(paths["tracking_home"], paths["tracking_away"])
    n_frames = len(arrays["ball_xy"])
    stop_map = period_stop_map(period)
    home_ids = tracking_slot_ids(paths["tracking_home"])
    away_ids = tracking_slot_ids(paths["tracking_away"])

    atd_raw = folder / "{}_atd.txt".format(game)
    atd_meta = folder / "{}_atd.xml".format(game)
    smart_tagging = folder / "{}_smart_tagging.xml".format(game)
    gt_csv = folder / "{}_possession_gt.csv".format(game)
    pred_csv = folder / "{}_possession_pred.csv".format(game)
    report_json = folder / "{}_external_report.json".format(game)

    print("converting {} tracking to ATD ...".format(game))
    players_written, ball_written = write_atd(
        arrays, home_ids, away_ids, atd_raw, atd_meta, period)
    arrays_atd = read_atd(atd_raw, atd_meta, arrays["pitch"], n_frames)
    if not np.allclose(arrays_atd["players_xy"], players_written, equal_nan=True):
        raise ValueError("ATD round-trip changed player positions")
    if not np.allclose(arrays_atd["ball_xy"], ball_written, equal_nan=True):
        raise ValueError("ATD round-trip changed ball positions")

    events_raw = load_metrica_events(paths["events"], frame_map)
    instances = smart_tagging_instances(events_raw, stop_map)
    write_smart_tagging(smart_tagging, instances)
    events_smart = read_smart_tagging(smart_tagging)
    print("events: {} Metrica rows -> {} Smart Tagging instances".format(
        len(events_raw), len(events_smart)))

    table = tr.build_possession_frame(
        arrays=arrays_atd, events=events_smart, fps=FPS)
    gt_labels = derive_gt(events_raw, n_frames, stop_map)

    pd.DataFrame({"frame": table["frame"], "label": table["state"]}).to_csv(
        pred_csv, index=False, lineterminator="\n")
    pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int32), "label": gt_labels}).to_csv(
        gt_csv, index=False, lineterminator="\n")

    result = score(gt_labels, table["state"].to_numpy(), period)
    result["game"] = game
    result["frames"] = n_frames
    result["periods"] = len(eg.period_runs(period))
    report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("wrote {}".format(atd_raw))
    print("wrote {}".format(atd_meta))
    print("wrote {}".format(smart_tagging))
    print("wrote {}".format(gt_csv))
    print("wrote {}".format(pred_csv))
    print("wrote {}".format(report_json))
    print_scoreboard(game, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="Sample_Game_1", choices=eg.SUPPORTED_GAMES)
    parser.add_argument("--data-dir", default=eg.DEFAULT_DATA_DIR,
                        help="download/cache root (default: data/external/metrica)")
    parser.add_argument("--force", action="store_true", help="redownload cached files")
    parser.add_argument("--skip-download", action="store_true",
                        help="use existing local files and never touch the network")
    args = parser.parse_args()
    run_game(args.game, args.data_dir, force=args.force, skip_download=args.skip_download)
