"""Run the event-driven possession state machine on the tactical match and show its boundaries.

    python -m solution.possession_demo                 # first 50 state changes
    python -m solution.possession_demo --limit 100     # more rows
    python -m solution.possession_demo --context 2     # 2 frames either side of each boundary

The script reads the cached ATD and the Smart Tagging XML only. It writes nothing and reads no
ground truth: this is a boundary check, not a scored submission.
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import tracking as tr  # noqa: E402


def _share(value, total):
    return 100.0 * value / total if total else 0.0


def _print_context(table, changes, context):
    """Show the frames around the first few boundaries, not just the boundary rows."""
    if context <= 0 or len(changes) == 0:
        return
    print("\nboundary context ({} frames either side of the first {} changes)".format(
        context, min(3, len(changes))))
    columns = ["frame", "ball_x", "ball_y", "outside_pitch", "dead", "dead_reason",
               "anchor", "anchor_reason", "state"]
    for frame in changes["frame"].head(3):
        low = max(0, int(frame) - context)
        high = min(len(table), int(frame) + context + 1)
        print("\n-- change at frame {} --".format(frame))
        print(table.iloc[low:high][columns].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=tr.GAME)
    parser.add_argument("--data-dir", default=None,
                        help="override the data bundle root (default: games.DATA_DIR)")
    parser.add_argument("--limit", type=int, default=50,
                        help="state changes to print (default: 50)")
    parser.add_argument("--context", type=int, default=2,
                        help="frames either side of the first boundaries (default: 2)")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="metres of tolerance around the touchlines (default: 0)")
    parser.add_argument("--nan-is-dead", action="store_true",
                        help="treat a missing ball as dead instead of unknown")
    args = parser.parse_args()

    arrays = tr.load(args.game, args.data_dir)
    events = tr.load_events(args.game, args.data_dir)
    table = tr.build_possession_frame(
        arrays=arrays, events=events, game=args.game, data_dir=args.data_dir,
        pitch_margin_m=args.margin, nan_is_dead=args.nan_is_dead)

    total = len(table)
    counts = table["state"].value_counts().reindex(tr.LABELS, fill_value=0)
    print("game {}  {} frames  {} Smart Tagging instances".format(
        args.game, total, len(events)))
    print("states   " + "  ".join(
        "{} {:>6} ({:5.1f}%)".format(label, counts[label], _share(counts[label], total))
        for label in tr.LABELS))
    print("dead     {:>6} frames ({:.1f}%)".format(
        int(table["dead"].sum()), _share(int(table["dead"].sum()), total)))
    print("\ndead reasons")
    print(table.loc[table["dead"], "dead_reason"].value_counts().to_string())
    print("\nanchor reasons")
    anchors = table.loc[table["anchor"] != "", "anchor_reason"].value_counts()
    print(anchors.to_string() if len(anchors) else "  (none)")

    changes = tr.possession_changes(table)
    print("\nfirst {} of {} state changes".format(min(args.limit, len(changes)), len(changes)))
    columns = ["frame", "prev_frame", "prev_state", "state", "change", "dead",
               "anchor", "anchor_reason", "dead_reason"]
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_columns", None):
        print(changes.head(args.limit)[columns].to_string(index=False))
    _print_context(table, changes, args.context)


if __name__ == "__main__":
    main()
