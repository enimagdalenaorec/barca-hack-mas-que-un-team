"""Write the frame,label submission produced by our possession state machine.

    python -m solution.make_submission
    python -m solution.make_submission --out evaluation/results/telemetry_state_machine.csv
    python -m solution.make_submission --no-physics --no-contact   # Phase 1 ablation

The only input is the cached tactical ATD plus the Smart Tagging XML. The output is exactly the
submission contract: one header line `frame,label`, then one row per video frame with labels in
H/A/D/X. `X` is written where there is no evidence yet; the scorer treats it as an abstention.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import tracking as tr  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "telemetry_state_machine.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=tr.GAME)
    parser.add_argument("--data-dir", default=None,
                        help="override the data bundle root (default: games.DATA_DIR)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="metres of tolerance around the touchlines (default: 0)")
    parser.add_argument("--nan-is-dead", action="store_true",
                        help="treat a missing ball as dead instead of unknown")
    parser.add_argument("--no-physics", action="store_true",
                        help="skip the sharp-trajectory anchors")
    parser.add_argument("--no-contact", action="store_true",
                        help="skip the sustained-control anchors")
    args = parser.parse_args()

    table = tr.build_possession_frame(
        game=args.game, data_dir=args.data_dir, pitch_margin_m=args.margin,
        nan_is_dead=args.nan_is_dead,
        use_physics=not args.no_physics, use_contact=not args.no_contact)

    folder = os.path.dirname(os.path.abspath(args.out))
    if folder:
        os.makedirs(folder, exist_ok=True)
    table[["frame", "state"]].rename(columns={"state": "label"}).to_csv(
        args.out, index=False, lineterminator="\n")

    counts = table["state"].value_counts().reindex(tr.LABELS, fill_value=0)
    total = len(table)
    print("wrote {}".format(args.out))
    print("  " + "  ".join(
        "{} {:>6} ({:5.1f}%)".format(label, counts[label], 100.0 * counts[label] / total)
        for label in tr.LABELS))
    print("  dead {:>6} ({:5.1f}%)".format(int(table["dead"].sum()),
                                           100.0 * table["dead"].mean()))
    reasons = table.loc[table["anchor_reason"] != "", "anchor_reason"].value_counts()
    print("\nanchors")
    print(reasons.to_string() if len(reasons) else "  (none)")


if __name__ == "__main__":
    main()
