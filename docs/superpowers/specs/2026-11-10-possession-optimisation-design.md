# Possession pipeline optimisation: design

Date: 2026-11-10. Branch: `bruno`, cut from `eni` at 0f60372. Deadline: 2026-11-17.

## Goal

Raise the full-match `spellF1` of Eni's pipeline (model + Viterbi, held-out 0.645 on 7 other
matches) without breaking the rules: labels from an algorithm only, no tuning on the public
5 minutes, and a repository that regenerates the exact same file.

Eni's pipeline is the fallback throughout. Every change sits behind a flag and becomes the default
only when it wins by the criterion below.

## Constraints

- **No match data to cloud LLMs** (FAQ, T&C). `.claude/settings.json` denies Claude file reads of
  `data/tactical/`, `data/broadcast/`, `data/cache/BAR-ATH*`, `evaluation/results/`,
  `evaluation/reports/`, and a PreToolUse hook blocks shell commands naming the match files.
  Scripts that touch our match print aggregate numbers only.
- **Public window**: run once at the end as a correctness check (teams not swapped, frames
  aligned, dead mask sensible). Its numbers are recorded, never used to choose a variant.
- **Selection** uses only leave-one-match-out scores on the 7 labelled training matches (3 Metrica
  sample games, 4 IDSSE), scored with `evaluation/possession_metrics.py`.

## Phase C: hygiene

1. Python 3.12 venv with `requirements-solution.txt` (numpy 2.0.2 / scipy 1.13.1 have no 3.13
   wheels). `SOLUTION.md` gains the 3.12 note and Windows commands.
2. `solution/download_external.py` pins the Metrica GitHub commit and the HuggingFace revision of
   `pysport/idsse-data` instead of `master` / `main`.
3. Reproduce Eni's held-out numbers (`python -m solution.model --evaluate`) before any change.
4. Determinism: build the submission with `OMP_NUM_THREADS=1` and with 16 threads; the SHA-256
   of both files must match.
5. Label granularity: per training source, possessions per live minute, median possession
   length, share under 1 s / 3 s, direct turnovers per minute; printed next to the public
   window's (17 possessions, 12 direct turnovers, median ~8 s in 5 min). Acted on only if a source
   is grossly off (about 3x); any label change would apply to training and held-out truth alike.

## Phase A: learned dead ball and one 3-state decode

### Components

- `solution/dead_features.py`: per-frame dead-ball evidence, mostly from players because that is
  real in the training matches: mean / p75 player speed over past and next 1 s and 3 s and its
  drop, share of near-stationary players, ball seconds-since-seen and until-seen, last-seen ball
  distance to touchlines and goal lines, last-seen ball speed, visible-but-stationary ball, players
  massed in a penalty box, own-half share (kick-off shape).
- `solution/dead_model.py`: HistGradientBoosting P(dead) trained on all graded frames of the
  labelled matches after `augment.augment`, leave-one-match-out, seeded.
- `solution/decode.py`: `viterbi3(p_dead, p_home, switch_cost, stop_cost, restart_cost)` over
  states H / A / D per half. Emissions: D = log p_dead; H = log(1 - p_dead) + log p_home;
  A = log(1 - p_dead) + log(1 - p_home). Frames before the detected kick-off are forced to D
  (Eni's rule kept as a hard constraint).
- `solution/pipeline.py`: `--dead rule|model`, default `rule` until the new path wins.

### Honest yardstick

The held-out matches' ball gaps are synthetic (`augment.py` hides the ball across 95% of
stoppages and adds 5% live gaps), which flatters any "ball missing" rule. So:

- measure aggregate ball-gap statistics on our match (share of graded frames without a ball,
  gap-length distribution, player speed inside gaps) -- unlabelled, aggregates only;
- add a `calibrated` augmentation regime whose training-match statistics match ours, plus a
  `harsh` regime with more and longer live-play gaps;
- score `rule` and `model` under `default`, `calibrated` and `harsh`.

### Promotion criterion

`--dead model` becomes the default only if, under `calibrated`, it beats `rule` on mean held-out
spellF1 and on at least 5 of 7 matches, and under `default` it is not worse by more than 0.01.
The three costs of `viterbi3` are tuned on held-out matches only.

## Testing

- Assert-based self-checks: `viterbi3` on synthetic sequences with known optimal paths (incl.
  forced pre-kick-off D), dead-feature shapes and NaN handling.
- The two-thread-count SHA-256 comparison is the reproducibility test.
- Every experiment is a `python -m solution.experiments <name>` command; results are appended to
  `docs/experiments.md` (mean spellF1, macroF1, wins of 7).

## Deliverables on `bruno`

Code, updated `SOLUTION.md`, `docs/experiments.md`, `.claude/settings.json`. The regenerated
`evaluation/results/model_viterbi.csv` is validated locally and stays gitignored.

## Out of scope

Broadcast-video synchronisation, own detection or tracking, neural sequence models, Smart
Tagging as a model input: none can be validated honestly in the time left.
