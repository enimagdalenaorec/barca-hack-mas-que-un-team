# Our solution: reproducing the submission

Everything we added lives in `solution/`; the rest of this repository is the organisers' starter
kit, unchanged. The technical document is [docs/technical_document.md](docs/technical_document.md);
every experiment behind it is in [docs/experiments.md](docs/experiments.md).

## What it does

1. **Features** (`solution/features.py`, `solution/dead_features.py`): numbers per frame, all per
   team, never per player.
2. **Models**: gradient-boosted trees (scikit-learn) give P(home has the ball)
   (`solution/model.py`) and P(ball out of play) (`solution/dead_model.py`) per frame. Trained on
   7 public matches of **other** games — 3 Metrica sample games and 4 IDSSE Bundesliga matches —
   made to look like automatic tracking first (`solution/augment.py`).
3. **Decoding** (`solution/decode.py`, `viterbi3`): one home / away / dead Viterbi per half, with
   costs for switching team, stopping and restarting, turns the per-frame probabilities into whole
   possessions and stoppages. Frames before each half's kick-off (found in the tracking: players
   set in their own halves, ball in the centre circle; `solution/dead_ball.py`) are forced dead.

Every choice is made by leave-one-game-out on 10 external matches (3 Metrica + 7 IDSSE), never on
the competition match ([docs/logo_external.md](docs/logo_external.md)): held-out spellF1 **0.655**,
against 0.645 for the same model with the "ball missing >= 2 s = dead" rule, and the learned dead
ball is ahead of that rule under every ball-gap regime tried. Switch cost 14 is the best of 6-60
and the nested choice in 10 of 10 folds. Tried and kept out: all 7 IDSSE matches for training,
controlled-possession training labels, minimum possession length, seed ensembles, noise
augmentation, kick-off variants. `--dead rule` switches back to the 2 s rule.

## Reproduce

Python 3.9 to 3.12 (numpy 2.0.2 has no 3.13 wheels), about 2 GB of free disk during the download
(~0.2 GB stays).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-solution.txt
```

On Windows create it with `py -3.12 -m venv .venv` and write `.venv/Scripts/python` wherever this
page says `.venv/bin/python` (`.venv/Scripts/python -m pip install -r requirements-solution.txt`
for the install).

Put the four tactical match files in `data/tactical/` (see [DATA.md](DATA.md)). Then convert the
training matches into compact arrays in `data/external/cache/`. Raw files already on disk are read
in place and kept: `data/external/metrica/Sample_Game_{1,2,3}/` and `data/external/idsse/`
(`DFL_02_01_matchinformation_*.xml`, `DFL_04_03_positions_raw_observed_*.xml`, original names).
Anything missing is downloaded from the pinned revisions (~1.7 GB in all), converted, and the
download deleted:

```bash
.venv/bin/python -m solution.download_external
```

Generate the submission. It trains the model on first use (seeded, so the same data always gives
the same model) and writes `evaluation/results/model_viterbi.csv`:

```bash
.venv/bin/python -m solution.pipeline
.venv/bin/python evaluation/validate_submission.py evaluation/results/model_viterbi.csv
```

## Other commands

```bash
.venv/bin/python -m solution.logo <name>             # leave-one-game-out on the 10 external matches (docs/logo_external.md)
.venv/bin/python -m solution.model --evaluate        # leave-one-match-out scores on the training matches
.venv/bin/python -m solution.experiments <name>      # earlier experiments (docs/experiments.md)
.venv/bin/python -m solution.diagnostics <csv> ...   # describe submissions without any ground truth
.venv/bin/python -m solution.timecode <frame>        # where a frame is in the video
.venv/bin/python -m solution.pipeline --method rule  # the baseline rule with our dead-ball fix
.venv/bin/python -m solution.pipeline --dead rule    # the 2 s dead-ball rule instead of the learned one
.venv/bin/python -m solution.selftest                # self-checks on synthetic data
```

The training data is pinned to fixed revisions (`solution/download_external.py`), and the
submission is byte-identical whether the model trains on 1 CPU thread or the default 8.

## Data sources

- **Competition match**: provided by the organisers; never committed (`data/` is gitignored).
- **Metrica Sports sample data** — https://github.com/metrica-sports/sample-data — 3 matches,
  tracking and events. Labels derived from Metrica's human-produced event data.
- **IDSSE** — Bassek, Rein, Weber & Memmert, *An integrated dataset of spatiotemporal and event data
  in elite soccer*, Scientific Data 12, 195 (2025), CC-BY 4.0 —
  https://huggingface.co/datasets/pysport/idsse-data — all 7 matches for validation, 4 of them
  for training, official DFL tracking with per-frame ball status and possession.
- Tried and dropped: **SkillCorner open data** (MIT) — training without it was better on every
  held-out match.

No data about the competition match other than the organisers' assets is used; no frame of it was
labelled by hand.
