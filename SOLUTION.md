# Our solution: reproducing the submission

Everything we added lives in `solution/`; the rest of this repository is the organisers' starter
kit, unchanged. The technical document is [docs/technical_document.md](docs/technical_document.md);
every experiment behind it is in [docs/experiments.md](docs/experiments.md).

## What it does

1. **Features** (`solution/features.py`, `solution/dead_features.py`): numbers per frame, all per
   team, never per player.
2. **Models**: gradient-boosted trees (scikit-learn) give P(home has the ball)
   (`solution/model.py`) and P(ball out of play) (`solution/dead_model.py`) per frame. Trained on
   public matches of **other** games — 3 Metrica sample games and 4 (team model) or 7 (dead model)
   IDSSE Bundesliga matches —
   made to look like automatic tracking first (`solution/augment.py`).
3. **Decoding** (`solution/decode.py`, `viterbi3`): one home / away / dead Viterbi per half, with
   costs for switching team, stopping and restarting, turns the per-frame probabilities into whole
   possessions and stoppages. Frames before each half's kick-off (found in the tracking: players
   set in their own halves, ball in the centre circle; `solution/dead_ball.py`) are forced dead.

Every choice is made by leave-one-game-out on 10 external matches (3 Metrica + 7 IDSSE), never on
the competition match ([docs/logo_external.md](docs/logo_external.md)). The learned dead ball beats
the "ball missing >= 2 s = dead" rule under every ball-gap regime tried. Round 2 added, together:
a lower switch cost (x0.4) within 10 frames of a "flip" (the team nearest the ball changes and the
new nearest player is within 2 m), switch cost 20 elsewhere, P(dead) trained on all 7 IDSSE
matches (the team model stays on 4) and temperature 2 on P(dead). Nested held-out spellF1 rises
by 0.0097, on 10 of 10 matches, and also under the harsher ball-gap regimes. Tried and kept out:
all 7 IDSSE matches for the team model, controlled-possession training labels, minimum possession
length, seed ensembles, noise augmentation, kick-off variants, a second-stage model, posterior
decoding. `--dead rule` switches back to the 2 s rule.

Round 3 added **touch features** (`features.TOUCH_FEATURES`): touches found as sudden changes of
ball velocity, credited to the clearly nearest team, plus "control" of a slow ball; who touched
last / next, time since / until each team's touch and control, touch counts. The competition
tracker smooths its ball, so a detector for sharp kicks saw almost no touches there
(`solution.touch_diag`); the submitted detector works on a ball smoothed over 1 s on every match.
Held-out spellF1 0.664 -> **0.703** (macroF1 0.922 -> 0.931), on 8 of 10 matches, and +0.038 /
+0.037 under the harsh / lag regimes (`python -m solution.touch_exp heavy`; docs/logo_external.md,
"Krug 3"). `python -m solution.touch_check` (run by a person: it reads the competition match)
compares the touch rate there (14.7 per minute) with the external matches (15.9–22.5).

Round 4 trains on the **competition's kind of ball** (`augment.TRACKER_SMOOTH_W = 21`, a trailing
average set from aggregate statistics of that ball). On held-out matches with that smoothed ball,
spellF1 0.515 -> **0.627** (with touches), 10 of 10 matches; on clean held-out balls it is
0.664 -> 0.584, because the model is now trained for the competition ball
(`python -m solution.smooth_train_exp`; docs/logo_external.md, "Krug 4").

Round 5 makes **every possession count the same in training** (`model.SPELL_WEIGHTS`: each frame
weighs clip(8 s / its possession's length, 0.25, 4)): tracker-like held-out ball 0.627 -> **0.649**
(9 of 10), clean ball 0.584 -> 0.605 (10 of 10), more of the short possessions recovered
(`python -m solution.smooth_train_exp weights`; docs/logo_external.md, "Krug 5").

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
.venv/bin/python -m solution.logo2 <name>            # round 2 of it: flip cost, temperature, dead model on 7 IDSSE
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
  https://huggingface.co/datasets/pysport/idsse-data — all 7 matches for validation and the dead-ball
  model, 4 of them for the team model, official DFL tracking with per-frame ball status and possession.
- Tried and dropped: **SkillCorner open data** (MIT) — training without it was better on every
  held-out match.

No data about the competition match other than the organisers' assets is used; no frame of it was
labelled by hand.
