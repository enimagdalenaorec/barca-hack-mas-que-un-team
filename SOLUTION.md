# Our solution: reproducing the submission

Everything we added lives in `solution/`; the rest of this repository is the organisers' starter
kit, unchanged. The approach and every experiment behind it are in
[docs/experiments.md](docs/experiments.md).

## What it does

1. **Dead ball** (`solution/dead_ball.py`): a frame is dead when the tracker has lost the ball for
   at least 2 s, and before each half's kick-off (found in the tracking: players set in their own
   halves, ball in the centre circle).
2. **Features** (`solution/features.py`): 51 numbers per frame, all per team, never per player.
3. **Model** (`solution/model.py`): gradient-boosted trees (scikit-learn) give P(home has the ball)
   on live frames. Trained on 7 public matches of **other** games — 3 Metrica sample games and 4
   IDSSE Bundesliga matches — made to look like our automatic tracking first
   (`solution/augment.py`).
4. **Smoothing** (`solution/decode.py`): Viterbi with a cost for switching teams turns the per-frame
   probabilities into whole possessions.

Held-out score on matches the model never saw: spellF1 **0.645** against **0.449** for the
published baseline's rule.

Tried and kept as an option, not the default: a learned dead-ball model (`solution/dead_model.py`)
decoded together with the team in one home / away / dead Viterbi. It wins on the 4 IDSSE matches
and loses on the 3 Metrica ones, so it missed the promotion bar set before the run
([docs/experiments.md](docs/experiments.md), 13). `--dead model` switches it on.

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

Put the four tactical match files in `data/tactical/` (see [DATA.md](DATA.md)). Then fetch and
convert the training matches — ~1.7 GB is downloaded, converted to compact arrays, and the raw
files are deleted:

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
.venv/bin/python -m solution.model --evaluate        # leave-one-match-out scores on the training matches
.venv/bin/python -m solution.experiments <name>      # rerun an experiment from docs/experiments.md
.venv/bin/python -m solution.diagnostics <csv> ...   # describe submissions without any ground truth
.venv/bin/python -m solution.timecode <frame>        # where a frame is in the video
.venv/bin/python -m solution.pipeline --method rule  # the baseline rule with our dead-ball fix
.venv/bin/python -m solution.pipeline --dead model   # learned dead ball instead of the 2 s rule
.venv/bin/python -m solution.selftest                # self-checks on synthetic data
.venv/bin/python -m solution.gap_stats               # aggregate ball-gap statistics of our match
```

The training data is pinned to fixed revisions (`solution/download_external.py`), and the
submission is byte-identical whether the model trains on 1 CPU thread or 16.

## Data sources

- **Competition match**: provided by the organisers; never committed (`data/` is gitignored).
- **Metrica Sports sample data** — https://github.com/metrica-sports/sample-data — 3 matches,
  tracking and events. Labels derived from Metrica's human-produced event data.
- **IDSSE** — Bassek, Rein, Weber & Memmert, *An integrated dataset of spatiotemporal and event data
  in elite soccer*, Scientific Data 12, 195 (2025), CC-BY 4.0 —
  https://huggingface.co/datasets/pysport/idsse-data — 4 of 7 matches, official DFL tracking with
  per-frame ball status and possession.
- Tried and dropped: **SkillCorner open data** (MIT) — training without it was better on every
  held-out match.

No data about the competition match other than the organisers' assets is used; no frame of it was
labelled by hand.
