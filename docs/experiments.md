# Experiments on the possession model

Raw material for the technical document. Every number below is a **held-out** score: for each of
the 7 matches with full H/A/D labels (3 Metrica, 4 IDSSE), the model was trained on the other
matches and scored on that one with the official metric library (`evaluation/possession_metrics.py`),
through the whole pipeline — dead-ball rule, model, Viterbi smoothing. Nothing here uses the
competition match or its public 5 minutes.

"Wins" = on how many of the 7 matches a variant beat the reference. With 7 matches, differences
under ~0.01–0.02 spellF1 are noise unless they win on most matches.

Code: `solution/experiments.py` (`python -m solution.experiments <name>`). Training matches at the
time of experiments 1–6 included 5 SkillCorner matches; experiment 4 is why they were dropped.

## Where the points are (error budget)

| | spellF1 |
|---|---|
| our pipeline (first model) | 0.568 |
| perfect dead-ball, our team decision | 0.596 |
| our dead-ball, perfect team decision | 0.956 |

The team decision holds almost all the remaining error, so the experiments target it.

## 1. Timing of team switches
Matched switches: median latency 0 frames (mean +6, quartiles −5 / +23). Shifting every switch
earlier by 5–35 frames lowered spellF1 every time. **No systematic delay; no change.**

## 2. Smoothing variants (reference: one switch cost of 6)
Separate costs with/without a visible ball, minimum possession length 0.5/1/2 s: all within
±0.001 of the reference, except 2 s minimum (−0.026). **Smoothing is not the bottleneck; no change.**

## 3. Feature groups (51 features)
| variant | spellF1 | wins vs all 51 |
|---|---|---|
| all 51 features | 0.578 | ref |
| original 39 features | 0.568 | 1/7 |
| without ball flight (new) | 0.570 | 1/7 |
| without around-the-ball | 0.564 | 0/7 |
| without team shape | 0.567 | 3/7 |
| without control / long context / 1–3 s windows / visibility | 0.577–0.580 | 3–5/7 |

**Kept all 51.** The useful new group is ball flight: distance from each team to where the ball is
heading in 0.4 s and 1 s. Control and long-context features are neutral (redundant with others).

## 4. Training mix
| variant | spellF1 | wins | Metrica | IDSSE |
|---|---|---|---|---|
| all 12 matches | 0.578 | ref | 0.686 | 0.496 |
| **without SkillCorner** | **0.626** | **7/7** | 0.720 | 0.555 |
| SkillCorner at half weight | 0.593 | 6/7 | 0.693 | 0.518 |
| without Metrica | 0.552 | 0/7 | 0.657 | 0.473 |
| without IDSSE | 0.547 | 0/7 | 0.651 | 0.469 |

**Dropped SkillCorner.** Its labels come from SkillCorner's own software and its tracking from a
10 fps broadcast feed. Metrica and IDSSE each help the other.

Learning curve without SkillCorner: 2 matches 0.608, 4 matches 0.625, 6 matches 0.626 — flat, so
the 3 IDSSE matches we did not download were not needed.

## 5. Models (all 12 matches, cost 6)
| model | spellF1 | wins |
|---|---|---|
| boosting 300 trees / 31 leaves / lr 0.05 | 0.578 | ref |
| boosting 150 / 15 | 0.554 | 0/7 |
| boosting 600 / 63 | 0.592 | 6/7 |
| boosting 300 / 31 / lr 0.1 | 0.585 | 7/7 |
| logistic regression | 0.514 | 0/7 |
| random forest, 200 trees | 0.557 | 1/7 |

Gradient boosting is clearly the right family; a larger model helps.

## 6. Augmentation sensitivity (training side; held-out kept at the reference)
No live-play gaps 0.580, 3× live gaps 0.572, hide 80% / 100% of stoppages 0.579 / 0.578, no dropped
players 0.570, 10% dropped 0.568, no augmentation at all 0.577 (reference 0.578). **Insensitive:
our guesses about how the competition tracking loses the ball do not drive the result.**

## 7. Stacking (second model over the first model's guesses, nested leave-one-out)
First stage 0.626; stacked 0.618–0.625 at costs 2–10. **No gain; dropped.**

## Final configuration
No SkillCorner; boosting 600 trees / 63 leaves / lr 0.1; all 51 features; switch cost 10.

| switch cost | 2 | 4 | 6 | 8 | **10** | 14 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| held-out spellF1 | 0.592 | 0.630 | 0.642 | 0.643 | **0.645** | 0.648 | 0.641 | 0.625 |

**Held-out spellF1 0.645, macroF1 0.904** — against 0.449 / 0.851 for the published baseline's rule
on the same matches, and 0.568 for the first model. Better than the first model on 7 of 7 matches.

# Phase 4: robustness on the competition match, and more ideas

## Checks on the competition match (no ground truth used)
- **Kick-off bug found and fixed.** FAQ.md: the 2,934 graded frames before the second-half kick-off
  are all dead. Only 18% of them were `D`: the ball is tracked while it is placed on the centre
  spot, so "ball missing ≥ 2 s" called them live. New rule in `solution/dead_ball.py`: nothing is
  live before a half's kick-off, which is detected in the tracking (≥ 85% of players in their own
  half for 3 s, then the formation breaks with the ball inside the centre circle, and nobody
  sprinted before it). Detected at frame 113,750 against the documented 113,742; on the 14
  training halves it fires only before one retaken kick-off. Two refinements came from failures
  on the training matches: the centre-circle condition, and the "calm before" condition, which
  keeps a kick-off after an early goal from turning 2.5 minutes of real play dead.
- **Stability.** Switch cost 6 or 14 instead of 10 changes 0.3% / 0.2% of graded frames;
  dead-ball threshold 1–3 s instead of 2 s changes none.
- **Tracking duplicates** (more than 22 tracked objects on 1.6% of live frames) do not cause extra
  team switches (1.3 per 1000 frames there vs 2.6 elsewhere).
- **Feature drift**, live frames, competition match vs augmented training matches: every feature
  within a third of the training interquartile range of its median, except opponents pressing
  the nearest player (median 0 vs 1) — a feature group that was neutral in experiment 3.
- **Hypothesis tested and rejected:** that the tracker puts goalkeepers (different shirt colour) in
  the wrong team. Both keepers' slots are labelled correctly and spend >80% of the time in their
  own penalty box.

## 8. Restart features (the laws decide who restarts)
Nine per-live-stretch features: who was closer to the ball in the last second before the stoppage,
where the ball was last seen then, stoppage length, restart position, who was closer in the first
second, time since the restart and until the next stoppage.

| variant | spellF1 | wins | first possession after a stoppage right |
|---|---|---|---|
| without | 0.645 | ref | 96.8% of 761 |
| with | 0.647 | 4/7 | 96.8% of 761 |

**No gain — dropped.** The restarting team is already right 97% of the time from who is nearest.

## 9. Ensembles and source weights (with restart features, reference 0.647)
Ensemble of 5 models, one per interleaved fifth of the frames: 0.646 (3/7). IDSSE counted double:
0.638 (1/7). Metrica counted double: 0.645 (3/7). **No gain.**

## 10. Self-training on the unlabelled held-out match (simulating our match)
Retrain twice on the model's own confident, smoothed guesses about the held-out match:

| confidence / weight | 0.7 / 0.5 | 0.7 / 1.0 | 0.85 / 0.5 | 0.85 / 1.0 |
|---|---|---|---|---|
| spellF1 (ref 0.647) | 0.641 | 0.643 | 0.635 | 0.635 |

**Worse every time — dropped.** The model reinforces its own mistakes.

## 11. Smart Tagging hints at restarts — not run
Restart team is already 97% right on the training matches, and only the competition match has
Smart Tagging, so the idea could not be tested honestly. Left out.

# Phase 5 (bruno): reproducibility, learned dead ball

Eni's held-out numbers reproduce exactly on Python 3.12 with the pinned training data (cost 10:
0.645, baseline 0.449).

## 12. Label granularity (`python -m solution.experiments labels`)
Do the training labels cut possessions as finely as the public ground truth? Raw truth, runs as
the metric builds them (non-X frames):

| source | poss / live min | median s | < 1 s | < 3 s | direct H<->A / live min |
|---|---|---|---|---|---|
| Metrica (mean of 3) | 5.92 | 5.4 | 0.08 | 0.32 | 4.29 |
| IDSSE (mean of 4) | 9.32 | 3.4 | 0.13 | 0.45 | 7.21 |
| public 5 min (17 possessions) | 4.23 | 7.6 | 0.06 | 0.18 | 2.98 |

No source is 3x off (the stop threshold; worst single match: IDSSE J03WN1 at 2.9x on direct
turnovers), so the labels stay as they are. But IDSSE cuts possessions about 2.3x finer than the
public window, Metrica 1.4x. The held-out yardstick therefore rewards more switches than our
ground truth probably does: a switch cost tuned on it leans low. The public window is 5 minutes,
so this is a direction, not a number to tune to.

## 13a. Our match's ball gaps, aggregates only (`python -m solution.gap_stats`)
How much of our long ball-missing time is live play? Mean player speed is the tell: 1.09 m/s on
dead frames and 2.31 m/s on live frames of the 7 labelled matches, 1.14 m/s inside our gaps.

| our match, graded frames | value |
|---|---|
| share inside ball gaps >= 2 s | 0.406 (133 gaps) |
| gaps at live pace (mean speed above 1.70 m/s) | 26 |
| estimated live share of gap frames, f | 0.04 |

So nearly all of our long gaps are stoppages, and the `calibrated` augmentation regime is milder
than Eni's default: `LIVE_GAP_SHARE` 0.028 (default 0.05), `LIVE_GAP_S` (6.1, 9.2) s (default
2-15). `harsh` (0.15, 3-30 s, 85% of stoppages hidden) is the stress test.

Checked on the training matches as the held-out side sees them
(`python -m solution.experiments calibration`): the share of live frames inside ball gaps >= 2 s,
which is 0.028 on ours.

| regime | IDSSE (4) | Metrica (3) |
|---|---|---|
| default | 0.041-0.049 | 0.040-0.052 |
| calibrated | 0.026-0.029 | 0.023-0.052 |

IDSSE hits the target. Metrica's own 1-2 s tracker gaps already use up much of the budget, so
metrica2 gets no injected gaps and stays harsher than our match (0.052). Under `calibrated` the
held-out matches have as much live play hidden in long gaps as ours or more: the yardstick does not
flatter the 2 s rule.

## 13. Learned dead ball vs the 2 s rule (`python -m solution.experiments dead`)
`solution/dead_model.py` learns P(dead) from 16 mostly player-movement features
(`solution/dead_features.py`); `decode.viterbi3` decodes home / away / dead in one pass per half,
frames before the detected kick-off forced dead. Training matches augmented under regime T, the
held-out match under regime E (default, calibrated from 13a, harsh). Stop / restart costs chosen
on held-out under E = calibrated, as Eni's switch cost was; the grid is flat (0.657-0.659), best
(20, 8) for T default.

| held-out E | Eni: T default, rule | T calibrated, rule | T default, model (20, 8) | T calibrated, model (20, 20) |
|---|---|---|---|---|
| default | 0.645 / 0.904 | 0.641 / 0.903 (2 of 7) | 0.651 / 0.915 (4 of 7) | 0.647 / 0.912 (4 of 7) |
| calibrated | 0.656 / 0.910 | 0.650 / 0.909 (2 of 7) | 0.659 / 0.917 (4 of 7) | 0.655 / 0.917 (2 of 7) |
| harsh | 0.605 / 0.840 | 0.598 / 0.840 (1 of 7) | 0.622 / 0.880 (5 of 7) | 0.612 / 0.868 (4 of 7) |

spellF1 / macroF1, wins of 7 against Eni in brackets.

**Kept the rule.** The promotion rule set before the run (under calibrated: better mean and at
least 5 of 7 wins; under default: not worse by more than 0.01) fails on wins: the model beats the
rule on all 4 IDSSE matches and loses on all 3 Metrica ones (by 0.01-0.015 at costs 8 / 8). On
Metrica the ball tracker's own gaps already mark the stoppages almost perfectly (dead F1
0.94-0.97, solution/dead_ball.py), and 13a says our match behaves the same way: nearly all of its
long gaps are at stoppage pace. The model's advantage grows when live-play gaps are common
(harsh: +0.017, 5 of 7) and it lifts macroF1 by ~0.01 everywhere, so it stays available as
`python -m solution.pipeline --dead model`.

Training the team model under the calibrated regime instead of the default is slightly worse
everywhere (2 of 7): `AUG_REGIME` stays "default".

## Phase 5 summary
- Reproducible: training data pinned to fixed revisions; the submission built from scratch with
  `OMP_NUM_THREADS=1` and with the default thread count (8 here) is byte-identical (same SHA-256); format validated.
- Final configuration: Eni's model + 2 s dead-ball rule + Viterbi; switch cost 14 since
  experiment 14 (was 10; the public-window row below was run at 10).
  The learned dead ball (13) is available behind `--dead model`; it did not meet the bar.
- Public 5 minutes, run once as a correctness check after every choice above was made (not
  used to choose anything):

| submission | spellF1 | macroF1 | alive/dead F1 | team acc (both alive) | possessions (truth 17) |
|---|---|---|---|---|---|
| model_viterbi | 0.506 | 0.682 | 0.893 | 0.920 | 17 |
| baseline_smooth50 | 0.326 | 0.621 | 0.899 | 0.858 | 26 |

  Teams are not swapped and frames line up (team accuracy 0.92, possession count exact).
  Alive/dead F1 sits 0.006 under the smoothed baseline: within noise on 1,469 dead frames,
  not a bug signature.

## 14. Switch cost chosen on Metrica only (`python -m solution.experiments switch`)
Experiment 12 found IDSSE's labels cut possessions ~2.3x finer than the public ground truth and
Metrica's ~1.4x, so a switch cost tuned on all 7 held-out matches leans towards too many
switches. Metrica is the closer proxy for our labels.

Rule, written and committed before the run:
- Grid: switch cost 6, 8, 10, 14, 20, 30, 40, 60, 80; everything else as the final configuration.
- c_M = the cost with the best mean held-out spellF1 over the 3 Metrica matches.
- Adopt c_M as `decode.SWITCH_COST` only if all three hold: (a) c_M is not 10; (b) c_M beats
  cost 10 on at least 2 of the 3 Metrica matches; (c) the mean over all 7 matches is not worse
  than cost 10's by more than 0.01 (a guard against a cost that fits only three matches).
- Predicted possessions per live minute on Metrica are printed next to the truth as a sanity
  check, not a criterion. The public window plays no part.

Result (held-out spellF1; possessions per live minute on Metrica, truth 5.92):

| switch cost | Metrica (3) | wins vs 10 | all 7 | IDSSE (4) | Metrica poss/min |
|---|---|---|---|---|---|
| 6 | 0.730 | 1 | 0.642 | 0.575 | 5.66 |
| 8 | 0.732 | 0 | 0.643 | 0.575 | 5.51 |
| 10 | 0.739 | - | 0.645 | 0.575 | 5.34 |
| **14** | **0.744** | **3** | **0.647** | 0.575 | 5.12 |
| 20 | 0.736 | 1 | 0.640 | 0.569 | 4.79 |
| 30 | 0.723 | 1 | 0.624 | 0.551 | 4.54 |
| 40 | 0.712 | 0 | 0.605 | 0.525 | 4.37 |
| 60 | 0.689 | 0 | 0.583 | 0.503 | 4.12 |
| 80 | 0.676 | 0 | 0.564 | 0.481 | 3.99 |

c_M = 14: not 10, 3 of 3 wins, all-7 mean up 0.002 -> **adopted** (`decode.SWITCH_COST = 14`).
A small, consistent gain (+0.005 on Metrica), not a big one. The premise held only in part: IDSSE
is flat from 6 to 14 rather than pulling the cost down, and 14 was already the top of the all-7
curve, where 10 had been picked as the plateau's middle. The public window was not rerun.

## 15. Bigger team model (`python -m solution.experiments bigger`)
Experiment 5 found bigger boosting better, and 600 trees / 63 leaves is the largest tried.

Rule, written and committed before the run:
- Variants (max_iter, max_leaf_nodes): (1000, 63), (600, 127), (1000, 127); every other GBM
  setting unchanged; decoded at the current switch cost (14); original training labels.
- Adopt the variant with the best mean held-out spellF1 only if it beats the current (600, 63)
  on the mean and on at least 5 of the 7 matches. Otherwise keep (600, 63).

## 16. Training labels relabelled to "controlled" possession (`python -m solution.experiments control`)
CHALLENGE.md: possession changes only when the other team gains *controlled* possession; a touch
without control, a deflection, or an aerial duel without subsequent control is no change, and in
contested or loose phases possession stays with the last team that had it. Our training labels
switch on any touch (experiment 12: 1.4-2.3x more possessions than the public ground truth).

Rule, written and committed before the run:
- Relabelling: inside each live stretch, possession passes to the other team only when that
  team holds the ball for at least T seconds in a row; its shorter spells go to the team that
  had the ball. The first spell after a restart stands (the restart has a clear possessor).
- T = 2 s, fixed from CHALLENGE.md's proxy "a player keeps the ball for a few seconds".
  T = 1 s and 3 s are reported for robustness only, never chosen.
- Applied to every training match, so training rows and held-out truth both change. The switch
  cost is re-chosen against the relabelled held-out truth: best mean over all 7 matches, grid
  2, 4, 6, 8, 10, 14, 20, 30, 40.
- Held-out scores cannot compare label definitions (each is scored against its own), so the
  decision is one public-window run, after experiment 15 is settled, with two files built from
  the same team-model settings: the current configuration and the T = 2 one. Adopt T = 2 only
  if its public spellF1 is at least 0.03 higher (beyond the noise of 17 possessions); otherwise
  keep the current file. Nothing else is scored on the public window afterwards.

## 15. Result: bigger team model -- kept 600 / 63

| variant | spellF1 | macroF1 | wins vs 600 / 63 | Metrica | IDSSE |
|---|---|---|---|---|---|
| 600 trees / 63 leaves (current) | 0.647 | 0.904 | ref | 0.744 | 0.575 |
| 1000 / 63 | 0.646 | 0.904 | 3/7 | 0.738 | 0.577 |
| 600 / 127 | 0.646 | 0.904 | 4/7 | 0.734 | 0.580 |
| 1000 / 127 | 0.648 | 0.904 | 4/7 | 0.729 | 0.587 |

Best mean 1000 / 127, but 4 of 7 wins (5 needed): **kept**. Model size is saturated.

## 16. Result: controlled-possession labels -- kept the original labels
Relabelled training truth (held-out, scored against its own relabelling; comparable within a T
only):

| T | Metrica poss / live min | median s | direct / live min | best switch cost | held-out spellF1 |
|---|---|---|---|---|---|
| 1 s | 5.32 | 6.6 | 3.69 | 14 | 0.690 |
| **2 s** | **4.48** | **8.9** | **2.84** | **30** | **0.722** |
| 3 s | 3.87 | 10.9 | 2.24 | 40 | 0.750 |
| public window | 4.23 | 7.6 | 2.98 | | |

T = 2 s brings Metrica's labels to the public ground truth's granularity, as the written
definition suggested. The decision run on the public window, once, both files from the same
600 / 63 model:

| public 5 min | spellF1 | macroF1 | possessions (truth 17) | mean s (truth 14.2) |
|---|---|---|---|---|
| current (original labels, cost 14) | 0.533 | 0.685 | 16 | 16.6 |
| T = 2 s, cost 30 | 0.452 | 0.684 | 14 | 18.9 |

**Not adopted** (needed +0.03, got -0.08). The premise failed on our match: although the training
labels switch more often than the public ground truth, the model's predictions on our match
already have about the right number of possessions (16 vs 17), and relabelling made them too long.
`model.CONTROL_S` stays None; the candidate file was deleted and nothing else is scored on the
public window.
