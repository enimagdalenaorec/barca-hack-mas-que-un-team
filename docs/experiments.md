# Experiments on the possession model

Raw material for the technical document. Every number below is a **held-out** score: for each of
the 7 matches with full H/A/D labels (3 Metrica, 4 IDSSE), the model was trained on the other
matches and scored on that one with the official metric library (`evaluation/possession_metrics.py`),
through the whole pipeline — dead-ball rule, model, Viterbi smoothing.

Experiments 1-15 use the first yardstick (7 held-out matches). The submission is decided by
Phase 6 at the end: leave one game out on 10 external matches.

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

# Phase 5 (bruno): reproducibility, bigger model

Eni's held-out numbers reproduce exactly on Python 3.12 with the pinned training data (cost 10:
0.645, baseline 0.449).

## Phase 5 summary
- Reproducible: training data pinned to fixed revisions; the submission built from scratch with
  `OMP_NUM_THREADS=1` and with the default thread count (8 here) is byte-identical (same SHA-256); format validated.
- Configuration at the end of Phase 5: Eni's model + 2 s dead-ball rule + Viterbi, switch cost 14.
- Public 5 minutes, run once as a correctness check of that configuration:

| submission | spellF1 | macroF1 | alive/dead F1 | team acc (both alive) | possessions (truth 17) |
|---|---|---|---|---|---|
| model_viterbi | 0.506 | 0.682 | 0.893 | 0.920 | 17 |
| baseline_smooth50 | 0.326 | 0.621 | 0.899 | 0.858 | 26 |

  Teams are not swapped and frames line up (team accuracy 0.92, possession count exact).
  Alive/dead F1 sits 0.006 under the smoothed baseline: within noise on 1,469 dead frames,
  not a bug signature.

## 15. Bigger team model (`python -m solution.experiments bigger`)
Experiment 5 found bigger boosting better, and 600 trees / 63 leaves is the largest tried.

Rule, written and committed before the run:
- Variants (max_iter, max_leaf_nodes): (1000, 63), (600, 127), (1000, 127); every other GBM
  setting unchanged; decoded at the current switch cost (14); original training labels.
- Adopt the variant with the best mean held-out spellF1 only if it beats the current (600, 63)
  on the mean and on at least 5 of the 7 matches. Otherwise keep (600, 63).

## 15. Result: bigger team model -- kept 600 / 63

| variant | spellF1 | macroF1 | wins vs 600 / 63 | Metrica | IDSSE |
|---|---|---|---|---|---|
| 600 trees / 63 leaves (current) | 0.647 | 0.904 | ref | 0.744 | 0.575 |
| 1000 / 63 | 0.646 | 0.904 | 3/7 | 0.738 | 0.577 |
| 600 / 127 | 0.646 | 0.904 | 4/7 | 0.734 | 0.580 |
| 1000 / 127 | 0.648 | 0.904 | 4/7 | 0.729 | 0.587 |

Best mean 1000 / 127, but 4 of 7 wins (5 needed): **kept**. Model size is saturated.

# Phase 6 (karlo): final choices on 10 external matches

Harnesses `solution/logo.py` (round 1) and `solution/logo2.py` (round 2); full tables and the
adoption rules, written before each round, in [logo_external.md](logo_external.md). Leave one game
out over 3 Metrica + 7 IDSSE matches; nothing reads the competition match or its public window.

- **Round 1** (one held-out seed; reference: the Phase 5 configuration, 0.645). Learned dead ball
  (`solution/dead_model.py` + `decode.viterbi3`, STOP 20 / RESTART 8) adopted: +0.010 (7/10)
  under the default regime, +0.021 / +0.007 / +0.044 under the `harsh`, `real` and `lag`
  regimes. Switch cost 14 confirmed by the external grid (nested choice 14 in 10 of 10 folds).
  Rejected: 7 IDSSE matches for the team model, controlled-possession labels, minimum
  possession length, seed and subsample ensembles, training with position noise or harsher
  regimes, kick-off variants.
- **Round 2** (two held-out seeds; reference: round 1's result, 0.6526). Adopted together:
  switch cost x0.4 within 10 frames of a "flip" (nearest team changes, new nearest player within
  2 m) and 20 elsewhere; dead model trained on all 7 IDSSE matches; temperature 2 on P(dead).
  Nested estimate +0.0097, 10/10 matches, `harsh` +0.011, `lag` +0.024, macroF1 0.915 -> 0.922.
  Rejected: joint cost grid, other evidence-dependent costs, temperature on P(home), second-stage
  model, posterior decoding, other dead-model GBM settings.

Same harness, all versions (spellF1 over 10 matches x 2 seeds):

| version | default | Metrica | IDSSE | macroF1 | harsh | lag |
|---|---|---|---|---|---|---|
| published baseline rule (+/-50 vote) | 0.347 | | | 0.799 | 0.328 | 0.343 |
| Phase 4 (2 s rule, switch cost 10) | 0.644 | 0.745 | 0.600 | 0.901 | 0.596 | 0.580 |
| Phase 5 (2 s rule, switch cost 14) | 0.644 | 0.749 | 0.599 | 0.901 | 0.592 | 0.581 |
| round 1 (learned dead ball) | 0.653 | 0.740 | 0.615 | 0.915 | 0.616 | 0.624 |
| **round 2, submitted** (nested estimate 0.662) | **0.664** | 0.751 | 0.627 | 0.922 | 0.628 | 0.650 |
