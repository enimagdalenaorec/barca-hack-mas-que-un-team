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
