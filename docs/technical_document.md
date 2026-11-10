# Frame-level ball possession: technical document

More than a Hack 2027, Phase 1. FC Barcelona vs Athletic Club: 199,319 frames, each labelled H, A,
D or X.

**In one paragraph.** The pipeline makes two decisions, one after the other. First, a frame is
dead when the automatic tracking has lost the ball for at least 2 s, or when the half has not yet
kicked off. Second, on live frames, a gradient-boosted tree model trained on 7 matches from other
games gives P(home has the ball) from 51 per-team features. A Viterbi pass with a switching cost
then turns those per-frame probabilities into whole possessions. Every choice was made by scoring
the full pipeline on matches it had not seen, using the official metric. On those 7 held-out
matches it reaches spellF1 **0.647** (macroF1 0.904), against 0.449 (0.851) for the published
baseline rule. The code is in `solution/`. Every experiment, with the rule set before it was run,
is in [experiments.md](experiments.md).

## 1. Approach, and why

**Two decisions in this order: alive or dead, then which team.** The published baseline smooths
everything at once, so every stoppage shrinks at both ends. Any stoppage shorter than twice the
window disappears. Deciding dead first, then smoothing only inside each live stretch, keeps
stoppages where the tracking puts them. It also means possession is never carried across a
stoppage.

**Where the points are.** Before tuning anything, we measured the error budget on the held-out
matches by swapping in perfect answers:

| first pipeline | perfect dead ball | perfect team decision |
|---|---|---|
| 0.568 | 0.596 | 0.956 |

The team decision held almost all the remaining error, so most of the work went there.

**Dead ball is a rule, not a model.** On Metrica's sample games, where events give the true
stoppages, the tracked ball is missing on 86–99% of dead frames and on only 2–6% of live ones.
Each gap is almost entirely one or the other (466 of 476). A minimum gap length of 2 s gives dead
F1 0.90 / 0.94 / 0.97 on the three games; 1 s or 3 s makes almost no difference. Our match behaves
the same way, and we checked this without ground truth:

- 133 gaps of at least 2 s cover 41% of graded frames.
- Players move at half speed inside them (1.2 against 2.2 m/s outside them).
- Only 26 gaps move at live pace, so we estimate about 4% of gap frames are live.

A second rule fixes a bug the first one had. Nothing is live before a half's kick-off. The ball is
tracked while it sits on the centre spot, so "ball missing" alone called 82% of the frames before
the second-half kick-off live. The kick-off is found in the tracking, not copied from the docs:
the players' set formation breaks with the ball inside the centre circle. It is detected at frame
113,750 against the documented 113,742.

We also built a learned dead-ball model (16 player-movement features, decoded jointly with the
team in a home / away / dead Viterbi). It beat the rule on all 4 IDSSE matches and lost on all 3
Metrica ones, so it missed the promotion bar we had set before the run. It is kept as an option
(`--dead model`).

**Which team: gradient boosting on per-team features.** The tracking has no player identities: 46
track slots for 22 players, swapping as the tracker loses people. So every feature describes a
team, never a player. Each feature is computed from one team's side ("us"), with the pitch flipped
so that we attack towards +x. Computing them again with the teams swapped doubles the training
set and removes any home/away bias; at prediction time both views are averaged. The 51 features
fall into seven groups:

- **Around the ball:** each team's nearest and second-nearest player to the ball; how many of
  each team are within 3 m and 7 m; ball speed; ball position.
- **Over time:** the same "who is closer" evidence over the last and next 1 s and 3 s.
- **Ball visibility:** seconds since the ball was last seen and until it is next seen.
- **Team shape:** centroid, spread, speed and forward velocity. The team in possession spreads out
  and pushes up, and this still works while the ball is missing.
- **Ball flight:** each team's distance to where the ball will be in 0.4 s and 1 s, which points
  to the intended receiver of a pass.
- **Control:** how closely the ball moves with each team's nearest player, and how many opponents
  press that player.
- **Long context:** "who is closer" over 5 s and 10 s.

Missing values stay missing, and the trees handle them natively.

We used scikit-learn's `HistGradientBoostingClassifier` with 600 trees, 63 leaves, learning rate
0.1, at least 200 samples per leaf, L2 regularisation 1.0, and a fixed seed. Other families scored
lower on the same held-out matches: logistic regression 0.514, random forest 0.557, against 0.578
for the boosting reference at the time. Bigger boosting (1000 trees, 127 leaves) did not win
consistently (4 of 7 matches), so model size is saturated.

**Making clean training data look like ours.** The training matches have clean, often corrected
tracking. Ours is automatic: the ball is missing on 41% of graded frames, in long blocks, and
about 20.5 of 22 players are tracked per frame. Before computing features, each training match is
changed, with a fixed seed:

- short ball gaps (under 1 s) are bridged by interpolation, as our tracker appears to do;
- the ball is hidden across 95% of stoppages;
- long live-play gaps are cut until 5% of live frames have no ball;
- players are dropped in random blocks covering 3% of frames.

The result is insensitive to these settings: every variant scored within ±0.01 of the reference.
That tells us our guesses about the tracker do not drive the score.

**Viterbi smoothing.** Inside each live stretch there are two states, home and away. Each frame
scores log P(state), and changing team costs 14 in the same units. This is a two-state HMM. The
cost was chosen on the 3 Metrica held-out matches only, by a rule committed before the run.
Metrica's labels are the closest to the organisers' in how finely they cut possessions; see
section 4.

**How choices were made.** We used leave-one-match-out over the 7 matches with full H/A/D
labels. For each match, we trained on the other six, then ran the whole pipeline on it (dead-ball
rule on the made-to-look-automatic tracking, model, Viterbi) and scored it with the organisers'
`possession_metrics.py`. With 7 matches, differences under about 0.01–0.02 are noise unless they
win on most matches. So from experiment 12 onwards, each experiment had a written adoption rule
(mean and number of wins), committed before it was run.

| held-out, 7 matches | spellF1 | macroF1 | Metrica (3) | IDSSE (4) |
|---|---|---|---|---|
| **ours** | **0.647** | **0.904** | 0.744 | 0.575 |
| published baseline rule (nearest player, ±50-frame vote) | 0.449 | 0.851 | | |

**Tried and dropped** (details and numbers in [experiments.md](experiments.md)):

- SkillCorner as training data (section 3).
- Shifting switch timing; more smoothing variants (minimum possession length, separate costs with
  and without a visible ball).
- Restart features built from the laws of the game. The team that restarts is already right 97%
  of the time from proximity.
- Ensembles and per-source weights.
- Self-training on the unlabelled held-out match. It got worse every time, because the model
  reinforced its own mistakes.
- Stacking a second model.
- Training under augmentation calibrated to our match's gaps.
- A bigger model.
- Relabelling training data to "controlled" possession (section 4).

## 2. Assets we used, and how

| asset | used? | how |
|---|---|---|
| Tactical ATD (tracking) | **yes, the only input** | Ball and player positions in metres; team of each slot. The exported ball speed is unusable, so speed is computed from positions. |
| Smart Tagging | checks only | Without ground truth, a sanity check: 45 of 59 set pieces are taken just as a ball gap ends, and a consistency report (`solution/diagnostics.py`). Not an input. Using it for restart hints could not be tested honestly, because no training match has the same tagging. |
| Tactical homography | no | The ATD is already in pitch coordinates. |
| Tactical video | no | Not a pipeline input. `solution/timecode.py` maps a frame to a time in the video so a person can look at it. |
| Broadcast video, ATD, homography | no | Not synchronised to the tactical timeline, and only 65.8% of its homography is usable. |
| Public 5-minute ground truth | checks only | See below. |

**The public 5 minutes.** We scored the public window only after the configuration was fixed, to
check correctness: teams not swapped, frames aligned, the possession count right. Our file scores
spellF1 0.533 / macroF1 0.685 there, with 16 possessions against 17 in the truth. The smoothed
published baseline scores 0.326 / 0.621.

We also measured how finely the public truth cuts possessions, to understand the definition
(section 4). One experiment (16) used one public-window run, set up in advance, to decide between
two label definitions. It needed +0.03 to adopt the new labels, got −0.08, and was rejected, so
the submitted configuration is the one the held-out matches chose. Nothing was tuned on the
window.

## 3. Training data

No labels from the provided game, in any amount. We made no labels by hand at all: every label
below was produced by its data provider.

| dataset | source and licence | who labelled it | used | labelled live play | possessions |
|---|---|---|---|---|---|
| Metrica Sports sample data | github.com/metrica-sports/sample-data, commit `e706dd5`; README asks for responsible use and acknowledgement | Metrica's human-produced event data. We derive per-frame labels mechanically (below). | 3 matches | 269,754 frames (3.0 h) | 1,063 |
| IDSSE (Bassek, Rein, Weber & Memmert, *Scientific Data* 12, 195, 2025) | huggingface.co/datasets/pysport/idsse-data, revision `a715a38`; CC-BY 4.0 | The DFL feed's own per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away), from official corrected tracking | 4 of 7 matches: J03WMX, J03WN1 (Bundesliga), J03WOH, J03WOY (2. Bundesliga) | 301,977 frames (3.4 h) | 1,869 |
| SkillCorner open data | github.com/SkillCorner/opendata; MIT | SkillCorner's software (possession group) | **tried and dropped**, 5 matches | | |

That makes 7 matches with 571,731 labelled live frames (6.4 h) and 2,932 possessions. Every frame
is also used mirrored, with the teams swapped.

**Metrica labels.**

- *Dead ball:* every restart (set piece or goal kick) ends a stoppage. The stoppage began when the
  last event before it ended: ball out, foul or goal.
- *Possession:* the team of the latest control event (set piece, pass, carry, shot, recovery).
  Challenges and ball-lost / ball-out events never hand over possession, so a loose or contested
  ball stays with the last team.

**Why only 4 IDSSE matches.** The learning curve is flat from 4 to 6 training matches (0.625 to
0.626), so the other three were not downloaded.

**Why SkillCorner was dropped.** Leaving it out was better on all 7 held-out matches (0.626
against 0.578). Its labels come from software rather than human or official sources, its
tracking is a 10 fps broadcast feed, and it has no dead-ball label.

## 4. Possession rules we implemented

The challenge defines possession by *controlled* possession. A touch, a deflection or an aerial
duel won without control is no change, and a loose or contested ball stays with the last team.
Here is where our system draws those lines.

- **Dead ball → D.**
  - The tracker has lost the ball for ≥ 2 s, or the half has not kicked off yet.
  - We never assign a dead-ball owner (none is asked for), and no possession carries across a
    stoppage.
  - A dead ball that the tracker keeps seeing, such as a ball placed for a free kick, stays live
    in our file. This is a known limitation.
- **Outside the graded halves → X.**
- **Restarts.** The first possession after a stoppage is whatever the model sees when play
  resumes: usually the team nearest the ball. On the held-out matches this is right for 96.8% of
  761 restarts. Restart features built from the laws of the game (who touched it last, where it
  went out) added nothing.
- **Loose balls and challenges.** Two mechanisms keep the ball with the last team.
  - *The training labels* already do it. In Metrica, only control events hand over possession,
    not challenges. In IDSSE, the DFL's possession channel does.
  - *The Viterbi switch cost* does the rest. One switch costs 14 log units. A brief spell for the
    other team pays two switches (28), so it must outweigh 28 in summed log-odds before it
    appears. At P = 0.9 per frame that takes about 13 frames (0.5 s). At P = 0.7 it takes about
    33 frames (1.3 s).
  - The result: a deflection, a 50/50 or a one-touch block is absorbed into the current
    possession, while a clear change of control of about half a second or more switches it.
- **What we tried and did not adopt.** We also tried relabelling the training data to CHALLENGE.md's
  proxy literally ("keeps the ball for a few seconds"): a switch only after the new team holds the
  ball ≥ 2 s. This brought Metrica's labels exactly to the public truth's granularity (4.48
  against 4.23 possessions per live minute). But our predictions already had about the right
  number of possessions (16 against 17), and the relabelled model made them too long. Rejected
  (section 2).

The training labels cut possessions more finely than the public truth: per live minute, IDSSE has
9.3 possessions, Metrica 5.9 and the public window 4.2. So the switch cost was chosen on Metrica
only (14 instead of 10). This won on all 3 Metrica matches and did not hurt the all-7 mean.

## 5. Limitations, and what we would fix with more time

- **The dead ball rests on the tracker losing the ball.**
  - It fails both ways: a dead ball that stays tracked (placed for a free kick, held for a throw-in)
    reads as live, and a long live-play loss reads as dead (about 4% of gap frames by our
    estimate).
  - Fix: run our own ball detector on the tactical video for the 41% of frames without a ball.
    The learned dead-ball model already does better when live-play gaps are common (+0.017
    spellF1 under heavy simulated gaps).
- **No ball height.** The ATD's ball is 2D, so a header, a high clearance and an aerial duel all
  look alike. The ball-flight features guess the receiver by extrapolating in 2D only.
- **The label definition is only approximated.** Our training labels switch on any controlled
  touch. The organisers' cut possessions more coarsely, and the one switch cost absorbs only
  part of the difference.
  - Fix: label a few *other* matches to the challenge's own definition (the rules allow labels on
    other games), and train and choose the switch cost on those.
- **Switch timing.** On the public window our switches tend to come about 10 frames early. That
  is 17 possessions, too few to act on, and not tuned. A held-out timing study found no
  systematic delay on the training matches.
- **A small yardstick.** Seven held-out matches, from two providers whose tracking is cleaner than
  ours. Augmentation narrows the gap between them and our match, and a feature-drift check found
  every feature within a third of the training interquartile range except one neutral pressing
  feature. Still, our match is a different tracker, league and camera.
- **Error budget.** With the first model, a perfect team decision was worth +0.39 spellF1 and a
  perfect dead ball only +0.03. Most of the room is still in the team decision during fast
  exchanges.

## 6. Reproduction

Python 3.9 to 3.12, and about 2 GB of free disk during the download (about 0.2 GB stays). From the
repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-solution.txt
# put the four tactical match files in data/tactical/ (see DATA.md)
.venv/bin/python -m solution.download_external      # fetch + convert the 7 training matches
.venv/bin/python -m solution.pipeline               # train (first run) and write the file
.venv/bin/python evaluation/validate_submission.py evaluation/results/model_viterbi.csv
```

On Windows, use `py -3.12 -m venv .venv` and `.venv/Scripts/python`.

The output, `evaluation/results/model_viterbi.csv`, is the submission. It is uploaded under the
team's name.

- **Deterministic:** the training data is pinned to fixed commits, augmentation and model are
  seeded, and the file is byte-identical whether training runs on 1 CPU thread or 8.
- `python -m solution.model --evaluate` reruns the held-out scores.
- `python -m solution.experiments <name>` reruns any experiment above.
- `python -m solution.selftest` runs 15 self-checks on synthetic data.

The match data is never committed: `data/` is gitignored.
