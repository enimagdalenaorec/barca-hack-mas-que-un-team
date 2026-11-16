# Frame-level ball possession: technical document

More than a Hack 2027, Phase 1. FC Barcelona vs Athletic Club: 199,319 frames, each labelled H, A,
D or X.

**In one paragraph.** Two gradient-boosted tree models, trained only on matches from other games,
give two probabilities for every frame: P(home has the ball), from 51 per-team features, and
P(ball out of play), from 16 player-movement features. One three-state Viterbi pass per half
(home / away / dead) turns them into whole possessions and stoppages. Frames before each half's
kick-off, which is found in the tracking, are forced dead. Every choice was made by
leave-one-game-out on 10 external matches (3 Metrica, 7 IDSSE), scored with the organisers'
metric library, under a written adoption rule. The competition match and its public 5 minutes
chose nothing. On those 10 held-out matches the submitted configuration reaches spellF1
**0.662** (nested estimate; macroF1 0.922), against 0.644 for our earlier rule-based dead ball and
0.347 for the published baseline rule. The code is in `solution/`; the experiments are in
[logo_external.md](logo_external.md) (current) and [experiments.md](experiments.md) (history).

## 1. Approach, and why

**Where the points are.** Before tuning anything, we measured the error budget of the first
pipeline on 7 held-out matches by swapping in perfect answers:

| first pipeline | perfect dead ball | perfect team decision |
|---|---|---|
| 0.568 | 0.596 | 0.956 |

The team decision held almost all the error, so most of the work went there. The dead ball came
back later, once the team model was good: deciding it jointly with the team, from a learned
probability, was the largest single gain after the team model itself.

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
lower: logistic regression 0.514, random forest 0.557, against 0.578 for the boosting reference
at the time. Bigger boosting (1000 trees, 127 leaves) did not win consistently, so model size is
saturated.

**Dead ball: a learned probability, decided together with the team.** On Metrica's sample games,
where events give the true stoppages, the tracked ball is missing on 86–99% of dead frames and on
only 2–6% of live ones. So "ball missing for at least 2 s" is a strong first rule, and it was our
dead ball until the last round. Its weakness is that it only works when the tracker loses the ball
exactly at the stoppage: a dead ball that stays tracked (placed for a free kick, held for a
throw-in) reads as live, and a live-play gap reads as dead. Automatic tracking gives no guarantee
of either.

The learned model (`solution/dead_model.py`, features in `solution/dead_features.py`) adds what
players do: during a stoppage they slow down, regroup and stand still, whether the ball is seen or
not. It gives P(dead) per frame, which we soften with temperature 2 (sigmoid(logit(p) / 2)) before
decoding. One home / away / dead Viterbi per half (`decode.viterbi3`) then decides stoppages and
possessions together. Leaving play costs 20, coming back costs 8, and possession never carries
across a stoppage. On the 10 held-out matches it beats the 2 s rule under every ball-loss regime we
simulated (section 1, "How choices were made"). The gain is largest exactly where the rule is
weakest: +0.04 to +0.07 when the ball stays visible for up to 3 s after a stoppage.
`--dead rule` still produces the rule-based file.

A second rule fixes a bug the first pipeline had: nothing is live before a half's kick-off. The
ball is tracked while it sits on the centre spot, so "ball missing" alone called most frames
before the second-half kick-off live. The kick-off is found in the tracking, not copied from the
docs: the players' set formation breaks with the ball inside the centre circle. It is detected at
frame 113,750 against the documented 113,742.

**Making clean training data look like ours.** The training matches have clean, often corrected
tracking. Ours is automatic: according to DATA.md the ball is missing on 41% of graded frames, in
long blocks, and about 20.5 of 22 players are tracked per frame. Before computing features, each
training match is changed, with a fixed seed:

- short ball gaps (under 1 s) are bridged by interpolation;
- the ball is hidden across 95% of stoppages;
- long live-play gaps are cut until 5% of live frames have no ball;
- players are dropped in random blocks covering 3% of frames.

The result is insensitive to these settings: every variant scored within ±0.01 of the reference.
Training under harsher regimes, or with position noise, did not help either. Our guesses about the
tracker do not drive the score.

**Switching team: a cost that depends on the evidence.** Changing team normally costs 20 log
units. Within 10 frames of a "flip" it costs 0.4 × 20 = 8. A flip is a frame where the team
nearest the ball changes and the new nearest player is within 2 m of it. A real change of
possession nearly always passes through such a moment, while flicker in P(home) during a loose
ball usually does not. So short possessions that start and end on a clear change of the nearest
team are cheap to keep, and switching elsewhere is dearer than before (20 instead of 14), which
removes flicker. Section 4 gives the arithmetic.

**How choices were made.** Leave-one-game-out on 10 external matches: 3 Metrica sample games and
7 IDSSE matches. For each match, the models are trained without it (on the remaining Metrica games
plus the IDSSE training set; section 3). The held-out match is first made to look like automatic
tracking, with two different seeds, then run through the whole pipeline and scored with the
organisers' `possession_metrics.py`. The same harness also scores every variant under two harsher
held-out regimes:

- `harsh`: more and longer live-play gaps, and fewer stoppages hidden;
- `lag`: the ball stays visible for up to 3 s after a stoppage begins and reappears up to 3 s
  before it ends.

The adoption rule was written before each round was run:

- the mean spellF1 rises by at least 0.008;
- the variant wins on at least 7 of 10 matches;
- it loses no more than 0.005 under `harsh` and `lag`.

A hyperparameter chosen on held-out scores is judged on its nested estimate: chosen on 9 matches,
scored on the 10th. The seed of the held-out degradation alone moves the mean by up to 0.011, so
smaller differences are noise.

| held-out, 10 matches, 2 seeds | spellF1 | Metrica (3) | IDSSE (7) | macroF1 | `harsh` | `lag` |
|---|---|---|---|---|---|---|
| **submitted** (fixed settings) | **0.664** | 0.751 | 0.627 | **0.922** | 0.628 | 0.650 |
| submitted, nested estimate | 0.662 | 0.749 | 0.625 | 0.922 | 0.627 | 0.648 |
| learned dead ball, before round 2 | 0.653 | 0.740 | 0.615 | 0.915 | 0.616 | 0.624 |
| 2 s dead-ball rule, switch cost 14 | 0.644 | 0.749 | 0.599 | 0.901 | 0.592 | 0.581 |
| published baseline rule (nearest player, ±50-frame vote) | 0.347 | | | 0.799 | 0.328 | 0.343 |

The submitted configuration beats the rule-based version on 9 of 10 matches, and the version
before round 2 on all 10. Under simulated position noise (players 0.25–0.5 m, ball 0.5–1 m, an
assumption) the flip gain shrinks to zero: the submission is then level with the version before
round 2 (within 0.001) and still above the rule-based one.

**Tried and dropped** (details and numbers in [logo_external.md](logo_external.md) and
[experiments.md](experiments.md)):

- SkillCorner as training data (section 3).
- All 7 IDSSE matches for the team model (the dead model does use all 7).
- Relabelling the training data to "controlled" possession; a minimum possession length.
- Restart features built from the laws of the game. The team that restarts is already right 97%
  of the time from proximity.
- Ensembles (over seeds and subsamples) and per-source weights.
- Self-training on an unlabelled held-out match. It got worse every time, because the model
  reinforced its own mistakes.
- A second-stage model over the probabilities; posterior (max-marginal) decoding.
- Temperature on P(home); a joint grid of switch, stop and restart costs; other GBM settings for
  the dead model.
- Training with position noise or under harsher ball-loss regimes.
- A bigger team model.

## 2. Assets we used, and how

| asset | used? | how |
|---|---|---|
| Tactical ATD (tracking) | **yes, the only input** | Ball and player positions in metres; team of each slot. The exported ball speed is unusable, so speed is computed from positions. |
| Smart Tagging | no | Not an input. Using it for restart hints could not be tested honestly, because no training match has the same tagging. |
| Tactical homography | no | The ATD is already in pitch coordinates. |
| Tactical video | no | Not a pipeline input. `solution/timecode.py` maps a frame to a time in the video so a person can look at it. |
| Broadcast video, ATD, homography | no | Not synchronised to the tactical timeline, and only 65.8% of its homography is usable. |
| Public 5-minute ground truth | checks only | See below. |

**The public 5 minutes.** The public window is used only to check that the file is correct: teams
not swapped, frames aligned, a sensible possession count. Every setting of the submitted
configuration was chosen by the external harness (section 1).

## 3. Training data

No labels from the provided game, in any amount. We made no labels by hand at all: every label
below was produced by its data provider.

| dataset | source and licence | who labelled it | used | labelled live play | possessions |
|---|---|---|---|---|---|
| Metrica Sports sample data | github.com/metrica-sports/sample-data, commit `e706dd5`; README asks for responsible use and acknowledgement | Metrica's human-produced event data. We derive per-frame labels mechanically (below). | 3 matches | 269,754 frames (3.0 h) | 1,063 |
| IDSSE (Bassek, Rein, Weber & Memmert, *Scientific Data* 12, 195, 2025) | huggingface.co/datasets/pysport/idsse-data, revision `a715a38`; CC-BY 4.0 | The DFL feed's own per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away), from official corrected tracking | team model: 4 matches (J03WMX, J03WN1, J03WOH, J03WOY); dead model and validation: all 7 (also J03WPY, J03WQQ, J03WR9) | the 4 team-model matches: 301,977 frames (3.4 h) | 1,869 (4 matches) |
| SkillCorner open data | github.com/SkillCorner/opendata; MIT | SkillCorner's software (possession group) | **tried and dropped**, 5 matches | | |

Every frame is also used mirrored, with the teams swapped.

**Metrica labels.**

- *Dead ball:* every restart (set piece or goal kick) ends a stoppage. The stoppage began when the
  last event before it ended: ball out, foul or goal.
- *Possession:* the team of the latest control event (set piece, pass, carry, shot, recovery).
  Challenges and ball-lost / ball-out events never hand over possession, so a loose or contested
  ball stays with the last team.

**Why 4 IDSSE matches for the team model and 7 for the dead model.** Training the team model on
all 7 did not help on the held-out matches (0.641 against 0.645, 5 of 10), while training the dead
model on all 7 did (+0.002, 9 of 10, and more under `harsh` and `lag`), and it is part of the
adopted combination.

**Why SkillCorner was dropped.** Leaving it out was better on all 7 held-out matches of the first
yardstick (0.626 against 0.578). Its labels come from software rather than human or official
sources, its tracking is a 10 fps broadcast feed, and it has no dead-ball label.

## 4. Possession rules we implemented

The challenge defines possession by *controlled* possession. A touch, a deflection or an aerial
duel won without control is no change, and a loose or contested ball stays with the last team.
Here is where our system draws those lines.

- **Dead ball → D.**
  - Decided jointly with the team by the three-state Viterbi from the learned P(dead), and forced
    before each half's kick-off.
  - We never assign a dead-ball owner (none is asked for), and no possession carries across a
    stoppage: a new possession after a stoppage pays the restart cost, whichever team it goes to.
- **Outside the graded halves → X.**
- **Restarts.** The first possession after a stoppage is whatever the model sees when play
  resumes: usually the team nearest the ball. On the held-out matches this is right for 96.8% of
  761 restarts. Restart features built from the laws of the game (who touched it last, where it
  went out) added nothing.
- **Loose balls and challenges.** Two mechanisms keep the ball with the last team.
  - *The training labels* already do it. In Metrica, only control events hand over possession,
    not challenges. In IDSSE, the DFL's possession channel does.
  - *The switch cost* does the rest. A brief spell for the other team pays two switches. Away from
    a flip that is 40 log units: at P = 0.9 per frame the spell must last about 18 frames
    (0.7 s), and at P = 0.7 about 47 frames (1.9 s). If the spell starts and ends at a flip (the
    nearest team changes, within 2 m), it costs only 16: about 7 frames (0.3 s) at P = 0.9.
  - The result: a deflection, a 50/50 or a one-touch block with no clear change of the nearest
    player is absorbed into the current possession. A clean change of control, where the other
    team's player arrives at the ball, switches after a fraction of a second.
- **What we tried and did not adopt.** We tried relabelling the training data to CHALLENGE.md's
  proxy literally ("keeps the ball for a few seconds"): a switch only after the new team holds the
  ball for 0.5, 1 or 2 s. Measured against the providers' own labels, every version lost (−0.004
  to −0.047, at most 1 of 10 wins), because it only removes possessions and our predictions
  already have fewer than the labels (about 307 against 412 per match). A minimum possession
  length lost for the same reason.

## 5. Limitations, and what we would fix with more time

- **The dead ball and the tracker.** The learned model reads player movement, so it no longer
  needs the tracker to lose the ball at a stoppage. But external data cannot tell us how far the
  automatic tracker's ball gaps drift from the true stoppages: Metrica's gaps match its stoppages
  to the frame, and IDSSE's ball is complete. We cover this with stress regimes, not a measurement.
  - Fix: run our own ball detector on the tactical video for the frames without a ball.
- **The flip needs accurate positions.** Under simulated position noise its gain disappears.
  The size of the real tracker's position noise is unknown to us.
- **No ball height.** The ATD's ball is 2D, so a header, a high clearance and an aerial duel all
  look alike. The ball-flight features guess the receiver by extrapolating in 2D only.
- **The label definition is only approximated.** Our training labels switch on any controlled
  touch, and the organisers' may cut possessions more coarsely. The switch cost absorbs part of
  any difference, but we cannot measure it without labels made to the challenge's definition.
  - Fix: label a few *other* matches to the challenge's own definition (the rules allow labels on
    other games), and train and choose the costs on those.
- **A small yardstick.** Ten held-out matches, from two providers whose tracking is cleaner than
  ours. Augmentation narrows the gap between them and our match, but our match is a different
  tracker, league and camera.
- **Too few short possessions.** On the held-out matches we predict about 307 possessions where
  the labels have 412. Most of the remaining error is in fast exchanges.

## 6. Reproduction

Python 3.9 to 3.12, and about 2 GB of free disk during the download (about 0.2 GB stays). From the
repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-solution.txt
# put the four tactical match files in data/tactical/ (see DATA.md)
.venv/bin/python -m solution.download_external      # fetch (or read local copies) + convert the training matches
.venv/bin/python -m solution.pipeline               # train (first run) and write the file
.venv/bin/python evaluation/validate_submission.py evaluation/results/model_viterbi.csv
```

On Windows, use `py -3.12 -m venv .venv` and `.venv/Scripts/python`.

The output, `evaluation/results/model_viterbi.csv`, is the submission. It is uploaded under the
team's name.

- **Deterministic:** the training data is pinned to fixed revisions, augmentation and models are
  seeded, and the file is byte-identical whether training runs on 1 CPU thread or 8. Raw files
  already in `data/external/metrica/` and `data/external/idsse/` are read in place; they are
  byte-identical to the pinned revisions.
- `python -m solution.logo <name>` and `python -m solution.logo2 <name>` rerun the external
  leave-one-game-out experiments.
- `python -m solution.selftest` runs 17 self-checks on synthetic data.

The match data is never committed: `data/` is gitignored.
