# Possession Pipeline Optimisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Eni's pipeline reproducible, then replace the 2-second dead-ball rule with a learned
P(dead) and one H/A/D Viterbi, promoted only if it wins on held-out matches under realistic ball gaps.

**Architecture:** New files `dead_features.py`, `dead_model.py`, `gap_stats.py`, `selftest.py` in
`solution/`; `decode.py` gains `viterbi3`; `augment.augment` gains a `params` override so training
and held-out matches can be degraded under named regimes. Eni's path stays the default until
Task 7's promotion rule says otherwise.

**Tech Stack:** Python 3.12, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1 (HistGradientBoosting),
scipy 1.13.1. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-10-possession-optimisation-design.md`

## Global Constraints

- Python: `py -3.12 -m venv .venv`; every command runs with `.venv/Scripts/python` (Windows) /
  `.venv/bin/python` (Linux, macOS).
- Never print, cat, head or read match data or files derived from it (`data/tactical/`,
  `data/broadcast/`, `data/cache/BAR-ATH*`, `evaluation/results/`, `evaluation/reports/`).
  Scripts that touch our match print aggregate numbers only.
- The public 5 minutes are run once, in Task 8, as a correctness check. No choice depends on them.
- Selection uses leave-one-match-out scores on the 7 labelled matches (3 Metrica, 4 IDSSE) via
  `evaluation/possession_metrics.py`.
- Promotion rule: under the `calibrated` held-out regime, beat Eni's config on mean spellF1 and on
  at least 5 of 7 matches; under the `default` held-out regime, not worse by more than 0.01.
- Seeds stay as Eni's (`model.seed_of`); every model is seeded so the same data gives the same file.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A half where no kick-off formation is found (`kickoff_frame` returns `start`): the forced-dead
  mask is all False there and decoding must still work. Pinned in Task 3.
- NaN probabilities (frames with no features, e.g. nobody tracked) inside a half: treated as 0.5,
  never crash or propagate NaN into the path. Pinned in Task 3.
- Very short halves or runs (length 0 or 1) reaching `viterbi3`. Pinned in Task 3.
- Our match has fewer than 5 fast ball gaps: `LIVE_GAP_S` falls back to the default (2.0, 15.0)
  instead of a percentile of two numbers. Pinned in Task 6.
- Output depends on CPU thread count (OpenMP in HistGradientBoosting): the submission must be
  byte-identical with 1 and with 16 threads. Pinned in Task 8.

---

### Task 1: Environment, pinned training data, reproduce Eni

**Files:**
- Modify: `solution/download_external.py` (`METRICA_URL`, `IDSSE_URL`)
- Modify: `SOLUTION.md` (Python 3.12 note, Windows commands)

**Interfaces:**
- Produces: `data/cache/sample_game_{1,2,3}_v3.npz`, `data/cache/ext_idsse_*_v1.npz`; a working `.venv`.

- [ ] **Step 1: Create the venv and install**

Run: `py -3.12 -m venv .venv && .venv/Scripts/python -m pip install -r requirements-solution.txt`
Expected: installs numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1 without building from source.

- [ ] **Step 2: Look up the revisions to pin**

Run: `gh api repos/metrica-sports/sample-data/commits/master --jq .sha` and
`curl -s https://huggingface.co/api/datasets/pysport/idsse-data | python -c "import json,sys;print(json.load(sys.stdin)['sha'])"`
Expected: two 40-character SHAs.

- [ ] **Step 3: Pin both URLs**

`METRICA_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/<sha>/data/"`,
`IDSSE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/<sha>/"`, each with a
one-line comment: pinned so the training set cannot change under the submission.

- [ ] **Step 4: Download and convert (background, ~1.7 GB)**

Run: `.venv/Scripts/python -m solution.download_external`
Expected: ends with a `total ... MB` line listing 4 idsse matches; the 3 Metrica caches exist.

- [ ] **Step 5: Reproduce Eni's held-out score**

Run: `.venv/Scripts/python -m solution.model --evaluate`
Expected: `viterbi cost 10` row shows spellF1 0.645 (±0.002) and baseline 0.449. If not, stop and report.

- [ ] **Step 6: Update SOLUTION.md** with "Python 3.9-3.12 (numpy 2.0.2 has no 3.13 wheels)" and a
Windows line next to each `.venv/bin/python` command.

- [ ] **Step 7: Commit**

```bash
git add solution/download_external.py SOLUTION.md
git commit -m "Pin training-data revisions; document Python 3.12 and Windows commands"
```

### Task 2: Label-granularity check

**Files:**
- Modify: `solution/experiments.py` (add `exp_labels`, dispatch name `labels`)
- Modify: `docs/experiments.md`

**Interfaces:**
- Consumes: `labelled()`, `M.truth_codes(arrays)`, `pm.segments`, `pm.read_label_csv`, `pm.label_codes`.
- Produces: printed table only.

- [ ] **Step 1: Implement `exp_labels()`**

For each labelled match (raw truth, no augmentation) and for the public ground truth
(`evaluation/gt/BAR-ATH_possession_gt_public.csv`, graded frames only), print: possessions per live
minute, median possession s, share of possessions under 1 s and under 3 s, direct H<->A
turnovers per live minute. Group rows by source (metrica, idsse, public).

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python -m solution.experiments labels`
Expected: one row per match plus a `public` row (17 possessions over the window).

- [ ] **Step 3: Decision point.** If any source's direct turnovers per live minute or median
possession length differs from the public row by 3x or more, stop and report to the user before
Task 3. Otherwise record the table in `docs/experiments.md` under "Phase 5 (bruno): label granularity".

- [ ] **Step 4: Commit**

```bash
git add solution/experiments.py docs/experiments.md
git commit -m "Compare training-label granularity with the public window"
```

### Task 3: Three-state decoder and forced pre-kick-off dead mask

**Files:**
- Modify: `solution/dead_ball.py` (extract `pre_kickoff_mask`; `dead_mask` calls it)
- Modify: `solution/decode.py` (add `viterbi3`, `STOP_COST`, `RESTART_COST`)
- Modify: `solution/model.py` (add `labels_from3`)
- Create: `solution/selftest.py`

**Interfaces:**
- Produces:
  - `dead_ball.pre_kickoff_mask(arrays, runs) -> np.ndarray[bool]` (N,): True on `[start, kickoff_frame(arrays, start, stop))` of each run.
  - `decode.viterbi3(p_dead, p_home, forced_dead, switch_cost, stop_cost, restart_cost) -> np.ndarray[int8]`, one run; 0 = H, 1 = A, 2 = D.
  - `decode.STOP_COST = 8.0`, `decode.RESTART_COST = 8.0` (placeholders until Task 7 sets them).
  - `model.labels_from3(p_home, p_dead, arrays, runs, costs) -> np.ndarray[int8]` of `pm` codes; `costs = (switch, stop, restart)`; X outside runs.
  - `python -m solution.selftest` runs every `test_*` function in the module and prints `ok N`.

- [ ] **Step 1: Write the failing tests in `solution/selftest.py`**

```python
def test_viterbi3_follows_clear_evidence():
    p_dead = np.r_[np.full(50, .02), np.full(50, .98), np.full(50, .02)]
    p_home = np.r_[np.full(50, .95), np.full(50, .5), np.full(50, .05)]
    path = decode.viterbi3(p_dead, p_home, np.zeros(150, bool), 10, 8, 8)
    assert (path[:50] == 0).all() and (path[50:100] == 2).all() and (path[100:] == 1).all()

def test_viterbi3_ignores_one_frame_blip():
    p_home = np.full(100, .9); p_home[40] = .01
    path = decode.viterbi3(np.full(100, .02), p_home, np.zeros(100, bool), 10, 8, 8)
    assert (path == 0).all()

def test_viterbi3_forced_dead_wins_over_evidence():
    forced = np.zeros(60, bool); forced[:20] = True
    path = decode.viterbi3(np.full(60, .01), np.full(60, .9), forced, 10, 8, 8)
    assert (path[:20] == 2).all() and (path[20:] == 0).all()

def test_viterbi3_nan_and_tiny_inputs():
    path = decode.viterbi3(np.full(30, np.nan), np.full(30, np.nan), np.zeros(30, bool), 10, 8, 8)
    assert path.shape == (30,) and set(path) <= {0, 1, 2}
    assert decode.viterbi3(np.r_[.9], np.r_[.5], np.r_[False], 10, 8, 8).tolist() == [2]
    assert decode.viterbi3(np.array([]), np.array([]), np.array([], bool), 10, 8, 8).shape == (0,)

def test_pre_kickoff_mask_no_formation_is_all_false():
    arrays = synthetic_match(200)            # players scattered: no kick-off formation
    assert dead_ball.pre_kickoff_mask(arrays, [(0, 200)]).sum() == 0
```

Also in `selftest.py`, the shared helper used by Tasks 3-6:
`synthetic_match(n, moving=True, with_labels=False, seed=0) -> dict` with `pitch` `[105, 68]`,
`slot_team` 11 zeros + 11 ones, `players_xy` uniform over the pitch (`default_rng(seed)`) and
random-walking ~2 m/s when `moving` else fixed, `ball_xy` following player 0. With `with_labels`:
`runs` `[(0, n)]`, `dead` alternating 400-frame live / 150-frame dead blocks (players fixed and ball
NaN on dead blocks), `owner` 0/1 alternating per live block (-1 on dead), `has_dead` 1.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m solution.selftest`
Expected: AttributeError for `viterbi3` / `pre_kickoff_mask`.

- [ ] **Step 3: Implement `pre_kickoff_mask`** and make `dead_mask` use it (behaviour unchanged).

- [ ] **Step 4: Implement `viterbi3`**, Eni's `viterbi` generalised to three states. Clip NaN to
0.5 and probabilities to `[EPS, 1-EPS]`. Emissions: D = log p_dead; H = log(1-p_dead) + log p_home;
A = log(1-p_dead) + log(1-p_home); where `forced_dead`, H and A emissions are -inf. Transition
costs: H<->A `switch_cost`, H/A->D `stop_cost`, D->H/A `restart_cost`, staying free. Empty input
returns an empty int8 array.

- [ ] **Step 5: Implement `labels_from3`**: per run, `viterbi3` over the whole run with
`forced_dead = dead_ball.pre_kickoff_mask(arrays, runs)`, mapped 0/1/2 -> `pm.H/pm.A/pm.D`.

- [ ] **Step 6: Run self-tests and Eni's regression**

Run: `.venv/Scripts/python -m solution.selftest` -> `ok 5`.
Run: `.venv/Scripts/python -m solution.model --evaluate` -> still 0.645 (dead_mask refactor changed nothing).

- [ ] **Step 7: Commit**

```bash
git add solution/dead_ball.py solution/decode.py solution/model.py solution/selftest.py
git commit -m "Add three-state H/A/D Viterbi with forced pre-kick-off dead frames"
```

### Task 4: Dead-ball features

**Files:**
- Create: `solution/dead_features.py`
- Modify: `solution/selftest.py`

**Interfaces:**
- Consumes: `features._rolling_mean`, `features._time_since_seen` (reuse, do not copy).
- Produces: `dead_features.compute(arrays, runs) -> (names: list[str], matrix: float32 [N, k])`, NaN outside runs, team-agnostic.

Features (names fixed so Task 5's fingerprint is stable): `spd_mean`, `spd_p75`, `spd_mean_past1`,
`spd_mean_next1`, `spd_mean_past3`, `spd_mean_next3`, `spd_drop3` (past3 - next3), `still_share`
(players under 1.0 m/s), `since_seen_s`, `until_seen_s`, `last_ball_touch_d` (last-seen ball
distance to the nearest touchline), `last_ball_goal_d` (to the nearest goal line), `last_ball_speed`,
`ball_still` (visible and under 0.5 m/s over 0.4 s), `box_count_max` (players inside the busier
penalty area), `own_half_share` (from `dead_ball.own_half_share`). Player speeds from positions
(`VEL_LAG` 2 frames, as in `features.py`).

- [ ] **Step 1: Write failing tests**

```python
def test_dead_features_shape_and_nan_outside_runs():
    arrays = synthetic_match(300)          # helper from Task 3
    names, X = dead_features.compute(arrays, [(50, 250)])
    assert X.shape == (300, len(names)) and len(names) == 16
    assert np.isnan(X[:50]).all() and np.isnan(X[250:]).all()

def test_dead_features_still_players_read_as_still():
    arrays = synthetic_match(300, moving=False)
    names, X = dead_features.compute(arrays, [(0, 300)])
    assert np.nanmedian(X[:, names.index("still_share")]) > 0.9
```

- [ ] **Step 2: Run** `python -m solution.selftest` -> fails on import.
- [ ] **Step 3: Implement `compute`** with the names above.
- [ ] **Step 4: Run** `python -m solution.selftest` -> `ok 7`.
- [ ] **Step 5: Commit** `git commit -m "Add dead-ball features from player movement and ball context"`

### Task 5: Dead-ball model and pipeline flag

**Files:**
- Modify: `solution/augment.py` (`augment(arrays, seed, params=None)`)
- Create: `solution/dead_model.py`
- Modify: `solution/model.py` (`rows`/`fit`/`train_and_save`/`load_or_train` accept `params=None`, passed to `augment.augment`; the `rows` memo key includes `params`; `AUG_REGIME = "default"`, resolved with `augment.regime(AUG_REGIME)`; fingerprint includes the resolved params)
- Modify: `solution/pipeline.py` (`--dead rule|model`, default `rule`)

**Interfaces:**
- Produces:
  - `augment.augment(arrays, seed, params=None)`: `params` dict overrides `BRIDGE_S`, `HIDE_DEAD_P`, `LIVE_GAP_SHARE`, `LIVE_GAP_S`, `PLAYER_DROP_SHARE`, `PLAYER_DROP_S` for this call only (no module mutation).
  - `augment.REGIMES = {"default": {}, "harsh": {"LIVE_GAP_SHARE": 0.15, "LIVE_GAP_S": (3.0, 30.0), "HIDE_DEAD_P": 0.85}}` and `augment.regime(name) -> dict` returning `REGIMES[name]`; Task 6 extends it with `"calibrated"`.
  - `dead_model.fit(matches, params=None) -> HistGradientBoostingClassifier` (GBM params = `model.GBM`; rows: every graded frame of every match with `has_dead`, augmented with `seed_of(name)`, subsampled by `model.SUBSAMPLE`, label `dead`).
  - `dead_model.p_dead(clf, arrays, runs) -> np.ndarray[float]` (NaN outside runs).
  - `dead_model.load_or_train(params=None)` caching `data/cache/dead_model.pkl` with a fingerprint like `model.fingerprint`.
  - `pipeline.build_model(arrays, runs, dead="rule")`.

- [ ] **Step 1: Write failing tests**

```python
def test_augment_params_do_not_leak():
    arrays = synthetic_match(500, with_labels=True)
    a = augment.augment(arrays, 1, {"LIVE_GAP_SHARE": 0.5})
    b = augment.augment(arrays, 1)
    assert augment.LIVE_GAP_SHARE == 0.05
    assert np.isnan(a["ball_xy"][:, 0]).sum() > np.isnan(b["ball_xy"][:, 0]).sum()

def test_dead_model_learns_synthetic_stoppages():
    m = [("s{}".format(i), synthetic_match(3000, with_labels=True, seed=i)) for i in range(3)]
    clf = dead_model.fit(m[:2])
    p = dead_model.p_dead(clf, m[2][1], m[2][1]["runs"])
    dead = m[2][1]["dead"]
    assert np.nanmean(p[dead]) > np.nanmean(p[~dead]) + 0.3
```

- [ ] **Step 2: Run** selftest -> fails.
- [ ] **Step 3: Implement** the `augment` override, `dead_model.py`, the `regime` plumbing in `model.py`, and the `--dead` flag.
- [ ] **Step 4: Run** selftest -> `ok 9`; `python -m solution.model --evaluate` -> still 0.645.
- [ ] **Step 5: Commit** `git commit -m "Add learned dead-ball model behind --dead flag"`

### Task 6: Calibrated regime from our match's ball gaps

Needs `data/tactical/` on disk.

**Files:**
- Create: `solution/gap_stats.py`
- Modify: `solution/augment.py` (`regime("calibrated")` computed lazily, once per process, via `gap_stats.from_disk()`)

**Interfaces:**
- Consumes: `tracking.load()`, `diagnostics.graded_runs()`, `model.training_matches()` raw arrays.
- Produces:
  - `gap_stats.calibrated(ours, our_runs, labelled) -> {"LIVE_GAP_SHARE": float, "LIVE_GAP_S": (float, float)}`, pure: `ours` is a tracking dict, `labelled` a list of training dicts with `dead`.
  - `gap_stats.from_disk() -> dict`, loading our match and the training matches, printing the aggregates below; `python -m solution.gap_stats` calls it.
  - `augment.regime("default"|"calibrated"|"harsh")`.

Algorithm (aggregates only, no labels of our match):

```
v_dead, v_live = mean per-frame player speed on dead / live frames of the 7 labelled matches (raw)
gaps           = ball-missing stretches >= dead_ball.MIN_DEAD_S inside our graded runs
m_long         = frames in gaps / graded frames
v_gap          = mean per-frame player speed over gap frames
f              = clip((v_gap - v_dead) / (v_live - v_dead), 0, 1)     # share of gap frames that are live
dead           = (1 - f) * m_long / HIDE_DEAD_P
LIVE_GAP_SHARE = f * m_long / (1 - dead)
LIVE_GAP_S     = (p25, p75) of lengths (s) of gaps whose mean speed > (v_dead + v_live) / 2;
                 default (2.0, 15.0) if fewer than 5 such gaps
```

- [ ] **Step 1: Write failing tests**

```python
def test_calibrated_all_gaps_dead_gives_no_live_gaps():
    train = [synthetic_match(5500, with_labels=True, seed=s) for s in range(2)]
    ours = dict(train[0]); ours["ball_xy"] = ours["ball_xy"].copy()   # gaps only on dead blocks
    out = gap_stats.calibrated(ours, ours["runs"], train)
    assert out["LIVE_GAP_SHARE"] < 0.01 and out["LIVE_GAP_S"] == (2.0, 15.0)   # < 5 fast gaps -> default

def test_calibrated_live_gaps_raise_share():
    train = [synthetic_match(5500, with_labels=True, seed=s) for s in range(2)]
    ours = synthetic_match(5500, seed=7)                                 # moving players everywhere
    for a in range(0, 5500, 500): ours["ball_xy"][a:a + 150] = np.nan    # 11 fast 6 s gaps
    out = gap_stats.calibrated(ours, [(0, 5500)], train)
    assert out["LIVE_GAP_SHARE"] > 0.2 and 5.0 <= out["LIVE_GAP_S"][0] <= out["LIVE_GAP_S"][1] <= 7.0
```

- [ ] **Step 2: Run** selftest -> fails. **Step 3: Implement.** **Step 4: Run** selftest -> `ok 11`.
- [ ] **Step 5: Run** `.venv/Scripts/python -m solution.gap_stats` -> prints `m_long`, `v_gap`, `v_dead`, `v_live`, `f`, and the regime dict (aggregates only). Record them in `docs/experiments.md`.
- [ ] **Step 6: Commit** `git commit -m "Calibrate augmentation to aggregate ball-gap statistics of our match"`

### Task 7: Dead-ball experiment and promotion

**Files:**
- Modify: `solution/experiments.py` (add `exp_dead`, dispatch name `dead`)
- Modify: `solution/decode.py` (`STOP_COST`, `RESTART_COST` from the experiment)
- Modify: `solution/model.py` (`AUG_REGIME`) and `solution/pipeline.py` (default of `--dead`) only if promoted
- Modify: `docs/experiments.md`

**Interfaces:**
- Consumes: everything above.

Grid: training regime T in {default, calibrated}; dead method in {rule, model}; held-out regime E
in {default, calibrated, harsh}; for `model`, stop and restart costs each in {4, 8, 12, 20},
switch cost `decode.SWITCH_COST`. Reference = (T default, rule) = Eni. For each (T, method) report
mean spellF1, macroF1 and wins of 7 vs reference under each E; for `model` pick the cost pair with
the best mean under E=calibrated (state that it is chosen on held-out, as Eni's switch cost was).

- [ ] **Step 1: Implement `exp_dead`** (team and dead models fitted once per fold and T; held-out
augmented with `seed_of(name) + 1` and `augment.regime(E)`). Run it in the background:
`.venv/Scripts/python -m solution.experiments dead`. Expected: a 4-row table x 3 regimes.
- [ ] **Step 2: Apply the promotion rule** (Global Constraints). Promoted -> set `AUG_REGIME`,
the `--dead` default, `STOP_COST`, `RESTART_COST`. Not promoted -> leave defaults, record why.
- [ ] **Step 3: Record** the table and the decision in `docs/experiments.md`.
- [ ] **Step 4: Run** selftest -> `ok 11`.
- [ ] **Step 5: Commit** `git commit -m "Dead-ball experiment: <promoted|kept rule>, held-out <numbers>"`

### Task 8: Determinism, public-window check, docs

**Files:**
- Modify: `SOLUTION.md`, `docs/experiments.md`

- [ ] **Step 1: Determinism.** Delete `data/cache/model.pkl` and `data/cache/dead_model.pkl`, run
`OMP_NUM_THREADS=1 .venv/Scripts/python -m solution.pipeline --name det1`; delete both again, run
`.venv/Scripts/python -m solution.pipeline --name det16`; run `sha256sum evaluation/results/det1.csv evaluation/results/det16.csv`.
Expected: identical hashes. If not, stop and report. Then delete `det1.csv` and `det16.csv` so they
stay off the scoreboard in Step 3.
- [ ] **Step 2: Validate** `python evaluation/validate_submission.py evaluation/results/model_viterbi.csv` -> passes (prints counts only).
- [ ] **Step 3: Public-window correctness check, once.** `python evaluation/run_evaluation.py`;
record the board rows for `model_viterbi` and the baselines. Expected: alive/dead F1 and team
accuracy at or above `baseline_smooth50`'s; anything far below means a bug (swap, offset), not a tuning signal.
- [ ] **Step 4: Docs.** `SOLUTION.md`: the final configuration and the regenerate command;
`docs/experiments.md`: Phase 5 summary.
- [ ] **Step 5: Commit** `git commit -m "Document final configuration and reproducibility check"`
