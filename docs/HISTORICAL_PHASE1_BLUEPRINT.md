# Historical Phase 1 — Engineering Blueprint

**Status:** Active plan  
**Scope:** Historical race-state prediction only  
**Base branch:** `fix/historical-feature-contract`  
**Working branch:** `feat/historical-phase1-blueprint`

## 1. Objective

Build a historically trustworthy, race-state winner-probability model. The model must estimate each active driver's probability of winning from information that would genuinely have been available at the stated prediction time.

The goal is not to reproduce another repository or optimize a headline accuracy number. External projects are research references; this repository owns its data contracts, feature definitions, implementation, tests and evaluation.

**Phase 1 does not include live-provider integration, UI expansion, or deployment work.** Live prediction and replay product integration remain later milestones.

## 2. Research references and what we adopt

These repositories were inspected as public research references:

| Reference | Relevant ideas | What we will not copy |
|---|---|---|
| [Jared-Chan/f1ml](https://github.com/Jared-Chan/f1ml) (190 stars; archived; last push 2022-03-25) | Race-level sequence representation; lap-by-lap prediction; driver/constructor context; pit strategy as a time-to-event problem; empirical caution against unnecessarily large LSTMs | Its old Ergast-era data pipeline, notebook architecture, fixed encodings, or LSTM implementation |
| [Akhil-0412/FormulAI](https://github.com/Akhil-0412/FormulAI) (1 star at review; active in October 2026) | Strictly historical features; walk-forward evaluation; learning-to-rank; probability calibration; baseline comparison; tests for feature integrity | Its exact feature/model stack, API, storage, and reported metrics |
| [mehmetkahya0/f1-race-prediction](https://github.com/mehmetkahya0/f1-race-prediction) (50 stars at review; active in October 2026) | Monte Carlo race uncertainty; integration of historical pace and simulation context | Its race simulator, assumptions, ratings or hard-coded 2026 context |

Star counts and repository activity are time-sensitive snapshots, not evidence that a model is scientifically correct. Reported scores in external READMEs are not accepted as our benchmarks unless we can reproduce them on our own data and protocol.

## 3. Current project baseline

The current historical dataset under evaluation is:

`data/processed/features/jolpica_race_state_repaired.parquet`

Known dataset summary from the current project audit:
- Seasons: 2023–2025
- Races: 70
- Rows: 77,617
- Duplicate `(session_key, driver_number, lap_number)` keys: 0
- Missing race/driver/lap sequence transitions: 0 in the audited driver-lap sequence
- Several tyre and weather fields are entirely null and must not be treated as informative features.

Current chronological evaluation:
- Train: 2023
- Validation: 2024
- Final holdout: 2025

Previously observed validation/test results are recorded in the project handoff and experiment output. They show that the current model is not yet a reliable improvement over the current-leader baseline on the 2025 holdout. The 2025 results are diagnostic only and must not guide further feature selection or tuning.

## 4. Critical correctness issue to resolve first: snapshot time

The static archive builder reconstructs each driver's timestamp from race start plus that driver's cumulative lap duration. It also derives positions and gaps from lap-end records. Drivers completing the same numbered lap do not finish it at the same instant.

The current evaluation groups rows by `(session_key, lap_number)` as if every driver row in that group were a simultaneous snapshot. This assumption is not yet justified.

Before training or tuning, quantify the timestamp spread among drivers grouped by race and lap:
- median, 90th, 95th percentile and maximum spread;
- counts of snapshots exceeding 10s and 30s;
- missing reconstructed timestamps;
- examples of the largest spreads.

Do not silently treat reconstructed timestamps as official timing-feed timestamps.

### Snapshot policy decision gate

If same-lap timestamp spreads are material, evaluation must be redesigned around a clearly defined prediction-time protocol. Candidate approach for investigation:
1. Define common race-clock evaluation instants.
2. For each driver, select only their latest observation at or before that instant.
3. Enforce `observation_time <= prediction_time` for every selected row.
4. Define active/retired/unknown driver handling explicitly.
5. Recompute or validate race-relative features at that common instant; do not assume lap-end gap/position values are automatically synchronized.
6. Record maximum observation age and reject snapshots exceeding a documented staleness threshold.

This is a research/design candidate, not an approved implementation. Because the archive's timing is reconstructed, any resulting common-clock dataset must be labelled approximate. If the available source cannot support a defensible synchronized state, narrow the prediction task rather than manufacture precision.

## 5. Prediction contract

Every training/evaluation row must be traceable to:
- race/session identity;
- driver identity;
- source observation identity;
- observation timestamp and its provenance;
- prediction timestamp or prediction checkpoint;
- lap/progress context;
- target definition and race winner label;
- feature-generation version.

The model estimates one winner distribution per race snapshot. For a valid snapshot with (N) eligible drivers:

[
0 leq P_i(mathrm{win}) leq 1,qquad
sum_{i=1}^{N}P_i(mathrm{win})=1
]

The eligibility rules must state how DNS, DNF, retired and not-yet-started drivers are represented at each prediction point. Do not infer these rules from a final classification alone.

### Leakage rules

- Features may only use data available by the prediction timestamp.
- Race outcome, final classification, post-race points, future laps, future pit stops and future weather observations must not enter predictors.
- Historical driver/team form must use strictly prior races.
- Transformations, imputers, encoders and calibration parameters are fitted on training data only.
- Grouped race snapshots must never be randomly split across train and test.
- Each experiment records the exact data file/version, feature list, code revision and temporal split.

## 6. Feature families to investigate

Feature families are hypotheses, not automatic inclusions.

| Family | Candidate information | Guardrail |
|---|---|---|
| Race state | Lap/progress, position, gap, interval, recent position/gap/interval changes | Only synchronized or explicitly approximate point-in-time observations |
| Pace | Previous lap time, rolling lap-time summaries, pace trend | Exclude invalid laps and ensure windows use prior observations only |
| Pit strategy | Stops completed, laps since pit, known stop history, estimated stint state | No future pit-stop leakage; distinguish observed from inferred tyre data |
| Driver form | Prior-race finishes, points, qualifying performance, reliability | Strictly prior-race joins |
| Constructor form | Prior team pace/results/reliability | Strictly prior-race joins; handle driver/team changes |
| Circuit context | Circuit identity and historical driver/team performance at circuit | Only historical results available before the race |
| Conditions | Weather, track and tyre state when sourced | Missing means unknown; never fabricate values |

The first pass should establish a clean baseline with current valid features, then add feature families one at a time. All-null columns are excluded until reliable data exists.

## 7. Modelling strategy

Start with interpretable, strong tabular baselines. Add complexity only when it improves validation performance and passes integrity checks.

### Required baselines
1. Current race leader / P1 heuristic.
2. Position-only baseline.
3. Regularized logistic or equivalent simple probabilistic baseline.
4. Current XGBoost implementation.

### Candidate challengers
- LightGBM and CatBoost classifiers, if dependencies and data contracts support them.
- Learning-to-rank model for within-race ordering, only with race-group-aware training and evaluation.
- Probability conversion/calibration layer, evaluated out-of-sample.
- Sequence models (LSTM/GRU/Transformer) only after a correctly synchronized sequence dataset exists and simpler models have been exhausted.

Do not select a model solely by accuracy. The primary metric remains log loss, with Brier score and calibration diagnostics. Winner accuracy, top-k and phase metrics are secondary diagnostics.

## 8. Evaluation protocol

Keep the current temporal protocol fixed unless a documented data-availability reason requires a revision:
- Train: 2023
- Validation: 2024
- Final test: 2025 (locked)

Use validation only for feature selection, model selection, thresholding and calibration design. The final 2025 holdout is not a tuning set.

Report metrics:
- race-snapshot multiclass log loss;
- multiclass Brier score (with the exact definition documented);
- calibration / reliability by probability bucket;
- winner accuracy;
- current-P1 baseline accuracy;
- coverage and number of scorable races/snapshots;
- metrics by race phase and prediction checkpoint;
- uncertainty intervals aggregated at race level where feasible.

Avoid misleading row-level confidence intervals because snapshots within one race are correlated. Report both snapshot-weighted and race-aggregated summaries where meaningful.

### Accuracy target

The aspirational 85% target must be tied to a specific prediction checkpoint/phase and a fixed evaluation unit. It is not a guaranteed outcome and must not replace proper probabilistic scoring. Early-race 85% accuracy is a different task from final-lap accuracy.

## 9. Phase 1 milestones and acceptance criteria

### P1.0 — Reproducible baseline and data audit
- [ ] Confirm repository state, dataset provenance and reproducible command.
- [ ] Run timestamp-spread audit on repaired race-state data.
- [ ] Audit timestamp nulls, duplicate keys, driver eligibility and target uniqueness.
- [ ] Document whether same-lap grouping is valid.
- [ ] Freeze 2025 holdout usage rules.

**Exit:** We can explain exactly what one evaluated snapshot represents and prove the information cutoff.

### P1.1 — Point-in-time snapshot contract
- [ ] Define the prediction checkpoint and eligible-driver policy.
- [ ] Implement synchronized/causal snapshot construction only if supported by source data.
- [ ] Add tests for as-of selection, no future observations, staleness and race boundaries.
- [ ] Preserve original raw and repaired datasets; write new derived artifacts separately.
- [ ] Add provenance and schema/version metadata.

**Exit:** Every model row has a defensible point-in-time meaning.

### P1.2 — Feature audit and baseline suite
- [ ] Validate each feature's source, availability time, null rate and semantics.
- [ ] Exclude all-null or unsupported fields.
- [ ] Compare P1, position-only, simple probabilistic and XGBoost baselines on 2024.
- [ ] Add feature-family ablations without touching 2025 for selection.
- [ ] Persist experiment configuration and metrics.

**Exit:** Baselines are reproducible and feature gains are measured without leakage.

### P1.3 — Model research
- [ ] Evaluate LightGBM/CatBoost and/or ranking models where appropriate.
- [ ] Compare probability calibration methods using validation-only fitting.
- [ ] Run controlled phase-specific analyses.
- [ ] Keep complex sequence models deferred until justified.

**Exit:** A candidate model beats meaningful baselines on validation log loss and calibration without integrity regressions.

### P1.4 — Locked final evaluation
- [ ] Freeze the candidate, features and calibration procedure.
- [ ] Evaluate once on the 2025 holdout.
- [ ] Compare against all baselines and report uncertainty/coverage.
- [ ] Document limitations and whether the 85% checkpoint target was achieved.

**Exit:** A reproducible model report exists; no claim exceeds the evidence.

## 10. Experiment record format

Every experiment must record:
- experiment ID and timestamp;
- git commit;
- dataset path, checksum or stable version;
- feature contract/version;
- train/validation years;
- model and hyperparameters;
- random seed;
- metrics and coverage;
- baseline deltas;
- known caveats and decision.

No experiment is considered complete if its data split, target definition or feature cutoff is unclear.

## 11. External references

- Jared-Chan, lap-by-lap model: https://github.com/Jared-Chan/f1ml
- Jared-Chan, model/data preparation notes: https://github.com/Jared-Chan/f1ml/blob/master/notebooks/README.md
- FormulAI: https://github.com/Akhil-0412/FormulAI
- FormulAI feature engineering: https://github.com/Akhil-0412/FormulAI/blob/main/features/fast_features.py
- FormulAI feature tests: https://github.com/Akhil-0412/FormulAI/tree/main/tests
- Mehmet's F1 race prediction simulator: https://github.com/mehmetkahya0/f1-race-prediction

These links are research references only. We do not copy their source code, data, branding or architecture wholesale. Re-check licenses and upstream data terms before reusing any code or redistributing any dataset.
