# F1 Predict — Race Intelligence Platform
## Engineering Handoff Report

**Repository:** https://github.com/mossheadzoro19-star/f1-predict-platform  
**Default branch:** `main`  
**Prepared:** 13 October 2026  
**Role / perspective:** Data engineering, ML evaluation, and production-readiness handoff  
**Purpose:** Preserve the objective, technical decisions, evidence, current state, blockers, feasibility, and next execution steps so work can continue without restarting or drifting.

---

## 1. Executive summary

F1 Predict is a race-intelligence platform intended to estimate **each driver's probability of winning**, before a race and as a race evolves. It combines historical performance with race-state evidence, while keeping prediction-time information separate from future outcomes.

The project has completed meaningful groundwork: historical Jolpica ingestion and normalization for 2018–2025, point-in-time pre-race features and a leakage audit, an original historical race-state dataset from OpenF1, and a substantial set of live-state ML experiments. A first XGBoost race-state model outperformed the earlier logistic baseline on the historical 2025 test set.

The current critical path is narrower than the full product: **generate and audit the FastF1-enriched 2023–2025 race-state dataset, then run the frozen OLD-versus-ENRICHED model comparison on the chronological 2025 holdout.** This has not been verified as complete. Until this is done, do not claim the enriched features improve the model.

The live-provider problem remains open. OpenF1 access encountered authentication/paid-access restrictions, and paid APIs are out of scope. A free-access live timing provider must be validated and isolated behind an adapter. The application must never present stale, replayed, fallback, or synthetic data as live.

### Current status in one view

| Workstream | Status | Evidence / caveat |
|---|---|---|
| Historical Jolpica ingestion, 2018–2025 | Completed previously | Raw ingestion and normalized season Parquet files were produced and audited |
| Pre-race point-in-time features | Completed previously | 3,458 rows, 173 races, 31 columns; leakage audit passed |
| Original race-state dataset | Completed previously | 77,322 rows, 70 races, 2023–2025; built while OpenF1 historical access was available |
| Live-state modeling experiments | Completed on old dataset | Logistic baseline, XGBoost, ablation, calibration, phase, trajectory, attribution |
| FastF1 race-state ingestion code | Present in repository | Runtime result and coverage have not been verified in this handoff |
| FastF1 enriched dataset, 2023–2025 | Pending verification | Roadmap still marks generation incomplete |
| OLD vs ENRICHED evaluation | Pending | Comparison script exists; no current result verified |
| Free live timing provider | Unresolved | Must remain free-access and licensing-aware |
| Replay, robust live adapter, freshness states, WebSocket | Not yet validated as production path | Keep explicitly scoped after data/model milestone |
| Current test suite / CI | Unverified for latest code | Last known checkpoint: 73 passed, 2 warnings, before FastF1 additions; GitHub returned no status checks/workflow runs for the inspected commit |
| Documentation consistency | Needs repair | README still describes OpenF1 as the live/historical adapter foundation |

---

## 2. Product objective and scope

### Primary objective

Build a credible, production-style F1 race intelligence platform that estimates a **probability distribution over drivers** and updates those probabilities as race evidence changes.

It is not merely a winner-label classifier or a dashboard displaying current positions.

### Two prediction modes

1. **Pre-race probability**
   - Historical driver and constructor form
   - Circuit-specific history
   - Qualifying and grid position when available at prediction time
   - Weather information only when it would have been available at the prediction timestamp
   - Historical knowledge is the prior; no future race outcomes may enter the feature set.

2. **Live-race probability**
   - Current observed position and timing gaps
   - Tyre compound/age, pit-stop history, lap pace, and weather when observed
   - Historical driver/constructor/circuit knowledge where the model design supports it
   - Recompute inference when new race-state observations arrive; **do not retrain the model every second**.

### Product requirements

- Show per-driver winner probabilities, not only a predicted winner.
- Distinguish `LIVE`, `REPLAY`, `FALLBACK`, `STALE`, and `DISCONNECTED` states.
- Expose data freshness and source/mode information.
- Never silently substitute old or replay data and label it live.
- Preserve a historical replay path for testing and demonstration.
- Keep model evaluation leakage-safe and chronological.
- Keep the system small until experiments justify extra complexity.

### Explicit exclusions / constraints

- No paid APIs as a mandatory dependency.
- No fake live data or deceptive status labels.
- No random row-level train/test split across correlated snapshots.
- No future information in prediction features.
- No deep-learning model just for novelty; introduce one only if a controlled experiment justifies it.
- No microservices or extra infrastructure without a clear need.
- Do not keep adding analysis modules while the agreed dataset/evaluation milestone remains unfinished.

---

## 3. Feasibility assessment

### Technical feasibility: feasible, with an important live-data dependency

The historical research and batch training path is feasible with the current Python stack and public historical data sources. The existing experiments demonstrate that a useful first race-state model can be trained and evaluated.

The full real-time product is **conditionally feasible** because free access to a reliable live timing feed, its availability during active sessions, and its permitted use/redistribution have not yet been confirmed. This is the primary external dependency. A free endpoint is not automatically a grant of commercial redistribution rights.

### Feasibility by layer

| Layer | Assessment | Main condition |
|---|---|---|
| Historical results and qualifying | High | Continue using Jolpica within its terms and rate limits |
| Rich historical race-state features | High / needs current validation | FastF1 ingestion must run successfully and pass coverage, timing, and leakage audits |
| Pre-race ML | Feasible baseline exists | Continue chronological evaluation and avoid post-race fields |
| Live-state ML | Feasible research baseline exists | Validate the new source's feature semantics and reproduce results on FastF1 data |
| Live timing ingestion | Conditional | Identify a free, stable provider and verify access/licensing |
| Replay | Feasible | Reuse recorded historical snapshots and label replay mode honestly |
| API and UI | Feasible | Stabilize data contracts and mode/freshness semantics first |
| Public/commercial deployment | Not yet cleared | Review provider/data licensing, secrets, operational reliability, and monitoring |

### Practical conclusion

Do not attempt to solve the entire production product in one step. The most defensible path is to finish the historical enriched-data experiment, lock a validated feature contract, then validate the free live source and build the live adapter against that contract.

---

## 4. Agreed architecture

```text
Historical sources
  ├── Jolpica ──> results, qualifying, schedule, driver/constructor metadata
  └── FastF1  ──> laps, session timing, tyres, weather, historical race state

Free live timing provider
  └── provider adapter ──> normalized live observations
                              |
                              v
                     raw/source-aware storage
                              |
                              v
                    normalization and audits
                              |
                              v
                  point-in-time feature pipeline
                              |
                 ┌────────────┴────────────┐
                 v                         v
           pre-race model             live-state model
                 |                         |
                 └────────────┬────────────┘
                              v
                    probability / calibration
                              |
                       FastAPI service
                              |
                   REST + later WebSocket
                              |
                  Pixel-grid F1 dashboard
```

### Architectural principles

- Source-specific logic stays in adapters.
- Normalize data before it enters feature engineering.
- Keep raw provider responses separate from derived ML datasets.
- Keep source identifiers and retrieval timestamps for auditability.
- Treat final race result as a target label only, never as a predictor for an earlier snapshot.
- Keep the old dataset and old feature contract available for controlled comparison.
- Do not add online retraining; run inference against newly observed race state.
- A failed feed must produce a clear stale/disconnected state, not a fabricated live response.

---

## 5. Data sources and access decisions

### Jolpica-F1 — historical structured data

Used for race schedule, results, qualifying, drivers, and constructors. Ingestion supports pagination and rate-limit handling. Historical seasons 2018–2025 were ingested and normalized.

Reference: https://github.com/jolpica/jolpica-f1

### FastF1 — richer historical session data

Chosen for lap/session timing, tyre information, and weather data to build a richer race-state dataset.

Reference: https://docs.fastf1.dev/  
Repository: https://github.com/theOehrly/Fast-F1

Important caveat: FastF1's software license does not automatically grant unrestricted redistribution rights for all upstream motorsport data. Review the applicable terms before public/commercial use.

### OpenF1 — no longer the required provider

During earlier attempts, active-session API access returned HTTP 401 and indicated authenticated access was required. The user has explicitly ruled out paid APIs. Therefore:
- Do not purchase OpenF1 access.
- Do not rely on OpenF1 for the project's required historical ingestion.
- Retain the prior OpenF1 dataset as an existing research benchmark, not as the long-term source dependency.
- Correct the README so it no longer presents OpenF1 as the required live/historical foundation.

Reference: https://openf1.org/  
Authentication context: https://openf1.org/auth.html

### Live timing provider — unresolved

Must be free-access, usable during an active race, and suitable under its access and licensing terms. Validate actual session access and reconnect/failure behavior before selecting it. Keep it behind an adapter to avoid coupling the rest of the project to one provider.

**Free access is not equivalent to permission for unrestricted commercial redistribution.**

---

## 6. Repository and implementation inventory

Repository: https://github.com/mossheadzoro19-star/f1-predict-platform

Key areas observed in the current repository:

- `f1_predict/data_sources/jolpica.py` — Jolpica client.
- `f1_predict/data_sources/openf1.py` — existing OpenF1 client retained for previous pipeline/context.
- `f1_predict/ingestion/historical.py` — historical ingestion entry point.
- `f1_predict/ingestion/normalize.py` — normalized historical outputs.
- `f1_predict/ingestion/audit.py`, `audit_normalized.py` — ingestion/data audits.
- `f1_predict/ingestion/openf1_race_state.py` — original race-state dataset pipeline and shared point-in-time dynamics transformation.
- `f1_predict/ingestion/fastf1_race_state.py` — FastF1 enriched race-state ingestion.
- `f1_predict/features/point_in_time.py` — pre-race historical features.
- `f1_predict/features/leakage.py` — leakage checks.
- `f1_predict/modeling/features.py` — pre-race feature contract.
- `f1_predict/modeling/split.py` — pre-race chronological split.
- `f1_predict/modeling/baseline.py` — pre-race baseline.
- `f1_predict/modeling/live_features.py` — old/enriched race-state feature contracts and selection.
- `f1_predict/modeling/live_split.py` — chronological live-state split.
- `f1_predict/modeling/live_baseline.py` — logistic baseline and probability metrics.
- `f1_predict/modeling/live_xgboost.py` — XGBoost live-state model.
- `f1_predict/modeling/live_ablation.py` — feature ablation.
- `f1_predict/modeling/live_calibration.py` — calibration evaluation.
- `f1_predict/modeling/live_phase_eval.py` — early/mid/late race evaluation.
- `f1_predict/modeling/live_trajectory.py` — predicted leader and probability trajectory diagnostics.
- `f1_predict/modeling/live_attribution.py` — feature attribution.
- `f1_predict/modeling/race_state_evaluation.py` — race-state metrics and phase/final-lap/confidence reports.
- `f1_predict/modeling/race_state_comparison.py` — OLD vs ENRICHED experiment entry point.
- `f1_predict/live/service.py`, `f1_predict/web.py`, `f1_predict/web_static/index.html` — existing service/UI foundation.
- `docs/` — PRD, architecture, API, database, design system, UX flows, security, testing, environment, deployment, observability, decisions, changelog, roadmap, and data provenance documents.

This is an inventory of known/current files, not a claim that every path is fully production-ready.

---

## 7. Completed work and evidence

### 7.1 Historical Jolpica ingestion and normalization

Historical seasons 2018–2025 were ingested. Normalized per-season outputs include:

```text
data/processed/jolpica/{year}/
  races.parquet
  drivers.parquet
  constructors.parquet
  qualifying.parquet
  race_results.parquet
```

Representative season counts from the previous audit:

| Season | Races | Race result rows | Qualifying rows |
|---|---:|---:|---:|
| 2018 | 21 | 420 | 420 |
| 2019 | 21 | 420 | 420 |
| 2021 | 22 | 440 | 440 |
| 2022 | 22 | 440 | 440 |
| 2023 | 22 | 440 | 440 |
| 2024 | 24 | 479 | 479 |
| 2025 | 24 | 479 | 479 |

2020 was also ingested, with 17 races and approximately 340 result rows. Counts vary with event/result completeness and should be confirmed from local audit output if needed. Rate limits were encountered during some ingestion attempts; retry handling was added and later runs succeeded.

### 7.2 Pre-race point-in-time features

Output previously produced:

`data/processed/features/race_driver_features.parquet`

- 3,458 rows
- 173 races
- 31 columns
- Seasons 2018–2025

Features include prior driver/constructor form, prior circuit results, qualifying, grid, and other race-time-appropriate predictors.

Known historical leakage/feature bugs fixed:
- Historical feature history resetting at season boundaries.
- Constructor history computed at driver-row granularity, which risked same-race teammate leakage.

Previously reported leakage audit:
- Driver race keys: PASS
- One winner per race: PASS
- First driver start: PASS
- Prior counts monotonic: PASS
- First-race history: PASS
- Prediction timestamps: PASS
- Overall leakage audit: PASS

A mutation-based temporal leakage test was also added.

### 7.3 Pre-race baseline

Chronological split:
- Train: seasons through 2023
- Validation: 2024
- Test: 2025

Baseline: logistic regression with median imputation and `class_weight="balanced"`.

| Split | Winner accuracy | Log loss | Brier |
|---|---:|---:|---:|
| Validation 2024 | 0.4583 | 1.8602 | 0.03941 |
| Test 2025 | 0.4583 | 1.5566 | 0.03669 |

Interpret metrics in context: winner-row classification is imbalanced and race-level probability quality matters. Do not compare Brier values across different formulations without checking normalization and aggregation.

### 7.4 Original OpenF1 race-state dataset

The original historical race-state dataset was previously generated for 2023–2025:

- 77,322 rows
- 70 races
- 32 drivers
- 20 columns

It used lap-based snapshots, observed positions/intervals/stints/weather, and the final result only as the eventual winner label.

Prior audit:
- One unique winner per race: passed
- 70 winner snapshots/races as expected in that dataset audit
- Duplicate snapshots: 0
- Global timestamp backward jumps: 0

Some interval fields were extremely sparse:
- `interval_laps_behind_car_ahead`: ~99.877% missing
- `interval_laps_behind_leader`: ~91.94% missing
- `interval_gap_to_leader`: ~10.85% missing
- `interval_interval`: ~2.98% missing
- Stint compound/tyre age: ~0.03% missing

This is the **old dataset**, not the pending FastF1 enriched dataset.

### 7.5 Live-state feature engineering

The old numeric contract includes lap number, position, gap/interval features, tyre age, and weather. The enriched contract adds:
- Position changes over 1, 3, and 5 laps
- Gap changes over 1, 3, and 5 laps
- Interval changes over 1, 3, and 5 laps
- Previous lap duration
- Previous three-lap mean pace
- Pit stops completed
- Laps since pit

Pit-stop timing semantics were designed so a pit on lap N only affects later snapshots, not snapshots before that pit becomes observable.

A feature-contract regression was fixed: validation now checks structural columns, and dataset selection reindexes the expected feature list so older fixtures missing enriched columns produce missing values for pipeline imputation.

### 7.6 Live-state logistic baseline and ablation

Chronological split:
- Train 2023
- Validation 2024
- Test 2025

Logistic baseline:
- Validation: accuracy 0.7344, log loss 1.1649, Brier 0.5451
- Test: accuracy 0.7768, log loss 1.1449, Brier 0.5751

Ablation results:

| Variant | Validation Log loss | Test Log loss | Test accuracy |
|---|---:|---:|---:|
| Position only | 1.3165 | 1.2647 | 0.7923 |
| Position + gap | 1.1997 | 1.1652 | 0.7923 |
| Position + gap + tyre | 1.1960 | 1.1506 | 0.7775 |
| Position + gap + tyre + weather | 1.1649 | 1.1449 | 0.7768 |
| All features | 1.1649 | 1.1449 | 0.7768 |

Interpretation: position is dominant; gap improves log loss; tyre adds a small log-loss improvement; weather provides a modest further improvement for the logistic baseline. Extra features did not improve this model in that experiment.

### 7.7 Live-state XGBoost

Previously evaluated configuration:
- 400 estimators
- max depth 4
- learning rate 0.05
- subsample 0.8
- column sample 0.8
- min child weight 5
- L2 regularization 2
- random state 42

Reported results:

| Split | Accuracy | Log loss | Brier |
|---|---:|---:|---:|
| Validation 2024 | 0.7344 | 1.0259 | 0.4263 |
| Test 2025 | 0.7486 | 0.8836 | 0.4224 |

This is the strongest previously tested live-state probability model on the old dataset. It is a benchmark, not proof that the new FastF1 enriched dataset will perform better.

### 7.8 Calibration experiment

Validation showed lower loss for isotonic calibration, but on the untouched 2025 test the raw model performed better:
- Raw test log loss: 0.8836, Brier 0.4224
- Sigmoid test log loss: 0.9342, Brier 0.4514
- Isotonic test log loss: 0.9133, Brier 0.4326

Decision: **do not deploy the calibrator based on this experiment**. Raw XGBoost remains the test-set winner. Do not select calibration solely on validation without respecting untouched test results.

### 7.9 Race-phase diagnostics

The model was substantially more predictive in late-race snapshots than early-race snapshots. On the old dataset, reported validation winner accuracy was approximately 0.596 early, 0.635 mid, and 0.846 late; test accuracy was approximately 0.674 early, 0.665 mid, and 0.826 late.

This is expected in part because race position becomes more informative later. Avoid overinterpreting the absolute Brier numbers: the existing race-state Brier is the sum of per-driver squared errors per snapshot, not a normalized multiclass Brier.

### 7.10 Probability trajectory diagnostics

On the old 2025 test set:
- 24 races
- First correct predicted leader reported as 1.0
- Persistent correct leader reported as 1.0
- Mean predicted leader changes: 3.67
- Mean maximum predicted winner probability: 0.9272
- Mean final winner probability: 0.8503
- Mean winner-probability volatility: 0.0545
- Mean snapshot entropy: 0.8838

The reported 0.9272 is **mean maximum predicted probability**, not accuracy. The trajectory results describe how the prediction evolves; they do not replace log loss, Brier, calibration, or holdout evaluation.

### 7.11 Feature attribution

Native XGBoost importance ranked current position first (~0.407), then gap to leader (~0.224), with rainfall and tyre compound among smaller contributors.

Permutation impact on the old 2025 test:
- Position increased log loss by ~0.0357 when permuted.
- Gap increased log loss by ~0.0032.
- Tyre age increased log loss by ~0.0020.
- Compound increased log loss by ~0.0015.
- Several weather variables had near-zero or slightly negative measured impacts.

Interpretation: the model relies strongly on current position, with measurable but much smaller contributions from timing gap and tyre information. Attribution is dataset/model-specific, not a universal causal claim.

### 7.12 API/UI foundation and design direction

FastAPI/Uvicorn service and a static dashboard foundation exist. Port 8000 is the intended Codespaces development port. The UI direction uses a dark telemetry aesthetic and Pixel Code-inspired pixel-grid styling, including a CSS/Unicode F1 car motif rather than heavy image assets.

UI reference requested by the user: https://qwerasd205.github.io/PixelCode/

The `Pixel Code` font family is referenced in CSS but is not known to be bundled; browsers may fall back to monospace. Do not bundle fonts until licensing is checked.

The live service has mode/fallback concepts, but production readiness of provider, replay, stale detection, and WebSocket behavior remains unverified.

---

## 8. Current FastF1 enriched ingestion: review notes and risks

Current file: `f1_predict/ingestion/fastf1_race_state.py`

The script:
- Enables FastF1 cache.
- Loads event schedules for requested seasons.
- Loads race sessions with `telemetry=False, weather=True, messages=False`.
- Extracts driver number, lap number, session time, position, lap duration, tyre compound/age, pit-in time, lap start date, weather, and final result.
- Builds lap-based gap and interval approximations.
- Calls the existing point-in-time dynamics helper.
- Labels the eventual winner from final session results.
- Writes `data/processed/features/fastf1_race_state_enriched.parquet`.
- Catches race-specific exceptions, prints `SKIP`, and continues.

### Risks to resolve before trusting this dataset

1. **Race coverage:** a successful process exit does not prove all expected races were loaded. Summarize every skipped event and define acceptable coverage.
2. **Gap semantics:** the current `interval_gap_to_leader` uses each driver's lap-end session time minus the minimum lap-end time for that lap. These are not necessarily same-instant gaps. Validate this feature against the intended prediction snapshot and FastF1 timing definitions; revise or rename it if needed.
3. **Prediction timestamp:** inspect `LapStartDate + LapTime` handling and fallback timestamps for timezone consistency and availability at prediction time.
4. **Feature observability:** ensure tyre age, lap duration, and position changes are aligned to the time the prediction would have been made. Confirm that shifts are within driver/session boundaries.
5. **Weather join:** backward as-of joining is directionally appropriate, but confirm both sides use compatible session-relative timestamps and no future weather record is included.
6. **Winner label:** final classification result is allowed only as a target. Verify exactly one winner per included race and handle red flags, shortened races, and unusual result records.
7. **Snapshot keys:** current evaluator groups by `session_key + lap_number`. Confirm there is one coherent snapshot per driver/lap and no unintended duplicate driver-lap rows.
8. **Old/new comparability:** ensure the old and enriched datasets use comparable races, targets, snapshot grouping, chronological splits, and evaluation code. If coverage differs, report it clearly and avoid misleading direct comparisons.
9. **Tests:** no dedicated `tests/test_fastf1_race_state.py` was found in the repository inspection. Add focused unit tests for transformations without requiring live network access.

Do not change many things at once. First reproduce the current data build, record coverage and defects, then make the smallest justified correction.

---

## 9. Evaluation protocol and metric interpretation

### Frozen temporal protocol

For race-state comparisons:
- Train: 2023
- Validation: 2024
- Test: 2025

For pre-race modeling:
- Train through 2023
- Validation 2024
- Test 2025

Do not use random row splits. Race snapshots within a race are highly correlated, and random splitting would leak event-specific context across partitions.

### Primary metric

**Log loss** is the primary probability-quality metric. It penalizes confidently wrong probabilities.

Secondary metrics:
- Winner accuracy
- Brier score, with its exact aggregation/normalization explicitly stated
- Scorable coverage
- Current leader accuracy
- Model-minus-current-leader accuracy
- Calibration diagnostics

Always compare against a simple baseline, including current P1 for live race-state prediction.

### Test discipline

- Use 2024 validation for model/feature choices.
- Keep 2025 untouched while choosing features and hyperparameters.
- Use 2025 once for the final comparison after decisions are locked.
- If the 2025 results have already been inspected in earlier experiments, describe it accurately as the existing historical holdout benchmark; do not pretend it is a brand-new unseen test. For the immediate OLD-vs-ENRICHED comparison, retain the agreed split for comparability and be transparent that previous test-driven experimentation occurred.

### Existing evaluator caveat

The race-state Brier score currently sums squared errors over drivers for each snapshot rather than calculating a normalized multiclass score. Keep it consistent for like-for-like comparison, label it accurately, and consider adding a standard normalized multiclass Brier as a separate metric rather than silently changing the historical metric.

---

## 10. Current blockers

### Blocker A — Enriched data build is unverified
The code is present, but a current successful build, expected race count, and data audit have not been verified.

### Blocker B — FastF1 timing/feature semantics need validation
The gap calculation is an approximation; point-in-time timestamp alignment must be confirmed before trusting model results.

### Blocker C — OLD vs ENRICHED result is missing
The evaluation script exists, but there is no verified current output demonstrating whether the enriched data improves log loss and other metrics.

### Blocker D — Free live timing source is unresolved
OpenF1's access restriction conflicts with the no-paid-API requirement. No replacement provider has yet been verified as usable and appropriate.

### Blocker E — Current tests/CI are unverified
The last known test checkpoint was 73 passed and 2 warnings before FastF1 additions. The inspected current commit had no status checks/workflow runs returned, and `.github/workflows/tests.yml` was not found. Do not report current tests as passing until run.

### Blocker F — README is stale
README still says OpenF1 is the live/historical foundation and describes OpenF1-based LIVE/REPLAY behavior. Align it with the agreed Jolpica + FastF1 historical plan and free live adapter strategy.

### Blocker G — Product-mode semantics need end-to-end validation
The dashboard/service must never confuse fallback, replay, stale, disconnected, and live data. Validate the mode and freshness contracts end to end before public deployment.

---

## 11. Immediate execution plan

### Milestone 1 — Re-establish repository health

Run in GitHub Codespaces:

```bash
python --version
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

Record the exact test count, failures, and warnings. Do not assume the earlier result is current.

### Milestone 2 — Generate FastF1 race-state dataset

```bash
python -m f1_predict.ingestion.fastf1_race_state \
  --start-year 2023 \
  --end-year 2025 \
  --output data/processed/features/fastf1_race_state_enriched.parquet
```

Expected target is around 70 race events across 2023–2025 if all races load; treat this as an expectation to verify, not a guaranteed result. Review every `SKIP` line.

### Milestone 3 — Audit the dataset

At minimum, report:
- Row count, race count, drivers per race
- Included/missing event list by season
- Exactly one unique winner per race
- Duplicate `session_key + lap_number + driver_number` rows
- Prediction timestamp nulls and ordering anomalies
- Feature missingness and all-NaN columns
- Winner-label validity
- Lap/driver continuity and dynamics alignment
- Pit-stop feature timing
- Weather join alignment
- A point-in-time leakage test

Do not run the comparison on a dataset that fails the audit.

### Milestone 4 — Run the controlled comparison

```bash
python -m f1_predict.modeling.race_state_comparison \
  --old-data data/processed/features/openf1_race_state.parquet \
  --enriched-data data/processed/features/fastf1_race_state_enriched.parquet
```

The old data file must exist in the Codespace. Raw/processed data may be ignored by Git, so a clean Codespace may need the historical data generation step first.

Report:
- Old vs enriched validation metrics
- Old vs enriched 2025 metrics
- Scorable coverage and race coverage
- Current P1 baseline
- Any dataset differences that make direct comparison imperfect

Use log loss as primary. Do not choose the winner based on accuracy alone.

### Milestone 5 — Repair documentation and tests

- Update README provider/mode claims.
- Add unit tests for FastF1 transformation logic and failure/skip reporting.
- Add a CI workflow for install + tests + lint if appropriate.
- Re-run the full suite after fixes.
- Update `docs/ROADMAP.md`, `docs/CHANGELOG.md`, and `docs/DATA_PROVENANCE.md` with actual evidence, not intended results.

### Milestone 6 — Validate free live timing

Only after the historical comparison:
- Test actual access during a live session.
- Check rate limits, authentication, reconnect behavior, timestamps, driver identifiers, and data freshness.
- Check terms/licensing.
- Implement a small provider adapter against the normalized race-state schema.
- Add explicit LIVE/STALE/DISCONNECTED behavior.
- Do not use the live provider to retrain every second.

### Milestone 7 — Replay and product hardening

- Implement historical replay from recorded snapshots.
- Test probability trajectories against recorded races.
- Validate REST responses and mode/freshness fields.
- Add WebSocket updates only after the polling/inference path is reliable.
- Keep the Pixel-grid UI lightweight and accessible.
- Review licensing, secrets, error handling, observability, and deployment before public launch.

---

## 12. Commands currently relevant

Install and test:

```bash
pip install -e ".[dev]"
pytest -q
```

Historical Jolpica ingestion example:

```bash
python -m f1_predict.ingestion.historical --year 2025 --source jolpica
```

FastF1 enriched dataset:

```bash
python -m f1_predict.ingestion.fastf1_race_state \
  --start-year 2023 \
  --end-year 2025 \
  --output data/processed/features/fastf1_race_state_enriched.parquet
```

OLD vs ENRICHED evaluation:

```bash
python -m f1_predict.modeling.race_state_comparison \
  --old-data data/processed/features/openf1_race_state.parquet \
  --enriched-data data/processed/features/fastf1_race_state_enriched.parquet
```

Run the web MVP:

```bash
make web
```

The intended local port is 8000.

---

## 13. Definition of done

### Historical enriched-model milestone
- [ ] Current tests run and result recorded
- [ ] FastF1 2023–2025 dataset generated
- [ ] Skipped races enumerated and explained
- [ ] Dataset integrity checks pass
- [ ] Exactly one eventual winner per included race
- [ ] Duplicate snapshot and timestamp audits pass
- [ ] Point-in-time feature timing validated
- [ ] OLD and ENRICHED models evaluated with the same protocol
- [ ] 2025 comparison reported transparently
- [ ] Useful and weak features documented
- [ ] README and roadmap aligned with evidence

### Live provider milestone
- [ ] Free-access provider tested during an active session
- [ ] Provider terms/licensing reviewed
- [ ] Adapter normalizes live observations
- [ ] Freshness, stale and disconnected states verified
- [ ] No fallback/replay output mislabeled as live

### Product milestone
- [ ] Replay path validated against historical sessions
- [ ] REST inference path tested
- [ ] WebSocket only after stable inference path
- [ ] Dashboard displays probabilities, source/mode, and freshness
- [ ] Secrets, errors, observability and deployment reviewed

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Historical provider changes/rate limits | Cache responses, respect terms/rate limits, retain source metadata |
| FastF1 skips events or returns incomplete laps | Emit coverage report and fail/flag below a defined coverage threshold |
| Incorrect gap/timing semantics | Validate against session timing documentation and known race snapshots |
| Leakage through timing or final results | Point-in-time tests, target/feature separation, chronological splits |
| Correlated snapshots inflate confidence | Evaluate at race-snapshot level and compare to current P1; document aggregation |
| Overfitting to 2025 | Use 2024 validation for choices and keep 2025 results transparent |
| Live source unavailable or restricted | Provider adapter, replay mode, explicit stale/disconnected status |
| Licensing risk | Review source terms before public/commercial redistribution |
| Documentation drifts from implementation | Update roadmap/provenance/changelog with each verified milestone |
| Premature infrastructure/ML complexity | Execute the current milestone before adding new modules or deep learning |

---

## 15. Handoff rules for the next engineer/agent

1. Continue from the current repository; do not restart or rewrite the project.
2. Preserve the no-paid-API requirement.
3. Use Jolpica + FastF1 for historical data unless evidence or terms require a change.
4. Do not label stale, fallback, replay, or synthetic output as LIVE.
5. Preserve chronological evaluation and point-in-time correctness.
6. Do not report old model metrics as results for the new FastF1 dataset.
7. Run the existing code and inspect real output before declaring a task complete.
8. Make the smallest correct fix; reuse current functions and libraries.
9. Prefer direct repository changes with tests over instructions that require manual copy/paste.
10. After every milestone, report: changed files, exact commands, actual test/data results, remaining risks, and the next single action.
11. Avoid unbounded experimentation and unnecessary features. The next action is data generation and audit, not UI polish or a new model family.

---

## 16. References

- Repository: https://github.com/mossheadzoro19-star/f1-predict-platform
- Jolpica-F1: https://github.com/jolpica/jolpica-f1
- FastF1 documentation: https://docs.fastf1.dev/
- FastF1 repository: https://github.com/theOehrly/Fast-F1
- OpenF1: https://openf1.org/
- FastAPI: https://fastapi.tiangolo.com/
- Pixel Code UI reference: https://qwerasd205.github.io/PixelCode/

---

## Final handoff statement

F1 Predict has completed the historical data foundation, point-in-time pre-race feature work, leakage auditing, and a meaningful first round of live-state model research. The existing XGBoost result is a useful benchmark. However, the FastF1 enriched dataset and OLD-versus-ENRICHED comparison are still unverified, the live provider remains unresolved under the no-paid-API constraint, and current tests/CI must be rerun.

**The next engineer should first run the current test suite, generate and audit the FastF1 2023–2025 dataset, and execute the controlled comparison.** Only after those results are recorded should work proceed to selecting a free live provider and hardening the live product path.
