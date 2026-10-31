# Historical Pre-Race Feature Ablation

## Purpose

Compare fixed pre-race feature groups under the same expanding-window chronological folds. The experiment investigates whether the baseline relies too heavily on career-level driver and constructor statistics, and how much predictive value comes from qualifying versus historical form.

## Run

```bash
python -m f1_predict.modeling.pre_race_ablation
```

Default input:

`data/processed/features/pre_race_driver_features.parquet`

Default validation seasons: 2021, 2022, 2023 and 2024. Each validation season is predicted using only seasons strictly before it. The module rejects any dataset containing a season after 2024. Do not change this guard or include 2025 during feature selection.

## Feature sets

- `full_baseline`: all existing model features.
- `no_career_dominance`: baseline without cumulative driver/constructor wins, win rates and constructor cumulative points. Prior driver and constructor starts remain as experience signals; recent form and circuit history remain.
- `recent_form_qualifying`: qualifying/grid context, recent driver and constructor form, DNF rates and driver circuit history; excludes all career start/win totals, win rates and cumulative constructor points.

These two groups are intentionally distinct: the first retains prior-start experience, while the second removes all career-total features.
- `qualifying_only`: field size, grid and qualifying position/time/gap features.
- `history_only`: driver/constructor career and recent history plus driver circuit history; excludes grid and qualifying features.
- `baseline_plus_relative_constructor`: full baseline plus rolling constructor points-share features over three and five prior races and a three-versus-preceding-three performance trend.
- `no_career_dominance_plus_relative_constructor`: removes career-dominance features from the baseline and adds the same relative constructor features.
- `baseline_plus_recent_lineage`: adds predecessor/current organizational-lineage points over the previous three and five races and points share over those windows.
- `baseline_plus_full_lineage`: adds lineage-level prior race starts, wins, cumulative points and win rate.
- `baseline_plus_all_lineage`: adds both recent and full lineage feature groups.

The experimental lineage mapping is explicit in `f1_predict.features.pre_race.CONSTRUCTOR_LINEAGE_IDS`: Force India/Racing Point/Aston Martin; Renault/Alpine; Toro Rosso/AlphaTauri/RB; and Alfa Romeo/Sauber. Unmapped constructors retain their own IDs. Red Bull Racing (`red_bull`) is not merged with the AlphaTauri/RB sister-team lineage. The mapping expresses organizational continuity for a hypothesis test; it does not imply unchanged technical performance. Source-ID constructor features remain unchanged.

Feature sets are defined explicitly in `f1_predict/modeling/pre_race_ablation.py`. All lineage columns are generated point-in-time from completed earlier races. No hyperparameter search occurs inside this model-comparison experiment.

## Models and evaluation

Each feature set is evaluated with the existing fixed Logistic Regression and XGBoost configurations. The same training/validation years are used for every feature set. Per-race driver scores are normalized into a race-level probability distribution using the existing baseline method.

Primary metric: mean multiclass Log Loss (lower is better). Secondary metrics: winner accuracy and multiclass Brier score (lower is better). Compare season-level results as well as pooled out-of-fold results; a pooled improvement that hides a severe season-specific regression is not sufficient.

This is a diagnostic ablation, not a final model-selection or hyperparameter-tuning exercise. The existing score-normalization architecture remains unchanged so that feature groups are the controlled variable.

## Outputs

- `reports/pre_race_ablation/metrics.json`: fold-level and aggregate metrics, plus exact feature-set definitions.
- `reports/pre_race_ablation/feature_set_comparison.csv`: aggregate comparison sorted by model and Log Loss.
- `reports/pre_race_ablation/race_predictions.csv`: per-race actual winner, predicted winner, winner probability and rank for each feature set/model.

## Interpretation cautions

- The 2021–2024 folds are a small historical sample; do not overstate statistical certainty.
- The 2023 season is unusually easy for the current baseline relative to 2021 and 2024. Inspect every fold.
- The experiment does not establish causality. Feature sets overlap by design, and removing a group can change interactions. The relative constructor features measure share of points awarded in prior races, not latent car pace or expected race pace.
- Do not use the reserved 2025 season for feature selection, tuning, debugging decisions based on model scores, or reporting interim performance.
- Do not interpret top-pick ECE as the primary selection metric.
