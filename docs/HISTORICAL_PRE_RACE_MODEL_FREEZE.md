# Historical pre-race model freeze

## Decision

The candidate for the final historical holdout is the existing XGBoost baseline from `f1_predict.modeling.pre_race_model`.

The selection is based on the expanding walk-forward comparison through 2024:

- XGBoost aggregate Log Loss: 1.1797.
- Logistic regression aggregate Log Loss: 1.3449.
- XGBoost aggregate multiclass Brier score: 0.5677.
- Logistic regression aggregate multiclass Brier score: 0.6075.

XGBoost is the candidate because Log Loss is the primary metric and it also has the better aggregate Brier score. Its weak 2024 fold remains a material limitation and must be disclosed.

## Frozen feature set

The candidate uses the exact `FEATURE_COLUMNS` list in `f1_predict/modeling/pre_race_model.py` (27 features). Experimental constructor-relative and constructor-lineage features are excluded.

## Frozen estimator

Use `_make_xgboost()` in `f1_predict/modeling/pre_race_model.py` without changing its parameters. No new feature selection, hyperparameter search, or probability calibration is permitted before the final holdout evaluation.

Temperature scaling was rejected: it improved 2023 but degraded 2024, including when fit on 2021–2023 and evaluated on 2024. The uncalibrated XGBoost output remains the candidate.

## Final 2025 holdout protocol

1. Build a separate point-in-time feature file for 2018–2025 from normalized Jolpica data:
   ```bash
   python -m f1_predict.features.pre_race \
     --start-year 2018 \
     --end-year 2025 \
     --output data/processed/features/pre_race_driver_features_2018_2025_holdout.parquet
   ```
2. Run the holdout evaluator exactly after the candidate has been frozen:
   ```bash
   python -m f1_predict.modeling.pre_race_holdout
   ```
3. The evaluator trains XGBoost only on rows from seasons through 2024 and scores only 2025. It compares against a uniform reference.
4. Do not use the 2025 result to tune features, hyperparameters, or calibration and then continue calling it a final holdout.

Outputs:
- `reports/pre_race_final_holdout_2025/metrics.json`
- `reports/pre_race_final_holdout_2025/driver_probabilities.csv`

The 2025 feature rows are generated chronologically: each race's features are frozen before that race's results update historical state. Earlier 2025 race results may inform later 2025 pre-race rows, as they would in real use; no future 2025 results may enter an earlier race's features.

## Interpretation

A poor 2025 result is not a reason to silently tune against 2025. It is evidence that the frozen candidate did not generalize to that holdout. Any redesign after inspecting it requires a new, explicitly named development/evaluation protocol and a different untouched test period.
