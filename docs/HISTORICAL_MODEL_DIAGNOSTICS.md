# Historical Pre-Race Model Diagnostics

## Purpose

Generate an auditable validation report for the pre-race winner model. The diagnostics use the existing feature dataset and saved model artifact; they do not retrain or tune the model.

## Data boundaries

- Training history: seasons through 2022.
- Validation: 2023–2024.
- Final holdout: 2025 remains excluded.
- XGBoost feature importance is extracted from the already-trained model, which was fitted only on the training partition.

## Run

After building features and training the baseline model:

```bash
python -m f1_predict.modeling.pre_race_diagnostics
```

Optional arguments:

- `--data`: feature Parquet path.
- `--model`: saved joblib model artifact.
- `--output-dir`: diagnostics output directory.
- `--train-end-year`, `--validation-start-year`, `--validation-end-year`: chronological boundaries.

## Outputs

The default output directory is `reports/pre_race_diagnostics/`.

- `race_level_audit.csv`: one row per validation race, including actual winner, predicted top driver, actual winner probability, predicted top probability, top-two probability margin, actual winner rank, and whether the top prediction was correct.
- `xgboost_feature_importance.csv`: gain-based feature importance from the fitted XGBoost model, including normalized importance share.
- `diagnostics_summary.json`: overall and season-level metrics plus the ten validation races where the actual winner received the lowest probability.

## Interpretation cautions

- Feature importance describes how the fitted model used features; it does not establish causality.
- Gain importance can be biased toward some feature types and can distribute importance across correlated features.
- The race audit is diagnostic, not a license to tune against individual validation outcomes repeatedly.
- Validation has only 46 races. Treat season-level differences as signals to investigate, not conclusive proof of a stable trend.
- Do not inspect or load 2025 while choosing features or model settings. Use it only after the model and feature contract are frozen.
