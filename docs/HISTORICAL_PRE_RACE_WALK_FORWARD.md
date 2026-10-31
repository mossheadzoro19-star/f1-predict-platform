# Historical pre-race walk-forward validation

## Purpose

Measure whether the pre-race winner model generalises to future seasons instead of relying on one train/validation boundary.

## Evaluation design

The evaluation uses expanding chronological windows:

| Training seasons | Validation season |
| --- | --- |
| 2018–2020 | 2021 |
| 2018–2021 | 2022 |
| 2018–2022 | 2023 |
| 2018–2023 | 2024 |

Each validation season is predicted by models trained only on earlier seasons. No race from a validation season is used to fit that fold's models.

The comparison includes:
- Uniform per-driver probabilities as a reference baseline.
- Logistic regression.
- XGBoost.

Log Loss is the primary selection metric. Multiclass Brier score and winner accuracy are secondary metrics. Top-label ECE is also reported as a diagnostic, not as a model-selection target by itself.

## Run

The input Parquet must contain the historical feature dataset through 2024:

```bash
python -m f1_predict.modeling.pre_race_walk_forward
```

Optional paths:

```bash
python -m f1_predict.modeling.pre_race_walk_forward \
  --data data/processed/features/pre_race_driver_features.parquet \
  --report-output reports/pre_race_walk_forward/metrics.json \
  --predictions-output reports/pre_race_walk_forward/race_predictions.csv \
  --driver-probabilities-output reports/pre_race_walk_forward/driver_probabilities.csv
```

Outputs:
- `reports/pre_race_walk_forward/metrics.json`: fold-level and combined out-of-fold metrics, training seasons and race counts.
- `reports/pre_race_walk_forward/race_predictions.csv`: one row per race and model, including the actual winner's predicted probability/rank and the predicted winner.
- `reports/pre_race_walk_forward/driver_probabilities.csv`: one row per driver, race and model, with the out-of-fold winning probability, actual-winner flag, rank and training cutoff. Probabilities sum to one within each race/model. This file is the input for the separate calibration experiment.

This command does not save or replace the baseline model artifact. It is an evaluation workflow, not a production-model training command.

## Data boundary

The command rejects any input containing a season after 2024. The 2025 season is reserved as a final holdout and must not be loaded, evaluated, or used for feature/model selection.

## Interpretation

Compare season-level Log Loss first. A model that performs well in one season but degrades sharply in another is not considered robust based on aggregate accuracy alone. Do not change features or hyperparameters from these results until the fold-level behaviour and race-level errors have been reviewed.
