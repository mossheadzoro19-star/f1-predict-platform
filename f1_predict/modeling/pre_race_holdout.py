from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_predict.modeling.pre_race_model import (
    FEATURE_COLUMNS,
    _make_xgboost,
    _race_probabilities,
    evaluate_predictions,
    uniform_predictions,
    validate_dataset,
)

TRAIN_CUTOFF_YEAR = 2024
HOLDOUT_YEAR = 2025


def validate_holdout_frame(frame: pd.DataFrame) -> None:
    """Require historical training data through 2024 and a separate 2025 holdout."""
    validate_dataset(frame)
    seasons = set(frame["season"].astype(int).unique())
    if HOLDOUT_YEAR not in seasons:
        raise ValueError("The 2025 final holdout season is missing")
    if max(seasons) != HOLDOUT_YEAR or any(
        season > HOLDOUT_YEAR for season in seasons
    ):
        raise ValueError("Holdout input must not contain seasons after 2025")
    training_seasons = {season for season in seasons if season <= TRAIN_CUTOFF_YEAR}
    if not training_seasons:
        raise ValueError("No pre-2025 training seasons are available")
    if max(training_seasons) != TRAIN_CUTOFF_YEAR:
        raise ValueError("Training data must extend through 2024")
    if frame.loc[frame["season"] == HOLDOUT_YEAR].empty:
        raise ValueError("The 2025 holdout partition is empty")


def run_final_holdout(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the frozen XGBoost baseline through 2024 and evaluate only on 2025."""
    validate_holdout_frame(frame)
    train = frame.loc[frame["season"] <= TRAIN_CUTOFF_YEAR].copy()
    holdout = frame.loc[frame["season"] == HOLDOUT_YEAR].copy()

    X_train = train[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X_holdout = holdout[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y_train = train["winner"].astype("int8")

    model = _make_xgboost()
    model.fit(X_train, y_train, verbose=False)
    model_probabilities = _race_probabilities(
        holdout, model.predict_proba(X_holdout)[:, 1]
    )
    baseline_probabilities = uniform_predictions(holdout)

    xgb_metrics = evaluate_predictions(model_probabilities)
    uniform_metrics = evaluate_predictions(baseline_probabilities)
    ordered = model_probabilities.copy()
    ordered["probability_rank"] = ordered.groupby(
        ["season", "round"]
    )["win_probability"].rank(method="min", ascending=False).astype(int)
    ordered = ordered.rename(
        columns={"winner": "actual_winner", "win_probability": "predicted_win_probability"}
    )
    ordered["model"] = "xgboost"
    ordered["train_end_year"] = TRAIN_CUTOFF_YEAR
    output = ordered[
        [
            "season", "round", "model", "train_end_year", "driver_id",
            "actual_winner", "predicted_win_probability", "probability_rank",
        ]
    ].sort_values(["season", "round", "probability_rank", "driver_id"])

    report = {
        "evaluation_type": "single final chronological holdout",
        "training_seasons": sorted(train["season"].astype(int).unique().tolist()),
        "training_races": int(train[["season", "round"]].drop_duplicates().shape[0]),
        "holdout_season": HOLDOUT_YEAR,
        "holdout_races": int(holdout[["season", "round"]].drop_duplicates().shape[0]),
        "feature_columns": FEATURE_COLUMNS,
        "model": {
            "name": "XGBoost",
            "configuration_source": "f1_predict.modeling.pre_race_model._make_xgboost",
            "calibration": "none; temperature scaling rejected",
        },
        "metrics": {
            "uniform_reference": uniform_metrics,
            "frozen_xgboost": xgb_metrics,
        },
        "holdout_policy": (
            "The model was fitted only on seasons through 2024. "
            "2025 is evaluated as a final holdout and must not be used for "
            "feature selection, hyperparameter tuning, or calibration."
        ),
    }
    return report, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time final 2025 holdout evaluation for the frozen pre-race "
            "XGBoost baseline. Do not tune from this result."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/processed/features/pre_race_driver_features_2018_2025_holdout.parquet"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/pre_race_final_holdout_2025/metrics.json"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("reports/pre_race_final_holdout_2025/driver_probabilities.csv"),
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    report, predictions = run_final_holdout(frame)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_csv(args.predictions_output, index=False)

    print("FINAL HISTORICAL MODEL HOLDOUT — 2025")
    print(f"Training seasons: {report['training_seasons']}")
    print(f"Training races: {report['training_races']}")
    print(f"Holdout races: {report['holdout_races']}")
    print("\nMetrics")
    for name, metrics in report["metrics"].items():
        print(
            f"{name:<20} accuracy={metrics['winner_accuracy']:.4f} "
            f"log_loss={metrics['log_loss']:.4f} "
            f"brier={metrics['multiclass_brier_score']:.4f} "
            f"top_ece={metrics['top_label_ece']:.4f}"
        )
    print(f"Metrics report: {args.report_output}")
    print(f"Driver probabilities: {args.predictions_output}")
    print("Do not tune the model using this final holdout.")


if __name__ == "__main__":
    main()
