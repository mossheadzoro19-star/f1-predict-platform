from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_predict.modeling.pre_race_model import (
    FEATURE_COLUMNS,
    RACE_KEYS,
    _race_probabilities,
    evaluate_predictions,
    run_experiment,
    validate_dataset,
)

DEFAULT_VALIDATION_YEARS = (2021, 2022, 2023, 2024)
MAX_ALLOWED_SEASON = 2024


def validate_walk_forward_frame(
    frame: pd.DataFrame,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
) -> None:
    """Validate the historical frame and enforce the untouched-2025 boundary."""
    validate_dataset(frame)
    seasons = set(frame["season"].astype(int).unique().tolist())
    if not seasons:
        raise ValueError("Walk-forward dataset contains no seasons")
    if max(seasons) > MAX_ALLOWED_SEASON:
        raise ValueError(
            "Walk-forward evaluation must not load seasons after 2024; "
            "2025 is reserved as the final holdout"
        )
    if tuple(sorted(set(validation_years))) != validation_years:
        raise ValueError("Validation years must be unique and in ascending order")
    for year in validation_years:
        if year not in seasons:
            raise ValueError(f"Validation season {year} is missing from the dataset")
        if not any(season < year for season in seasons):
            raise ValueError(f"No earlier training seasons are available for {year}")


def _prediction_audit(
    predictions: pd.DataFrame,
    model_name: str,
    train_end_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (season, round_number), race in predictions.groupby(RACE_KEYS, sort=True):
        race = race.sort_values(
            ["win_probability", "driver_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        actual = race.loc[race["winner"] == 1, "driver_id"].tolist()
        if len(actual) != 1:
            raise ValueError(
                f"Season {season}, round {round_number} must have exactly one winner"
            )
        actual_driver = str(actual[0])
        actual_row = race.loc[race["driver_id"] == actual_driver].iloc[0]
        predicted_driver = str(race.iloc[0]["driver_id"])
        rows.append(
            {
                "season": int(season),
                "round": int(round_number),
                "model": model_name,
                "train_end_year": int(train_end_year),
                "actual_winner_driver_id": actual_driver,
                "predicted_winner_driver_id": predicted_driver,
                "actual_winner_probability": float(actual_row["win_probability"]),
                "actual_winner_rank": int(
                    race["win_probability"].rank(method="min", ascending=False)
                    .loc[race["driver_id"] == actual_driver].iloc[0]
                ),
                "predicted_winner_probability": float(race.iloc[0]["win_probability"]),
                "field_size": int(len(race)),
            }
        )
    return pd.DataFrame(rows)


def run_walk_forward(
    frame: pd.DataFrame,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate each future season using only seasons strictly before it."""
    validate_walk_forward_frame(frame, validation_years)

    model_predictions: dict[str, list[pd.DataFrame]] = {
        "uniform": [],
        "logistic_regression": [],
        "xgboost": [],
    }
    fold_reports: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []

    for validation_year in validation_years:
        train_end_year = validation_year - 1
        _, artifacts = run_experiment(
            frame,
            train_end_year=train_end_year,
            validation_start_year=validation_year,
            validation_end_year=validation_year,
        )
        validation = frame.loc[frame["season"] == validation_year].copy()
        X_validation = validation[FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )

        fold_predictions = {
            "uniform": _race_probabilities(
                validation, np.ones(len(validation), dtype=float)
            ),
            "logistic_regression": _race_probabilities(
                validation,
                artifacts["logistic_regression"].predict_proba(X_validation)[:, 1],
            ),
            "xgboost": _race_probabilities(
                validation,
                artifacts["xgboost"].predict_proba(X_validation)[:, 1],
            ),
        }

        fold_metrics = {
            name: evaluate_predictions(prediction)
            for name, prediction in fold_predictions.items()
        }
        for name, prediction in fold_predictions.items():
            model_predictions[name].append(prediction)
            audit_frames.append(
                _prediction_audit(prediction, name, train_end_year)
            )

        fold_reports.append(
            {
                "validation_season": validation_year,
                "training_seasons": sorted(
                    frame.loc[frame["season"] <= train_end_year, "season"]
                    .astype(int).unique().tolist()
                ),
                "training_races": int(
                    frame.loc[frame["season"] <= train_end_year, RACE_KEYS]
                    .drop_duplicates().shape[0]
                ),
                "validation_races": int(
                    validation[RACE_KEYS].drop_duplicates().shape[0]
                ),
                "metrics": fold_metrics,
            }
        )

    combined_metrics = {
        name: evaluate_predictions(pd.concat(predictions, ignore_index=True))
        for name, predictions in model_predictions.items()
    }
    report = {
        "validation_years": list(validation_years),
        "validation_races": int(
            sum(fold["validation_races"] for fold in fold_reports)
        ),
        "primary_metric": "log_loss",
        "folds": fold_reports,
        "combined_out_of_fold_metrics": combined_metrics,
        "holdout_policy": (
            "Only seasons through 2024 are allowed. 2025 is reserved and "
            "must not be loaded, evaluated, or used for model selection."
        ),
    }
    audit = pd.concat(audit_frames, ignore_index=True).sort_values(
        ["season", "round", "model"]
    ).reset_index(drop=True)
    return report, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run expanding-window, chronological validation for pre-race "
            "winner prediction. The 2025 holdout is excluded."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/features/pre_race_driver_features.parquet"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/pre_race_walk_forward/metrics.json"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("reports/pre_race_walk_forward/race_predictions.csv"),
    )
    args = parser.parse_args()

    # The feature file is expected to contain 2018–2024 only. The validation
    # guard rejects any accidental inclusion of 2025 or later.
    frame = pd.read_parquet(args.data)
    report, audit = run_walk_forward(frame)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    audit.to_csv(args.predictions_output, index=False)

    print("PRE-RACE WALK-FORWARD VALIDATION")
    print(f"Validation seasons: {report['validation_years']}")
    print(f"Validation races: {report['validation_races']}")
    print("\nCombined out-of-fold metrics")
    print(
        f"{'Model':<24}{'Accuracy':>12}{'Log loss':>12}"
        f"{'Brier':>12}{'Top ECE':>12}"
    )
    print("-" * 72)
    for name, metrics in report["combined_out_of_fold_metrics"].items():
        print(
            f"{name:<24}{metrics['winner_accuracy']:>12.4f}"
            f"{metrics['log_loss']:>12.4f}"
            f"{metrics['multiclass_brier_score']:>12.4f}"
            f"{metrics['top_label_ece']:>12.4f}"
        )
    print("\nSeason-by-season results")
    for fold in report["folds"]:
        print(
            f"  {fold['validation_season']} "
            f"(trained through {fold['training_seasons'][-1]}):"
        )
        for name, metrics in fold["metrics"].items():
            print(
                f"    {name:<22} accuracy={metrics['winner_accuracy']:.4f} "
                f"log_loss={metrics['log_loss']:.4f} "
                f"brier={metrics['multiclass_brier_score']:.4f}"
            )
    print(f"\nMetrics report: {args.report_output}")
    print(f"Race predictions: {args.predictions_output}")
    print("2025 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
