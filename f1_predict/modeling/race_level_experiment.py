from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_predict.modeling.pre_race_model import (
    FEATURE_COLUMNS, RACE_KEYS, _make_xgboost, _race_probabilities,
    evaluate_predictions, validate_dataset,
)
from f1_predict.modeling.race_level_model import ConditionalLogitRaceModel

DEFAULT_VALIDATION_YEARS = (2021, 2022, 2023, 2024)
MAX_ALLOWED_SEASON = 2024


def validate_experiment_frame(
    frame: pd.DataFrame,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
) -> None:
    validate_dataset(frame)
    seasons = set(frame["season"].astype(int).unique().tolist())
    if not seasons:
        raise ValueError("Experiment dataset contains no seasons")
    if max(seasons) > MAX_ALLOWED_SEASON:
        raise ValueError(
            "Race-level experiment must not load seasons after 2024; "
            "2025 is reserved as the final holdout"
        )
    if tuple(sorted(set(validation_years))) != validation_years:
        raise ValueError("Validation years must be unique and in ascending order")
    for year in validation_years:
        if year not in seasons:
            raise ValueError(f"Validation season {year} is missing from the dataset")
        if not any(season < year for season in seasons):
            raise ValueError(f"No earlier training seasons are available for {year}")


def _race_audit(
    predictions: pd.DataFrame, model_name: str, train_end_year: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (season, round_number), race in predictions.groupby(RACE_KEYS, sort=True):
        race = race.sort_values(
            ["win_probability", "driver_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        winners = race.loc[race["winner"] == 1]
        if len(winners) != 1:
            raise ValueError(
                f"Season {season}, round {round_number} must have exactly one winner"
            )
        actual, predicted = winners.iloc[0], race.iloc[0]
        rows.append({
            "season": int(season),
            "round": int(round_number),
            "model": model_name,
            "train_end_year": int(train_end_year),
            "actual_winner_driver_id": str(actual["driver_id"]),
            "predicted_winner_driver_id": str(predicted["driver_id"]),
            "actual_winner_probability": float(actual["win_probability"]),
            "actual_winner_rank": int(
                race["win_probability"].rank(method="min", ascending=False)
                .loc[race["driver_id"] == actual["driver_id"]].iloc[0]
            ),
            "predicted_winner_probability": float(predicted["win_probability"]),
            "field_size": int(len(race)),
        })
    return pd.DataFrame(rows)


def run_race_level_experiment(
    frame: pd.DataFrame,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
    l2_strength: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare grouped conditional likelihood with normalized binary XGBoost."""
    validate_experiment_frame(frame, validation_years)
    if l2_strength < 0:
        raise ValueError("l2_strength must be non-negative")

    names = ("uniform", "xgboost_normalized_binary", "conditional_logit")
    predictions_by_model: dict[str, list[pd.DataFrame]] = {
        name: [] for name in names
    }
    folds: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []

    for year in validation_years:
        train_end = year - 1
        train = frame.loc[frame["season"] <= train_end].copy()
        validation = frame.loc[frame["season"] == year].copy()
        x_train = train.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        x_validation = validation.loc[:, FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )

        xgb = _make_xgboost()
        xgb.fit(x_train, train["winner"].astype("int8"), verbose=False)
        xgb_predictions = _race_probabilities(
            validation, xgb.predict_proba(x_validation)[:, 1]
        )

        grouped_model = ConditionalLogitRaceModel(
            feature_columns=tuple(FEATURE_COLUMNS),
            l2_strength=l2_strength,
        ).fit(train)
        grouped_predictions = grouped_model.predict_race_probabilities(validation)
        uniform = _race_probabilities(validation, np.ones(len(validation), dtype=float))

        fold_predictions = {
            "uniform": uniform,
            "xgboost_normalized_binary": xgb_predictions,
            "conditional_logit": grouped_predictions,
        }
        fold_metrics = {
            name: evaluate_predictions(prediction)
            for name, prediction in fold_predictions.items()
        }
        for name, prediction in fold_predictions.items():
            predictions_by_model[name].append(prediction)
            audit_frames.append(_race_audit(prediction, name, train_end))

        folds.append({
            "validation_season": int(year),
            "train_end_year": int(train_end),
            "training_seasons": sorted(train["season"].astype(int).unique().tolist()),
            "training_races": int(train[RACE_KEYS].drop_duplicates().shape[0]),
            "validation_races": int(validation[RACE_KEYS].drop_duplicates().shape[0]),
            "conditional_logit_optimization": grouped_model.optimization_,
            "metrics": fold_metrics,
        })

    combined = {
        name: evaluate_predictions(pd.concat(predictions, ignore_index=True))
        for name, predictions in predictions_by_model.items()
    }
    report = {
        "experiment": "conditional_logit_vs_normalized_binary_xgboost",
        "validation_years": list(validation_years),
        "validation_races": int(sum(fold["validation_races"] for fold in folds)),
        "primary_metric": "log_loss",
        "secondary_metrics": ["multiclass_brier_score", "winner_accuracy"],
        "mathematics": {
            "utility": "u_ir = x_ir^T beta",
            "probability": "p_ir = exp(u_ir) / sum_{j in race r} exp(u_jr)",
            "objective": (
                "L(beta) = -(1/R) sum_r [u_w(r),r - "
                "log(sum_{j in r} exp(u_jr))] + (lambda/2)||beta||_2^2"
            ),
            "gradient": "(1/R) sum_r X_r^T(p_r-y_r) + lambda beta",
            "optimizer": "L-BFGS-B using the analytic gradient",
            "intercept": "No global intercept; a race-constant term cancels in softmax.",
        },
        "configuration": {
            "features": list(FEATURE_COLUMNS),
            "l2_strength": float(l2_strength),
            "max_iterations": 2000,
            "tolerance": 1e-9,
            "preprocessing": (
                "Median imputation and standardization fitted independently "
                "on each fold's training rows only."
            ),
        },
        "folds": folds,
        "combined_out_of_fold_metrics": combined,
        "holdout_policy": (
            "Only seasons through 2024 are allowed. 2025 is reserved as the "
            "final untouched holdout."
        ),
    }
    audit = pd.concat(audit_frames, ignore_index=True).sort_values(
        ["season", "round", "model"]
    ).reset_index(drop=True)
    return report, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare conditional-logit probabilities with normalized binary XGBoost."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/features/pre_race_driver_features.parquet"),
    )
    parser.add_argument("--l2-strength", type=float, default=1.0)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/race_level_experiment/metrics.json"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("reports/race_level_experiment/race_predictions.csv"),
    )
    args = parser.parse_args()
    frame = pd.read_parquet(args.data)
    report, audit = run_race_level_experiment(frame, l2_strength=args.l2_strength)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    audit.to_csv(args.predictions_output, index=False)

    print("RACE-LEVEL PROBABILITY ARCHITECTURE EXPERIMENT")
    print(f"Validation seasons: {report['validation_years']}")
    print(f"Validation races: {report['validation_races']}")
    print(f"Fixed L2 strength: {args.l2_strength}")
    print("\nCombined out-of-fold metrics (lower Log Loss/Brier is better)")
    for name, metrics in report["combined_out_of_fold_metrics"].items():
        print(
            f"{name:<30} accuracy={metrics['winner_accuracy']:.4f} "
            f"log_loss={metrics['log_loss']:.4f} "
            f"brier={metrics['multiclass_brier_score']:.4f}"
        )
    print("\nSeason-level Log Loss")
    for fold in report["folds"]:
        print(f"  {fold['validation_season']}:")
        for name, metrics in fold["metrics"].items():
            print(f"    {name:<30} {metrics['log_loss']:.4f}")
    print(f"\nMetrics report: {args.report_output}")
    print(f"Race predictions: {args.predictions_output}")
    print("2025 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
