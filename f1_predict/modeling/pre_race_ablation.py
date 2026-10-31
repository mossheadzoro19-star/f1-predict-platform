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
    ROW_KEYS,
    _make_logistic,
    _make_xgboost,
    _race_probabilities,
    evaluate_predictions,
    validate_dataset,
)

DEFAULT_VALIDATION_YEARS = (2021, 2022, 2023, 2024)
MAX_ALLOWED_SEASON = 2024

QUALIFYING_FEATURES = (
    "field_size",
    "grid",
    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "qualifying_gap_to_pole_seconds",
    "teammate_q3_gap_seconds",
)
EXPERIENCE_FEATURES = (
    "driver_prior_starts",
    "constructor_prior_starts",
)
CAREER_DOMINANCE_FEATURES = (
    "driver_prior_wins",
    "driver_prior_win_rate",
    "constructor_prior_wins",
    "constructor_prior_points",
    "constructor_prior_win_rate",
)
CAREER_FEATURES = EXPERIENCE_FEATURES + CAREER_DOMINANCE_FEATURES
RECENT_FEATURES = (
    "driver_finish_last_3",
    "driver_finish_last_5",
    "driver_points_last_3",
    "driver_points_last_5",
    "driver_weighted_form_5",
    "driver_prior_dnf_rate",
    "constructor_points_last_3",
    "constructor_points_last_5",
    "constructor_prior_dnf_rate",
)
CIRCUIT_FEATURES = (
    "driver_circuit_prior_starts",
    "driver_circuit_finish_last_3",
    "driver_circuit_finish_last_5",
)
RELATIVE_CONSTRUCTOR_FEATURES = (
    "constructor_points_share_last_3",
    "constructor_points_share_last_5",
    "constructor_points_share_trend_3v3",
)
RECENT_LINEAGE_FEATURES = (
    "lineage_points_last_3",
    "lineage_points_last_5",
    "lineage_points_share_last_3",
    "lineage_points_share_last_5",
)
FULL_LINEAGE_FEATURES = (
    "lineage_prior_starts",
    "lineage_prior_wins",
    "lineage_prior_points",
    "lineage_prior_win_rate",
)
ALL_LINEAGE_FEATURES = RECENT_LINEAGE_FEATURES + FULL_LINEAGE_FEATURES

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "full_baseline": tuple(FEATURE_COLUMNS),
    "no_career_dominance": tuple(
        feature
        for feature in FEATURE_COLUMNS
        if feature not in CAREER_DOMINANCE_FEATURES
    ),
    "recent_form_qualifying": (
        QUALIFYING_FEATURES + RECENT_FEATURES + CIRCUIT_FEATURES
    ),
    "qualifying_only": QUALIFYING_FEATURES,
    "history_only": RECENT_FEATURES + CAREER_FEATURES + CIRCUIT_FEATURES,
    "baseline_plus_relative_constructor": (
        tuple(FEATURE_COLUMNS) + RELATIVE_CONSTRUCTOR_FEATURES
    ),
    "no_career_dominance_plus_relative_constructor": (
        tuple(
            feature
            for feature in FEATURE_COLUMNS
            if feature not in CAREER_DOMINANCE_FEATURES
        )
        + RELATIVE_CONSTRUCTOR_FEATURES
    ),
    "baseline_plus_recent_lineage": (
        tuple(FEATURE_COLUMNS) + RECENT_LINEAGE_FEATURES
    ),
    "baseline_plus_full_lineage": (
        tuple(FEATURE_COLUMNS) + FULL_LINEAGE_FEATURES
    ),
    "baseline_plus_all_lineage": (
        tuple(FEATURE_COLUMNS) + ALL_LINEAGE_FEATURES
    ),
}


def validate_ablation_frame(
    frame: pd.DataFrame,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
) -> None:
    validate_dataset(frame)
    seasons = set(frame["season"].astype(int).unique().tolist())
    if not seasons:
        raise ValueError("Ablation dataset contains no seasons")
    if max(seasons) > MAX_ALLOWED_SEASON:
        raise ValueError(
            "Ablation evaluation must not load seasons after 2024; "
            "2025 is reserved as the final holdout"
        )
    if tuple(sorted(set(validation_years))) != validation_years:
        raise ValueError("Validation years must be unique and in ascending order")
    for year in validation_years:
        if year not in seasons:
            raise ValueError(f"Validation season {year} is missing from the dataset")
        if not any(season < year for season in seasons):
            raise ValueError(f"No earlier training seasons are available for {year}")


def _fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: tuple[str, ...],
    model_name: str,
) -> pd.DataFrame:
    X_train = train.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    X_validation = validation.loc[:, feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    y_train = train["winner"].astype("int8")

    if model_name == "logistic_regression":
        model = _make_logistic()
    elif model_name == "xgboost":
        model = _make_xgboost()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    scores = model.predict_proba(X_validation)[:, 1]
    return _race_probabilities(validation, scores)


def _audit_predictions(
    predictions: pd.DataFrame,
    feature_set: str,
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
        winner_rows = race.loc[race["winner"] == 1]
        if len(winner_rows) != 1:
            raise ValueError(
                f"Season {season}, round {round_number} must have exactly one winner"
            )
        actual = winner_rows.iloc[0]
        predicted = race.iloc[0]
        rows.append(
            {
                "season": int(season),
                "round": int(round_number),
                "feature_set": feature_set,
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
            }
        )
    return pd.DataFrame(rows)


def run_ablation(
    frame: pd.DataFrame,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare fixed feature sets on identical expanding chronological folds."""
    validate_ablation_frame(frame, validation_years)

    metric_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []

    for year in validation_years:
        train_end = year - 1
        train = frame.loc[frame["season"] <= train_end].copy()
        validation = frame.loc[frame["season"] == year].copy()
        fold_models: dict[str, dict[str, Any]] = {}

        for feature_set, columns in FEATURE_SETS.items():
            missing = set(columns) - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Feature set {feature_set} is missing columns: {sorted(missing)}"
                )
            for model_name in ("logistic_regression", "xgboost"):
                prediction = _fit_predict(
                    train, validation, columns, model_name
                )
                metrics = evaluate_predictions(prediction)
                fold_models.setdefault(feature_set, {})[model_name] = metrics
                metric_rows.append(
                    {
                        "validation_season": year,
                        "train_end_year": train_end,
                        "feature_set": feature_set,
                        "model": model_name,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key != "reliability_bins"
                        },
                    }
                )
                audit_frames.append(
                    _audit_predictions(
                        prediction, feature_set, model_name, train_end
                    )
                )
        fold_reports.append(
            {
                "validation_season": year,
                "train_end_year": train_end,
                "training_races": int(
                    train[RACE_KEYS].drop_duplicates().shape[0]
                ),
                "validation_races": int(
                    validation[RACE_KEYS].drop_duplicates().shape[0]
                ),
                "metrics": fold_models,
            }
        )

    audit = pd.concat(audit_frames, ignore_index=True).sort_values(
        ["feature_set", "model", "season", "round"]
    ).reset_index(drop=True)
    metrics_frame = pd.DataFrame(metric_rows)
    # Aggregate proper scoring rules from the per-race fold metrics, weighted
    # by race count. Every fold contains a disjoint set of validation races.
    summary_rows: list[dict[str, Any]] = []
    for (feature_set, model_name), group in metrics_frame.groupby(
        ["feature_set", "model"], sort=True
    ):
        weights = group["races"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "validation_races": int(weights.sum()),
                "winner_accuracy": float(
                    np.average(group["winner_accuracy"], weights=weights)
                ),
                "log_loss": float(np.average(group["log_loss"], weights=weights)),
                "multiclass_brier_score": float(
                    np.average(group["multiclass_brier_score"], weights=weights)
                ),
            }
        )

    report = {
        "validation_years": list(validation_years),
        "validation_races": int(
            sum(fold["validation_races"] for fold in fold_reports)
        ),
        "primary_metric": "log_loss",
        "models": ["logistic_regression", "xgboost"],
        "feature_sets": {
            name: list(columns) for name, columns in FEATURE_SETS.items()
        },
        "folds": fold_reports,
        "combined_out_of_fold_metrics": summary_rows,
        "holdout_policy": (
            "Only seasons through 2024 are allowed. 2025 is reserved and "
            "must not be loaded, evaluated, or used for model selection."
        ),
    }
    return report, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled feature ablations with chronological validation."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/features/pre_race_driver_features.parquet"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/pre_race_ablation/metrics.json"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/pre_race_ablation/feature_set_comparison.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("reports/pre_race_ablation/race_predictions.csv"),
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    report, audit = run_ablation(frame)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)

    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    comparison = pd.DataFrame(report["combined_out_of_fold_metrics"]).sort_values(
        ["model", "log_loss"]
    )
    comparison.to_csv(args.comparison_output, index=False)
    audit.to_csv(args.predictions_output, index=False)

    print("PRE-RACE FEATURE ABLATION")
    print(f"Validation seasons: {report['validation_years']}")
    print(f"Validation races: {report['validation_races']}")
    print("\nCombined out-of-fold comparison (lower Log Loss is better)")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nMetrics report: {args.report_output}")
    print(f"Feature comparison: {args.comparison_output}")
    print(f"Race predictions: {args.predictions_output}")
    print("2025 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
