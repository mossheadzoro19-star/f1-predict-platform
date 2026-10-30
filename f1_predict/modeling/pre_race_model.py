from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "grid",
    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "driver_prior_starts",
    "driver_prior_wins",
    "driver_prior_win_rate",
    "driver_finish_last_3",
    "driver_finish_last_5",
    "driver_points_last_3",
    "driver_points_last_5",
    "constructor_prior_starts",
    "constructor_prior_wins",
    "constructor_prior_points",
    "constructor_prior_win_rate",
    "driver_circuit_prior_starts",
    "driver_circuit_finish_last_3",
]
RACE_KEYS = ["season", "round"]
ROW_KEYS = ["season", "round", "driver_id"]


def validate_dataset(frame: pd.DataFrame) -> None:
    required = set(ROW_KEYS + ["winner"] + FEATURE_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.duplicated(ROW_KEYS).any():
        raise ValueError("Dataset contains duplicate driver-race rows")
    if frame["winner"].isna().any() or not frame["winner"].isin([0, 1]).all():
        raise ValueError("winner must be a non-null binary target")
    race_winners = frame.groupby(RACE_KEYS)["winner"].sum()
    if not race_winners.eq(1).all():
        raise ValueError("Every race must contain exactly one winner")
    numeric = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Feature matrix contains infinite values")


def chronological_split(
    frame: pd.DataFrame,
    train_end_year: int = 2022,
    validation_start_year: int = 2023,
    validation_end_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_end_year < validation_start_year <= validation_end_year:
        raise ValueError("Invalid chronological year boundaries")
    train = frame[frame["season"] <= train_end_year].copy()
    validation = frame[
        frame["season"].between(validation_start_year, validation_end_year)
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("Chronological split produced an empty partition")
    if train["season"].max() >= validation["season"].min():
        raise AssertionError("Training contains a season not earlier than validation")
    return train, validation


def _race_probabilities(frame: pd.DataFrame, row_scores: np.ndarray) -> pd.DataFrame:
    if len(frame) != len(row_scores):
        raise ValueError("Score count does not match the number of driver rows")
    if not np.isfinite(row_scores).all() or (row_scores < 0).any():
        raise ValueError("Model scores must be finite and non-negative")

    result = frame.loc[:, ROW_KEYS + ["winner"]].copy()
    result["score"] = row_scores.astype(float)
    totals = result.groupby(RACE_KEYS)["score"].transform("sum")
    counts = result.groupby(RACE_KEYS)["driver_id"].transform("count")
    result["win_probability"] = np.where(
        totals > 0,
        result["score"] / totals,
        1.0 / counts,
    )
    return result


def uniform_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return _race_probabilities(
        frame,
        np.ones(len(frame), dtype=float),
    )


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    required = set(ROW_KEYS + ["winner", "win_probability"])
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    losses: list[float] = []
    briers: list[float] = []
    correct: list[bool] = []
    for key, race in predictions.groupby(RACE_KEYS, sort=True):
        if int(race["winner"].sum()) != 1:
            raise ValueError(f"Race {key} must have exactly one winner")
        probabilities = race["win_probability"].to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-8)
        ):
            raise ValueError(f"Race {key} probabilities must sum to one")

        winner_index = int(np.flatnonzero(race["winner"].to_numpy() == 1)[0])
        losses.append(-float(np.log(np.clip(probabilities[winner_index], 1e-15, 1.0))))
        target = race["winner"].to_numpy(dtype=float)
        briers.append(float(np.sum((probabilities - target) ** 2)))
        correct.append(int(np.argmax(probabilities)) == winner_index)

    return {
        "races": len(losses),
        "winner_accuracy": float(np.mean(correct)),
        "log_loss": float(np.mean(losses)),
        "multiclass_brier_score": float(np.mean(briers)),
    }


def _make_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )


def _make_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def run_experiment(
    frame: pd.DataFrame,
    train_end_year: int = 2022,
    validation_start_year: int = 2023,
    validation_end_year: int = 2024,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_dataset(frame)
    train, validation = chronological_split(
        frame, train_end_year, validation_start_year, validation_end_year
    )
    X_train = train[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X_validation = validation[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y_train = train["winner"].astype("int8")

    logistic = _make_logistic()
    logistic.fit(X_train, y_train)
    logistic_predictions = _race_probabilities(
        validation, logistic.predict_proba(X_validation)[:, 1]
    )

    xgboost = _make_xgboost()
    xgboost.fit(X_train, y_train, verbose=False)
    xgb_predictions = _race_probabilities(
        validation, xgboost.predict_proba(X_validation)[:, 1]
    )

    predictions = {
        "uniform": uniform_predictions(validation),
        "logistic_regression": logistic_predictions,
        "xgboost": xgb_predictions,
    }
    metrics = {
        name: evaluate_predictions(prediction)
        for name, prediction in predictions.items()
    }
    metadata = {
        "train_seasons": sorted(train["season"].unique().astype(int).tolist()),
        "validation_seasons": sorted(validation["season"].unique().astype(int).tolist()),
        "train_races": int(train[RACE_KEYS].drop_duplicates().shape[0]),
        "validation_races": int(validation[RACE_KEYS].drop_duplicates().shape[0]),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }
    artifacts = {
        "logistic_regression": logistic,
        "xgboost": xgboost,
        "feature_columns": FEATURE_COLUMNS,
        "metadata": metadata,
    }
    return metadata, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and chronologically evaluate pre-race winner models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/features/pre_race_driver_features.parquet"),
    )
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--validation-start-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2024)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/pre_race_winner_models.joblib"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/pre_race_baseline_metrics.json"),
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    metadata, artifacts = run_experiment(
        frame,
        args.train_end_year,
        args.validation_start_year,
        args.validation_end_year,
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, args.model_output)
    args.report_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("PRE-RACE WINNER MODEL — CHRONOLOGICAL VALIDATION")
    print(f"Training seasons: {metadata['train_seasons']}")
    print(f"Training races: {metadata['train_races']}")
    print(f"Validation seasons: {metadata['validation_seasons']}")
    print(f"Validation races: {metadata['validation_races']}")
    print("\nMetrics (one probability distribution per race)")
    print(f"{'Model':<24}{'Accuracy':>12}{'Log loss':>12}{'Brier':>12}")
    print("-" * 60)
    for name, values in metadata["metrics"].items():
        print(
            f"{name:<24}{values['winner_accuracy']:>12.4f}"
            f"{values['log_loss']:>12.4f}"
            f"{values['multiclass_brier_score']:>12.4f}"
        )
    print(f"\nModel artifact: {args.model_output}")
    print(f"Metrics report: {args.report_output}")
    print("2025 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
