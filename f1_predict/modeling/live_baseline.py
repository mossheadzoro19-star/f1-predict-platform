from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from f1_predict.modeling.live_features import (
    LiveFeatureContract,
    select_live_dataset,
)
from f1_predict.modeling.live_split import chronological_live_race_split


SNAPSHOT_KEYS = ["session_key", "lap_number"]


@dataclass(frozen=True)
class LivePredictionMetrics:
    """Metrics for winner probabilities at each lap snapshot."""

    winner_accuracy: float
    log_loss: float
    brier_score: float
    snapshots: int
    races: int


def build_live_logistic_baseline(
    contract: LiveFeatureContract | None = None,
) -> Pipeline:
    """Build a transparent point-in-time logistic regression baseline."""
    contract = contract or LiveFeatureContract.default()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(contract.numeric_features)),
            ("categorical", categorical_pipeline, list(contract.categorical_features)),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def fit_live_baseline(
    train: pd.DataFrame,
    contract: LiveFeatureContract | None = None,
) -> Pipeline:
    """Fit the live baseline using only the supplied historical race snapshots."""
    X, y = select_live_dataset(train, contract)
    model = build_live_logistic_baseline(contract)
    model.fit(X, y)
    return model


def _normalize_within_snapshot(
    probabilities: pd.Series,
    frame: pd.DataFrame,
) -> pd.Series:
    """Convert independent binary probabilities into a race-state distribution."""
    denominator = probabilities.groupby(
        [frame["session_key"], frame["lap_number"]],
        sort=False,
    ).transform("sum")

    normalized = probabilities / denominator.replace(0.0, np.nan)

    # A fitted classifier should produce positive probabilities, but retaining
    # this guard makes the output contract explicit if a future model changes.
    return normalized.fillna(1.0 / frame.groupby(SNAPSHOT_KEYS)["driver_number"].transform("count"))


def predict_live_probabilities(
    model: Pipeline,
    frame: pd.DataFrame,
    contract: LiveFeatureContract | None = None,
) -> pd.DataFrame:
    """Predict and normalize driver win probabilities for every lap snapshot."""
    X, _ = select_live_dataset(frame, contract)
    raw = pd.Series(
        model.predict_proba(X)[:, 1],
        index=frame.index,
        dtype="float64",
        name="raw_win_probability",
    )

    predictions = frame[
        [
            "season",
            "session_key",
            "driver_number",
            "lap_number",
            "prediction_time",
            "winner",
        ]
    ].copy()
    predictions["raw_win_probability"] = raw
    predictions["win_probability"] = _normalize_within_snapshot(raw, frame)
    return predictions.sort_values(
        ["session_key", "lap_number", "win_probability"],
        ascending=[True, True, False],
        ignore_index=True,
    )


def evaluate_live_probabilities(predictions: pd.DataFrame) -> LivePredictionMetrics:
    """Evaluate race-state probabilities at the lap-snapshot level."""
    required = {
        "session_key",
        "lap_number",
        "driver_number",
        "winner",
        "win_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    rows: list[tuple[float, float, float]] = []

    skipped_snapshots = 0

    for _, group in predictions.groupby(SNAPSHOT_KEYS, sort=False):
        group = group.copy()
        probability_sum = float(group["win_probability"].sum())
        if not np.isfinite(probability_sum) or not np.isclose(probability_sum, 1.0, atol=1e-6):
            raise ValueError("Each lap snapshot must have probabilities summing to 1")

        actual = group.loc[group["winner"] == 1, "driver_number"]
        if len(actual) == 0:
            # A driver can disappear from the OpenF1 lap-state stream after a
            # terminal event. Such a snapshot cannot be scored against the
            # eventual winner because the winner is no longer in the candidate
            # field. Keep it in predictions, but exclude it from supervised
            # snapshot metrics and report the resulting coverage.
            skipped_snapshots += 1
            continue
        if len(actual) != 1:
            raise ValueError("Each evaluated snapshot must contain at most one eventual winner")

        winner_driver = actual.iloc[0]
        winner_probability = float(
            group.loc[group["driver_number"] == winner_driver, "win_probability"].iloc[0]
        )
        predicted_driver = group.loc[group["win_probability"].idxmax(), "driver_number"]

        # Multiclass Brier score: sum of squared probability errors across the
        # drivers in this snapshot. Averaging below gives every snapshot equal weight.
        target = group["winner"].to_numpy(dtype=float)
        probability = group["win_probability"].to_numpy(dtype=float)
        brier = float(np.sum((probability - target) ** 2))

        rows.append(
            (
                float(predicted_driver == winner_driver),
                -float(np.log(np.clip(winner_probability, 1e-15, 1.0))),
                brier,
            )
        )

    if not rows:
        raise ValueError("No scorable lap snapshots available for evaluation")

    metrics = np.asarray(rows, dtype=float)
    coverage = len(rows) / (len(rows) + skipped_snapshots)

    print(
        f"  scorable_coverage = {coverage:.4f} "
        f"({len(rows)}/{len(rows) + skipped_snapshots} snapshots)"
    )

    return LivePredictionMetrics(
        winner_accuracy=float(metrics[:, 0].mean()),
        log_loss=float(metrics[:, 1].mean()),
        brier_score=float(metrics[:, 2].mean()),
        snapshots=len(rows),
        races=int(predictions["session_key"].nunique()),
    )


def _format_metrics(label: str, metrics: LivePredictionMetrics) -> str:
    return (
        f"{label}\n"
        f"  winner_accuracy = {metrics.winner_accuracy:.4f}\n"
        f"  log_loss        = {metrics.log_loss:.4f}\n"
        f"  brier_score     = {metrics.brier_score:.4f}\n"
        f"  snapshots       = {metrics.snapshots}\n"
        f"  races           = {metrics.races}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the first live-race ML baseline."
    )
    parser.add_argument(
        "--data",
        default="data/processed/features/openf1_race_state.parquet",
    )
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2024)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)

    split = chronological_live_race_split(
        frame,
        train_end_year=args.train_end_year,
        validation_end_year=args.validation_end_year,
    )

    print(
        "LIVE TEMPORAL SPLIT\n"
        f"TRAIN: {len(split.train)} rows, {split.train['session_key'].nunique()} races\n"
        f"VALIDATION: {len(split.validation)} rows, {split.validation['session_key'].nunique()} races\n"
        f"TEST: {len(split.test)} rows, {split.test['session_key'].nunique()} races"
    )

    validation_model = fit_live_baseline(split.train)
    validation_predictions = predict_live_probabilities(
        validation_model,
        split.validation,
    )
    validation_metrics = evaluate_live_probabilities(validation_predictions)
    print(_format_metrics("VALIDATION", validation_metrics))

    # Freeze the feature/model design after validation, then refit on all
    # pre-test seasons. The 2025 test set remains untouched by model selection.
    final_train = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = fit_live_baseline(final_train)
    test_predictions = predict_live_probabilities(final_model, split.test)
    test_metrics = evaluate_live_probabilities(test_predictions)
    print(_format_metrics("TEST", test_metrics))


if __name__ == "__main__":
    main()
