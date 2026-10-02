from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from f1_predict.modeling.features import select_pre_race_dataset


@dataclass(frozen=True)
class RacePredictionMetrics:
    """Race-level metrics for winner-probability predictions."""

    winner_accuracy: float
    log_loss: float
    brier_score: float
    races: int


@dataclass(frozen=True)
class BaselineProbabilityModel:
    """A simple, reproducible binary win model converted to race probabilities."""

    pipeline: Pipeline


def build_logistic_baseline() -> BaselineProbabilityModel:
    """Build a leakage-safe baseline with median imputation and logistic regression."""
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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
    return BaselineProbabilityModel(pipeline=pipeline)


def fit_baseline(
    train: pd.DataFrame,
) -> BaselineProbabilityModel:
    """Fit the baseline using only the training partition."""
    X_train, y_train = select_pre_race_dataset(train)
    model = build_logistic_baseline()
    model.pipeline.fit(X_train, y_train)
    return model


def predict_race_probabilities(
    model: BaselineProbabilityModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Predict P(P1) for each driver and normalize probabilities within each race."""
    X, _ = select_pre_race_dataset(frame)
    row_probability = model.pipeline.predict_proba(X)[:, 1]

    result = frame.loc[:, ["season", "round", "driver_id"]].copy()
    result["raw_win_probability"] = row_probability

    def normalize(group: pd.DataFrame) -> pd.DataFrame:
        total = group["raw_win_probability"].sum()
        if total <= 0 or not np.isfinite(total):
            group["win_probability"] = 1.0 / len(group)
        else:
            group["win_probability"] = group["raw_win_probability"] / total
        return group

    result["race_probability_total"] = result.groupby(
        ["season", "round"]
    )["raw_win_probability"].transform("sum")

    result["win_probability"] = np.where(
        result["race_probability_total"] > 0
        & np.isfinite(result["race_probability_total"]),
        result["raw_win_probability"] / result["race_probability_total"],
        1.0 / result.groupby(["season", "round"])["driver_id"].transform("count"),
    )

    return result.drop(columns=["race_probability_total"])


def evaluate_race_probabilities(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> RacePredictionMetrics:
    """Evaluate one P(P1) distribution per race using log loss, Brier score and accuracy."""
    required_prediction = {
        "season",
        "round",
        "driver_id",
        "win_probability",
    }
    required_actual = {"season", "round", "driver_id", "winner"}

    missing_prediction = required_prediction - set(predictions.columns)
    missing_actual = required_actual - set(actuals.columns)

    if missing_prediction:
        raise ValueError(f"Missing prediction columns: {sorted(missing_prediction)}")
    if missing_actual:
        raise ValueError(f"Missing actual columns: {sorted(missing_actual)}")

    merged = predictions.merge(
        actuals.loc[:, ["season", "round", "driver_id", "winner"]],
        on=["season", "round", "driver_id"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("No matching driver-race rows for evaluation")

    race_metrics: list[tuple[float, float, bool]] = []
    for _, race in merged.groupby(["season", "round"], sort=True):
        if int(race["winner"].sum()) != 1:
            raise ValueError("Each evaluated race must contain exactly one winner")

        probabilities = race["win_probability"].to_numpy(dtype=float)
        winner_mask = race["winner"].to_numpy(dtype=bool)
        winner_probability = float(probabilities[winner_mask][0])

        clipped = float(np.clip(winner_probability, 1e-15, 1.0))
        log_loss = -np.log(clipped)
        brier = float(np.mean((probabilities - winner_mask.astype(float)) ** 2))
        predicted_driver = race.iloc[int(np.argmax(probabilities))]["driver_id"]
        actual_driver = race.loc[race["winner"].eq(1), "driver_id"].iloc[0]
        race_metrics.append((log_loss, brier, predicted_driver == actual_driver))

    metrics = np.asarray(race_metrics, dtype=object)
    return RacePredictionMetrics(
        winner_accuracy=float(np.mean(metrics[:, 2].astype(bool))),
        log_loss=float(np.mean(metrics[:, 0].astype(float))),
        brier_score=float(np.mean(metrics[:, 1].astype(float))),
        races=len(race_metrics),
    )
