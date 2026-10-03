from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from f1_predict.modeling.live_baseline import (
    LivePredictionMetrics,
    evaluate_live_probabilities,
    predict_live_probabilities,
)
from f1_predict.modeling.live_features import LiveFeatureContract
from f1_predict.modeling.live_split import chronological_live_race_split
from f1_predict.modeling.live_xgboost import fit_live_xgboost


EPSILON = 1e-6


@dataclass(frozen=True)
class CalibrationMetrics:
    """Probability calibration diagnostics."""

    expected_calibration_error: float
    maximum_calibration_error: float
    bins: int
    observations: int


class ProbabilityCalibrator:
    """Fit a one-dimensional calibration map on out-of-sample probabilities."""

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        self.method = method
        self._model: LogisticRegression | IsotonicRegression | None = None

    @staticmethod
    def _clip(probabilities: np.ndarray) -> np.ndarray:
        return np.clip(probabilities.astype(float), EPSILON, 1.0 - EPSILON)

    def fit(self, probabilities: pd.Series, targets: pd.Series) -> "ProbabilityCalibrator":
        x = self._clip(probabilities.to_numpy()).reshape(-1, 1)
        y = targets.to_numpy(dtype=int)

        if len(np.unique(y)) != 2:
            raise ValueError("Calibration data must contain both winner classes")

        if self.method == "sigmoid":
            # Platt-style scaling: fit logistic regression on the model's
            # log-odds, without class weighting, so the empirical frequency is
            # preserved rather than rebalanced.
            logit = np.log(x / (1.0 - x))
            model = LogisticRegression(
                C=1e6,
                solver="lbfgs",
                max_iter=2000,
                random_state=42,
            )
            model.fit(logit, y)
            self._model = model
        else:
            model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            model.fit(x.ravel(), y)
            self._model = model

        return self

    def transform(self, probabilities: pd.Series) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Calibrator must be fitted before transform")

        values = self._clip(probabilities.to_numpy())

        if self.method == "sigmoid":
            logit = np.log(values / (1.0 - values)).reshape(-1, 1)
            calibrated = self._model.predict_proba(logit)[:, 1]
        else:
            calibrated = self._model.predict(values)

        return pd.Series(calibrated, index=probabilities.index, name=probabilities.name)


def calibrate_prediction_frame(
    predictions: pd.DataFrame,
    calibrator: ProbabilityCalibrator,
) -> pd.DataFrame:
    """Apply a fitted binary calibrator and renormalize each lap snapshot."""
    required = {
        "session_key",
        "lap_number",
        "driver_number",
        "winner",
        "win_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")

    calibrated_raw = calibrator.transform(predictions["win_probability"])

    denominator = calibrated_raw.groupby(
        [predictions["session_key"], predictions["lap_number"]],
        sort=False,
    ).transform("sum")

    output = predictions.copy()
    output["calibrated_raw_probability"] = calibrated_raw
    output["win_probability"] = calibrated_raw / denominator.replace(0.0, np.nan)

    if output["win_probability"].isna().any():
        raise ValueError("Calibration produced an invalid probability distribution")

    return output


def calibration_metrics(
    predictions: pd.DataFrame,
    bins: int = 10,
) -> CalibrationMetrics:
    """Compute equal-width reliability diagnostics on driver-level probabilities."""
    if bins < 2:
        raise ValueError("bins must be at least 2")

    probabilities = predictions["win_probability"].to_numpy(dtype=float)
    targets = predictions["winner"].to_numpy(dtype=int)

    edges = np.linspace(0.0, 1.0, bins + 1)
    weighted_error = 0.0
    maximum_error = 0.0
    total = len(probabilities)

    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])

        count = int(mask.sum())
        if count == 0:
            continue

        confidence = float(probabilities[mask].mean())
        frequency = float(targets[mask].mean())
        error = abs(confidence - frequency)

        weighted_error += (count / total) * error
        maximum_error = max(maximum_error, error)

    return CalibrationMetrics(
        expected_calibration_error=float(weighted_error),
        maximum_calibration_error=float(maximum_error),
        bins=bins,
        observations=total,
    )


def _evaluate(
    predictions: pd.DataFrame,
) -> tuple[LivePredictionMetrics, CalibrationMetrics]:
    return (
        evaluate_live_probabilities(predictions),
        calibration_metrics(predictions),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate the live XGBoost winner probabilities."
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

    contract = LiveFeatureContract.default()

    # Model fitting is restricted to pre-calibration races.
    model = fit_live_xgboost(split.train, contract)

    validation_predictions = predict_live_probabilities(
        model,
        split.validation,
        contract,
    )
    validation_metrics, validation_calibration = _evaluate(validation_predictions)

    # The calibrator sees validation probabilities and labels only. The 2025
    # test race states remain untouched until the final evaluation.
    sigmoid = ProbabilityCalibrator("sigmoid").fit(
        validation_predictions["win_probability"],
        validation_predictions["winner"],
    )
    isotonic = ProbabilityCalibrator("isotonic").fit(
        validation_predictions["win_probability"],
        validation_predictions["winner"],
    )

    validation_sigmoid = calibrate_prediction_frame(validation_predictions, sigmoid)
    validation_isotonic = calibrate_prediction_frame(validation_predictions, isotonic)

    final_train = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = fit_live_xgboost(final_train, contract)
    test_predictions = predict_live_probabilities(
        final_model,
        split.test,
        contract,
    )

    test_uncalibrated_metrics, test_uncalibrated_calibration = _evaluate(test_predictions)
    test_sigmoid = calibrate_prediction_frame(test_predictions, sigmoid)
    test_isotonic = calibrate_prediction_frame(test_predictions, isotonic)

    test_sigmoid_metrics, test_sigmoid_calibration = _evaluate(test_sigmoid)
    test_isotonic_metrics, test_isotonic_calibration = _evaluate(test_isotonic)

    print(
        "LIVE XGBOOST CALIBRATION\n"
        f"TRAIN: {len(split.train)} rows, {split.train['session_key'].nunique()} races\n"
        f"CALIBRATION: {len(split.validation)} rows, {split.validation['session_key'].nunique()} races\n"
        f"TEST: {len(split.test)} rows, {split.test['session_key'].nunique()} races\n"
    )

    print(
        f"{'Validation':<16} "
        f"LogLoss={validation_metrics.log_loss:.4f} "
        f"Brier={validation_metrics.brier_score:.4f} "
        f"ECE={validation_calibration.expected_calibration_error:.4f} "
        f"MCE={validation_calibration.maximum_calibration_error:.4f}"
    )
    print(
        f"{'Val + Sigmoid':<16} "
        f"LogLoss={evaluate_live_probabilities(validation_sigmoid).log_loss:.4f} "
        f"Brier={evaluate_live_probabilities(validation_sigmoid).brier_score:.4f} "
        f"ECE={calibration_metrics(validation_sigmoid).expected_calibration_error:.4f} "
        f"MCE={calibration_metrics(validation_sigmoid).maximum_calibration_error:.4f}"
    )
    print(
        f"{'Val + Isotonic':<16} "
        f"LogLoss={evaluate_live_probabilities(validation_isotonic).log_loss:.4f} "
        f"Brier={evaluate_live_probabilities(validation_isotonic).brier_score:.4f} "
        f"ECE={calibration_metrics(validation_isotonic).expected_calibration_error:.4f} "
        f"MCE={calibration_metrics(validation_isotonic).maximum_calibration_error:.4f}"
    )

    print("\nTEST")
    print(
        f"{'XGBoost raw':<16} "
        f"Accuracy={test_uncalibrated_metrics.winner_accuracy:.4f} "
        f"LogLoss={test_uncalibrated_metrics.log_loss:.4f} "
        f"Brier={test_uncalibrated_metrics.brier_score:.4f} "
        f"ECE={test_uncalibrated_calibration.expected_calibration_error:.4f} "
        f"MCE={test_uncalibrated_calibration.maximum_calibration_error:.4f}"
    )
    print(
        f"{'XGB + Sigmoid':<16} "
        f"Accuracy={test_sigmoid_metrics.winner_accuracy:.4f} "
        f"LogLoss={test_sigmoid_metrics.log_loss:.4f} "
        f"Brier={test_sigmoid_metrics.brier_score:.4f} "
        f"ECE={test_sigmoid_calibration.expected_calibration_error:.4f} "
        f"MCE={test_sigmoid_calibration.maximum_calibration_error:.4f}"
    )
    print(
        f"{'XGB + Isotonic':<16} "
        f"Accuracy={test_isotonic_metrics.winner_accuracy:.4f} "
        f"LogLoss={test_isotonic_metrics.log_loss:.4f} "
        f"Brier={test_isotonic_metrics.brier_score:.4f} "
        f"ECE={test_isotonic_calibration.expected_calibration_error:.4f} "
        f"MCE={test_isotonic_calibration.maximum_calibration_error:.4f}"
    )


if __name__ == "__main__":
    main()
