from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predict.modeling.live_calibration import (
    ProbabilityCalibrator,
    calibration_metrics,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_key": ["s1"] * 4,
            "lap_number": [1] * 4,
            "driver_number": [1, 2, 3, 4],
            "winner": [1, 0, 0, 0],
            "win_probability": [0.7, 0.1, 0.1, 0.1],
        }
    )


def test_sigmoid_calibrator_maps_probabilities() -> None:
    probabilities = pd.Series(np.linspace(0.05, 0.95, 20))
    targets = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    calibrator = ProbabilityCalibrator("sigmoid").fit(probabilities, targets)
    calibrated = calibrator.transform(probabilities)

    assert np.isfinite(calibrated).all()
    assert ((calibrated > 0) & (calibrated < 1)).all()


def test_isotonic_calibrator_maps_probabilities() -> None:
    probabilities = pd.Series(np.linspace(0.05, 0.95, 20))
    targets = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    calibrator = ProbabilityCalibrator("isotonic").fit(probabilities, targets)
    calibrated = calibrator.transform(probabilities)

    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_calibration_metrics_are_zero_for_perfectly_matched_bin() -> None:
    predictions = pd.DataFrame(
        {
            "win_probability": [0.0, 0.0, 1.0, 1.0],
            "winner": [0, 0, 1, 1],
        }
    )
    metrics = calibration_metrics(predictions, bins=2)
    assert metrics.expected_calibration_error == pytest.approx(0.0)
    assert metrics.maximum_calibration_error == pytest.approx(0.0)


def test_calibrator_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both winner classes"):
        ProbabilityCalibrator("sigmoid").fit(
            pd.Series([0.2, 0.3]),
            pd.Series([0, 0]),
        )
