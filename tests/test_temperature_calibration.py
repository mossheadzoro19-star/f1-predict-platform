from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predict.modeling.temperature_calibration_experiment import (
    _fit_temperature,
    _multiclass_log_loss,
    _temperature_scale_predictions,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"session_key": "race1", "lap_number": 1, "driver_number": 1, "winner": 1, "win_probability": 0.8},
            {"session_key": "race1", "lap_number": 1, "driver_number": 2, "winner": 0, "win_probability": 0.2},
            {"session_key": "race2", "lap_number": 1, "driver_number": 1, "winner": 0, "win_probability": 0.6},
            {"session_key": "race2", "lap_number": 1, "driver_number": 2, "winner": 1, "win_probability": 0.4},
        ]
    )


def test_temperature_scaling_preserves_snapshot_probability_sums() -> None:
    scaled = _temperature_scale_predictions(_predictions(), temperature=1.7)
    sums = scaled.groupby(["session_key", "lap_number"])["win_probability"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert np.isfinite(scaled["win_probability"]).all()


def test_temperature_one_preserves_probabilities() -> None:
    frame = _predictions()
    scaled = _temperature_scale_predictions(frame, temperature=1.0)
    assert np.allclose(scaled["win_probability"], frame["win_probability"])


def test_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        _temperature_scale_predictions(_predictions(), temperature=0.0)


def test_log_loss_scores_winner_probabilities() -> None:
    assert _multiclass_log_loss(_predictions()) == pytest.approx(
        (-np.log(0.8) - np.log(0.4)) / 2
    )


def test_fitted_temperature_is_finite_and_positive() -> None:
    temperature = _fit_temperature(_predictions())
    assert np.isfinite(temperature)
    assert temperature > 0
