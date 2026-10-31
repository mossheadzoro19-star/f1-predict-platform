import numpy as np
import pandas as pd
import pytest

from f1_predict.modeling.pre_race_calibration import (
    run_calibration_experiment,
    validate_probability_frame,
)


def _probabilities():
    rows = []
    for season in (2021, 2022, 2023, 2024):
        for rnd in (1, 2, 3):
            for driver, probability, winner in (
                ("a", 0.8, 0),
                ("b", 0.2, 1),
            ):
                rows.append({
                    "season": season, "round": rnd, "model": "xgboost",
                    "train_end_year": season - 1, "driver_id": driver,
                    "actual_winner": winner,
                    "predicted_win_probability": probability,
                    "probability_rank": 1 if driver == "a" else 2,
                })
    return pd.DataFrame(rows)


def test_temperature_calibration_preserves_ranking_and_probability_sums():
    report, evaluation = run_calibration_experiment(_probabilities())
    assert report["fit_seasons"] == [2021, 2022]
    assert report["evaluation_seasons"] == [2023, 2024]
    assert evaluation["calibrated_win_probability"].between(0, 1).all()
    sums = evaluation.groupby(["season", "round"])[
        "calibrated_win_probability"
    ].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    for _, race in evaluation.groupby(["season", "round"]):
        assert race.loc[
            race["predicted_win_probability"].idxmax(), "driver_id"
        ] == race.loc[
            race["calibrated_win_probability"].idxmax(), "driver_id"
        ]
    for values in report["evaluation"].values():
        assert values["baseline"]["winner_accuracy"] == values[
            "temperature_scaled"
        ]["winner_accuracy"]


def test_calibration_rejects_overlapping_fit_and_evaluation_seasons():
    with pytest.raises(ValueError, match="must not overlap"):
        run_calibration_experiment(
            _probabilities(), fit_seasons=(2021, 2022),
            evaluation_seasons=(2022, 2024),
        )


def test_calibration_rejects_reserved_2025():
    frame = _probabilities()
    frame.loc[0, "season"] = 2025
    with pytest.raises(ValueError, match="2025 is reserved"):
        validate_probability_frame(frame)


def test_calibration_rejects_non_normalized_probabilities():
    frame = _probabilities()
    frame.loc[0, "predicted_win_probability"] = 0.7
    with pytest.raises(ValueError, match="must sum to one"):
        validate_probability_frame(frame)
