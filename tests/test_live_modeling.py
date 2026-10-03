from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predict.modeling.live_baseline import (
    evaluate_live_probabilities,
    fit_live_baseline,
    predict_live_probabilities,
)
from f1_predict.modeling.live_split import chronological_live_race_split


def _rows(season: int, session: str, winner: int, laps: int = 2) -> list[dict]:
    rows = []
    for lap in range(1, laps + 1):
        for driver, position in [(1, 1), (2, 2), (3, 3)]:
            rows.append(
                {
                    "season": season,
                    "session_key": session,
                    "driver_number": driver,
                    "lap_number": lap,
                    "prediction_time": pd.Timestamp(
                        f"{season}-01-01T00:{lap:02d}:{driver:02d}Z"
                    ),
                    "winner": int(driver == winner),
                    "position_position": position,
                    "interval_gap_to_leader": float(position - 1),
                    "interval_laps_behind_leader": 0.0,
                    "interval_interval": 1.0,
                    "interval_laps_behind_car_ahead": 0.0,
                    "stint_tyre_age_at_start": float(lap),
                    "weather_air_temperature": 25.0,
                    "weather_track_temperature": 40.0,
                    "weather_humidity": 50.0,
                    "weather_rainfall": 0.0,
                    "weather_wind_speed": 2.0,
                    "stint_compound": "MEDIUM",
                }
            )
    return rows


def test_live_split_keeps_sessions_disjoint() -> None:
    frame = pd.DataFrame(
        _rows(2023, "s23", 1)
        + _rows(2024, "s24", 2)
        + _rows(2025, "s25", 3)
    )

    split = chronological_live_race_split(frame, 2023, 2024)

    assert split.train["season"].unique().tolist() == [2023]
    assert split.validation["season"].unique().tolist() == [2024]
    assert split.test["season"].unique().tolist() == [2025]
    assert set(split.train["session_key"]).isdisjoint(split.validation["session_key"])
    assert set(split.validation["session_key"]).isdisjoint(split.test["session_key"])


def test_live_split_rejects_empty_partition() -> None:
    frame = pd.DataFrame(_rows(2025, "s25", 1))
    with pytest.raises(ValueError, match="empty partition"):
        chronological_live_race_split(frame, 2023, 2024)


def test_live_baseline_predicts_one_distribution_per_lap() -> None:
    train = pd.DataFrame(_rows(2023, "train", 1, laps=3))
    test = pd.DataFrame(_rows(2024, "test", 2, laps=2))

    model = fit_live_baseline(train)
    predictions = predict_live_probabilities(model, test)

    sums = predictions.groupby(["session_key", "lap_number"])["win_probability"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert len(predictions) == len(test)


def test_live_evaluation_scores_valid_predictions() -> None:
    frame = pd.DataFrame(
        [
            {
                "session_key": "s1",
                "lap_number": 1,
                "driver_number": 1,
                "winner": 1,
                "win_probability": 0.8,
            },
            {
                "session_key": "s1",
                "lap_number": 1,
                "driver_number": 2,
                "winner": 0,
                "win_probability": 0.2,
            },
        ]
    )

    metrics = evaluate_live_probabilities(frame)

    assert metrics.winner_accuracy == 1.0
    assert metrics.log_loss == pytest.approx(-np.log(0.8))
    assert metrics.brier_score == pytest.approx((0.2**2 + 0.2**2))
    assert metrics.snapshots == 1
    assert metrics.races == 1


def test_live_evaluation_rejects_non_normalized_snapshot() -> None:
    frame = pd.DataFrame(
        [
            {"session_key": "s1", "lap_number": 1, "driver_number": 1, "winner": 1, "win_probability": 0.7},
            {"session_key": "s1", "lap_number": 1, "driver_number": 2, "winner": 0, "win_probability": 0.7},
        ]
    )

    with pytest.raises(ValueError, match="summing to 1"):
        evaluate_live_probabilities(frame)
