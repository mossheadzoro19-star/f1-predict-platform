import pandas as pd
import pytest

from f1_predict.modeling.live_features import (
    LiveFeatureContract,
    select_live_dataset,
    summarize_live_dataset,
    validate_race_snapshot_targets,
)


def _frame() -> pd.DataFrame:
    rows = []
    for session_key in (100, 200):
        for driver in (1, 2):
            for lap in (1, 2):
                rows.append(
                    {
                        "session_key": session_key,
                        "driver_number": driver,
                        "prediction_time": pd.Timestamp(
                            f"2025-01-01T00:0{session_key // 100}{lap}:00Z"
                        ),
                        "lap_number": lap,
                        "winner": int(driver == 1),
                        "stint_compound": "MEDIUM",
                        "position_position": driver,
                        "interval_gap_to_leader": float(driver - 1),
                        "interval_laps_behind_leader": None,
                        "interval_interval": float(driver - 1),
                        "interval_laps_behind_car_ahead": None,
                        "stint_tyre_age_at_start": float(lap),
                        "weather_air_temperature": 25.0,
                        "weather_track_temperature": 35.0,
                        "weather_humidity": 50.0,
                        "weather_rainfall": 0.0,
                        "weather_wind_speed": 1.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["season"] = 2025
    return frame


def test_live_contract_has_only_point_in_time_features():
    contract = LiveFeatureContract.default()
    assert "winner" not in contract.features
    assert "position_position" in contract.features
    assert "stint_compound" in contract.features


def test_select_live_dataset_preserves_missing_values_for_pipeline():
    X, y = select_live_dataset(_frame())
    assert list(X.columns) == list(LiveFeatureContract.default().features)
    assert len(X) == len(y)
    assert X["interval_laps_behind_leader"].isna().all()


def test_validate_race_snapshot_targets():
    validate_race_snapshot_targets(_frame())


def test_invalid_multiple_winners_are_rejected():
    frame = _frame()
    frame.loc[
        (frame["session_key"] == 100) & (frame["driver_number"] == 2),
        "winner",
    ] = 1
    with pytest.raises(ValueError, match="exactly one unique winner"):
        validate_race_snapshot_targets(frame)


def test_live_dataset_summary():
    summary = summarize_live_dataset(_frame())
    assert summary["rows"] == 8
    assert summary["races"] == 2
    assert summary["drivers"] == 2
    assert summary["duplicate_snapshots"] == 0
    assert summary["winner_drivers_per_race"] == 1
