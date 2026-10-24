import pandas as pd
import pytest

from f1_predict.ingestion.repair_race_state_features import (
    recompute_multi_lap_features,
)


def _source_frame() -> pd.DataFrame:
    rows = []
    # Interleave races and drivers so grouping and sorting are both exercised.
    for session, driver, offset in ((100, 1, 0.0), (200, 1, 100.0)):
        for lap in range(1, 7):
            rows.append(
                {
                    "session_key": session,
                    "driver_number": driver,
                    "lap_number": lap,
                    "position_position": float(lap + (session == 200)),
                    "interval_gap_to_leader": offset + float(lap),
                    "interval_interval": float(lap * 0.5),
                    "winner": int(driver == 1),
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)


def test_recomputes_multi_lap_deltas_within_each_race_and_driver():
    result = recompute_multi_lap_features(_source_frame())

    race_100_lap_4 = result.loc[
        (result["session_key"] == 100) & (result["lap_number"] == 4)
    ].iloc[0]
    race_200_lap_4 = result.loc[
        (result["session_key"] == 200) & (result["lap_number"] == 4)
    ].iloc[0]

    assert race_100_lap_4["position_change_3_laps"] == 3.0
    assert race_100_lap_4["gap_change_3_laps"] == 3.0
    assert race_100_lap_4["interval_change_3_laps"] == 1.5
    # The first 3 laps of a new race must not borrow the prior race's values.
    assert result.loc[
        (result["session_key"] == 200) & (result["lap_number"] == 1),
        "position_change_3_laps",
    ].isna().all()
    assert race_200_lap_4["position_change_3_laps"] == 3.0


def test_five_lap_features_need_five_prior_observations():
    result = recompute_multi_lap_features(_source_frame())
    assert result.loc[
        result["lap_number"].le(5), "position_change_5_laps"
    ].isna().all()
    assert result.loc[
        (result["session_key"] == 100) & (result["lap_number"] == 6),
        "position_change_5_laps",
    ].iloc[0] == 5.0


def test_rejects_duplicate_race_driver_lap_rows():
    frame = _source_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate historical observations"):
        recompute_multi_lap_features(frame)


def test_rejects_missing_source_columns():
    with pytest.raises(ValueError, match="Missing source columns"):
        recompute_multi_lap_features(pd.DataFrame({"session_key": [1]}))
