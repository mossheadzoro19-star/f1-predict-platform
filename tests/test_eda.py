import pandas as pd
import pytest

from f1_predict.analysis.eda import (
    build_eda_report,
    summarize_qualifying_and_grid,
    summarize_target,
)


def _frame():
    return pd.DataFrame(
        [
            {"season": 2024, "round": 1, "driver_id": "a", "constructor_id": "x", "circuit_id": "c",
             "winner": 1, "qualifying_position": 1, "grid": 1},
            {"season": 2024, "round": 1, "driver_id": "b", "constructor_id": "y", "circuit_id": "c",
             "winner": 0, "qualifying_position": 2, "grid": 2},
            {"season": 2024, "round": 2, "driver_id": "a", "constructor_id": "x", "circuit_id": "d",
             "winner": 0, "qualifying_position": 2, "grid": 2},
            {"season": 2024, "round": 2, "driver_id": "b", "constructor_id": "y", "circuit_id": "d",
             "winner": 1, "qualifying_position": 1, "grid": 1},
        ]
    )


def test_target_summary():
    result = summarize_target(_frame())
    assert result["winner_rows"] == 2
    assert result["non_winner_rows"] == 2
    assert result["races"] == 2
    assert result["one_winner_per_race"] is True
    assert result["winner_row_rate"] == pytest.approx(0.5)


def test_qualifying_and_grid_summary():
    result = summarize_qualifying_and_grid(_frame())
    assert set(result["feature"]) == {"qualifying_position", "grid"}
    first = result[(result["feature"] == "qualifying_position") & (result["position"] == 1)].iloc[0]
    assert first["starts"] == 2
    assert first["wins"] == 2
    assert first["win_rate"] == pytest.approx(1.0)


def test_eda_report_is_json_serializable():
    report = build_eda_report(_frame())
    assert report["dataset"]["races"] == 2
    assert report["target"]["winner_rows"] == 2
    assert len(report["seasons"]) == 1
    assert "numeric_target_association" in report
