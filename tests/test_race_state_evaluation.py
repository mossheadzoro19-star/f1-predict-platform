import pandas as pd
import pytest

from f1_predict.modeling.race_state_evaluation import (
    confidence_diagnostics,
    evaluate_race_state_predictions,
)


def _fixture() -> pd.DataFrame:
    rows = []
    for lap in (1, 5):
        rows.extend(
            [
                {"session_key": 1, "lap_number": lap, "driver_number": 1, "winner": 1,
                 "win_probability": 0.7, "position_position": 1},
                {"session_key": 1, "lap_number": lap, "driver_number": 2, "winner": 0,
                 "win_probability": 0.2, "position_position": 2},
                {"session_key": 1, "lap_number": lap, "driver_number": 3, "winner": 0,
                 "win_probability": 0.1, "position_position": 3},
            ]
        )
    return pd.DataFrame(rows)


def test_evaluation_reports_model_and_current_leader():
    metrics = evaluate_race_state_predictions(_fixture())

    assert metrics.snapshots == 2
    assert metrics.races == 1
    assert metrics.winner_accuracy == pytest.approx(1.0)
    assert metrics.current_leader_accuracy == pytest.approx(1.0)
    assert metrics.model_vs_leader_accuracy_delta == pytest.approx(0.0)


def test_confidence_diagnostics_has_expected_columns():
    result = confidence_diagnostics(_fixture(), bins=5)

    assert {"snapshots", "mean_confidence", "empirical_accuracy"} <= set(result.columns)
