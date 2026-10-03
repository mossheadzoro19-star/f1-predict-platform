from __future__ import annotations

import pandas as pd
import pytest

from f1_predict.modeling.live_phase_eval import (
    assign_race_phase,
    evaluate_race_phases,
)


def test_assign_race_phase_uses_fixed_lap_boundaries() -> None:
    laps = pd.Series([1, 10, 11, 30, 31, 50])
    phases = assign_race_phase(laps)

    assert phases.tolist() == ["early", "early", "mid", "mid", "late", "late"]


def test_assign_race_phase_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="bounds"):
        assign_race_phase(pd.Series([1]), (("bad", 3, 2),))


def test_phase_evaluation_returns_all_configured_phases() -> None:
    rows = []
    for lap in [1, 11, 31]:
        rows.extend(
            [
                {
                    "session_key": "s1",
                    "lap_number": lap,
                    "driver_number": 1,
                    "winner": 1,
                    "win_probability": 0.8,
                },
                {
                    "session_key": "s1",
                    "lap_number": lap,
                    "driver_number": 2,
                    "winner": 0,
                    "win_probability": 0.2,
                },
            ]
        )

    result = evaluate_race_phases(pd.DataFrame(rows))

    assert result["phase"].tolist() == ["early", "mid", "late"]
    assert result["snapshots"].tolist() == [1, 1, 1]
    assert result["winner_accuracy"].tolist() == [1.0, 1.0, 1.0]


def test_phase_evaluation_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing phase-evaluation columns"):
        evaluate_race_phases(
            pd.DataFrame(
                {
                    "session_key": ["s1"],
                    "lap_number": [1],
                    "driver_number": [1],
                    "winner": [1],
                }
            )
        )
