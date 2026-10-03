from __future__ import annotations

import pandas as pd
import pytest

from f1_predict.modeling.live_attribution import aggregate_source_importance


def test_aggregate_source_importance_collapses_encoded_features():
    importance = pd.DataFrame(
        {
            "feature": [
                "numeric__lap_number",
                "numeric__position_position",
                "categorical__stint_compound__SOFT",
                "categorical__stint_compound__MEDIUM",
            ],
            "importance": [0.2, 0.5, 0.1, 0.2],
        }
    )
    result = aggregate_source_importance(importance)
    assert result.iloc[0]["source_feature"] == "position_position"
    compound = result[result["source_feature"] == "stint_compound"].iloc[0]
    assert compound["importance"] == pytest.approx(0.3)
    assert result["relative_importance"].sum() == pytest.approx(1.0)
