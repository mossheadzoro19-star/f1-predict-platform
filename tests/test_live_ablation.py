from __future__ import annotations

import pandas as pd

from f1_predict.modeling.live_ablation import ABLATIONS, make_contract


def test_ablation_variants_are_cumulative() -> None:
    assert list(ABLATIONS) == [
        "A_position",
        "B_position_gap",
        "C_position_gap_tyre",
        "D_position_gap_tyre_weather",
        "E_all_features",
    ]

    contracts = [make_contract(name) for name in ABLATIONS]

    assert "position_position" in contracts[0].features
    assert "interval_gap_to_leader" not in contracts[0].features
    assert "interval_gap_to_leader" in contracts[1].features
    assert "stint_tyre_age_at_start" in contracts[2].features
    assert "stint_compound" in contracts[2].features
    assert "weather_air_temperature" in contracts[3].features
    assert set(contracts[4].features) == set(contracts[4].numeric_features + contracts[4].categorical_features)


def test_ablation_contracts_keep_target_out_of_features() -> None:
    for name in ABLATIONS:
        contract = make_contract(name)
        assert contract.target not in contract.features
