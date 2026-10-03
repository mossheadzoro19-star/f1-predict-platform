from __future__ import annotations

import pandas as pd

from f1_predict.modeling.live_xgboost import (
    LiveXGBoostConfig,
    build_live_xgboost,
)


def test_xgboost_config_is_deterministic() -> None:
    config = LiveXGBoostConfig()
    assert config.random_state == 42
    assert config.n_estimators == 400


def test_xgboost_builds_with_default_live_contract() -> None:
    model = build_live_xgboost()
    assert "preprocess" in model.named_steps
    assert "classifier" in model.named_steps


def test_xgboost_supports_numeric_only_contract() -> None:
    from f1_predict.modeling.live_features import LiveFeatureContract

    contract = LiveFeatureContract(
        numeric_features=("position_position",),
        categorical_features=(),
        target="winner",
    )
    model = build_live_xgboost(contract)
    assert "classifier" in model.named_steps
