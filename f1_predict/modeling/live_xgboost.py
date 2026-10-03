from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from f1_predict.modeling.live_baseline import (
    LivePredictionMetrics,
    evaluate_live_probabilities,
    predict_live_probabilities,
)
from f1_predict.modeling.live_features import (
    LiveFeatureContract,
    select_live_dataset,
)
from f1_predict.modeling.live_split import chronological_live_race_split


@dataclass(frozen=True)
class LiveXGBoostConfig:
    """Conservative first XGBoost configuration for live-race research."""

    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 2.0
    random_state: int = 42


def build_live_xgboost(
    contract: LiveFeatureContract | None = None,
    config: LiveXGBoostConfig | None = None,
) -> Pipeline:
    """Build a point-in-time XGBoost classifier with the same feature contract."""
    contract = contract or LiveFeatureContract.default()
    config = config or LiveXGBoostConfig()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = [
        ("numeric", numeric_pipeline, list(contract.numeric_features)),
    ]
    if contract.categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, list(contract.categorical_features))
        )

    preprocess = ColumnTransformer(transformers=transformers)

    classifier = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=config.random_state,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )


def fit_live_xgboost(
    train: pd.DataFrame,
    contract: LiveFeatureContract | None = None,
    config: LiveXGBoostConfig | None = None,
) -> Pipeline:
    """Fit XGBoost only on historical race-state observations."""
    X, y = select_live_dataset(train, contract)
    model = build_live_xgboost(contract, config)

    # The winner class is rare: one eventual winner among the drivers in each
    # lap snapshot. Balance training gradients without changing the evaluation.
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError("Training data contains no winner observations")
    classifier = model.named_steps["classifier"]
    classifier.set_params(scale_pos_weight=negatives / positives)

    model.fit(X, y)
    return model


def compare_logistic_and_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    contract: LiveFeatureContract | None = None,
) -> dict[str, tuple[LivePredictionMetrics, LivePredictionMetrics]]:
    """Compare both models under the identical temporal evaluation protocol."""
    from f1_predict.modeling.live_baseline import fit_live_baseline

    logistic = fit_live_baseline(train, contract)
    logistic_validation = evaluate_live_probabilities(
        predict_live_probabilities(logistic, validation, contract)
    )

    final_train = pd.concat([train, validation], ignore_index=True)
    final_logistic = fit_live_baseline(final_train, contract)
    logistic_test = evaluate_live_probabilities(
        predict_live_probabilities(final_logistic, test, contract)
    )

    xgb = fit_live_xgboost(train, contract)
    xgb_validation = evaluate_live_probabilities(
        predict_live_probabilities(xgb, validation, contract)
    )

    final_xgb = fit_live_xgboost(final_train, contract)
    xgb_test = evaluate_live_probabilities(
        predict_live_probabilities(final_xgb, test, contract)
    )

    return {
        "Logistic": (logistic_validation, logistic_test),
        "XGBoost": (xgb_validation, xgb_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Logistic Regression with the first live XGBoost model."
    )
    parser.add_argument(
        "--data",
        default="data/processed/features/openf1_race_state.parquet",
    )
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2024)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    split = chronological_live_race_split(
        frame,
        train_end_year=args.train_end_year,
        validation_end_year=args.validation_end_year,
    )

    contract = LiveFeatureContract.default()
    results = compare_logistic_and_xgboost(
        split.train,
        split.validation,
        split.test,
        contract,
    )

    print(
        "LIVE MODEL COMPARISON\n"
        f"TRAIN: {len(split.train)} rows, {split.train['session_key'].nunique()} races\n"
        f"VALIDATION: {len(split.validation)} rows, {split.validation['session_key'].nunique()} races\n"
        f"TEST: {len(split.test)} rows, {split.test['session_key'].nunique()} races\n"
    )
    print(
        f"{'Model':<14} {'Val Acc':>9} {'Val LL':>9} {'Val Brier':>10} "
        f"{'Test Acc':>9} {'Test LL':>9} {'Test Brier':>10}"
    )
    print("-" * 82)

    for name, (validation, test) in results.items():
        print(
            f"{name:<14} "
            f"{validation.winner_accuracy:>9.4f} "
            f"{validation.log_loss:>9.4f} "
            f"{validation.brier_score:>10.4f} "
            f"{test.winner_accuracy:>9.4f} "
            f"{test.log_loss:>9.4f} "
            f"{test.brier_score:>10.4f}"
        )


if __name__ == "__main__":
    main()
