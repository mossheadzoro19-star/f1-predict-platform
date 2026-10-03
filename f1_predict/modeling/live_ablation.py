from __future__ import annotations

import argparse

import pandas as pd

from f1_predict.modeling.live_baseline import (
    LivePredictionMetrics,
    build_live_logistic_baseline,
    evaluate_live_probabilities,
    predict_live_probabilities,
)
from f1_predict.modeling.live_features import (
    LiveFeatureContract,
    LIVE_CATEGORICAL_FEATURES,
    LIVE_NUMERIC_FEATURES,
    TARGET_COLUMN,
    select_live_dataset,
)
from f1_predict.modeling.live_split import chronological_live_race_split


BASE_NUMERIC = {
    "lap_number",
    "position_position",
}

GAP_NUMERIC = {
    "interval_gap_to_leader",
    "interval_laps_behind_leader",
    "interval_interval",
    "interval_laps_behind_car_ahead",
}

TYRE_NUMERIC = {
    "stint_tyre_age_at_start",
}

WEATHER_NUMERIC = {
    "weather_air_temperature",
    "weather_track_temperature",
    "weather_humidity",
    "weather_rainfall",
    "weather_wind_speed",
}


ABLATIONS: dict[str, tuple[str, ...]] = {
    "A_position": tuple(BASE_NUMERIC),
    "B_position_gap": tuple(BASE_NUMERIC | GAP_NUMERIC),
    "C_position_gap_tyre": tuple(BASE_NUMERIC | GAP_NUMERIC | TYRE_NUMERIC),
    "D_position_gap_tyre_weather": tuple(
        BASE_NUMERIC | GAP_NUMERIC | TYRE_NUMERIC | WEATHER_NUMERIC
    ),
    "E_all_features": tuple(LIVE_NUMERIC_FEATURES),
}


def make_contract(name: str) -> LiveFeatureContract:
    """Build a feature contract for one pre-registered ablation."""
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {name}")

    numeric = tuple(
        feature
        for feature in LIVE_NUMERIC_FEATURES
        if feature in ABLATIONS[name]
    )
    categorical = tuple(
        feature
        for feature in LIVE_CATEGORICAL_FEATURES
        if name in {"C_position_gap_tyre", "D_position_gap_tyre_weather", "E_all_features"}
    )
    return LiveFeatureContract(
        numeric_features=numeric,
        categorical_features=categorical,
        target=TARGET_COLUMN,
    )


def run_ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    contract: LiveFeatureContract,
) -> tuple[LivePredictionMetrics, LivePredictionMetrics]:
    """Train one fixed feature variant and evaluate validation and test."""
    X_train, y_train = select_live_dataset(train, contract)
    model = build_live_logistic_baseline(contract)
    model.fit(X_train, y_train)

    validation_predictions = predict_live_probabilities(model, validation, contract)
    validation_metrics = evaluate_live_probabilities(validation_predictions)

    final_train = pd.concat([train, validation], ignore_index=True)
    X_final, y_final = select_live_dataset(final_train, contract)
    final_model = build_live_logistic_baseline(contract)
    final_model.fit(X_final, y_final)

    test_predictions = predict_live_probabilities(final_model, test, contract)
    test_metrics = evaluate_live_probabilities(test_predictions)

    return validation_metrics, test_metrics


def _print_row(
    name: str,
    features: int,
    validation: LivePredictionMetrics,
    test: LivePredictionMetrics,
) -> None:
    print(
        f"{name:<30} "
        f"{features:>2} "
        f"{validation.winner_accuracy:>9.4f} "
        f"{validation.log_loss:>9.4f} "
        f"{validation.brier_score:>9.4f} "
        f"{test.winner_accuracy:>9.4f} "
        f"{test.log_loss:>9.4f} "
        f"{test.brier_score:>9.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered live-race feature ablation study."
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

    print(
        "LIVE ABLATION STUDY\n"
        f"TRAIN: {len(split.train)} rows, {split.train['session_key'].nunique()} races\n"
        f"VALIDATION: {len(split.validation)} rows, {split.validation['session_key'].nunique()} races\n"
        f"TEST: {len(split.test)} rows, {split.test['session_key'].nunique()} races\n"
    )
    print(
        f"{'Variant':<30} {'N':>2} "
        f"{'Val Acc':>9} {'Val LL':>9} {'Val Brier':>9} "
        f"{'Test Acc':>9} {'Test LL':>9} {'Test Brier':>9}"
    )
    print("-" * 100)

    for name, _ in ABLATIONS.items():
        contract = make_contract(name)
        validation, test = run_ablation(
            split.train,
            split.validation,
            split.test,
            contract,
        )
        _print_row(name, len(contract.features), validation, test)


if __name__ == "__main__":
    main()
