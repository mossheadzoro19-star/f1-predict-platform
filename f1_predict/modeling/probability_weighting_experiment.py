from __future__ import annotations

"""Compare weighted and unweighted XGBoost probabilities on validation only."""

import argparse

import pandas as pd

from f1_predict.modeling.live_features import LiveFeatureContract
from f1_predict.modeling.live_split import chronological_live_race_split
from f1_predict.modeling.live_xgboost import (
    LiveXGBoostConfig,
    fit_live_xgboost,
    predict_live_probabilities,
)
from f1_predict.modeling.race_state_evaluation import build_race_state_report


def _evaluate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    balanced: bool,
    contract: LiveFeatureContract,
) -> dict[str, object]:
    config = LiveXGBoostConfig(balance_classes=balanced)
    model = fit_live_xgboost(train, contract=contract, config=config)
    predictions = predict_live_probabilities(model, validation, contract)
    return build_race_state_report(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare class-weighted and unweighted XGBoost on the 2024 validation "
            "season. Does not evaluate or use the test season."
        )
    )
    parser.add_argument(
        "--data",
        default="data/processed/features/jolpica_race_state_enriched.parquet",
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

    reports = {
        "WEIGHTED": _evaluate(split.train, split.validation, True, contract),
        "UNWEIGHTED": _evaluate(split.train, split.validation, False, contract),
    }

    print(
        "VALIDATION-ONLY CLASS-WEIGHTING EXPERIMENT\n"
        f"TRAIN: {len(split.train):,} rows, {split.train['session_key'].nunique()} races "
        f"(seasons <= {args.train_end_year})\n"
        f"VALIDATION: {len(split.validation):,} rows, "
        f"{split.validation['session_key'].nunique()} races "
        f"(seasons {args.train_end_year + 1}-{args.validation_end_year})\n"
        "The test season is not evaluated."
    )
    print(
        f"{'Metric':<28}{'WEIGHTED':>12}{'UNWEIGHTED':>14}"
    )
    print("-" * 54)
    for metric in (
        "winner_accuracy",
        "current_leader_accuracy",
        "model_vs_leader_accuracy_delta",
        "log_loss",
        "brier_score",
    ):
        a = reports["WEIGHTED"]["overall"][metric]
        b = reports["UNWEIGHTED"]["overall"][metric]
        print(f"{metric:<28}{a:>12.4f}{b:>14.4f}")

    for name, report in reports.items():
        print(f"\n{name} CONFIDENCE DIAGNOSTICS")
        print(
            pd.DataFrame(report["confidence"]).to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )


if __name__ == "__main__":
    main()
