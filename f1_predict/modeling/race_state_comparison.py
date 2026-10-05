from __future__ import annotations

"""Compare the frozen old race-state contract with the enriched contract."""

import argparse

import pandas as pd

from f1_predict.modeling.live_features import LiveFeatureContract
from f1_predict.modeling.race_state_evaluation import evaluate_current_xgboost


def _overall(report: dict[str, object], split: str) -> dict[str, float]:
    return report[split]["overall"]  # type: ignore[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-data",
        default="data/processed/features/openf1_race_state.parquet",
    )
    parser.add_argument(
        "--enriched-data",
        default="data/processed/features/fastf1_race_state_enriched.parquet",
    )
    args = parser.parse_args()

    old = evaluate_current_xgboost(
        pd.read_parquet(args.old_data),
        contract=LiveFeatureContract.old(),
    )
    enriched = evaluate_current_xgboost(
        pd.read_parquet(args.enriched_data),
        contract=LiveFeatureContract.default(),
    )

    print("\nOLD vs ENRICHED XGBOOST")
    print(f"{'Metric':<28}{'OLD':>12}{'ENRICHED':>12}{'DELTA':>12}")
    print("-" * 64)
    for split in ("validation", "test"):
        print(f"\n{split.upper()}")
        for metric in (
            "winner_accuracy",
            "current_leader_accuracy",
            "model_vs_leader_accuracy_delta",
            "log_loss",
            "brier_score",
        ):
            a = _overall(old, split)[metric]
            b = _overall(enriched, split)[metric]
            print(f"{metric:<28}{a:>12.4f}{b:>12.4f}{b-a:>+12.4f}")

        print(
            f"scorable_coverage{'':<12}"
            f"{_overall(old, split)['scorable_coverage']:>12.4f}"
            f"{_overall(enriched, split)['scorable_coverage']:>12.4f}"
        )


if __name__ == "__main__":
    main()
