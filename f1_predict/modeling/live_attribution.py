from __future__ import annotations

import argparse

import pandas as pd

from f1_predict.modeling.live_baseline import predict_live_probabilities
from f1_predict.modeling.live_features import LiveFeatureContract, select_live_dataset
from f1_predict.modeling.live_split import chronological_live_race_split
from f1_predict.modeling.live_xgboost import fit_live_xgboost


def xgboost_feature_importance(model, contract: LiveFeatureContract | None = None) -> pd.DataFrame:
    """Return gain-based importance for the expanded preprocessed XGBoost features."""
    contract = contract or LiveFeatureContract.default()
    classifier = model.named_steps["classifier"]
    preprocess = model.named_steps["preprocess"]

    names = list(preprocess.get_feature_names_out())
    importance = classifier.feature_importances_
    if len(names) != len(importance):
        raise ValueError("Feature-name and importance lengths do not match")

    result = pd.DataFrame({"feature": names, "importance": importance})
    result["importance"] = result["importance"].astype(float)
    result = result.sort_values("importance", ascending=False, ignore_index=True)
    total = result["importance"].sum()
    result["relative_importance"] = (
        result["importance"] / total if total > 0 else 0.0
    )
    return result


def aggregate_source_importance(
    importance: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse one-hot encoded features back to their source feature groups."""
    result = importance.copy()
    result["source_feature"] = (
        result["feature"]
        .str.replace(r"^(numeric|categorical)__", "", regex=True)
        .str.replace(r"__.*$", "", regex=True)
    )
    grouped = (
        result.groupby("source_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False, ignore_index=True)
    )
    total = grouped["importance"].sum()
    grouped["relative_importance"] = (
        grouped["importance"] / total if total > 0 else 0.0
    )
    return grouped


def permutation_importance_by_race(
    model,
    frame: pd.DataFrame,
    contract: LiveFeatureContract | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure source-feature impact by shuffling within each race snapshot."""
    contract = contract or LiveFeatureContract.default()
    baseline = predict_live_probabilities(model, frame, contract)
    baseline_loss = (
        baseline.assign(
            clipped_probability=baseline["win_probability"].clip(1e-7, 1 - 1e-7)
        )
        .assign(
            row_loss=lambda x: -(
                x["winner"] * x["clipped_probability"]
                + (1 - x["winner"])
                * (1 - x["clipped_probability"]).apply(lambda v: v if v > 0 else 1e-7)
            ).apply(lambda v: __import__("math").log(max(v, 1e-7)))
        )
    )
    baseline_score = float(baseline_loss["row_loss"].mean())

    rng = pd.Series(range(len(frame)), index=frame.index).sample(
        frac=1.0, random_state=random_state
    ).index
    results = []
    for feature in contract.features:
        shuffled = frame.copy()
        if feature not in shuffled.columns:
            continue
        shuffled[feature] = (
            shuffled.groupby(["session_key", "lap_number"])[feature]
            .transform(lambda s: s.sample(frac=1.0, random_state=random_state).to_numpy())
        )
        pred = predict_live_probabilities(model, shuffled, contract)
        p = pred["win_probability"].clip(1e-7, 1 - 1e-7)
        loss = -(pred["winner"] * p + (1 - pred["winner"]) * (1 - p)).apply(
            lambda v: __import__("math").log(max(v, 1e-7))
        ).mean()
        results.append(
            {
                "feature": feature,
                "baseline_log_loss": baseline_score,
                "permuted_log_loss": float(loss),
                "log_loss_increase": float(loss - baseline_score),
            }
        )
    return pd.DataFrame(results).sort_values(
        "log_loss_increase", ascending=False, ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect live XGBoost feature importance and race-state signal impact."
    )
    parser.add_argument("--data", default="data/processed/features/openf1_race_state.parquet")
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2024)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    split = chronological_live_race_split(
        frame,
        train_end_year=args.train_end_year,
        validation_end_year=args.validation_end_year,
    )
    contract = LiveFeatureContract.default()
    train_validation = pd.concat([split.train, split.validation], ignore_index=True)
    model = fit_live_xgboost(train_validation, contract)

    detailed = xgboost_feature_importance(model, contract)
    grouped = aggregate_source_importance(detailed)

    print("LIVE XGBOOST FEATURE ATTRIBUTION")
    print("\nTOP PREPROCESSED FEATURES")
    print(detailed.head(args.top).to_string(index=False))
    print("\nSOURCE FEATURE IMPORTANCE")
    print(grouped.to_string(index=False))

    permutation = permutation_importance_by_race(
        model,
        split.test,
        contract,
    )
    print("\nTEST RACE-SNAPSHOT PERMUTATION IMPACT")
    print(permutation.to_string(index=False))


if __name__ == "__main__":
    main()
