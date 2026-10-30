from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from f1_predict.modeling.pre_race_model import (
    FEATURE_COLUMNS,
    RACE_KEYS,
    chronological_split,
    evaluate_predictions,
    validate_dataset,
    _race_probabilities,
)


def build_race_audit(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create one diagnostic row per race from normalized driver probabilities."""
    required = set(RACE_KEYS + ["driver_id", "winner", "win_probability"])
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    metadata_columns = [
        column for column in ("race_name", "circuit_id", "race_date")
        if column in frame.columns
    ]
    race_metadata = (
        frame[RACE_KEYS + metadata_columns]
        .drop_duplicates(RACE_KEYS)
        if metadata_columns else frame[RACE_KEYS].drop_duplicates()
    )
    rows: list[dict[str, Any]] = []

    for key, race in predictions.groupby(RACE_KEYS, sort=True):
        race = race.sort_values(
            ["win_probability", "driver_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        winners = race.loc[race["winner"] == 1, "driver_id"].tolist()
        if len(winners) != 1:
            raise ValueError(f"Race {key} must have exactly one actual winner")

        actual_winner = str(winners[0])
        winner_rows = race[race["driver_id"] == actual_winner]
        actual_probability = float(winner_rows.iloc[0]["win_probability"])
        top_probability = float(race.iloc[0]["win_probability"])
        second_probability = (
            float(race.iloc[1]["win_probability"]) if len(race) > 1 else 0.0
        )
        tied_top = np.isclose(
            race["win_probability"].to_numpy(dtype=float),
            top_probability,
            rtol=1e-12,
            atol=1e-12,
        )
        top_drivers = race.loc[tied_top, "driver_id"].astype(str).tolist()
        actual_in_top_tie = actual_winner in top_drivers
        actual_rank = int(
            race["win_probability"].rank(method="min", ascending=False)
            .loc[race["driver_id"] == actual_winner].iloc[0]
        )

        row: dict[str, Any] = {
            "season": int(key[0]),
            "round": int(key[1]),
            "actual_winner_driver_id": actual_winner,
            "predicted_winner_driver_id": str(race.iloc[0]["driver_id"]),
            "predicted_top_tie_driver_ids": "|".join(top_drivers),
            "actual_winner_probability": actual_probability,
            "predicted_top_probability": top_probability,
            "top_two_probability_margin": top_probability - second_probability,
            "actual_winner_probability_rank": actual_rank,
            "top_prediction_correct": bool(actual_in_top_tie),
            "field_size": int(len(race)),
        }
        rows.append(row)

    audit = pd.DataFrame(rows)
    if not race_metadata.empty:
        audit = audit.merge(race_metadata, on=RACE_KEYS, how="left", validate="one_to_one")
    return audit.sort_values(RACE_KEYS).reset_index(drop=True)


def extract_xgboost_gain_importance(model: Any) -> pd.DataFrame:
    """Return training-fitted XGBoost gain importance for every input feature."""
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    values: dict[str, float] = {}
    for index, feature in enumerate(FEATURE_COLUMNS):
        values[feature] = float(raw.get(feature, raw.get(f"f{index}", 0.0)))
    result = pd.DataFrame(
        [{"feature": name, "gain_importance": value} for name, value in values.items()]
    )
    total = float(result["gain_importance"].sum())
    result["gain_importance_share"] = (
        result["gain_importance"] / total if total > 0 else 0.0
    )
    return result.sort_values(
        ["gain_importance", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def run_diagnostics(
    frame: pd.DataFrame,
    artifacts: dict[str, Any],
    train_end_year: int = 2022,
    validation_start_year: int = 2023,
    validation_end_year: int = 2024,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Recreate held-out validation predictions and produce audit/importance tables."""
    validate_dataset(frame)
    _, validation = chronological_split(
        frame, train_end_year, validation_start_year, validation_end_year
    )
    features = artifacts.get("feature_columns", FEATURE_COLUMNS)
    if list(features) != list(FEATURE_COLUMNS):
        raise ValueError("Model artifact feature columns do not match current contract")

    x_validation = validation[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    xgb_model = artifacts["xgboost"]
    raw_scores = xgb_model.predict_proba(x_validation)[:, 1]
    predictions = _race_probabilities(validation, raw_scores)

    audit = build_race_audit(validation, predictions)
    importance = extract_xgboost_gain_importance(xgb_model)

    by_season: dict[str, Any] = {}
    for season in sorted(validation["season"].unique().astype(int).tolist()):
        season_predictions = predictions[predictions["season"] == season]
        by_season[str(season)] = evaluate_predictions(season_predictions)

    summary = {
        "training_seasons": sorted(
            frame.loc[frame["season"] <= train_end_year, "season"]
            .unique().astype(int).tolist()
        ),
        "validation_seasons": sorted(
            validation["season"].unique().astype(int).tolist()
        ),
        "validation_races": int(len(audit)),
        "xgboost_validation_metrics": evaluate_predictions(predictions),
        "xgboost_season_metrics": by_season,
        "worst_log_loss_races": audit.sort_values(
            "actual_winner_probability", ascending=True
        ).head(10)[
            RACE_KEYS + ["actual_winner_driver_id", "predicted_winner_driver_id",
                         "actual_winner_probability", "actual_winner_probability_rank"]
        ].to_dict(orient="records"),
        "holdout_policy": "2025 was not loaded or evaluated.",
        "importance_method": "XGBoost booster gain; fitted on training seasons only.",
    }
    return summary, audit, importance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate race-level diagnostics for the pre-race winner model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/features/pre_race_driver_features.parquet"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pre_race_winner_models.joblib"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pre_race_diagnostics"),
    )
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--validation-start-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2024)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    artifacts = joblib.load(args.model)
    summary, audit, importance = run_diagnostics(
        frame,
        artifacts,
        args.train_end_year,
        args.validation_start_year,
        args.validation_end_year,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "race_level_audit.csv"
    importance_path = args.output_dir / "xgboost_feature_importance.csv"
    summary_path = args.output_dir / "diagnostics_summary.json"
    audit.to_csv(audit_path, index=False)
    importance.to_csv(importance_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("PRE-RACE MODEL DIAGNOSTICS")
    print(f"Validation races: {summary['validation_races']}")
    print("\nXGBoost season metrics")
    for season, metrics in summary["xgboost_season_metrics"].items():
        print(
            f"{season}: accuracy={metrics['winner_accuracy']:.4f}, "
            f"log_loss={metrics['log_loss']:.4f}, "
            f"brier={metrics['multiclass_brier_score']:.4f}, "
            f"top_ECE={metrics['top_label_ece']:.4f}"
        )
    print("\nTop 10 XGBoost features by gain")
    print(importance.head(10).to_string(index=False))
    print("\nLowest actual-winner probabilities")
    print(
        audit.sort_values("actual_winner_probability", ascending=True)
        .head(10)[
            ["season", "round", "actual_winner_driver_id",
             "predicted_winner_driver_id", "actual_winner_probability",
             "actual_winner_probability_rank"]
        ].to_string(index=False)
    )
    print(f"\nRace audit: {audit_path}")
    print(f"Feature importance: {importance_path}")
    print(f"Summary: {summary_path}")
    print("2025 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
