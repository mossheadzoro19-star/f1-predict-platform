from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_predict.modeling.pre_race_model import (
    FEATURE_COLUMNS,
    RACE_KEYS,
    _make_xgboost,
    _race_probabilities,
    evaluate_predictions,
    validate_dataset,
)

DEFAULT_VALIDATION_YEARS = (2021, 2022, 2023, 2024)
MAX_ALLOWED_SEASON = 2024
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_LOW_WINNER_PROBABILITY = 0.05
DEFAULT_BINS = 10

AUDIT_FEATURES = (
    "qualifying_position",
    "grid",
    "qualifying_gap_to_pole_seconds",
    "teammate_q3_gap_seconds",
    "driver_points_last_5",
    "driver_weighted_form_5",
    "driver_prior_win_rate",
    "driver_prior_dnf_rate",
    "constructor_prior_points",
    "constructor_prior_win_rate",
    "constructor_prior_dnf_rate",
    "driver_circuit_prior_starts",
)


def validate_audit_frame(
    frame: pd.DataFrame,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
) -> None:
    validate_dataset(frame)
    seasons = set(frame["season"].astype(int).unique().tolist())
    if not seasons:
        raise ValueError("Audit dataset contains no seasons")
    if max(seasons) > MAX_ALLOWED_SEASON:
        raise ValueError(
            "Prediction audit must not contain seasons after 2024; "
            "2025 is reserved as the final untouched holdout"
        )
    if tuple(sorted(set(validation_years))) != validation_years:
        raise ValueError("Validation years must be unique and in ascending order")
    for year in validation_years:
        if year not in seasons:
            raise ValueError(f"Validation season {year} is missing from the dataset")
        if not any(season < year for season in seasons):
            raise ValueError(f"No earlier training seasons are available for {year}")


def build_driver_predictions(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    train_end_year: int,
) -> pd.DataFrame:
    """Return one out-of-fold probability per driver/race candidate."""
    if len(validation) != len(probabilities):
        raise ValueError("Probability count does not match validation rows")
    result = validation.loc[:, RACE_KEYS + ["driver_id", "winner"]].copy()
    result["win_probability"] = probabilities.astype(float)
    result["train_end_year"] = int(train_end_year)
    result["season"] = result["season"].astype(int)
    result["round"] = result["round"].astype(int)

    if (
        not np.isfinite(result["win_probability"]).all()
        or (result["win_probability"] < 0).any()
        or (result["win_probability"] > 1).any()
    ):
        raise ValueError("Driver probabilities must be finite and in [0, 1]")
    totals = result.groupby(RACE_KEYS)["win_probability"].sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-8):
        raise ValueError("Driver probabilities must sum to one within every race")
    return result


def build_race_error_audit(
    frame: pd.DataFrame,
    driver_predictions: pd.DataFrame,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    low_winner_probability: float = DEFAULT_LOW_WINNER_PROBABILITY,
) -> pd.DataFrame:
    """Summarize winner misses, high-confidence errors and useful race context."""
    if not 0 < confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in (0, 1]")
    if not 0 <= low_winner_probability <= 1:
        raise ValueError("low_winner_probability must be in [0, 1]")

    metadata = [
        name for name in ("race_name", "circuit_id", "race_date")
        if name in frame.columns
    ]
    context_columns = [
        name for name in AUDIT_FEATURES if name in frame.columns
    ]
    context = frame[RACE_KEYS + ["driver_id"] + metadata + context_columns].copy()
    rows: list[dict[str, Any]] = []

    for key, group in driver_predictions.groupby(RACE_KEYS, sort=True):
        ranked = group.sort_values(
            ["win_probability", "driver_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        winners = ranked.loc[ranked["winner"] == 1]
        if len(winners) != 1:
            raise ValueError(f"Race {key} must contain exactly one winner")
        actual = winners.iloc[0]
        predicted = ranked.iloc[0]
        actual_probability = float(actual["win_probability"])
        top_probability = float(predicted["win_probability"])
        second_probability = (
            float(ranked.iloc[1]["win_probability"]) if len(ranked) > 1 else 0.0
        )
        actual_rank = int(
            ranked["win_probability"].rank(method="min", ascending=False)
            .loc[ranked["driver_id"] == actual["driver_id"]].iloc[0]
        )
        row: dict[str, Any] = {
            "season": int(key[0]),
            "round": int(key[1]),
            "train_end_year": int(actual["train_end_year"]),
            "actual_winner_driver_id": str(actual["driver_id"]),
            "predicted_winner_driver_id": str(predicted["driver_id"]),
            "actual_winner_probability": actual_probability,
            "actual_winner_rank": actual_rank,
            "predicted_winner_probability": top_probability,
            "top_two_probability_margin": top_probability - second_probability,
            "field_size": int(len(ranked)),
            "top_prediction_correct": bool(predicted["driver_id"] == actual["driver_id"]),
            "high_confidence_wrong": bool(
                predicted["driver_id"] != actual["driver_id"]
                and top_probability >= confidence_threshold
            ),
            "low_probability_actual_winner": bool(
                actual_probability <= low_winner_probability
            ),
        }

        for role, candidate in (("actual_winner", actual), ("predicted_winner", predicted)):
            lookup = context.loc[
                (context["season"] == key[0])
                & (context["round"] == key[1])
                & (context["driver_id"].astype(str) == str(candidate["driver_id"]))
            ]
            if len(lookup) != 1:
                raise ValueError(f"Could not uniquely resolve {role} context for race {key}")
            details = lookup.iloc[0]
            for name in metadata + context_columns:
                row[f"{role}_{name}"] = details[name]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(RACE_KEYS).reset_index(drop=True)


def build_candidate_reliability_bins(
    predictions: pd.DataFrame,
    bins: int = DEFAULT_BINS,
) -> pd.DataFrame:
    """Bin all driver-race probabilities against binary winner outcomes."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    probability = predictions["win_probability"].to_numpy(dtype=float)
    outcome = predictions["winner"].to_numpy(dtype=float)
    records: list[dict[str, Any]] = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        count = int(mask.sum())
        if count == 0:
            continue
        records.append({
            "bin_lower": lower,
            "bin_upper": upper,
            "candidate_rows": count,
            "mean_predicted_probability": float(probability[mask].mean()),
            "observed_winner_frequency": float(outcome[mask].mean()),
            "absolute_calibration_gap": float(
                abs(probability[mask].mean() - outcome[mask].mean())
            ),
        })
    return pd.DataFrame(records)


def build_top_choice_reliability_bins(
    race_audit: pd.DataFrame,
    bins: int = DEFAULT_BINS,
) -> pd.DataFrame:
    """Assess whether the top-ranked driver's confidence matches hit frequency."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    confidence = race_audit["predicted_winner_probability"].to_numpy(dtype=float)
    correct = race_audit["top_prediction_correct"].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    records: list[dict[str, Any]] = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        hit_rate = float(correct[mask].mean())
        records.append({
            "bin_lower": lower,
            "bin_upper": upper,
            "races": count,
            "mean_top_probability": mean_confidence,
            "top_pick_hit_rate": hit_rate,
            "absolute_calibration_gap": abs(mean_confidence - hit_rate),
        })
    return pd.DataFrame(records)


def _ece_from_bins(
    bins: pd.DataFrame,
    count_column: str,
    gap_column: str,
    total: int,
) -> float:
    if bins.empty or total == 0:
        return 0.0
    return float(
        (bins[count_column] * bins[gap_column]).sum() / total
    )


def run_prediction_audit(
    frame: pd.DataFrame,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    low_winner_probability: float = DEFAULT_LOW_WINNER_PROBABILITY,
    bins: int = DEFAULT_BINS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train XGBoost on prior seasons and retain all driver-level OOF probabilities."""
    validate_audit_frame(frame, validation_years)
    if bins < 2:
        raise ValueError("bins must be at least 2")

    prediction_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for year in validation_years:
        train_end = year - 1
        train = frame.loc[frame["season"] <= train_end].copy()
        validation = frame.loc[frame["season"] == year].copy()
        x_train = train[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        x_validation = validation[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        model = _make_xgboost()
        model.fit(x_train, train["winner"].astype("int8"), verbose=False)
        raw_scores = model.predict_proba(x_validation)[:, 1]
        race_predictions = _race_probabilities(validation, raw_scores)
        candidate_predictions = build_driver_predictions(
            validation,
            race_predictions["win_probability"].to_numpy(dtype=float),
            train_end,
        )
        prediction_frames.append(candidate_predictions)
        fold_summaries.append({
            "validation_season": int(year),
            "train_end_year": int(train_end),
            "training_seasons": sorted(train["season"].astype(int).unique().tolist()),
            "training_races": int(train[RACE_KEYS].drop_duplicates().shape[0]),
            "validation_races": int(validation[RACE_KEYS].drop_duplicates().shape[0]),
            "metrics": evaluate_predictions(race_predictions),
        })

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        RACE_KEYS + ["driver_id"]
    ).reset_index(drop=True)
    race_audit = build_race_error_audit(
        frame,
        predictions,
        confidence_threshold=confidence_threshold,
        low_winner_probability=low_winner_probability,
    )
    candidate_bins = build_candidate_reliability_bins(predictions, bins=bins)
    top_bins = build_top_choice_reliability_bins(race_audit, bins=bins)
    candidate_ece = _ece_from_bins(
        candidate_bins, "candidate_rows", "absolute_calibration_gap", len(predictions)
    )
    top_ece = _ece_from_bins(top_bins, "races", "absolute_calibration_gap", len(race_audit))

    season_metrics: dict[str, Any] = {}
    for year in validation_years:
        season_predictions = predictions.loc[predictions["season"] == year]
        season_races = race_audit.loc[race_audit["season"] == year]
        season_candidate_bins = build_candidate_reliability_bins(
            season_predictions, bins=bins
        )
        season_top_bins = build_top_choice_reliability_bins(season_races, bins=bins)
        season_metrics[str(year)] = {
            "race_metrics": evaluate_predictions(season_predictions),
            "candidate_calibration_ece": _ece_from_bins(
                season_candidate_bins,
                "candidate_rows",
                "absolute_calibration_gap",
                len(season_predictions),
            ),
            "top_choice_calibration_ece": _ece_from_bins(
                season_top_bins, "races", "absolute_calibration_gap", len(season_races)
            ),
            "low_probability_actual_winner_races": int(
                season_races["low_probability_actual_winner"].sum()
            ),
            "high_confidence_wrong_races": int(
                season_races["high_confidence_wrong"].sum()
            ),
        }

    summary = {
        "experiment": "xgboost_expanding_window_prediction_error_and_calibration_audit",
        "validation_years": list(validation_years),
        "validation_races": int(len(race_audit)),
        "driver_race_candidate_rows": int(len(predictions)),
        "primary_metric": "race-level log_loss",
        "configuration": {
            "features": list(FEATURE_COLUMNS),
            "confidence_threshold": float(confidence_threshold),
            "low_winner_probability_threshold": float(low_winner_probability),
            "calibration_bins": int(bins),
            "probability_method": (
                "Binary XGBoost scores normalized within each race; identical "
                "to the existing walk-forward champion architecture."
            ),
        },
        "overall_race_metrics": evaluate_predictions(predictions),
        "candidate_probability_calibration": {
            "ece": candidate_ece,
            "definition": (
                "Count-weighted absolute gap between mean predicted probability "
                "and observed winner frequency across all driver-race candidates."
            ),
        },
        "top_choice_calibration": {
            "ece": top_ece,
            "definition": (
                "Count-weighted absolute gap between mean top-driver probability "
                "and top-pick hit rate across races."
            ),
        },
        "season_metrics": season_metrics,
        "error_counts": {
            "low_probability_actual_winner_races": int(
                race_audit["low_probability_actual_winner"].sum()
            ),
            "high_confidence_wrong_races": int(
                race_audit["high_confidence_wrong"].sum()
            ),
        },
        "high_confidence_wrong_races": int(
            race_audit["high_confidence_wrong"].sum()
        ),
        "low_probability_actual_winner_races": int(
            race_audit["low_probability_actual_winner"].sum()
        ),
        "worst_misses": race_audit.sort_values(
            ["actual_winner_probability", "season", "round"]
        ).head(15).to_dict(orient="records"),
        "largest_confident_errors": race_audit.loc[
            race_audit["high_confidence_wrong"]
        ].sort_values(
            ["predicted_winner_probability", "season", "round"],
            ascending=[False, True, True],
        ).head(15).to_dict(orient="records"),
        "holdout_policy": (
            "The input is restricted to seasons through 2024. 2025 is reserved "
            "and is not used for fitting, evaluation, calibration or model selection."
        ),
        "interpretation_warning": (
            "Calibration bins are descriptive diagnostics, not a fitted calibrator. "
            "Do not fit calibration parameters on these out-of-fold outcomes and "
            "then report performance on the same folds."
        ),
    }
    return summary, predictions, race_audit, candidate_bins


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit expanding-window XGBoost winner probabilities, race errors "
            "and calibration. The 2025 final holdout is excluded."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/features/pre_race_driver_features.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pre_race_prediction_audit"),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.50)
    parser.add_argument("--low-winner-probability", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    summary, predictions, race_audit, _ = run_prediction_audit(
        frame,
        confidence_threshold=args.confidence_threshold,
        low_winner_probability=args.low_winner_probability,
        bins=args.bins,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "driver_race_probabilities.csv"
    race_audit_path = args.output_dir / "race_error_audit.csv"
    candidate_bins_path = args.output_dir / "candidate_calibration_bins.csv"
    top_bins_path = args.output_dir / "top_choice_calibration_bins.csv"
    summary_path = args.output_dir / "audit_summary.json"

    predictions.to_csv(predictions_path, index=False)
    race_audit.to_csv(race_audit_path, index=False)
    build_candidate_reliability_bins(predictions, bins=args.bins).to_csv(
        candidate_bins_path, index=False
    )
    build_top_choice_reliability_bins(race_audit, bins=args.bins).to_csv(
        top_bins_path, index=False
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("PRE-RACE PREDICTION ERROR AND CALIBRATION AUDIT")
    print(f"Validation seasons: {summary['validation_years']}")
    print(f"Validation races: {summary['validation_races']}")
    print(f"Driver-race rows: {summary['driver_race_candidate_rows']}")
    print(f"Log Loss: {summary['overall_race_metrics']['log_loss']:.4f}")
    print(f"Brier: {summary['overall_race_metrics']['multiclass_brier_score']:.4f}")
    print(f"Low-probability winner misses: {summary['low_probability_actual_winner_races']}")
    print(f"High-confidence wrong picks: {summary['high_confidence_wrong_races']}")
    print(f"Candidate calibration ECE: {summary['candidate_probability_calibration']['ece']:.4f}")
    print(f"Top-choice calibration ECE: {summary['top_choice_calibration']['ece']:.4f}")
    print(f"Driver probabilities: {predictions_path}")
    print(f"Race error audit: {race_audit_path}")
    print(f"Calibration bins: {candidate_bins_path}, {top_bins_path}")
    print(f"Summary: {summary_path}")
    print("2025 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
