from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RaceStateMetrics:
    """Compact diagnostics for a normalized race-state winner model."""

    winner_accuracy: float
    log_loss: float
    brier_score: float
    snapshots: int
    races: int
    scorable_coverage: float
    current_leader_accuracy: float
    model_vs_leader_accuracy_delta: float


def _score_snapshot(group: pd.DataFrame) -> tuple[float, float, float, bool] | None:
    """Score one snapshot whose eventual winner is still in the candidate field."""
    actual = group.loc[group["winner"] == 1, "driver_number"]
    if len(actual) != 1:
        return None

    winner = actual.iloc[0]
    winner_probability = float(
        group.loc[group["driver_number"] == winner, "win_probability"].iloc[0]
    )
    predicted = group.loc[group["win_probability"].idxmax(), "driver_number"]
    current_leader = pd.to_numeric(
        group["position_position"], errors="coerce"
    ).idxmin()

    probabilities = group["win_probability"].to_numpy(dtype=float)
    target = group["winner"].to_numpy(dtype=float)
    brier = float(np.sum((probabilities - target) ** 2))
    log_loss = -float(np.log(np.clip(winner_probability, 1e-15, 1.0)))

    leader_position = group.loc[current_leader, "position_position"]
    leader_driver = group.loc[current_leader, "driver_number"]
    leader_valid = pd.notna(leader_position)

    return (
        float(predicted == winner),
        log_loss,
        brier,
        bool(leader_valid and leader_driver == winner),
    )


def evaluate_race_state_predictions(predictions: pd.DataFrame) -> RaceStateMetrics:
    """Evaluate normalized winner probabilities and compare them with current P1."""
    required = {
        "session_key",
        "lap_number",
        "driver_number",
        "winner",
        "win_probability",
        "position_position",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")

    rows: list[tuple[float, float, float, bool]] = []
    total = 0
    for _, group in predictions.groupby(["session_key", "lap_number"], sort=False):
        total += 1
        scored = _score_snapshot(group)
        if scored is not None:
            rows.append(scored)

    if not rows:
        raise ValueError("No scorable race-state snapshots")

    values = np.asarray([row[:3] for row in rows], dtype=float)
    leader_hits = np.asarray([row[3] for row in rows], dtype=float)

    return RaceStateMetrics(
        winner_accuracy=float(values[:, 0].mean()),
        log_loss=float(values[:, 1].mean()),
        brier_score=float(values[:, 2].mean()),
        snapshots=len(rows),
        races=int(predictions["session_key"].nunique()),
        scorable_coverage=float(len(rows) / total),
        current_leader_accuracy=float(leader_hits.mean()),
        model_vs_leader_accuracy_delta=float(values[:, 0].mean() - leader_hits.mean()),
    )


def phase_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Break winner accuracy, LogLoss and Brier score into early/mid/late race."""
    frame = predictions.copy()
    frame["race_progress"] = frame.groupby("session_key")["lap_number"].transform(
        lambda s: s / max(float(s.max()), 1.0)
    )
    frame["phase"] = pd.cut(
        frame["race_progress"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "mid", "late"],
    )

    rows: list[dict[str, Any]] = []
    for phase, phase_frame in frame.groupby("phase", observed=True):
        metrics = evaluate_race_state_predictions(phase_frame)
        rows.append(
            {
                "phase": str(phase),
                "snapshots": metrics.snapshots,
                "races": metrics.races,
                "winner_accuracy": metrics.winner_accuracy,
                "log_loss": metrics.log_loss,
                "brier_score": metrics.brier_score,
                "current_leader_accuracy": metrics.current_leader_accuracy,
                "model_vs_leader_accuracy_delta": metrics.model_vs_leader_accuracy_delta,
            }
        )
    return pd.DataFrame(rows)


def final_lap_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:
    """Evaluate the last observed lap of each race separately from all snapshots."""
    frame = predictions.copy()
    max_lap = frame.groupby("session_key")["lap_number"].transform("max")
    final = frame[frame["lap_number"] == max_lap].copy()
    metrics = evaluate_race_state_predictions(final)
    return asdict(metrics)


def confidence_diagnostics(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Check whether the model's top probability corresponds to realized wins."""
    rows: list[dict[str, Any]] = []
    for _, group in predictions.groupby(["session_key", "lap_number"], sort=False):
        actual = group.loc[group["winner"] == 1, "driver_number"]
        if len(actual) != 1:
            continue
        top = group.loc[group["win_probability"].idxmax()]
        rows.append(
            {
                "confidence": float(top["win_probability"]),
                "correct": float(top["driver_number"] == actual.iloc[0]),
            }
        )

    if not rows:
        raise ValueError("No snapshots available for confidence diagnostics")

    frame = pd.DataFrame(rows)
    frame["confidence_bin"] = pd.cut(
        frame["confidence"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
    )
    return (
        frame.groupby("confidence_bin", observed=True)
        .agg(
            snapshots=("correct", "size"),
            mean_confidence=("confidence", "mean"),
            empirical_accuracy=("correct", "mean"),
        )
        .reset_index()
    )


def build_race_state_report(predictions: pd.DataFrame) -> dict[str, Any]:
    """Return one compact report suitable for JSON or notebook display."""
    overall = evaluate_race_state_predictions(predictions)
    return {
        "overall": asdict(overall),
        "phase": phase_diagnostics(predictions).to_dict(orient="records"),
        "final_lap": final_lap_diagnostics(predictions),
        "confidence": confidence_diagnostics(predictions).to_dict(orient="records"),
    }


def evaluate_current_xgboost(
    frame: pd.DataFrame,
    train_end_year: int = 2023,
    validation_end_year: int = 2024,
) -> dict[str, Any]:
    """Run the existing XGBoost model under the frozen chronological protocol."""
    from f1_predict.modeling.live_features import LiveFeatureContract
    from f1_predict.modeling.live_split import chronological_live_race_split
    from f1_predict.modeling.live_xgboost import fit_live_xgboost, predict_live_probabilities

    split = chronological_live_race_split(
        frame,
        train_end_year=train_end_year,
        validation_end_year=validation_end_year,
    )
    contract = LiveFeatureContract.default()

    validation_model = fit_live_xgboost(split.train, contract)
    validation_predictions = predict_live_probabilities(
        validation_model, split.validation, contract
    )

    final_train = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = fit_live_xgboost(final_train, contract)
    test_predictions = predict_live_probabilities(
        final_model, split.test, contract
    )

    return {
        "validation": build_race_state_report(validation_predictions),
        "test": build_race_state_report(test_predictions),
        "split": {
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.validation)),
            "test_rows": int(len(split.test)),
            "train_races": int(split.train["session_key"].nunique()),
            "validation_races": int(split.validation["session_key"].nunique()),
            "test_races": int(split.test["session_key"].nunique()),
        },
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Cross-check current live XGBoost against a current-P1 baseline."
    )
    parser.add_argument(
        "--data",
        default="data/processed/features/openf1_race_state.parquet",
    )
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2024)
    args = parser.parse_args()

    report = evaluate_current_xgboost(
        pd.read_parquet(args.data),
        train_end_year=args.train_end_year,
        validation_end_year=args.validation_end_year,
    )

    for split_name in ("validation", "test"):
        overall = report[split_name]["overall"]
        print(f"\n{split_name.upper()}")
        print(f"  winner accuracy       = {overall['winner_accuracy']:.4f}")
        print(f"  current P1 accuracy   = {overall['current_leader_accuracy']:.4f}")
        print(f"  model - P1 delta      = {overall['model_vs_leader_accuracy_delta']:+.4f}")
        print(f"  log loss              = {overall['log_loss']:.4f}")
        print(f"  brier score           = {overall['brier_score']:.4f}")
        print(f"  scorable coverage     = {overall['scorable_coverage']:.4f}")

        print("  phase:")
        for row in report[split_name]["phase"]:
            print(
                f"    {row['phase']:<5} "
                f"acc={row['winner_accuracy']:.4f} "
                f"p1={row['current_leader_accuracy']:.4f} "
                f"LL={row['log_loss']:.4f} "
                f"Brier={row['brier_score']:.4f}"
            )

        final = report[split_name]["final_lap"]
        print(
            "  final lap: "
            f"acc={final['winner_accuracy']:.4f} "
            f"p1={final['current_leader_accuracy']:.4f} "
            f"LL={final['log_loss']:.4f} "
            f"Brier={final['brier_score']:.4f}"
        )

    print("\nJSON REPORT")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
