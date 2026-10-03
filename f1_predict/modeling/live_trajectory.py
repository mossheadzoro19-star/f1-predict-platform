from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1_predict.modeling.live_features import LiveFeatureContract
from f1_predict.modeling.live_split import chronological_live_race_split
from f1_predict.modeling.live_xgboost import fit_live_xgboost
from f1_predict.modeling.live_baseline import predict_live_probabilities


@dataclass(frozen=True)
class RaceTrajectoryMetrics:
    session_key: str
    first_correct_leader_lap: int | None
    persistent_correct_leader_lap: int | None
    predicted_leader_changes: int
    max_winner_probability: float
    final_winner_probability: float
    winner_probability_volatility: float
    mean_entropy: float


def _snapshot_table(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"session_key","lap_number","driver_number","winner","win_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing trajectory columns: {sorted(missing)}")

    frame = predictions.copy()
    frame["win_probability"] = pd.to_numeric(frame["win_probability"], errors="raise")
    rows = []
    for (session_key, lap), group in frame.groupby(["session_key", "lap_number"], sort=True):
        group = group.sort_values("driver_number")
        winner_rows = group[group["winner"] == 1]
        if len(winner_rows) != 1:
            continue
        leader = group.loc[group["win_probability"].idxmax()]
        entropy = float(-(group["win_probability"] * np.log(np.clip(group["win_probability"], 1e-12, 1))).sum())
        winner_probability = float(winner_rows["win_probability"].iloc[0])
        rows.append({
            "session_key": session_key,
            "lap_number": int(lap),
            "leader": int(leader["driver_number"]),
            "winner_driver": int(winner_rows["driver_number"].iloc[0]),
            "winner_probability": winner_probability,
            "entropy": entropy,
        })
    return pd.DataFrame(rows)


def _persistent_lap(table: pd.DataFrame, winner: int) -> int | None:
    ordered = table.sort_values("lap_number")
    for idx in range(len(ordered)):
        suffix = ordered.iloc[idx:]
        if (suffix["leader"] == winner).all():
            return int(ordered.iloc[idx]["lap_number"])
    return None


def evaluate_trajectories(predictions: pd.DataFrame) -> pd.DataFrame:
    table = _snapshot_table(predictions)
    if table.empty:
        raise ValueError("No scorable snapshots available")

    output = []
    for session_key, race in table.groupby("session_key", sort=True):
        race = race.sort_values("lap_number").reset_index(drop=True)
        winner = int(race["winner_driver"].iloc[0])
        leaders = race["leader"].to_numpy()
        changes = int(np.sum(leaders[1:] != leaders[:-1]))

        correct = race.index[race["leader"] == winner]
        first_correct = int(race.loc[correct[0], "lap_number"]) if len(correct) else None
        persistent = _persistent_lap(race, winner)

        winner_probs = race["winner_probability"].to_numpy(dtype=float)
        volatility = float(np.mean(np.abs(np.diff(winner_probs)))) if len(winner_probs) > 1 else 0.0

        output.append(RaceTrajectoryMetrics(
            session_key=str(session_key),
            first_correct_leader_lap=first_correct,
            persistent_correct_leader_lap=persistent,
            predicted_leader_changes=changes,
            max_winner_probability=float(winner_probs.max()),
            final_winner_probability=float(winner_probs[-1]),
            winner_probability_volatility=volatility,
            mean_entropy=float(race["entropy"].mean()),
        ).__dict__)

    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze live winner-probability trajectories.")
    parser.add_argument("--data", default="data/processed/features/openf1_race_state.parquet")
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2024)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    split = chronological_live_race_split(frame, args.train_end_year, args.validation_end_year)
    contract = LiveFeatureContract.default()

    model = fit_live_xgboost(
        pd.concat([split.train, split.validation], ignore_index=True),
        contract,
    )
    predictions = predict_live_probabilities(model, split.test, contract)
    result = evaluate_trajectories(predictions)

    print("LIVE XGBOOST PROBABILITY TRAJECTORY ANALYSIS")
    print(f"TEST RACES: {len(result)}")
    print(f"FIRST CORRECT LEADER: {result['first_correct_leader_lap'].notna().mean():.4f}")
    print(f"PERSISTENT CORRECT LEADER: {result['persistent_correct_leader_lap'].notna().mean():.4f}")
    print(f"MEAN PREDICTED LEADER CHANGES: {result['predicted_leader_changes'].mean():.2f}")
    print(f"MEAN MAX WINNER PROBABILITY: {result['max_winner_probability'].mean():.4f}")
    print(f"MEAN FINAL WINNER PROBABILITY: {result['final_winner_probability'].mean():.4f}")
    print(f"MEAN WINNER-PROBABILITY VOLATILITY: {result['winner_probability_volatility'].mean():.4f}")
    print(f"MEAN SNAPSHOT ENTROPY: {result['mean_entropy'].mean():.4f}")
    print("\nRACES WITH NO PERSISTENT CORRECT LEADER")
    print(result[result["persistent_correct_leader_lap"].isna()][
        ["session_key","first_correct_leader_lap","predicted_leader_changes","max_winner_probability","final_winner_probability"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
