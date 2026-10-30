# Historical pre-race pipeline — Phase 1

## Goal

Create a clean, reproducible driver-race training table for race-winner prediction using only information available before each Grand Prix starts.

## Scope

- Inputs: existing normalized Jolpica season Parquet files.
- Default seasons: 2018–2024.
- Output: `data/processed/features/pre_race_driver_features.parquet`.
- Unit of observation: one driver entry in one race.
- Target: `winner`, derived from final classified position.
- 2025 is deliberately excluded by default and remains the final holdout.

## Point-in-time rules

Every feature row is constructed before the current race's results are added to historical state. The result position is used only as the target and to update driver, constructor and circuit history after all rows for that race have been created.

Included feature groups:

- Pre-race grid and qualifying position/times.
- Driver prior starts, wins, win rate, recent finishes and recent points.
- Constructor prior starts, wins, cumulative points and win rate.
- Driver prior experience and recent finishes at the current circuit.

Race outcome fields such as current-race finish position, points, status, laps completed, race time and fastest lap are not model features.

## Build

From the repository root:

```bash
python -m f1_predict.features.pre_race
```

Optional explicit range and output:

```bash
python -m f1_predict.features.pre_race \
  --start-year 2018 \
  --end-year 2024 \
  --output data/processed/features/pre_race_driver_features.parquet
```

## Tests

```bash
pytest -q tests/test_pre_race_features.py
```

## Important boundaries

This is the pre-race winner-probability dataset. It does not use lap timing or pit stops because those occur after the race begins. The TracingInsights archive will be handled in a separate historical race-state/replay phase with explicit timing semantics and data-quality checks.

No existing raw, normalized or legacy data files are overwritten. The generated Parquet output is local and ignored by Git.
