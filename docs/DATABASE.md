# Data Model

This project currently uses Parquet as the primary analytical storage format rather than a transactional database.

## 1. Storage layers
- `data/raw/<provider>/<year>/`: provider responses.
- `data/processed/jolpica/<year>/`: normalized historical tables.
- `data/processed/features/`: point-in-time model features.
- `data/processed/race_state/`: historical race-state snapshots.

## 2. Core entities
### races
Race/session identity, year, round, circuit and timestamps.

### drivers
Stable driver identifiers, numbers and names.

### constructors
Stable constructor identifiers and names.

### qualifying
Driver qualifying result and session timing available before race start.

### race_results
Final race result used as target, never as a feature at prediction time.

### race_state_snapshots
Point-in-time driver state: session, timestamp/lap, position, gaps, tyre/stint state, pit state and weather.

## 3. Keys
Race-state snapshot key: session + driver + prediction timestamp/lap boundary.

## 4. Integrity
- One winner per race.
- No duplicate race/driver/time snapshots.
- Prediction timestamp must not occur after the information used.
- Final race result is target only.
