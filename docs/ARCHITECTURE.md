# Architecture

## 1. System overview
F1 sources → adapters → raw storage → normalization → point-in-time features → models → prediction service → FastAPI → UI.

## 2. Pattern
**Modular Python application.** Keep source adapters, feature engineering, modeling and serving separated without introducing microservices prematurely.

## 3. Sources
- Jolpica: structured historical results/schedule/qualifying.
- FastF1: richer historical lap/session/telemetry/weather processing.
- Free-access live timing provider: live state behind an adapter.
- Historical replay: derived from stored historical race-state data.

## 4. Components
### Data sources
Own provider-specific API/file behavior.

### Ingestion
Downloads and stores raw provider responses.

### Normalization
Converts provider data into stable schemas.

### Features
Builds point-in-time historical and race-state features.

### Modeling
Trains/evaluates pre-race and live-state models.

### Live service
Combines current race state with trained probabilities.

### API/UI
Expose state and predictions; never embed provider credentials in frontend code.

## 5. Data flow
1. Retrieve source data.
2. Preserve raw data separately.
3. Normalize.
4. Build only information available at prediction time.
5. Train chronologically.
6. Evaluate on untouched later seasons.
7. Serve inference.
8. Update UI from current/replay snapshots.

## 6. Dependency rules
- UI does not access providers directly.
- Models do not perform provider I/O.
- Provider adapters do not contain model logic.
- Feature generation must remain point-in-time.
- No paid provider becomes mandatory without an explicit project decision.

## 7. Performance
Prefer vectorized pandas operations, parquet, cached historical data and batch inference. Do not call external APIs once per driver if one request can provide the required snapshot.

## 8. Failure behavior
Provider failure must surface as stale/disconnected/fallback. Never label cached historical output as LIVE.
