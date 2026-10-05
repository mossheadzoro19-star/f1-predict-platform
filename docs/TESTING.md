# Testing Strategy

## Goals
Protect point-in-time correctness, provider boundaries, model evaluation and API state semantics.

## Test types
### Unit
- Feature transformations.
- Leakage rules.
- Provider parsing.
- Probability normalization.
- Replay offsets.

### Integration
- Historical ingestion.
- FastF1/Jolpica adapter integration where practical.
- API snapshot path.

### Model evaluation
Use chronological:
- Train: 2023
- Validation: 2024
- Untouched test: 2025

Primary metric: LogLoss. Secondary: Brier, winner accuracy and calibration.

## Critical checks
- One winner per race.
- No duplicate snapshots.
- Monotonic timestamps within session.
- No future information in features.
- Prediction probabilities sum to 1 within each snapshot where required.
- LIVE is never returned when the source is stale/disconnected.

## Commands
```bash
pytest -q
ruff check .
mypy f1_predict
```

## Definition of done
Tests pass, lint/type checks pass where configured, model comparison uses the fixed protocol, and no test-set tuning leaks into model selection.
