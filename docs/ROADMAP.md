# Roadmap

## Milestone 1 — Data foundation
- Provider adapters
- Raw historical ingestion
- Provenance and licensing notes
- Codespaces environment

## Milestone 2 — Historical dataset
- Normalize races, drivers, constructors, qualifying and results
- Add sprint and pit-stop history where appropriate
- Build point-in-time feature snapshots
- Establish chronological train/validation/test splits
- Add schema/data-quality checks

## Milestone 3 — Baselines
- Grid-position baseline
- Driver/constructor baseline
- Logistic regression
- Gradient-boosted trees
- Log loss, Brier score, calibration and winner accuracy

## Milestone 4 — Historical replay
- Reconstruct race state lap by lap
- Add lap timing, stints, tyres, pit stops and weather
- Generate predictions at multiple timestamps
- Measure probability trajectory quality

## Milestone 5 — Live inference
- OpenF1 live adapter
- Event-driven update pipeline
- Prediction service
- WebSocket stream
- Stale-data detection

## Milestone 6 — Research
- Temporal models
- Driver/circuit representations
- Advanced calibration
- Ablation studies

## Milestone 7 — Product
- Next.js interface
- Race dashboard
- Replay UI
- Deployment and observability
