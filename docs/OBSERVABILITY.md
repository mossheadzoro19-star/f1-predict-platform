# Observability

## Logs
Record:
- provider state changes
- request failures
- prediction latency
- replay/live mode
- model version
- data freshness

Never record secrets.

## Metrics
Track:
- API latency
- provider latency
- provider error rate
- stale duration
- snapshot update rate
- prediction generation latency
- model version usage

## ML monitoring
Track live/replay distributions for:
- position
- gap
- tyre state
- probability entropy
- winner probability
- missingness

Do not silently retrain from production observations.
