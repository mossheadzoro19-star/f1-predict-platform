# Roadmap

## NOW — Historical race-state milestone
- [x] Historical Jolpica season ingestion and normalization
- [x] Point-in-time pre-race feature pipeline
- [x] Pre-race leakage audit
- [x] Old OpenF1-derived race-state dataset and baseline experiments
- [x] Enriched live feature contract
- [x] Jolpica race-state ingestion implementation
- [ ] Generate/refresh 2023–2025 Jolpica race-state dataset
- [ ] Audit Jolpica race-state coverage, reconstructed timing, labels, duplicates, and missingness
- [ ] Verify point-in-time integrity of reconstructed snapshots
- [ ] Compare old OpenF1-derived benchmark against Jolpica-derived data using the same evaluation protocol
- [ ] Lock useful/weak features using untouched 2025 test
- [ ] Keep FastF1 optional unless a compliant, no-cost data path is available

## NEXT — Free live provider
- [ ] Select and validate free-access live timing provider
- [ ] Implement provider adapter
- [ ] Live freshness/stale detection
- [ ] Live inference validation

## NEXT — Replay and product
- [ ] Historical replay engine
- [ ] Replay probability trajectories
- [ ] FastAPI production path
- [ ] WebSocket updates
- [ ] Pixel-grid F1 dashboard

## LATER — Research
- [ ] Advanced calibration
- [ ] Temporal models only if justified by controlled experiments
- [ ] Driver/circuit representations
- [ ] Deployment and observability hardening

## DEFERRED
- Paid live APIs
- Deep learning without evidence
- Microservices and unnecessary infrastructure
- Treating reconstructed Jolpica snapshots as exact live timing
