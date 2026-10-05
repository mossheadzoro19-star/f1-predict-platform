# Product Requirements Document

## 1. Product
**Name:** F1 Predict — Race Intelligence Platform  
**Purpose:** Predict each driver's probability of winning an F1 race using leakage-safe historical knowledge and current race-state evidence.

## 2. Problem
Race outcome models often collapse a race into a static pre-race prediction. This platform must update winner probabilities as observable race evidence changes while preserving point-in-time correctness.

## 3. Users
- ML/AI researchers studying probabilistic race prediction.
- F1 users who want an interpretable live/replay race intelligence view.

## 4. Core requirements
1. Historical ingestion from free/appropriate sources.
2. Point-in-time feature generation with leakage audits.
3. Pre-race winner probabilities.
4. Historical race replay.
5. Free-access live timing adapter.
6. Dynamic live winner probabilities without retraining every second.
7. Probability evaluation using log loss, Brier score, calibration and winner accuracy.
8. Honest LIVE / REPLAY / FALLBACK / STALE / DISCONNECTED states.
9. Lightweight dashboard with race order, P(P1), gaps, tyres and weather.

## 5. Scope
### Must have
- Jolpica + FastF1 historical pipeline.
- Race-state dataset for 2023–2025.
- OLD vs ENRICHED XGBoost evaluation.
- Free live-provider adapter behind an interface.
- Replay and API integration.

### Later
- Temporal/deep models only when experiments justify them.
- Advanced calibration.
- Next.js product UI and deployment hardening.

## 6. Non-goals
- Paid API dependency.
- Fake live data.
- Retraining on every live update.
- Random train/test splitting of correlated race snapshots.
- Large infrastructure before product need exists.

## 7. Success metrics
Primary: untouched 2025 log loss.  
Secondary: Brier score, calibration error, winner accuracy and probability trajectory quality.

## 8. Definition of done
Requirements implemented, leakage checks pass, tests pass, 2025 remains untouched during model selection, and provider failures are represented honestly.
