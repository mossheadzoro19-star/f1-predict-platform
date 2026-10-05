# Architecture Decisions

## 001 — Free historical sources
**Status:** Accepted  
Use Jolpica + FastF1 for historical data.  
**Reason:** Jolpica provides structured historical records; FastF1 provides richer session/lap/telemetry processing.

## 002 — No paid API dependency
**Status:** Accepted  
A paid OpenF1 plan is not required by the platform.  
**Reason:** The project goal requires a free-access live path.

## 003 — Historical replay
**Status:** Accepted  
Replay is a first-class validation and product mode.  
**Reason:** It enables realistic point-in-time evaluation without pretending historical data is live.

## 004 — Chronological evaluation
**Status:** Accepted  
Train 2023, validate 2024, test 2025.  
**Reason:** Race snapshots are temporally correlated and future seasons must remain unseen.

## 005 — XGBoost before deep learning
**Status:** Accepted  
Use strong tabular baselines before temporal/deep models.  
**Reason:** Complexity must be justified by measurable improvement.

## 006 — Honest provider state
**Status:** Accepted  
Never label stale or fallback output as LIVE.  
**Reason:** Trustworthiness is a product requirement.
