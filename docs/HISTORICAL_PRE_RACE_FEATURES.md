# Enhanced Historical Pre-Race Feature Contract

## Scope

This feature set supports a prediction made after qualifying and before the Grand Prix start. It uses the normalized Jolpica race calendar, race results, and qualifying tables already stored in the repository. The default build ends in 2024; 2025 remains reserved as the final holdout.

## Feature families

### Driver form and reliability
- Prior starts, wins, and win rate.
- Mean finishing position over the last three and five classified finishes.
- Points scored over the last three and five starts.
- Recency-weighted finishing-position form over up to five prior finishes (larger weights go to more recent finishes).
- Prior DNF rate, calculated only from completed previous starts.

### Constructor strength and reliability
- Prior constructor race starts, wins, cumulative points, and win rate.
- Points scored over the previous three and five constructor races.
- Prior constructor DNF rate, using prior driver entries as the denominator.

### Qualifying and grid
- Grid position and qualifying position.
- Q1, Q2, and Q3 times in seconds.
- Q3 gap to the fastest recorded Q3 time in that qualifying session.
- Q3 gap to the best recorded Q3 time by the driver's teammate.
- Field size.

### Circuit experience
- Driver starts at the circuit.
- Mean finishing position over the driver's previous three and five starts at that circuit.

## Point-in-time safeguards

1. All driver rows for a race are constructed before any result from that race updates driver, constructor, or circuit history.
2. Current race position, points, and status are used only for the target and subsequent-race history.
3. Qualifying comparisons are computed only from the current qualifying session, which is available at the pre-race prediction cutoff.
4. Missing qualifying times remain missing and are imputed inside the training pipeline; they are not fabricated.
5. A driver with no prior history receives zero for count/rate features where zero has a defined meaning, and null for unavailable averages.
6. 2025 is not loaded by the default feature build or default validation experiment.

## Interpretation notes

- A positive teammate Q3 gap means the driver was slower than the teammate; a negative gap means faster.
- A positive gap-to-pole means slower than the session's fastest Q3 time.
- Jolpica status values such as `Finished` and `+1 Lap` are treated as classified finishes. Other status strings are counted as non-finishes for the DNF-rate feature. This is a coarse historical reliability signal, not a mechanical-failure label.
- Historical results are season-context sensitive: constructor identifiers change when team identity changes. No attempt is made to merge predecessor/successor teams.
- Weather, tyre degradation, pit-stop execution, safety-car exposure, and telemetry-derived pace are intentionally not included in this pre-race feature set yet. They require separate source-quality and availability checks and may be introduced in later phases.

## Rebuild

```bash
python -m f1_predict.features.pre_race
```

## Train the initial comparison

```bash
python -m f1_predict.modeling.pre_race_model
```

The initial comparison is a baseline experiment, not a production model selection. Evaluate log loss first, then multiclass Brier score and winner accuracy. Keep the final 2025 holdout untouched until the feature and model design is frozen.
