# Historical Feature and Evaluation Audit

Status: in progress  
Branch: `fix/historical-feature-contract`

## Goal

Establish that every historical race-state prediction uses only information that would have been available at its stated prediction time. Do not tune against the 2025 holdout while changing the feature contract.

## Known project constraints

- The current enriched race-state archive covers the 2023–2025 seasons.
- Some race timestamps are reconstructed from race metadata and cumulative lap durations; these are approximate and are not official live timing timestamps.
- The archive does not consistently provide weather or tyre compound/age fields. Missing values must not be interpreted as observed values.
- Driver/lap snapshots from the same race are correlated; snapshot-level metrics alone can overstate the amount of independent evidence.
- Earlier held-out 2025 results showed the model slightly behind the current-leader accuracy baseline. Preserve this as a warning, not as a final benchmark, until the evaluation protocol and snapshot coverage are verified.

## Feature audit checklist

For every field in `LIVE_NUMERIC_FEATURES` and `LIVE_CATEGORICAL_FEATURES`, record:

1. Source table/file and source column(s).
2. How it is computed and grouped (race/session, driver, lap).
3. Earliest time it could be known in a real race.
4. Whether its value is observed, reconstructed, or imputed.
5. Missing-value rate overall and by season.
6. Whether it can encode a final result or a later lap.
7. Decision: safe, safe only after a timing shift, unavailable in this archive, or exclude pending evidence.

## High-priority timing checks

- `position_position`, gaps, and intervals: confirm whether they represent end-of-lap values and whether `prediction_time` is after that observation.
- `position_change_*`, `gap_change_*`, and `interval_change_*`: verify shifts use only prior completed laps and do not cross session boundaries.
- `previous_lap_duration` and `previous_3_lap_mean`: verify current-lap timing is excluded.
- `pit_stops_completed` and `laps_since_pit`: verify pit events are only counted after their occurrence is observable.
- `stint_tyre_age_at_start` and `stint_compound`: verify stint metadata does not use future stint changes or retrospectively reconstructed tyre information.
- Weather fields: establish source availability and timestamp alignment; all-missing archive fields should not be treated as useful signals.
- `winner`: target only; never include it or any post-race classification/result field in predictors.

## Evaluation checks

- Keep race sessions disjoint across chronological train, validation, and test splits.
- Freeze the feature contract before the final held-out test.
- Compare model and current-P1 baseline on the exact same scorable snapshots.
- Report scorable snapshot coverage and races included, not accuracy alone.
- Check each snapshot has one eventual winner in the candidate field and probabilities sum to one.
- Report metrics by early/middle/late race phase and by race, as snapshots within a race are not independent.
- Do not adopt temperature calibration unless it improves validation-only proper scoring rules and remains frozen before test evaluation.

## Exit criteria

Do not declare the historical model reliable until the feature timing/source audit, snapshot alignment, leakage tests, and reproducible chronological evaluation are complete. Live prediction remains out of scope until then.
