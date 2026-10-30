# Pre-race prediction error and calibration audit

## Purpose

This audit examines the existing XGBoost champion using chronological out-of-fold (OOF) predictions. It preserves every driver-race probability instead of retaining only a single summary row per race. The outputs support inspection of under-rated winners, confident wrong predictions, and probability reliability.

This is a diagnostic workflow. It does not change model parameters, fit a calibrator, replace the champion, or use the 2025 final holdout.

## Evaluation protocol

The default expanding folds are:

| Training seasons | Validation season |
| --- | --- |
| 2018–2020 | 2021 |
| 2018–2021 | 2022 |
| 2018–2022 | 2023 |
| 2018–2023 | 2024 |

For each fold, the fixed existing XGBoost configuration is fitted only on earlier seasons. Binary driver scores are normalized within each race using the same `_race_probabilities` implementation used by the walk-forward benchmark. The feature set and model configuration are not tuned here.

The input is rejected if it contains any season after 2024. The 2025 season remains reserved and must not be loaded, evaluated, calibrated against, or used for model selection.

## Mathematical definitions

For race (r), driver (i), and XGBoost raw score (s_{ir}), the normalized probability is:

\[
p_{ir}=\frac{s_{ir}}{\sum_{j\in r}s_{jr}}
\]

If all scores in a race sum to zero, the existing implementation falls back to a uniform distribution.

### Race-level Log Loss

Let (w(r)) denote the actual winner of race (r):

\[
\operatorname{LogLoss}
=-\frac{1}{R}\sum_{r=1}^{R}\log(p_{w(r),r})
\]

### Multiclass Brier score

With binary winner indicator (y_{ir}\in\{0,1\}):

\[
\operatorname{Brier}
=\frac{1}{R}\sum_{r=1}^{R}
\sum_{i\in r}(p_{ir}-y_{ir})^2
\]

### Candidate-level reliability

All driver-race candidates are grouped into probability bins (B_k). For each bin:

\[
\operatorname{MeanPred}_k
=\frac{1}{|B_k|}\sum_{(i,r)\in B_k}p_{ir}
\]

\[
\operatorname{Observed}_k
=\frac{1}{|B_k|}\sum_{(i,r)\in B_k}y_{ir}
\]

The candidate-level expected calibration error is:

\[
\operatorname{ECE}_{candidate}
=\sum_k\frac{|B_k|}{N}
\left|\operatorname{MeanPred}_k-\operatorname{Observed}_k\right|
\]

Here (N) is the number of driver-race candidate rows. This diagnostic treats each candidate row as one binary outcome. Rows within the same race are dependent, so it must not be interpreted as an independent-sample confidence interval.

### Top-choice reliability

For each race, let (c_r=\max_i p_{ir}) and let (a_r=1) if the highest-probability driver is the actual winner, otherwise 0. Ties are resolved deterministically by driver ID in the race error table.

\[
\operatorname{ECE}_{top}
=\sum_k\frac{|T_k|}{R}
\left|\operatorname{mean}_{r\in T_k}(c_r)
-\operatorname{mean}_{r\in T_k}(a_r)\right|
\]

Both ECE values depend on the selected bin count and are descriptive diagnostics, not primary model-selection metrics.

## Error definitions

- **Low-probability actual winner:** actual winner probability is at or below 0.05 by default.
- **High-confidence wrong pick:** predicted winner is incorrect and its probability is at least 0.50 by default.
- **Winner rank:** rank of the actual winner among all drivers, with tied probabilities sharing the minimum rank.
- **Top-two margin:** difference between the largest and second-largest driver probabilities.

Thresholds are configurable command-line arguments. They are review flags, not claims that a probability threshold is universally meaningful.

## Run

From the repository root:

```bash
python -m f1_predict.modeling.pre_race_prediction_audit
```

Optional configuration:

```bash
python -m f1_predict.modeling.pre_race_prediction_audit \
  --confidence-threshold 0.50 \
  --low-winner-probability 0.05 \
  --bins 10
```

## Outputs

Created under `reports/pre_race_prediction_audit/`:

- `driver_race_probabilities.csv`: one OOF probability per driver per validation race, including the training cutoff.
- `race_error_audit.csv`: actual and predicted winner, probabilities, rank, margin, error flags, and selected feature context for both drivers.
- `candidate_calibration_bins.csv`: candidate-level probability reliability bins.
- `top_choice_calibration_bins.csv`: top-pick confidence versus hit-rate bins.
- `audit_summary.json`: aggregate and season-level metrics, error counts, worst misses, and largest confident errors.

## Interpretation and safeguards

1. Compare season-level Log Loss and Brier before interpreting calibration bins.
2. Inspect low-probability winner races and high-confidence wrong picks individually. A single feature association is not evidence of causation.
3. Review whether errors cluster by season, qualifying context, driver form, constructor history, or circuit context. These are hypotheses to inspect, not pre-established explanations.
4. Do not fit a probability calibrator and evaluate it on the same OOF outcomes used to fit it. Any future calibration method must be nested inside the chronological evaluation protocol.
5. Do not tune XGBoost hyperparameters from this audit.
6. Do not load or use 2025 until model, feature, and calibration choices are frozen.

## Scope

This audit uses the existing point-in-time feature table and existing champion model recipe. It does not add data sources, change feature engineering, or introduce live/replay functionality.
