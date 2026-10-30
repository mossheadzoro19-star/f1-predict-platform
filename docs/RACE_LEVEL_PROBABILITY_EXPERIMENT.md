# Race-level probability architecture experiment

## Purpose and status

This is a separate research experiment. It does not replace the existing pre-race XGBoost baseline or overwrite its artifacts.

Research question: does directly optimizing the probability of the winner within each race improve race-level probabilities compared with independently estimating driver win scores and normalizing them afterward?

The experiment uses the existing point-in-time pre-race features and expanding chronological folds. It rejects datasets containing seasons after 2024.

## Architecture A: normalized binary XGBoost

For driver i in race r, the existing binary classifier estimates a driver-level score:

    s_ir = estimated P(Y_ir = 1 | x_ir)

The scores are normalized within each race:

    p_hat_ir = s_ir / sum(j in race r) s_jr

This creates a probability distribution, but the binary training objective is not a race-level conditional likelihood.

## Architecture B: conditional logistic regression

### Utility

Let x_ir in R^p be the pre-race feature vector for driver i in race r, and beta in R^p be shared coefficients.

    u_ir = x_ir^T beta

There is no global intercept. A term constant across every driver in a race cancels in the within-race softmax and is not identifiable.

### Race-level winner probability

    P(Y_ir = 1 | X_r) = exp(u_ir) / sum(j in race r) exp(u_jr)

For each race:

    sum(i in race r) P(Y_ir = 1 | X_r) = 1

The implementation uses numerically stable softmax and log-sum-exp calculations.

### Conditional likelihood

Let w(r) denote the actual winner of race r and R the number of training races.

    ell(beta) = sum over races r [
        u_w(r),r - log(sum(j in race r) exp(u_jr))
    ]

The optimized objective is mean negative log-likelihood with L2 regularization:

    L(beta) = -(1/R) ell(beta) + (lambda/2) ||beta||_2^2

Here lambda >= 0 is the fixed l2_strength argument. The initial experiment uses lambda = 1.0. It is not tuned against the validation folds.

### Analytic gradient

Let p_r be the vector of predicted probabilities for race r, y_r the one-hot winner vector, and X_r the matrix of driver features for that race.

    gradient L(beta) = (1/R) sum over races r [X_r^T (p_r - y_r)] + lambda beta

The objective and analytic gradient are optimized using L-BFGS-B.

### Preprocessing and grouping

- Numeric conversion is explicit.
- Missing values are median-imputed using statistics fitted only on the fold's training rows.
- Standardization is fitted only on the fold's training rows.
- No global intercept is fitted.
- Both architectures use the same declared feature columns.
- Race groups are defined by (season, round).
- A feature constant among all drivers in a race cannot distinguish candidates under the conditional softmax; its common contribution cancels. This is an inherent property of the objective.

## Evaluation protocol

Expanding chronological folds:

| Training seasons | Validation season |
| --- | --- |
| 2018–2020 | 2021 |
| 2018–2021 | 2022 |
| 2018–2022 | 2023 |
| 2018–2023 | 2024 |

For each fold, preprocessing is fitted on training seasons only. Both models use identical training and validation rows.

Compared models:
1. Uniform race probabilities.
2. Existing XGBoost binary classifier followed by within-race score normalization.
3. Conditional logistic regression trained with grouped race winner likelihood.

Primary metric: race-level Log Loss.

    LogLoss = -(1/R) sum over races r log(p_hat_w(r),r)

Secondary metrics:

    Brier = (1/R) sum over races r sum over drivers i (p_hat_ir - y_ir)^2

Winner accuracy is the fraction of races where the highest-probability driver is the actual winner. Ties receive fractional credit, consistent with the existing evaluator.

Lower Log Loss and Brier are better. Results must be examined per season, not only in aggregate. An aggregate improvement with severe seasonal degradation is not sufficient evidence of robustness.

## Run

    python -m f1_predict.modeling.race_level_experiment

Optional arguments:

    python -m f1_predict.modeling.race_level_experiment \
      --data data/processed/features/pre_race_driver_features.parquet \
      --l2-strength 1.0 \
      --report-output reports/race_level_experiment/metrics.json \
      --predictions-output reports/race_level_experiment/race_predictions.csv

Outputs:
- reports/race_level_experiment/metrics.json: fold metrics, combined out-of-fold metrics, optimizer diagnostics and experiment metadata.
- reports/race_level_experiment/race_predictions.csv: one row per race and model, with actual/predicted winner audit fields.

## Decision criteria and limitations

- Keep the existing XGBoost baseline unchanged.
- Do not tune lambda using these same validation folds.
- Do not use 2025 for training, validation, model selection or tuning.
- Conditional logistic regression is a linear utility model. It may underfit nonlinear driver/team interactions; this isolates the objective/probability architecture, not model capacity.
- This experiment does not prove calibration by itself. Calibration diagnostics remain a separate task.
- This is a research comparison, not production model promotion.

## Holdout policy

The runner rejects any input containing a season after 2024 before fitting models. The 2025 season remains reserved as the final untouched holdout until features, architecture, regularization and model-selection methodology are frozen.
