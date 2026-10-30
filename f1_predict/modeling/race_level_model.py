from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from f1_predict.modeling.pre_race_model import (
    FEATURE_COLUMNS, RACE_KEYS, ROW_KEYS, validate_dataset,
)


@dataclass
class ConditionalLogitRaceModel:
    """Race-grouped conditional logistic regression with a softmax output."""

    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    l2_strength: float = 1.0
    max_iter: int = 2000
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if self.l2_strength < 0:
            raise ValueError("l2_strength must be non-negative")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.coef_: np.ndarray | None = None
        self.optimization_: dict[str, Any] | None = None

    @staticmethod
    def _groups(frame: pd.DataFrame) -> list[np.ndarray]:
        groups = [
            np.asarray(indices, dtype=int)
            for indices in frame.groupby(RACE_KEYS, sort=False).indices.values()
        ]
        if not groups:
            raise ValueError("No race groups found")
        return groups

    @staticmethod
    def _validate_races(frame: pd.DataFrame, require_target: bool) -> None:
        required = set(ROW_KEYS + (["winner"] if require_target else []))
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Race frame is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Race frame is empty")
        if frame.duplicated(ROW_KEYS).any():
            raise ValueError("Race frame contains duplicate driver-race rows")
        if require_target:
            if frame["winner"].isna().any() or not frame["winner"].isin([0, 1]).all():
                raise ValueError("winner must be a non-null binary target")
            counts = frame.groupby(RACE_KEYS)["winner"].sum()
            if not counts.eq(1).all():
                raise ValueError("Every race must contain exactly one winner")

    def _matrix(self, frame: pd.DataFrame, fitting: bool) -> np.ndarray:
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Feature frame is missing columns: {sorted(missing)}")
        values = frame.loc[:, self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if fitting:
            self.imputer_ = SimpleImputer(
                strategy="median", keep_empty_features=True
            )
            imputed = self.imputer_.fit_transform(values)
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(imputed)
        if self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.scaler_.transform(self.imputer_.transform(values))

    def _objective(
        self,
        beta: np.ndarray,
        matrix: np.ndarray,
        groups: list[np.ndarray],
        targets: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        loss = 0.0
        gradient = np.zeros_like(beta)
        for indices in groups:
            group_x = matrix[indices]
            group_y = targets[indices]
            utilities = group_x @ beta
            probabilities = softmax(utilities)
            loss += logsumexp(utilities) - float(group_y @ utilities)
            gradient += group_x.T @ (probabilities - group_y)

        race_count = len(groups)
        loss /= race_count
        gradient /= race_count
        loss += 0.5 * self.l2_strength * float(beta @ beta)
        gradient += self.l2_strength * beta
        return float(loss), gradient

    def fit(self, frame: pd.DataFrame) -> "ConditionalLogitRaceModel":
        self._validate_races(frame, require_target=True)
        validate_dataset(frame)
        matrix = self._matrix(frame, fitting=True)
        groups = self._groups(frame)
        targets = frame["winner"].to_numpy(dtype=float)
        initial = np.zeros(matrix.shape[1], dtype=float)

        result = minimize(
            fun=lambda beta: self._objective(beta, matrix, groups, targets),
            x0=initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tolerance},
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("Conditional-logit optimization produced non-finite values")
        if not result.success:
            raise RuntimeError(
                f"Conditional-logit optimization did not converge: {result.message}"
            )

        self.coef_ = np.asarray(result.x, dtype=float)
        self.optimization_ = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "gradient_norm": float(np.linalg.norm(result.jac)),
            "l2_strength": float(self.l2_strength),
        }
        return self

    def predict_race_probabilities(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._validate_races(frame, require_target=False)
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        matrix = self._matrix(frame, fitting=False)
        probabilities = np.empty(len(frame), dtype=float)
        for indices in self._groups(frame):
            probabilities[indices] = softmax(matrix[indices] @ self.coef_)

        result = frame.loc[:, ROW_KEYS].copy()
        if "winner" in frame.columns:
            result["winner"] = frame["winner"].to_numpy()
        result["win_probability"] = probabilities
        return result
