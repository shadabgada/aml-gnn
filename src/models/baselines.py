"""Conventional supervised classifiers for AML edge classification baseline.

Provides unified wrappers for Logistic Regression, Random Forest, and XGBoost.
These models operate on flat edge feature vectors — they have no access to
graph structure, neighbor information, or transaction ordering. This makes
them the "conventional ML" baseline against which GNNs are compared (SQ3).

All models handle severe class imbalance via class weights (LR, RF) or
scale_pos_weight (XGBoost), computed from the training set.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """Abstract interface for baseline classifiers."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on (N, F) features and (N,) binary labels."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probability of the positive (laundering) class, shape (N,)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for logging."""


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------


class LogisticRegressionBaseline(BaseClassifier):
    """L2-regularised logistic regression with balanced class weights."""

    def __init__(self, pos_weight: float, max_iter: int = 1000):
        # class_weight='balanced' auto-computes weights from y.
        # We additionally allow manual scaling via pos_weight on the positive class.
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        )
        self._pos_weight = pos_weight
        self._name = "LogisticRegression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # class_weight='balanced' handles class imbalance — no extra sample_weight
        self.model.fit(X, y)
        logger.info("%s trained on %d samples", self._name, len(y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class RandomForestBaseline(BaseClassifier):
    """Random Forest with class-balanced subsampling."""

    def __init__(self, pos_weight: float, n_estimators: int = 200):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            max_depth=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        self._pos_weight = pos_weight
        self._name = "RandomForest"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # class_weight='balanced' handles class imbalance — no extra sample_weight
        self.model.fit(X, y)
        logger.info("%s trained on %d samples", self._name, len(y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class XGBoostBaseline(BaseClassifier):
    """Gradient-boosted trees with scale_pos_weight for class imbalance."""

    def __init__(
        self,
        pos_weight: float,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self._name = "XGBoost"

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> None:
        eval_set = [(X_val, y_val)] if X_val is not None else [(X, y)]
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info("%s trained on %d samples", self._name, len(y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_baselines(pos_weight: float) -> list[BaseClassifier]:
    """Create all three baseline classifiers with the given pos_weight."""
    return [
        LogisticRegressionBaseline(pos_weight=pos_weight),
        RandomForestBaseline(pos_weight=pos_weight),
        XGBoostBaseline(pos_weight=pos_weight),
    ]
