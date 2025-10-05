import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import joblib
from lightning import LightningModule
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from numerlab.utils.types import Metrics, Path_

log = logging.getLogger(__name__)
Criterion = Callable[
    [
        ArrayLike,  # t (target)
        ArrayLike,  # p (preds)
        ArrayLike,  # m (meta)
        ArrayLike,  # e (era)
    ],
    float,
]


class SKLearnModule:
    """A wrapper class for scikit-learn models with Numerai-specific
    functionality.

    This class provides a unified interface for scikit-learn estimators, adding
    support for Numerai-specific metrics, model persistence, and evaluation
    methods. It handles training state tracking and provides validation/testing
    capabilities with proper metric computation.

    Attributes:
        model: The underlying scikit-learn estimator
        metrics: List of metric functions for evaluation
        _trained: Internal flag tracking training state
    """

    def __init__(self, model: BaseEstimator, metrics: list[Criterion]):
        """Initialize a SKLearnModule.

        Args:
            model: A scikit-learn compatible estimator
            metrics: List of metric functions, each taking (target, predictions, meta, eras)
                    and returning a float
        """
        self.model = model
        self.metrics = metrics
        self._trained = False

    @property
    def trained(self) -> bool:
        """Whether the model has been trained.

        Returns:
            bool: True if the model has been trained, False otherwise
        """
        return self._trained

    def __call__(self, X: ArrayLike) -> ArrayLike:
        """Make predictions on input data X.

        Args:
            X (ArrayLike): Input features to make predictions on.

        Returns:
            ArrayLike: Model predictions.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        log.info(f"Predicting on data with shape {X.shape = }")
        return self.model.predict(X)

    def train(self, X: ArrayLike, y: ArrayLike) -> None:
        """Train the model on input features X and target y.

        Args:
            X (ArrayLike): Input features for training.
            y (ArrayLike): Target values for training.
        """
        log.info(f"Training model on data with shape {X.shape = } and {y.shape = }")
        self.model.fit(X, y)
        self._trained = True

    def validate(self, X: ArrayLike, y: ArrayLike, m: ArrayLike, e: ArrayLike) -> Metrics:
        """Validate the model on input features X and target y.

        Args:
            X: Input features for validation
            y: Target values for validation
            m: Meta model predictions for validation
            e: Era indices for validation

        Returns:
            Metrics: Dictionary with metric names (prefixed with 'val/') as keys and their values

        Raises:
            RuntimeError: If the model has not been trained yet
        """
        return self.evaluate(X, y, m, e, prefix="val/")

    def test(self, X: ArrayLike, y: ArrayLike, m: ArrayLike, e: ArrayLike) -> Metrics:
        """Test the model on input features X and target y.

        Args:
            X: Input features for testing
            y: Target values for testing
            m: Meta model predictions for testing
            e: Era indices for testing

        Returns:
            Metrics: Dictionary with metric names (prefixed with 'test/') as keys and their values

        Raises:
            RuntimeError: If the model has not been trained yet
        """
        return self.evaluate(X, y, m, e, prefix="test/")

    def evaluate(self, X: ArrayLike, y: ArrayLike, m: ArrayLike, e: ArrayLike, prefix: str) -> Metrics:
        """Evaluate the model on input features X and target y.

        Args:
            X: Input features for evaluation
            y: Target values for evaluation
            m: Meta model predictions for evaluation
            e: Era indices for evaluation
            prefix: Prefix for metric names (e.g., 'val/' or 'test/')

        Returns:
            Metrics: Dictionary with metric names (prefixed) as keys and their values
        """
        p = self(X)
        log.info(f"Evaluating model on data with shape {X.shape = }, {y.shape = }, {m.shape = }, {e.shape = }")
        results = {}
        for metric in self.metrics:
            metric_name = metric.func.__name__ if isinstance(metric, partial) else metric.__name__
            metric_value = metric(y, p, m, e)
            results[f"{prefix}{metric_name}"] = metric_value
        return results

    def save(self, path: Path_) -> None:
        """Save the entire SKLearnModule object to the specified path.

        Args:
            path: Path where to save the model
        """
        path = Path(path)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path_) -> "SKLearnModule":
        """Load a SKLearnModule object from the specified path.

        Args:
            path: Path from where to load the model

        Returns:
            SKLearnModule: The loaded module object
        """
        path = Path(path)
        return joblib.load(path)


Module = LightningModule | SKLearnModule
