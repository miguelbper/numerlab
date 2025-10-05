import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

from numerlab.model.components.linreg_gpu import LinearRegressionGPU
from tests.conftest import param_namer


@pytest.fixture
def rng() -> np.random.RandomState:
    """Fixture providing a random number generator for consistent testing."""
    return check_random_state(42)


@pytest.fixture(params=[1, 2], ids=param_namer("n_targets"))
def n_targets(request) -> int:
    """Fixture providing different numbers of targets for testing."""
    return request.param


@pytest.fixture
def synthetic_data(rng: np.random.RandomState, n_targets: int) -> tuple[NDArray, NDArray]:
    """Fixture providing synthetic data for testing."""
    n_samples, n_features = 100, 5
    X: NDArray = rng.randn(n_samples, n_features)
    y: NDArray = rng.randn(n_samples) if n_targets == 1 else rng.randn(n_samples, n_targets)
    return X.astype(np.int8), y.astype(np.float32)


@pytest.fixture(params=[None, 1, 2, 3, 4, 5], ids=param_namer("batch_size"))
def batch_size(request) -> int | None:
    """Fixture providing different values for batch_size."""
    return request.param


@pytest.fixture(params=[False, True], ids=param_namer("fit_intercept"))
def fit_intercept(request) -> bool:
    """Fixture providing different values for fit_intercept."""
    return request.param


@pytest.fixture
def cpu_model(fit_intercept: bool) -> LinearRegression:
    """Fixture providing a LinearRegressionGPU model on CPU."""
    return LinearRegression(fit_intercept=fit_intercept)


@pytest.fixture
def gpu_model(fit_intercept: bool, batch_size: int | None) -> LinearRegressionGPU:
    """Fixture providing a LinearRegressionGPU model on GPU."""
    return LinearRegressionGPU(fit_intercept=fit_intercept, batch_size=batch_size)


class TestLinearRegressionGPU:
    """Test suite for LinearRegressionGPU."""

    def test_fit(
        self,
        synthetic_data: tuple[NDArray, NDArray],
        cpu_model: LinearRegression,
        gpu_model: LinearRegressionGPU,
    ) -> None:
        """Test that LinearRegressionGPU produces same coefficients as sklearn
        LinearRegression."""
        X, y = synthetic_data

        cpu_model.fit(X, y)
        cpu_coef: NDArray = cpu_model.coef_
        cpu_intercept: float | NDArray = cpu_model.intercept_

        gpu_model.fit(X, y)
        gpu_coef: NDArray = gpu_model.coef_
        gpu_intercept: float | NDArray = gpu_model.intercept_

        np.testing.assert_array_almost_equal(cpu_coef, gpu_coef)
        np.testing.assert_array_almost_equal(cpu_intercept, gpu_intercept)

    def test_predict(
        self,
        synthetic_data: tuple[NDArray, NDArray],
        cpu_model: LinearRegression,
        gpu_model: LinearRegressionGPU,
    ) -> None:
        """Test that LinearRegressionGPU produces same predictions as sklearn
        LinearRegression."""
        X, y = synthetic_data

        # Fit both models
        cpu_model.fit(X, y)
        gpu_model.fit(X, y)

        X_test: NDArray = np.random.randn(50, X.shape[1])
        cpu_pred: NDArray = cpu_model.predict(X_test)
        gpu_pred: NDArray = gpu_model.predict(X_test)

        np.testing.assert_array_almost_equal(cpu_pred, gpu_pred, decimal=6)
