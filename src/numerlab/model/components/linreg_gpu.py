from itertools import product

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import Tensor


class LinearRegressionGPU(BaseEstimator, RegressorMixin):
    """A linear regression model that uses GPU acceleration.

    This class implements linear regression using GPU acceleration via PyTorch.
    It follows the scikit-learn estimator interface and can be used as a drop-in
    replacement for sklearn.linear_model.LinearRegression.

    The model solves the linear regression problem by computing the exact solution:
    W = (X^T X)^(-1) X^T y

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False,
        no intercept will be used in calculations.
    streaming_batch_size : int or None, default=None
        Size of batches to use for streaming matrix multiplication. If None,
        regular matrix multiplication is used. Use this parameter to reduce
        memory usage when working with large matrices.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if fit_intercept=False.
    n_features_in_ : int
        Number of features seen during fit.
    device : str
        Device used for computations ('cuda' if GPU is available, 'cpu' otherwise).
    """

    def __init__(self, fit_intercept: bool = True, batch_size: int | None = None) -> None:
        self.fit_intercept: bool = fit_intercept
        self.batch_size: int | None = batch_size
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: NDArray[np.number], y: NDArray[np.number]) -> "LinearRegressionGPU":
        """Fit linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : LinearRegressionGPU
            Fitted estimator.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X = check_array(X)  # (m, n)
        y = check_array(y)  # (m, t)

        self.n_features_in_ = X.shape[1]

        # Main Fit code
        X: Tensor = torch.from_numpy(X).to(self.device)  # (m, n)
        y: Tensor = torch.from_numpy(y).to(self.device)  # (m, t)

        if self.fit_intercept:
            m, _ = X.shape
            N: Tensor = torch.ones(size=(m, 1), device=self.device, dtype=X.dtype)  # (m, 1)
            X: Tensor = torch.cat([N, X], dim=1)  # (m, n + 1)  Note: will ignore "+ 1" in dimension comments below

        A: Tensor = streaming_matmul(X.T, X, dtype=torch.float32, batch_size=self.batch_size)  # (n, n)
        b: Tensor = streaming_matmul(X.T, y, dtype=torch.float32, batch_size=self.batch_size)  # (n, t)

        W: Tensor = torch.linalg.solve(A, b)  # (n, t)

        self.W = W.to(torch.float32)

        if self.fit_intercept:
            self.coef_ = W[1:, :].cpu().numpy().squeeze().T  # (n, t)
            self.intercept_ = W[0, :].cpu().numpy().squeeze()  # (t,)
        else:
            self.coef_ = W.cpu().numpy().squeeze().T  # (n, t)
            self.intercept_ = 0.0

        return self

    def predict(self, X: NDArray[np.number]) -> NDArray[np.number]:
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.

        Raises
        ------
        ValueError
            If the number of features in X does not match the number of features
            the model was trained on.
        """
        check_is_fitted(self)
        X = check_array(X)  # (b, n)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but LinearRegressionGPU is expecting "
                f"{self.n_features_in_} features as input."
            )

        # Main Predict code
        X: Tensor = torch.from_numpy(X).to(self.device)  # (b, n)

        if self.fit_intercept:
            m, _ = X.shape
            N: Tensor = torch.ones(size=(m, 1), device=self.device, dtype=X.dtype)  # (m, 1)
            X: Tensor = torch.cat([N, X], dim=1)  # (m, n + 1)  Note: will ignore "+ 1" in dimension comments below

        X: Tensor = X.to(torch.float32)  # (m, n)
        y: Tensor = X @ self.W  # (b, t)

        return y.cpu().numpy().squeeze()


def streaming_matmul(A: Tensor, B: Tensor, dtype: torch.dtype, batch_size: int | None = None) -> Tensor:
    """Performs matrix multiplication in a streaming fashion to reduce memory
    usage.

    This function multiplies two matrices A and B using batched operations to avoid loading
    the entire matrices into memory at once. This is useful for very large matrices that
    may not fit in GPU memory.

    Args:
        A (Tensor): First input matrix of shape (m, n)
        B (Tensor): Second input matrix of shape (n, p)
        dtype (torch.dtype): Data type to use for computation (e.g. torch.float32)
        batch_size (int | None): Size of batches to process at once. If None or >= n,
            performs regular matrix multiplication. Default: None

    Returns:
        Tensor: Result of matrix multiplication A @ B with shape (m, p)

    Raises:
        ValueError: If A and B are not 2D tensors, have incompatible shapes,
            or are not on the same device

    Note:
        The function processes the matrices in blocks of size batch_size x batch_size
        to reduce memory usage. Each block computation is:
        C[i:i+batch_size, j:j+batch_size] = A[i:i+batch_size, :] @ B[:, j:j+batch_size]
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors. {A.ndim = }, {B.ndim = }")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"A and B must have compatible shapes. {A.shape = }, {B.shape = }")
    if A.device != B.device:
        raise ValueError(f"A and B must be on the same device. {A.device = }, {B.device = }")

    m, _ = A.shape
    _, p = B.shape

    if batch_size is None:
        A = A.to(dtype)
        B = B.to(dtype)
        return A @ B

    C: Tensor = torch.empty(size=(m, p), dtype=dtype, device=A.device)

    for i, j in product(range(0, m, batch_size), range(0, p, batch_size)):
        i0: int = i
        i1: int = min(i + batch_size, m)
        j0: int = j
        j1: int = min(j + batch_size, p)

        A_block: Tensor = A[i0:i1, :].to(dtype)  # (batch_size, n)
        B_block: Tensor = B[:, j0:j1].to(dtype)  # (n, batch_size)

        C[i0:i1, j0:j1] = A_block @ B_block  # (batch_size, batch_size)

    return C
