import numpy as np
from einops import rearrange
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from numerlab.metrics.preprocessing import center, exponentiate, gaussianize, orthogonal_projection


def corr_vector(target: NDArray[np.number]) -> NDArray[np.float32]:
    """Calculate the correlation contribution vector.

    This vector can be used to calculate the correlation score by taking its
    dot product with the exponentiated and gaussianized predictions.

    Args:
        target: Target values

    Returns:
        Vector of correlation contributions
    """
    n = len(target)
    p = np.arange(n, dtype=np.float32)
    t_ = exponentiate(center(target.astype(np.float32)))
    p_ = exponentiate(gaussianize(p))
    return t_ / (n * np.std(t_) * np.std(p_))


def mmc_vector(target: NDArray[np.number], meta: NDArray[np.number]) -> NDArray[np.float32]:
    """Calculate the MMC contribution vector.

    This vector can be used to calculate the MMC score by taking its
    dot product with the gaussianized predictions.

    Args:
        target: Target values
        meta: Meta model prediction values

    Returns:
        Vector of MMC contributions
    """
    n = len(target)
    t_ = center(target.astype(np.float32))
    m_ = gaussianize(meta.astype(np.float32))
    proj = orthogonal_projection(m_, t_)
    return (4 / n) * proj


def payoff_vector(target: NDArray[np.number], meta: NDArray[np.number]) -> NDArray[np.float32]:
    """Returns the vector that maximizes the payoff.

    For n large, it is possible to show experimentally that there is a high
    probability that this is equal to mmc_vector.

    Args:
        target: Target values
        meta: Meta model prediction values

    Returns:
        Vector that maximizes the payoff
    """
    n = len(target)
    x = rearrange(0.5 * corr_vector(target), "n -> n 1")
    y = rearrange(2 * mmc_vector(target, meta), "n -> n 1")
    v = rearrange(gaussianize(np.arange(n, dtype=np.float32)), "n -> n 1")
    u = exponentiate(v)
    W = x @ u.T + y @ v.T
    _, col_ind = linear_sum_assignment(W, maximize=True)
    return gaussianize(col_ind)
