import numpy as np
import polars as pl
import scipy
from numpy.typing import NDArray


def center(x: NDArray[np.number]) -> NDArray[np.float32]:
    """Center a vector by subtracting its mean.

    Args:
        x (NDArray[np.number]): Input vector of any numeric type.

    Returns:
        NDArray[np.float32]: Centered vector as float32.
    """
    return x - np.mean(x, dtype=np.float32)


def gaussianize(x: NDArray[np.number]) -> NDArray[np.float32]:
    """Transform a vector to follow a standard normal distribution using rank-
    based inverse normal transformation.

    Args:
        x (NDArray[np.number]): Input vector of any numeric type.

    Returns:
        NDArray[np.float32]: Vector transformed to follow standard normal distribution.
    """
    n = len(x)
    x_series = pl.Series("x", x.astype(np.float32))
    x_ranked = (x_series.rank().to_numpy() - 0.5) / n
    return scipy.stats.norm.ppf(x_ranked)


def exponentiate(x: NDArray[np.number]) -> NDArray[np.float32]:
    """Apply a signed power transformation to a vector.

    The transformation is sign(x) * |x|^1.5, which preserves the sign while
    applying a power transformation to the absolute values.

    Args:
        x (NDArray[np.number]): Input vector of any numeric type.

    Returns:
        NDArray[np.float32]: Transformed vector as float32.
    """
    x = x.astype(np.float32)
    return np.sign(x) * np.abs(x) ** 1.5


def orthogonal_projection(meta: NDArray[np.number], vect: NDArray[np.number]) -> NDArray[np.float32]:
    """Calculate the projection of a vector onto the subspace orthogonal to the
    meta model.

    Args:
        meta (NDArray[np.number]): Meta model prediction values.
        vect (NDArray[np.number]): Vector to project.

    Returns:
        NDArray[np.float32]: Projection of the vector onto the subspace orthogonal to the meta model.
    """
    meta = meta.astype(np.float32)
    vect = vect.astype(np.float32)
    coef = np.dot(meta, vect) / np.linalg.norm(meta) ** 2
    return vect - coef * meta
