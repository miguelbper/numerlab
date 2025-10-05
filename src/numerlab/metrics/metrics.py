import numpy as np
from numpy.typing import NDArray

from numerlab.metrics.preprocessing import center, exponentiate, gaussianize, orthogonal_projection


def corr_(target: NDArray[np.number], preds: NDArray[np.number], meta: NDArray[np.number] | None = None) -> float:
    """Calculate the Numerai correlation between target and predictions.

    This implements the Numerai correlation metric which is the correlation
    of the transformed target and predictions.

    Args:
        target (NDArray[np.number]): Target values.
        preds (NDArray[np.number]): Prediction values.
        meta (NDArray[np.number]): Meta model prediction values.

    Returns:
        float: Correlation score between -1 and 1.
    """
    t_ = exponentiate(center(target.astype(np.float32)))
    p_ = exponentiate(gaussianize(preds.astype(np.float32)))
    corr = np.corrcoef(t_, p_)[0, 1]
    return float(corr)


def mmc_(target: NDArray[np.number], preds: NDArray[np.number], meta: NDArray[np.number]) -> float:
    """Calculate the Meta Model Contribution (MMC) metric.

    MMC measures the correlation between target and predictions after removing
    the component that can be explained by the meta model.

    Args:
        target (NDArray[np.number]): Target values.
        preds (NDArray[np.number]): Prediction values.
        meta (NDArray[np.number]): Meta model prediction values.

    Returns:
        float: MMC score between -1 and 1.
    """
    t_ = center(target.astype(np.float32))
    p_ = gaussianize(preds.astype(np.float32))
    m_ = gaussianize(meta.astype(np.float32))
    proj = orthogonal_projection(m_, p_)
    mmc = 4 * np.cov(t_, proj, bias=True)[0, 1]
    return float(mmc)


def payoff_(target: NDArray[np.number], preds: NDArray[np.number], meta: NDArray[np.number]) -> float:
    """Calculate the Numerai payoff score.

    The payoff is a weighted combination of correlation and MMC:
    payoff = 0.5 * corr + 2 * mmc

    Args:
        target (NDArray[np.number]): Target values.
        preds (NDArray[np.number]): Prediction values.
        meta (NDArray[np.number]): Meta model prediction values.

    Returns:
        float: Payoff score.
    """
    return 0.5 * corr_(target, preds) + 2 * mmc_(target, preds, meta)
