from collections.abc import Callable

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from scipy.stats import gmean

from numerlab.metrics.metrics import corr_, mmc_, payoff_
from numerlab.metrics.preprocessing import gaussianize

Target = NDArray[np.number]
Preds = NDArray[np.number]
Meta = NDArray[np.number]
Eras = NDArray[np.integer]
ScoringFnOneEra = Callable[[Target, Preds, Meta], float]
ScoringFn = Callable[[Target, Preds, Meta, Eras | None], float]


def ensemble(preds: NDArray[np.number]) -> NDArray[np.number]:
    """Calculate the ensemble of predictions.

    Args:
        preds: Prediction values
    """
    preds: NDArray[np.number] = np.atleast_2d(preds)
    normalized: NDArray[np.number] = np.apply_along_axis(gaussianize, axis=0, arr=preds)
    standardized: NDArray[np.number] = normalized / np.std(normalized, axis=0, keepdims=True)
    average: NDArray[np.number] = np.mean(standardized, axis=1)
    return gaussianize(average)


def erawise(scoring_fn_one_era: ScoringFnOneEra, agg_fn: Callable[[NDArray[np.number]], float] = np.mean) -> ScoringFn:
    def scoring_fn(target: Target, preds: Preds, meta: Meta, eras: Eras | None = None) -> float:
        if eras is None:
            return scoring_fn_one_era(target, preds, meta)

        target: NDArray[np.number] = np.squeeze(target)  # (m,)
        preds: NDArray[np.number] = preds if preds.ndim == 2 else preds.reshape(-1, 1)  # (m, num_preds)
        meta: NDArray[np.integer] = np.squeeze(meta)  # (m,)
        eras: NDArray[np.integer] = np.squeeze(eras)  # (m,)
        num_preds: int = preds.shape[1]

        df = pl.DataFrame(
            {
                "target": target,
                **{f"preds_{i}": preds[:, i] for i in range(num_preds)},
                "meta": meta,
                "eras": eras,
            }
        )

        def per_era(group: pl.DataFrame) -> pl.DataFrame:
            target: NDArray[np.number] = group["target"].to_numpy()  # (m,)
            preds: NDArray[np.number] = ensemble(group.select(cs.starts_with("preds_")).to_numpy())  # (m,)
            meta: NDArray[np.number] = group["meta"].to_numpy()  # (m,)
            return pl.DataFrame({"score": scoring_fn_one_era(target, preds, meta)})

        scores: pl.Series = df.group_by("eras").map_groups(per_era).select(pl.col("score"))
        return agg_fn(scores.to_numpy())

    return scoring_fn


def annualized_return(payoffs: NDArray[np.float32]) -> float:
    """Calculate the annualized return from a sequence of payoffs.

    The calculation assumes:
    - 52 weeks per year
    - 5 trading days per week
    - A payout factor of 0.1

    Args:
        payoffs (NDArray[np.float32]): Array of payoff values.

    Returns:
        float: The annualized return rate.
    """
    NUM_ROUNDS_PER_YEAR: int = 52 * 5
    PAYOUT_FACTOR: float = 0.1
    growth_factors: NDArray[np.float32] = 1 + PAYOUT_FACTOR * payoffs
    annual_growth_factor: float = float(gmean(growth_factors) ** NUM_ROUNDS_PER_YEAR)
    annual_return: float = annual_growth_factor - 1
    return annual_return


def corr(
    target: NDArray[np.number],
    preds: NDArray[np.number],
    meta: NDArray[np.number] | None = None,
    eras: NDArray[np.integer] | None = None,
) -> float:
    meta = meta if meta is not None else np.full_like(target, np.nan, dtype=np.float32)
    fn: ScoringFn = erawise(corr_)
    return fn(target, preds, meta, eras)


def mmc(
    target: NDArray[np.number],
    preds: NDArray[np.number],
    meta: NDArray[np.number],
    eras: NDArray[np.integer] | None = None,
) -> float:
    fn: ScoringFn = erawise(mmc_)
    return fn(target, preds, meta, eras)


def payoff(
    target: NDArray[np.number],
    preds: NDArray[np.number],
    meta: NDArray[np.number],
    eras: NDArray[np.integer] | None = None,
) -> float:
    fn: ScoringFn = erawise(payoff_)
    return fn(target, preds, meta, eras)


def annual_return(
    target: NDArray[np.number],
    preds: NDArray[np.number],
    meta: NDArray[np.number],
    eras: NDArray[np.integer] | None = None,
) -> float:
    fn: ScoringFn = erawise(payoff_, annualized_return)
    return fn(target, preds, meta, eras)
