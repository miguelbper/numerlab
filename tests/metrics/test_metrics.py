import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from einops import rearrange
from numerai_tools.scoring import correlation_contribution, numerai_corr
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

from numerlab.metrics.erawise import annual_return, corr, mmc, payoff
from numerlab.metrics.metrics import corr_, mmc_, payoff_
from numerlab.metrics.preprocessing import exponentiate, gaussianize
from numerlab.metrics.torchmetrics_ import MMC, AnnualReturn, Corr, Payoff
from numerlab.metrics.vectors import corr_vector, mmc_vector
from tests.conftest import param_namer

TOL = 1e-6


@pytest.fixture
def n() -> int:
    return 100


@pytest.fixture(params=range(0, 100), ids=param_namer("seed"))
def seed(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


@pytest.fixture
def target(n: int, rng: np.random.RandomState) -> NDArray[np.float32]:
    return rng.randint(0, 5, size=n).astype(np.float32) / 4


@pytest.fixture
def preds(n: int, rng: np.random.RandomState) -> NDArray[np.float32]:
    return rng.randn(n).astype(np.float32)


@pytest.fixture
def meta(n: int, rng: np.random.RandomState) -> NDArray[np.float32]:
    return rng.randn(n).astype(np.float32)


@pytest.fixture
def eras(n: int, rng: np.random.RandomState) -> NDArray[np.int32]:
    return rng.randint(0, 10, size=n).astype(np.int32)


class TestCorr:
    def test_equal_to_numerai(self, target: NDArray[np.float32], preds: NDArray[np.float32]) -> None:
        ts: pd.Series = pd.Series(target, name="t")
        ps: pd.DataFrame = pd.DataFrame({"p": preds})
        corr_0: float = corr_(target, preds)
        corr_1: float = numerai_corr(ps, ts).item()
        assert np.allclose(corr_0, corr_1)
        assert isinstance(corr_0, float)

    def test_equal_to_numerai_with_vector(self, target: NDArray[np.float32], preds: NDArray[np.float32]) -> None:
        ts: pd.Series = pd.Series(target, name="t")
        ps: pd.DataFrame = pd.DataFrame({"p": preds})
        vect: NDArray[np.float32] = corr_vector(target)
        corr_0: float = float(np.dot(vect, exponentiate(gaussianize(preds))))
        corr_1: float = numerai_corr(ps, ts).item()
        assert np.allclose(corr_0, corr_1)
        assert isinstance(corr_0, float)
        assert vect.dtype == np.float32

    def test_equal_to_erawise_no_era(self, target: NDArray[np.float32], preds: NDArray[np.float32]) -> None:
        corr_0 = corr_(target, preds)
        corr_1 = corr(target, preds)
        assert corr_0 == corr_1

    def test_equal_to_erawise_one_era(self, target: NDArray[np.float32], preds: NDArray[np.float32]) -> None:
        eras_ = np.ones_like(target, dtype=np.int32)
        corr_0 = corr_(target, preds)
        corr_1 = corr(target, preds, eras=eras_)
        assert np.allclose(corr_0, corr_1, rtol=TOL, atol=TOL)

    def test_torchmetric_one_batch(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        corr_np: float = corr(target, preds, meta, eras)

        t_: torch.Tensor = torch.tensor(target)
        p_: torch.Tensor = torch.tensor(preds)
        m_: torch.Tensor = torch.tensor(meta)
        e_: torch.Tensor = torch.tensor(eras)

        corr_metric = Corr()
        corr_torch = corr_metric(t_, p_, m_, e_)

        assert np.allclose(corr_torch.item(), corr_np, rtol=TOL, atol=TOL)

    def test_torchmetric_dataloader(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        corr_np: float = corr(target, preds, meta, eras)

        t_: torch.Tensor = torch.tensor(target)
        p_: torch.Tensor = torch.tensor(preds)
        m_: torch.Tensor = torch.tensor(meta)
        e_: torch.Tensor = torch.tensor(eras)

        corr_metric = Corr()

        dataset = TensorDataset(t_, p_, m_, e_)
        batch_size = len(dataset) // 10
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            t, p, m, e = batch
            corr_metric.update(t, p, m, e)

        corr_torch = corr_metric.compute()

        assert np.allclose(corr_torch.item(), corr_np, rtol=TOL, atol=TOL)


class TestMMC:
    def test_equal_to_numerai(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
    ) -> None:
        ts: pd.Series = pd.Series(target, name="t")
        ps: pd.DataFrame = pd.DataFrame({"p": preds})
        ms: pd.Series = pd.Series(meta, name="m")

        mmc_0: float = mmc_(target, preds, meta)
        mmc_1: float = correlation_contribution(ps, ms, ts).item()
        assert np.allclose(mmc_0, mmc_1)
        assert isinstance(mmc_0, float)

    def test_equal_to_numerai_with_vector(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
    ) -> None:
        ts: pd.Series = pd.Series(target, name="t")
        ps: pd.DataFrame = pd.DataFrame({"p": preds})
        ms: pd.Series = pd.Series(meta, name="m")

        vect: NDArray[np.float32] = mmc_vector(target, meta)
        mmc_0: float = float(np.dot(vect, gaussianize(preds)))
        mmc_1: float = correlation_contribution(ps, ms, ts).item()

        assert np.allclose(mmc_0, mmc_1)
        assert isinstance(mmc_0, float)
        assert vect.dtype == np.float32

    def test_equal_to_erawise_no_era(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
    ) -> None:
        mmc_0 = mmc_(target, preds, meta)
        mmc_1 = mmc(target, preds, meta)
        assert mmc_0 == mmc_1

    def test_equal_to_erawise_one_era(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
    ) -> None:
        eras_ = np.ones_like(target, dtype=np.int32)
        mmc_0 = mmc_(target, preds, meta)
        mmc_1 = mmc(target, preds, meta, eras=eras_)
        assert np.allclose(mmc_0, mmc_1, rtol=TOL, atol=TOL)

    def test_torchmetric_one_batch(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        mmc_np: float = mmc(target, preds, meta, eras)

        t_: torch.Tensor = torch.tensor(target)
        p_: torch.Tensor = torch.tensor(preds)
        m_: torch.Tensor = torch.tensor(meta)
        e_: torch.Tensor = torch.tensor(eras)

        mmc_metric = MMC()
        mmc_torch = mmc_metric(t_, p_, m_, e_)

        assert np.allclose(mmc_torch.item(), mmc_np, rtol=TOL, atol=TOL)

    def test_no_meta(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        meta = np.nan * np.ones_like(target)
        assert np.isnan(mmc(target, preds, meta, eras=None))
        assert np.isnan(mmc(target, preds, meta, eras=eras))

    def test_torchmetric_dataloader(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        mmc_np: float = mmc(target, preds, meta, eras)

        t_: torch.Tensor = torch.tensor(target)
        p_: torch.Tensor = torch.tensor(preds)
        m_: torch.Tensor = torch.tensor(meta)
        e_: torch.Tensor = torch.tensor(eras)

        mmc_metric = MMC()

        dataset = TensorDataset(t_, p_, m_, e_)
        batch_size = len(dataset) // 10
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            t, p, m, e = batch
            mmc_metric.update(t, p, m, e)

        mmc_torch = mmc_metric.compute()

        assert np.allclose(mmc_torch.item(), mmc_np, rtol=TOL, atol=TOL)


class TestPayoff:
    def test_equal_to_erawise_no_era(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
    ) -> None:
        payoff_0 = payoff_(target, preds, meta)
        payoff_1 = payoff(target, preds, meta)
        assert payoff_0 == payoff_1

    def test_equal_to_erawise_one_era(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
    ) -> None:
        eras_ = np.ones_like(target, dtype=np.int32)
        payoff_0 = payoff_(target, preds, meta)
        payoff_1 = payoff(target, preds, meta, eras=eras_)
        assert np.allclose(payoff_0, payoff_1, rtol=TOL, atol=TOL)

    def test_torchmetric_one_batch(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        payoff_np: float = payoff(target, preds, meta, eras)

        t_: torch.Tensor = torch.tensor(target)
        p_: torch.Tensor = torch.tensor(preds)
        m_: torch.Tensor = torch.tensor(meta)
        e_: torch.Tensor = torch.tensor(eras)

        payoff_metric = Payoff()
        payoff_torch = payoff_metric(t_, p_, m_, e_)

        assert np.allclose(payoff_torch.item(), payoff_np, rtol=TOL, atol=TOL)

    def test_no_meta(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        meta = np.nan * np.ones_like(target)
        assert np.isnan(payoff(target, preds, meta, eras=None))
        assert np.isnan(payoff(target, preds, meta, eras=eras))

    def test_torchmetric_dataloader(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        payoff_np: float = payoff(target, preds, meta, eras)

        t_: torch.Tensor = torch.tensor(target)
        p_: torch.Tensor = torch.tensor(preds)
        m_: torch.Tensor = torch.tensor(meta)
        e_: torch.Tensor = torch.tensor(eras)

        payoff_metric = Payoff()

        dataset = TensorDataset(t_, p_, m_, e_)
        batch_size = len(dataset) // 10
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            t, p, m, e = batch
            payoff_metric.update(t, p, m, e)

        payoff_torch = payoff_metric.compute()

        assert np.allclose(payoff_torch.item(), payoff_np, rtol=TOL, atol=TOL)


class TestAnnualReturn:
    def test_one_year_of_equal_rounds(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
    ) -> None:
        num_annual_rounds = 52 * 5
        annual_target = np.tile(target, num_annual_rounds)
        annual_preds = np.tile(preds, num_annual_rounds)
        annual_meta = np.tile(meta, num_annual_rounds)
        annual_eras = np.repeat(np.arange(num_annual_rounds), len(target)).astype(np.int32)
        annual_ret_0 = annual_return(annual_target, annual_preds, annual_meta, annual_eras)

        PAYOUT_FACTOR = 0.1
        round_ret = PAYOUT_FACTOR * payoff(target, preds, meta)
        annual_ret_1 = (1 + round_ret) ** num_annual_rounds - 1

        assert np.allclose(annual_ret_0, annual_ret_1, rtol=TOL, atol=TOL)

    def test_torchmetric_one_batch(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        annual_ret_np: float = annual_return(target, preds, meta, eras)

        t_: torch.Tensor = torch.tensor(target)
        p_: torch.Tensor = torch.tensor(preds)
        m_: torch.Tensor = torch.tensor(meta)
        e_: torch.Tensor = torch.tensor(eras)

        annual_ret_metric = AnnualReturn()
        annual_ret_torch = annual_ret_metric(t_, p_, m_, e_)

        assert np.allclose(annual_ret_torch.item(), annual_ret_np, rtol=TOL, atol=TOL)

    def test_no_meta(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        meta = np.nan * np.ones_like(target)
        assert np.isnan(annual_return(target, preds, meta, eras=eras))

    def test_torchmetric_dataloader(
        self,
        target: NDArray[np.float32],
        preds: NDArray[np.float32],
        meta: NDArray[np.float32],
        eras: NDArray[np.int32],
    ) -> None:
        annual_ret_np: float = annual_return(target, preds, meta, eras)

        t_: torch.Tensor = torch.tensor(target)
        p_: torch.Tensor = torch.tensor(preds)
        m_: torch.Tensor = torch.tensor(meta)
        e_: torch.Tensor = torch.tensor(eras)

        annual_ret_metric = AnnualReturn()

        dataset = TensorDataset(t_, p_, m_, e_)
        batch_size = len(dataset) // 10
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            t, p, m, e = batch
            annual_ret_metric.update(t, p, m, e)

        annual_ret_torch = annual_ret_metric.compute()

        assert np.allclose(annual_ret_torch.item(), annual_ret_np, rtol=TOL, atol=TOL)


class TestOptimization:
    def test_optimal_payoff(self, target: NDArray[np.float32], meta: NDArray[np.float32]) -> None:
        n: int = len(target)
        row_idx: NDArray[np.int64] = np.arange(n)
        meta = (pl.Series(meta.astype(np.float32)).rank().to_numpy() - 0.5) / n

        def _payoff(W: NDArray[np.float64], col_idx: NDArray[np.int32]) -> np.float64:
            return np.sum(W[row_idx, col_idx])

        def rank(v: NDArray[np.float32]) -> NDArray[np.int32]:
            s = pl.Series(v.flatten())
            return s.rank(method="ordinal").to_numpy().astype(np.int32) - 1

        c: NDArray[np.float64] = rearrange(corr_vector(target), "n -> n 1").astype(np.float64)
        m: NDArray[np.float64] = rearrange(mmc_vector(target, meta), "n -> n 1").astype(np.float64)
        g: NDArray[np.float64] = rearrange(gaussianize(row_idx), "n -> n 1")
        g_exp: NDArray[np.float64] = exponentiate(g)

        W_c: NDArray[np.float64] = c @ g_exp.T
        W_m: NDArray[np.float64] = m @ g.T
        W_p: NDArray[np.float64] = W_c + W_m

        v_c: NDArray[np.int64] = linear_sum_assignment(W_c, maximize=True)[1]
        v_m: NDArray[np.int64] = linear_sum_assignment(W_m, maximize=True)[1]
        v_p: NDArray[np.int64] = linear_sum_assignment(W_p, maximize=True)[1]

        pi_c: NDArray[np.int32] = rank(c)
        pi_m: NDArray[np.int32] = rank(m)
        pi_p: NDArray[np.int32] = pi_m

        assert np.allclose(_payoff(W_c, v_c), _payoff(W_c, pi_c))
        assert np.allclose(_payoff(W_m, v_m), _payoff(W_m, pi_m))
        assert _payoff(W_p, v_p) >= _payoff(W_p, pi_p)
