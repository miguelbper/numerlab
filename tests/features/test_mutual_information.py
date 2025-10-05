from itertools import product

import infomeasure as im
import lightning as L
import pytest
import torch

from numerlab.features.mutual_information import (
    N,
    count_X,
    count_XX,
    count_XX_from_XXY,
    count_XXY,
    count_XXY_streaming,
    count_XY,
    count_XY_from_XXY,
    get_MI_XX,
    get_MI_XXY,
    get_MI_XY,
)
from tests.conftest import param_namer

I = 10  # noqa: E741
J = 4


@pytest.fixture(params=range(10), ids=param_namer("seed"))
def seed(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=["cpu", "cuda"], ids=param_namer("device"))
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device(request.param)


@pytest.fixture
def X(seed: int, device: torch.device) -> torch.Tensor:
    L.seed_everything(seed)
    return torch.randint(0, N, (I, J), dtype=torch.int8, device=device)


@pytest.fixture
def Y(seed: int, device: torch.device) -> torch.Tensor:
    L.seed_everything(seed + 1000)
    return torch.randint(0, N, (I,), dtype=torch.int8, device=device)


class TestCounts:
    def test_count_X_fuzz(self, X: torch.Tensor) -> None:
        C = count_X(X)
        assert C.shape == (J, N)

    def test_count_XY_fuzz(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        C = count_XY(X, Y)
        assert C.shape == (J, N, N)

    def test_count_XX_fuzz(self, X: torch.Tensor) -> None:
        C = count_XX(X)
        assert C.shape == (J, J, N, N)

    def test_count_XXY_fuzz(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        C = count_XXY(X, Y)
        assert C.shape == (J, J, N, N, N)

    def test_reduce_XY_to_X(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        C_XY = count_XY(X, Y)
        C_X0 = count_X(X)
        C_X1 = torch.sum(C_XY, dim=2)
        assert torch.equal(C_X0, C_X1)

    def test_reduce_XX_to_X(self, X: torch.Tensor) -> None:
        C_XX = count_XX(X)
        C_X0 = count_X(X)
        C_X1 = torch.sum(C_XX[0], dim=1)
        assert torch.equal(C_X0, C_X1)

    def test_reduce_XXY_to_XX(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        C_XXY = count_XXY(X, Y)
        C_XX0 = count_XX(X)
        C_XX1 = count_XX_from_XXY(C_XXY)
        assert torch.equal(C_XX0, C_XX1)

    def test_reduce_XXY_to_XY(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        C_XXY = count_XXY(X, Y)
        C_XY0 = count_XY(X, Y)
        C_XY1 = count_XY_from_XXY(C_XXY)
        assert torch.equal(C_XY0, C_XY1)

    def test_count_XXY_streaming(self, X: torch.Tensor, Y: torch.Tensor, device: torch.device) -> None:
        C_XXY0 = count_XXY_streaming(X, Y, device=device, batch_size=1)
        C_XXY1 = count_XXY(X, Y)
        assert torch.equal(C_XXY0, C_XXY1)


class TestMutualInformation:
    def test_get_MI_XY(self, X: torch.Tensor, Y: torch.Tensor, device: torch.device) -> None:
        MI_XY_0 = get_MI_XY(X, Y)

        MI_XY_1 = torch.zeros_like(MI_XY_0)
        for j in range(J):
            x = X[:, j].cpu().numpy().flatten()
            y = Y.cpu().numpy()
            MI_XY_1[j] = im.mutual_information(x, y, approach="discrete")

        assert torch.allclose(MI_XY_0, MI_XY_1)

    def test_get_MI_XX(self, X: torch.Tensor, device: torch.device) -> None:
        MI_XX_0 = get_MI_XX(X)

        MI_XX_1 = torch.zeros_like(MI_XX_0)

        for j, k in product(range(J), repeat=2):
            xj = X[:, j].cpu().numpy().flatten()
            xk = X[:, k].cpu().numpy().flatten()
            MI_XX_1[j, k] = im.mutual_information(xj, xk, approach="discrete")

        assert torch.allclose(MI_XX_0, MI_XX_1)

    def test_get_MI_XXY(self, X: torch.Tensor, Y: torch.Tensor, device: torch.device) -> None:
        MI_XXY_0 = get_MI_XXY(X, Y)

        MI_XXY_1 = torch.zeros_like(MI_XXY_0)

        for j, k in product(range(J), repeat=2):
            xj = X[:, j].cpu().numpy().flatten()
            xk = X[:, k].cpu().numpy().flatten()
            y = Y.cpu().numpy()
            MI_XXY_1[j, k] = im.mutual_information(xj, xk, cond=y, approach="discrete")

        assert torch.allclose(MI_XXY_0, MI_XXY_1)
