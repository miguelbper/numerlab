import os

import pytest
import torch
from pytest import FixtureRequest
from torch import nn

from numerlab.model.components.numerai_transformer import NumeraiTransformer
from tests.conftest import param_namer


@pytest.fixture(params=["cpu", "cuda"], ids=param_namer("device"))
def device(request: FixtureRequest) -> str:
    device_: str = request.param
    if device_ != "cpu" and os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping GPU tests on GitHub Actions")
    if device_ == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return device_


@pytest.fixture(params=[1, 2], ids=param_namer("batch_size"))
def batch_size(request: FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[1, 4], ids=param_namer("num_tokens"))
def num_tokens(request: FixtureRequest) -> int:
    return request.param


@pytest.fixture
def num_values() -> int:
    return 5


@pytest.fixture(params=[1, 2], ids=param_namer("num_targets"))
def num_targets(request: FixtureRequest) -> int:
    return request.param


@pytest.fixture
def numerai_era(batch_size: int, num_tokens: int, num_values: int, device: str) -> torch.Tensor:
    return torch.randint(0, num_values, (batch_size, num_tokens), dtype=torch.int8, device=device)


@pytest.fixture
def transformer(num_tokens: int, num_values: int, num_targets: int, device: str) -> NumeraiTransformer:
    return NumeraiTransformer(
        num_features=num_tokens,
        num_targets=num_targets,
        num_values=num_values,
    ).to(device)


class TestTransformer:
    def test_init(self, transformer: NumeraiTransformer) -> None:
        assert isinstance(transformer, nn.Module)

    def test_call(
        self,
        transformer: NumeraiTransformer,
        numerai_era: torch.Tensor,
        batch_size: int,
        num_targets: int,
    ) -> None:
        with torch.no_grad():
            output: torch.Tensor = transformer(numerai_era)

        assert isinstance(output, torch.Tensor)
        assert output.dtype == torch.float32
        assert output.shape == (batch_size, num_targets)

    def test_weights(
        self,
        transformer: NumeraiTransformer,
        numerai_era: torch.Tensor,
    ) -> None:
        output: torch.Tensor = transformer(numerai_era)
        target: torch.Tensor = torch.zeros_like(output, dtype=torch.float32)
        loss: torch.Tensor = nn.MSELoss()(output, target)
        loss.backward()

        for param in transformer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert param.grad.shape == param.shape
                assert param.grad.dtype == torch.float32

        assert transformer.net[0].embedding.requires_grad
