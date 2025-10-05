import pytest
import torch
from hydra_zen import instantiate
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

from numerlab.configs.utils.utils import fbuilds
from numerlab.metrics.torchmetrics_ import MMC, Corr, Payoff
from numerlab.model.module import Module
from numerlab.utils.types import Batch
from tests.conftest import param_namer


@pytest.fixture(params=[1, 2], ids=param_namer("batch_size"))
def batch_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[1, 2], ids=param_namer("num_features"))
def num_features(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture()
def batch(batch_size: int, num_features: int) -> Batch:
    X: Tensor = torch.randint(0, 5, (batch_size, num_features)).type(torch.float32)
    T: Tensor = torch.randint(0, 5, (batch_size, 1)).type(torch.float32) / 4
    M: Tensor = torch.rand(batch_size, 1).type(torch.float32)
    E: Tensor = torch.randint(0, 11, (batch_size, 1)).type(torch.int32)
    return (X, T, M, E)


@pytest.fixture
def model(num_features: int) -> Module:
    ModuleCfg = fbuilds(
        Module,
        model=fbuilds(
            nn.Linear,
            in_features=num_features,
            out_features=1,
        ),
        loss_fn=fbuilds(
            nn.MSELoss,
        ),
        optimizer=fbuilds(
            Adam,
            lr=0.001,
            zen_partial=True,
        ),
        scheduler=fbuilds(
            ReduceLROnPlateau,
            mode="min",
            factor=0.1,
            patience=10,
            zen_partial=True,
        ),
        metric_collection=fbuilds(
            MetricCollection,
            metrics=[
                fbuilds(Corr),
                fbuilds(MMC),
                fbuilds(Payoff),
            ],
        ),
    )
    return instantiate(ModuleCfg)


class TestModel:
    def test_init(self, model: Module):
        assert isinstance(model, LightningModule)

    def test_configure_optimizers(self, model: Module):
        optim_cfg = model.configure_optimizers()
        assert isinstance(optim_cfg, dict)
        assert "optimizer" in optim_cfg

    def test_configure_optimizers_no_scheduler(self, model: Module):
        model.scheduler = None
        optim_cfg = model.configure_optimizers()
        assert isinstance(optim_cfg, dict)
        assert "optimizer" in optim_cfg

    def test_training_step(self, model: Module, batch: Batch):
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    def test_validation_step(self, model: Module, batch: Batch):
        model.eval()
        with torch.no_grad():
            model.validation_step(batch, 0)

    def test_test_step(self, model: Module, batch: Batch):
        model.eval()
        with torch.no_grad():
            model.test_step(batch, 0)
