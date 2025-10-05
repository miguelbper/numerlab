from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from numerlab.metrics.erawise import corr, mmc
from numerlab.sklearn_.checkpoint import SKLearnCheckpoint
from numerlab.sklearn_.datamodule import SKLearnDataModule
from numerlab.sklearn_.module import SKLearnModule
from numerlab.sklearn_.trainer import SKLearnTrainer
from numerlab.utils.types import Data

N = 5
ERAS = 2
rng = np.random.RandomState(42)

X_TRAIN = rng.randn(N, N)
Y_TRAIN = rng.randn(N)
M_TRAIN = rng.randn(N)
E_TRAIN = rng.randint(0, ERAS, N)

X_VAL = rng.randn(N, N)
Y_VAL = rng.randn(N)
M_VAL = rng.randn(N)
E_VAL = rng.randint(0, ERAS, N)

X_TEST = rng.randn(N, N)
Y_TEST = rng.randn(N)
M_TEST = rng.randn(N)
E_TEST = rng.randint(0, ERAS, N)


class RandomDataModule(SKLearnDataModule):
    def train_dataset(self) -> Data:
        return X_TRAIN, Y_TRAIN, M_TRAIN, E_TRAIN

    def val_dataset(self) -> Data:
        return X_VAL, Y_VAL, M_VAL, E_VAL

    def test_dataset(self) -> Data:
        return X_TEST, Y_TEST, M_TEST, E_TEST


class TrainValSameDataModule(SKLearnDataModule):
    def train_dataset(self) -> Data:
        return X_TRAIN, Y_TRAIN, M_TRAIN, E_TRAIN

    def val_dataset(self) -> Data:
        return X_TRAIN, Y_TRAIN, M_TRAIN, E_TRAIN  # Same as train

    def test_dataset(self) -> Data:
        return X_TEST, Y_TEST, M_TEST, E_TEST


class ValTestSameDataModule(SKLearnDataModule):
    def train_dataset(self) -> Data:
        return X_TRAIN, Y_TRAIN, M_TRAIN, E_TRAIN

    def val_dataset(self) -> Data:
        return X_VAL, Y_VAL, M_VAL, E_VAL

    def test_dataset(self) -> Data:
        return X_VAL, Y_VAL, M_VAL, E_VAL  # Same as validation


@pytest.fixture
def model() -> SKLearnModule:
    return SKLearnModule(model=LinearRegression(fit_intercept=False), metrics=[corr, mmc])


@pytest.fixture
def datamodule() -> SKLearnDataModule:
    return RandomDataModule()


@pytest.fixture
def ckpt_path(tmp_path: Path) -> Path:
    return tmp_path / "model.pkl"


@pytest.fixture
def checkpoint_callback(tmp_path: Path) -> SKLearnCheckpoint:
    return SKLearnCheckpoint(
        dirpath=tmp_path,
        monitor="val/corr",
        mode="max",
    )


@pytest.fixture
def trainer(checkpoint_callback: SKLearnCheckpoint) -> SKLearnTrainer:
    return SKLearnTrainer(callbacks=[checkpoint_callback])


class TestTrainer:
    def test_fit(
        self,
        trainer: SKLearnTrainer,
        datamodule: SKLearnDataModule,
        model: SKLearnModule,
        ckpt_path: Path,
    ) -> None:
        trainer.fit(model, datamodule, ckpt_path)
        assert hasattr(model.model, "coef_")
        assert hasattr(model.model, "intercept_")

    def test_validate(self, trainer: SKLearnTrainer, datamodule: SKLearnDataModule, model: SKLearnModule) -> None:
        trainer.fit(model, datamodule)
        metrics = trainer.validate(model, datamodule)[0]
        assert isinstance(metrics, dict)
        assert len(metrics) == 2
        assert "val/corr" in metrics
        assert "val/mmc" in metrics
        assert isinstance(metrics["val/corr"], float)
        assert isinstance(metrics["val/mmc"], float)

    def test_test(self, trainer: SKLearnTrainer, datamodule: SKLearnDataModule, model: SKLearnModule) -> None:
        trainer.fit(model, datamodule)
        metrics = trainer.test(model, datamodule)[0]
        assert isinstance(metrics, dict)
        assert len(metrics) == 2
        assert "test/corr" in metrics
        assert "test/mmc" in metrics
        assert isinstance(metrics["test/corr"], float)
        assert isinstance(metrics["test/mmc"], float)

    def test_checkpoint(
        self,
        trainer: SKLearnTrainer,
        datamodule: SKLearnDataModule,
        model: SKLearnModule,
        ckpt_path: Path,
    ) -> None:
        trainer.fit(model, datamodule, ckpt_path)

        val_metrics_ckpt = trainer.validate(None, datamodule, ckpt_path)
        val_metrics_model = trainer.validate(model, datamodule, None)
        assert val_metrics_ckpt == val_metrics_model

        test_metrics_ckpt = trainer.test(None, datamodule, ckpt_path)
        test_metrics_model = trainer.test(model, datamodule, None)
        assert test_metrics_ckpt == test_metrics_model

    def test_invalid_inputs(self, trainer: SKLearnTrainer, datamodule: SKLearnDataModule, model: SKLearnModule) -> None:
        with pytest.raises(ValueError, match="Either model or ckpt_path must be provided"):
            trainer.validate(None, datamodule, None)
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            trainer.validate(model, datamodule, None)

    def test_perfect_fit(self, trainer: SKLearnTrainer, model: SKLearnModule) -> None:
        datamodule = TrainValSameDataModule()
        trainer.fit(model, datamodule)
        metrics = trainer.validate(model, datamodule)[0]
        assert isinstance(metrics, dict)
        assert len(metrics) == 2
        assert "val/corr" in metrics
        assert "val/mmc" in metrics
        assert isinstance(metrics["val/corr"], float)
        assert isinstance(metrics["val/mmc"], float)
        assert metrics["val/corr"] > 0.9
