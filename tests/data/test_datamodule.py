import os

import polars as pl
import pytest
import torch
from rootutils import find_root
from torch.utils.data import DataLoader

from numerlab.data.datamodule import NumeraiDataModule, NumeraiDataset
from numerlab.utils.types import Data
from tests.conftest import param_namer

data_dir = find_root() / "data"

ERA_0 = 1
ERA_1 = 2
FEATURE_SET = "small"
NUM_FEATURES = 42  # Number of features in the small feature set, hardcoded for simplicity
TARGETS = ["target"]
NUM_TARGETS = 1


@pytest.fixture
def numerai_dataset() -> NumeraiDataset:
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping NumeraiDataset tests on GitHub Actions")
    return NumeraiDataset(
        data_dir=data_dir,
        era_0=ERA_0,
        era_1=ERA_1,
        features=FEATURE_SET,
        targets=TARGETS,
    )


@pytest.fixture
def numerai_datamodule() -> NumeraiDataModule:
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping NumeraiDataModule tests on GitHub Actions")
    return NumeraiDataModule(
        data_dir=data_dir,
        train_era_0=ERA_0,
        train_era_1=ERA_1,
        validation_era_0=ERA_0,
        validation_era_1=ERA_1,
        test_era_0=ERA_0,
        test_era_1=ERA_1,
        features=FEATURE_SET,
        targets=TARGETS,
    )


@pytest.fixture(
    params=[
        ("fit", "train_dataloader"),
        ("fit", "val_dataloader"),
        ("test", "test_dataloader"),
    ],
    ids=lambda x: f"<split={x[1]}>",
)
def split_dataloader(request: pytest.FixtureRequest, numerai_datamodule: NumeraiDataModule) -> DataLoader:
    """Fixture that provides configured dataloaders for different stages."""
    setup_stage, method_name = request.param
    numerai_datamodule.setup(setup_stage)
    dataloader = getattr(numerai_datamodule, method_name)()
    return dataloader


@pytest.fixture(params=["train_dataset", "val_dataset", "test_dataset"], ids=param_namer("split_dataset"))
def split_dataset(request: pytest.FixtureRequest, numerai_datamodule: NumeraiDataModule) -> DataLoader:
    """Fixture that provides configured datasets for different stages."""
    method_name = request.param
    dataset = getattr(numerai_datamodule, method_name)()
    return dataset


class TestNumeraiDataset:
    @pytest.mark.requires_data
    def test_init(self, numerai_dataset: NumeraiDataset) -> None:
        assert numerai_dataset is not None

    @pytest.mark.requires_data
    def test_len(self, numerai_dataset: NumeraiDataset) -> None:
        dataset_path = data_dir / "numerai.parquet"
        len_0: int = len(numerai_dataset)
        len_1: int = (
            pl
            .scan_parquet(dataset_path)
            .filter(pl.col("era").is_between(ERA_0, ERA_1, closed="left"))
            .select(pl.len())
            .collect()
            .item()
        )  # fmt: skip
        assert len_0 == len_1

    @pytest.mark.requires_data
    def test_getitem(self, numerai_dataset: NumeraiDataset) -> None:
        item = numerai_dataset[0]
        assert len(item) == 4  # X, T, M, E
        assert all(isinstance(t, torch.Tensor) for t in item)

    @pytest.mark.requires_data
    def test_tensor_shapes(self, numerai_dataset: NumeraiDataset) -> None:
        m = len(numerai_dataset)
        X, T, M, E = numerai_dataset[:]
        assert X.shape == (m, NUM_FEATURES)
        assert T.shape == (m, NUM_TARGETS)
        assert M.shape == (m, 1)
        assert E.shape == (m, 1)

    @pytest.mark.requires_data
    def test_check_exists_after_download(self) -> None:
        dataset = NumeraiDataset(
            data_dir=data_dir,
            era_0=ERA_0,
            era_1=ERA_1,
            features=FEATURE_SET,
            targets=TARGETS,
            download=True,
        )
        assert dataset.check_exists(data_dir)


class TestNumeraiDataModule:
    @pytest.mark.requires_data
    def test_init(self, numerai_datamodule: NumeraiDataModule) -> None:
        assert numerai_datamodule is not None

    @pytest.mark.requires_data
    def test_prepare_data(self, numerai_datamodule: NumeraiDataModule) -> None:
        numerai_datamodule.prepare_data()

    @pytest.mark.requires_data
    def test_setup_fit(self, numerai_datamodule: NumeraiDataModule) -> None:
        numerai_datamodule.setup("fit")
        assert hasattr(numerai_datamodule, "numerai_train")
        assert hasattr(numerai_datamodule, "numerai_val")

    @pytest.mark.requires_data
    def test_setup_test(self, numerai_datamodule: NumeraiDataModule) -> None:
        numerai_datamodule.setup("test")
        assert hasattr(numerai_datamodule, "numerai_test")

    @pytest.mark.requires_data
    def test_dataloader(self, split_dataloader: DataLoader) -> None:
        """Test that all dataloaders return batches with correct shapes."""
        batch = next(iter(split_dataloader))
        X, T, M, E = batch
        batch_size = split_dataloader.batch_size
        assert X.shape == (batch_size, NUM_FEATURES)
        assert T.shape == (batch_size, NUM_TARGETS)
        assert M.shape == (batch_size, 1)
        assert E.shape == (batch_size, 1)

    @pytest.mark.requires_data
    def test_datasets(self, split_dataset: Data) -> None:
        """Test that all datasets return numpy arrays with correct shapes."""
        X, T, M, E = split_dataset
        m = X.shape[0]
        assert X.shape == (m, NUM_FEATURES)
        assert T.shape == (m, NUM_TARGETS)
        assert M.shape == (m, 1)
        assert E.shape == (m, 1)
