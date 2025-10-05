import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import polars as pl
import polars.selectors as cs
from lightning import LightningDataModule
from lightning.fabric.utilities.data import suggested_max_num_workers
from numerapi import NumerAPI
from numpy.typing import NDArray
from rootutils import find_root
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from numerlab.sklearn_.datamodule import SKLearnDataModule
from numerlab.utils.splits import (
    TEST_ERA_0,
    TEST_ERA_1,
    TRAIN_ERA_0,
    TRAIN_ERA_1,
    VALIDATION_ERA_0,
    VALIDATION_ERA_1,
)
from numerlab.utils.types import Batch, Data, Path_

Array = NDArray | Tensor

log = logging.getLogger(__name__)

data_dir: str = str(find_root() / "data")


def load_feature_metadata(data_dir: Path_ = data_dir) -> dict[str, Any]:
    data_dir: Path = Path(data_dir)
    with open(data_dir / "features.json") as f:
        feature_metadata = json.load(f)
    extra_sets_path = data_dir / "extra_feature_sets.json"
    if extra_sets_path.exists():
        with open(extra_sets_path) as f:
            feature_metadata["feature_sets"].update(json.load(f))
    return feature_metadata


def features_list(
    features: Iterable[str] | str | None = None,
    first_n_features: int | None = None,
    data_dir: Path_ = data_dir,
) -> list[str]:
    if features is None or isinstance(features, str):
        feature_metadata = load_feature_metadata(data_dir)
        feature_set: str = features or "small"
        features: list[str] = feature_metadata["feature_sets"][feature_set]
    else:
        features: list[str] = list(features)
    if first_n_features is not None:
        features = features[:first_n_features]
    return features


def targets_list(targets: Iterable[str] | None = None) -> list[str]:
    return list(targets) if targets else ["target"]


def numerai_dataset(
    data_dir: str = data_dir,
    era_0: int | None = None,
    era_1: int | None = None,
    features: Iterable[str] | str | None = None,
    first_n_features: int | None = None,
    targets: Iterable[str] | None = None,
    tensor_framework: Literal["torch", "numpy"] = "torch",
    features_dtype: pl.DataType = pl.Int8,
) -> tuple[Array, Array, Array, Array]:
    """Get Numerai dataset as tensor arrays.

    Args:
        data_dir: Directory where the dataset is stored. Defaults to the global data_dir.
        era_0: Starting era (inclusive). Defaults to 1 if None.
        era_1: Ending era (not inclusive). Defaults to 2 if None.
        features: Iterable of feature names or feature set name ('small', 'medium', 'large').
            If None, uses 'small' feature set.
        first_n_features: Number of features to use. Defaults to None.
        targets: Iterable of target column names. Defaults to ['target'] if None.
        tensor_framework: Framework for tensor output ('torch' or 'numpy'). Defaults to 'torch'.
        features_dtype: Data type for features. Defaults to pl.Int8.

    Returns:
        A tuple of (features, targets, meta_model, eras) as tensor arrays.
    """
    if era_0 is None or era_1 is None:
        era_0: int = 1
        era_1: int = 2

    features: list[str] = features_list(features, first_n_features, data_dir)
    targets: list[str] = targets_list(targets)

    df: pl.LazyFrame = (
        pl.scan_parquet(data_dir / "numerai.parquet")
        .filter(pl.col("era").is_between(era_0, era_1, closed="left"))
    )  # fmt: off

    def get_tensor(columns: Iterable[str] | pl.Expr, dtype: pl.DataType) -> Array:
        A = df.select(columns).collect().to_torch(dtype=dtype)
        if tensor_framework == "torch":
            return A
        elif tensor_framework == "numpy":
            return A.detach().cpu().numpy()

    X: Array = get_tensor(features, features_dtype)
    T: Array = get_tensor(targets, pl.Float32)
    M: Array = get_tensor(pl.col("numerai_meta_model"), pl.Float32)
    E: Array = get_tensor(pl.col("era"), pl.Int32)
    return X, T, M, E


class NumeraiDataset(Dataset):
    """A PyTorch Dataset for the Numerai competition data.

    This dataset provides access to Numerai data in a format suitable
    for deep learning, including features, targets, benchmark models,
    and meta model predictions.
    """

    def __init__(
        self,
        data_dir: str = data_dir,
        era_0: int | None = None,
        era_1: int | None = None,
        features: Iterable[str] | str | None = None,
        first_n_features: int | None = None,
        targets: Iterable[str] | None = None,
        download: bool = False,
        features_dtype: str = "Int8",
    ) -> None:
        """Initialize the NumeraiDataset.

        Args:
            data_dir: Directory where the dataset is stored or will be downloaded. Defaults to the global data_dir.
            era_0: Starting era (inclusive). Defaults to 1 if None.
            era_1: Ending era (not inclusive). Defaults to 2 if None.
            features: Iterable of feature names or feature set name ('small', 'medium', 'large').
                If None, uses 'small' feature set.
            first_n_features: Number of features to use. Defaults to None.
            targets: Iterable of target column names. Defaults to ['target'] if None.
            download: Whether to download the dataset if it doesn't exist.
            features_dtype: Data type for features. Defaults to 'Int8'.
        """
        data_dir: Path = Path(data_dir)
        if download:
            self.download(data_dir)

        features_dtype: pl.DataType = getattr(pl, features_dtype)

        X, T, M, E = numerai_dataset(
            data_dir=data_dir,
            era_0=era_0,
            era_1=era_1,
            features=features,
            first_n_features=first_n_features,
            targets=targets,
            tensor_framework="torch",
            features_dtype=features_dtype,
        )

        self.dataset: TensorDataset = TensorDataset(X, T, M, E)

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Batch:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A Batch tuple containing (features, targets, meta_model, era) tensors.
        """
        return self.dataset[index]

    def download(self, data_dir: Path) -> None:
        """Download and process the Numerai dataset.

        This method downloads the dataset if it doesn't exist, processes it by joining
        multiple parquet files, computing special eras, filling missing values, and
        saving the processed dataset.

        Args:
            data_dir: Directory where the dataset will be stored.
        """
        if self.check_exists(data_dir):
            return

        log.info("Downloading dataset")
        self.download_dataset(data_dir)

        log.info("Computing lazy join of datasets")
        df: pl.LazyFrame = self.join_dataset(data_dir / "v5.0")

        log.info("Computing special eras")
        eras: dict[str, int] = self.special_eras(df)

        log.info("Filling missing values")
        df: pl.LazyFrame = self.fill_missing_values(df, eras)

        log.info("Checking dataset properties")
        self.check_properties(df, eras)

        log.info(f"Saving eras file: {data_dir / 'eras.json'}")
        with open(data_dir / "eras.json", "w") as f:
            json.dump(eras, f, indent=4)

        log.info(f"Copying features from {data_dir / 'v5.0' / 'features.json'} to {data_dir / 'features.json'}")
        with open(data_dir / "v5.0" / "features.json") as src:
            features = json.load(src)
        with open(data_dir / "features.json", "w") as dst:
            json.dump(features, dst, indent=4)

        log.info(f"Saving dataset to parquet: {data_dir / 'numerai.parquet'}")
        df.sink_parquet(data_dir / "numerai.parquet")

    def check_exists(self, data_dir: Path) -> bool:
        """Check if the required dataset files exist.

        Args:
            data_dir: Directory to check for dataset files.

        Returns:
            True if all required files (numerai.parquet, eras.json, features.json) exist,
            False otherwise.
        """
        files = ["numerai.parquet", "eras.json", "features.json"]
        return all((data_dir / file).exists() for file in files)

    @staticmethod
    def download_dataset(dest_dir: Path) -> None:
        """Download all available Numerai datasets to the specified directory.

        Args:
            dest_dir: Directory where the datasets will be stored.
        """
        napi = NumerAPI()
        for dataset in napi.list_datasets():
            log.info(f"Downloading {dataset}")
            dataset_path = dest_dir / dataset
            napi.download_dataset(dataset, str(dataset_path))

    @staticmethod
    def join_dataset(data_dir: Path) -> pl.LazyFrame:
        """Join multiple Numerai datasets into a single LazyFrame.

        Args:
            data_dir: Directory containing the downloaded Numerai datasets.

        Returns:
            A lazy DataFrame containing all joined data with proper types.
        """
        df_train = pl.scan_parquet(data_dir / "train.parquet")
        df_validation = pl.scan_parquet(data_dir / "validation.parquet")
        df_live = pl.scan_parquet(data_dir / "live.parquet")
        df_train_benchmark = pl.scan_parquet(data_dir / "train_benchmark_models.parquet")
        df_validation_benchmark = pl.scan_parquet(data_dir / "validation_benchmark_models.parquet")
        df_live_benchmark = pl.scan_parquet(data_dir / "live_benchmark_models.parquet")
        df_meta_model = pl.scan_parquet(data_dir / "meta_model.parquet")

        df = pl.concat([df_train, df_validation, df_live], how="vertical")
        df_benchmark_models = pl.concat(
            [df_train_benchmark, df_validation_benchmark, df_live_benchmark], how="vertical"
        )

        df = df.join(df_benchmark_models, on="id", how="left").drop("era_right")
        df = df.join(df_meta_model, on="id", how="left").drop("era_right", "data_type_right")

        era_live = 1 + int(df.filter(pl.col("era") != "X").select(pl.col("era").max()).collect().item())
        df = df.with_columns(pl.when(pl.col("era") == "X").then(pl.lit(era_live)).otherwise(pl.col("era")).alias("era"))
        df = df.cast(
            {
                "id": pl.String,
                "era": pl.Int32,
                "data_type": pl.Enum(["train", "validation", "test", "live"]),
                cs.starts_with("feature"): pl.Int8,
                cs.starts_with("target"): pl.Float32,
                cs.starts_with("v5_lgbm_"): pl.Float32,
                "numerai_meta_model": pl.Float32,
            }
        )

        return df

    @staticmethod
    def special_eras(df: pl.LazyFrame) -> dict[str, int]:
        """Get the starting era numbers for different data types in the Numerai
        dataset.

        Args:
            df: The joined Numerai dataset.

        Returns:
            A dictionary mapping data types to their starting era numbers:
                - train: Era 1 (beginning of training data)
                - benchmark: First era with benchmark model predictions
                - validation: First era of validation data
                - metamodel: First era with meta model predictions
                - test: First era of test data
                - live: Last era in the dataset
        """

        def minimal_era(df: pl.LazyFrame) -> int:
            return df.select(pl.col("era")).min().collect().item()

        eras = {
            "train": 1,
            "benchmark": minimal_era(df.filter(pl.col("v5_lgbm_cyrusd20").is_not_null())),
            "validation": minimal_era(df.filter(pl.col("data_type") == "validation")),
            "metamodel": minimal_era(df.filter(pl.col("numerai_meta_model").is_not_null())),
            "test": minimal_era(df.filter(pl.col("data_type") == "test")),
            "live": df.select(pl.col("era")).max().collect().item(),
        }

        return eras

    @staticmethod
    def fill_missing_values(df: pl.LazyFrame, eras: dict[str, int]) -> pl.LazyFrame:
        """Fill missing target values in the dataset with appropriate defaults.

        Args:
            df: The joined Numerai dataset.
            eras: Dictionary containing special era numbers for different data types.

        Returns:
            Dataset with filled missing values:
                - target_jeremy_20: Filled with 0.5 for eras before test
                - target_jeremy_60: Filled with 0.5 for eras before (test - 8)
        """
        df = df.with_columns(
            pl.when(pl.col("era") < eras["test"])
            .then(pl.col("target_jeremy_20").fill_null(0.5))
            .otherwise(pl.col("target_jeremy_20"))
            .alias("target_jeremy_20")
        )  # fmt: skip

        df = df.with_columns(
            pl.when(pl.col("era") < eras["test"] - 8)
            .then(pl.col("target_jeremy_60").fill_null(0.5))
            .otherwise(pl.col("target_jeremy_60"))
            .alias("target_jeremy_60")
        )  # fmt: skip

        return df

    @staticmethod
    def check_properties(df: pl.LazyFrame, eras: dict[str, int]) -> None:
        """Verify that the dataset meets all expected properties and
        constraints.

        This function performs several assertions to ensure data quality:
        1. No NaN values in float columns
        2. No nulls in era, data_type, features, and id columns
        3. Benchmark models are present/absent in appropriate eras
        4. Meta model predictions are present/absent in appropriate eras
        5. Target values are present/absent in appropriate eras

        Args:
            df: The joined Numerai dataset.
            eras: Dictionary containing special era numbers for different data types.
        """
        era_benchmark: int = eras["benchmark"]
        era_metamodel: int = eras["metamodel"]
        era_test: int = eras["test"]

        def check_nulls(df: pl.LazyFrame, select_expr: pl.Expr, filter_expr: pl.Expr) -> bool:
            not_null_in_eras = all(df.filter(filter_expr).select(select_expr.is_not_null().all()).collect().row(0))
            null_outside_eras = all(df.filter(~filter_expr).select(select_expr.is_null().all()).collect().row(0))
            return not_null_in_eras and null_outside_eras

        assert all(df.select(cs.by_dtype(pl.Float32, pl.Float64).is_not_nan().all()).collect().row(0))

        assert all(
            df.select(pl.col("era"), pl.col("data_type"), pl.col("id"), cs.starts_with("feature"))
            .select(cs.all().is_not_null().all())
            .collect()
            .row(0)
        )

        benchmark_eras = pl.col("era").is_between(era_benchmark, era_test + 5, "left")
        assert check_nulls(df, cs.starts_with("v5_lgbm_"), benchmark_eras | (pl.col("data_type") == "live"))
        assert check_nulls(df, pl.col("numerai_meta_model"), pl.col("era").is_between(era_metamodel, era_test, "left"))
        assert check_nulls(df, cs.starts_with("target") & cs.ends_with("20"), pl.col("era") < era_test)
        assert check_nulls(df, cs.starts_with("target") & cs.ends_with("60"), pl.col("era") < era_test - 8)


class NumeraiDataModule(LightningDataModule, SKLearnDataModule):
    """A PyTorch Lightning DataModule for the Numerai dataset.

    This class handles the loading and preparation of Numerai data for
    training, validation, and testing in a PyTorch Lightning workflow.
    """

    def __init__(
        self,
        data_dir: str = data_dir,
        train_era_0: int = TRAIN_ERA_0,
        train_era_1: int = TRAIN_ERA_1,
        validation_era_0: int = VALIDATION_ERA_0,
        validation_era_1: int = VALIDATION_ERA_1,
        test_era_0: int = TEST_ERA_0,
        test_era_1: int = TEST_ERA_1,
        features: Iterable[str] | str | None = None,
        first_n_features: int | None = None,
        targets: Iterable[str] | None = None,
        features_dtype: str = "Int8",
        batch_size: int = 1024,
        num_workers: int | None = None,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the NumeraiDataModule.

        Args:
            data_dir: Directory where the dataset is stored. Defaults to the global data_dir.
            train_era_0: Starting era for training data (inclusive).
            train_era_1: Ending era for training data (not inclusive).
            validation_era_0: Starting era for validation data (inclusive).
            validation_era_1: Ending era for validation data (not inclusive).
            test_era_0: Starting era for test data (inclusive).
            test_era_1: Ending era for test data (not inclusive).
            features: Iterable of feature names or feature set name ('small', 'medium', 'large').
                If None, uses 'small' feature set.
            targets: Iterable of target column names. Defaults to ['target'] if None.
            features_dtype: Data type for features. Defaults to 'Int8'.
            batch_size: Batch size for DataLoader. Defaults to 1024.
            num_workers: Number of workers for DataLoader. Defaults to suggested_max_num_workers(1) if None.
            pin_memory: Whether to pin memory in DataLoader. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.train_era_0 = train_era_0
        self.train_era_1 = train_era_1
        self.validation_era_0 = validation_era_0
        self.validation_era_1 = validation_era_1
        self.test_era_0 = test_era_0
        self.test_era_1 = test_era_1
        self.features = features
        self.first_n_features = first_n_features
        self.features_list = features_list(features, first_n_features, data_dir)
        self.targets = targets
        self.targets_list = targets_list(targets)
        self.features_dtype = features_dtype
        self.features_dtype_pl = getattr(pl, features_dtype)
        self.batch_size = batch_size
        self.num_workers = suggested_max_num_workers(1) if num_workers is None else num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = self.num_workers > 0

    def prepare_data(self) -> None:
        """Download and prepare the dataset if it doesn't exist.

        This method is called automatically by PyTorch Lightning before
        setup(). It ensures the Numerai dataset is downloaded and
        processed.
        """
        NumeraiDataset(self.data_dir, download=True)

    def setup(self, stage: str) -> None:
        """Set up the dataset for the specified stage.

        Args:
            stage: Either 'fit' or 'test'.
                - 'fit': Sets up training and validation datasets
                - 'test': Sets up test dataset
        """
        if stage == "fit":
            self.numerai_train: Dataset[Batch] = NumeraiDataset(
                data_dir=self.data_dir,
                era_0=self.train_era_0,
                era_1=self.train_era_1,
                features=self.features_list,
                first_n_features=self.first_n_features,
                targets=self.targets_list,
                features_dtype=self.features_dtype,
            )
            self.numerai_val: Dataset[Batch] = NumeraiDataset(
                data_dir=self.data_dir,
                era_0=self.validation_era_0,
                era_1=self.validation_era_1,
                features=self.features_list,
                first_n_features=self.first_n_features,
                features_dtype=self.features_dtype,
            )
        if stage == "test":
            self.numerai_test: Dataset[Batch] = NumeraiDataset(
                data_dir=self.data_dir,
                era_0=self.test_era_0,
                era_1=self.test_era_1,
                features=self.features_list,
                first_n_features=self.first_n_features,
                features_dtype=self.features_dtype,
            )

    def train_dataloader(self) -> DataLoader[Batch]:
        """Get the training DataLoader.

        Returns:
            DataLoader for training data with shuffling enabled.
        """
        return DataLoader(
            self.numerai_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        """Get the validation DataLoader.

        Returns:
            DataLoader for validation data with shuffling disabled.
        """
        return DataLoader(
            self.numerai_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[Batch]:
        """Get the test DataLoader.

        Returns:
            DataLoader for test data with shuffling disabled.
        """
        return DataLoader(
            self.numerai_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataset(self) -> Data:
        """Get the training dataset as tensor arrays.

        Returns:
            A Data tuple containing (features, targets, meta_model, eras) as tensor arrays.
            Features are int8/float32, targets and meta_model are float32, eras are int32.
        """
        return numerai_dataset(
            data_dir=self.data_dir,
            era_0=self.train_era_0,
            era_1=self.train_era_1,
            features=self.features_list,
            first_n_features=self.first_n_features,
            targets=self.targets_list,
            tensor_framework="numpy",
            features_dtype=self.features_dtype_pl,
        )

    def val_dataset(self) -> Data:
        """Get the validation dataset as tensor arrays.

        Returns:
            A Data tuple containing (features, targets, meta_model, eras) as tensor arrays.
            Features are int8/float32, targets and meta_model are float32, eras are int32.
        """
        return numerai_dataset(
            data_dir=self.data_dir,
            era_0=self.validation_era_0,
            era_1=self.validation_era_1,
            features=self.features_list,
            first_n_features=self.first_n_features,
            tensor_framework="numpy",
            features_dtype=self.features_dtype_pl,
        )

    def test_dataset(self) -> Data:
        """Get the test dataset as tensor arrays.

        Returns:
            A Data tuple containing (features, targets, meta_model, eras) as tensor arrays.
            Features are int8/float32, targets and meta_model are float32, eras are int32.
        """
        return numerai_dataset(
            data_dir=self.data_dir,
            era_0=self.test_era_0,
            era_1=self.test_era_1,
            features=self.features_list,
            first_n_features=self.first_n_features,
            tensor_framework="numpy",
            features_dtype=self.features_dtype_pl,
        )
