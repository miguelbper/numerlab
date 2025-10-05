import gc
import logging
import os
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import polars.selectors as cs
from catboost import CatBoostRegressor
from flash_select import flash_select
from lightgbm import LGBMRegressor
from omegaconf import DictConfig
from xgboost import XGBRegressor

from numerlab.features.feature_selection import FeatureSelectionStrategy
from numerlab.utils.splits import EMBARGO

log = logging.getLogger(__name__)
TreeModel = XGBRegressor | LGBMRegressor | CatBoostRegressor


FEATURE_NAME = "feature name"
SELECTED = "selected"


class FlashSelect(FeatureSelectionStrategy):
    """SHAP-based iterative feature selection strategy.

    This class implements a feature selection method that uses SHAP values to identify
    features with positive linear relationships to the target. The method works by
    fitting a linear regression model between target values and SHAP values,
    then iteratively removing features with the most negative t-statistics.

    The algorithm:
    1. Trains a tree-based regressor (XGBoost, LightGBM, or CatBoost) on embargoed training data
    2. Computes SHAP values for validation data
    3. Fits linear regression: targets ~ SHAP_values
    4. Iteratively removes features with the most negative t-statistics
    5. Continues until all remaining features have positive coefficients or meet significance threshold

    This approach ensures that selected features have positive contributions to
    the target prediction when explained through SHAP values, filtering out
    features that may be correlated but not directly predictive.

    Args:
        model: Tree-based regressor (XGBoost, LightGBM, or CatBoost) to use for SHAP computation
        num_val_eras: Number of validation eras to use
        embargo: Number of eras to embargo from training (default from EMBARGO constant)
        starting_features: Initial feature set to consider (None for all features)
        threshold: p-value threshold for feature significance (default 0.05)
    """

    def __init__(
        self,
        model: TreeModel,
        num_val_eras: int,
        embargo: int = EMBARGO,
        starting_features: list[str] | None = None,
        threshold: float = 0.05,
    ) -> None:
        """Initialize the SHAP Select feature selection strategy.

        Args:
            model: Tree-based regressor (XGBoost, LightGBM, or CatBoost) to use for SHAP computation
            num_val_eras: Number of validation eras to use
            embargo: Number of eras to embargo from training (default from EMBARGO constant)
            starting_features: Initial feature set to consider (None for all features)
            threshold: p-value threshold for feature significance (default 0.05)
        """
        self.model = model
        self.num_val_eras = num_val_eras
        self.embargo = embargo
        self.starting_features = starting_features
        self.threshold = threshold

    def get_dataset(self, dataset_path: Path, era_0: int, era_1: int) -> tuple[pl.DataFrame, pl.Series]:
        """Load and filter dataset for the specified era range.

        Args:
            dataset_path: Path to the Numerai parquet dataset
            era_0: Starting era (inclusive)
            era_1: Ending era (exclusive)

        Returns:
            Tuple of (features_df, target_series) where features_df contains
            the selected features and target_series contains the target values.
        """
        df = pl.scan_parquet(dataset_path).filter(pl.col("era").is_between(era_0, era_1, closed="left"))
        X = df.select(self.starting_features or cs.starts_with("feature")).collect()
        y = df.select(pl.col("target")).collect().to_series()
        return X, y

    def strategy(self, cfg: DictConfig) -> list[str]:
        """Execute the SHAP Select feature selection strategy.

        This method implements the core feature selection algorithm:
        1. Trains the tree-based model on embargoed training data
        2. Computes SHAP values for the validation data
        3. Iteratively fits linear regression of targets on SHAP values
        4. Removes features with the most negative t-statistics
        5. Continues until all remaining features have positive coefficients or meet significance threshold

        The linear regression model is: targets = β * SHAP_values
        Features are removed based on their t-statistics, prioritizing those with the most negative values.
        The final selection includes features with positive coefficients and significant p-values.

        Args:
            cfg: DictConfig with data and path settings.

        Returns:
            List of selected feature names with positive linear relationships.
        """
        train_era_0_outer: int = cfg.data.train_era_0
        train_era_1_outer: int = cfg.data.train_era_1

        val_era_1: int = train_era_1_outer
        val_era_0: int = val_era_1 - self.num_val_eras
        train_era_1: int = val_era_0 - self.embargo
        train_era_0: int = train_era_0_outer

        dataset_path = os.path.join(cfg.data.data_dir, "numerai.parquet")

        log.info(f"Training explainer model in eras [{train_era_0}, {train_era_1})")
        L.seed_everything(42)
        X_train, y_train = self.get_dataset(dataset_path, train_era_0, train_era_1)
        self.model.fit(X_train.to_numpy(), y_train.to_numpy())

        del X_train, y_train
        gc.collect()

        log.info(f"Running SHAP Select in eras [{val_era_0}, {val_era_1})")
        X_val, y_val = self.get_dataset(dataset_path, val_era_0, val_era_1)
        X = X_val.to_numpy()
        y = y_val.to_numpy()
        features = X_val.columns
        df: pl.DataFrame = pl.from_pandas(
            flash_select(
                self.model,
                X,
                y,
                features,
                self.threshold,
                dtype=np.float64,
                batch_size=4096,
            )
        )

        return df.filter(pl.col(SELECTED) == 1).select(FEATURE_NAME).to_series().to_list()
