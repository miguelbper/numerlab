import logging
from abc import ABC, abstractmethod
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class FeatureSelectionStrategy(ABC):
    """Abstract base class for feature selection strategies.

    This class provides a template for implementing feature selection algorithms
    that can be applied to the Numerai dataset. Subclasses must implement the
    `inputs()` and `strategy()` methods to define how features are selected.

    The class follows a two-step process:
    1. `inputs()`: Extracts and prepares data from the configuration
    2. `strategy()`: Executes the actual feature selection algorithm

    Usage:
        class MyFeatureSelector(FeatureSelectionStrategy):
            def inputs(self, cfg):
                # Extract data from config
                return (data,)

            def strategy(self, data):
                # Implement selection logic
                return selected_features
    """

    def __call__(self, cfg: DictConfig) -> None:
        """Execute the feature selection strategy and update the configuration.

        This method orchestrates the feature selection process by:
        1. Extracting inputs from the configuration using `inputs()`
        2. Executing the selection strategy using `strategy()`
        3. Updating the configuration with selected features

        Args:
            cfg: The configuration object containing data and parameters
        """
        log.info(f"Selecting features with {self.__class__.__name__}")
        selected_features = self.strategy(cfg)

        num_selected_features: int = len(selected_features)
        features_path: Path = Path(cfg.output_dir) / "selected_features.txt"
        log.info(f"Selected {num_selected_features} features. Writing selected features to {features_path}")
        with open(features_path, "w") as f:
            f.write("\n".join(selected_features))

        log.info("Setting data.features to be the computed features")
        cfg.data.features = selected_features

    @abstractmethod
    def strategy(self, cfg: DictConfig) -> list[str]:
        """Execute the feature selection strategy.

        Args:
            cfg: The configuration object containing data and parameters

        Returns:
            List of selected feature names
        """
        pass


def select_features(cfg: DictConfig) -> None:
    """Call the feature selection strategy if one is configured.

    This function checks if a feature selection strategy is configured in the
    configuration object and executes it if present. If no strategy is configured,
    the function does nothing.

    Args:
        cfg: The configuration object that may contain a 'feature_strategy' key
    """
    feature_strategy_cfg: DictConfig | None = cfg.get("feature_strategy", None)
    if feature_strategy_cfg is not None:
        strategy: FeatureSelectionStrategy = instantiate(feature_strategy_cfg)
        strategy(cfg)
