import os
from typing import Literal

from torch import Tensor, tensor

from numerlab.sklearn_.datamodule import SKLearnDataModule
from numerlab.sklearn_.module import SKLearnModule
from numerlab.utils.types import Metrics


class SKLearnCheckpoint:
    """A checkpoint callback for scikit-learn models that mimics PyTorch
    Lightning's ModelCheckpoint.

    This class is designed to be used as an attribute inside SKLearnTrainer to provide
    the same interface as Lightning's ModelCheckpoint callback. It handles model saving,
    metric tracking, and provides the same properties that the training script expects.

    The properties `best_model_path` and `best_model_score` are specifically designed
    to match the interface of Lightning's ModelCheckpoint, allowing the same training
    script to work with both Lightning and scikit-learn workflows.

    Attributes:
        dirpath (str): Directory path where checkpoints are saved
        monitor (str): Metric name to monitor for best model selection
        mode (Literal["min", "max"]): Whether to minimize or maximize the monitored metric
        model_path (str): Path to the saved model file
        val_metrics (Metrics): Validation metrics dictionary
        test_metrics (Metrics): Test metrics dictionary
    """

    def __init__(
        self,
        dirpath: str,
        monitor: str,
        mode: Literal["min", "max"],
    ) -> None:
        """Initialize the SKLearnCheckpoint.

        Args:
            dirpath (str): Directory path where checkpoints will be saved
            monitor (str): Metric name to monitor for best model selection
            mode (Literal["min", "max"]): Whether to minimize or maximize the monitored metric
        """
        self.dirpath: str = dirpath
        self.monitor: str = monitor
        self.mode: Literal["min", "max"] = mode
        self.model_path: str = os.path.join(self.dirpath, "ckpt.pkl")
        self.val_metrics: Metrics = {}
        self.test_metrics: Metrics = {}

    def save(self, model: SKLearnModule) -> None:
        """Save the model to the checkpoint directory.

        Creates the directory if it doesn't exist and saves the model using joblib.

        Args:
            model (SKLearnModule): The model to save

        Returns:
            None: This method saves the model to disk but does not return values.
        """
        os.makedirs(self.dirpath, exist_ok=True)
        model.save(self.model_path)

    def populate_metrics(self, model: SKLearnModule, datamodule: SKLearnDataModule) -> None:
        """Compute and store validation and test metrics for the model.

        This method evaluates the model on both validation and test datasets
        and stores the results for later access via the properties.

        Args:
            model (SKLearnModule): The trained model to evaluate
            datamodule (SKLearnDataModule): Data module containing validation and test data

        Returns:
            None: This method computes metrics but does not return values.
        """
        X_val, y_val, m_val, e_val = datamodule.val_dataset()
        X_test, y_test, m_test, e_test = datamodule.test_dataset()
        self.val_metrics: Metrics = model.validate(X_val, y_val, m_val, e_val)
        self.test_metrics: Metrics = model.test(X_test, y_test, m_test, e_test)

    @property
    def best_model_path(self) -> str:
        """Get the path to the best model checkpoint.

        This property mimics Lightning's ModelCheckpoint.best_model_path
        to ensure compatibility with the training script.

        Returns:
            str: Path to the saved model checkpoint
        """
        return self.model_path

    @property
    def best_model_score(self) -> Tensor | None:
        """Get the best model score from the monitored metric.

        This property mimics Lightning's ModelCheckpoint.best_model_score
        to ensure compatibility with the training script. It searches for
        the monitored metric in both validation and test metrics.

        Returns:
            Tensor | None: The best score as a tensor, or None if the metric is not found
        """
        metrics: Metrics = {**self.val_metrics, **self.test_metrics}
        score: float | None = metrics.get(self.monitor, None)
        return tensor(score) if score is not None else None
