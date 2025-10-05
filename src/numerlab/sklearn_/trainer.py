from collections.abc import Callable
from typing import Literal

from lightning import Trainer
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT
from numpy.typing import ArrayLike
from rich.console import Console
from rich.table import Table

from numerlab.sklearn_.checkpoint import SKLearnCheckpoint
from numerlab.sklearn_.datamodule import SKLearnDataModule
from numerlab.sklearn_.module import SKLearnModule
from numerlab.utils.logging_ import SKLearnMLflowLogger
from numerlab.utils.types import Metrics, Path_


class SKLearnTrainer:
    """A trainer class for scikit-learn models that mimics PyTorch Lightning's
    Trainer interface.

    This class provides training, validation, and testing capabilities for scikit-learn
    models with the same method signatures as Lightning's Trainer. This allows the same
    training script to work with both Lightning and scikit-learn workflows without
    modification.

    The trainer includes a checkpoint callback as an attribute (similar to Lightning's
    trainer.checkpoint_callback) and provides the same fit(), validate(), and test()
    methods with identical parameter signatures.

    Attributes:
        checkpoint_callback (SKLearnCheckpoint): Checkpoint callback for model persistence
    """

    def __init__(
        self,
        logger: SKLearnMLflowLogger | None = None,
        callbacks: list[SKLearnCheckpoint] | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
    ) -> None:
        """Initialize the SKLearnTrainer.

        Args:
            checkpoint_callback (SKLearnCheckpoint): Checkpoint callback that will be
                stored as an attribute to mimic Lightning's trainer.checkpoint_callback
        """
        self.logger = logger
        self.callbacks = callbacks
        self.checkpoint_callback = callbacks[0] if callbacks else None
        self.run_val = limit_val_batches is None or limit_val_batches > 0
        self.run_test = limit_test_batches is None or limit_test_batches > 0

    def fit(self, model: SKLearnModule, datamodule: SKLearnDataModule, ckpt_path: Path_ | None = None) -> None:
        """Train a model using the provided datamodule.

        This method has the same signature as Lightning's Trainer.fit() to ensure
        compatibility with the training script. It trains the model and automatically
        saves it using the checkpoint callback.

        Args:
            model (SKLearnModule): The model to train
            datamodule (SKLearnDataModule): DataModule containing the training data
            ckpt_path (Path_ | None, optional): Path to resume training from (not used in sklearn)

        Returns:
            None: This method trains the model in-place but does not return values.
        """
        if self.logger:
            self.logger.log_files()

        X, y, _, _ = datamodule.train_dataset()
        model.train(X, y)

        if self.checkpoint_callback:
            self.checkpoint_callback.save(model=model)
            self.checkpoint_callback.populate_metrics(model=model, datamodule=datamodule)

            if self.logger:
                self.logger.log_metrics(self.checkpoint_callback.val_metrics | self.checkpoint_callback.test_metrics)
                self.logger.log_checkpoint(model=model, data=datamodule, ckpt_path=self.checkpoint_callback.model_path)
                self.logger.end_run()

    def validate(
        self,
        model: SKLearnModule | None,
        datamodule: SKLearnDataModule,
        ckpt_path: Path_ | None = None,
    ) -> _EVALUATE_OUTPUT:
        """Evaluate the model on validation data.

        This method has the same signature as Lightning's Trainer.validate() to ensure
        compatibility with the training script. It can load a model from checkpoint
        if no model is provided.

        Args:
            model (SKLearnModule | None): The model to validate, can be None if ckpt_path is provided
            datamodule (SKLearnDataModule): DataModule containing the validation data
            ckpt_path (Path_ | None, optional): Path to load model from, required if model is None

        Returns:
            _EVALUATE_OUTPUT: List containing a dictionary of validation metrics (matching Lightning format)

        Raises:
            ValueError: If neither model nor ckpt_path is provided, or if model is not trained
        """
        return self.evaluate(model, datamodule, ckpt_path, "validation") if self.run_val else []

    def test(
        self,
        model: SKLearnModule | None,
        datamodule: SKLearnDataModule,
        ckpt_path: Path_ | None = None,
    ) -> _EVALUATE_OUTPUT:
        """Evaluate the model on test data.

        This method has the same signature as Lightning's Trainer.test() to ensure
        compatibility with the training script. It can load a model from checkpoint
        if no model is provided.

        Args:
            model (SKLearnModule | None): The model to test, can be None if ckpt_path is provided
            datamodule (SKLearnDataModule): DataModule containing the test data
            ckpt_path (Path_ | None, optional): Path to load model from, required if model is None

        Returns:
            _EVALUATE_OUTPUT: List containing a dictionary of test metrics (matching Lightning format)

        Raises:
            ValueError: If neither model nor ckpt_path is provided, or if model is not trained
        """
        return self.evaluate(model, datamodule, ckpt_path, "test") if self.run_test else []

    def evaluate(
        self,
        model: SKLearnModule | None,
        datamodule: SKLearnDataModule,
        ckpt_path: Path_ | None,
        split: Literal["validation", "test"],
    ) -> _EVALUATE_OUTPUT:
        """Common evaluation logic for validation and test.

        This internal method handles the common evaluation logic for both validation
        and test splits. It first checks if metrics are already computed in the
        checkpoint callback, and if not, loads the model and computes them.

        Args:
            model (SKLearnModule | None): Model to evaluate, can be None if ckpt_path is provided
            datamodule (SKLearnDataModule): DataModule containing the data
            ckpt_path (Path_ | None): Optional path to load model from, required if model is None
            split (Literal["validation", "test"]): Which split to evaluate ('validation' or 'test')

        Returns:
            _EVALUATE_OUTPUT: List containing a dictionary of evaluation metrics (matching Lightning format)

        Raises:
            ValueError: If neither model nor ckpt_path is provided, or if model is not trained
        """
        checkpoint_attr: str = "val_metrics" if split == "validation" else "test_metrics"
        metrics: Metrics = getattr(self.checkpoint_callback, checkpoint_attr) if self.checkpoint_callback else None

        if not metrics:
            if model is None and ckpt_path is None:
                raise ValueError("Either model or ckpt_path must be provided")
            if ckpt_path:
                model = SKLearnModule.load(ckpt_path)
            if not model.trained:
                raise ValueError("Model must be trained before evaluation")

            datamodule_func_name: str = "val_dataset" if split == "validation" else "test_dataset"
            datamodule_func: Callable[[], tuple[ArrayLike, ArrayLike]] = getattr(datamodule, datamodule_func_name)
            split_prefix: str = "val/" if split == "validation" else "test/"
            X, y, m, e = datamodule_func()
            metrics: Metrics = model.evaluate(X, y, m, e, split_prefix)

        self.print_metrics(metrics, split.capitalize())
        return [metrics]

    @staticmethod
    def print_metrics(metrics: Metrics, prefix: str) -> None:
        """Pretty print metrics in a table format using Rich.

        Args:
            metrics (Metrics): Dictionary of metric names and values
            prefix (str): Prefix to use in the title (e.g., 'Validation' or 'Test')

        Returns:
            None: This method prints to console but does not return values.
        """
        console = Console()
        table = Table()
        table.add_column(f"{prefix} metric", style="cyan")
        table.add_column("Value", style="magenta")

        for name, value in metrics.items():
            table.add_row(name, f"{value:.16f}")

        console.print(table)


Trainer_ = Trainer | SKLearnTrainer
