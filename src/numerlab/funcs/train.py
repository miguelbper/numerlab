import logging
from dataclasses import dataclass

import lightning as L
import torch
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT

from numerlab.sklearn_.datamodule import DataModule
from numerlab.sklearn_.module import Module
from numerlab.sklearn_.trainer import Trainer_

log = logging.getLogger(__name__)


@dataclass
class TrainOutput:
    val_score: float | None
    test_score: float | None
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    data: DataModule | None
    module: Module | None
    trainer: Trainer_ | None


def train(
    data: DataModule,
    module: Module,
    trainer: Trainer_,
    monitor: str,
    return_objects: bool = False,
    ckpt_path: str | None = None,
    matmul_precision: str | None = "high",
    compile: bool = False,
) -> TrainOutput:
    """Train, validate and test a PyTorch Lightning or scikit-learn model.

    Args:
        data (DataModule): The data module containing training, validation and test data.
        module (Module): The model to train.
        trainer (Trainer_): The trainer instance.
        monitor (str): Metric name to monitor for best model selection.
        return_objects (bool, optional): Whether to return the data, module, and trainer objects in the output.
        ckpt_path (str | None, optional): Path to a checkpoint to resume training from. Defaults to None.
        matmul_precision (str | None, optional): Precision for matrix multiplication. Defaults to "high".
        compile (bool, optional): Whether to compile the model using torch.compile(). Defaults to False.

    Returns:
        TrainOutput: A dataclass containing metrics and optionally the data, module, and trainer objects.
    """
    if matmul_precision:
        log.info(f"Setting matmul precision to {matmul_precision}")
        torch.set_float32_matmul_precision(matmul_precision)

    if compile and isinstance(module, torch.nn.Module):
        log.info("Compiling model")
        module = torch.compile(module)

    log.info("Training model")
    trainer.fit(model=module, datamodule=data, ckpt_path=ckpt_path)
    ckpt_path: str = trainer.checkpoint_callback.best_model_path

    log.info("Validating model")
    val_out: _EVALUATE_OUTPUT = trainer.validate(model=module, datamodule=data, ckpt_path=ckpt_path)

    log.info("Testing model")
    test_out: _EVALUATE_OUTPUT = trainer.test(model=module, datamodule=data, ckpt_path=ckpt_path)

    return get_train_output(val_out, test_out, monitor, data, module, trainer, return_objects)


def get_train_output(
    val_out: _EVALUATE_OUTPUT,
    test_out: _EVALUATE_OUTPUT,
    monitor: str,
    data: DataModule,
    module: Module,
    trainer: Trainer_,
    return_objects: bool = False,
) -> TrainOutput:
    """Extract the training output from validation and test results.

    Args:
        val_out (_EVALUATE_OUTPUT): Validation output from trainer.
        test_out (_EVALUATE_OUTPUT): Test output from trainer.
        monitor (str): Metric name to monitor for best model selection.
        data (DataModule): The data module used for training.
        module (Module): The trained model.
        trainer (Trainer_): The trainer instance.
        return_objects (bool, optional): Whether to include the data, module, and trainer objects in the output.

    Returns:
        TrainOutput: A dataclass containing metrics and optionally the data, module, and trainer objects.
    """
    val_metrics: dict[str, float] = val_out[0] if val_out else {}
    test_metrics: dict[str, float] = test_out[0] if test_out else {}
    val_score: float = val_metrics.get(monitor)
    test_score: float = test_metrics.get(monitor)
    out = TrainOutput(
        val_score=val_score,
        test_score=test_score,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        data=data if return_objects else None,
        module=module if return_objects else None,
        trainer=trainer if return_objects else None,
    )
    return out


def seed_fn(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set for all random number generators.
    """
    log.info(f"Setting seed to {seed}")
    L.seed_everything(seed, workers=True, verbose=False)
