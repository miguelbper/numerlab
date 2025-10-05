import copy
import json
import logging
import os
from importlib.metadata import distributions
from itertools import chain
from pathlib import Path
from typing import Any

import mlflow
import rich
import rich.syntax
import rich.tree
from git import Repo
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, ListConfig, OmegaConf, flag_override
from rich.tree import Tree
from rootutils import find_root

from numerlab.data.datamodule import numerai_dataset
from numerlab.sklearn_.datamodule import SKLearnDataModule
from numerlab.sklearn_.module import SKLearnModule
from numerlab.utils.types import Metrics, Path_

log = logging.getLogger(__name__)


def print_config(cfg: DictConfig) -> None:
    """Print and save the configuration tree to a file.

    This function takes a Hydra configuration object, removes specified packages,
    converts it to a rich tree structure, and saves it to a config_tree.log file
    in the output directory.

    Args:
        cfg (DictConfig): The Hydra configuration object to print and save.
    """
    cfg: DictConfig = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    output_dir: str = cfg.output_dir
    log.info(f"Output directory is {output_dir}")
    config_file: str = os.path.join(output_dir, "config_tree.log")

    cfg: DictConfig = remove_packages(cfg)
    tree: Tree = cfg_to_tree(cfg)

    log.info(f"Saving config to {config_file}")
    with open(config_file, "w") as file:
        rich.print(tree, file=file)
    rich.print(tree)

    flat_cfg: dict[str, Any] = flatten_cfg(cfg)
    with open(os.path.join(output_dir, "config_flat.json"), "w") as file:
        json.dump(flat_cfg, file, indent=2)


def flatten_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Convert a DictConfig to a flattened dictionary.

    Args:
        cfg (DictConfig): The DictConfig to convert.

    Returns:
        dict[str, Any]: A flattened dictionary where nested keys are joined with dots.
    """
    result: dict[str, Any] = {}

    def _process_dict(d: DictConfig, prefix: str = "") -> None:
        for key, value in d.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, DictConfig):
                new_prefix = f"{full_key}."
                _process_dict(value, new_prefix)
            elif isinstance(value, ListConfig):
                for i, item in enumerate(value):
                    list_key = f"{full_key}.{i}"
                    if isinstance(item, DictConfig | ListConfig):
                        _process_dict(item, f"{list_key}.")
                    else:
                        result[list_key] = item
            else:
                result[full_key] = value

    _process_dict(cfg)
    return result


def remove_packages(cfg: DictConfig, packages: tuple[str, ...] = ("hydra", "paths")) -> DictConfig:
    """Remove specified packages from the configuration object.

    Args:
        cfg (DictConfig): The configuration object to modify.
        packages (tuple[str, ...], optional): Tuple of package names to remove.
            Defaults to ("hydra", "paths").

    Returns:
        DictConfig: A copy of the configuration object with specified packages removed.
    """
    for package in packages:
        if package in cfg:
            cfg = copy.copy(cfg)
            with flag_override(cfg, ["struct", "readonly"], [False, False]):
                cfg.pop(package)
    return cfg


def cfg_to_tree(cfg: DictConfig) -> Tree:
    """Convert a configuration object to a rich tree structure.

    Args:
        cfg (DictConfig): The configuration object to convert.

    Returns:
        rich.tree.Tree: A rich tree structure representing the configuration.
    """

    def add_to_tree(tree: Tree, level: int, cfg: DictConfig) -> None:
        colors = ["red", "cyan", "blue", "green"]
        color = colors[min(level, len(colors) - 1)]
        for key, value in cfg.items():
            key_format = f"[bold {color}]{key}[/bold {color}]"
            if isinstance(value, DictConfig):
                sub_tree = tree.add(key_format)
                add_to_tree(sub_tree, level + 1, value)
            elif isinstance(value, ListConfig):
                sub_tree = tree.add(key_format)
                add_to_tree(sub_tree, level + 1, OmegaConf.create({str(i): item for i, item in enumerate(value)}))
            else:
                tree.add(f"{key_format}: {value}")

    tree = Tree("config")
    add_to_tree(tree, 0, cfg)
    return tree


def log_python_env(cfg: DictConfig) -> None:
    """Log the current Python environment to a file.

    This function creates a file containing a list of all installed Python packages
    and their versions in the output directory. The packages are sorted alphabetically
    and formatted as 'package_name==version'.

    Args:
        cfg (DictConfig): The configuration object containing output directory path.
    """
    output_dir: str = cfg.output_dir
    python_env_file: str = os.path.join(output_dir, "python_env.log")
    log.info(f"Logging Python environment to {python_env_file}")
    installed_packages = sorted(f"{dist.metadata['Name']}=={dist.version}\n" for dist in distributions())
    with open(python_env_file, "w") as file:
        file.writelines(installed_packages)


def log_git_status(cfg: DictConfig) -> None:
    """Log git repository status and changes to a file.

    This function creates a file containing:
    1. The current commit hash
    2. A detailed diff of all uncommitted changes
    3. A summary of the git status

    The information is written to a file in the output directory. If the repository
    cannot be accessed or an error occurs, an error message is written instead.

    Args:
        cfg (DictConfig): The configuration object containing output directory path.
    """
    output_dir: str = cfg.output_dir
    git_status_file: str = os.path.join(output_dir, "git_status.log")
    log.info(f"Logging git status to {git_status_file}")

    repo = Repo(search_parent_directories=True)
    with open(git_status_file, "w") as file:
        file.write(f"Commit hash: {repo.head.commit.hexsha}\n\n")
        file.write(repo.git.diff() + "\n\n")
        file.write(repo.git.status())


class LogConfigToMLflow(Callback):
    """A PyTorch Lightning callback that logs configuration files to MLflow.

    This callback logs all configuration files from the output directory and
    .hydra directory to MLflow when training starts. It requires an MLflow
    logger to be present in the trainer's loggers.

    Attributes:
        None
    """

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log configuration files to MLflow when training starts.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (LightningModule): The PyTorch Lightning module being trained.
        """
        mlf_logger: MLFlowLogger | None = next((lg for lg in trainer.loggers if isinstance(lg, MLFlowLogger)), None)
        if mlf_logger is None:
            log.warning("No MLflow logger found, skipping logging to MLflow")
            return
        mlf_client: MlflowClient = mlf_logger.experiment

        output_dir: str = trainer.default_root_dir
        hydra_dir: str = os.path.join(output_dir, ".hydra")
        dirs: list[str] = [output_dir, hydra_dir]
        entries: list[os.DirEntry] = list(chain.from_iterable(os.scandir(dir) for dir in dirs))
        files: list[str] = [entry.path for entry in entries if entry.is_file()]

        for file_path in files:
            parent_folder: Path = Path(file_path).relative_to(output_dir).parent
            artifact_path: str | None = str(parent_folder) if parent_folder.parts else None
            mlf_client.log_artifact(
                run_id=mlf_logger.run_id,
                local_path=file_path,
                artifact_path=artifact_path,
            )

        flat_config_path = os.path.join(output_dir, "config_flat.json")
        if os.path.exists(flat_config_path):
            with open(flat_config_path) as f:
                params = json.load(f)
                mlf_logger.log_hyperparams(params)


class SKLearnMLflowLogger:
    def __init__(
        self,
        experiment_name: str = "sklearn_logs",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        log_model: bool = False,
        output_dir: str | None = None,
    ) -> None:
        self.experiment_name: str = experiment_name
        self.run_name: str | None = run_name
        self.tracking_uri: str | None = tracking_uri
        self.log_model: bool = log_model
        self.output_dir: str | None = output_dir

        mlflow_dir = find_root() / "logs" / "mlflow" / "mlruns"
        mlflow.set_tracking_uri(self.tracking_uri or mlflow_dir)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

    def log_files(self) -> None:
        """Log configuration files to MLflow.

        This method logs all configuration files from the output
        directory and .hydra directory to MLflow. It also logs flattened
        configuration parameters.
        """
        if self.output_dir is None:
            log.warning("No output directory provided, skipping logging of files to MLflow")
            return
        output_dir: str = self.output_dir
        hydra_dir: str = os.path.join(output_dir, ".hydra")
        dirs: list[str] = [output_dir, hydra_dir]
        entries: list[os.DirEntry] = list(chain.from_iterable(os.scandir(dir) for dir in dirs))
        files: list[str] = [entry.path for entry in entries if entry.is_file()]

        for file_path in files:
            parent_folder: Path = Path(file_path).relative_to(output_dir).parent
            artifact_path: str | None = str(parent_folder) if parent_folder.parts else None
            mlflow.log_artifact(local_path=file_path, artifact_path=artifact_path)

        flat_config_path = os.path.join(output_dir, "config_flat.json")
        if os.path.exists(flat_config_path):
            with open(flat_config_path) as f:
                params = json.load(f)
                for key, value in params.items():
                    mlflow.log_param(key, value)

    def log_checkpoint(
        self,
        model: SKLearnModule,
        data: SKLearnDataModule,
        ckpt_path: Path_ | None = None,
    ) -> None:
        """Log model checkpoint to MLflow.

        Args:
            model (SKLearnModule): The trained model to log.
            data (SKLearnDataModule): Data module containing validation data.
            ckpt_path (Path_ | None): Path to the checkpoint file. If None, logs the model directly.
        """
        if not ckpt_path:
            log.warning("No checkpoint path provided, skipping logging of checkpoint to MLflow")
            return
        mlflow.log_artifact(ckpt_path)

        X, y, _, _ = numerai_dataset(
            data_dir=data.data_dir,
            features=data.features,
            targets=data.targets,
            tensor_framework="numpy",
            features_dtype=data.features_dtype_pl,
        )

        mlflow.sklearn.log_model(
            sk_model=model.model,
            signature=infer_signature(X, y),
            input_example=X[:1],
        )

    def log_metrics(self, metrics: Metrics) -> None:
        """Log metrics to MLflow.

        Args:
            metrics (Metrics): Dictionary of metric names and values to log.
        """
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    def end_run(self) -> None:
        """Stop the current MLflow run.

        This method stops the current MLflow run and logs the run to
        MLflow.
        """
        mlflow.end_run()
