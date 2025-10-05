from hydra_zen import launch, store, zen

from numerlab.configs import TrainCfg
from numerlab.sklearn_.datamodule import DataModule
from numerlab.sklearn_.module import Module
from numerlab.sklearn_.trainer import Trainer_


def mock(
    data: DataModule,
    module: Module,
    trainer: Trainer_,
    monitor: str,
    return_objects: bool = False,
    ckpt_path: str | None = None,
    matmul_precision: str | None = "high",
    compile: bool = False,
) -> None:
    pass


def test_train_config() -> None:
    store.add_to_hydra_store(overwrite_ok=True)
    launch(TrainCfg, zen(mock), version_base="1.3")
