from hydra_zen import store
from lightning import Trainer

from numerlab.configs.utils.paths import output_dir
from numerlab.configs.utils.utils import fbuilds, log_instantiation
from numerlab.sklearn_.trainer import SKLearnTrainer

TrainerCfg = fbuilds(
    Trainer,
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir=output_dir,
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)


SKLearnTrainerCfg = fbuilds(
    SKLearnTrainer,
    zen_wrappers=log_instantiation,
)


store(TrainerCfg, group="trainer", name="lightning")
store(SKLearnTrainerCfg, group="trainer", name="sklearn")
