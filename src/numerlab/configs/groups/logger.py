import os

from hydra_zen import make_config, store
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger

from numerlab.configs.utils.paths import log_dir, output_dir
from numerlab.configs.utils.utils import fbuilds, remove_types
from numerlab.utils.logging_ import SKLearnMLflowLogger

# ------------------------------------------------------------------------------
# Lightning
# ------------------------------------------------------------------------------

CSVLoggerCfg = fbuilds(
    CSVLogger,
    save_dir=output_dir,
    name="csv",
)

TensorBoardLoggerCfg = fbuilds(
    TensorBoardLogger,
    save_dir=output_dir,
    name="tensorboard",
)

MLFlowLoggerCfg = fbuilds(
    MLFlowLogger,
    tracking_uri=os.path.join(log_dir, "mlflow", "mlruns"),
    experiment_name="${task_name}",
    log_model=True,
)

LightningLoggerCfg = make_config(
    logger=[
        CSVLoggerCfg,
        TensorBoardLoggerCfg,
        MLFlowLoggerCfg,
    ],
)


# ------------------------------------------------------------------------------
# SKLearn
# ------------------------------------------------------------------------------


SKLearnMLflowLoggerCfg = fbuilds(
    SKLearnMLflowLogger,
    tracking_uri=os.path.join(log_dir, "mlflow", "mlruns"),
    experiment_name="${task_name}",
    log_model=True,
    output_dir=output_dir,
)

SKLearnLoggerCfg = make_config(
    logger=SKLearnMLflowLoggerCfg,
)


# ------------------------------------------------------------------------------
# Store
# ------------------------------------------------------------------------------

logger_store = store(group="logger", package="trainer", to_config=remove_types)
logger_store(LightningLoggerCfg, name="lightning")
logger_store(SKLearnLoggerCfg, name="sklearn")
