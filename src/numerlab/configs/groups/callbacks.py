import os

from hydra_zen import make_config, store
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar

from numerlab.configs.utils.paths import output_dir
from numerlab.configs.utils.utils import fbuilds, remove_types
from numerlab.sklearn_.checkpoint import SKLearnCheckpoint
from numerlab.utils.logging_ import LogConfigToMLflow

# ------------------------------------------------------------------------------
# Lightning
# ------------------------------------------------------------------------------

RichProgressBarCfg = fbuilds(
    RichProgressBar,
)

RichModelSummaryCfg = fbuilds(
    RichModelSummary,
)

EarlyStoppingCfg = fbuilds(
    EarlyStopping,
    monitor="${monitor}",
    patience=3,
    mode="${mode}",
)

ModelCheckpointCfg = fbuilds(
    ModelCheckpoint,
    dirpath=os.path.join(output_dir, "checkpoints"),
    filename="epoch_{epoch:03d}",
    monitor="${monitor}",
    save_last=True,
    mode="${mode}",
    auto_insert_metric_name=False,
)

LogConfigToMLflowCfg = fbuilds(
    LogConfigToMLflow,
)

LightningCallbacksCfg = make_config(
    callbacks=[
        RichProgressBarCfg,
        RichModelSummaryCfg,
        EarlyStoppingCfg,
        ModelCheckpointCfg,
        LogConfigToMLflowCfg,
    ],
)

# ------------------------------------------------------------------------------
# SKLearn
# ------------------------------------------------------------------------------

SKLearnCheckpointCfg = fbuilds(
    SKLearnCheckpoint,
    dirpath=os.path.join(output_dir, "checkpoints"),
    monitor="${monitor}",
    mode="${mode}",
)

SKLearnCallbacksCfg = make_config(
    callbacks=[
        SKLearnCheckpointCfg,
    ],
)

# ------------------------------------------------------------------------------
# Store
# ------------------------------------------------------------------------------

callbacks_store = store(group="callbacks", package="trainer", to_config=remove_types)
callbacks_store(LightningCallbacksCfg, name="lightning")
callbacks_store(SKLearnCallbacksCfg, name="sklearn")
