from hydra_zen import store
from omegaconf import MISSING
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

from numerlab.configs.utils.utils import fbuilds, log_instantiation
from numerlab.metrics.erawise import corr, mmc, payoff
from numerlab.metrics.torchmetrics_ import MMC, Corr, Payoff
from numerlab.model.module import Module
from numerlab.sklearn_.module import SKLearnModule

# ------------------------------------------------------------------------------
# Lightning
# ------------------------------------------------------------------------------

ModuleCfg = fbuilds(
    Module,
    model=MISSING,
    loss_fn=fbuilds(
        nn.MSELoss,
    ),
    optimizer=fbuilds(
        Adam,
        lr=0.001,
        zen_partial=True,
    ),
    scheduler=fbuilds(
        ReduceLROnPlateau,
        mode="min",
        factor=0.1,
        patience=10,
        zen_partial=True,
    ),
    metric_collection=fbuilds(
        MetricCollection,
        metrics=[
            fbuilds(Corr),
            fbuilds(MMC),
            fbuilds(Payoff),
        ],
    ),
    zen_wrappers=log_instantiation,
)


# ------------------------------------------------------------------------------
# SKLearn
# ------------------------------------------------------------------------------

SKLearnModuleCfg = fbuilds(
    SKLearnModule,
    model=MISSING,
    metrics=[
        corr,
        mmc,
        payoff,
    ],
    zen_wrappers=log_instantiation,
)


# ------------------------------------------------------------------------------
# Store
# ------------------------------------------------------------------------------

module_store = store(group="module")
module_store(ModuleCfg, name="lightning")
module_store(SKLearnModuleCfg, name="sklearn")
