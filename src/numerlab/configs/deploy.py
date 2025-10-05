from hydra_zen import make_config
from omegaconf import MISSING

from numerlab.configs.train import TrainCfg
from numerlab.configs.utils.paths import output_dir
from numerlab.configs.utils.utils import fbuilds
from numerlab.funcs.deploy import deploy

DeployCfg = make_config(
    bases=(fbuilds(deploy), TrainCfg),
    output_dir=output_dir,
    model_name=MISSING,
    seed=42,
    task_name="deploy",
    monitor="val/Payoff",
    mode="max",
    feature_strategy=None,
)
