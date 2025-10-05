from hydra_zen import make_config

from numerlab.configs.utils.paths import output_dir
from numerlab.configs.utils.utils import add_colorlog, fbuilds
from numerlab.funcs.train import train

TrainCfg = make_config(
    bases=(fbuilds(train),),
    hydra_defaults=add_colorlog(
        [
            "_self_",
            {"data": "numerai_f32"},
            {"model": "linear"},
            {"module": "lightning"},
            {"trainer": "lightning"},
            {"logger": "lightning"},
            {"callbacks": "lightning"},
        ]
    ),
    seed=42,
    task_name="train",
    monitor="val/Payoff",
    mode="max",
    output_dir=output_dir,
    feature_strategy=None,
)
