from hydra_zen import store

from numerlab.configs.utils.utils import fbuilds, log_instantiation
from numerlab.data.datamodule import NumeraiDataModule

NumeraiDataModuleCfg = fbuilds(
    NumeraiDataModule,
    zen_wrappers=log_instantiation,
)

NumeraiDataModuleFloat32Cfg = fbuilds(
    NumeraiDataModule,
    features_dtype="Float32",
    zen_wrappers=log_instantiation,
)

store(NumeraiDataModuleCfg, group="data", name="numerai")
store(NumeraiDataModuleFloat32Cfg, group="data", name="numerai_f32")
