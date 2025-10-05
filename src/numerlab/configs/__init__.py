from pathlib import Path

from rootutils import find_root

from numerlab.configs.deploy import DeployCfg
from numerlab.configs.train import TrainCfg
from numerlab.utils.imports import import_modules

__all__ = ["TrainCfg", "DeployCfg"]

root_dir: Path = find_root()
numerai_dir: Path = root_dir / "src" / "numerlab"
configs_dir: Path = numerai_dir / "configs"

import_modules(configs_dir)
