from hydra_zen import store

from numerlab.configs.utils.utils import remove_types

experiment_store = store(group="experiment", package="_global_", to_config=remove_types)
