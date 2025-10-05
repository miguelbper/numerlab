import logging

from hydra_zen import store, zen

from numerlab.configs import DeployCfg
from numerlab.features.feature_selection import select_features
from numerlab.funcs.deploy import deploy
from numerlab.funcs.train import seed_fn
from numerlab.utils.logging_ import log_git_status, log_python_env, print_config

log = logging.getLogger(__name__)


def main() -> None:
    store(DeployCfg, name="config")
    store.add_to_hydra_store(overwrite_ok=True)
    task_fn = zen(deploy, pre_call=[log_git_status, log_python_env, select_features, zen(seed_fn), print_config])
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
