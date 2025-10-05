from lightgbm.callback import CallbackEnv
from tqdm import tqdm
from xgboost.callback import TrainingCallback
from xgboost.core import Booster


class XGBProgressCallback(TrainingCallback):
    def __init__(self, n_estimators: int):
        self.pbar = None
        self.n_estimators = n_estimators

    def before_training(self, model: Booster) -> Booster:
        """Initialize progress bar before training starts.

        Args:
            model (Booster): The XGBoost model.

        Returns:
            Booster: The model unchanged.
        """
        self.pbar = tqdm(total=self.n_estimators, desc="Training XGBoost")
        return model

    def after_iteration(self, model: Booster, epoch: int, evals_log: dict[str, dict]) -> bool:
        """Update progress bar after each iteration.

        Args:
            model (Booster): The XGBoost model.
            epoch (int): Current epoch number.
            evals_log (dict[str, dict]): Evaluation logs.

        Returns:
            bool: False to continue training.
        """
        self.pbar.update(1)
        return False  # Continue training

    def after_training(self, model: Booster) -> Booster:
        """Clean up progress bar after training completes.

        Args:
            model (Booster): The XGBoost model.

        Returns:
            Booster: The trained model.
        """
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        return model


class CatBoostProgressCallback:
    def __init__(self, n_estimators: int) -> None:
        self.n_estimators = n_estimators
        self.pbar = None

    def after_iteration(self, info) -> bool:
        if self.pbar is None:
            self.pbar = tqdm(total=self.n_estimators, desc="Training CatBoost")

        self.pbar.update(1)

        if info.iteration >= self.n_estimators:
            self.pbar.close()
            self.pbar = None

        return True


class LGBMProgressCallback:
    def __init__(self, n_estimators: int) -> None:
        self.n_estimators = n_estimators
        self.pbar = None

    def _init(self, env: CallbackEnv) -> None:
        """Initialize progress bar before training starts.

        Args:
            env (CallbackEnv): The LightGBM callback environment.
        """
        self.pbar = tqdm(total=self.n_estimators, desc="Training LightGBM")

    def __call__(self, env: CallbackEnv) -> None:
        """Update progress bar after each iteration.

        Args:
            env (CallbackEnv): The LightGBM callback environment.
        """
        if env.iteration == env.begin_iteration:
            self._init(env)

        if self.pbar is not None:
            self.pbar.update(1)

        if (env.iteration >= self.n_estimators - 1) and (self.pbar is not None):
            self.pbar.close()
            self.pbar = None
