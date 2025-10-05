from hydra_zen import make_config, store
from omegaconf import MISSING
from sklearn.linear_model import LinearRegression
from torch import nn
from xgboost import XGBRegressor

from numerlab.configs.utils.utils import fbuilds, remove_types
from numerlab.model.components.progress import XGBProgressCallback

model_store = store(group="model", package="module.model", to_config=remove_types)

# ------------------------------------------------------------------------------
# Lightning
# ------------------------------------------------------------------------------

LinearCfg = fbuilds(
    nn.Linear,
    in_features=42,  # Will only work for default value of the NumeraiDataModule of using small feature set
    out_features=1,
)
model_store(LinearCfg, name="linear")


# ------------------------------------------------------------------------------
# SKLearn
# ------------------------------------------------------------------------------

LinearRegressionCfg = fbuilds(
    LinearRegression,
)
model_store(LinearRegressionCfg, name="linear_regression")

XGBoostBaseCfg = fbuilds(
    XGBRegressor,
    n_estimators=MISSING,
    learning_rate=MISSING,
    max_depth=MISSING,
    max_leaves=MISSING,
    colsample_bytree=0.1,
    tree_method="hist",
    device="cuda",
    callbacks=[
        fbuilds(
            XGBProgressCallback,
            n_estimators="${module.model.n_estimators}",
        ),
    ],
)

XGBoostSmallCfg = make_config(
    bases=(XGBoostBaseCfg,),
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    max_leaves=2**5,
)
model_store(XGBoostSmallCfg, name="xgboost_small")

XGBoostLargeCfg = make_config(
    bases=(XGBoostBaseCfg,),
    n_estimators=20000,
    learning_rate=0.001,
    max_depth=6,
    max_leaves=2**6,
)
model_store(XGBoostLargeCfg, name="xgboost_large")

XGBoostDeepCfg = make_config(
    bases=(XGBoostBaseCfg,),
    n_estimators=30000,
    learning_rate=0.001,
    max_depth=10,
    max_leaves=2**10,
)
model_store(XGBoostDeepCfg, name="xgboost_deep")
