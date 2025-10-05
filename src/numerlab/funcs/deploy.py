import logging
import os
from collections.abc import Callable
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numerapi import NumerAPI
from numpy.typing import NDArray

from numerlab.funcs.train import TrainOutput, train
from numerlab.sklearn_.datamodule import DataModule
from numerlab.sklearn_.module import Module
from numerlab.sklearn_.trainer import Trainer_
from numerlab.utils.types import Path_

log = logging.getLogger(__name__)

PredictFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def get_predict_fn(
    data: DataModule,
    module: Module,
) -> PredictFn:
    features = data.features_list
    model = module.model

    def inference(X: pd.DataFrame) -> NDArray[np.number]:
        X = X.values
        if isinstance(model, nn.Module):
            model.cpu()
            X = torch.from_numpy(X)
            Y = model(X).numpy()
        else:
            Y = model.predict(X)
        return Y.flatten()

    def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        live_predictions = inference(live_features[features])
        submission = pd.Series(live_predictions, index=live_features.index)
        return submission.to_frame("prediction")

    return predict


def upload_predict_fn(
    predict_fn: PredictFn,
    output_dir: Path_,
    model_name: str,
) -> None:
    predict_path = Path(output_dir) / "predict.pkl"
    p = cloudpickle.dumps(predict_fn)
    with open(predict_path, "wb") as f:
        f.write(p)

    public_id = os.getenv("NUMERAI_PUBLIC_ID")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    napi = NumerAPI(public_id=public_id, secret_key=secret_key)

    model_id = napi.get_models()[model_name]
    docker_image = napi.model_upload_docker_images()["Python 3.12"]
    napi.model_upload(file_path=predict_path, model_id=model_id, docker_image=docker_image)


def deploy(
    data: DataModule,
    module: Module,
    trainer: Trainer_,
    monitor: str,
    output_dir: Path_,
    model_name: str,
    ckpt_path: str | None = None,
    matmul_precision: str | None = "high",
    compile: bool = False,
) -> None:
    train_output: TrainOutput = train(
        data=data,
        module=module,
        trainer=trainer,
        monitor=monitor,
        return_objects=True,
        ckpt_path=ckpt_path,
        matmul_precision=matmul_precision,
        compile=compile,
    )

    data: DataModule = train_output.data
    module: Module = train_output.module

    log.info("Creating predict function")
    predict_fn: PredictFn = get_predict_fn(data, module)

    log.info("Uploading predict function")
    upload_predict_fn(predict_fn, output_dir, model_name)
