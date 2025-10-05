import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra_zen import launch, store, zen
from hydra_zen._launch import JobReturn, OverrideDict
from pytest import FixtureRequest
from rootutils import find_root

from numerlab.configs import TrainCfg
from numerlab.funcs.train import TrainOutput, train
from tests.conftest import param_namer


@pytest.fixture(params=["cpu", "cuda"], ids=param_namer("accelerator"))
def accelerator(request: FixtureRequest) -> str:
    device: str = request.param
    if device != "cpu" and os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping GPU tests on GitHub Actions")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return device


@pytest.fixture(params=[None, "high"], ids=param_namer("matmul_precision"))
def matmul_precision(request: FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=[False, True], ids=param_namer("validate"))
def validate(request: FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[False, True], ids=param_namer("return_objects"))
def return_objects(request: FixtureRequest) -> bool:
    return request.param


@pytest.fixture()
def overrides(tmp_path: Path) -> OverrideDict:
    overrides = {
        "data.batch_size": 2,
        "data.num_workers": 0,
        "data.features_dtype": "Float32",
        "hydra.run.dir": str(tmp_path),
        "trainer.accelerator": "cpu",
        "trainer.callbacks": None,
        "trainer.devices": 1,
        "trainer.limit_test_batches": 1,
        "trainer.limit_train_batches": 1,
        "trainer.limit_val_batches": 1,
        "trainer.logger": None,
        "trainer.max_epochs": 1,
        "matmul_precision": None,
    }
    return overrides


class TestTrain:
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_train(
        self,
        overrides: OverrideDict,
        accelerator: str,
        matmul_precision: str | None,
        validate: bool,
        return_objects: bool,
    ) -> None:
        overrides.update(
            {
                "trainer.accelerator": accelerator,
                "matmul_precision": matmul_precision,
                "trainer.limit_val_batches": 1 if validate else 0.0,
                "return_objects": return_objects,
            }
        )
        store.add_to_hydra_store(overwrite_ok=True)
        job: JobReturn = launch(TrainCfg, zen(train), version_base="1.3", overrides=overrides)
        assert isinstance(job.return_value, TrainOutput)

    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_main(self, overrides: OverrideDict) -> None:
        def value_to_str(value: Any) -> str:
            if value is None:
                return "null"
            if isinstance(value, bool):
                return str(value).lower()
            return str(value)

        args = [f"{key}={value_to_str(value)}" for key, value in overrides.items()]
        train_script = find_root() / "src" / "numerlab" / "train.py"
        subprocess.run([sys.executable, str(train_script)] + args, check=True)
