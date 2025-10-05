from hydra_zen import make_config, store

from numerlab.configs.utils.utils import remove_types

DebugDefaultCfg = make_config(
    task_name="debug",
    hydra=dict(
        job_logging=dict(
            root=dict(
                level="DEBUG",
            ),
        ),
    ),
    trainer=dict(
        logger=None,
        callbacks=None,
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        detect_anomaly=True,
    ),
    data=dict(
        num_workers=0,
        pin_memory=False,
    ),
)

DebugFDRCfg = make_config(
    hydra_defaults=[
        "default",
        "_self_",
    ],
    trainer=dict(
        fast_dev_run=True,
    ),
)

DebugLimitCfg = make_config(
    hydra_defaults=[
        "default",
        "_self_",
    ],
    trainer=dict(
        max_epochs=3,
        limit_train_batches=0.01,
        limit_val_batches=0.05,
        limit_test_batches=0.05,
    ),
)

DebugOverfitCfg = make_config(
    hydra_defaults=[
        "default",
        "_self_",
    ],
    trainer=dict(
        max_epochs=20,
        overfit_batches=1,
    ),
)

DebugProfilerCfg = make_config(
    hydra_defaults=[
        "default",
        "_self_",
    ],
    trainer=dict(
        max_epochs=1,
        profiler="simple",
    ),
)

debug_store = store(group="debug", package="_global_", to_config=remove_types)
debug_store(DebugDefaultCfg, name="default")
debug_store(DebugFDRCfg, name="fdr")
debug_store(DebugLimitCfg, name="limit")
debug_store(DebugOverfitCfg, name="overfit")
debug_store(DebugProfilerCfg, name="profiler")
