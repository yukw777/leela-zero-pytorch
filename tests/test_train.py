import pytest
import sys

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from leela_zero_pytorch.train import main as train_main
from leela_zero_pytorch.weights import main as weights_main


@pytest.mark.parametrize(
    "logger", [[], ["logger=wandb", "logger.params.offline=true"]],
)
def test_train(monkeypatch, tmp_path, capsys, clear_hydra, logger):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lzp-train",
            "network=small",
            "dataset.train.dir_path=tests/test-data",
            "dataset.train.batch_size=2",
            "dataset.val.dir_path=tests/test-data",
            "dataset.val.batch_size=2",
            "dataset.test.dir_path=tests/test-data",
            "dataset.test.batch_size=2",
            f"pl_trainer.default_root_dir={tmp_path}",
            "pl_trainer.fast_dev_run=true",
        ]
        + logger
        + ([f"logger.params.save_dir={tmp_path}"] if len(logger) > 0 else []),
    )
    with capsys.disabled():
        # both pytest and wandb capture stdout, and they cause
        # a deadlock, so don't capture when running the trainer
        trainer = train_main()

    checkpoint_path = "checkpoints/epoch=0.ckpt"
    if isinstance(trainer.logger, TensorBoardLogger):
        checkpoint_path = f"{trainer.logger.log_dir}/{checkpoint_path}"
    elif isinstance(trainer.logger, WandbLogger):
        checkpoint_path = (
            f"{tmp_path}/{trainer.logger.name}/"
            f"{trainer.logger.version}/{checkpoint_path}"
        )

    monkeypatch.setattr(
        sys, "argv", ["lzp-weights", checkpoint_path, f"{tmp_path}/weights.txt"],
    )
    weights_main()
