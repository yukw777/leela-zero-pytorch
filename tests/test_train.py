import pytest
import sys

from hydra.experimental import initialize, compose
from omegaconf import open_dict
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from leela_zero_pytorch.train import main as train_main
from leela_zero_pytorch.weights import main as weights_main


@pytest.mark.parametrize(
    "logger", [[], ["logger=null"]],
)
def test_train(monkeypatch, tmp_path, capsys, logger):
    with initialize(config_path="../leela_zero_pytorch/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "+network=small",
                "dataset.train.dir_path=tests/test-data",
                "dataset.train.batch_size=2",
                "dataset.val.dir_path=tests/test-data",
                "dataset.val.batch_size=2",
                "dataset.test.dir_path=tests/test-data",
                "dataset.test.batch_size=2",
                f"+pl_trainer.default_root_dir={tmp_path}",
                "+pl_trainer.fast_dev_run=true",
            ]
            + logger,
        )
        if len(logger) == 0:
            # default wandb logger
            with open_dict(cfg):
                cfg.logger.offline = True
                cfg.logger.save_dir = str(tmp_path)
        with capsys.disabled():
            # both pytest and wandb capture stdout, and they cause
            # a deadlock, so don't capture when running the trainer
            trainer = train_main(cfg)

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
