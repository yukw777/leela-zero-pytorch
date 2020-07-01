import sys

from leela_zero_pytorch.train import main as train_main
from leela_zero_pytorch.weights import main as weights_main


def test_train(monkeypatch, tmp_path):
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
        ],
    )
    train_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lzp-weights",
            f"{tmp_path}/lightning_logs/version_0/checkpoints/epoch=0.ckpt",
            f"{tmp_path}/weights.txt",
        ],
    )
    weights_main()
