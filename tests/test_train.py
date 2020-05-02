import sys

from leela_zero_pytorch.train import main as train_main
from leela_zero_pytorch.weights import main as weights_main


def test_train(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["lzp-train", "network=tiny", f"pl_trainer.default_root_dir={tmp_path}"],
    )
    train_main("../tests/conf/config.yaml")

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
