import sys

from leela_zero_pytorch.train import main


def test_train(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["lzp-train", "network=tiny", f"hydra.run.dir={tmp_path}"],
    )
    main()
