import logging
import argparse

from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from hydra.experimental import compose, initialize

from leela_zero_pytorch.network import NetworkLightningModule
from leela_zero_pytorch.dataset import Dataset


logger = logging.getLogger(__name__)


def main(config_file: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    parsed = Path(config_file)
    initialize(config_dir=str(parsed.parent), strict=False)
    cfg = compose(parsed.name, overrides=args.overrides)
    logger.info(f"Training with the following config:\n{cfg.pretty()}")

    # we want to pass in dictionaries as OmegaConf doesn't play nicely with
    # loggers and doesn't allow non-native types
    module = NetworkLightningModule(OmegaConf.to_container(cfg, resolve=True))
    trainer = Trainer(**OmegaConf.to_container(cfg.pl_trainer, resolve=True))
    trainer.fit(
        module,
        train_dataloader=DataLoader(
            Dataset.from_data_dir(cfg.dataset.train.dir_path, transform=True),
            shuffle=True,
            batch_size=cfg.dataset.train.batch_size,
            num_workers=cfg.dataset.train.num_workers,
        ),
        val_dataloaders=DataLoader(
            Dataset.from_data_dir(cfg.dataset.val.dir_path),
            batch_size=cfg.dataset.val.batch_size,
            num_workers=cfg.dataset.val.num_workers,
        ),
    )
    if cfg.train.run_test:
        trainer.test(
            test_dataloaders=DataLoader(
                Dataset.from_data_dir(cfg.dataset.test.dir_path),
                batch_size=cfg.datset.train.batch_size,
                num_workers=cfg.dataset.test.num_workers,
            )
        )


if __name__ == "__main__":
    main("conf/config.yaml")
