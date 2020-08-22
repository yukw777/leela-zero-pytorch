import logging
import hydra

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from leela_zero_pytorch.network import NetworkLightningModule
from leela_zero_pytorch.dataset import Dataset


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Trainer:
    logger.info(f"Training with the following config:\n{cfg.pretty()}")

    module = NetworkLightningModule(cfg.network, cfg.train)
    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger)
    trainer.fit(
        module,
        train_dataloader=DataLoader(
            Dataset.from_data_dir(
                hydra.utils.to_absolute_path(cfg.dataset.train.dir_path), transform=True
            ),
            shuffle=True,
            batch_size=cfg.dataset.train.batch_size,
            num_workers=cfg.dataset.train.num_workers,
        ),
        val_dataloaders=DataLoader(
            Dataset.from_data_dir(
                hydra.utils.to_absolute_path(cfg.dataset.val.dir_path)
            ),
            batch_size=cfg.dataset.val.batch_size,
            num_workers=cfg.dataset.val.num_workers,
        ),
    )
    if cfg.train.run_test:
        trainer.test(
            test_dataloaders=DataLoader(
                Dataset.from_data_dir(
                    hydra.utils.to_absolute_path(cfg.dataset.test.dir_path)
                ),
                batch_size=cfg.dataset.train.batch_size,
                num_workers=cfg.dataset.test.num_workers,
            )
        )

    return trainer


if __name__ == "__main__":
    main()
