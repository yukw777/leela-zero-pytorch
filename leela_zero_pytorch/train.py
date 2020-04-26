import hydra
import logging

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from leela_zero_pytorch.network import NetworkLightningModule
from leela_zero_pytorch.dataset import Dataset


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml", strict=False)
def main(cfg: DictConfig):
    logger.info(f"Training with the following config:\n{cfg.pretty()}")
    module = NetworkLightningModule(cfg)
    trainer = Trainer(**cfg.pl_trainer)
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
                batch_size=cfg.datset.train.batch_size,
                num_workers=cfg.dataset.test.num_workers,
            )
        )


# this function is required to allow automatic detection of the module name when running
# from a binary script. it should be called from the executable script and not the
# hydra.main() function directly.
def entry():
    main()


if __name__ == "__main__":
    main()
