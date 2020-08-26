import logging
import hydra

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from hydra.utils import instantiate

from leela_zero_pytorch.network import NetworkLightningModule


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Trainer:
    logger.info(f"Training with the following config:\n{cfg.pretty()}")

    network = NetworkLightningModule(cfg.network, cfg.train)
    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger)
    data = instantiate(cfg.data)
    trainer.fit(network, data)
    if cfg.train.run_test:
        trainer.test(datamodule=data)

    return trainer


if __name__ == "__main__":
    main()
