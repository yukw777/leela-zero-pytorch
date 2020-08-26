import logging
import hydra

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Trainer:
    logger.info(f"Training with the following config:\n{cfg.pretty()}")

    network = instantiate(cfg.network, cfg.train)
    data = instantiate(cfg.data)
    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger)
    trainer.fit(network, data)
    if cfg.train.run_test:
        trainer.test(datamodule=data)

    return trainer


if __name__ == "__main__":
    main()
