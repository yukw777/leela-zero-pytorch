import hydra

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from leela_zero_pytorch.network import Network
from leela_zero_pytorch.dataset import Dataset


@hydra.main(config_path='../conf/config.yaml')
def train(cfg):
    print(cfg.pretty())
    network = Network(
        cfg.network.board_size,
        cfg.network.in_channels,
        cfg.network.residual_channels,
        cfg.network.residual_layers,
    )
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        gpus=cfg.train.gpus,
    )
    trainer.fit(
        network,
        train_dataloader=DataLoader(
            Dataset.from_data_dir(hydra.utils.to_absolute_path(cfg.dataset.train_dir)),
            shuffle=True,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.n_data_workers,
        ),
        val_dataloaders=DataLoader(
            Dataset.from_data_dir(hydra.utils.to_absolute_path(cfg.dataset.val_dir)),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.n_data_workers,
        ),
        test_dataloaders=DataLoader(
            Dataset.from_data_dir(hydra.utils.to_absolute_path(cfg.dataset.test_dir)),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.n_data_workers,
        ),
    )


if __name__ == "__main__":
    train()
