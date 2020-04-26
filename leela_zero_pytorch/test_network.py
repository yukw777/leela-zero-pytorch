import torch
import pytest
import tempfile
import os

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from leela_zero_pytorch.network import Network, NetworkLightningModule
from leela_zero_pytorch.dataset import Dataset


@pytest.mark.parametrize(
    "board_size,in_channels,residual_channels,residual_layers,input_size,"
    "pol_output_size,val_output_size",
    (
        (19, 9, 64, 3, (36, 9, 19, 19), (36, 19 * 19 + 1), (36, 1)),
        (19, 18, 128, 6, (36, 18, 19, 19), (36, 19 * 19 + 1), (36, 1)),
    ),
)
def test_network(
    board_size,
    in_channels,
    residual_channels,
    residual_layers,
    input_size,
    pol_output_size,
    val_output_size,
):
    n = Network(board_size, in_channels, residual_channels, residual_layers)
    (pol, val), (target_pol, target_val) = n(
        torch.randn(*input_size),
        torch.randint(19 * 19 + 1, (input_size[0],)),
        torch.ones((input_size[0],)),
    )
    assert pol.size() == pol_output_size
    assert val.size() == val_output_size
    assert target_pol.size() == (input_size[0],)
    assert target_val.size() == (input_size[0],)


@pytest.fixture
def weight_file():
    return open("test-data/weights.txt", "r")


def test_to_leela_weights(weight_file):
    n = Network(19, 18, 16, 3)
    _, tmp = tempfile.mkstemp()
    n.to_leela_weights(tmp)

    with open(tmp, "r") as tmpfile:
        tmplines = tmpfile.readlines()
        weightlines = weight_file.readlines()
        assert len(tmplines) == len(weightlines)
        for l1, l2 in zip(tmplines, weightlines):
            t1 = torch.tensor([float(n) for n in l1.split()])
            t2 = torch.tensor([float(n) for n in l2.split()])
            assert t1.size() == t2.size()

    os.remove(tmp)


def test_train(tmp_path):
    module = NetworkLightningModule(
        {
            "board_size": 19,
            "in_channels": 18,
            "residual_channels": 1,
            "residual_layers": 1,
            "learning_rate": 0.05,
        }
    )
    trainer = Trainer(fast_dev_run=True, default_save_path=tmp_path)
    train_dataset = Dataset.from_data_dir("test-data", transform=True)
    dataset = Dataset.from_data_dir("test-data")
    trainer.fit(
        module,
        train_dataloader=DataLoader(train_dataset, batch_size=2, shuffle=True),
        val_dataloaders=DataLoader(dataset, batch_size=2),
        test_dataloaders=DataLoader(dataset, batch_size=2),
    )
