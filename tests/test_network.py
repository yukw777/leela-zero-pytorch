import torch
import pytest
import tempfile
import os

from leela_zero_pytorch.network import Network


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
