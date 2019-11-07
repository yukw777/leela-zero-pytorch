import torch
import pytest
import tempfile
import os

from network import Network


@pytest.mark.parametrize(
    "board_size,in_channels,residual_channels,residual_layers,input_size,pol_output_size,val_output_size",
    (
        (19, 9, 64, 3, (36, 9, 19, 19), (36, 19 * 19 + 1), (36, 1)),
        (19, 18, 128, 6, (36, 18, 19, 19), (36, 19 * 19 + 1), (36, 1)),
    )
)
def test_network(board_size, in_channels, residual_channels,
                 residual_layers, input_size, pol_output_size, val_output_size):
    n = Network(board_size, in_channels, residual_channels, residual_layers)
    pol, val = n(torch.randn(*input_size))
    assert pol.size() == pol_output_size
    assert val.size() == val_output_size


@pytest.fixture
def weight_file():
    return open('test-data/weights.txt', 'r')


def test_to_leela_weights(weight_file):
    torch.manual_seed(42)
    n = Network(19, 18, 16, 3)
    _, tmp = tempfile.mkstemp()
    n.to_leela_weights(tmp)

    with open(tmp, 'r') as tmpfile:
        tmplines = tmpfile.readlines()
        weightlines = weight_file.readlines()
        assert len(tmplines) == len(weightlines)
        for l1, l2 in zip(tmplines, weightlines):
            assert l1 == l2

    os.remove(tmp)
