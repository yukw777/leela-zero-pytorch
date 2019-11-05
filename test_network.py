import torch
import pytest

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
