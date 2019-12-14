import pytest
import torch

from yureka_go.dataset import move_plane, stone_plane


@pytest.mark.parametrize(
    'hex_bytes,plane',
    (
        (
            # nothing
            hex(int(b'0000000000000000000' * 19, 2))[2:],
            torch.zeros(19, 19),
        ),
        (
            # upper left corner
            hex(int(b'1000000000000000000' + b'0000000000000000000' * 18, 2))[2:],
            torch.tensor([1] + [0] * 360).float().view(19, 19),
        ),
        (
            # upper right corner
            hex(int(b'0000000000000000001' + b'0000000000000000000' * 18, 2))[2:],
            torch.tensor([0] * 18 + [1] + [0] * 342).float().view(19, 19),
        ),
        (
            # bottom left corner
            hex(int(b'0000000000000000000' * 18 + b'1000000000000000000', 2))[2:],
            torch.tensor([0] * 342 + [1] + [0] * 18).float().view(19, 19),
        ),
        (
            # bottom right corner
            hex(int(b'0000000000000000000' * 18 + b'0000000000000000001', 2))[2:],
            torch.tensor([0] * 360 + [1]).float().view(19, 19),
        ),
        (
            # middle
            hex(int(b'0000000000000000000' * 9 + b'0000000001000000000' + b'0000000000000000000' * 9, 2))[2:],
            torch.tensor([0] * 180 + [1] + [0] * 180).float().view(19, 19),
        ),
    )
)
def test_stone_plane(hex_bytes: bytes, plane: torch.Tensor):
    assert stone_plane(hex_bytes).equal(plane)


@pytest.mark.parametrize(
    'turn_bytes,planes',
    (
        (b'0', torch.stack([torch.ones(19, 19), torch.zeros(19, 19)])),
        (b'1', torch.stack([torch.zeros(19, 19), torch.ones(19, 19)])),
    )
)
def test_move_plane(turn_bytes: bytes, planes: torch.Tensor):
    assert move_plane(turn_bytes).equal(planes)
