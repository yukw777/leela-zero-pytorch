import pytest
import torch

from yureka_go.dataset import move_plane


def test_stone_plane():
    pass


@pytest.mark.parametrize(
    "turn_bytes,planes",
    (
        (b'0', torch.stack([torch.ones(19, 19), torch.zeros(19, 19)])),
        (b'1', torch.stack([torch.zeros(19, 19), torch.ones(19, 19)])),
    )
)
def test_move_plane(turn_bytes: bytes, planes: torch.Tensor):
    assert move_plane(turn_bytes).equal(planes)
