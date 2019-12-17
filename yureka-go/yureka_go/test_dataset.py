import pytest
import torch

from typing import List

from yureka_go.dataset import move_plane, stone_plane, parse_file, get_datapoints


@pytest.mark.parametrize(
    'plane,plane_tensor',
    (
        (
            # nothing
            hex(int('0000000000000000000' * 19, 2))[2:].zfill(91),
            torch.zeros(19, 19),
        ),
        (
            # upper left corner
            hex(int('1000000000000000000' + '0000000000000000000' * 18, 2))[2:].zfill(91),
            torch.tensor([1] + [0] * 360).float().view(19, 19),
        ),
        (
            # upper right corner
            hex(int('0000000000000000001' + '0000000000000000000' * 18, 2))[2:].zfill(91),
            torch.tensor([0] * 18 + [1] + [0] * 342).float().view(19, 19),
        ),
        (
            # bottom left corner
            hex(int('0000000000000000000' * 18 + '1000000000000000000', 2))[2:].zfill(91),
            torch.tensor([0] * 342 + [1] + [0] * 18).float().view(19, 19),
        ),
        (
            # bottom right corner
            hex(int('0000000000000000000' * 18 + '0000000000000000001', 2))[2:].zfill(91),
            torch.tensor([0] * 360 + [1]).float().view(19, 19),
        ),
        (
            # middle
            hex(int('0000000000000000000' * 9 + '0000000001000000000' + '0000000000000000000' * 9, 2))[2:].zfill(91),
            torch.tensor([0] * 180 + [1] + [0] * 180).float().view(19, 19),
        ),
    )
)
def test_stone_plane(plane: str, plane_tensor: torch.Tensor):
    assert stone_plane(plane).equal(plane_tensor)


@pytest.mark.parametrize(
    'turn,planes',
    (
        ('0', [torch.ones(19, 19), torch.zeros(19, 19)]),
        ('1', [torch.zeros(19, 19), torch.ones(19, 19)]),
    )
)
def test_move_plane(turn: str, planes: List[torch.Tensor]):
    assert all(a.equal(b) for a, b in zip(move_plane(turn), planes))


def test_parse_file():
    for input_tensor, move_probs, outcome in parse_file('test-data/kgs.0.gz'):
        assert input_tensor.size() == (18, 19, 19)
        assert move_probs.size() == (19 * 19 + 1,)
        assert outcome.item() in (1, -1)


def test_get_datapoints():
    assert len(get_datapoints(['test-data/kgs.0.gz'])) == 6366
