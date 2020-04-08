import pytest
import torch
import random

from typing import List

from leela_zero_pytorch.dataset import move_plane, stone_plane, Dataset, hex_to_ndarray


@pytest.mark.parametrize(
    'plane,plane_tensor',
    (
        (
            # nothing
            hex_to_ndarray(hex(int('0000000000000000000' * 19, 2))[2:].zfill(91)),
            torch.zeros(19, 19),
        ),
        (
            # upper left corner
            hex_to_ndarray(
                (hex(int('1000000000000000000' +
                         '0000000000000000000' * 17 +
                         '000000000000000000', 2)) + '0')[2:].zfill(91)),
            torch.tensor([1] + [0] * 360).float().view(19, 19),
        ),
        (
            # upper right corner
            hex_to_ndarray(
                (hex(int('0000000000000000001' +
                         '0000000000000000000' * 17 +
                         '000000000000000000', 2)) + '0')[2:].zfill(91)),
            torch.tensor([0] * 18 + [1] + [0] * 342).float().view(19, 19),
        ),
        (
            # bottom left corner
            hex_to_ndarray(
                (hex(int('0000000000000000000' * 18 +
                         '100000000000000000', 2)) + '0')[2:].zfill(91)),
            torch.tensor([0] * 342 + [1] + [0] * 18).float().view(19, 19),
        ),
        (
            # bottom right corner
            hex_to_ndarray(
                (hex(int('0000000000000000000' * 18 +
                         '000000000000000000', 2)) + '1')[2:].zfill(91)),
            torch.tensor([0] * 360 + [1]).float().view(19, 19),
        ),
        (
            # middle
            hex_to_ndarray(
                (hex(int('0000000000000000000' * 9 +
                         '0000000001000000000' +
                         '0000000000000000000' * 8 +
                         '000000000000000000', 2)) + '0')[2:].zfill(91)),
            torch.tensor([0] * 180 + [1] + [0] * 180).float().view(19, 19),
        ),
    )
)
def test_stone_plane(plane: str, plane_tensor: torch.Tensor):
    assert stone_plane(plane).equal(plane_tensor)


@pytest.mark.parametrize(
    'turn,planes',
    (
        (0, [torch.ones(19, 19), torch.zeros(19, 19)]),
        (1, [torch.zeros(19, 19), torch.ones(19, 19)]),
    )
)
def test_move_plane(turn: int, planes: List[torch.Tensor]):
    assert all(a.equal(b) for a, b in zip(move_plane(turn), planes))


@pytest.mark.parametrize(
    'filenames,length',
    (
        (['test-data/kgs.0.gz'], 6366),
        (['test-data/kgs.1.gz'], 6658),
        (['test-data/kgs.0.gz', 'test-data/kgs.1.gz'], 13024),
    )
)
def test_go_data_view(filenames: List[str], length: int):
    view = Dataset(filenames)
    assert len(view) == length
    random_idx = random.randrange(0, len(view))

    planes, probs, outcome = view[random_idx]
    assert planes.size() == (18, 19, 19)
    assert probs.item() in list(range(19 * 19 + 1))
    assert probs.dtype == torch.int64
    assert outcome.item() in (-1, 1)
