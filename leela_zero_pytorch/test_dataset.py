import pytest
import torch
import random

from typing import List

from leela_zero_pytorch.dataset import (
    turn_plane,
    stone_plane,
    Dataset,
    hex_to_ndarray,
    transform,
    transform_move_prob_plane,
)


@pytest.mark.parametrize(
    "plane,plane_tensor",
    (
        (
            # nothing
            hex_to_ndarray(hex(int("0000000000000000000" * 19, 2))[2:].zfill(91)),
            torch.zeros(19, 19),
        ),
        (
            # upper left corner
            hex_to_ndarray(
                (
                    hex(
                        int(
                            "1000000000000000000"
                            + "0000000000000000000" * 17
                            + "000000000000000000",
                            2,
                        )
                    )
                    + "0"
                )[2:].zfill(91)
            ),
            torch.tensor([1] + [0] * 360).float().view(19, 19),
        ),
        (
            # upper right corner
            hex_to_ndarray(
                (
                    hex(
                        int(
                            "0000000000000000001"
                            + "0000000000000000000" * 17
                            + "000000000000000000",
                            2,
                        )
                    )
                    + "0"
                )[2:].zfill(91)
            ),
            torch.tensor([0] * 18 + [1] + [0] * 342).float().view(19, 19),
        ),
        (
            # bottom left corner
            hex_to_ndarray(
                (hex(int("0000000000000000000" * 18 + "100000000000000000", 2)) + "0")[
                    2:
                ].zfill(91)
            ),
            torch.tensor([0] * 342 + [1] + [0] * 18).float().view(19, 19),
        ),
        (
            # bottom right corner
            hex_to_ndarray(
                (hex(int("0000000000000000000" * 18 + "000000000000000000", 2)) + "1")[
                    2:
                ].zfill(91)
            ),
            torch.tensor([0] * 360 + [1]).float().view(19, 19),
        ),
        (
            # middle
            hex_to_ndarray(
                (
                    hex(
                        int(
                            "0000000000000000000" * 9
                            + "0000000001000000000"
                            + "0000000000000000000" * 8
                            + "000000000000000000",
                            2,
                        )
                    )
                    + "0"
                )[2:].zfill(91)
            ),
            torch.tensor([0] * 180 + [1] + [0] * 180).float().view(19, 19),
        ),
    ),
)
def test_stone_plane(plane: str, plane_tensor: torch.Tensor):
    assert stone_plane(plane).equal(plane_tensor)


@pytest.mark.parametrize(
    "turn,planes",
    (
        (0, [torch.ones(19, 19), torch.zeros(19, 19)]),
        (1, [torch.zeros(19, 19), torch.ones(19, 19)]),
    ),
)
def test_turn_plane(turn: int, planes: List[torch.Tensor]):
    assert all(a.equal(b) for a, b in zip(turn_plane(turn), planes))


@pytest.mark.parametrize("transform", [True, False])
@pytest.mark.parametrize(
    "filenames,length",
    (
        (["test-data/kgs.0.gz"], 6366),
        (["test-data/kgs.1.gz"], 6658),
        (["test-data/kgs.0.gz", "test-data/kgs.1.gz"], 13024),
    ),
)
def test_go_dataset(filenames: List[str], length: int, transform: bool):
    view = Dataset(filenames, transform)
    assert len(view) == length
    random_idx = random.randrange(0, len(view))

    planes, moves, outcome = view[random_idx]
    assert planes.size() == (18, 19, 19)
    assert moves.item() in list(range(19 * 19 + 1))
    assert moves.dtype == torch.int64
    assert outcome.item() in (-1, 1)


@pytest.mark.parametrize(
    "planes,k,hflip,transformed",
    [
        (
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
            0,
            False,
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
            0,
            True,
            torch.tensor([[0, 0, 1], [0, 0, 0], [0, 0, 2],]),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
            1,
            False,
            torch.tensor([[2, 0, 1], [0, 0, 0], [0, 0, 0],]),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
            1,
            True,
            torch.tensor([[1, 0, 2], [0, 0, 0], [0, 0, 0],]),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
            2,
            False,
            torch.tensor([[0, 0, 2], [0, 0, 0], [0, 0, 1],]),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
            2,
            True,
            torch.tensor([[2, 0, 0], [0, 0, 0], [1, 0, 0],]),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
            3,
            False,
            torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 2],]),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 0, 0], [2, 0, 0],]),
            3,
            True,
            torch.tensor([[0, 0, 0], [0, 0, 0], [2, 0, 1],]),
        ),
        (
            torch.tensor(
                [
                    [
                        [[1, 0, 0], [0, 0, 0], [2, 0, 0],],
                        [[1, 0, 0], [0, 0, 0], [2, 0, 0],],
                    ],
                    [
                        [[1, 0, 0], [0, 0, 0], [2, 0, 0],],
                        [[1, 0, 0], [0, 0, 0], [2, 0, 0],],
                    ],
                ]
            ),
            3,
            True,
            torch.tensor(
                [
                    [
                        [[0, 0, 0], [0, 0, 0], [2, 0, 1],],
                        [[0, 0, 0], [0, 0, 0], [2, 0, 1],],
                    ],
                    [
                        [[0, 0, 0], [0, 0, 0], [2, 0, 1],],
                        [[0, 0, 0], [0, 0, 0], [2, 0, 1],],
                    ],
                ]
            ),
        ),
    ],
)
def test_transform(planes, k, hflip, transformed):
    assert transform(planes, k, hflip).equal(transformed)


@pytest.mark.parametrize(
    "plane,k,hflip,transformed",
    [
        (
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
            0,
            False,
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
        ),
        (
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
            0,
            True,
            torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 2, 1]),
        ),
        (
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
            1,
            False,
            torch.tensor([2, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
        ),
        (
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
            1,
            True,
            torch.tensor([1, 0, 2, 0, 0, 0, 0, 0, 0, 1]),
        ),
        (
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
            2,
            False,
            torch.tensor([0, 0, 2, 0, 0, 0, 0, 0, 1, 1]),
        ),
        (
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
            2,
            True,
            torch.tensor([2, 0, 0, 0, 0, 0, 1, 0, 0, 1]),
        ),
        (
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
            3,
            False,
            torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 2, 1]),
        ),
        (
            torch.tensor([1, 0, 0, 0, 0, 0, 2, 0, 0, 1]),
            3,
            True,
            torch.tensor([0, 0, 0, 0, 0, 0, 2, 0, 1, 1]),
        ),
    ],
)
def test_transform_move_prob_plane(plane, k, hflip, transformed):
    assert transform_move_prob_plane(plane, 3, k, hflip).equal(transformed)
