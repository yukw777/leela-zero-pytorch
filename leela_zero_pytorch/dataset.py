import logging
import glob
import os
import torch
import numpy as np
import gzip
import random

from typing import Tuple, List
from itertools import cycle
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

# (planes, move probs, game outcome)
DataPoint = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def stone_plane(plane: np.ndarray) -> torch.Tensor:
    # throw away the padding added by the last bit
    bits = np.unpackbits(plane)[:-7]
    return torch.tensor(bits).float().view(19, 19)


def turn_plane(turn: int) -> List[torch.Tensor]:
    # 0 = black, 1 = white
    # 17) All 1 if it's black's turn, 0 otherwise
    # 18) All 1 if it's white's turn, 0 otherwise
    ones = torch.ones(19, 19)
    zeros = torch.zeros(19, 19)
    if turn == 0:
        # black's turn to move
        return [ones, zeros]
    return [zeros, ones]


def hex_to_ndarray(hex: str) -> np.ndarray:
    # For a board with an odd number of rows and columns,
    # there are a total of (2*k+1)^2 = 4*k^2 + 4*k + 1 positions or "bits"
    # If we write this out as a hexadecimal, i.e. group by 4 bits (2^4 = 16),
    # there will always be 1 bit left, i.e. (4*k^2 + 4*k + 1) % 4 = 1.
    # LeelaZero handles this in a unique way:
    # Instead of treating the bit array as one hexadecimal (by prepending it with 0s to make the length divisible by 4),
    # it just appends the last bit at the end as '0' or '1'. So we first need to parse the hex string without the last
    # digit, then append a bit to the parsed bit array at the end.
    # More details in the code below:
    # https://github.com/leela-zero/leela-zero/blob/b259e50d5cce34a12176846534f369ef5ffcebc1/src/Training.cpp#L260-L264

    # turn the first bytes into a bit array
    bit_array = np.unpackbits(np.array(bytearray.fromhex(hex[:-1])))

    # append the last bit and pack it
    return np.packbits(np.append(bit_array, int(hex[-1])))


def get_data_from_file(fname: str) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    stone_planes: List[np.ndarray] = []
    turn_planes: List[int] = []
    move_probs: List[np.ndarray] = []
    outcomes: List[int] = []
    with gzip.open(fname, 'rt') as f:
        for i in cycle(range(19)):
            try:
                line = next(f).strip()
            except StopIteration:
                break
            if i < 16:
                stone_planes.append(hex_to_ndarray(line))
            elif i == 16:
                turn_planes.append(int(line))
            elif i == 17:
                move_probs.append(np.array([int(p) for p in line.split()], dtype=np.uint8))
            else:
                # i == 18
                outcomes.append(int(line))
    return stone_planes, turn_planes, move_probs, outcomes


def transform(planes: torch.Tensor, k: int, hflip: bool) -> torch.Tensor:
    """
    Rotate the planes 90 degrees `k` times and flip them horizontally if `hflip` is True.
    """
    dim = planes.dim()
    transformed = planes.rot90(k, (dim - 1, dim - 2))
    if hflip:
        return transformed.flip(dim - 1)
    return transformed


def transform_move_prob_plane(plane: torch.Tensor, board_size: int, k: int, hflip: bool) -> torch.Tensor:
    """
    Transform the move prob plane. The last bit is for passing, so transform everything before that.
    """
    # extract the board
    board, pass_move = plane[:-1], plane[-1]

    # transform the board
    # we need to use reshape as the tensor may not be contiguous in memory at this point
    transformed = transform(board.view(board_size, board_size), k, hflip).reshape(-1, 1)

    # append the pass move and return the flat tensor
    return torch.cat((transformed, pass_move.view(1, 1))).flatten()


class Dataset:

    def __init__(self, filenames: List[str]):
        stone_planes: List[np.ndarray] = []
        turn_planes: List[int] = []
        move_probs: List[np.ndarray] = []
        outcomes: List[int] = []
        self.raw_datapoints: List[List[str]] = []
        with ProcessPoolExecutor() as executor:
            for data in executor.map(get_data_from_file, filenames):
                f_stone_planes, f_turn_planes, f_move_probs, f_outcomes = data
                stone_planes.extend(f_stone_planes)
                turn_planes.extend(f_turn_planes)
                move_probs.extend(f_move_probs)
                outcomes.extend(f_outcomes)
        self.stone_planes = np.stack(stone_planes) if len(stone_planes) > 0 else np.empty((0, 19, 19))
        self.turn_planes = np.array(turn_planes)
        self.move_probs = np.stack(move_probs) if len(move_probs) > 0 else np.empty((0, 19*19+1))
        self.outcomes = np.array(outcomes)

    def __getitem__(self, idx: int) -> DataPoint:
        # parameters for random transformation
        rotations = random.randrange(4)
        hflip = bool(random.getrandbits(1))

        # prepare the stone planes with transformation
        input_planes: List[torch.Tensor] = []
        for plane in self.stone_planes[idx * 16: (idx + 1) * 16]:
            input_planes.append(transform(stone_plane(plane), rotations, hflip))

        # prepare the turn planes
        # no need for transformation as they're just all 1s and 0s
        input_planes.extend(turn_plane(self.turn_planes[idx]))

        # transform the move prob plane
        move_probs = transform_move_prob_plane(torch.from_numpy(self.move_probs[idx]), 19, rotations, hflip)
        return (
            torch.stack(input_planes),
            move_probs.argmax(),
            torch.tensor(self.outcomes[idx]).float(),
        )

    def __len__(self):
        return len(self.outcomes)

    @classmethod
    def from_data_dir(cls, path: str):
        return cls(glob.glob(os.path.join(path, '*.gz')))
