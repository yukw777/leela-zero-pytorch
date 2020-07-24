import logging
import glob
import os
import torch
import numpy as np
import gzip
import random
import bisect

from typing import Tuple, List

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
    # Instead of treating the bit array as one hexadecimal (by prepending it with 0s to
    # make the length divisible by 4), it just appends the last bit at the end as
    # '0' or '1'. So we first need to parse the hex string without the last
    # digit, then append a bit to the parsed bit array at the end.
    # More details in the code below:
    # https://github.com/leela-zero/leela-zero/blob/b259e50d5cce34a12176846534f369ef5ffcebc1/src/Training.cpp#L260-L264

    # turn the first bytes into a bit array
    bit_array = np.unpackbits(np.array(bytearray.fromhex(hex[:-1])))

    # append the last bit and pack it
    return np.packbits(np.append(bit_array, int(hex[-1])))


def transform(planes: torch.Tensor, k: int, hflip: bool) -> torch.Tensor:
    """
    Rotate the planes 90 degrees `k` times and flip them horizontally if `hflip`
    is True.
    """
    dim = planes.dim()
    transformed = planes.rot90(k, (dim - 1, dim - 2))
    if hflip:
        return transformed.flip(dim - 1)
    return transformed


def transform_move_prob_plane(
    plane: torch.Tensor, board_size: int, k: int, hflip: bool
) -> torch.Tensor:
    """
    Transform the move prob plane. The last bit is for passing, so transform everything
    before that.
    """
    # extract the board
    board, pass_move = plane[:-1], plane[-1]

    # transform the board
    # we need to use reshape as the tensor may not be contiguous in memory at this point
    transformed = transform(board.view(board_size, board_size), k, hflip).reshape(-1, 1)

    # append the pass move and return the flat tensor
    return torch.cat((transformed, pass_move.view(1, 1))).flatten()


class Dataset:
    def __init__(self, filenames: List[str], transform: bool):
        self.filenames = filenames
        self.transform = transform
        self._build_item_positions()

    def _build_item_positions(self):
        """
        Calculate the file breakpoints (how many items per file) and
        the file position for each item within a file.
        """
        self.breakpoints: List[int] = []
        self.file_positions: List[List[int]] = []
        count = 0
        for fname in self.filenames:
            with gzip.open(fname, "rt") as fileh:
                positions = []
                while True:
                    position = fileh.tell()
                    for _ in range(19):
                        line = fileh.readline()
                    if len(line) == 0:
                        break
                    positions.append(position)
                    count += 1
            self.file_positions.append(positions)
            self.breakpoints.append(count)

    def __getitem__(self, idx: int) -> DataPoint:
        # find the right file for the idx
        file_idx = bisect.bisect(self.breakpoints, idx)
        with gzip.open(self.filenames[file_idx], "rt") as fileh:
            # find the position within the file
            in_file_idx = idx if file_idx == 0 else idx - self.breakpoints[file_idx - 1]

            # prepare the input planes
            input_planes: List[torch.Tensor] = []
            fileh.seek(self.file_positions[file_idx][in_file_idx])
            for i in range(19):
                line = fileh.readline().strip()
                if i < 16:
                    # prepare the stone planes
                    input_planes.append(stone_plane(hex_to_ndarray(line)))
                elif i == 16:
                    # prepare the turn planes
                    input_planes.extend(turn_plane(int(line)))
                elif i == 17:
                    # move probabilities
                    move_probs = torch.from_numpy(
                        np.array([int(p) for p in line.split()], dtype=np.uint8)
                    )
                else:
                    # i == 18
                    # outcome
                    outcome = torch.tensor(int(line)).float()

        # stack all the planes
        stacked_input = torch.stack(input_planes)

        if self.transform:
            # transform for data augmentation

            # parameters for random transformation
            rotations = random.randrange(4)
            hflip = bool(random.getrandbits(1))

            stacked_input = transform(stacked_input, rotations, hflip)
            move_probs = transform_move_prob_plane(move_probs, 19, rotations, hflip)
        return (
            stacked_input,
            move_probs.argmax(),
            outcome,
        )

    def __len__(self):
        return self.breakpoints[-1] if len(self.breakpoints) > 0 else 0

    @classmethod
    def from_data_dir(cls, path: str, transform: bool = False):
        return cls(glob.glob(os.path.join(path, "*.gz")), transform)
