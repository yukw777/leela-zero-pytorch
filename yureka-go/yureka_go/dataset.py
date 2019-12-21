import logging
import glob
import os
import torch
import numpy as np
import gzip

from typing import Tuple, List
from itertools import islice
from concurrent.futures import ThreadPoolExecutor

from flambe.dataset import Dataset
from flambe.compile import registrable_factory

logger = logging.getLogger(__name__)

# (planes, move probs, game outcome)
DataPoint = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def stone_plane(plane: str) -> torch.Tensor:
    bits = np.unpackbits(np.array(bytearray.fromhex('0' + plane)))[7:]
    return torch.tensor(bits).float().view(19, 19)


def move_plane(turn: str) -> List[torch.Tensor]:
    # 0 = black, 1 = white
    # 17) All 1 if black is to move, 0 otherwise
    # 18) All 1 if white is to move, 0 otherwise
    ones = torch.ones(19, 19)
    zeros = torch.zeros(19, 19)
    if turn == '0':
        # black's turn to move
        return [ones, zeros]
    return [zeros, ones]


def parse(lines: List[str]) -> DataPoint:
    assert len(lines) == 19
    input_planes: List[torch.Tensor] = []
    for i, line in enumerate(lines):
        if i < 16:
            input_planes.append(stone_plane(line.strip()))
        elif i == 16:
            input_planes.extend(move_plane(line.strip()))
        elif i == 17:
            move_probs = torch.argmax(torch.tensor([float(p) for p in line.split()]))
        else:
            outcome = torch.tensor(float(line))
            # i == 18
    return torch.stack(input_planes), move_probs, outcome


def get_raw_datapoints(filename: str) -> np.ndarray:
    raw_datapoints = []
    with gzip.open(filename, 'rt') as f:
        while True:
            # read in 19 lines at a time and append
            lines = list(islice(f, 19))
            if len(lines) != 19:
                break
            raw_datapoints.append(lines)
    return np.array(raw_datapoints)


class GoDataView():

    def __init__(self, filenames: List[str]):
        with ThreadPoolExecutor() as executor:
            raw_datapoints = [datapoints for datapoints in executor.map(get_raw_datapoints, filenames)]
        self.raw_datapoints = np.concatenate(raw_datapoints, axis=0)

    def __getitem__(self, idx: int) -> DataPoint:
        return parse(self.raw_datapoints[idx])

    def __len__(self):
        return len(self.raw_datapoints)


class GoDataset(Dataset):

    def __init__(self, train_files: List[str], val_files: List[str], test_files: List[str]):
        self._train = GoDataView(train_files)
        self._val = GoDataView(val_files)
        self._test = GoDataView(test_files)

    @property
    def train(self) -> GoDataView:
        return self._train

    @property
    def val(self) -> GoDataView:
        return self._val

    @property
    def test(self) -> GoDataView:
        return self._test

    @registrable_factory
    @classmethod
    def from_data_dir(cls, train_dir: str, val_dir: str, test_dir: str) -> 'GoDataset':
        train_files = glob.glob(os.path.join(train_dir, '*.gz'))
        val_files = glob.glob(os.path.join(val_dir, '*.gz'))
        test_files = glob.glob(os.path.join(test_dir, '*.gz'))
        return cls(train_files, val_files, test_files)
