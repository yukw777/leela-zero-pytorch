import torch

from typing import Tuple, List

from flambe.dataset import Dataset


# (input, move probs, game outcome)
DataPoint = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def stone_plane(hex_bytes: bytes) -> torch.Tensor:
    n = int(hex_bytes, 16)
    return torch.tensor([n >> i & 1 for i in range(361 - 1, -1, -1)]).float().view(19, 19)


def move_plane(turn_bytes: bytes) -> torch.Tensor:
    # 0 = black, 1 = white
    # 17) All 1 if black is to move, 0 otherwise
    # 18) All 1 if white is to move, 0 otherwise
    ones = torch.ones(19, 19)
    zeros = torch.zeros(19, 19)
    if turn_bytes == b'0':
        # black's turn to move
        return torch.stack([ones, zeros])
    return torch.stack([zeros, ones])


class GoDataset(Dataset):

    def __init__(self, train_files: List[str], val_files: List[str], test_files: List[str]):
        pass

    @property
    def train(self) -> List[DataPoint]:
        return self._train
