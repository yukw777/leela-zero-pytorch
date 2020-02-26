import torch
import torch.nn.functional as F

from typing import Tuple


class MSECrossEntropyLoss:

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def compute(self, pred: Tuple[torch.Tensor, torch.Tensor],
                target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pred_move, pred_val = pred
        pred_move = pred_move.squeeze()
        pred_val = pred_val.squeeze()
        target_move, target_val = target
        cross_entropy_loss = F.cross_entropy(pred_move, target_move)
        mse_loss = F.mse_loss(pred_val, target_val)
        return self.alpha * mse_loss + cross_entropy_loss
