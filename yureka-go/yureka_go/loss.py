import torch
import torch.nn.functional as F

from flambe.metric import Metric


class MSECrossEntropyLoss(Metric):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def compute(self, pred_move: torch.Tensor, pred_val: torch.Tensor,
                target_move: torch.Tensor, target_val: torch.Tensor) -> torch.Tensor:
        cross_entropy_loss = F.cross_entropy(pred_move, target_move)
        mse_loss = F.mse_loss(pred_val, target_val)
        return self.alpha * mse_loss + cross_entropy_loss
