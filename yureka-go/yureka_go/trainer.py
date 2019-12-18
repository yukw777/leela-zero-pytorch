import torch
import torch.nn as nn
import logging

from typing import Tuple

from flambe.learn import Trainer


logger = logging.getLogger(__name__)


class YurekaGoTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_model = None
        if torch.cuda.device_count() > 1:
            logger.info(f'{torch.cuda.device_count()} GPUs, using DataParallel')
            self.parallel_model = nn.DataParallel(self.model)

    def _compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        model = self.parallel_model if self.parallel_model is not None else self.model
        pred, target = model(*batch)
        loss = self.loss_fn(pred, target)
        return loss

    def _aggregate_preds(self, data_iterator):
        pol_preds, val_preds, pol_targets, val_targets = [], [], [], []
        for batch in data_iterator:
            batch = self._batch_to_device(batch)
            (pred_pol, pred_val), (target_pol, target_val) = self.model(*batch)
            pol_preds.append(pred_pol.cpu())
            val_preds.append(pred_val.cpu())
            pol_targets.append(target_pol.cpu())
            val_targets.append(target_val.cpu())

        pol_preds = torch.cat(pol_preds, dim=0)
        val_preds = torch.cat(val_preds, dim=0)
        pol_targets = torch.cat(pol_targets, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        return (pol_preds, val_preds), (pol_targets, val_targets)
