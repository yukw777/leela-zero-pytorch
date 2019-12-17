import torch

from flambe.learn import Trainer


class YurekaGoTrainer(Trainer):

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
