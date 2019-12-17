import torch

from flambe.logging import log
from flambe.learn.eval import Evaluator


class YurekaGoEvaluator(Evaluator):

    def run(self, block_name: str = None) -> bool:
        """Run the evaluation.

        Returns
        ------
        bool
            Whether the component should continue running.

        """
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            pol_preds, val_preds, pol_targets, val_targets = [], [], [], []

            for batch in self._eval_iterator:
                (pred_pol, pred_val), (target_pol, target_val) = self.model(*[t.to(self.device) for t in batch])
                pol_preds.append(pred_pol.cpu())
                val_preds.append(pred_val.cpu())
                pol_targets.append(target_pol.cpu())
                val_targets.append(target_val.cpu())

            pol_preds = torch.cat(pol_preds, dim=0)  # type: ignore
            val_preds = torch.cat(val_preds, dim=0)  # type: ignore
            pol_targets = torch.cat(pol_targets, dim=0)  # type: ignore
            val_targets = torch.cat(val_targets, dim=0)  # type: ignore
            self.eval_metric = self.metric_fn((pol_preds, val_preds), (pol_targets, val_targets)).item()

            tb_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""

            log(f'{tb_prefix}Eval {self.metric_fn}',  # type: ignore
                self.eval_metric, global_step=0)  # type: ignore

        return False
