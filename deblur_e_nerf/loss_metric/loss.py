import easydict
import torch
from ..utils import modules


class Loss(torch.nn.Module):
    LOSS_NAMES = [
        "log_intensity_diff",
        "log_intensity_tv"
    ]

    def __init__(self, loss_weight, loss_error_fn, loss_normalize):
        super().__init__()

        # assert loss names & weights
        assert set(self.LOSS_NAMES) <= set(loss_weight.keys())
        for loss_weight_value in loss_weight.values():
            assert isinstance(loss_weight_value, (int, float)) \
                   and loss_weight_value >= 0
        assert sum(loss_weight.values()) > 0

        # save some hyperparameters as attributes
        self.loss_weight = loss_weight
        self.error_fn = easydict.EasyDict()
        for key in self.LOSS_NAMES:
            self.error_fn[key] = {
                "l1": torch.nn.L1Loss(reduction="none"),
                "mse": torch.nn.MSELoss(reduction="none"),
                "huber": torch.nn.HuberLoss(reduction="none", delta=1.0),
                "mape": modules.MAPELoss(reduction="none")
            }[loss_error_fn[key]]
        self.normalize = loss_normalize

    def compute(
        self,
        batch_event,
        batch_diff=None,
        batch_subdiff=None,
        mean_contrast_threshold=None
    ):
        """
        NOTE:
            Care must be taken to handle the computation of losses with
            zero-shaped inputs.
        """
        batch_mean_loss = easydict.EasyDict({})
        batch_event.log_intensity_grad = (                                      # (batch.size)
            batch_event.log_intensity_diff
            / (batch_event.end_ts - batch_event.start_ts)
        )

        if self.loss_weight.log_intensity_diff > 0:
            batch_mean_loss.log_intensity_diff = self.log_intensity_diff(
                batch_event, batch_diff, mean_contrast_threshold
            )
        if self.loss_weight.log_intensity_tv > 0:
            batch_mean_loss.log_intensity_tv = self.log_intensity_tv(
                batch_subdiff, mean_contrast_threshold
            )
        return batch_mean_loss

    def log_intensity_diff(
        self,
        batch_event,
        batch_diff,
        mean_contrast_threshold
    ):
        if self.normalize.log_intensity_diff:
            normalizing_const = mean_contrast_threshold
        else:
            normalizing_const = 1
        log_intensity_diff_err = self.error_fn.log_intensity_diff(              # (batch.size)
            input=batch_diff.log_intensity_diff / normalizing_const,
            target=(batch_diff.ts_diff * batch_event.log_intensity_grad
                    / normalizing_const).to(
                        batch_diff.log_intensity_diff.dtype
                   )
        )
        mean_log_intensity_diff_err = (                                         # ()
            log_intensity_diff_err[batch_diff.is_valid].mean()
        )
        return mean_log_intensity_diff_err
    
    def log_intensity_tv(self, batch_subdiff, mean_contrast_threshold):
        if self.normalize.log_intensity_tv:
            normalizing_const = mean_contrast_threshold
        else:
            normalizing_const = 1
        log_intensity_tv_err = self.error_fn.log_intensity_tv(                  # (batch.size)
            input=batch_subdiff.log_intensity_diff / normalizing_const,
            target=torch.zeros_like(batch_subdiff.log_intensity_diff)
        )
        mean_log_intensity_tv_err = (                                           # ()
            log_intensity_tv_err[batch_subdiff.is_valid].mean()
        )
        return mean_log_intensity_tv_err
