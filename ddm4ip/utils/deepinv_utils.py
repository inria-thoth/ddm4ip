import numpy as np
import torch
import deepinv
import skimage

from .metrics import calc_psnr
from .torch_utils import center, uncenter


class DinvReconstructorWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        get_output=lambda X: X,
    ):
        super().__init__()
        self.model = model
        self.get_output = get_output
        self.custom_metrics = None
        self.batch_size = None

    def forward(self, y, physics, x_gt=None, compute_metrics=False):
        x_init = physics.A_adjoint(y)
        device = y.device
        metrics = self.init_metrics_fn(x_init=x_init, x_gt=x_gt)
        out = self.model(y, physics, x_init=x_init)
        if isinstance(out, torch.Tensor):
            # Iterates were not saved, `out` is solution
            metrics = self.update_metrics_fn(metrics, x_init, out, x_gt=x_gt)
            final_out = self.get_output(out)
        else:
            for i in range(1, len(out)):
                x_prev = out[i - 1].to(device=device)
                x = out[i].to(device=device)
                metrics = self.update_metrics_fn(metrics, x_prev, x, x_gt=x_gt)
            final_out = self.get_output(out[-1])

        if compute_metrics:
            return final_out, metrics
        return final_out

    def init_metrics_fn(self, x_init, x_gt=None):
        r"""
        Initializes the metrics.

        Metrics are computed for each batch and for each iteration.
        They are represented by a list of list, and ``metrics[metric_name][i,j]`` contains the metric ``metric_name``
        computed for batch i, at iteration j.

        :param dict X_init: dictionary containing the primal and auxiliary initial iterates.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: ``None``.
        :return dict: A dictionary containing the metrics.
        """
        init = {}
        x_init = self.get_output(x_init)
        self.batch_size = x_init.shape[0]
        if x_gt is not None:
            psnr = [
                [calc_psnr(x_init[i], x_gt[i])]
                for i in range(self.batch_size)
            ]
        else:
            psnr = [[] for i in range(self.batch_size)]
        init["psnr"] = psnr
        init["residual"] = [[] for i in range(self.batch_size)]
        if self.custom_metrics is not None:
            for custom_metric_name in self.custom_metrics.keys():
                init[custom_metric_name] = [[] for i in range(self.batch_size)]
        return init

    def update_metrics_fn(self, metrics, x_prev, x, x_gt=None):
        r"""
        Function that compute all the metrics, across all batches, for the current iteration.

        :param dict metrics: dictionary containing the metrics. Each metric is computed for each batch.
        :param dict X_prev: dictionary containing the primal and dual previous iterates.
        :param dict X: dictionary containing the current primal and dual iterates.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: None.
        :return dict: a dictionary containing the updated metrics.
        """
        if metrics is not None:
            x_prev = self.get_output(x_prev)
            x = self.get_output(x)
            for i in range(self.batch_size):
                residual = (
                    ((x_prev[i] - x[i]).norm() / (x[i].norm() + 1e-06))
                    .detach()
                    .cpu()
                    .item()
                )
                metrics["residual"][i].append(residual)
                if x_gt is not None:
                    psnr = calc_psnr(x[i], x_gt[i])
                    metrics["psnr"][i].append(psnr)
                if self.custom_metrics is not None:
                    for custom_metric_name, custom_metric_fn in zip(
                        self.custom_metrics.keys(), self.custom_metrics.values()
                    ):
                        metrics[custom_metric_name][i].append(
                            custom_metric_fn(
                                metrics[custom_metric_name], x_prev[i], x[i]
                            )
                        )
        return metrics


# The GSPnP prior corresponds to a RED prior with an explicit `g`.
# We thus write a class that inherits from RED for this custom prior.
class GSPnP(deepinv.optim.prior.RED):
    """
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)


class DPSWrapper(deepinv.sampling.DPS):
    def forward(
        self,
        y,
        physics: deepinv.physics.Physics,
        seed=None,
        x_init=None,
    ):
        # The `x_init is None` path is never taken
        if x_init is None:
            # Need to uncenter, because the first step in DPS is a centering step
            x_init = uncenter(torch.randn_like(physics.A_adjoint(y)))
        # But for some reason y is not centered so we do it here!
        y = center(y)

        out = super().forward(y, physics, seed, x_init)
        if isinstance(out, list):
            out = [uncenter(o) for o in out]
        else:
            out = uncenter(out)

        return out


class WienerSolver(torch.nn.Module):
    def __init__(self, balance: float):
        super().__init__()
        self.balance = balance

    def forward(self, y, physics, **kwargs):
        batch_size = y.shape[0]
        channels = y.shape[1]
        y_np = y.numpy(force=True)
        x_np = np.empty_like(y_np)
        filters_np = physics.filter.numpy(force=True)

        for i in range(batch_size):
            for ch in range(channels):
                c_filter = filters_np[i, ch]
                c_y = y_np[i, ch]
                x_np[i, ch] = skimage.restoration.wiener(c_y, c_filter, balance=self.balance)
                # x_np[i, ch] = skimage.restoration.unsupervised_wiener(c_y, c_filter)[0]
        x_pt = torch.from_numpy(x_np).to(dtype=y.dtype, device=y.device)
        return x_pt

