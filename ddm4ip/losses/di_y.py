from typing import Any, Mapping
import warnings

import torch
import torch.nn as nn

from ddm4ip.data.base import Batch, Datasplit
from ddm4ip.losses.base import AbstractLoss, OptState, OptimizationState
from ddm4ip.losses.flow_matching import FlowMatchingLoss
from ddm4ip.losses.kernel_regularizers import kernel_regularization
from ddm4ip.losses.utils import create_optimizer, make_ddp
from ddm4ip.networks.blur_nets import AbstractDegradationNetwork
from ddm4ip.trainers.base import BaseTrainer
from ddm4ip.utils.torch_utils import center, equate_kernel_shapes


class DiffInstructOnY(AbstractLoss):
    def __init__(
        self,
        config: Mapping[str, Any],
        aux_flow_nn: nn.Module,
        prtr_flow_nn: nn.Module,
        kernel_nn: AbstractDegradationNetwork,
    ):
        super().__init__(has_val_loss=True, n_accum_steps=config['loss']['n_accum_steps'])

        self.opt_state = OptimizationState(0, OptState.INNER)
        self.kernel_opt, self.lr_scheduler = create_optimizer(
            config['optim']['kernel'], kernel_nn.parameters()
        )
        gs_enabled = getattr(kernel_nn, "use_fp16", False)
        self.grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=gs_enabled)

        # Note that `clean_images` does not imply true clean images. The images used for
        # the auxiliary loss are the ones generated with `kernel_nn`
        config['loss']['aux']['n_accum_steps'] = self.n_accum_steps  # Forced to be equal!
        aux_cfg = dict(loss=config['loss']['aux'], optim=config['optim']['aux'])
        self.aux_flow_loss = FlowMatchingLoss(
            config=aux_cfg, flow_nn=aux_flow_nn, clean_images=True
        )
        self.aux_flow_nn = aux_flow_nn
        self.kernel_nn = kernel_nn
        self.kernel_nn_ddp = make_ddp(kernel_nn, broadcast_buffers=False)
        self.prtr_flow_nn = prtr_flow_nn.requires_grad_(False)

        # Regularizers
        self.sparse_reg = config['loss'].get('sparse_reg', None)
        self.symmetry_reg = config['loss'].get('symmetry_reg', None)
        self.center_reg = config['loss'].get('center_reg', None)
        self.sum_to_one_reg = config['loss'].get('sum_to_one_reg', None)
        self.gaussian_reg = config['loss'].get('gaussian_reg', None)

    def loss_di(self, clean_img, noise_level, conditioning, **kernel_kwargs):
        diff = self.aux_flow_loss.diffusion
        y1, physics = self.kernel_nn_ddp(
            clean_img,
            return_physics=True,
            sigma=noise_level,
            conditioning=conditioning,
            **kernel_kwargs
        )
        y1 = center(y1)
        y0 = diff.sample_noisy(y1)
        t = diff.sample_time(y0.shape[0], y0.device)
        yt = diff.make_xt(x_clean=y1, x_noise=y0, t=t)

        with torch.no_grad():
            cuda_rng_state = torch.cuda.get_rng_state()
            prtr_velocity = self.prtr_flow_nn(yt, t, conditioning=conditioning)
            torch.cuda.set_rng_state(cuda_rng_state)
            aux_velocity = self.aux_flow_nn(yt, t, conditioning=conditioning)
            flow_diff = aux_velocity - prtr_velocity

        loss_di = (flow_diff * y1).mean()
        loss_info = {"loss/di": loss_di.detach()}

        if hasattr(physics, "filter") and physics.filter is not None:  # apply regularization on the predicted kernels/filters
            total_reg, reg_info = kernel_regularization(
                physics.filter,
                sparse=self.sparse_reg, center=self.center_reg,
                sum_to_one=self.sum_to_one_reg, gaussian=self.gaussian_reg
            )
            loss_info.update(reg_info)
            loss_di = loss_di + total_reg

        return loss_di, loss_info

    def __call__(self, trainer: BaseTrainer, batch: Batch):
        batch_dev = batch.cuda()
        assert batch_dev.clean is not None
        self.kernel_nn_ddp.train()

        if self.opt_state.state == OptState.OUTER:  # Outer loss (learn the kernel)
            with self.optimizer_step(self.kernel_opt, self.lr_scheduler, self.kernel_nn_ddp, grad_scaler=self.grad_scaler):
                loss_di, loss_info = self.loss_di(
                    batch_dev.clean,
                    batch_dev.noise_level,
                    batch_dev.clean_conditioning,
                    **batch_dev.meta
                )
                loss_di = loss_di / self.n_accum_steps
                self.grad_scaler.scale(loss_di).backward()

            loss_info["learning_rate/kernel_nn"] = self.kernel_opt.param_groups[0]['lr']
            if super().is_full_step_complete():
                self.opt_state.increment_state()
        else:   # Inner loss (learn the auxiliary score network)
            with torch.no_grad():
                self.kernel_nn.eval()
                y1 = self.kernel_nn(
                    batch_dev.clean,
                    return_physics=False,
                    sigma=batch_dev.noise_level,
                    conditioning=batch_dev.clean_conditioning,
                    **batch_dev.meta
                )
            ker_batch = Batch(
                clean=y1,
                clean_conditioning=batch_dev.clean_conditioning,
                clean_label=None, corrupt=None, corrupt_label=None,
                corrupt_conditioning=None, noise_level=None,
            )
            aux_loss_info = self.aux_flow_loss(trainer, ker_batch)
            loss_info = {
                k.replace("loss/", "loss/aux_"): v for k, v in aux_loss_info.items()
            }
            if self.aux_flow_loss.is_full_step_complete():
                self.opt_state.increment_state()

        return loss_info

    def is_full_step_complete(self) -> bool:
        # full_step: both inner and outer
        # Always called after `__call__`
        return super().is_full_step_complete() and self.opt_state.state == OptState.INNER

    def state_dict(self):
        return {
            "aux": self.aux_flow_loss.state_dict(),
            "kernel_opt": self.kernel_opt.state_dict(),
            "kernel_lr": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "gs": self.grad_scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.aux_flow_loss.load_state_dict(state_dict["aux"])
        self.kernel_opt.load_state_dict(state_dict["kernel_opt"])
        if self.lr_scheduler is not None:
            if "kernel_lr" not in state_dict:
                raise ValueError(
                    "Created loss with lr scheduler but state-dict has no lr scheduler state."
                )
            self.lr_scheduler.load_state_dict(state_dict["kernel_lr"])
        self.grad_scaler.load_state_dict(state_dict["gs"])

    @torch.no_grad()
    def val_loss(self, trainer: BaseTrainer, batch: Batch) -> Mapping[str, float | torch.Tensor]:
        vl = {}
        # Here we assume batch contains paired data
        dev_batch = batch.cuda()
        pred_corrupt, physics = self.kernel_nn(
            dev_batch.clean,
            return_physics=True,
            sigma=dev_batch.noise_level,
            conditioning=dev_batch.clean_conditioning,
            **dev_batch.meta
        )
        if dev_batch.corrupt is not None and dev_batch.corrupt.shape == pred_corrupt.shape:
            vl["val_loss/y-mae"] = torch.mean(torch.abs(dev_batch.corrupt - pred_corrupt))
        else:
            warnings.warn("Cannot compute validation loss on y.")

        try:
            if hasattr(physics, "filter") and physics.filter is not None:
                if dev_batch.kernel is not None:
                    true_kernel = dev_batch.kernel
                elif Datasplit.TRAIN in trainer.dsets and hasattr(trainer.dsets[Datasplit.TRAIN].corruption, 'get_kernel'):
                    true_kernel: torch.Tensor = trainer.dsets[Datasplit.TRAIN].corruption.get_kernel(
                        conditioning=batch.clean_conditioning
                    )
                else:
                    raise ValueError("Cannot extract true kernel...")

                pred_kernel = physics.filter
                pred_kernel, true_kernel = equate_kernel_shapes(pred_kernel, true_kernel)
                vl["val_loss/kernel-mae"] = torch.mean(torch.abs(true_kernel - pred_kernel.to(true_kernel.device)))
        except ValueError as e:
            warnings.warn(f"Cannot compute validation loss on kernels. Reason: {e}")

        return vl
