from typing import Any, Mapping
import warnings

import torch
import torch.nn as nn

from ddm4ip.data.base import Batch, Datasplit
from ddm4ip.losses.base import AbstractLoss, OptState, OptimizationState
from ddm4ip.losses.flow_matching import FlowMatchingLoss
from ddm4ip.losses.utils import create_optimizer
from ddm4ip.trainers.base import BaseTrainer
from ddm4ip.utils import distributed
from ddm4ip.utils.torch_utils import center, equate_kernel_shapes


def gaussian_reg(kernel):
    if kernel.dim() == 5:
        # kernel.shape: B, H*W, kc, kh, kw
        kernel = kernel.reshape(-1, *kernel.shape[2:])
    B, C, H, W = kernel.shape

    # 🔹 Compute center of gravity (x_c, y_c)
    x_coords = torch.linspace(-1, 1, W, device=kernel.device).view(1, 1, 1, W)
    y_coords = torch.linspace(-1, 1, H, device=kernel.device).view(1, 1, H, 1)

    x_c = (kernel * x_coords).sum(dim=(-2, -1), keepdim=True)  # [-1, 1]
    y_c = (kernel * y_coords).sum(dim=(-2, -1), keepdim=True)  # [-1, 1]

    # 🔹 Compute spread loss (variance around center of gravity, normalized)
    spread_loss = ((x_coords - x_c) ** 2 + (y_coords - y_c) ** 2) * kernel
    return spread_loss.mean()


def sparse_reg(kernel: torch.Tensor):
    return torch.abs(kernel).mean()


def com(kernel):
    xx, yy = torch.meshgrid(
        torch.arange(kernel.shape[-2], device=kernel.device),
        torch.arange(kernel.shape[-1], device=kernel.device),
        indexing='ij'
    )
    xx = xx[None, None, ...]
    yy = yy[None, None, ...]
    ksum = kernel.sum((-1, -2, -3)) + 1e-5
    center_of_mass = (kernel * xx).sum((-1, -2, -3)) / ksum, (kernel * yy).sum((-1, -2, -3)) / ksum
    return center_of_mass


def center_reg(kernel: torch.Tensor):
    kernel_com = com(kernel)
    com_reg = (kernel.shape[-2] // 2 - kernel_com[-2]) ** 2 + (kernel.shape[-1] // 2 - kernel_com[-1]) ** 2
    return com_reg.mean()


def sum_to_one_reg(kernel_pre_norm: torch.Tensor):
    return torch.abs(kernel_pre_norm.sum(dim=(-1, -2)) - 1).mean()


class DiffInstructOnY(AbstractLoss):
    def __init__(
        self,
        config: Mapping[str, Any],
        aux_flow_nn: nn.Module,
        prtr_flow_nn: nn.Module,
        kernel_nn: nn.Module
    ):
        super().__init__(has_val_loss=True, n_accum_steps=config['loss']['n_accum_steps'])

        self.opt_state = OptimizationState(0, OptState.INNER)
        self.kernel_opt, self.lr_scheduler = create_optimizer(
            config['optim']['kernel'], kernel_nn.parameters()
        )
        gs_enabled = getattr(kernel_nn, "use_fp16", False)
        self.grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=gs_enabled)

        # Note that clean_images does not imply true clean images. The images used for
        # the auxiliary loss are the ones generated with `kernel_nn`
        config['loss']['aux']['n_accum_steps'] = self.n_accum_steps  # Forced to be equal!
        aux_cfg = dict(loss=config['loss']['aux'], optim=config['optim']['aux'])
        self.aux_flow_loss = FlowMatchingLoss(
            config=aux_cfg, flow_nn=aux_flow_nn, clean_images=True, pretr_flow_nn=prtr_flow_nn
        )
        self.aux_flow_nn = aux_flow_nn
        self.kernel_nn = kernel_nn
        self.explicit_kernel = hasattr(kernel_nn, 'get_kernel')
        if distributed.get_world_size() > 1:
            self.kernel_nn_ddp = torch.nn.parallel.DistributedDataParallel(
                kernel_nn,
                device_ids=[distributed.get_local_rank()],
                find_unused_parameters=False,
                broadcast_buffers=False
            )
        else:
            self.kernel_nn_ddp = kernel_nn
        self.prtr_flow_nn = prtr_flow_nn.requires_grad_(False)

        # Regularizers
        self.boundary_mask = None
        self.sparse_reg = config['loss'].get('sparse_reg', None)
        self.symmetry_reg = config['loss'].get('symmetry_reg', None)
        self.center_reg = config['loss'].get('center_reg', None)
        self.sum_to_one_reg = config['loss'].get('sum_to_one_reg', None)
        self.gaussian_reg = config['loss'].get('gaussian_reg', None)

        self.conditioning_acc = []

    def loss_di(self, clean_img, clean_label, noise_level, conditioning, **kernel_kwargs):
        dev = clean_img.device

        # in DiY input to kernel-nn is the clean image,
        # output is the noisy image. Here noise_level is used explicitly.
        if self.explicit_kernel:
            y1, kernel = self.kernel_nn_ddp(
                clean_img,
                noise_level,
                clean_label,
                conditioning,
                get_kernel=True,
                **kernel_kwargs
            )
        else:
            y1 = self.kernel_nn_ddp(clean_img, noise_level, clean_label, conditioning, **kernel_kwargs)
            kernel = None
        corrupt_shape = y1.shape
        y0 = torch.randn(corrupt_shape, device=dev)
        y1 = center(y1)
        t = torch.rand([corrupt_shape[0], 1, 1, 1], device=dev)
        yt = (1 - t) * y0 + t * y1

        with torch.no_grad():
            cuda_rng_state = torch.cuda.get_rng_state()
            prtr_velocity = self.prtr_flow_nn(yt, t, clean_label, conditioning)
            torch.cuda.set_rng_state(cuda_rng_state)
            aux_velocity = self.aux_flow_nn(yt, t, clean_label, conditioning)
            flow_diff = aux_velocity - prtr_velocity

        loss_di = (flow_diff * y1).mean()
        loss_info = {"loss/di": loss_di.detach()}

        if kernel is not None:  # apply regularization on the predicted kernels/filters
            total_reg, reg_info = self.kernel_regularization(kernel)
            loss_info.update(reg_info)
            loss_di = loss_di + total_reg

        return loss_di, loss_info

    def kernel_regularization(self, kernel: torch.Tensor):
        loss_info = {}
        loss_sparse = 0
        if self.sparse_reg is not None and self.sparse_reg > 0:
            loss_sparse = self.sparse_reg * sparse_reg(kernel)
            loss_info["loss/sparse"] = loss_sparse.detach()
        loss_center = 0
        if self.center_reg is not None and self.center_reg > 0:
            loss_center = self.center_reg * center_reg(kernel)
            loss_info["loss/center"] = loss_center.detach()
        loss_sum_to_one = 0
        if self.sum_to_one_reg is not None and self.sum_to_one_reg > 0:
            loss_sum_to_one = self.sum_to_one_reg * sum_to_one_reg(kernel)
            loss_info["loss/sum_to_one"] = loss_sum_to_one.detach()
        loss_gaussian = 0
        if self.gaussian_reg is not None and self.gaussian_reg > 0:
            loss_gaussian = self.gaussian_reg * gaussian_reg(kernel)
            loss_info['loss/gaussian'] = loss_gaussian.detach()

        total_reg = loss_sparse + loss_center + loss_sum_to_one + loss_gaussian
        return total_reg, loss_info

    def __call__(self, trainer: BaseTrainer, batch: Batch):
        batch_dev = batch.cuda()
        assert batch_dev.clean is not None
        assert batch_dev.noise_level is not None
        self.kernel_nn_ddp.train()

        if self.opt_state.state == OptState.OUTER:  # Outer loss (learn the kernel)
            with self.optimizer_step(self.kernel_opt, self.lr_scheduler, self.kernel_nn_ddp, grad_scaler=self.grad_scaler):
                loss_di, loss_info = self.loss_di(
                    batch_dev.clean,
                    batch_dev.clean_label,
                    batch_dev.noise_level,
                    batch_dev.clean_conditioning,
                    **batch_dev.meta
                )
                loss_di = loss_di / self.n_accum_steps
                self.grad_scaler.scale(loss_di).backward()

                if hasattr(self, "conditioning_acc") and batch_dev.clean_conditioning is not None:
                    if len(self.conditioning_acc) > 5_000:
                        self.conditioning_acc.pop(0)
                    self.conditioning_acc.append(batch_dev.clean_conditioning.detach().cpu().mean(dim=(-1, -2)))

            loss_info["learning_rate/kernel_nn"] = self.kernel_opt.param_groups[0]['lr']
            if super().is_full_step_complete():
                self.opt_state.increment_state()
        else:   # Inner loss (learn the auxiliary score network)
            with torch.no_grad():
                self.kernel_nn.eval()
                y1 = self.kernel_nn(
                    batch_dev.clean,
                    noise_level=batch_dev.noise_level,
                    class_labels=batch_dev.clean_label,
                    conditioning=batch_dev.clean_conditioning,
                    **batch_dev.meta
                )
                self.kernel_nn.train()
            ker_batch = Batch(
                clean=y1,
                clean_label=batch_dev.clean_label,
                clean_conditioning=batch_dev.clean_conditioning,
                corrupt=None,
                corrupt_label=None,
                corrupt_conditioning=None,
                noise_level=None,
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
        pred_corrupt = self.kernel_nn(
            dev_batch.clean,
            dev_batch.noise_level,
            dev_batch.clean_label,
            dev_batch.clean_conditioning,
            **dev_batch.meta
        )
        if dev_batch.corrupt is not None and dev_batch.corrupt.shape == pred_corrupt.shape:
            vl["val_loss/y-mae"] = torch.mean(torch.abs(dev_batch.corrupt - pred_corrupt))
        else:
            warnings.warn("Cannot compute validation loss on y.")

        try:
            if self.explicit_kernel:
                if dev_batch.kernel is not None:
                    true_kernel = dev_batch.kernel
                elif Datasplit.TRAIN in trainer.dsets and hasattr(trainer.dsets[Datasplit.TRAIN].corruption, 'get_kernel'):
                    true_kernel: torch.Tensor = trainer.dsets[Datasplit.TRAIN].corruption.get_kernel(
                        conditioning=batch.clean_conditioning
                    )
                else:
                    raise ValueError("Cannot extract true kernel...")

                pred_kernel = self.kernel_nn.get_kernel(
                    img=dev_batch.clean,
                    conditioning=dev_batch.clean_conditioning
                )
                pred_kernel, true_kernel = equate_kernel_shapes(pred_kernel, true_kernel)
                vl["val_loss/kernel-mae"] = torch.mean(torch.abs(true_kernel - pred_kernel.to(true_kernel.device)))
        except ValueError as e:
            warnings.warn(f"Cannot compute validation loss on kernels. Reason: {e}")

        return vl
