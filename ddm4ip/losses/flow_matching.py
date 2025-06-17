from typing import Any, Dict, Mapping

import torch
import torch.amp
from torchvision.transforms import v2

from ddm4ip.data.base import Batch
from ddm4ip.losses.base import AbstractLoss
from ddm4ip.losses.utils import create_optimizer
from ddm4ip.networks.unets import RFNoPrecond
from ddm4ip.utils import distributed
from ddm4ip.utils.torch_utils import center


def flow_matching_loss(
    net: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor | None = None,
    conditioning: torch.Tensor | None = None,
    pretrained_net: torch.nn.Module | None = None,
    pretrained_net_factor: float = 1.0,
):
    images = center(images)           # x1
    noisy = torch.randn_like(images)  # x0
    t = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
    xt = (1 - t) * noisy + t * images
    target_direction = images - noisy

    direction = net(xt, t.squeeze(), labels, conditioning=conditioning)
    loss = torch.mean((direction - target_direction) ** 2)
    loss_dict = {'flow_matching': loss.detach()}
    if pretrained_net is not None:
        # regularizer to follow what pretrained_net says
        with torch.no_grad():
            pretr_direction = pretrained_net(xt, t.squeeze(), labels, conditioning=conditioning)
        pretr_loss = pretrained_net_factor * torch.mean((pretr_direction - direction) ** 2)
        loss = loss + pretr_loss
        loss_dict['pretr_matching'] = pretr_loss.detach()
    return loss, loss_dict


class FlowMatchingLoss(AbstractLoss):
    def __init__(self,
                 config: Mapping[str, Any],
                 flow_nn: torch.nn.Module,
                 clean_images: bool,
                 pretr_flow_nn: torch.nn.Module | None = None):
        super().__init__(has_val_loss=True, n_accum_steps=config['loss']['n_accum_steps'])
        self.opt, self.lr_scheduler = create_optimizer(config['optim'], flow_nn.parameters())
        assert isinstance(flow_nn, RFNoPrecond)
        self.flow_nn = flow_nn
        if distributed.get_world_size() > 1:
            self.flow_nn_ddp = torch.nn.parallel.DistributedDataParallel(
                flow_nn,
                device_ids=[distributed.get_local_rank()],
                find_unused_parameters=False
            )
        else:
            self.flow_nn_ddp = self.flow_nn
        self.clean_images = clean_images
        loss_cfg = config['loss']
        self.crop_patch_sizes = loss_cfg['crop_patch_sizes']
        self.train_transform = None
        if self.crop_patch_sizes is not None and len(self.crop_patch_sizes) > 0:
            # Uniform probabilities
            self.train_transform = v2.RandomChoice([
                v2.RandomCrop(ps, pad_if_needed=False)
                for ps in self.crop_patch_sizes
            ])
        self.grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=self.flow_nn.use_fp16)

        self.pretr_matching_reg = loss_cfg.get('pretr_matching', None)
        self.pretr_flow_nn = None
        if self.pretr_matching_reg is not None and self.pretr_matching_reg > 0:
            assert pretr_flow_nn is not None
            self.pretr_flow_nn = pretr_flow_nn

    def get_data_from_batch(self, batch: Batch):
        if self.clean_images:
            out_img, out_lbl, cond = batch.clean, batch.clean_label, batch.clean_conditioning
        else:
            out_img, out_lbl, cond = batch.corrupt, batch.corrupt_label, batch.corrupt_conditioning
        assert out_img is not None
        return out_img, out_lbl, cond

    def __call__(self, trainer, batch: Batch) -> Dict[str, torch.Tensor]:
        dev_batch = batch.cuda()
        self.flow_nn_ddp.train()

        target_img, target_lbl, cond = self.get_data_from_batch(dev_batch)
        if self.train_transform is not None:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state()
            target_img = self.train_transform(target_img)
            if cond is not None:
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(cuda_rng_state)
                cond = self.train_transform(cond)

        with self.optimizer_step(self.opt, self.lr_scheduler, self.flow_nn_ddp, grad_scaler=self.grad_scaler):
            loss, loss_dict = flow_matching_loss(self.flow_nn_ddp, target_img, target_lbl, cond, self.pretr_flow_nn, self.pretr_matching_reg)
            loss = loss / self.n_accum_steps
            self.grad_scaler.scale(loss).backward()
        return {
            "learning_rate/flow_nn": self.opt.param_groups[0]['lr'],
        } | {f"loss/{k}": v for k, v in loss_dict.items()}

    @torch.no_grad()
    def val_loss(self, trainer, batch: Batch):
        dev_batch = batch.cuda()
        self.flow_nn.eval()
        target_img, target_lbl, cond = self.get_data_from_batch(dev_batch)
        loss, loss_dict = flow_matching_loss(self.flow_nn, target_img, target_lbl, cond, self.pretr_flow_nn, self.pretr_matching_reg)
        return {
            f"val_loss/{k}": v for k, v in loss_dict.items()
        }

    def state_dict(self):
        return {
            "opt": self.opt.state_dict(),
            "lr": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "gs": self.grad_scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict["opt"])
        if self.lr_scheduler is not None:
            if "lr" not in state_dict:
                raise ValueError(
                    "Created loss with lr scheduler but state-dict has no lr scheduler state."
                )
            self.lr_scheduler.load_state_dict(state_dict["lr"])
        self.grad_scaler.load_state_dict(state_dict['gs'])
