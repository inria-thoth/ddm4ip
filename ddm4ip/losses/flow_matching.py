import abc
from typing import Any, Dict, Mapping

import torch
import torch.amp
from torchvision.transforms import v2

from ddm4ip.data.base import Batch
from ddm4ip.losses.base import AbstractLoss
from ddm4ip.losses.utils import create_optimizer, make_ddp
from ddm4ip.networks.unets import RFNoPrecond
from ddm4ip.utils.torch_utils import center


class AbstractDiffusion(abc.ABC):
    @abc.abstractmethod
    def sample_noisy(self, clean_img: torch.Tensor) -> torch.Tensor:
        pass
    @abc.abstractmethod
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pass
    @abc.abstractmethod
    def make_xt(self, x_clean, x_noise, t) -> torch.Tensor:
        pass
    @abc.abstractmethod
    def diff_loss_weight(self, t) -> torch.Tensor:
        pass
    @abc.abstractmethod
    def diff_loss_target(self, x_clean, x_noisy) -> torch.Tensor:
        pass

    @staticmethod
    def from_config(config: Mapping[str, Any]) -> 'AbstractDiffusion':
        if config['loss']['diffusion']['type'] == "flowmatching":
            return FlowMatchingDiffusion()
        elif config['loss']['diffusion']['type'] == "edm":
            return EDMDiffusion(
                P_mean=config['loss']['diffusion'].get('P_mean', -1.2),
                P_std=config['loss']['diffusion'].get('P_std', 1.2),
                sigma_data=config['loss']['diffusion'].get('sigma_data', 0.5),
            )
        else:
            raise ValueError(f"Diffusion type {config['loss']['diffusion']['type']} not recognized.")


class FlowMatchingDiffusion(AbstractDiffusion):
    def sample_noisy(self, clean_img: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(clean_img)  # x0

    def sample_time(self, batch_size: int, device: torch.device):
        return torch.rand([batch_size, 1, 1, 1], device=device)

    def make_xt(self, x_clean, x_noise, t):
        return (1 - t) * x_noise + t * x_clean

    def diff_loss_weight(self, t):
        return torch.ones((1,), device=t.device, dtype=t.dtype)  # 1.0

    def diff_loss_target(self, x_clean, x_noisy):
        return x_clean - x_noisy


class EDMDiffusion(AbstractDiffusion):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def sample_noisy(self, clean_img: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(clean_img)  # x0

    def sample_time(self, batch_size: int, device: torch.device):
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
        return (rnd_normal * self.P_std + self.P_mean).exp()

    def make_xt(self, x_clean, x_noise, t):
        return x_clean + t * x_noise

    def diff_loss_weight(self, t):
        return (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2

    def diff_loss_target(self, x_clean, x_noisy):
        return x_clean


def flow_matching_loss(
    net: torch.nn.Module,
    diff: AbstractDiffusion,
    images: torch.Tensor,
    labels: torch.Tensor | None = None,
    conditioning: torch.Tensor | None = None,
):
    images = center(images)            # x1
    noisy = diff.sample_noisy(images)  # x0
    t = diff.sample_time(noisy.shape[0], noisy.device)
    xt = diff.make_xt(x_clean=images, x_noise=noisy, t=t)
    target = diff.diff_loss_target(images, noisy)
    weight = diff.diff_loss_weight(t)

    direction = net(xt, t.squeeze(), labels, conditioning=conditioning)
    loss = torch.mean(weight * ((direction - target) ** 2))
    loss_dict = {'flow_matching': loss.detach()}
    return loss, loss_dict


class FlowMatchingLoss(AbstractLoss):
    def __init__(self,
                 config: Mapping[str, Any],
                 flow_nn: torch.nn.Module,
                 clean_images: bool,):
        super().__init__(has_val_loss=True, n_accum_steps=config['loss']['n_accum_steps'])
        self.opt, self.lr_scheduler = create_optimizer(config['optim'], flow_nn.parameters())
        assert isinstance(flow_nn, RFNoPrecond)
        self.flow_nn = flow_nn
        self.flow_nn_ddp = make_ddp(flow_nn)
        self.clean_images = clean_images
        self.diffusion = AbstractDiffusion.from_config(config)
        self.crop_patch_sizes = config['loss']['crop_patch_sizes']
        self.train_transform = None
        if self.crop_patch_sizes is not None and len(self.crop_patch_sizes) > 0:
            # Uniform probabilities
            self.train_transform = v2.RandomChoice([
                v2.RandomCrop(ps, pad_if_needed=False)
                for ps in self.crop_patch_sizes
            ])
        self.grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=self.flow_nn.use_fp16)

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
            loss, loss_dict = flow_matching_loss(self.flow_nn_ddp, self.diffusion, target_img, target_lbl, cond)
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
        loss, loss_dict = flow_matching_loss(self.flow_nn, self.diffusion, target_img, target_lbl, cond)
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
