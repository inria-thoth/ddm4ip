import itertools
import math

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR

from ddm4ip.utils import distributed


def create_optimizer(config, *parameters):
    concat_params = itertools.chain(*parameters)
    lr = config['lr']
    betas = config.get('betas', (0.9, 0.99))
    wd = config.get('weight_decay', 0)
    optim = torch.optim.Adam(
        concat_params,
        lr=lr,
        betas=betas,
        weight_decay=wd,
    )
    lr_scheduler = None
    if config.get('lr_scheduler') is not None:
        if config['lr_scheduler']['type'] == "EDMLRScheduler":
            lr_scheduler = EDMLRScheduler(
                optimizer=optim,
                batch_size=config['lr_scheduler']['batch_size'] * distributed.get_world_size(),
                ref_nimg=config['lr_scheduler']['ref_nimg'],
                rampup_nimg=config['lr_scheduler']['rampup_nimg'],
            )
        elif config['lr_scheduler']['type'] == "CosineAnnealingLR":
            lr_scheduler = CosineAnnealingLR(optim, 1e7)
        elif config['lr_scheduler']['type'] == "EDMLRScheduler":
            lr_scheduler = LinearLR(optim)
        elif config['lr_scheduler']['type'] == "MultiStepLR":
            lr_scheduler = MultiStepLR(optim, milestones=[400000, 800000], gamma=100)
    return optim, lr_scheduler


def learning_rate_schedule(cur_step, batch_size, ref_nimg=70e6, rampup_nimg=10e6):
    mul = 1.0
    if ref_nimg > 0:
        # From ref_nimg, decrease towards 0 with sqrt law
        mul /= math.sqrt(max((cur_step * batch_size) / ref_nimg, 1))
    if rampup_nimg > 0:
        # Increasing towards rampup_nimg
        mul *= min(cur_step * batch_size / rampup_nimg, 1)
    return mul


class EDMLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size,
        ref_nimg=70_000_000,
        rampup_nimg=10_000_000,
        last_epoch=-1,
    ):
        self.ref_nimg = ref_nimg
        self.batch_size = batch_size
        self.rampup_nimg = rampup_nimg
        self.num_imgs = 0
        self._prev_last_epoch = None
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        """Compute learning rate."""
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)
        if self._prev_last_epoch is not None:
            num_steps = max(0, self.last_epoch - self._prev_last_epoch)
        else:
            if self.num_imgs == 0:
                # backwards compatible with StepLR we were previously using
                num_steps = self.last_epoch
            else:
                num_steps = 1
        self._prev_last_epoch = self.last_epoch
        self.num_imgs += self.batch_size * num_steps

        mul = 1.0
        if self.ref_nimg > 0:
            # From ref_nimg, decrease towards 0 with sqrt law
            mul /= math.sqrt(max(self.num_imgs / self.ref_nimg, 1))
        if self.rampup_nimg > 0:
            # Increasing towards rampup_nimg
            mul *= min(self.num_imgs / self.rampup_nimg, 1)

        return [base_lr * mul for base_lr in self.base_lrs]


def force_finite_grads(net):
    for param in net.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
