import abc
import contextlib
import dataclasses
from enum import Enum
from typing import Any, Mapping
import torch

from ddm4ip.data.base import Batch
from ddm4ip.trainers.base import BaseTrainer


class AbstractLoss(abc.ABC):
    def __init__(
        self,
        has_val_loss: bool,
        n_accum_steps: int = 1,
    ):
        self.has_val_loss = has_val_loss
        self.n_accum_steps = n_accum_steps
        self.cur_n_accum_steps = 0

    @abc.abstractmethod
    def __call__(self, trainer: 'BaseTrainer', batch: Batch) -> Mapping[str, float | torch.Tensor]:
        pass

    @abc.abstractmethod
    def val_loss(self, trainer: 'BaseTrainer', batch: Batch) -> Mapping[str, float | torch.Tensor]:
        pass

    def is_full_step_complete(self) -> bool:
        return self.cur_n_accum_steps == 0

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @contextlib.contextmanager
    def optimizer_step(self, optimizer, scheduler, module, grad_scaler):
        do_optim_step = (self.cur_n_accum_steps + 1) % self.n_accum_steps == 0
        try:
            if isinstance(module, torch.nn.parallel.DistributedDataParallel) and not do_optim_step:
                with module.no_sync():
                    yield optimizer
            else:
                yield optimizer
        finally:
            if do_optim_step:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                if scheduler is not None:
                    scheduler.step()
                # if hasattr(module, "initial_param"):
                #     print(f"{torch.linalg.norm(module.initial_param.grad.view(64, -1), dim=1)=}")
                optimizer.zero_grad(set_to_none=True)
            self.cur_n_accum_steps = (self.cur_n_accum_steps + 1) % self.n_accum_steps


class OptState(Enum):
    INNER = 1
    OUTER = 2


@dataclasses.dataclass
class OptimizationState:
    current_steps: int
    state: OptState
    loss_dict: dict[str, float] = dataclasses.field(default_factory=dict)

    def is_start_of_cycle(self) -> bool:
        return self.state == OptState.INNER and self.current_steps == 0

    def increment_state(self):
        if self.state == OptState.INNER:
            self.state = OptState.OUTER
        elif self.state == OptState.OUTER:
            self.state = OptState.INNER
        else:
            raise NotImplementedError(self.state)
        self.current_steps = 0