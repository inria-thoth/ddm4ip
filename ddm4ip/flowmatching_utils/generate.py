import abc
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from ddm4ip.utils import distributed
from ddm4ip.utils.torch_utils import uncenter


class HeunSample(abc.ABC):
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def sample(self, noise, labels=None, dtype=torch.float32, conditioning=None, return_trajectory=False):
        device = noise.device
        traj: List[torch.Tensor] = []
        t_steps = self.get_time_steps(dtype, device)
        x_next = noise.to(dtype) * self.get_initial_std(t_steps)
        if return_trajectory:
            traj.append(uncenter(x_next))
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            # Euler step
            d_cur = self.get_direction(x_cur, t_cur, labels, conditioning)
            x_next = x_cur + (t_next - t_cur) * d_cur
            # 2nd order correction
            if i < self.num_steps - 1:
                d_prime = self.get_direction(x_next, t_next, labels, conditioning)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

            if return_trajectory:
                traj.append(uncenter(x_next))
        if return_trajectory:
            return traj[-1], traj
        return uncenter(x_next)

    @abc.abstractmethod
    def get_time_steps(self, dtype, device) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_initial_std(self, t_steps) -> float:
        pass

    @abc.abstractmethod
    def get_direction(self, x, t, labels, conditioning) -> torch.Tensor:
        pass


class FlowSample(HeunSample):
    def __init__(self, net, num_steps):
        super().__init__(num_steps)
        self.net = net

    def get_time_steps(self, dtype, device):
        return torch.linspace(0, 1, steps=self.num_steps + 1, dtype=dtype, device=device)

    def get_initial_std(self, t_steps) -> float:
        return 1.0

    def get_direction(self, x, t, labels, conditioning) -> torch.Tensor:
        return self.net(x, t, labels, conditioning).to(x.dtype)


@dataclass(frozen=False)
class ImageGenerationOutput:
    seeds: List[int]
    num_batches: int
    batch_idx: int
    solution: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    trajectory: List[torch.Tensor] | None = None


def imggen(
    net,
    img_size: tuple[int, int, int],
    label_dim: int = 0,
    seeds: Sequence[int] = range(16, 24),
    class_idx=None,
    batch_size=32,
    device=torch.device('cuda'),
    return_trajectory=False,
    dtype=torch.float32,
    conditioning=None,
    **sampler_kwargs
):
    sampler = FlowSample(net=net, **sampler_kwargs)

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (batch_size * distributed.get_world_size()) + 1, 1) * distributed.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[distributed.get_local_rank()::distributed.get_world_size()]

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                batch_seeds = [seeds[idx] for idx in indices]
                batch_conditioning = None
                if conditioning is not None:
                    batch_conditioning = torch.stack([conditioning[idx] for idx in indices], 0)
                ig_out = ImageGenerationOutput(
                    seeds=batch_seeds,
                    num_batches=len(rank_batches),
                    batch_idx=batch_idx
                )
                if len(batch_seeds) > 0:
                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, batch_seeds)
                    noise = rnd.randn([len(batch_seeds), *img_size], device=device)
                    labels = None
                    if label_dim > 0:
                        labels = torch.eye(label_dim, device=device)[rnd.randint(label_dim, size=[len(batch_seeds)], device=device)]
                        if class_idx is not None:
                            labels[:, :] = 0
                            labels[:, class_idx] = 1
                        ig_out.labels = labels

                    # Generate images.
                    images = sampler.sample(
                        noise=noise,
                        labels=labels,
                        dtype=dtype,
                        conditioning=batch_conditioning,
                        return_trajectory=return_trajectory
                    )
                    if return_trajectory:
                        assert isinstance(images, tuple)
                        ig_out.solution = images[0]
                        ig_out.trajectory = images[1]
                    else:
                        assert isinstance(images, torch.Tensor)
                        ig_out.solution = images
                distributed.barrier()  # keep the ranks in sync
                yield ig_out

    return ImageIterable()

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
