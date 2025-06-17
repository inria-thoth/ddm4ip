from functools import cache
import math
from typing import Tuple
import torch
import torchvision
import torchvision.transforms.v2.functional as TF


def cart2pol(x: torch.Tensor, y: torch.Tensor, width, height) -> Tuple[torch.Tensor, torch.Tensor]:
    # polar coordinates wrt to the center of image (so do translation here)
    x = x - (width - 1) / 2
    y = y - (height - 1) / 2
    r = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.arctan2(-y, x)
    return r, theta


def rotate_patch(p: torch.Tensor, angle: float) -> torch.Tensor:
    return TF.rotate(
        p, angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )


def norm_sum_to_one(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    return x / (x.sum(dim=(-1, -2), keepdim=True) + eps)


def get_psf_at_pos(
    psfs: torch.Tensor,
    grid: torch.Tensor,
    pos: torch.Tensor,
    norm_output: bool = True,
) -> torch.Tensor:
    """
    psfs: H*W, kc, kh, kw
    grid: H*W, 2
    pos: B, 2
    output: B, kc, kh, kw
    """
    # Need to put PSFs on a grid given by x, y
    x = torch.unique(grid[:, 0])
    y = torch.unique(grid[:, 1])

    ## `grid_sample` version - backward a lot faster but not thoroughly checked for correctness
    psf_shape = psfs.shape[1:]
    psfs = psfs.view(len(y), len(x), -1).permute(2, 0, 1)[None, :]

    # pos: N, 2
    num_queries = pos.shape[0]
    pos = pos.view(1, num_queries, 1, 2).to(torch.float32) * 2 - 1
    interp = torch.nn.functional.grid_sample(psfs, pos, align_corners=True)
    interp = interp.view(-1, num_queries).T.reshape(num_queries, *psf_shape)
    if norm_output:
        return norm_sum_to_one(interp, eps=1e-5)
    return interp


class PSF(torch.nn.Module):
    def __init__(
        self,
        positions: torch.Tensor,
        kernels: torch.Tensor,
        do_rotation: bool = False,
    ):
        super().__init__()
        if kernels.dim() != 4:
            raise ValueError(f"Kernels must have 4 dimensions: B, C, H, W but found shape {kernels.shape}.")
        if positions.shape[0] != kernels.shape[0]:
            raise ValueError(f"Positions and kernels must have the same batch dimension but found {positions.shape[0]} and {kernels.shape[0]}.")
        if positions.dim() != 2 or positions.shape[1] != 2:
            raise ValueError(f"Positions must have shape B, 2 for x, y positions but found shape {positions.shape}.")
        if positions.amin() < 0 or positions.amax() > 1:
            raise ValueError(f"Positions should be within the [0, 1] range but found {positions.amin():.4f} and {positions.amax():.4f}")
        self.filter_size = (kernels.shape[-2], kernels.shape[-1])
        self.N = positions.shape[0]

        self.register_buffer("loc", positions)
        self.register_buffer("psfs", norm_sum_to_one(kernels))

        self.rot_psfs = None
        if do_rotation:
            radii, angles = cart2pol(self.loc[:, 0], self.loc[:, 1], 2, 2)
            rot_psfs = torch.stack([
                rotate_patch(psf, math.degrees(angle)) for psf, angle in zip(self.psfs, angles)
            ], dim=0)
            self.register_buffer("rot_psfs", norm_sum_to_one(rot_psfs))

    def __len__(self):
        return self.N

    def get_psf_by_index(self, index):
        index = index % self.N
        psfs = self.psfs if self.rot_psfs is None else self.rot_psfs
        return {
            "x": self.loc[index, 0],
            "y": self.loc[index, 1],
            "psf": psfs[index],
        }

    def get_psf_by_location(self, x: float, y: float):
        distances = torch.linalg.norm(self.loc - torch.tensor([x, y], dtype=self.loc.dtype, device=self.loc.device), dim=1)
        index = torch.argmin(distances)
        return self.get_psf_by_index(index)

    @property
    @cache
    def is_regular_grid(self) -> bool:
        uniq_x = torch.unique(self.loc[:, 0])
        uniq_y = torch.unique(self.loc[:, 1])
        if len(uniq_x) * len(uniq_y) == self.N:
            return True
        return False

    @staticmethod
    def from_path(psf_path, filter_size=None, **kwargs):
        psf_data = torch.load(psf_path, weights_only=True)

        psfs = psf_data["psf"]
        if filter_size is not None:
            psfs = TF.center_crop(psfs, filter_size)

        return PSF(
            torch.stack([psf_data["x"], psf_data["y"]], 1),
            psfs,
            **kwargs
        )
