
from typing import Literal
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity

from .torch_utils import equate_kernel_shapes


class LPIPS(torch.nn.Module):
    def __init__(self, net_type: Literal["vgg", "alex"] = "vgg"):
        super().__init__()
        self.net_type = net_type
        self.net = lpips.LPIPS(net=net_type)

    @torch.no_grad()
    def forward(self, i1, i2):
        """
        Inputs should be in [0, 1]
        """
        if i1.dim() == 3:
            i1 = i1[None, ...]
        if i2.dim() == 3:
            i2 = i2[None, ...]
        return self.net(i1, i2, normalize=True)


def calc_lpips(i1: torch.Tensor, i2: torch.Tensor, net_type: Literal["vgg", "alex"]):
    return LPIPS(net_type)(i1, i2)


def calc_psnr(i1: torch.Tensor | np.ndarray, i2: torch.Tensor | np.ndarray) -> float | torch.Tensor:
    # Assumes normalized inputs between [0, 1]
    # Also assumes no batch size
    if isinstance(i1, np.ndarray):
        i1 = torch.from_numpy(i1)
    if isinstance(i2, np.ndarray):
        i2 = torch.from_numpy(i2)
    assert i1.ndim == i2.ndim
    if i1.ndim == 4:
        if i1.shape[0] != i2.shape[0]:
            raise ValueError(f"Batch sizes of `i1` and `i2` do not match. Found {i1.shape[0]} and {i2.shape[0]}")
    else:
        i1 = i1[None, ...]
        i2 = i2[None, ...]

    psnrs = []
    for j in range(i1.shape[0]):
        mse = torch.mean(torch.square(i1[j] - i2[j]))
        psnrs.append(20 * torch.log10(1 / (torch.sqrt(mse) + 1e-8)))

    if i1.shape[0] == 1:
        return psnrs[0].item()
    return torch.as_tensor(psnrs).flatten()


def calc_psnr_matlab(i1: torch.Tensor, i2: torch.Tensor) -> float:
    if i1.ndim == 4:
        assert i1.shape[0] == 1 and i2.shape[0] == 1, "calc_psnr_matplab only supports single images."
        i1 = i1.squeeze(0)
        i2 = i2.squeeze(0)
    assert i1.shape[0] == 3, "Input images to calc_psnr_matlab must have 3 channels"
    assert i2.shape[0] == 3, "Input images to calc_psnr_matlab must have 3 channels"

    # Range here should be 0, 1
    weight = torch.Tensor([[65.481, -37.797, 112.0],
                            [128.553, -74.203, -93.786],
                            [24.966, 112.0, -18.214]]).to(i1)
    bias = torch.Tensor([16, 128, 128]).view(3, 1, 1).to(i1)
    ycbcr_i1 = torch.matmul(i1.permute(1, 2, 0), weight).permute(2, 0, 1) + bias
    ycbcr_i2 = torch.matmul(i2.permute(1, 2, 0), weight).permute(2, 0, 1) + bias
    # Now range is 0, 255
    imdiff = ycbcr_i1[0].double() - ycbcr_i2[0].double()  # only compare Y channel
    mse_value = torch.mean(imdiff ** 2) + 1e-8
    psnr_metrics = 10 * torch.log10((255.0 ** 2) / mse_value)
    return psnr_metrics.item()


def calc_ssim(i1: torch.Tensor, i2: torch.Tensor) -> float | torch.Tensor:
    assert i1.ndim == i2.ndim
    if i1.ndim == 4:
        if i1.shape[0] != i2.shape[0]:
            raise ValueError(f"Batch sizes of `i1` and `i2` do not match. Found {i1.shape[0]} and {i2.shape[0]}")
    else:
        i1 = i1[None, ...]
        i2 = i2[None, ...]

    ssims = []
    for j in range(i1.shape[0]):
        ssims.append(structural_similarity(
            i1[j].numpy(force=True),
            i2[j].numpy(force=True),
            full=False,
            data_range=1,
            channel_axis=0,
        )) # type: ignore

    if i1.shape[0] == 1:
        return ssims[0].item()
    return torch.as_tensor(ssims).flatten()


def calc_ncc(k1: torch.Tensor, k2: torch.Tensor, eps=1e-6):
    """ 2-dimensional normalized cross correlation (NCC)
    """
    if k1.ndim == 3:
        k1 = k1[None, :]
    if k2.ndim == 3:
        k2 = k2[None, :]
    bs = k1.shape[0]
    assert k2.shape[0] == bs, "Batch-sizes of k1 and k2 do not match"
    k1, k2 = equate_kernel_shapes(k1, k2)

    k1 = k1.reshape(bs, -1)
    k2 = k2.reshape(bs, -1)
    k1_mean = torch.mean(k1, dim=1, keepdim=True)
    k2_mean = torch.mean(k2, dim=1, keepdim=True)
    # deviation
    k1 = k1 - k1_mean
    k2 = k2 - k2_mean

    dev_xy = torch.mul(k1,k2)
    dev_xx = torch.mul(k1,k1)
    dev_yy = torch.mul(k2,k2)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = (
        (dev_xy + eps / dev_xy.shape[1])
        / (torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    )
    ncc = torch.sum(ncc, dim=1)
    return torch.mean(ncc)
