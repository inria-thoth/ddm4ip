import math

import matplotlib
import matplotlib.axes
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms.v2.functional as TF
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from ddm4ip.data.base import Batch, DatasetType
from ddm4ip.data.utils import get_space_varying_patches
from ddm4ip.flowmatching_utils.generate import imggen
from ddm4ip.losses.di_y import DiffInstructOnY
from ddm4ip.utils import distributed
from ddm4ip.utils.metrics import calc_psnr
from ddm4ip.utils.torch_utils import equate_kernel_shapes, get_center_kernel


def imshow_with_cbar(fig, ax: matplotlib.axes.Axes, img, axis_off=True, cmap='viridis', vmin=None, vmax=None):
    # cmap='RdBu', vmin=-1.0, vmax=1.0
    img_shape = img.shape
    if len(img_shape) == 2:
        num_ch = 1
    else:
        num_ch = img_shape[-1]
    # if num_ch == 3
    plt_img = ax.imshow(img, aspect='equal', interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = num_ch == 1
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        fig.colorbar(plt_img, cax=cax, orientation='horizontal')
    if axis_off:
        ax.set_axis_off()


def plot_3ch_kernels(pred_filters: torch.Tensor, title: str = ""):
    # pred_filters: H, W, C
    assert pred_filters.shape[-1] == 3
    fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
    if pred_filters.max() <= 0.25:
        pred_filters = pred_filters * 10
        title = f"{title} x10"
        if pred_filters.max() <= 0.15:
            pred_filters = pred_filters * 10
            title = f"{title} x10"
    imshow_with_cbar(fig, ax, pred_filters)
    ax.set_title(title)

    # imshow_with_cbar(fig, ax[0], pred_filters[:, :, 0], cmap='Reds')
    # ax[0].set_title(f"{title} R")
    # imshow_with_cbar(fig, ax[1], pred_filters[:, :, 1], cmap='Greens')
    # ax[1].set_title(f"{title} G")
    # imshow_with_cbar(fig, ax[2], pred_filters[:, :, 2], cmap='Blues')
    # ax[2].set_title(f"{title} B")
    return fig


"""
Plot generated with diffusion
"""
@torch.no_grad()
def plot_imggen(config, flow_nn, batch: Batch, img_size: tuple[int, int, int], dset: DatasetType, device) -> torch.Tensor | None:
    if "generation" not in config:
        return None

    batch_size = config['training']['batch_size']
    is_space_varying = getattr(dset, "space_conditioning", False) or getattr(dset, "random_space_conditioning", False)
    if is_space_varying:
        all_patches = get_space_varying_patches(img_size=1024, patch_size=img_size[-1])
        n_h_patches, nrow = all_patches.shape[:2]
        all_patches = all_patches.view(-1, *all_patches.shape[2:])
        seeds = [42] * len(all_patches)
        conditioning = all_patches.to(device)
    else:
        num_img_for_sample = min(64, batch_size)
        seeds = range(42, 42 + num_img_for_sample)
        conditioning = batch.corrupt_conditioning[:num_img_for_sample].to(device) if batch.corrupt_conditioning is not None else None
        nrow = int(math.sqrt(num_img_for_sample))
        if conditioning is not None and (conditioning.shape[-2] != img_size[-2] or conditioning.shape[-1] != img_size[-1]):
            conditioning = TF.center_crop(conditioning, [img_size[-2], img_size[-1]])

    image_iter = imggen(
        net=flow_nn,
        img_size=img_size,
        label_dim=dset.label_dim,
        seeds=seeds,
        batch_size=batch_size,
        device=device,
        conditioning=conditioning,
        return_trajectory=False,
        **config['generation'],
    )
    # Gather within process
    gen_imgs = [img.solution for img in image_iter if img.solution is not None]
    gen_imgs = torch.cat(gen_imgs, dim=0) if len(gen_imgs) > 0 else None
    assert gen_imgs is not None
    # Gather across processes
    if torch.distributed.is_initialized():
        gen_imgs = torch.cat([img.cpu() for img in distributed._simple_gather_all_tensors(gen_imgs, None, distributed.get_world_size())], dim=0)
    image_grid = make_grid(
        gen_imgs.clamp_(0, 1), nrow=nrow, padding=0
    )
    return image_grid

"""
Plot explicit kernels
"""

def fix_kernel_dimensions(kernel, conditioning=None):
    """
    Fix kernel dimensions to be C, H, W. There are a variety of possible input dimensions.
    """
    if kernel.dim() == 2:
        kernel = kernel[None, :]  # kC, kH, kW
    elif kernel.dim() == 4:
        num_ch = kernel.shape[-3]
        if kernel.shape[0] > 1:
            kernel = make_grid(kernel, nrow=4, padding=0)
        else:
            kernel = kernel[0]
        if num_ch == 1:
            kernel = kernel[0:1]  # kC, H, W
    elif kernel.dim() == 5:
        # Multiple kernels for each image (B, H*W, kc, kh, kw). Visualize the middle one
        assert conditioning is not None
        num_ch = kernel.shape[-3]
        kernel = get_center_kernel(kernel, img_h=conditioning.shape[-2], img_w=conditioning.shape[-1])
        kernel = make_grid(kernel, nrow=4, padding=0)
        if num_ch == 1:
            kernel = kernel[0:1]  # kC, H, W
    return kernel


def kernel_diff(pred_kernel: torch.Tensor, true_kernel: torch.Tensor):
    # Now there are only two different possibilities based on number of kernel channels
    num_ch = pred_kernel.shape[0]
    if num_ch == 1:
        true_kernel_np = true_kernel[0].numpy(force=True)
        pred_kernel_np = pred_kernel[0].numpy(force=True)
        difference = pred_kernel_np - true_kernel_np
        mae = np.mean(np.abs(difference))
    elif num_ch == 3:
        true_kernel_np = true_kernel.permute(1, 2, 0).numpy(force=True)  # need channels-last for matplotlib
        pred_kernel_np = pred_kernel.permute(1, 2, 0).numpy(force=True)
        difference = np.mean(np.abs(pred_kernel_np - true_kernel_np), axis=-1)  # average of three channels
        mae = np.mean(difference)
    else:
        raise ValueError(f"Invalid number of channels: {num_ch}")
    return difference, mae


@torch.no_grad()
def plot_explicit_kernels(dataset, kernel_nn, batch: Batch | None, batch_size: int, device):
    if not hasattr(kernel_nn, "get_kernel"):
        return None, None

    batch_dev = None
    if batch is not None:
        num_imgs = min(16, batch_size)
        batch_dev = batch[:num_imgs].to(device)
    conditioning = None
    if batch_dev is not None:
        conditioning = batch_dev.clean_conditioning
    img = None
    if batch_dev is not None:
        img = batch_dev.clean

    pred_kernel = kernel_nn.get_kernel(img=img, conditioning=conditioning)
    true_kernel = None
    if dataset.corruption is not None:
        true_kernel = dataset.corruption.get_kernel(conditioning=conditioning)
        pred_kernel, true_kernel = equate_kernel_shapes(pred_kernel, true_kernel)
        true_kernel = fix_kernel_dimensions(true_kernel, conditioning)
        true_kernel = true_kernel.cpu().detach()

    pred_kernel = fix_kernel_dimensions(pred_kernel, conditioning)
    pred_kernel = pred_kernel.cpu().detach()
    pred_filters_flat = pred_kernel.clone()

    if true_kernel is not None:
        difference, mae = kernel_diff(pred_kernel, true_kernel)
        fig, ax = plt.subplots(ncols=3, figsize=(9, 4))
        imshow_with_cbar(fig, ax[0], true_kernel.permute(1, 2, 0))
        ax[0].set_title("True")
        imshow_with_cbar(fig, ax[1], pred_kernel.permute(1, 2, 0))
        ax[1].set_title("Pred")
        imshow_with_cbar(fig, ax[2], difference)
        ax[2].set_title(f"Difference. MAE={mae:.3e}")
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        imshow_with_cbar(fig, ax, pred_kernel.permute(1, 2, 0))
        ax.set_title("Predicted")

    fig.tight_layout()
    return fig, pred_filters_flat


def make_kernels_grid(kernels: torch.Tensor, nrow: int) -> torch.Tensor | None:
    if kernels.dim() != 4:
        return None
    num_ch = kernels.shape[1]
    kernels = make_grid(kernels, nrow=nrow, padding=0)
    if num_ch == 1:
        kernels = kernels[0]
    else:
        kernels = kernels.permute(1, 2, 0)
    return kernels


@torch.no_grad()
def plot_spacevarying_kernels(
    corruption,
    patch_size: int,
    kernel_nn: torch.nn.Module,
    batch_size: int, device
):
    true_ker_avail = True
    if corruption is None or not hasattr(corruption, "get_kernel"):
        true_ker_avail = False

    # nH, nW, 2, patch_size, patch_size
    all_patches = get_space_varying_patches(img_size=1024, patch_size=patch_size)
    n_h_patches, n_w_patches = all_patches.shape[:2]
    all_patches = all_patches.view(-1, *all_patches.shape[2:])
    pred_filters, true_filters = [], []
    for i in range(0, len(all_patches), batch_size):
        batch = all_patches[i: i + batch_size].to(device)
        try:
            pf = kernel_nn.get_kernel(img=None, conditioning=batch).cpu()
        except Exception:
            return None, None
        pf = get_center_kernel(pf, img_h=batch.shape[-2], img_w=batch.shape[-1])
        if pf.dim() != 4 or pf.shape[0] != batch.shape[0]:
            return None, None
        if true_ker_avail:
            tf = corruption.get_kernel(batch.cpu())
            tf = get_center_kernel(tf, img_h=batch.shape[-2], img_w=batch.shape[-1])
            if tf.dim() != 4 or tf.shape[0] != batch.shape[0]:
                return None, None
            tf, pf = equate_kernel_shapes(tf, pf)
            true_filters.append(tf)
        pred_filters.append(pf)
    pred_filters = torch.cat(pred_filters, 0)
    pred_filters_flat = pred_filters.clone()
    pred_filters = make_kernels_grid(pred_filters, nrow=n_w_patches)
    if true_ker_avail:
        true_filters = torch.cat(true_filters, 0)
        true_filters = make_kernels_grid(true_filters, nrow=n_w_patches)

    assert pred_filters is not None

    if (not true_ker_avail) and pred_filters.ndim == 3 and pred_filters.shape[-1] == 3:
        fig = plot_3ch_kernels(pred_filters, title="Predicted: ")
    else:
        ncols = 3 if true_ker_avail else 1
        fig, ax = plt.subplots(ncols=ncols, figsize=(ncols * 5, 5))
        if not isinstance(ax, np.ndarray):
            ax = [ax]
        imshow_with_cbar(fig, ax[0], pred_filters)
        ax[0].set_title("Predicted")
        if true_ker_avail:
            imshow_with_cbar(fig, ax[1], true_filters)
            ax[1].set_title("True")
            diff = (true_filters - pred_filters).abs()
            imshow_with_cbar(fig, ax[2], diff)
            ax[2].set_title(f"Difference. MAE={diff.mean():.2e}")

    fig.tight_layout()
    return fig, pred_filters_flat


"""
Plot implicit kernels
"""

@torch.no_grad()
def plot_implicit_kernels(batch: Batch, kernel_nn, num_imgs: int = 16, is_diy: bool = True):
    batch_dev = batch[:num_imgs].cuda()
    if batch_dev.clean is None or batch_dev.corrupt is None or batch_dev.noise_level is None:
        return None
    noise_level = batch_dev.noise_level
    if is_diy:
        clean, clean_lbl = batch_dev.clean, r"$x$"
        labels = batch_dev if batch_dev.clean_label is not None else None
        corrupt_true, corrupt_true_lbl = batch_dev.corrupt, r"$y$"
        cond = batch_dev.clean_conditioning if batch_dev.clean_conditioning is not None else None
        corrupt_pred, corrupt_pred_lbl = kernel_nn(clean, noise_level, labels, cond, **batch_dev.meta), r"$A_{\hat{\omega}} x + \epsilon$"
        title = r"$y - A_{\hat{\omega}} x$"
    else:
        psnr_y = calc_psnr(batch_dev.corrupt, batch_dev.clean)
        clean, clean_lbl = batch_dev.corrupt, r"$y$ " + f"PSNR: {psnr_y:.2f}"
        labels = batch_dev.corrupt_label if batch_dev.corrupt_label is not None else None
        corrupt_true, corrupt_true_lbl = batch_dev.clean, r"$x$"
        cond = batch_dev.corrupt_conditioning if batch_dev.corrupt_conditioning is not None else None
        corrupt_pred, corrupt_pred_lbl = kernel_nn(clean, noise_level, labels, cond, **batch_dev.meta), r"$h_{\hat{\omega}}(A^*x + \epsilon)$"
        psnr = calc_psnr(corrupt_true, corrupt_pred)
        title = r"$x - h_{\hat{\omega}}(y)$ " + f"PSNR: {psnr:.2f}"

    batch_size = clean.shape[0]
    n_img_row = int(math.sqrt(batch_size))

    clean = make_grid(
        clean.clamp_(0, 1),
        nrow=n_img_row, padding=0,
    ).permute(1, 2, 0).numpy(force=True)
    corrupt_true = make_grid(
        corrupt_true.clamp_(0, 1),
        nrow=n_img_row, padding=0,
    ).permute(1, 2, 0).numpy(force=True)
    corrupt_pred = make_grid(
        corrupt_pred.clamp_(0, 1),
        nrow=n_img_row, padding=0,
    ).permute(1, 2, 0).numpy(force=True)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 16))

    imshow_with_cbar(fig, ax[0, 0], clean)
    ax[0, 0].set_title(clean_lbl)

    imshow_with_cbar(fig, ax[0, 1], corrupt_true)
    ax[0, 1].set_title(corrupt_true_lbl)

    imshow_with_cbar(fig, ax[1, 0], corrupt_pred)
    ax[1, 0].set_title(corrupt_pred_lbl)

    if corrupt_true.shape == corrupt_pred.shape:
        diff = (corrupt_true - corrupt_pred).mean(2)  # average over channels
        if np.abs(diff).max() < 0.1:
            diff *= 10
            title = rf"{title} (x10)"
        ax[1, 1].imshow(diff, aspect='equal', interpolation='none', cmap='RdBu', vmin=-1.0, vmax=1.0)
        ax[1, 1].set_title(title)
    ax[1, 1].set_axis_off()

    fig.tight_layout()
    return fig


"""
Plot conditioning distribution
"""
@torch.no_grad()
def plot_conditioning_hist(diy_loss: DiffInstructOnY):
    if not hasattr(diy_loss, "conditioning_acc"):
        return None
    if len(diy_loss.conditioning_acc) == 0:
        return None
    cond = torch.cat(diy_loss.conditioning_acc, dim=0)
    fig, ax = plt.subplots()
    im = ax.hist2d(cond[:,0], cond[:, 1], bins=[8, 8])
    cb = fig.colorbar(im[-1], ax=ax)
    cb.set_label("Counts")
    fig.tight_layout()
    return fig