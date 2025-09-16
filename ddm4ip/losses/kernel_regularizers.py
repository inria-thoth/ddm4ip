import torch


__all__ = (
    "kernel_regularization",
)

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


def kernel_regularization(
    kernel: torch.Tensor,
    sparse: float | None,
    center: float | None,
    sum_to_one: float | None,
    gaussian: float | None,
):
    loss_info = {}
    loss_sparse = 0
    if sparse is not None and sparse > 0:
        loss_sparse = sparse * sparse_reg(kernel)
        loss_info["loss/sparse"] = loss_sparse.detach()
    loss_center = 0
    if center is not None and center > 0:
        loss_center = center * center_reg(kernel)
        loss_info["loss/center"] = loss_center.detach()
    loss_sum_to_one = 0
    if sum_to_one is not None and sum_to_one > 0:
        loss_sum_to_one = sum_to_one * sum_to_one_reg(kernel)
        loss_info["loss/sum_to_one"] = loss_sum_to_one.detach()
    loss_gaussian = 0
    if gaussian is not None and gaussian > 0:
        loss_gaussian = gaussian * gaussian_reg(kernel)
        loss_info['loss/gaussian'] = loss_gaussian.detach()

    total_reg = loss_sparse + loss_center + loss_sum_to_one + loss_gaussian
    return total_reg, loss_info