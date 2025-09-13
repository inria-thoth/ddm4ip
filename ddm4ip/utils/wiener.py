import math
import torch
import skimage.restoration.deconvolution

from ddm4ip.utils.torch_utils import center, uncenter


__all__ = (
    "WienerSolver",
)


class WienerSolver(torch.nn.Module):
    def __init__(self, balance: float, normalize: bool = True):
        super().__init__()
        self.balance = balance
        self.normalize = normalize

    def forward(self, y, physics, **kwargs):
        batch_size = y.shape[0]
        channels = y.shape[1]
        if self.normalize:
            y = center(y)
        filter = physics.filter
        if filter.shape[1] != channels and filter.shape[1] == 1:
            filter = filter.expand(filter.shape[0], channels, -1, -1)
        elif filter.shape[1] != channels:
            raise ValueError(f"PSF has incorrect number of channels. Expected {channels} but found {filter.shape[1]}.")

        x = torch.empty_like(y)
        for i in range(batch_size):
            for ch in range(channels):
                x[i, ch] = wiener_pt(y[i, ch], filter[i, ch], balance=self.balance)

        if self.normalize:
            x = uncenter(x)
        return x


def wiener_pt(image: torch.Tensor, psf: torch.Tensor, balance: float, reg=None, clip=True) -> torch.Tensor:
    r"""Pytorch port of scikit-learn Wiener-Hunt deconvolution
    """
    if reg is None:
        reg, _ = skimage.restoration.deconvolution.uft.laplacian(image.ndim, image.shape, is_real=True)
        reg = torch.from_numpy(reg).to(device=image.device)
    if not torch.is_complex(reg):
        reg = ir2tf(reg, image.shape)
    psf = psf.real.to(dtype=image.dtype)
    reg = reg.real.to(dtype=image.dtype)

    if psf.shape != reg.shape:
        trans_func = ir2tf(psf, image.shape)
    else:
        trans_func = psf

    wiener_filter = torch.conj(trans_func) / (
        torch.abs(trans_func) ** 2 + balance * torch.abs(reg) ** 2
    )
    deconv = uirfftn(wiener_filter * urfftn(image), shape=image.shape)

    if clip:
        deconv = deconv.clamp(-1, 1)

    return deconv


def ir2tf(imp_resp: torch.Tensor, shape: tuple[int] | torch.Size, dim=None) -> torch.Tensor:
    if not dim:
        dim = imp_resp.ndim
    # Zero padding and fill
    irpadded_dtype = imp_resp.dtype
    irpadded_device = imp_resp.device
    irpadded = torch.zeros(shape, dtype=irpadded_dtype, device=irpadded_device)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    # Roll for zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    for j, dim_size in enumerate(imp_resp.shape):
        if j >= imp_resp.ndim - dim:
            irpadded = torch.roll(irpadded, shifts=-int(math.floor(dim_size / 2)), dims=j)
    out = torch.fft.rfftn(irpadded, dim=tuple(range(-dim, 0)))
    return out


def uirfftn(inarray: torch.Tensor, dim=None, shape=None) -> torch.Tensor:
    if dim is None:
        dim = inarray.ndim
    outarray = torch.fft.irfftn(inarray, shape, dim=tuple(range(-dim, 0)), norm='ortho')
    return outarray


def urfftn(inarray, dim=None):
    if dim is None:
        dim = inarray.ndim
    outarray = torch.fft.rfftn(inarray, dim=tuple(range(-dim, 0)), norm='ortho')
    return outarray