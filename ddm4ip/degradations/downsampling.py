import deepinv
import torch
from deepinv.physics.functional import (
    filter_fft_2d,
    conv_transpose2d,
)

from .degradation import fix_deepinv_state


class Downsampling(deepinv.physics.Downsampling):
    """Wraps deepinv's `Downsampling` class"""
    def __init__(
        self,
        img_size=None,
        filter=None,
        factor=2,
        device="cpu",
        padding="circular",
        **kwargs,
    ):
        if img_size is None:
            img_size = (3, 16, 16)
        self.device = device
        super().__init__(img_size, filter, factor, device, padding, **kwargs)

    def get_kernel(self, img=None, conditioning=None):
        return self.filter

    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d

    def A_adjoint(self, y):
        imsize = (
            y.shape[-3],
            y.shape[-2] * self.factor,
            y.shape[-1] * self.factor,
        )
        if imsize != self.imsize:
            print(f"Updating high-res image size from {self.imsize} to {imsize}")
            self.imsize = imsize
            self.update_prox_params()

        x = torch.zeros((y.shape[0],) + imsize, device=y.device, dtype=y.dtype)
        x[:, :, :: self.factor, :: self.factor] = y  # upsample
        if self.filter is not None:
            x = conv_transpose2d(x, self.filter, padding=self.padding)
        return x

    def update_parameters(self, filter=None, **kwargs):
        r"""
        Updates the current filter.
        For some reason the base-class does not implement this!.

        :param torch.Tensor filter: New filter to be applied to the input image.
        """
        if filter is not None:
            self.filter = torch.nn.Parameter(
                filter.to(self.device), requires_grad=False
            )
            self.update_prox_params()
        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)

    def update_prox_params(self):
        self.Fh = filter_fft_2d(self.filter, self.imsize, real_fft=False).to(self.device)
        Fhc = torch.conj(self.Fh)
        Fh2 = Fhc * self.Fh
        self.Fhc = torch.nn.Parameter(Fhc, requires_grad=False)
        self.Fh2 = torch.nn.Parameter(Fh2, requires_grad=False)

