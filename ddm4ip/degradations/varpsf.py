"""Various operators implementing non-uniform blur"""
from pathlib import Path
from typing import List, Literal, Sequence, Tuple
import torch

import deepinv
from deepinv.physics.functional import (
    conv2d, conv_transpose2d
)

from .degradation import fix_deepinv_state
from ddm4ip.psf.psf import PSF, get_psf_at_pos
from ddm4ip.utils.torch_utils import get_center_kernel
from ddm4ip.utils.torch_utils import crop_valid, pad_valid, pad_kernel


def kernels_from_spec(
    kernel_spec: Sequence[Tuple[float, float, float]],
    kernel_channels: int,
    pad_kernels: bool = True
) -> List[torch.Tensor]:
    """
    Args:
        kernel_spec (Sequence[Tuple[float, float, float]]): describes the parameters
            of the (gaussian) kernels
        kernel_channels (int): number of channels of the kernels
        pad_kernels (bool): whether all kernels should be padded to be of the same size

    Returns:
        List[torch.Tensor]: List of kernels, each of shape [kernel_channels, kernel_size, kernel_size]
            note that each kernel may be of different shape, unless `pad_kernels` is set to True
    """
    kernels = []
    max_size = 0
    for k in kernel_spec:
        if len(k) != 3:
            raise ValueError(
                f"Kernel specification must be a tuple containing "
                f"(std-x, std-y, angle) but found sequence of length {len(k)}."
            )
        kernel = deepinv.physics.blur.gaussian_blur(
            sigma=[k[0], k[1]],
            angle=k[2], # type: ignore
        )
        # Remove first dimension which is unneeded and
        # expand the second dimension to the required number of channels
        # TODO: kernel_channels > 1 is untested and I'm not sure if it's correct.
        kernel = kernel.squeeze(0).expand(kernel_channels, -1, -1)
        max_size = max(max_size, kernel.shape[-1])
        kernels.append(kernel)
    if pad_kernels:
        for i in range(len(kernels)):
            kernels[i] = pad_kernel(kernels[i], max_size)
    return kernels


def check_conditioning(conditioning: torch.Tensor, img: torch.Tensor | None, check_uniform: bool) -> torch.Tensor:
    # If conditioning doesn't have batch-dim add it in
    if conditioning.dim() == 3:
        conditioning = conditioning[None, ...]
    if img is not None:
        if img.dim() != 4:
            raise ValueError(f"Image must be 4D but found image with shape {img.shape}")
        if conditioning.dim() != img.dim():
            raise ValueError(
                f"Conditioning and image must have the same number of "
                f"dimensions. Found {conditioning.shape=} and {img.shape=}"
            )
        # exclude channels dim
        cond_shape_noch = conditioning.shape[:-3] + conditioning.shape[-2:]
        img_shape_noch = img.shape[:-3] + img.shape[-2:]
        if cond_shape_noch != img_shape_noch:
            raise ValueError(
                f"Conditioning and image must have the same shape "
                f"apart from the channels dimension. Found shapes "
                f"{conditioning.shape=} and {img.shape=}"
            )
    if check_uniform:
        for c_ch in range(conditioning.shape[-3]):
            if not torch.all(conditioning[..., c_ch, :, :] == conditioning[..., c_ch, 0, 0][..., None, None]):
                raise ValueError(f"Conditioning does not have uniform value at channel {c_ch}.")
    return conditioning


def crop_for_padding_mode(padding, x, ph, pw, ih, iw):
    if padding == "valid":
        out = x
    elif padding == "circular":
        out = x[:, :, ph : -ph + ih, pw : -pw + iw]
        # sides
        out[:, :, : ph - ih, :] += x[:, :, -ph + ih :, pw : -pw + iw]
        out[:, :, -ph:, :] += x[:, :, :ph, pw : -pw + iw]
        out[:, :, :, : pw - iw] += x[:, :, ph : -ph + ih, -pw + iw :]
        out[:, :, :, -pw:] += x[:, :, ph : -ph + ih, :pw]
        # corners
        out[:, :, : ph - ih, : pw - iw] += x[:, :, -ph + ih :, -pw + iw :]
        out[:, :, -ph:, -pw:] += x[:, :, :ph, :pw]
        out[:, :, : ph - ih, -pw:] += x[:, :, -ph + ih :, :pw]
        out[:, :, -ph:, : pw - iw] += x[:, :, :ph, -pw + iw :]

    elif padding == "reflect":
        out = x[:, :, ph : -ph + ih, pw : -pw + iw]
        # sides
        out[:, :, 1 : 1 + ph, :] += x[:, :, :ph, pw : -pw + iw].flip(dims=(2,))
        out[:, :, -ph + ih - 1 : -1, :] += x[:, :, -ph + ih :, pw : -pw + iw].flip(
            dims=(2,)
        )
        out[:, :, :, 1 : 1 + pw] += x[:, :, ph : -ph + ih, :pw].flip(dims=(3,))
        out[:, :, :, -pw + iw - 1 : -1] += x[:, :, ph : -ph + ih, -pw + iw :].flip(
            dims=(3,)
        )
        # corners
        out[:, :, 1 : 1 + ph, 1 : 1 + pw] += x[:, :, :ph, :pw].flip(dims=(2, 3))
        out[:, :, -ph + ih - 1 : -1, -pw + iw - 1 : -1] += x[
            :, :, -ph + ih :, -pw + iw :
        ].flip(dims=(2, 3))
        out[:, :, -ph + ih - 1 : -1, 1 : 1 + pw] += x[:, :, -ph + ih :, :pw].flip(
            dims=(2, 3)
        )
        out[:, :, 1 : 1 + ph, -pw + iw - 1 : -1] += x[:, :, :ph, -pw + iw :].flip(
            dims=(2, 3)
        )

    elif padding == "replicate":
        out = x[:, :, ph : -ph + ih, pw : -pw + iw]
        # sides
        out[:, :, 0, :] += x[:, :, :ph, pw : -pw + iw].sum(2)
        out[:, :, -1, :] += x[:, :, -ph + ih :, pw : -pw + iw].sum(2)
        out[:, :, :, 0] += x[:, :, ph : -ph + ih, :pw].sum(3)
        out[:, :, :, -1] += x[:, :, ph : -ph + ih, -pw + iw :].sum(3)
        # corners
        out[:, :, 0, 0] += x[:, :, :ph, :pw].sum(3).sum(2)
        out[:, :, -1, -1] += x[:, :, -ph + ih :, -pw + iw :].sum(3).sum(2)
        out[:, :, -1, 0] += x[:, :, -ph + ih :, :pw].sum(3).sum(2)
        out[:, :, 0, -1] += x[:, :, :ph, -pw + iw :].sum(3).sum(2)
    elif padding == "constant":
        out = x[:, :, ph : -(ph - ih), pw : -(pw - iw)]
    else:
        raise ValueError(f"Padding mode {padding} not valid.")
    return out


class PerPixelBlur(deepinv.physics.LinearPhysics):
    """
    Handling of padding matches the deepinv Blur implementation
    """
    def __init__(self, filters=None, padding="replicate", **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        self.update_parameters(filters=filters)

    def A(self, x, *, filters=None, **kwargs):
        self.update_parameters(filters=filters, **kwargs)

        # flip is to do convolution, not cross-correlation (like in deepinv conv2d)
        kernels = self.filters.flip(-1, -2)

        B, C, H, W = x.shape
        b, hw, kc, kh, kw = kernels.shape
        if self.padding != "valid":
            ph = kh // 2
            ih = (kh - 1) % 2
            pw = kw // 2
            iw = (kw - 1) % 2
            pad = (pw, pw - iw, ph, ph - ih)  # because functional.pad is w,h instead of h,w

            x = torch.nn.functional.pad(x, pad, mode=self.padding, value=0)
            B, C, H, W = x.size()

        # Maybe crop kernels
        crp_x_h, crp_x_w = H - kh + 1, W - kw + 1
        if hw != crp_x_h * crp_x_w:
            if hw != H * W:
                raise ValueError(
                    f"Kernels hw dimension error. Found {hw} and expected "
                    f"either {crp_x_h * crp_x_w} (cropped) or {H * W} (full)."
                )
            # Crop kernel from H, W to crp_x_h, crp_x_w
            kernels = crop_valid(kernels.view(b, H, W, kc, kh, kw), (kh, kw), dims=(1, 2))  # B, ch, cw, kc, kh, kw
            # Permute kernels to move image H,W after kernel kH,KW
            kernels = kernels.permute(0, 3, 4, 5, 1, 2).contiguous()
            kernels = kernels.view(b, kc, kh * kw, crp_x_h * crp_x_w)  # B, kc, kh*kw, cH*cW
        else:
            # Permute kernels to move image H,W after kernel kH,KW
            kernels = kernels.permute(0, 2, 3, 4, 1).contiguous()
            kernels = kernels.view(b, kc, kh * kw, crp_x_h * crp_x_w)  # B, kc, kh*kw, cH*cW

        if b != B:
            if b != 1:
                raise ValueError(f"Batch sizes must match, found {B} and {b}.")
        if kc != C:
            if kc != 1:
                raise ValueError(f"Channels must match found {C} and {kc}.")

        x = torch.nn.functional.unfold(x, (kh, kw), padding=0)
        x = x.view(B, C, kh * kw, crp_x_h * crp_x_w)
        # kernels: B, kc, kh*kw, H*W
        # x      : B, C,  kh*kw, H*W
        # out    : B, C,  H*W
        # y = torch.einsum("bcks,bcks->bcs", kernels, x)
        y = torch.sum(kernels * x, dim=2)
        y = y.view(B, C, crp_x_h, crp_x_w)
        return y

    def A_adjoint(self, y, *, filters=None, **kwargs):
        self.update_parameters(filters=filters, **kwargs)

        # kernels: B, H*W, kc, kh, kw
        # flip is to do convolution, not cross-correlation (like in deepinv conv2d)
        kernels = self.filters.flip(-1, -2)
        # This is for transpose convolution
        kernels = torch.rot90(kernels, k=2, dims=(-2, -1))

        B, C, H, W = y.shape
        # kernels: B, H*W, kc, kh, kw
        b, hw, kc, kh, kw = kernels.shape
        ph = kh // 2
        pw = kw // 2
        ih = (kh - 1) % 2
        iw = (kw - 1) % 2
        if self.padding != "valid":
            if ph == 0 or pw == 0:
                raise ValueError(
                    "Both dimensions of the filter must be strictly greater than 2 if padding != 'valid'"
                )

        if b != B:
            if b != 1:
                raise ValueError(f"Batch sizes must match, found {B} and {b}.")
        if kc != C:
            if kc != 1:
                raise ValueError(f"Channels must match found {C} and {kc}.")

        # reshape to: B, kc, kh*kw, H*W
        kernels = kernels.reshape(B, hw, kc, kh * kw).permute(0, 2, 3, 1)

        pad_x_h, pad_x_w = H + kh - 1, W + kw - 1

        if hw != pad_x_h * pad_x_w:
            if hw != H * W:
                raise ValueError(
                    f"Kernels hw dimension error. Found {hw} and expected "
                    f"either {pad_x_h * pad_x_w} (padded) or {H * W} (full)."
                )
            # Pad kernel from H, W to pad_x_h, pad_x_w
            kernels = pad_valid(kernels.view(B, kc, kh*kw, H, W), (kh, kw))
            kernels = kernels.reshape(B, kc, kh*kw, pad_x_h*pad_x_w)

        y = pad_valid(y, (kh * 2 - 1, kw * 2 - 1))
        y = torch.nn.functional.unfold(y, (kh, kw), padding=0)
        y = y.view(B, C, -1, pad_x_h * pad_x_w)
        x = (kernels * y).sum(2).view(B, C, pad_x_h, pad_x_w)
        x = crop_for_padding_mode(self.padding, x, ph, pw, ih, iw)
        return x

    def update_parameters(self, filters: torch.Tensor | None = None, **kwargs):
        if filters is not None:
            if filters.dim() != 5:
                raise ValueError(
                    f"Filters must have 5 dimensions (batch, image_height*image_width, "
                    f"channels, height, width) but found {filters.shape}"
                )
            if hasattr(self, "filters"):
                filters = filters.to(self.filters.device)
            self.filters = torch.nn.Parameter(
                filters, requires_grad=False
            )
        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)

    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d


class PerPixelInterpolatedBlur(PerPixelBlur):
    def __init__(
        self,
        psf_path: str,
        kernel_size: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.psf = PSF.from_path(
            psf_path=psf_path, filter_size=kernel_size, do_rotation=False
        )

    def A(self, x, *, conditioning=None, noise_level=None, **kwargs):
        r"""
        Applies the filter to the input image.

        :param torch.Tensor x: input image.
        """
        new_filters = None
        if conditioning is not None:
            conditioning = check_conditioning(conditioning, x, check_uniform=False)
            new_filters = self.get_kernel(conditioning)

        return super().A(x, filters=new_filters, sigma=noise_level, **kwargs)

    def A_adjoint(self, y, *, conditioning=None, noise_level=None, **kwargs):
        new_filters = None
        if conditioning is not None:
            conditioning = check_conditioning(conditioning, y, check_uniform=False)
            new_filters = self.get_kernel(conditioning)

        return super().A_adjoint(y, filters=new_filters, sigma=noise_level, **kwargs)

    def get_kernel(self, conditioning, img=None):
        # check_conditioning will add batch if absent
        conditioning = check_conditioning(conditioning, None, check_uniform=False)
        cond_dev = conditioning.device
        conditioning = conditioning.to(self.psf.psfs.device)
        # Conditioning: B, 2, H, W same size as x (apart from channels-dimension)
        # each H, W slice contains location in the image normalized between 0, 1
        # the first slice (channels-dim) is height and the second is width
        bs, _, h, w = conditioning.shape
        # 1. reshape such that we have B*H*W, 2 (the points at which we should interpolate)
        #    we make `conditioning` column contiguous to avoid a warning in `get_psf_at_pos`
        conditioning = conditioning.permute(1, 0, 2, 3).reshape(2, -1).T
        # 2. Get the kernels from the PSF object. Output of shape B*H*W, kc, kh, kw
        kernels = get_psf_at_pos(
            psfs=self.psf.psfs,
            grid=self.psf.loc,
            pos=conditioning
        )
        kc, kh, kw = kernels.shape[1:4]
        # 3. reshape to B, H*W, kc, kh, kw
        kernels = kernels.reshape(bs, h * w, kc, kh, kw)
        return kernels.to(device=cond_dev)


class PerPatchInterpolatedBlur(deepinv.physics.LinearPhysics):
    def __init__(
        self,
        psf_path: str | PSF,
        kernel_size: int,
        filter=None,
        padding: Literal["valid", "replicate"] = "replicate",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        if isinstance(psf_path, (str, Path)):
            self.psf = PSF.from_path(
                psf_path=psf_path, filter_size=kernel_size, do_rotation=False
            )
        else:
            self.psf = psf_path
        self.padding = padding
        self.update_parameters(filter=filter, **kwargs)

    def A(self, x, *, conditioning=None, noise_level=None, **kwargs):
        r"""
        Applies the filter to the input image.

        :param torch.Tensor x: input image.
        """
        new_filters = None
        if conditioning is not None:
            conditioning = check_conditioning(conditioning, x, check_uniform=False)
            new_filters = self.get_kernel(conditioning)
        self.update_parameters(sigma=noise_level, filter=new_filters, **kwargs)

        return conv2d(x, self.filter, padding=self.padding)

    def A_adjoint(self, y, *, conditioning=None, noise_level=None, **kwargs):
        new_filters = None
        if conditioning is not None:
            conditioning = check_conditioning(conditioning, y, check_uniform=False)
            new_filters = self.get_kernel(conditioning)
        self.update_parameters(sigma=noise_level, filter=new_filters, **kwargs)

        return conv_transpose2d(y, filter=self.filter, padding=self.padding)

    def get_kernel(self, conditioning, **kwargs):
        # check_conditioning will add batch dimension if absent
        conditioning = check_conditioning(conditioning, None, check_uniform=False)
        # Conditioning: B, 2, H, W same size as x (apart from channels-dimension)
        # each H, W slice contains location in the image normalized between 0, 1
        # the first slice (channels-dim) is height and the second is width
        # 1. Compute the mean (center coordinate) of x and y values -> B, 2
        centers = conditioning.mean(dim=(-1, -2))
        # 2. Get the kernels from the PSF object. Output of shape B, kc, kh, kw
        kernels = get_psf_at_pos(
            psfs=self.psf.psfs,
            grid=self.psf.loc,
            pos=centers
        )
        return kernels

    def update_parameters(self, filter=None, **kwargs):
        if filter is not None:
            if hasattr(self, "filter"):
                filter = filter.to(device=self.filter.device)
            self.filter = torch.nn.Parameter(filter, requires_grad=False)

        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)

    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d


class PixelToPatchAdapter(PerPatchInterpolatedBlur):
    def __init__(self, space_blur: PerPixelInterpolatedBlur, img_h: int, img_w: int):
        super().__init__(
            kernel_size=space_blur.kernel_size,
            padding=space_blur.padding,
            psf_path=space_blur.psf
        )
        self.img_h = img_h
        self.img_w = img_w
    def update_parameters(self, filter=None, **kwargs):
        if filter is not None:
            if hasattr(self, "filter"):
                filter = filter.to(device=self.filter.device)
            filter = get_center_kernel(filter, img_h=self.img_h, img_w=self.img_w)
            self.filter = torch.nn.Parameter(filter, requires_grad=False)

        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)
