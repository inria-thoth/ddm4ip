import abc
from enum import Enum
from typing import Literal, Union
import torch
import torch.nn as nn

from ddm4ip.degradations.blur import Blur
from ddm4ip.degradations.degradation import GaussianNoise
from ddm4ip.degradations.downsampling import Downsampling
from ddm4ip.degradations.varpsf import PerPixelBlur
from ddm4ip.networks.unets import Block, MPConv, mp_silu
from ddm4ip.psf.psf import PSF, get_psf_at_pos, norm_sum_to_one
from ddm4ip.utils import distributed, persistence
from ddm4ip.utils.torch_utils import center


class BlurDegradationType(Enum):
    BLUR = 1
    PER_PIXEL_BLUR = 2
    DOWNSAMPLING = 3


class BlurNetBase(nn.Module, abc.ABC):
    def __init__(
        self,
        degradation_type: BlurDegradationType,
        padding: str,
        is_output_noisy: bool = True,
        learn_output_noise: bool = False,
        **deg_kwargs
    ):
        super().__init__()
        self.is_output_noisy = is_output_noisy
        self.learn_output_noise = learn_output_noise
        self.degradation_type = degradation_type
        self.padding = padding
        if learn_output_noise:
            assert is_output_noisy, "Cannot learn output noise if `is_output_noisy` is False"
            self.sigma = torch.nn.Parameter(torch.tensor([0.01]))

        if degradation_type == BlurDegradationType.BLUR:
            self.degradation = Blur(
                noise_model=GaussianNoise(sigma=0.0),
                padding=padding,
                **deg_kwargs
            )
        elif degradation_type == BlurDegradationType.PER_PIXEL_BLUR:
            self.degradation = PerPixelBlur(
                noise_model=GaussianNoise(sigma=0.0),
                padding=padding,
                **deg_kwargs
            )
        elif degradation_type == BlurDegradationType.DOWNSAMPLING:
            self.degradation = Downsampling(
                noise_model=GaussianNoise(sigma=0.0),
                padding=padding,
                **deg_kwargs
            )
        else:
            raise NotImplementedError(degradation_type)

        self.register_load_state_dict_pre_hook(self.fix_state_dict)

    def forward(
        self,
        clean_img: torch.Tensor,
        noise_level: Union[float, torch.Tensor] = 0.0,
        class_labels=None,
        conditioning=None,
        get_kernel: bool = False,
        **kwargs
    ):
        kernels = self.get_kernel(clean_img, conditioning, **kwargs)
        if isinstance(noise_level, float):
            # Necessary to prevent noise level going on CPU
            noise_level = torch.tensor(noise_level, device=kernels.device)
        if self.learn_output_noise:
            sigma = 0.0
        else:
            sigma = noise_level

        # NOTE: Must set filter like this because the `Blur` class will reset
        #      `requires_grad` to False on the filters
        if self.degradation_type == BlurDegradationType.BLUR:
            self.degradation.filter = kernels
        elif self.degradation_type == BlurDegradationType.PER_PIXEL_BLUR:
            self.degradation.filters = kernels
        elif self.degradation_type == BlurDegradationType.DOWNSAMPLING:
            self.degradation.filter = kernels
        else:
            raise NotImplementedError(self.degradation_type)
        out = self.degradation(clean_img, sigma=sigma)
        if self.learn_output_noise:
            out = out + torch.randn_like(out) * self.sigma

        if get_kernel:
            return out, kernels
        return out

    @abc.abstractmethod
    def get_kernel(self, img, conditioning, **kwargs) -> torch.Tensor:
        pass

    def fix_state_dict(self, module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        sd_sigma = state_dict["degradation.noise_model.sigma"]
        self_sigma = self.degradation.noise_model.sigma
        if sd_sigma.shape != self_sigma.shape:
            print("Warn: modifying noise-model sigma in state dict")
            state_dict["degradation.noise_model.sigma"] = self_sigma


class FourierEmbedding(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        freqs = (1000 ** (torch.arange(0, n_channels, 2).float() / n_channels))
        self.register_buffer("freqs", freqs)

    def forward(self, cond):
        cond_freq = torch.sin(cond.unsqueeze(1) * self.freqs[None, :, None, None, None])
        cond_freq = cond_freq.reshape(cond.shape[0], -1, *cond.shape[2:])
        return cond_freq


def get_default_psf_grid(num_psfs: tuple[int, int]):
    return torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, num_psfs[0]),  # x
            torch.linspace(0, 1, num_psfs[1]),  # y
            indexing='ij'
        ),
        dim=-1
    ).view(-1, 2)


@persistence.persistent_class
class DirectKernel(BlurNetBase):
    def __init__(
        self,
        kernel_size: int,
        padding: str,
        factor: int = 1,
        sum_to_one: bool = True,
        learn_output_noise: bool = False,
    ):
        deg_type = BlurDegradationType.BLUR if factor == 1 else BlurDegradationType.DOWNSAMPLING
        # Extra kwargs are simply ignored in deepinv.
        super().__init__(deg_type, padding=padding, factor=factor, learn_output_noise=learn_output_noise)
        self.sum_to_one = sum_to_one
        kernel = torch.randn(1, 1, kernel_size, kernel_size).abs_()
        kernel = norm_sum_to_one(kernel)
        self.kernel = torch.nn.Parameter(kernel, requires_grad=True)

    def get_kernel(self, img, conditioning, **kwargs):
        if self.sum_to_one:
            return norm_sum_to_one(self.kernel)
        return self.kernel


@persistence.persistent_class
class DirectCenterKernel(BlurNetBase):
    def __init__(
        self,
        psf_path: str | None,
        kernel_size: int,
        padding: str,
        num_psfs: tuple[int, int] | None = None,
        kernel_ch: int | None = None,
        sum_to_one: bool = True,
        learn_output_noise: bool = False,
        initialization: Literal["random", "square"] = "random",
    ):
        super().__init__(BlurDegradationType.BLUR, padding=padding, learn_output_noise=learn_output_noise)
        self.sum_to_one = sum_to_one
        if psf_path is None:
            assert num_psfs is not None, "psf_path is None so num_psfs must be specified"
            assert kernel_ch is not None, "psf_path is None so kernel_ch must be specified"
            psf_grid = get_default_psf_grid(num_psfs)
            psf_shape = (psf_grid.shape[0], kernel_ch, kernel_size, kernel_size)
        else:
            psf = PSF.from_path(
                psf_path=psf_path, filter_size=kernel_size, do_rotation=False
            )
            psf_grid = psf.loc
            psf_shape = psf.psfs.shape

        self.register_buffer("psf_grid", psf_grid)

        if initialization == "random":
            empty_kernels = torch.randn(psf_shape).abs_()
        elif initialization == "square":
            kernel_grid = torch.zeros(psf_shape)
            center_y, center_x = psf_shape[-2] // 2, psf_shape[-1] // 2
            kernel_grid[:, :, center_y - 2 : center_y + 2 , center_x - 2 : center_x + 2] = 1
            empty_kernels = kernel_grid
        else:
            raise NotImplementedError(
                f"initialization {initialization} is not implemented for DirectCenterKernel network."
            )
        empty_kernels = empty_kernels / ((empty_kernels.sum(dim=(-1, -2), keepdim=True) + 1e-5))
        self.kernels = torch.nn.Parameter(empty_kernels, requires_grad=True)

        distributed.print0("Initialized DirectCenterKernel model:")
        distributed.print0(f"Direct spatially-varying PSF shape:    {self.kernels.shape}")
        distributed.print0(f"Padding:                               {self.padding}")
        distributed.print0(f"Initialization:                        {initialization}")
        distributed.print0()

    def get_kernel(self, img, conditioning: torch.Tensor, sum_to_one=None, **kwargs):
        # conditioning: B, 2(xy), H, W
        if conditioning is None:
            raise ValueError("conditioning cannot be None for `DirectCenterKernel`")
        bs, c, H, W = conditioning.shape
        if c != 2:
            raise ValueError(f"`DirectCenterKernel` expects conditioning with 2 channels but found {conditioning.shape[1]}")

        # Compute the mean (center-coordinate) of x, y values
        centers = conditioning.mean(dim=(-1, -2))  # B, 2
        sum_to_one = sum_to_one if sum_to_one is not None else self.sum_to_one
        interp_kernels = get_psf_at_pos(
            psfs=self.kernels,
            grid=self.psf_grid,
            pos=centers,
            norm_output=False,
        )
        if sum_to_one:
            interp_kernels = interp_kernels.abs()
            interp_kernels = norm_sum_to_one(interp_kernels)

        # kernel needs to be of shape  B, kc, kh, kw
        return interp_kernels


@persistence.persistent_class
class DirectPerPixelKernel(BlurNetBase):
    def __init__(self, psf_path: str, kernel_size: int, padding: str):
        super().__init__(BlurDegradationType.PER_PIXEL_BLUR, padding=padding)
        self.psf = PSF.from_path(
            psf_path=psf_path, filter_size=kernel_size, do_rotation=False
        )
        empty_kernels = torch.randn_like(self.psf.psfs).abs_()
        empty_kernels = norm_sum_to_one(empty_kernels)
        self.initial_param = torch.nn.Parameter(empty_kernels, requires_grad=True)

    def get_kernel(self, img, conditioning, **kwargs):
        # conditioning: B, 2(xy), H, W
        if conditioning is None:
            raise ValueError("conditioning cannot be None for `DirectPerPixelKernel`")
        B, cch, H, W = conditioning.shape
        if cch != 2:
            raise ValueError(f"`DirectPerPixelKernel` expects conditioning with 2 channels but found {conditioning.shape[1]}")

        # reshape such that we have B*H*W, 2 (the points at which we should interpolate)
        # `conditioning` column-contiguous to avoid warning in `get_psf_at_pos`
        positions = conditioning.permute(1, 0, 2, 3).reshape(2, -1).T
        interp_kernels = get_psf_at_pos(
            psfs=self.initial_param,
            grid=self.psf.loc,
            pos=positions,
            norm_output=False,
        )
        # interp_kernels = interp_kernels.clamp_min(0)
        interp_kernels = norm_sum_to_one(interp_kernels, eps=1e-5)
        # kernel needs to be of shape  B, H*W, kc, kh, kw
        interp_kernels = interp_kernels.reshape(B, H*W, *interp_kernels.shape[1:])
        return interp_kernels


@persistence.persistent_class
class ConvCenterKernel(BlurNetBase):
    def __init__(
        self,
        kernel_size: int,
        kernel_ch: int,
        img_ch: int,
        cond_ch: int,
        img_res: int,
        padding: Literal["valid", "replicate"] = "replicate",
        is_output_noisy: bool = True,
        sum_to_one: bool = True,
        learn_output_noise: bool = False,
    ):
        super().__init__(
            BlurDegradationType.BLUR,
            padding=padding,
            is_output_noisy=is_output_noisy,
            learn_output_noise=learn_output_noise
        )
        self.sum_to_one = sum_to_one
        self.cond_ch = cond_ch
        self.img_ch = img_ch
        self.kernel_ch = kernel_ch
        self.kernel_size = kernel_size

        cblock = [32 * i for i in [1, 1, 1, 1]]
        num_blocks = 1

        self.cond_emb = FourierEmbedding(n_channels=32)
        in_ch = 1 + self.cond_emb.n_channels

        self.enc = torch.nn.ModuleDict()
        cout = in_ch
        for level, channels in enumerate(cblock):
            res = img_res >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, 0, flavor='enc', resample_mode='down')
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, 0, flavor='enc')

        self.out_red_ch = nn.Linear(
            cout * (img_res >> (len(cblock) - 1)) * (img_res >> (len(cblock) - 1)),
            256,
            bias=False
        )
        self.out_mid_ch = nn.Linear(256, 256, bias=False)
        self.out_inc_ch = nn.Linear(256, kernel_ch * kernel_size ** 2, bias=False)
        # Note std=0.005 works okay for kernels of size 25,25,3. TODO: Derive a formula for correct initialization (one that makes init ~ sum to 1)
        torch.nn.init.normal_(self.out_red_ch.weight, std=0.005)
        torch.nn.init.normal_(self.out_mid_ch.weight, std=0.005)
        torch.nn.init.normal_(self.out_inc_ch.weight, std=0.005)

    def get_kernel(self, img=None, conditioning=None, sum_to_one=None, **kwargs):
        assert self.cond_ch > 0 and conditioning is not None
        x = center(conditioning)
        x = self.cond_emb(x)
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        # Encoder
        for name, block in self.enc.items():
            x = block(x)
        x = x.view(x.shape[0], -1)
        x = self.out_red_ch(x)
        x = mp_silu(x)
        x = self.out_mid_ch(x)
        x = mp_silu(x)
        kernel = self.out_inc_ch(x)
        kernel = kernel.reshape(kernel.shape[0], self.kernel_ch, self.kernel_size, self.kernel_size)
        # kernel = uncenter(kernel).abs()
        # kernel = kernel.abs()
        sum_to_one = sum_to_one if sum_to_one is not None else self.sum_to_one
        if sum_to_one:
            kernel = norm_sum_to_one(kernel, eps=1e-5).abs()
        return kernel


@persistence.persistent_class
class ConvKernel(BlurNetBase):
    def __init__(self, kernel_size: int, kernel_ch: int, padding: str, factor: int = 1, sum_to_one=True):
        deg_type = BlurDegradationType.BLUR if factor == 1 else BlurDegradationType.DOWNSAMPLING
        super().__init__(deg_type, padding=padding, factor=factor)

        cmid = 64
        self.sum_to_one = sum_to_one
        self.initial_param = nn.Parameter(torch.empty(1, 1, kernel_size * 16, kernel_size * 16))
        self.layers = nn.Sequential(
            MPConv(2, cmid, kernel=[3,3]),
            Block(cmid, cmid, 0, flavor='enc', resample_mode='down'),
            Block(cmid, cmid, 0, flavor='enc', resample_mode='keep', attention=False),
            Block(cmid, cmid, 0, flavor='enc', resample_mode='down'),
            Block(cmid, cmid, 0, flavor='enc', resample_mode='keep', attention=False),
            Block(cmid, cmid, 0, flavor='enc', resample_mode='down'),
            Block(cmid, cmid, 0, flavor='enc', resample_mode='keep', attention=True),
            Block(cmid, cmid, 0, flavor='enc', resample_mode='down'),
            Block(cmid, cmid, 0, flavor='enc', resample_mode='keep', attention=True),
            MPConv(cmid, kernel_ch, kernel=[3, 3])
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.initial_param, std=1)

    def get_kernel(self, *args, **kwargs):
        x = self.initial_param
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        out = self.layers(x)
        # out_shape = out.shape
        # out = out.view(*out_shape[:-3], -1)
        # out = torch.nn.functional.softmax(out, dim=-1)
        # out = out.view(out_shape)
        if self.sum_to_one:
            out = norm_sum_to_one(out)
        return out


@persistence.persistent_class
class ConvSR(nn.Module):
    """
    Blur kernel with optional downsampling, implemented via a deep linear CNN.
    """
    def __init__(
        self,
        factor: int,
        padding: str,
        img_ch: int,
        img_res: int,
        kernel_ch: int,
        kernel_size: int,
        learn_output_noise: bool = False,
        num_kernels: tuple[int, int] | None = None
    ):
        super().__init__()
        self.learn_output_noise = learn_output_noise
        if learn_output_noise:
            # 0.005 is just a random initialization for the noise standard-deviation
            self.sigma = torch.nn.Parameter(torch.tensor([0.005]))
        self.degradation_type = BlurDegradationType.BLUR if factor == 1 else BlurDegradationType.DOWNSAMPLING
        self.factor = factor
        self.padding = padding

        padding_strat = "valid" if padding == "valid" else "same"
        padding_mode = "zeros" if padding == "valid" else padding
        self.kernel_size = kernel_size
        self.kernel_ch = kernel_ch
        if self.kernel_ch != 1 and self.kernel_ch != img_ch:
            raise ValueError("Kernel channels must either be 1 or same as image channels.")

        self.tot_kernels = 1
        if num_kernels is not None:
            self.register_buffer("psf_grid", get_default_psf_grid(num_kernels))
            self.tot_kernels = num_kernels[0] * num_kernels[1]

        groups = self.kernel_ch * self.tot_kernels
        ch = 64 * groups
        # Each channel is processed independently by the same kernel, so we put the
        # channel-dimension of the image into the batch.
        layers = [
            nn.Conv2d(in_channels=self.kernel_ch, out_channels=ch, groups=groups,
                      kernel_size=7, bias=False, padding=padding_strat, padding_mode=padding_mode),
            nn.Conv2d(in_channels=ch, out_channels=ch, groups=groups,
                      kernel_size=5, bias=False, padding=padding_strat, padding_mode=padding_mode),
        ]
        cur_field = 10  # (7-1) + (5-1)
        req_field = self.kernel_size - 1
        while cur_field < req_field:
            if req_field - cur_field == 2:
                layers.append(nn.Conv2d(in_channels=ch, out_channels=ch, groups=groups,
                                        kernel_size=3, bias=False, padding=padding_strat, padding_mode=padding_mode),)
                cur_field += 2
            else:
                layers.append(nn.Conv2d(in_channels=ch, out_channels=ch, groups=groups,
                                        kernel_size=5, bias=False, padding=padding_strat, padding_mode=padding_mode),)
                cur_field += 4
        layers.extend([
            nn.Conv2d(in_channels=ch, out_channels=ch, groups=groups,
                      kernel_size=1, bias=False, padding=padding_strat, padding_mode=padding_mode),
            nn.Conv2d(in_channels=ch, out_channels=groups, groups=groups,
                      kernel_size=1, stride=factor, bias=False, padding=padding_strat, padding_mode=padding_mode),
        ])
        self.layers = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Scale is important: optimization should start from small kernel values.
                nn.init.xavier_normal_(m.weight, 0.1)

        distributed.print0("Initialized ConvSR model:")
        distributed.print0(f"Learning {self.tot_kernels} kernels with {self.kernel_ch} channels.")
        distributed.print0(f"Padding:    {padding_strat}")
        distributed.print0(f"Factor:     {self.factor}")
        distributed.print0()

    def forward(
        self,
        clean_img: torch.Tensor,
        noise_level: Union[float, torch.Tensor] = 0.0,
        class_labels=None,
        conditioning=None,
        get_kernel: bool = False,
        **kwargs
    ):
        if isinstance(noise_level, (float, int)):
            # Necessary to prevent noise level going on CPU
            noise_level = torch.tensor([noise_level], device=clean_img.device)
        B, C, H, W = clean_img.shape
        if self.kernel_ch == 1:
            clean_img = clean_img.view(B * C, 1, H, W)  # merge batch and channel
        else:
            assert clean_img.shape[1] == self.kernel_ch
        out = self.layers(clean_img)
        out = out.reshape(B, C, *out.shape[-2:])  # H, W may have changed

        if self.learn_output_noise:
            noise_level = self.sigma
        out = out + torch.randn_like(out) * noise_level.view(-1, *([1] * (out.ndim - 1)))

        if get_kernel:
            kernels = self.get_kernel(clean_img, conditioning, **kwargs)
            return out, kernels
        return out

    def get_kernel(self, img, conditioning, **kwargs) -> torch.Tensor:
        delta = torch.ones((self.kernel_ch,)).cuda()
        delta = delta[None, :, None, None]
        for i, w in enumerate(self.layers.parameters()):
            if i == 0:
                curr_k = torch.nn.functional.conv2d(delta, w, padding=self.kernel_size - 1, groups=self.kernel_ch * self.tot_kernels)
            else:
                curr_k = torch.nn.functional.conv2d(curr_k, w, groups=self.kernel_ch * self.tot_kernels) # type: ignore
        kernel = curr_k  # type: ignore
        return kernel
