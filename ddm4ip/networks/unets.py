# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import numpy as np
import torch
from ddm4ip.utils import distributed, persistence
from ddm4ip.utils.torch_utils import center

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


#----------------------------------------------------------------------------
# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.

def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)


#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

@persistence.persistent_class
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, groups=1):
        super().__init__()
        self.out_channels = out_channels
        self.groups = groups
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        # if self.training and torch.is_grad_enabled():  # GIACOMO: This allows for CT
        with torch.no_grad():
            self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,), groups=self.groups)

#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

@persistence.persistent_class
class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        if emb_channels > 0:
            self.emb_gain = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        else:
            self.emb_gain = None
            self.emb_linear = None
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb=None):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        if self.emb_linear is not None:
            assert self.emb_gain is not None
            assert emb is not None
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        else:
            y = mp_silu(y)
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            assert self.attn_qkv is not None and self.attn_proj is not None
            y = self.attn_qkv(x)
            if True:
                y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
                q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
                w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
                y = torch.einsum('nhqk,nhck->nhcq', w, v)
            else:
                y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3]).permute(0, 1, 4, 3, 2)
                q, k, v = normalize(y, dim=4).unbind(3) # pixel norm & split
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v
                )
                y = y.permute(0, 1, 3, 2)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        out_channels        = 3,
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None
        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        if out_channels is None:
            out_channels = img_channels
        self.out_conv = MPConv(cout, out_channels, kernel=[3,3])
        self.register_load_state_dict_pre_hook(self.allow_adding_input_channels)

    def allow_adding_input_channels(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        sd_key = f"{prefix}enc.{self.img_resolution}x{self.img_resolution}_conv.weight"
        sd_img_channels = state_dict[sd_key].shape[1] - 1
        if self.img_channels == sd_img_channels:
            pass
        elif self.img_channels < sd_img_channels:
            num_remove_ch = sd_img_channels - self.img_channels
            distributed.print0(f"Removing {num_remove_ch} channels from loaded state-dict")
            # shape: out_channels, in_channels, *kernel_size
            # removing channels from `in_channels` (remove from the end)
            old_tensor = state_dict[sd_key]
            new_tensor = old_tensor[:, :-num_remove_ch]
            state_dict[sd_key] = new_tensor
        else:
            num_new_channels = self.img_channels - sd_img_channels
            distributed.print0(f"Adding {num_new_channels} channels to loaded state-dict")
            # shape: out_channels, in_channels, *kernel_size
            # new-shape adds channels to in_channels
            old_tensor = state_dict[sd_key]
            new_tensor = torch.cat([
                old_tensor,
                torch.randn([old_tensor.shape[0], num_new_channels] + list(old_tensor.shape[2:]), )
            ], dim=1)
            state_dict[sd_key] = new_tensor

    def forward(self, x, noise_labels, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x

#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

@persistence.persistent_class
class Precond(torch.nn.Module):
    def __init__(self,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        label_dim,              # Class label dimensionality. 0 = unconditional.
        cond_channels   = 0,    # Number of channels for conditioning
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.cond_channels = cond_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = UNet(
            img_resolution=img_resolution,
            img_channels=img_channels + cond_channels,
            label_dim=label_dim,
            **unet_kwargs
        )
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, class_labels=None, conditioning=None, force_fp32=False, return_logvar=False, **unet_kwargs):
        x = x.to(torch.float32)
        if conditioning is not None:
            conditioning = conditioning.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
        if conditioning is not None:
            x_in = torch.cat([c_in * x, conditioning], dim=-3).to(dtype)
        else:
            x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, c_noise, class_labels, **unet_kwargs)
        if conditioning is not None:
            F_x = F_x[:, :self.img_channels]
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x


@persistence.persistent_class
class RFNoPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution: int | tuple[int, int], # Image resolution.
        img_channels,           # Image channels.
        label_dim,              # Class label dimensionality. 0 = unconditional.
        cond_channels   = 0,    # Number of channels for conditioning
        use_fp16        = True, # Run the model at FP16 precision?
        use_logvar      = False,  # ignored
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        assert use_logvar is False
        if isinstance(img_resolution, (tuple, list)):
            assert len(img_resolution) == 2, f"Found invalid resolution {img_resolution}"
            assert img_resolution[0] == img_resolution[1], f"Found invalid resolution {img_resolution}"
            img_resolution = img_resolution[0]
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.cond_channels = cond_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.unet = UNet(
            img_resolution=img_resolution,
            img_channels=img_channels + cond_channels,
            label_dim=label_dim,
            **unet_kwargs
        )

        distributed.print0("Initialized RFNoPrecond model:")
        distributed.print0(f"Input channels:               {self.img_channels}")
        distributed.print0(f"Conditioning channels:        {self.cond_channels}")
        distributed.print0(f"Image resolution:             {self.img_resolution}")
        distributed.print0(f"Class labels:                 {self.label_dim}")
        distributed.print0()

    def forward(self, x, sigma, class_labels=None, conditioning=None, force_fp32=False, **unet_kwargs):
        x = x.to(torch.float32)
        if conditioning is not None and self.cond_channels > 0:
            conditioning = conditioning.to(torch.float32)
        else:
            conditioning = None
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        if class_labels is None:
            if self.label_dim != 0:
                class_labels = torch.zeros([1, self.label_dim], device=x.device)
        else:
            class_labels = class_labels.to(torch.float32)

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_noise = (sigma * 2 - 1).flatten()  # approximately match range of EDM?

        # Run the model.
        if conditioning is not None:
            conditioning = center(conditioning) # CAREFUL! Expectes conditioning between 0, 1
            x = torch.cat([x, conditioning], dim=-3)
        x_in = (x).to(dtype)
        F_x = self.unet(x_in, c_noise, class_labels, **unet_kwargs)
        if conditioning is not None:
            F_x = F_x[:, :self.img_channels]
        v_x =  F_x.to(torch.float32)

        return v_x

    def velocity(self, x, sigma, class_labels=None, force_fp32=False, **unet_kwargs):
        return self.forward(x, sigma, class_labels, force_fp32, **unet_kwargs)

    def score(self, x, sigma, class_labels=None, force_fp32=False, **unet_kwargs):
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        # We need to force sigma to be slightly far from 1, for numerical stability.
        sigma = sigma.clamp(0, 1 - 1e-3)
        velocity = self.velocity(x, sigma, class_labels, force_fp32, **unet_kwargs)
        return x / (1 - sigma) + (sigma / (1 - sigma)) * velocity

    def denoise(self, x, sigma, class_labels=None, force_fp32=False, **unet_kwargs):
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        velocity = self.velocity(x, sigma, class_labels, force_fp32, **unet_kwargs)
        x1 = (1 - sigma) * velocity + x
        return x1


def init_unet_with_defaults(cfg, img_size: tuple[int, int, int], cond_ch: int, label_dim: int = 0):
    if "channels" not in cfg:
        raise KeyError("'channels' must be specified when initializing a UNet model.")
    kind = cfg.get("kind", "flow")
    if kind == "flow":
        return RFNoPrecond(
            img_resolution=cfg.get("img_res", img_size[1:]),
            img_channels=img_size[0],
            cond_channels=cond_ch,
            label_dim=label_dim,
            use_fp16=cfg.get("use_fp16", True),
            model_channels=cfg["channels"],
            channel_mult=cfg.get("channel_mult", [1, 2, 4]),
            num_blocks=cfg.get("num_blocks", 2),
            attn_resolutions=cfg.get("attn_resolutions", [8, 4]),
            dropout=cfg.get("dropout", 0.0),
            channels_per_head=cfg.get("channels_per_head", 16),
            out_channels=cfg.get("out_channels", min(img_size[0], 3)),
        )
    else:
        raise ValueError(
            f"Model kind '{kind}' invalid. Valid values are 'flow' and 'edm2'."
        )


@persistence.persistent_class
class CrossAttentionLayer(torch.nn.Module):
    pass