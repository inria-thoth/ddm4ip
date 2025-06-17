from collections.abc import Callable
from io import BytesIO
import math
from pathlib import Path
from typing import IO, Generator, Sequence, Tuple, TypeVar
import warnings

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.v2.functional as TF
import cv2


FT = TypeVar("FT", float, torch.Tensor)

def center(img: FT, max_val: float = 1.0) -> FT:
    # [0, max_val] -> [-max_val, max_val]
    return img * 2 - max_val


def uncenter(img: FT, max_val: float = 1.0) -> FT:
    # [-max_val, max_val] -> [0, max_val]
    out = (img + max_val) * 0.5
    return out


def set_random_seed(*args):
    seed = hash(args) % (1 << 31)
    torch.manual_seed(seed)
    np.random.seed(seed)


#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name])


#----------------------------------------------------------------------------
#

T = TypeVar("T", torch.Tensor, np.ndarray)

def equate_kernel_shapes(
    k1: T,
    k2: T
) -> Tuple[T, T]:
    if k1.ndim < 2:
        raise ValueError(f"Kernel k1 has {k1.ndim} dimensions, but expected 2. Shape: {k1.shape}")
    if k2.ndim < 2:
        raise ValueError(f"Kernel k2 has {k2.ndim} dimensions, but expected 2. Shape: {k2.shape}")
    if k1.ndim == 5 and k2.ndim == 4:
        # need to take center kernel for k1
        k1 = k1[:, k1.shape[1] // 2]
    elif k2.ndim == 5 and k1.ndim == 4:
        # need to take center kernel for k2
        k2 = k2[:, k2.shape[1] // 2]
    if k1.ndim != k2.ndim:
        raise ValueError(f"Kernels k1 and k2 must have the same number of dimensions. Found {k1.ndim} and {k2.ndim}")
    for dim in range(k1.ndim - 2):
        if k1.shape[dim] != k2.shape[dim]:
            raise ValueError(f"Kernels must have the same shape apart for the last two dimensions. "
                             f"Found shapes {k1.shape[dim]} and {k2.shape[dim]} at dimension {dim}")
    if k1.shape != k2.shape:
        # Expand the smaller kernel to the same size as the larger kernel with zeros
        k1_shp = k1.shape
        k2_shp = k2.shape
        if k1_shp[-1] < k2_shp[-1] and k1_shp[-2] > k2_shp[-2]:
            raise ValueError(
                f"Kernel k1 must be smaller or larger than k2"
                f" in both dimensions. Found k1 of shape {k1.shape}"
                f" and k2 of shape {k2.shape}"
            )
        if k2_shp[-1] < k1_shp[-1] and k2_shp[-2] > k1_shp[-2]:
            raise ValueError(
                f"Kernel k1 must be smaller or larger than k2"
                f" in both dimensions. Found k1 of shape {k1.shape}"
                f" and k2 of shape {k2.shape}"
            )
        revert = False
        if k2_shp[-1] < k1_shp[-1]:
            k1, k2 = k2, k1
            k1_shp, k2_shp = k2_shp, k1_shp
            revert = True
        # Now k1 < k2
        dim0_diff = k2_shp[-1] - k1_shp[-1]
        dim1_diff = k2_shp[-2] - k1_shp[-2]
        pad = [
            (dim0_diff // 2, dim0_diff // 2 + dim0_diff % 2),
            (dim1_diff // 2, dim1_diff // 2 + dim1_diff % 2),
        ]
        if isinstance(k1, torch.Tensor):
            # torch pad only pads the last dimensions so this should be good
            k1 = torch.nn.functional.pad(k1, [*pad[0], *pad[1]], value=0)
        else:
            np_pad = [(0, 0) for _ in range(k1.ndim - 2)] + pad
            k1 = np.pad(k1, np_pad, constant_values=0)
        if revert:
            k1, k2 = k2, k1
    return k1, k2


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, start_idx=0, is_infinite: bool = True):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        warnings.filterwarnings('ignore', '`data_source` argument is not used and will be removed')
        super().__init__(dataset)
        self.dataset_size = len(dataset)
        self.start_idx = start_idx + rank
        self.stride = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.is_infinite = is_infinite

    def __iter__(self):
        idx = self.start_idx
        epoch = None
        while True:
            if epoch != idx // self.dataset_size:
                epoch = idx // self.dataset_size
                order = np.arange(self.dataset_size)
                if self.shuffle:
                    np.random.RandomState(hash((self.seed, epoch)) % (1 << 31)).shuffle(order)
            assert epoch is not None
            if epoch > 0 and not self.is_infinite:
                break
            yield int(order[idx % self.dataset_size])
            idx += self.stride

    def __len__(self) -> int:
        if self.is_infinite:
            return 1_000_000_000_000_000
        return self.dataset_size


def crop_valid(img: torch.Tensor, kernel_size: int | tuple[int, int], dims: tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """Crop an image to the portion which is 'valid' when convolved with kernel of size `kernel_size`
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if kernel_size[0] == 0 and kernel_size[1] == 0:
        return img
    crop = (
        (math.ceil((kernel_size[0] + 1) / 2 - 1), img.shape[dims[0]] - (kernel_size[0] - 1)),
        (math.ceil((kernel_size[1] + 1) / 2 - 1), img.shape[dims[1]] - (kernel_size[1] - 1))
    )
    return img.narrow(dims[0], crop[0][0], crop[0][1]).narrow(dims[1], crop[1][0], crop[1][1])


def pad_valid(img: torch.Tensor, kernel_size: int | tuple[int, int]) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    # padding is specified on width then height (different order)
    pad = (
        math.ceil((kernel_size[1] + 1) / 2 - 1), math.floor((kernel_size[1] + 1) / 2 - 1),
        math.ceil((kernel_size[0] + 1) / 2 - 1), math.floor((kernel_size[0] + 1) / 2 - 1),
    )
    return torch.nn.functional.pad(img, pad, mode="constant", value=0)


def pad_kernel(kernel: torch.Tensor, kernel_size: int | Tuple[int, int]) -> torch.Tensor:
    """
    Pad a kernel with zeros until it reaches the desired size while
    keeping the original kernel in the center.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if kernel.shape[-2] > kernel_size[0] or kernel.shape[-1] > kernel_size[1]:
        raise ValueError(
            f"Cannot pad kernel. Kernel larger than desired size. "
            f"Found kernel of shape {kernel.shape} and desired size {kernel_size}."
        )
    pad = (
        (kernel_size[0] - kernel.shape[-2]) // 2,
        (kernel_size[0] - kernel.shape[-2]) // 2 + (kernel_size[0] - kernel.shape[-2]) % 2,
        (kernel_size[1] - kernel.shape[-1]) // 2,
        (kernel_size[1] - kernel.shape[-1]) // 2 + (kernel_size[1] - kernel.shape[-1]) % 2,
    )
    return torch.nn.functional.pad(kernel, pad, value=0)


def get_center_kernel(kernels: torch.Tensor, img_h: int | None, img_w: int | None) -> torch.Tensor:
    if kernels.dim() == 5:
        img_hw = kernels.shape[1]
        if img_h is not None and img_w is not None:
            pass
        elif img_h is None and img_w is None:
            img_h = img_w = int(math.sqrt(img_hw))
        elif img_w is None:
            assert img_h is not None
            img_w = img_hw // img_h
        else:
            assert img_h is None and img_w is not None
            img_h = img_hw // img_w
        assert img_hw == img_h * img_w, f"{img_hw=} {img_h=} {img_w=}"

        kernels = kernels.view(-1, img_h, img_w, *kernels.shape[2:])
        kernels = kernels[:, img_h // 2, img_w // 2, ...]
    return kernels


# ----------------
# --- Image IO ---
# ----------------

def read_img_pt(path: str | Path | IO[bytes]) -> torch.Tensor:
    if isinstance(path, (str, Path)):
        img_np = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        data = path.read()
        img_np = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img_np)
    img = img.permute(2, 0, 1)
    img = TF.to_dtype(img, torch.float32, scale=True)
    return img


def write_img_pt(img: torch.Tensor, path: str | Path | BytesIO) -> bool:
    img = img.clip(0, 1)
    img = TF.to_dtype(img, torch.uint8, scale=True)
    img = img.flip(dims=(0, ))  # RGB->BGR
    img = img.permute(1, 2, 0)
    img_np = img.numpy(force=True)
    if isinstance(path, BytesIO):
        is_success, buffer = cv2.imencode(".png", img_np)
        path.write(buffer.tobytes())
        return is_success
    return cv2.imwrite(str(path), img_np)


# -------------------------------------
# --- Full images and image patches ---
# -------------------------------------

def pad_image_to_size(image, final_height, final_width, mode: str="replicate", value=None):
    image_height, image_width = image.shape[-2:]
    pad_params = [
        (final_width - image_width) // 2 if final_width > image_width else 0,
        (final_width - image_width + 1) // 2 if final_width > image_width else 0,
        (final_height - image_height) // 2 if final_height > image_height else 0,
        (final_height - image_height + 1) // 2 if final_height > image_height else 0,
    ]
    return torch.nn.functional.pad(image, pad_params, mode=mode, value=value)


def img2patches(img: torch.Tensor, patch_size: int, stride: int) -> list[torch.Tensor]:
    overlap = patch_size - stride
    img = pad_image_to_size(img, img.shape[-2] + overlap, img.shape[-1] + overlap)
    h, w = img.shape[-2:]
    w1 = list(np.arange(0, w-patch_size, stride, dtype=int))
    h1 = list(np.arange(0, h-patch_size, stride, dtype=int))
    w1.append(w-patch_size)
    h1.append(h-patch_size)
    patches = []
    for i in h1:
        for j in w1:
            patches.append(img[..., i:i+patch_size, j:j+patch_size].contiguous())
    return patches


def patches2img(
        patches: Sequence[torch.Tensor],
        stride: int, imgh: int, imgw: int, func: Callable[[torch.Tensor], torch.Tensor] | None) -> torch.Tensor:
    num_patches = len(patches)
    C, patch_size, _ = patches[0].shape
    dt, dev = patches[0].dtype, patches[0].device
    overlap = patch_size - stride
    w1 = list(np.arange(0, imgw-(patch_size - overlap), stride, dtype=int))
    h1 = list(np.arange(0, imgh-(patch_size - overlap), stride, dtype=int))
    w1.append(imgw-(patch_size - overlap))
    h1.append(imgh-(patch_size - overlap))
    assert num_patches == len(w1) * len(h1)
    img = torch.empty((C, imgh, imgw), dtype=dt, device=dev)
    pid = 0
    if func is None:
        func = lambda x: x
    for i in h1:
        for j in w1:
            cpatch = func(patches[pid])  # C, pS, pS
            img[..., i: i + (patch_size - overlap), j: j + (patch_size - overlap)].copy_(cpatch)
            pid += 1
    return img
