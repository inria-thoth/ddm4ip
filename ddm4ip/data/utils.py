import torch
import torchvision.transforms.v2 as v2

from typing import Any, Dict, List


def get_dim_blocks(dim_in, kernel_size, padding=0, stride=1, dilation=1) -> int:
    return (dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def extract_patches(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1
) -> torch.Tensor:
    """Extract all patches from an image.

    Args:
        x: input image. Tensor of shape [C, H, W]
        kernel_size: size of the  patches to be extracted (kH, kW)
        stride: Stride of the sliding window. If equal to kernel_size, patches
            won't overlap. Defaults to 1.
        dilation: Controls the "stride" within each patch. e.g. if dilation is 2
            only one every two pixels will be taken from the input image.
            Defaults to 1.

    Returns:
        torch.Tensor: Patches tensor of shape [nH, nW, C, kH, kW] where nH and nW
            are the number of patches in vertical and horizontal axes.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    channels, height, width = x.shape[-3:]
    h_blocks = get_dim_blocks(height, kernel_size=kernel_size[0], stride=stride[0], dilation=dilation[0])
    w_blocks = get_dim_blocks(width, kernel_size=kernel_size[1], stride=stride[1], dilation=dilation[1])
    shape = (channels, h_blocks, w_blocks, kernel_size[0], kernel_size[1])
    strides = (
        width*height,
        stride[0]*width,
        stride[1],
        dilation[0]*width,
        dilation[1]
    )
    x = x.contiguous()
    x = x.as_strided(shape, strides)
    x = x.permute(1, 2, 0, 3, 4)
    return x


class GetAllPatchesTransform(v2.Transform):
    def __init__(self, patch_size: int, stride: int | None = None):
        super().__init__()
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # patches: H-blocks, W-blocks, Channels, H, W
        patches = extract_patches(inpt, self.patch_size, stride=self.stride)
        # merge the first two dimensions to get standard B,C,H,W
        return patches.reshape(-1, patches.shape[2], patches.shape[3], patches.shape[4])


class TopLeftCrop(v2.Transform):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return v2.functional.crop(inpt, top=0, left=0, height=self.patch_size, width=self.patch_size)


def get_locmap(height, width):
    map_w = torch.linspace(0, 1, width, dtype=torch.float32)
    map_w = map_w[None, :].repeat(height, 1)
    map_h = torch.linspace(0, 1, height, dtype=torch.float32)
    map_h = map_h[:, None].repeat(1, width)
    return torch.cat([map_h[None, ...], map_w[None, ...]], dim=-3)



class AddLocMapTransform(v2.Transform):
    def __init__(self):
        super().__init__()

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        height, width = inpt.shape[-2], inpt.shape[-1]
        assert inpt.dtype == torch.float32
        locmap = get_locmap(height, width)
        return torch.cat([inpt, locmap], dim=-3)


class ReplaceRandomLocMapTransform(v2.Transform):
    def __init__(self, p: float, full_img_size: tuple[int, int]):
        super().__init__()
        self.p = p
        self.full_img_size = full_img_size
        self.full_map = get_locmap(full_img_size[0], full_img_size[1])

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if torch.rand(1) >= self.p:
            return inpt
        ch, height, width = inpt.shape[-3], inpt.shape[-2], inpt.shape[-1]
        # Replace the old map with a new one
        inpt = inpt[..., :ch-2, :, :]
        top = torch.randint(0, self.full_img_size[0] - height, size=(1, ))
        left = torch.randint(0, self.full_img_size[1] - width, size=(1, ))
        sub_map = self.full_map[:, top:top + height, left: left + width]
        if inpt.dim() == 4:
            sub_map = sub_map.expand(inpt.shape[0], *sub_map.shape)
        return torch.cat([inpt, sub_map], dim=-3)


class AddRandomLocMapTransform(v2.Transform):
    def __init__(self, full_img_size: tuple[int, int]):
        super().__init__()
        self.full_img_size = full_img_size
        self.full_map = get_locmap(full_img_size[0], full_img_size[1])

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        height, width = inpt.shape[-2], inpt.shape[-1]
        top = torch.randint(0, self.full_img_size[0] - height, size=(1, ))
        left = torch.randint(0, self.full_img_size[1] - width, size=(1, ))
        sub_map = self.full_map[:, top:top + height, left: left + width]
        if inpt.dim() == 4:
            sub_map = sub_map.expand(inpt.shape[0], *sub_map.shape)
        return torch.cat([inpt, sub_map], dim=-3)


class SeededTransform(v2.Transform):
    """A transform which uses the same random numbers
    when called the same number of times in two separate instances.
    Used for
    """
    def __init__(self, init_seed, transforms):
        super().__init__()
        self.init_seed = init_seed
        self.transforms = transforms
        self.num_called = 0

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self.transforms._get_params(flat_inputs)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # Assumes we're doing this on CPU
        orig_state = torch.default_generator.get_state()
        try:
            torch.default_generator.manual_seed(
                hash((self.init_seed << 32) + self.num_called)
            )
            return self.transforms(inpt)
        finally:
            torch.default_generator.set_state(orig_state)
            self.num_called += 1


def get_space_varying_patches(img_size: int | tuple[int, int], patch_size: int) -> torch.Tensor:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    full_img = AddLocMapTransform()(torch.zeros(1, img_size[1], img_size[0]))[1:]
    # all_patches: nH*nW, 2, patch_size, patch_size
    all_patches = GetAllPatchesTransform(patch_size, stride=patch_size)(full_img)
    n_w_patches = get_dim_blocks(img_size[1], patch_size, stride=patch_size)
    # resize to nH, nW, 2, patch_size, patch_size
    return all_patches.view(-1, n_w_patches, *all_patches.shape[1:])