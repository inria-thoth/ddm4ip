import deepinv
import numpy
import torch
import torchvision.transforms.v2.functional as TF

from .degradation import fix_deepinv_state


def get_motion_blur_kernel(size: int, intensity: float, rnd_seed: int):
    from .motion_blur import Kernel
    numpy.random.seed(rnd_seed)
    mb_ker_obj = Kernel(size=(size, size), intensity=intensity)
    mb_ker = torch.from_numpy(mb_ker_obj.kernelMatrix).to(dtype=torch.float32)
    return mb_ker.view(1, 1, size, size)


class Blur(deepinv.physics.Blur):
    """Wraps deepinv `Blur`"""
    def get_kernel(self, img=None, conditioning=None):
        return self.filter

    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d


class Pad(torch.nn.Module):
    """Not a real degradation, applied 'valid' padding to the input image"""
    def __init__(self, size: int, padding: str):
        super().__init__()
        self.size = size
        self.padding = padding

    def forward(self, x, **kwargs):
        if self.padding == "valid":
            return TF.center_crop(
                x, [x.shape[-2] - (self.size - 1), x.shape[-1] - (self.size - 1)]
            )
        else:
            raise NotImplementedError(f"Padding mode {self.padding} not implemented.")