from functools import partial
import os
import pathlib
import sys
import argparse
import deepinv
import tqdm
import yaml
import pickle
import cv2

import numpy as np
import torch
from deepinv.physics import Blur

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from FastDiffusionEM.guided_diffusion.unet import create_model
from FastDiffusionEM.guided_diffusion.gaussian_diffusion import create_sampler
from FastDiffusionEM.util.logger import get_logger
import FastDiffusionEM.util.utils_image as util

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from ddm4ip.degradations.blur import get_motion_blur_kernel

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def pad_kernel(kernel: torch.Tensor, kernel_size) -> torch.Tensor:
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


def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='fastdiffem_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='FastDiffusionEM/configs/diffusion_config_fastem_pigdm.yaml')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='./testset')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--sigma', type=int, default=5)  # 5 / 255 or 10 / 255
    parser.add_argument('--n', type=int, default=1)

    args = parser.parse_args()

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Only model we have is for this kernel size!
    kernel_size = 33

    # Load configurations
    img_model_config = load_yaml(args.img_model_config)
    diffusion_config = load_yaml(args.diffusion_config)

    # Load model
    img_model = create_model(**img_model_config)
    img_model = img_model.to(device)
    img_model.eval()

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=img_model)

    # Load PSF and forward model
    ref_ker = get_motion_blur_kernel(size=32, intensity=0.5, rnd_seed=1)
    ref_ker = pad_kernel(ref_ker, kernel_size)
    sigma = args.sigma / 255
    fwd_model = Blur(
        filter=ref_ker, padding="circular", noise_model=deepinv.physics.GaussianNoise(sigma=sigma)
    )

    # Working directory
    out_path = args.save_dir
    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)

    # set seed for reproduce
    np.random.seed(123)
    torch.manual_seed(123)

    dir = args.input_dir
    paths = sorted(os.listdir(dir))

    for path in tqdm.tqdm(paths):
        if path.endswith(".png"):
            hq = cv2.imread(os.path.join(dir, path), cv2.IMREAD_UNCHANGED)
            hq = hq.astype(np.float32) / 255
            hq = hq[:, :, ::-1]  # bgr2rgb
            hq = hq.transpose(2, 0, 1)
            hq = torch.from_numpy(hq.copy())
            ref_ker = ref_ker
        elif path.endswith(".pickle"):
            with open(os.path.join(dir, path), "rb") as fh:
                data = pickle.load(fh)
            hq = data["H"].squeeze(0)
            ref_ker = data["kernel"]
        else:
            continue

        lq = fwd_model(hq[None, :], filter=ref_ker)
        lq = lq.to(device)

        # Set initial sample
        x_start = torch.randn([args.n, lq.shape[1], lq.shape[2], lq.shape[3]], device=device).requires_grad_()

        # sample
        sample = sample_fn(x_start=x_start, measurement=lq, record=False, save_root=out_path, sigma=sigma, ksize=kernel_size)

        est = sample[0].detach().cpu()[0]
        ker_est = sample[1].detach().cpu()

        # Write
        lq = lq[0].detach().cpu()
        out_img = torch.cat([lq, est, hq], dim=-1)
        out_ker = torch.cat([ker_est[0, 0] / ker_est.max(), ref_ker[0, 0] / ref_ker.max()], dim=-1)
        fname = os.path.split(path)[-1].replace("pickle", "png")
        util.imwrite(util.tensor2uint(out_img), os.path.join(out_path, f"img_{fname}"))
        util.imwrite(util.tensor2uint(out_ker), os.path.join(out_path, f"ker_{fname}"))


if __name__ == '__main__':
    main()
