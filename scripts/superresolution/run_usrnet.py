import os
import time
import argparse
from pathlib import Path
import sys

import torch
import torchvision.transforms.v2.functional as TF
import scipy.io

from USRNet.models.network_usrnet_v1 import USRNet
from KernelGAN.util import zeroize_negligible_val, kernel_shift

sys.path.append("../..")
from ddm4ip.utils.torch_utils import read_img_pt, write_img_pt


def preprocess_kernel(kernel):
    kernel = kernel_shift(kernel, sf=2)
    kernel = zeroize_negligible_val(kernel, 40)
    return kernel


def run_usrnet(kernel, image_path, model_path, output_dir, x4: bool, noise_scale: float):
    t_start = time.time()
    image_name = Path(image_path).stem
    sf = 4 if x4 else 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name = Path(model_path).stem
    if "tiny" in model_name:
        model = USRNet(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                       nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    else:
        model = USRNet(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                       nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    model.eval()
    model.to(device)

    kernel = kernel_shift(kernel, sf=2)

    sigma = torch.tensor(noise_scale).float().view([1, 1, 1, 1])
    img = read_img_pt(image_path)[None, ...]  # B, C, H, W
    kernel_pt = torch.from_numpy(kernel)[None, None, ...].float()  # B, 1, H, W

    img_pred = model(img.to(device), kernel_pt.to(device), sf, sigma.to(device)).to("cpu")[0]
    # There's a weird 1-pixel shift!
    img_pred = TF.affine(
        img_pred, angle=0, translate=[1, 1], scale=1, shear=0
    )
    write_img_pt(img_pred, os.path.join(output_dir, "USRNET_%s.png" % (image_name)))
    t_end = time.time()
    print(f"Saved HR image {image_name} with USRNET in {t_end - t_start:.2f}s")


if __name__ == "__main__":
    prog = argparse.ArgumentParser()
    prog.add_argument('--kernel-path', '-k', type=str, help='path to kernel (mat format).')
    prog.add_argument('--image-path', '-i', type=str, help='path to image.')
    prog.add_argument('--model-path', type=str, required=True, help='path to USRNET model (see https://github.com/cszn/USRNet)')
    prog.add_argument('--preprocess', action='store_true', default=False)
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--noise-scale', type=float, default=1./255., help='USRNET uses this to partially de-noise images')
    args = prog.parse_args()

    kernel = scipy.io.loadmat(args.kernel_path)["Kernel"]
    if args.preprocess:
        kernel = preprocess_kernel(kernel)

    run_usrnet(
        kernel=kernel,
        image_path=args.image_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        x4=args.X4,
        noise_scale=args.noise_scale
    )
