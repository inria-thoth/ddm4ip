import os
import time
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.io

from KernelGAN.ZSSRforKernelGAN.ZSSR import ZSSR
from KernelGAN.util import analytic_kernel, zeroize_negligible_val, kernel_shift


def preprocess_kernel(kernel):
    significant_k = zeroize_negligible_val(kernel, 60)
    centralized_k = kernel_shift(significant_k, sf=2)
    return centralized_k


def run_zssr(
    k_2,
    input_image_path: str,
    output_dir: str,
    x4: bool,
    real_image: bool,
    noise_scale: float,
    preprocess: bool,
):
    """Performs ZSSR with estimated kernel for wanted scale factor"""
    if preprocess:
        k_2 = preprocess_kernel(k_2)
    start_time = time.time()
    print('~' * 30 + '\nRunning ZSSR X%d...' % (4 if x4 else 2))
    if x4:
        sr = ZSSR(input_image_path, scale_factor=[[2, 2], [4, 4]], kernels=[k_2, analytic_kernel(k_2)],
                  is_real_img=real_image, noise_scale=noise_scale).run()
    else:
        sr = ZSSR(input_image_path, scale_factor=2, kernels=[k_2], is_real_img=real_image, noise_scale=noise_scale).run()
    max_val = 255 if sr.dtype == 'uint8' else 1.
    img_name = Path(input_image_path).stem
    plt.imsave(os.path.join(output_dir, 'ZSSR_%s.png' % img_name), sr, vmin=0, vmax=max_val, dpi=1)
    runtime = int(time.time() - start_time)
    print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)


if __name__ == "__main__":
    prog = argparse.ArgumentParser()
    prog.add_argument('--kernel-path', '-k', type=str, help='path to kernel (mat format).')
    prog.add_argument('--image-path', '-i', type=str, help='path to image.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--preprocess', action='store_true', default=False)
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    args = prog.parse_args()

    kernel = scipy.io.loadmat(args.kernel_path)["Kernel"]
    run_zssr(kernel, args.image_path, args.output_dir, args.X4, args.real, args.noise_scale, args.preprocess)
