import os
import argparse
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pathlib
import sys

from DKP.DIPDKP.DIPDKP.model.util import read_image, im2tensor01, map2tensor, tensor2im01
from DKP.DIPDKP.DIPDKP.config.configs import Config
from DKP.DIPDKP.DIPDKP.model.model import DIPDKP
from DKP.DIPDKP.DIPDKP.Settings import parameters_setting
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve() / "USRNet"))
from USRNet.models.network_usrnet_v1 import USRNet


def train(conf, lr_image, hr_image):
    ''' trainer for DIPDKP, etc.'''
    model = DIPDKP(conf, lr_image, hr_image)
    kernel, sr = model.train()
    return kernel, sr


def create_params(filename, args):
    ''' pass parameters to Config '''
    params = [
        '--model', "DIPDKP",
        '--input_image_path', filename,
        '--real',  # we set this to avoid having to load the GT kernel
        '--sf', args.sf
    ]
    if args.SR:
        params.append('--SR')
    return params


def main():
    prog = argparse.ArgumentParser()
    prog.add_argument('--lr-image', type=str, required=True, help='path to lr image')
    prog.add_argument('--out-path', type=str, required=True)
    prog.add_argument('--sf', type=str, default='4', help='The wanted SR scale factor')
    prog.add_argument('--path-nonblind', type=str, default='../data/pretrained_models/usrnet_tiny.pth',
                    help='path for trained nonblind model')
    prog.add_argument('--SR', action='store_true', default=False, help='when activated - nonblind SR is performed')
    prog.add_argument('--noise-scale', type=float, default=1./255., help='USRNET uses this to partially de-noise images')
    args = prog.parse_args()
    args.dataset = ""  # required by parameter_settings
    args.input_dir = ""  # required by parameter_settings

    I_loop_x = 5
    I_loop_k = 3
    D_loop = 5

    filename = args.lr_image
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)
    conf = Config().parse(create_params(filename, args))
    conf, args = parameters_setting(conf, args, I_loop_x, I_loop_k, D_loop, "DIPDKP", filename, None)
    # For no reason `parameters_setting` overrides stuff, which we re-override here:
    conf.output_dir_path = os.path.abspath(args.out_path)

    # Load low-res image
    lr_image = im2tensor01(read_image(filename)).unsqueeze(0)
    # fake HR image
    hr_image = torch.ones(lr_image.shape[0], lr_image.shape[1], lr_image.shape[2]*int(args.sf), lr_image.shape[3]*int(args.sf))

    # crop the image to 960x960 due to memory limit
    crop_size = 960
    size_min = min(hr_image.shape[2], hr_image.shape[3])
    if size_min > crop_size:
        crop = int(crop_size / 2 / conf.sf)
        lr_image_cropped = lr_image[:, :, lr_image.shape[2] // 2 - crop: lr_image.shape[2] // 2 + crop,
                lr_image.shape[3] // 2 - crop: lr_image.shape[3] // 2 + crop]
        hr_image_cropped = hr_image[:, :, hr_image.shape[2] // 2 - crop * 2: hr_image.shape[2] // 2 + crop * 2,
                hr_image.shape[3] // 2 - crop * 2: hr_image.shape[3] // 2 + crop * 2]
        conf.IF_DIV2K = True
        conf.crop = crop
        print(f"Cropped LR from {lr_image.shape=} to {lr_image_cropped.shape=}")
    else:
        lr_image_cropped = lr_image
        hr_image_cropped = hr_image

    dipdkp_start = time.time()
    kernel, sr_dip = train(conf, lr_image_cropped, hr_image_cropped)
    print(f"Kernel estimation time: {time.time() - dipdkp_start:.2f}s")
    img_name = os.path.split(filename)[-1]
    plt.imsave(
        os.path.join(conf.output_dir_path, f"DIP_{img_name}"), tensor2im01(sr_dip),
        vmin=0, vmax=1., dpi=1
    )

    # nonblind SR
    if args.SR:
        # Load non-blind USRNet model
        netG = USRNet(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        netG.load_state_dict(torch.load(args.path_nonblind), strict=True)
        netG.eval()
        for key, v in netG.named_parameters():
            v.requires_grad = False
        netG = netG.cuda()
        # Run USRNet reconstruction
        sr_start = time.time()
        kernel = map2tensor(kernel)
        sr = netG(
            lr_image,
            torch.flip(kernel, [2, 3]),
            int(args.sf),
            args.noise_scale * torch.ones([1, 1, 1, 1]).cuda()
        )
        plt.imsave(
            os.path.join(conf.output_dir_path, f"USRNET_{img_name}"), tensor2im01(sr),
            vmin=0, vmax=1., dpi=1
        )
        print(f"USRNET SR time: {time.time() - sr_start:.2f}s")


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
