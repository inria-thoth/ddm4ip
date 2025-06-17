import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
sys.path.append(str(pathlib.Path(__file__).parent.resolve() / "FKP"))
sys.path.append(str(pathlib.Path(__file__).parent.resolve() / "FKP" / "DIPFKP"))
from FKP.DIPFKP.util import read_image, im2tensor01, map2tensor, tensor2im01
from FKP.DIPFKP.config.configs import Config
from FKP.DIPFKP.model.model import DIPFKP
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve() / "USRNet"))
from USRNet.models.network_usrnet_v1 import USRNet


def train(conf, lr_image):
    model = DIPFKP(conf, lr_image)
    kernel, sr = model.train()
    return kernel, sr


def create_params(filename, args):
    params = [
        '--model', "DIPFKP",
        '--input_image_path', filename,
        '--output_dir_path', os.path.abspath(args.output_dir),
        '--path_KP', os.path.abspath(args.path_KP),
        '--sf', args.sf,
        '--real'
    ]
    if args.SR:
        params.append('--SR')
    return params


def main():
    prog = argparse.ArgumentParser()
    prog.add_argument('--sf', type=str, default='2', help='The wanted SR scale factor')
    prog.add_argument('--path-nonblind', type=str, default='../data/pretrained_models/usrnet_tiny.pth',
                      help='path for trained nonblind model')
    prog.add_argument('--SR', action='store_true', default=False, help='when activated - nonblind SR is performed')
    prog.add_argument('--path-KP', type=str, default='../data/pretrained_models/FKP_x2.pt',
                      help='path for trained kernel prior')
    prog.add_argument('--lr-image', type=str, required=True, help='path to lr image')
    prog.add_argument('--output-dir', '-o', type=str, required=True, help='path to image output directory')
    prog.add_argument('--noise-scale', type=float, default=1./255., help='USRNET uses this to partially de-noise images')
    args = prog.parse_args()

    filename = args.lr_image
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    # Create configuration object. This goes through a multiple layers of indirection.
    conf = Config().parse(create_params(filename, args))

    # Load low-resolution image
    lr_image = im2tensor01(read_image(filename)).unsqueeze(0)

    # crop the image to 960x960 due to memory limit
    if True:
        crop = int(960 / 2 / conf.sf)
        cropped_lr_image = lr_image[:, :, lr_image.shape[2] // 2 - crop: lr_image.shape[2] // 2 + crop,
                    lr_image.shape[3] // 2 - crop: lr_image.shape[3] // 2 + crop]

    # Run DIP-FKP, save output
    kernel, sr_dip = train(conf, cropped_lr_image)
    img_name = os.path.split(filename)[-1]
    plt.imsave(
        os.path.join(conf.output_dir_path, f"FKP_{img_name}"), tensor2im01(sr_dip),
        vmin=0, vmax=1., dpi=1
    )

    # nonblind SR
    if args.SR:
        # Load non-blind model
        netG = USRNet(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                      nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        netG.load_state_dict(torch.load(args.path_nonblind), strict=True)
        netG.eval()
        for key, v in netG.named_parameters():
            v.requires_grad = False
        netG = netG.cuda()

        # Run non-blind and save output
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


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
