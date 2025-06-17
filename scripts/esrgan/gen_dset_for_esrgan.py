"""
Need two folders, one for HQ and one for LQ.
The LQ data needs to be decimated.
"""
import argparse
import math
from pathlib import Path
import pickle
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm


def pad_div_by(img: torch.Tensor, div_by: int):
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = img.size()
    if (h % div_by != 0):
        mod_pad_h = (div_by - h % div_by)
    if (w % div_by != 0):
        mod_pad_w = (div_by - w % div_by)
    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return img, mod_pad_h, mod_pad_w


def unpad_div_by(img: torch.Tensor, pad_h: int, pad_w: int, scale: int):
    _, _, h, w = img.size()
    assert pad_h % scale == 0
    assert pad_w % scale == 0
    img = img[:, :, 0:h - pad_h // scale, 0:w - pad_w // scale]
    return img


class LQGenerator(torch.nn.Module):
    def __init__(
        self,
        model_scale,
        output_scale,
        model_path,
        device,
        add_noise,
        tile_size=0,
        tile_pad=10,
        pre_pad=10,
    ):
        super().__init__()
        self.model_scale = model_scale
        self.output_scale = output_scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.add_noise = add_noise
        if self.pre_pad % self.output_scale != 0:
            raise ValueError(f"Pre-pad ({self.pre_pad}) must be a multiple of output scale ({self.output_scale})")
        self.device = device

        # Load model
        with open(model_path, "rb") as fh:
            pretr_data = pickle.load(fh)
            kernel_nn = pretr_data["kernel_nn"]
        self.model = kernel_nn.to(device)

    def pre_process(self, img: torch.Tensor) -> torch.Tensor:
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = img.unsqueeze(0).to(self.device)

        # pre_pad
        if self.pre_pad != 0:
            img = F.pad(img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        print(f"After pre-pad: {img.shape=}")
        # mod pad for divisible borders
        img, self.mod_pad_h, self.mod_pad_w = pad_div_by(img, 16)
        print(f"After mod-pad: {img.shape=}")
        return img

    def post_process(self, img: torch.Tensor) -> torch.Tensor:
        print(f"Before post-process: {img.shape=}")
        # decimate
        img = img[:, :, ::self.output_scale, ::self.output_scale]
        print(f"After decimate: {img.shape=}")
        # remove extra pad
        img = unpad_div_by(img, self.mod_pad_h, self.mod_pad_w, self.output_scale)
        print(f"After removing mod-pad: {img.shape=}")
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = img.size()
            img = img[:, :, 0:h - self.pre_pad // self.output_scale, 0:w - self.pre_pad // self.output_scale]
        print(f"After removing pre-pad: {img.shape=}")
        return img

    def process(self, img: torch.Tensor) -> torch.Tensor:
        img, pad_h, pad_w = pad_div_by(img, div_by=16)
        out = self.model(img)
        if self.add_noise > 0:
            out = out + torch.randn_like(out) * self.add_noise
        out = unpad_div_by(out, pad_h, pad_w, scale=self.model_scale)
        return out

    def tile_process(self, img: torch.Tensor) -> torch.Tensor:
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/xinntao/Real-ESRGAN
        """
        batch, channel, height, width = img.shape
        output_height = height * self.model_scale
        output_width = width * self.model_scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.process(input_tile)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.model_scale
                output_end_x = input_end_x * self.model_scale
                output_start_y = input_start_y * self.model_scale
                output_end_y = input_end_y * self.model_scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.model_scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.model_scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.model_scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.model_scale

                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        return output

    def enhance(self, img: np.ndarray, alpha_upsampler='realesrgan'):
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pt = torch.from_numpy(img).permute(2, 0, 1)

        # ------------------- process image (without the alpha channel) ------------------- #
        img_pt = self.pre_process(img_pt)
        if self.tile_size > 0:
            img_pt = self.tile_process(img_pt)
        else:
            img_pt = self.process(img_pt)
        img_pt = self.post_process(img_pt)
        output_img = img_pt.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                alpha_pt = torch.from_numpy(alpha).permute(2, 0, 1)
                alpha_pt = self.pre_process(alpha_pt)
                if self.tile_size > 0:
                    alpha_pt = self.tile_process(alpha_pt)
                else:
                    alpha_pt= self.process(alpha_pt)
                alpha_pt = self.post_process(alpha_pt)
                output_alpha = alpha_pt.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        return output, img_mode


def gen(
    model_path,
    clean_images,
    num_out,
    hq_out_path: Path,
    lq_out_path: Path,
    pre_pad: int,
    tile_size: int,
    tile_pad: int,
    add_noise: float,
):
    hq_out_path.mkdir(parents=True, exist_ok=True)
    lq_out_path.mkdir(parents=True, exist_ok=True)
    if len(list(lq_out_path.glob("*"))) > 0:
        raise ValueError(
            "Low-quality output path is non-empty. Please delete any existing files first by running "
            f"`rm -r '{lq_out_path.resolve()}'`"
        )
    if len(list(hq_out_path.glob("*"))) > 0:
        raise ValueError(
            "High-quality output path is non-empty. Please delete any existing files first by running "
            f"`rm -r '{hq_out_path.resolve()}'`"
        )

    print(f"Generating {num_out} images from {len(clean_images)} clean images...")
    torch.manual_seed(0)
    gen = LQGenerator(
        model_scale=1,
        output_scale=4,
        model_path=model_path,
        tile_size=tile_size,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        add_noise=add_noise,
        device="cuda:0",
    )
    clean_images_iter = iter(clean_images)
    for i in tqdm.tqdm(range(num_out)):
        try:
            clean_img_path = next(clean_images_iter)
        except StopIteration:
            clean_images_iter = iter(clean_images)
            clean_img_path = next(clean_images_iter)

        hq_img = cv2.imread(str(clean_img_path.resolve()), cv2.IMREAD_UNCHANGED)
        with torch.inference_mode():
            lq_img, img_mode = gen.enhance(hq_img)

        cv2.imwrite(str(hq_out_path / f"{i:05d}.png"), hq_img)
        cv2.imwrite(str(lq_out_path / f"{i:05d}.png"), lq_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to the lq-generator model", type=str)
    parser.add_argument("--dset", help="path to the clean image dataset", type=str)
    parser.add_argument("--num", help="number of images to generate", type=int)
    parser.add_argument("--hq_path", help="path to the output HQ images", type=str)
    parser.add_argument("--lq_path", help="path to the output LQ images", type=str)
    parser.add_argument("--pre_pad", help="How much to pre-pad images", type=int, default=10)
    parser.add_argument("--tile_size", help="Size of tiling", type=int, default=128)
    parser.add_argument("--tile_pad", help="How much to pad each tile", type=int, default=10)
    parser.add_argument("--add_noise", type=float, default=0.0)

    args = parser.parse_args()
    clean_images = sorted(list(Path(args.dset).glob("*HR.png")))
    gen(
        model_path=args.model,
        clean_images=clean_images,
        num_out=args.num,
        hq_out_path=Path(args.hq_path),
        lq_out_path=Path(args.lq_path),
        pre_pad=args.pre_pad,
        tile_size=args.tile_size,
        tile_pad=args.tile_pad,
        add_noise=args.add_noise,
    )

"""
PYTHONPATH='/home/gmeanti/inverseproblems' python /home/gmeanti/inverseproblems/scripts/esrgan/gen_dset_for_esrgan.py \
  --model=/scratch/clear/gmeanti/inverseproblems/experiments/diy_realsr-x4-128-canon_lr1e-5x1_1is_xsunet_v4/checkpoints/network-snapshot-1048576.pkl \
  --dset="/scratch/clear/gmeanti/data/RealSR (ICCV2019)/Canon/Train/4" \
  --num=500 \
  --hq_path="/scratch/clear/gmeanti/data/RealSR (ICCV2019)/Canon_x4_generated_1/hq" \
  --lq_path="/scratch/clear/gmeanti/data/RealSR (ICCV2019)/Canon_x4_generated_1/lq" \
  --tile_size=1024 \
  --tile_pad=8 \
  --pre_pad=8
"""