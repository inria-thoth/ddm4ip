import argparse
from pathlib import Path

import numpy as np
import torchvision.transforms.v2.functional as TF

from ddm4ip.utils.metrics import calc_psnr, calc_ssim, calc_psnr_matlab
from ddm4ip.utils.torch_utils import read_img_pt


def img_num_from_name(img_name: str) -> int:
    spl = img_name.split("_")
    for s in spl:
        try:
            num = int(s)
            return num
        except ValueError:
            pass
    raise ValueError()

def compute_metrics(recon_path: str, gt_path: str, crop_border_pixels: int):
    # Match-up images
    if recon_path.endswith(".png"):
        recon_imgs = {img_num_from_name(Path(recon_path).stem): Path(recon_path)}
    else:
        recon_imgs = Path(recon_path).glob("*.png")
        recon_imgs = {img_num_from_name(img_path.stem): img_path for img_path in recon_imgs}
        recon_imgs = {key: recon_imgs[key] for key in sorted(recon_imgs.keys())}

    if gt_path.endswith(".png"):
        gt_imgs = {img_num_from_name(Path(gt_path).stem): Path(gt_path)}
    else:
        gt_imgs = Path(gt_path).glob("*.png")
        gt_imgs = {img_num_from_name(img_path.stem): img_path for img_path in gt_imgs}
    print(f"Reconstructed images found: {list(recon_imgs.keys())}")
    print(f"Ground-truth images found: {list(gt_imgs.keys())}")
    psnr, ssim, psnr_matlab = [], [], []
    for img_idx, recon_img_path in recon_imgs.items():
        if img_idx not in gt_imgs:
            raise KeyError(f"Image with index {img_idx} exists in reconstructed, not in ground-truth folder")
        recon_img = read_img_pt(recon_img_path)
        gt_img = read_img_pt(gt_imgs[img_idx])
        if crop_border_pixels > 0:
            recon_img = TF.center_crop(
                recon_img, [
                    recon_img.shape[-2] - crop_border_pixels * 2, recon_img.shape[-1] - crop_border_pixels * 2
                ]
            )
            gt_img = TF.center_crop(
                gt_img, [
                    gt_img.shape[-2] - crop_border_pixels * 2, gt_img.shape[-1] - crop_border_pixels * 2
                ]
            )
        psnr.append(calc_psnr(recon_img, gt_img))
        ssim.append(calc_ssim(recon_img, gt_img))
        psnr_matlab.append(calc_psnr_matlab(recon_img, gt_img))
        print(f"Image {img_idx}: PSNR={psnr[-1]:.3f}  SSIM={ssim[-1]:.5f}  MATLAB PSNR={psnr_matlab[-1]:.3f}")
    if len(recon_imgs) > 1:
        avg_ssim = np.mean(ssim)
        avg_psnr = np.mean(psnr)
        avg_psnr_matlab = np.mean(psnr_matlab)
        print(f"Average over {len(recon_imgs)} (cropped {crop_border_pixels} pixels). "
              f"PSNR={avg_psnr:.3f}  SSIM={avg_ssim:.5f}  PSNR_MATLAB={avg_psnr_matlab}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstructed")
    parser.add_argument("--ground-truth")
    parser.add_argument("--crop-border-pixels", type=int, default=0)
    args = parser.parse_args()
    compute_metrics(args.reconstructed, args.ground_truth, args.crop_border_pixels)
