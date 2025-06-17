from pathlib import Path
import argparse
import shutil
import sys
import zipfile

import torch
import tqdm

from ddm4ip.degradations.varpsf import PerPixelBlur
from ddm4ip.psf.psf import PSF, get_psf_at_pos
from ddm4ip.utils.torch_utils import read_img_pt, write_img_pt

"""
For DDPD we need:
 1. clean training
 2. clean validation
 3. training-set corrupted with the ground-truth space-varying PSF. Needed to run the non-blind ESRGAN.
"""


def extract(zip_file: Path, out_path: Path):
    # Take train_c/target for training and test_c/target for testing
    with zipfile.ZipFile(zip_file, mode="r") as zf:
        all_fnames = zf.namelist()
        train_fnames = [
            fn for fn in all_fnames if fn.startswith("dd_dp_dataset_png/train_c/target") and fn.endswith(".png")
        ]
        test_fnames = [
            fn for fn in all_fnames if fn.startswith("dd_dp_dataset_png/test_c/target") and fn.endswith(".png")
        ]
        train_out_path = out_path / "train"
        test_out_path = out_path / "test"
        train_out_path.mkdir(exist_ok=True, parents=True)
        test_out_path.mkdir(exist_ok=True, parents=True)

        for tr_fn in train_fnames:
            target_path = train_out_path / tr_fn.split("/")[-1]
            with zf.open(tr_fn) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
        print(f"Extracted DDPD training dataset: {len(train_fnames)} images to {train_out_path}")
        for ts_fn in test_fnames:
            target_path = test_out_path / ts_fn.split("/")[-1]
            with zf.open(ts_fn) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
        print(f"Extracted DDPD testing dataset: {len(test_fnames)} images to {test_out_path}")

def corrupt_img(img: torch.Tensor, psf: PSF):
    blur = PerPixelBlur(padding="replicate")
    kernels = get_psf_at_pos(
        psfs=psf.psfs,
        grid=psf.loc,
        pos=torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, img.shape[-2]),
                torch.linspace(0, 1, img.shape[-1]),
                indexing='ij'
            ), 0
        ).view(2, -1).T
    )
    kc, kh, kw = kernels.shape[1:4]
    kernels = kernels.reshape(1, img.shape[-2] * img.shape[-1], kc, kh, kw)
    out = blur.A(img[None, ...], filters=kernels, sigma=0)[0]
    return out

def corrupt_trset(data_path: Path, out_path: Path, psf_path: Path):
    corrupted_out_path = out_path / "train_corrupted"
    corrupted_out_path.mkdir(exist_ok=True)
    psf = PSF.from_path(
        psf_path=psf_path, do_rotation=False
    )
    for fname in tqdm.tqdm(list(data_path.glob("*.png")), desc="Corrupting training set"):
        img = read_img_pt(fname)
        corr_img = corrupt_img(img, psf)
        write_img_pt(corr_img, corrupted_out_path / fname.name)

def run(argv):
    parser = argparse.ArgumentParser(argv[0], description="Process the DDPD dataset for Diff4IP training. Note that the script needs around 32GB of RAM to complete the corruption process successfully.")
    parser.add_argument("--input-path", help="Path to original DDPD zip-file (`dd_pd_dataset_canon.zip`). This can be downloaded following the instructions here: https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel.")
    parser.add_argument("--output-path", help="A directory where all FFHQ subsets (at 256x256 resolution) needed for Diff4IP will be placed")
    parser.add_argument("--psf", help="Path to the space-varying PSF. Generate this using the `get_spacevarying_psf.py` script", required=False, defualt=None)
    args = parser.parse_args()

    out_path = Path(args.output_path)
    in_path = Path(args.input_path)

    out_path.mkdir(parents=True, exist_ok=True)
    extract(zip_file=in_path, out_path=out_path)

    if args.psf is not None:
        psf_path = Path(args.psf)
        corrupt_trset(data_path=out_path / "train", out_path=out_path, psf_path=psf_path)

if __name__ == "__main__":
    run(sys.argv)

"""
PYTHONPATH='..' python preprocess_ddpd.py \
    --input-path=/scratch/clear/gmeanti/data/dd_dp_dataset_canon.zip \
    --output-path=/scratch/clear/gmeanti/data/ddpd/ \
    --psf=/home/gmeanti/inverseproblems/notebooks/8x8_realpsf_27.pt
"""