import argparse
import io
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import torch
import torchvision.transforms.v2.functional as TF
from tqdm import tqdm

from ddm4ip.data.utils import extract_patches, AddLocMapTransform
from ddm4ip.utils.torch_utils import read_img_pt, write_img_pt


image_correspondences = [
    # Day 2
    {
        "5.6":  "P1081840_f5.6.png",
        "16.0": "P1081841_f16.0.png",
    },
    {
        "5.6":  "P1081843_f5.6.png",
        "16.0": "P1081842_f16.0.png",
    },
    {
        "5.6":  "P1081846_f5.6.png",
        "16.0": "P1081847_f16.0.png",
    },
    {
        "5.6":  "P1081849_f5.6.png",
        "16.0": "P1081848_f16.0.png",
    },
    {
        "5.6":  "P1081852_f5.6.png",
        "16.0": "P1081853_f16.0.png",
    },
    {
        "5.6":  "P1081855_f5.6.png",
        "16.0": "P1081854_f16.0.png",
    },
    {
        "5.6":  "P1081858_f5.6.png",  # bike shed
        "16.0": "P1081859_f16.0.png",
    },
    # {
    #     "5.6":  "P1081861_f5.6.png",  # car side
    #     "16.0": "P1081860_f16.0.png",
    # },
    # {
    #     "5.6":  "P1081864_f5.6.png",  # contains licence plates
    #     "16.0": "P1081865_f16.0.png",
    # },
    {
        "5.6":  "P1081867_f5.6.png",
        "16.0": "P1081866_f16.0.png",
    },
    # {
    #     "5.6":  "P1081870_f5.6.png",  # too much sky
    #     "16.0": "P1081871_f16.0.png",
    # },
    {
        "5.6":  "P1081873_f5.6.png",
        "16.0": "P1081872_f16.0.png",
    },
    # {
    #     "5.6":  "P1081876_f5.6.png",  # no texture
    #     "16.0": "P1081877_f16.0.png",
    # },
    {
        "5.6":  "P1081879_f5.6.png",
        "16.0": "P1081878_f16.0.png",
    },
    {
        "5.6":  "P1081882_f5.6.png",
        "16.0": "P1081883_f16.0.png",
    },
    {
        "5.6":  "P1081885_f5.6.png",  # too much sky
        "16.0": "P1081884_f16.0.png",
    },
    {
        "5.6":  "P1081890_f5.6.png",
        "16.0": "P1081891_f16.0.png",
    },
    {
        "5.6":  "P1081893_f5.6.png",
        "16.0": "P1081892_f16.0.png",
    },
    # {
    #     "5.6":  "P1081899_f5.6.png",  # car
    #     "16.0": "P1081898_f16.0.png",
    # },
    # Day 3
    {
        "5.6":  "P1081921_f5.6.png",
        "16.0": "P1081922_f16.0.png",
    },
    {
        "5.6":  "P1081928_f5.6.png",
        "16.0":  "P1081927_f16.0.png",
    },
    # {
    #     "5.6":  "P1081931_f5.6.png",  # out of focus
    #     "16.0":  "P1081932_f16.0.png",
    # },
    {
        "5.6":  "P1081936_f5.6.png",
        "16.0":  "P1081935_f16.0.png",
    },
    # {
    #     "5.6":  "P1081941_f5.6.png",  # does not exist?
    #     "16.0":  "P1081942_f16.0.png",
    # },
    {
        "5.6":  "P1081945_f5.6.png",
        "16.0":  "P1081946_f16.0.png",
    },
    {
        "5.6":  "P1081951_f5.6.png",
        "16.0":  "P1081949_f16.0.png",
    },
    {
        "5.6":  "P1081953_f5.6.png",
        "16.0":  "P1081954_f16.0.png",
    },
    {
        "5.6":  "P1081957_f5.6.png",
        "16.0":  "P1081956_f16.0.png",
    },
    {
        "5.6":  "P1081960_f5.6.png",
        "16.0":  "P1081961_f16.0.png",
    },
]


def run_processing(raw_path: Path, jpg_path: Path | None, out_base_path: Path):
    ppm_path = out_base_path / "ppm"
    ppm_path.mkdir(parents=True, exist_ok=True)
    png_path = out_base_path / "png"
    png_path.mkdir(parents=True, exist_ok=True)

    for raw_img in raw_path.glob("*.RW2"):
        output = subprocess.check_output(["dcraw", "-i", "-v", str(raw_img)])
        aperture = re.search(r"^Aperture: f/([0-9\.]+)$", output.decode('utf-8'), re.MULTILINE).group(1)
        # 1. Rename raw
        if aperture not in raw_img.name:
            img_number = raw_img.name[:-4]
            new_raw_img = raw_img.with_name(f"{raw_img.name[:-4]}_f{aperture}.RW2")
            print(f"Renaming {raw_img} to {new_raw_img}")
            raw_img.rename(new_raw_img)
            raw_img = new_raw_img
        else:
            img_number = raw_img.name.split("_")[0]

        # 2. Rename corresponding jpg to have f-stop in name
        if jpg_path is not None:
            if (old_jpg_path := jpg_path / f"{img_number}.JPG").exists():
                new_jpg_path = old_jpg_path.with_name(f"{img_number}_f{aperture}.JPG")
                print(f"Renaming {old_jpg_path} to {new_jpg_path}")
                old_jpg_path.rename(new_jpg_path)

        # 3. Process RAW with dcraw
        ppm_img = ppm_path / f"{raw_img.name[:-4]}.ppm"
        if not ppm_img.exists():
            output = subprocess.check_output([
                "dcraw", "-v",
                "-w",       # use camera white balance
                "-H", "2",  # highlight mode: blend
                "-q", "3",  # demosaick algorithm: AHD
                "-n", "50", # noise suppression threshold
                str(raw_img)
            ])
            print(output.decode('utf-8'))
            tmp_ppm_img = raw_img.with_name(f"{raw_img.name[:-4]}.ppm")
            tmp_ppm_img.replace(ppm_img)
            assert ppm_img.exists()

        # 4. Convert PPM to PNG with imagemagick
        png_img = png_path / f"{raw_img.name[:-4]}.png"
        if not png_img.exists():
            output = subprocess.check_output([
                "convert", str(ppm_img), "-quality", "7", str(png_img)
            ])
            print(output.decode('utf-8'))
            assert png_img.exists()


def make_zip_fullimgs(fstop, out_path: Path, img_info, base_path):
    if not out_path.name.endswith(".zip"):
        raise ValueError(f"Expected a path to a zip-file, bound found {out_path}")
    with zipfile.ZipFile(out_path, "w") as out:
        for i, img_pair_info in tqdm(enumerate(img_info)):
            img_path = base_path / img_pair_info[fstop]
            out.write(str(img_path), arcname=img_path.name)
    print(f"Written {len(img_info)} images at f-stop {fstop} to {out_path}")


def make_zip_patches(
    patch_size: int,
    patch_overlap: int,
    fstop: str,
    out_path: Path,
    img_info: list[dict[str, str]],
    base_path: Path,
):
    if not out_path.name.endswith(".zip"):
        raise ValueError(f"Expected a path to a zip-file, bound found {out_path}")

    trsf = AddLocMapTransform()
    num_imgs = 0
    with zipfile.ZipFile(out_path, "w") as out:
        for i, img_pair_info in tqdm(enumerate(img_info)):
            if not (img_path := base_path / img_pair_info[fstop]).is_file():
                print(f"Image at {img_path} not found. Skipping")
                continue
            num_imgs += 1
            img = read_img_pt(base_path / img_pair_info[fstop])
            img = trsf(img)

            patches = extract_patches(img, kernel_size=patch_size, stride=patch_size - patch_overlap)
            patches = patches.reshape(-1, *patches.shape[2:])
            for j in range(patches.shape[0]):
                fname = f"img_{i}_{j}.pt"

                patch = patches[j].clone()
                buf = io.BytesIO()
                torch.save(patch, buf)
                buf.seek(0)
                out.writestr(fname, buf.read())
    print(f"Written all patches of size {patch_size} with overlap {patch_overlap} at "
          f"f-stop {fstop} to {out_path} from {num_imgs} full images")


def make_zip_center_crop(
    crop_size: int,
    fstop: str,
    out_path: Path,
    img_info: list[dict[str, str]],
    base_path: Path,
):
    if not out_path.name.endswith(".zip"):
        raise ValueError(f"Expected a path to a zip-file, bound found {out_path}")

    num_imgs = 0
    with zipfile.ZipFile(out_path, "w") as out:
        for i, img_pair_info in tqdm(enumerate(img_info)):
            if not (img_path := base_path / img_pair_info[fstop]).is_file():
                print(f"Image at {img_path} not found. Skipping")
                continue
            num_imgs += 1
            img = read_img_pt(base_path / img_pair_info[fstop])
            center_img = TF.center_crop(img, [crop_size, crop_size])
            fname = img_pair_info[fstop]
            buf = io.BytesIO()
            write_img_pt(center_img, buf)
            buf.seek(0)
            out.writestr(fname, buf.read())
    print(f"Written {num_imgs} center-crops of size {crop_size} at "
          f"f-stop {fstop} to {out_path}.")


def run(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument("--path", help="Base directory of the parking-lot dataset.", required=True)
    parser.add_argument("--preprocess", action="store_true", default=False)
    args = parser.parse_args()
    base_path = Path(args.path)

    if args.preprocess:
        # Data is in two directories: one for JPEGs and one for RAW files.
        # We go through the following processing steps for the RAW files
        # A rename step to rename the files such that the f-stop is part of the filename
        # A RAW processing step with `dcraw` which outputs a PPM file
        # A conversion step to PNG using imagemagick
        # For the JPEG files we only perform the first step.
        if not base_path.is_dir():
            raise ValueError("Base path must exist and have the parking lot data in it.")
        raw_path = base_path / "raw"
        if not raw_path.is_dir():
            raise FileNotFoundError(raw_path)
        jpg_path = base_path / "jpg"
        if not jpg_path.is_dir():
            jpg_path = None
        run_processing(raw_path, jpg_path, base_path)
    make_zip_fullimgs("5.6", base_path / "parkinglot_5_6.zip", image_correspondences, base_path / "png")
    make_zip_fullimgs("16.0", base_path / "parkinglot_16.zip", image_correspondences, base_path / "png")
    # make_zip_patches(256, "5.6", base_path / "plot_patches_5.6.zip", image_correspondences, base_path)# / "png")
    # make_zip_patches(256, 32, "16.0", base_path / "parkingd3_patches_16.0.zip", image_correspondences, base_path / "png")
    make_zip_center_crop(768, "5.6", base_path / "parkinglot_5_6_center768.zip", image_correspondences, base_path / "png")
    make_zip_center_crop(768, "16.0", base_path / "parkinglot_16_center768.zip", image_correspondences, base_path / "png")

if __name__ == "__main__":
    run(sys.argv)