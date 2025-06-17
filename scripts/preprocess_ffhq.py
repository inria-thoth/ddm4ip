import io
import sys
import zipfile
from pathlib import Path
import numpy as np
import tqdm
from PIL import Image


def run_on_dset(in_path: Path, out_path: Path, crop_size, start_idx=0, end_idx=70_000, seed=42):
    if out_path.exists():
        print(f"Output path {out_path.resolve()} already exists, skipping.")
        return
    if crop_size >= 1024 or crop_size <= 0:
        raise ValueError(f"Cannot crop 1024x1024 images to desired size '{crop_size}'")
    if not out_path.name.endswith(".zip"):
        raise ValueError("Output path must be a zip-file.")

    rng = np.random.RandomState(seed=seed)
    if in_path.is_dir():
        def image_gen():
            all_images = sorted(
                list(in_path.glob("*.png")),
                key=lambda img_path: int(img_path.name[:-4])
            )
            if len(all_images) != 70_000:
                raise ValueError(f"Dataset contains {len(all_images)} instead of the expected 70000.")
            image_ids = rng.permutation(len(all_images))
            image_ids = image_ids[start_idx:end_idx]
            for image_id in tqdm.tqdm(image_ids, desc=f"Resizing FFHQ from {start_idx} to {end_idx}"):
                image = all_images[image_id]
                with open(image, 'rb') as f:
                    yield f, image.name
    elif in_path.suffix == ".zip":
        def image_gen():
            with zipfile.ZipFile(str(in_path)) as zfile:
                names = sorted([n for n in zfile.namelist() if n.endswith(".png")])
                if len(names) != 70_000:
                    raise ValueError(f"Dataset contains {len(names)} instead of the expected 70000.")
                image_ids = rng.permutation(len(names))
                image_ids = image_ids[start_idx:end_idx]
                for image_id in tqdm.tqdm(image_ids, desc=f"Resizing FFHQ from {start_idx} to {end_idx}"):
                    name = names[image_id]
                    with zfile.open(name, 'r') as f:
                        yield f, name
    else:
        raise ValueError(f"Input path {in_path} must be either a directory or a zip file.")

    with zipfile.ZipFile(out_path.resolve(), mode="w", compression=0) as zf:
        for image_buffer, image_name in image_gen():
            # Read, Process, Write
            in_img = Image.open(image_buffer)
            if in_img.size != (1024, 1024):
                raise ValueError(f"Expected input images of size 1024x1024, but found image with size {in_img.size}")
            # Default is bicubic downsampling
            out_img = in_img.resize((crop_size, crop_size))
            out_img_bytes = io.BytesIO()
            out_img.save(out_img_bytes, format='PNG')
            zf.writestr(
                image_name,
                out_img_bytes.getvalue()
            )


def run(argv):
    import argparse
    parser = argparse.ArgumentParser(argv[0], description="Process the FFHQ dataset to get smaller sub-datasets needed for Diff4IP training.")
    parser.add_argument("--input-path", help="Path to original FFHQ 1024x1024 data. This should be a directory with all FFHQ samples as pngs.")
    parser.add_argument("--output-path", help="A directory where all FFHQ subsets (at 256x256 resolution) needed for Diff4IP will be placed")
    args = parser.parse_args()

    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.input_path)
    # all datasets are disjoint
    # 1. validation dataset of 1000 random images
    run_on_dset(in_path, out_path / "ffhq_256_val.zip", 256, start_idx=60_000, end_idx=61_000, seed=42)
    # 2. training dataset of 1000 random images
    run_on_dset(in_path, out_path / "ffhq_256_tr_1k.zip", 256, start_idx=0, end_idx=1000, seed=42)
    # 3. training dataset of 100 random images
    run_on_dset(in_path, out_path / "ffhq_256_tr_100.zip", 256, start_idx=5000, end_idx=5100, seed=42)


if __name__ == "__main__":
    run(sys.argv)
