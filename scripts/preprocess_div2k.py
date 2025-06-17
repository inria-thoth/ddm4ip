import hashlib
import argparse
from pathlib import Path
import sys
import zipfile

import requests
import tqdm


def make_full(full_path: Path, out_path: Path):
    max_val_img = 100

    with zipfile.ZipFile(full_path, 'r') as input_zip:
        files = sorted([n for n in input_zip.namelist() if n.endswith(".png")])
        val_files = files[:max_val_img]
        tr_files = files[max_val_img:]
        with zipfile.ZipFile(out_path / "div2k_val_full.zip", 'w') as output_zip:
            for val_file in tqdm.tqdm(val_files):
                with input_zip.open(val_file, 'r') as vf_data:
                    output_zip.writestr(val_file, vf_data.read())
        print(f"Written {len(val_files)} files to {out_path / 'div2k_val_full.zip'}")
        with zipfile.ZipFile(out_path / "div2k_train_full.zip", 'w') as output_zip:
            for tr_file in tqdm.tqdm(tr_files):
                with input_zip.open(tr_file, 'r') as tf_data:
                    output_zip.writestr(tr_file, tf_data.read())
        print(f"Written {len(tr_files)} files to {out_path / 'div2k_train_full.zip'}")


def download(url: str, filename: Path, expected_size: int | None = None, expected_md5: str | None = None):
    if filename.is_file():
        print("DIV2K already exists, skipping download.")
        return filename

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192
    data_size = 0
    data_md5 = hashlib.md5()
    with open(filename.with_suffix(".temp"), "wb") as file, tqdm.tqdm(
        desc=str(filename),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            file.write(chunk)
            data_md5.update(chunk)
            data_size += len(chunk)
            bar.update(len(chunk))
    # validate
    if expected_size is not None and expected_size != data_size:
        raise IOError('Incorrect file size', filename)
    if expected_md5 is not None and data_md5.hexdigest() != expected_md5:
        raise IOError('Incorrect file MD5', filename)

    filename.with_suffix(".temp").replace(filename)
    return filename


def run(argv):
    parser = argparse.ArgumentParser(argv[0], description="Download and process Div2k dataset.")
    parser.add_argument("--path", help="Directory where to place the Div2k dataset.")
    args = parser.parse_args()
    data_path = Path(args.path)
    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = download(
        url="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        filename=data_path / "DIV2K_train_HR.zip",
        expected_size=3530603713,
        expected_md5="bdc2d9338d4e574fe81bf7d158758658"
    )
    make_full(zip_path, data_path)


if __name__ == "__main__":
    run(sys.argv)
