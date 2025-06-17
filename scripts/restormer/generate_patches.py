## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##### Data preparation file for training Restormer on the GoPro Dataset ########
from pathlib import Path
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def extract_paired_patches(
    lq_fpath: Path, hq_fpath: Path, lq_out_path: Path, hq_out_path: Path, patch_size: int, overlap: int
):
    filename = lq_fpath.stem
    lq_img = cv2.imread(str(lq_fpath))
    hq_img = cv2.imread(str(hq_fpath))
    num_patch = 0
    w, h = lq_img.shape[:2]
    w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
    h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
    w1.append(w-patch_size)
    h1.append(h-patch_size)
    for i in w1:
        for j in h1:
            num_patch += 1

            lq_patch = lq_img[i:i+patch_size, j:j+patch_size,:]
            hq_patch = hq_img[i:i+patch_size, j:j+patch_size,:]

            lq_savename = os.path.join(lq_out_path, f"{filename}-{num_patch}.png")
            hq_savename = os.path.join(hq_out_path, f"{filename}-{num_patch}.png")

            cv2.imwrite(lq_savename, lq_patch)
            cv2.imwrite(hq_savename, hq_patch)


def extract_center_patch(
    lq_fpath: Path, hq_fpath: Path, lq_out_path: Path, hq_out_path: Path, patch_size: int
):
    filename = lq_fpath.stem
    lq_img = cv2.imread(str(lq_fpath))
    hq_img = cv2.imread(str(hq_fpath))

    lq_savename = os.path.join(lq_out_path, f"{filename}.png")
    hq_savename = os.path.join(hq_out_path, f"{filename}.png")

    w, h = lq_img.shape[:2]

    i = (w - patch_size) // 2
    j = (h - patch_size) // 2

    lq_patch = lq_img[i: i + patch_size, j: j + patch_size,:]
    hq_patch = hq_img[i: i + patch_size, j: j + patch_size,:]

    cv2.imwrite(lq_savename, lq_patch)
    cv2.imwrite(hq_savename, hq_patch)


def extract_tr_val_patches():



############ Prepare Training data ####################
num_cores = 10
patch_size = 512
overlap = 256
p_max = 0

src = 'Datasets/Downloads/GoPro'
tar = 'Datasets/train/GoPro'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, 'input', '*.png')) + glob(os.path.join(src, 'input', '*.jpg')))
hr_files = natsorted(glob(os.path.join(src, 'target', '*.png')) + glob(os.path.join(src, 'target', '*.jpg')))

files = [(i, j) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))


############ Prepare validation data ####################
val_patch_size = 256
src = 'Datasets/test/GoPro'
tar = 'Datasets/val/GoPro'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, 'input', '*.png')) + glob(os.path.join(src, 'input', '*.jpg')))
hr_files = natsorted(glob(os.path.join(src, 'target', '*.png')) + glob(os.path.join(src, 'target', '*.jpg')))

files = [(i, j) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files))
