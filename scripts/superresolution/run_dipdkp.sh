#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/div2krk_exp

dkp_dir="${OUTPUT_DIR}"

for img_num in {1..100}; do
    lr_img="${DATA_DIR}/lr_x2/im_${img_num}.png"
    kernel="${DATA_DIR}/gt_k_x2/kernel_${img_num}.mat"
    out_img="${dkp_dir}/USRNET_im_${img_num}.png"

    if [ -f "$out_img" ]; then
        echo "Output image $img_num exists at ${out_img}. Skipping DKP."
    else
        PYTHONPATH='.' python run_dipdkp.py \
            --lr-image="$lr_img" \
            --out-path="$dkp_dir" \
            --sf=2 \
            --path-nonblind="./DKP/DIPDKP/data/pretrained_models/usrnet_tiny.pth" \
            --noise-scale=0.01 \
            --SR
    fi
done