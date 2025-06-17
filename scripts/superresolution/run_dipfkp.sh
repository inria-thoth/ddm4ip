#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/superresolution/dipfkp

fkp_dir="${OUTPUT_DIR}/baseline"  # will save in baseline_DIPFKP

for img_num in {1..100}; do
    lr_img="${DATA_DIR}/lr_x2/im_${img_num}.png"
    kernel="${DATA_DIR}/gt_k_x2/kernel_${img_num}.mat"
    out_img="${fkp_dir}_DIPFKP/USRNET_im_${img_num}.png"

    if [ -f "$out_img" ]; then
        echo "Output image $img_num exists at ${out_img}. Skipping FKP."
    else
        PYTHONPATH='.' python run_dipfkp.py \
            --lr-image="$lr_img" \
            --output-dir="$fkp_dir" \
            --sf=2 \
            --path-nonblind="./FKP/data/pretrained_models/usrnet_tiny.pth" \
            --path-KP="./FKP/data/pretrained_models/FKP_x2.pt" \
            --noise-scale=0.01 \
            --SR
    fi
done