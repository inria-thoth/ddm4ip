#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/div2krk_exp

zssr_dir="${OUTPUT_DIR}/zssr_baseline/"
mkdir -p "$zssr_dir"
for img_num in {1..100}; do
    lr_img="${DATA_DIR}/lr_x2/im_${img_num}.png"
    kernel="${DATA_DIR}/gt_k_x2/kernel_${img_num}.mat"
    out_img="${zssr_dir}/ZSSR_im_${img_num}.png"
    if [ -f "$out_img" ]; then
        echo "Output image $img_num exists at ${out_img}. Skipping ZSSR."
    else
        PYTHONPATH='.' python run_zssr.py \
            --kernel-path="$kernel" \
            --image-path="$lr_img" \
            --output-dir="$zssr_dir"  # e.g ZSSR_im_1.png
    fi
done

PYTHONPATH='../..' python compute_metrics.py \
    --reconstructed="$zssr_dir" \
    --ground-truth="${DATA_DIR}/gt" > "${zssr_dir}/metrics.txt"
