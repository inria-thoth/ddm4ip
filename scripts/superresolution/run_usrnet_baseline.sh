#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/div2krk_exp

usrnet_dir="${OUTPUT_DIR}/usrnet_x2_baseline/"
mkdir -p "$usrnet_dir"
for img_num in {1..100}; do
    lr_img="${DATA_DIR}/lr_x2/im_${img_num}.png"  # input to the super-resolution network
    kernel="${DATA_DIR}/gt_k_x2/kernel_${img_num}.mat"
    out_img="${usrnet_dir}/USRNET_im_${img_num}.png"  # output of run_usrnet.py

    if [ -f "$out_img" ] && false; then
        echo "Output image $img_num exists at ${out_img}. Skipping USRNET."
    else
        PYTHONPATH='.' python run_usrnet.py \
            --kernel-path="$kernel" \
            --image-path="$lr_img" \
            --model-path="/scratch/clear/gmeanti/model_cache/usrnet_tiny.pth" \
            --noise-scale=0.01 \
            --output-dir="$usrnet_dir"  # e.g USRNET_im_1.png
    fi
done

PYTHONPATH='../..' python compute_metrics.py \
    --reconstructed="$usrnet_dir" \
    --ground-truth="${DATA_DIR}/gt" | tee -a "${usrnet_dir}/metrics.txt"
