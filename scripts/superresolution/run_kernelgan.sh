#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/div2krk_exp

zssr_dir="${OUTPUT_DIR}/zssr_kernelgan/"
mkdir -p "$zssr_dir"

PYTHONPATH='./KernelGAN' python KernelGAN/train.py \
    --input-dir="${DATA_DIR}/lr_x2" \
    --output-dir="$zssr_dir" \
    --SR

for img_num in {1..100}; do
    mv "${zssr_dir}/im_${img_num}"/* "$zssr_dir/"
    rmdir "${zssr_dir}/im_${img_num}"
done

PYTHONPATH='../..' python compute_metrics.py \
    --reconstructed="$zssr_dir" \
    --ground-truth="${DATA_DIR}/gt" | tee "${zssr_dir}/metrics.txt"
