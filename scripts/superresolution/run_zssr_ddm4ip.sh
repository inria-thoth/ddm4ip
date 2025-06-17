#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/div2krk_exp
# NETWORK_BASE_NAME  sr_step2_lr1e-5x4_creg1e-2_bs64
# CHECKPOINT_STEP    1048576

zssr_dir="${OUTPUT_DIR}/zssr_test/"
mkdir -p "$zssr_dir"
for img_num in {1..100}; do
    gt_img="${DATA_DIR}/gt/img_${img_num}_gt.png"
    lr_img="${DATA_DIR}/lr_x2/im_${img_num}.png"  # input to the super-resolution network
    kernel="${zssr_dir}/kernel_${img_num}.mat"    # output of get_kernel_from_network.py
    out_img="${zssr_dir}/ZSSR_im_${img_num}.png"  # output of run_zssr.py

    if [ -f "$kernel" ]; then
        echo "Network kernel $img_num exists at ${kernel}. Skipping extraction."
    elif [ ! -f "${OUTPUT_DIR}/${NETWORK_BASE_NAME}/${NETWORK_BASE_NAME}_img${img_num}/checkpoints/network-snapshot-${CHECKPOINT_STEP}.pkl" ]; then
        echo "Checkpoint for image $img_num and step $CHECKPOINT_STEP does not exist. Skipping."
        continue
    else
        PYTHONPATH='../..' python get_kernel_from_network.py \
            --network="${OUTPUT_DIR}/${NETWORK_BASE_NAME}/${NETWORK_BASE_NAME}_img${img_num}/checkpoints/network-snapshot-${CHECKPOINT_STEP}.pkl" \
            --output-file="$kernel"
    fi

    if [ -f "$out_img" ]; then
        echo "Output image $img_num exists at ${out_img}. Skipping ZSSR."
    else
        PYTHONPATH='./KernelGAN' python run_zssr.py \
            --kernel-path="$kernel" \
            --image-path="$lr_img" \
            --noise_scale=1 \
            --preprocess \
            --output-dir="$zssr_dir"  # e.g ZSSR_im_1.png
    fi

    PYTHONPATH='../..' python compute_metrics.py \
        --reconstructed="$out_img" \
        --ground-truth="$gt_img" | tee -a "${zssr_dir}/metrics.txt"
done
