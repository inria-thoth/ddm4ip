#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/div2krk_exp
# NETWORK_BASE_NAME  sr_step2_lr1e-5x8_creg1e-2_bs64_centerv1_64ch_v3
# CHECKPOINT_STEP    1048576

usrnet_dir="${OUTPUT_DIR}/usrnet_x2_${NETWORK_BASE_NAME}/"
mkdir -p "$usrnet_dir"
for img_num in {1..100}; do
    lr_img="${DATA_DIR}/lr_x2/im_${img_num}.png"  # input to the super-resolution network
    kernel="${usrnet_dir}/kernel_${img_num}.mat"    # output of get_kernel_from_network.py
    out_img="${usrnet_dir}/USRNET_im_${img_num}.png"  # output of run_usrnet.py

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
        echo "Output image $img_num exists at ${out_img}. Skipping USRNET."
    else
        PYTHONPATH='.' python run_usrnet.py \
            --kernel-path="$kernel" \
            --image-path="$lr_img" \
            --model-path="/scratch/clear/gmeanti/model_cache/usrnet_tiny.pth" \
            --noise-scale=0.001 \
            --output-dir="$usrnet_dir"  # e.g ZSSR_im_1.png
    fi
done

PYTHONPATH='../..' python compute_metrics.py \
    --reconstructed="$usrnet_dir" \
    --ground-truth="${DATA_DIR}/gt" | tee -a "${usrnet_dir}/metrics.txt"
