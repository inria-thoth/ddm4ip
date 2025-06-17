#!/bin/bash

set -euxo pipefail

# Run the two DDM4IP steps on a full dataset.
# To replicate the first part of the KernelGAN paper, run this script
# with the DIV2KRK dataset. This script assumes the data structure is
# the same as for the DIV2KRK dataset.

# Defaults
if [ -z "${DATA_DIR+x}" ]; then
    DATA_DIR="/scratch/clear/gmeanti/data/div2krk"
fi
if [ -z "${OUTPUT_DIR+x}" ]; then
    OUTPUT_DIR="/scratch/clear/gmeanti/inverseproblems/div2krk_exp/"
fi

base_exp1_name="sr_step1_ema0.004_bs512_v2"
base_exp2_name="sr_step2_2M_lr1e-5x8_creg1e-2_bs64_centerv1_64ch_v3"
exp1_dir="${OUTPUT_DIR}/${base_exp1_name}"
exp2_dir="${OUTPUT_DIR}/${base_exp2_name}"
zssr_dir="${OUTPUT_DIR}/zssr_x2_${base_exp2_name}/"
mkdir -p "$zssr_dir"
checkpoint_step=524288

for img_num in {1..100}; do
    gt_img="${DATA_DIR}/gt/img_${img_num}_gt.png"
    lr_img="${DATA_DIR}/lr_x2/im_${img_num}.png"
    gt_kernel="${DATA_DIR}/gt_k_x2/kernel_${img_num}.mat"
    net_kernel="${zssr_dir}/kernel_${img_num}.mat"    # output of get_kernel_from_network.py
    out_img="${zssr_dir}/ZSSR_im_${img_num}.png"  # output of run_zssr.py
    exp1_name="${base_exp1_name}_img${img_num}"
    exp2_name="${base_exp2_name}_img${img_num}"

    # Step 1
    if [ -f "${exp1_dir}/${exp1_name}/checkpoints/training-state-1048576.pt" ]; then
        echo "Checkpoint for image ${img_num}, step 1 already exists. Will not rerun it."
    else
        PYTHONPATH='../..' python ../../ddm4ip/main.py exp=step1_sr exp_name="${exp1_name}" \
            dataset.train_path="$gt_img" \
            dataset.test_path="$gt_img" \
            dataset.inflate_patches=15 \
            dataset/degradation=file_downsampling \
            dataset.degradation.kernel_path="$gt_kernel" \
            dataset.degradation.padding="valid" \
            dataset.degradation.kernel_size=15 \
            dataset.noise.std=0 \
            ema.stds='[0.004]' \
            paths.out_path="$exp1_dir" \
            training.batch_size=128 \
            loss.n_accum_steps=4 \
            optim.lr=0.01 \
            training.max_steps='${parse_nimg:"2049Ki"}' \
            training.save_every_steps='${parse_nimg:"1024Ki"}' \
            training.plot_every_steps='${parse_nimg:"256Ki"}' \
            training.report_every_steps='${parse_nimg:"64Ki"}'
    fi

    # Step 2
    if [ -f "${exp2_dir}/${base_exp2_name}_img${img_num}/checkpoints/training-state-${checkpoint_step}.pt" ]; then
        echo "Checkpoint for image ${img_num}, step 2 already exists. Will not rerun it."
    else
        PYTHONPATH='../..' python ../../ddm4ip/main.py exp=step2_sr exp_name="${exp2_name}" \
            models.pretrained_flow.path="${exp1_dir}/${exp1_name}/checkpoints/training-state-1048576.pt" \
            dataset.train_path="$gt_img" \
            dataset.test_path="$gt_img" \
            dataset/degradation=file_downsampling \
            dataset.degradation.kernel_path="$gt_kernel" \
            dataset.degradation.padding="valid" \
            dataset.degradation.kernel_size=15 \
            dataset.noise.std=0 \
            optim.aux.lr=0.00001 \
            optim.kernel.lr=0.00008 \
            loss.center_reg=0.01 \
            loss.sparse_reg=0.1 \
            loss.sum_to_one_reg=0.1 \
            paths.out_path="${exp2_dir}" \
            training.batch_size=64 \
            training.max_steps='${parse_nimg:"513Ki"}' \
            training.save_every_steps='${parse_nimg:"512Ki"}' \
            training.plot_every_steps='${parse_nimg:"128Ki"}' \
            training.report_every_steps='${parse_nimg:"32Ki"}'
    fi

    # Extract kernel from network checkpoint to ZSSR directory
    if [ -f "$net_kernel" ]; then
        echo "Network kernel $img_num exists at ${net_kernel}. Skipping extraction."
    elif [ ! -f "${exp2_dir}/${exp2_name}/checkpoints/network-snapshot-${checkpoint_step}.pkl" ]; then
        echo "Checkpoint for image $img_num and step $checkpoint_step does not exist. Skipping."
        continue
    else
        PYTHONPATH='../..' python get_kernel_from_network.py \
            --network="${exp2_dir}/${exp2_name}/checkpoints/network-snapshot-${checkpoint_step}.pkl" \
            --output-file="$net_kernel"
    fi

    # Run ZSSR
    if [ -f "$out_img" ]; then
        echo "Output image $img_num exists at ${out_img}. Skipping ZSSR."
    else
        PYTHONPATH='./KernelGAN' python run_zssr.py \
            --kernel-path="$net_kernel" \
            --image-path="$lr_img" \
            --output-dir="$zssr_dir"  # e.g ZSSR_im_1.png
        PYTHONPATH='../..' python compute_metrics.py \
            --reconstructed="$out_img" \
            --ground-truth="$gt_img" | tee -a "${zssr_dir}/metrics.txt"
    fi

done
