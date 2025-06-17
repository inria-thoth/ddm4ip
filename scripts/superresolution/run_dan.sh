#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/div2krk_exp

dan_dir="${OUTPUT_DIR}/DANv2"
lr_dir="${DATA_DIR}/lr_x2"
hr_dir="${DATA_DIR}/gt"

PYTHONPATH='.' python run_dan.py \
    --hq-path="$hr_dir" \
    --lq-path="$lr_dir" \
    --out-path="$dan_dir"
