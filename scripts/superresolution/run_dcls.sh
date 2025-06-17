#!/bin/bash

set -euxo pipefail

export PYTHONUNBUFFERED=1

# Inputs:
# DATA_DIR    /scratch/clear/gmeanti/data/div2krk
# OUTPUT_DIR  /scratch/clear/gmeanti/inverseproblems/div2krk_exp

dcls_dir="${OUTPUT_DIR}/DCLS"
lr_dir="${DATA_DIR}/lr_x2"
hr_dir="${DATA_DIR}/gt"

PYTHONPATH='.' python run_dcls.py \
    --hq-path="$hr_dir" \
    --lq-path="$lr_dir" \
    --out-path="$dcls_dir"
