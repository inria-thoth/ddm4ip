#!/bin/bash
#OAR -n "div2krk"
#OAR -O /scratch/clear/gmeanti/inverseproblems/div2krk_exp/%jobid%.log
#OAR -E /scratch/clear/gmeanti/inverseproblems/div2krk_exp/%jobid%.log
#OAR -l host=1/gpuid=1,walltime=20:00:00
#OAR -p (host='gpuhost31')

source ~/.bashrc
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export DNNLIB_CACHE_DIR="/scratch/clear/gmeanti/dnnlib_cache/"
export TORCH_HOME="/scratch/clear/gmeanti/dnnlib_cache/"

export DATA_DIR=/scratch/clear/gmeanti/data/div2krk
export OUTPUT_DIR=/scratch/clear/gmeanti/inverseproblems/div2krk_exp/

source gpu_setVisibleDevices.sh
conda activate torch

# Run in same shell
source run_all_ddm4ip.sh
