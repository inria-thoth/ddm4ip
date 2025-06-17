#!/bin/bash
# Run as ./oar_run_exp.sh ../ddm4ip/configs/exp/step1_parkinglot.yaml

set -eux

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment.yaml>"
    exit 1
fi
INPUT_YAML="$1"


EXP_DIR="/scratch/clear/gmeanti/inverseproblems/experiments/"
CACHE_DIR="/scratch/clear/gmeanti/model_cache/"
if [ ! -d "$EXP_DIR" ]; then
    echo "ERROR: Experiment directory expected at '$EXP_DIR' does not exist. Please create it, or edit this file to point to an existing directory for experiments."
    exit 2
fi

# Extract exp_name
EXP_NAME=`awk -F': ' '/^exp_name:/ {print $2}' "$INPUT_YAML"`
if [ -z "$EXP_NAME" ]; then
    echo "ERROR: Could not extract experiment name (key 'exp_name') from input file at '$INPUT_YAML'."
    exit 3
fi
BASE_EXP_NAME=`basename "$INPUT_YAML" .yaml`

#OAR -t besteffort

mkdir -p "${EXP_DIR}/${EXP_NAME}"
# 1st heredoc with variable substitution
cat << EOF > "${EXP_DIR}/${EXP_NAME}/run.sh"
#!/bin/bash

#OAR -n "${EXP_NAME}"
#OAR -O ${EXP_DIR}/${EXP_NAME}/%jobid%.out
#OAR -E ${EXP_DIR}/${EXP_NAME}/%jobid%.out
#OAR -l host=1/gpuid=1,walltime=14:00:00
#OAR -p (host='gpuhost31')

source ~/.bashrc
source gpu_setVisibleDevices.sh
conda activate torch

export PYTHONUNBUFFERED=1
export DNNLIB_CACHE_DIR=$CACHE_DIR
export TORCH_HOME=$CACHE_DIR
BASE_EXP_NAME=$BASE_EXP_NAME
EOF
# 2nd heredoc without variable substitution (note the quotes around EOF)
cat >> "${EXP_DIR}/${EXP_NAME}/run.sh" << 'EOF'
cd ..
PYTHONPATH='.' python ddm4ip/main.py exp=${BASE_EXP_NAME} paths=edgar_giac
# for i in {5..5}; do
#     exp_name="di_tr1k_ffhq256-5000-5100_motionblur_ks28_noise0.02_direct_v2_seed${i}"
#     if [[ ! -d "/scratch/clear/gmeanti/inverseproblems/experiments/${exp_name}" ]]; then
#         PYTHONPATH='.' python ddm4ip/main.py exp=${BASE_EXP_NAME} paths=edgar_giac training.seed=${i} exp_name=${exp_name}
#     fi
#     PYTHONPATH='.' python ddm4ip/main.py exp=step3_ffhq paths=edgar_giac \
#         models.kernel.path="/scratch/clear/gmeanti/inverseproblems/experiments/${exp_name}/checkpoints/network-snapshot-524288.pkl" \
#         exp_name="diffpir_ffhq256_1k-100-imgs_motionblur_seed${i}" training.seed=${i}
# done
EOF

chmod a+x "${EXP_DIR}/${EXP_NAME}/run.sh"
oarsub -S "${EXP_DIR}/${EXP_NAME}/run.sh"
