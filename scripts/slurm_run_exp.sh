#!/bin/bash

# --qos=qos_gpu_a100-dev
# --qos=qos_gpu-dev

# Run as ./slurm_run_exp.sh ...

set -eux

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment.yaml>"
    exit 1
fi
INPUT_YAML="$1"

EXP_DIR="/lustre/fsn1/projects/rech/kzr/uou82zu/outputs"
CACHE_DIR="/lustre/fswork/projects/rech/kzr/uou82zu/dnnlib_cache/"
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


sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="${EXP_NAME}"
#SBATCH --output="${EXP_DIR}/${EXP_NAME}__%j.out"
#SBATCH --error="${EXP_DIR}/${EXP_NAME}__%j.out"
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --account=kzr@h100
#SBATCH --constraint=h100
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=10

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0
export DNNLIB_CACHE_DIR=$CACHE_DIR
export TORCH_HOME=$CACHE_DIR

cd ..
PYTHONPATH='.' python ddm4ip/main.py exp=${BASE_EXP_NAME} paths=jz

EOT