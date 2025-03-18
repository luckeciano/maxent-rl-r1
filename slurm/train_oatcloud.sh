#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="act-pm"
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/

export TMPDIR=/scratch/${USER}/tmp
mkdir -p $TMPDIR
BUILD_DIR=/scratch-ssd/${USER}/conda_envs/pip-build

# Clean pip cache and set pip configurations
rm -rf ~/.cache/pip
export PIP_NO_CACHE_DIR=1
export PIP_DEFAULT_TIMEOUT=100
export PYTHONWARNINGS="ignore::DeprecationWarning"

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/maxent-rl-r1/environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate maxent-r1

cd ~/maxent-rl-r1
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -e ".[dev]"

# Installing flash-attn
pip install flash-attn --no-build-isolation

echo $TMPDIR

nvidia-smi

huggingface-cli login --token $HUGGINGFACE_WRITETOKEN

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1


ACCELERATE_LOG_LEVEL=debug accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-7B/grpo/config_oatcloud.yaml
