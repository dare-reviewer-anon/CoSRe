#!/bin/bash
#SBATCH --job-name=anole-h100-4g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_h100          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END
# Cluster: <CLUSTER_PROVIDER>

#############################
# 1. Load modules
#############################
module load 2023
module load CUDA/12.4.0
module load Miniconda3/23.5.2-0

#############################
# 2. Environment setup
#############################
# Use your conda env without conda init
export PATH=/home/<USER>/.conda/envs/COSRE-env/bin:$PATH
export PYTHONNOUSERSITE=1

# HuggingFace / cache
export HF_HOME=<FS_ROOT>/hfcache
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DISABLE_TELEMETRY=1

export WANDB_MODE=online
export WANDB_API_KEY=YOUR_KEY 

# Triton / CUDA memory behavior
export TRITON_CACHE_DIR=<FS_ROOT>/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR"

# Threads / CUDA connections
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Use 4 GPUs on the node
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Project root (COSRE / Anole + COSRE training code)
cd <FS_ROOT>/COSRE
mkdir -p logs

echo "=== Environment check ==="
which python
python --version
which torchrun
echo "========================="

#############################
# 3. Training (4 GPUs + ZeRO-3)
#############################

echo "[$(date)] Starting torchrun..."
torchrun --nproc_per_node=4 traino.py \
  --model anole \
  --data interleaved_maze \
  --data_dir data_samples \
  --decoder_type anole \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/anole7b_zero3_4gpusoutput \
  --report_to wandb \
  --train_bz 2 \
  --val_bz 2 \
  --grad_acc 32 \
  --image_seq_length 1024 \
  --note "anole7b_zero3_4gpus_"
