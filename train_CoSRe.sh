#!/bin/bash
#SBATCH --job-name=CoSRe-anole-h100-4g
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

set -euo pipefail

#############################
# 1. Load modules
#############################
module load 2023
module load CUDA/12.4.0
module load Miniconda3/23.5.2-0

#############################
# 2. Environment setup
#############################
export PATH=/home/<USER>/.conda/envs/COSRE/bin:$PATH
export PYTHONNOUSERSITE=1

# HuggingFace / cache
export HF_HOME=<FS_ROOT>/hfcache
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DISABLE_TELEMETRY=1

# W&B (disabled below via --report_to none; keep for convenience)
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

#############################
# 3. Move to project root
#############################
cd <FS_ROOT>/COSRE
mkdir -p logs

echo "=== Environment check ==="
which python
python --version
which torchrun
nvidia-smi
echo "========================="

#############################
# 4. CoSRe training (4 GPUs + ZeRO-3)
#############################
echo "[$(date)] Starting CoSRe torchrun..."

MASTER_PORT=$((12000 + RANDOM % 20000))

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train_CoSRe.py \
  --model anole \
  --data interleaved_maze \
  --data_dir <FS_ROOT>/data-samples \
  --decoder_type anole \
  --input_format anole \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/COSRE-anole7b-maze \
  --note "CoSRe-maze-" \
  --image_seq_length 1024 \
  --report_to none \
  --train_bz 2 \
  --val_bz 2 \
  --grad_acc 32 \
  --enable_CoSRe \
  --cosre_block 8 \
  --cosre_keep_h 4 \
  --cosre_keep_w 4 \
  --cosre_slots 32 \
  --cosre_base_delta 0.5 \
  --model_ckpt <FS_ROOT>/COSRE/outputs/anole7b_zero3_4gpusoutput \
  --load_last_checkpoint

echo "[$(date)] Done."
