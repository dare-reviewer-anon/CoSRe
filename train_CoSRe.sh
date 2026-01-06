#!/bin/bash
#SBATCH --job-name=dare-anole-a100-4g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_a100
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
# Use the DARE conda env (replace with your env name if different)
export PATH=/home/<USER>/.conda/envs/DARE/bin:$PATH
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

#############################
# 3. Move to project root
#############################
cd <FS_ROOT>/DARE
mkdir -p logs

echo "=== Environment check ==="
which python
python --version
which torchrun
echo "========================="

#############################
# 4. DARE training (4 GPUs + ZeRO-3)
#############################

echo "[$(date)] Starting DARE torchrun..."

torchrun --nproc_per_node=4 train_dare.py \
  --model anole \
  --data interleaved_maze \
  --data_dir <FS_ROOT>/data-samples \
  --decoder_type anole \
  --input_format anole \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/dare-anole7b-maze \
  --note "dare-maze-image_seq_len-1024-" \
  --image_seq_length 1024 \
  --report_to none \
  --train_bz 2 \
  --val_bz 2 \
  --grad_acc 32 \
  --enable_dare \
  --rho_text_target 0.7 \
  --rho_vis_target 0.4 \
  --dare_prefix_kappa 2 \
  --model_ckpt <FS_ROOT>/DARE/outputs/anole7b_zero3_4gpusoutput \
  --load_last_checkpoint
