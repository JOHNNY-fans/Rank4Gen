#!/usr/bin/env bash
set -euo pipefail

# =========================
# Environment (safe to open-source)
# =========================
export MASTER_PORT="${MASTER_PORT:-88888}"

# CUDA path: keep generic; allow override by user env
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export PATH="$CUDA_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# =========================
# Paths (REDACTED -> configurable)
# =========================
# NOTE:
# - Replace these defaults with your local paths, or export them before running.
# - You can also put them into a .env file and source it (do NOT commit .env).

export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-8B}"
DATASET_FILES=(
  "${DATASET_INDEX_FILE:-/path/to/dataset/SFT/index.jsonl}"
  "${DATASET_SNAPSHOT_FILE:-/path/to/dataset/SFT/snapshot.jsonl}"
)

OUTPUT_DIR="${OUTPUT_DIR:-./output/Rank4Gen-SFT-8B-Qwen3}"
mkdir -p "$OUTPUT_DIR"

# =========================
# Distributed / NCCL / CUDA knobs (keep; reproducibility)
# =========================
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# GPU selection: keep configurable
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# =========================
# Train
# =========================
megatron sft \
  --model "$MODEL_PATH" \
  --use_hf true \
  --load_safetensors true \
  --save_safetensors true \
  --train_type full \
  --dataset "${DATASET_FILES[@]}" \
  --load_from_cache_file true \
  --tensor_model_parallel_size 8 \
  --pipeline_model_parallel_size 1 \
  --sequence_parallel true \
  --use_distributed_optimizer true \
  --check_model false \
  --packing true \
  --dataset_num_proc 32 \
  --torch_dtype bfloat16 \
  --split_dataset_ratio 0. \
  --seed 42 \
  --max_epochs 2 \
  --micro_batch_size 2 \
  --global_batch_size 32 \
  --max_length 40960 \
  --lr 1e-5 \
  --min_lr 1e-6 \
  --lr_warmup_fraction 0.05 \
  --save "$OUTPUT_DIR" \
  --save_interval 2000 \
  --no_save_optim true \
  --no_save_rng true \
  --log_interval 10 \
  --num_workers 32 \
  --truncation_strategy delete \
  --attention_backend flash \
  --cross_entropy_loss_fusion true \
  > "$OUTPUT_DIR/train.log" 2>&1
