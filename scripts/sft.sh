#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"

# SFT 分成两个串行阶段：
#   Base 权重 + LoRA 训练 -> ADAPTER_DIR（只有低秩增量）
#   Base 权重 + Adapter  -> MERGED_DIR（完整权重，可直接交给 vLLM/GRPO）
# GRPO 默认读取 MERGED_DIR，因为 veRL rollout workers 需要一个普通 HF checkpoint，
# 而不是“Base 路径 + PEFT adapter 路径”这组二元输入。
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"

cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output "$ADAPTER_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --attention-implementation sdpa

# 只有训练进程成功退出才会执行合并。exec 让合并进程的退出状态成为 sft.sh 的退出状态。
exec "$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" \
  --bf16
