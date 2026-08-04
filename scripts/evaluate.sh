#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# LABEL 只决定输出文件夹名称，不决定模型权重。Base、SFT、GRPO 的公平比较方式是：
# 1. 分别用 serve_model.sh 启动对应 checkpoint；
# 2. 保持下面的 benchmark、采样参数和 35 步上限不变；
# 3. 分别执行 evaluate.sh baseline|sft|grpo。
LABEL="${1:-model}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"

# 评测同时依赖两个 HTTP 服务：
# - ShopSimulator :5700 保存每条购物轨迹的环境状态和 Reward v3 终局；
# - vLLM/OpenAI API :8000 负责从当前对话生成一个 tool call。
# 这里只给默认地址，允许用环境变量覆盖，避免把机器路径或地址写死进仓库。
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

# 输出分两层：trajectories.jsonl 是可复算的逐题原始事实，summary.json 是派生汇总。
# 后续排查 bad case 应先读 trajectory，不能只看 summary 中的平均数。
exec "$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY"
