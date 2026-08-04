#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 学习提示：baseline.sh 本身不加载 Base 模型。它只把评测结果目录标记为
# outputs/evaluation/baseline。真正被评测的是已经由 scripts/serve_model.sh
# 暴露成 OpenAI-compatible API 的模型，因此跑 baseline 前必须确认服务端加载的
# 是原始 Base checkpoint，而不是 SFT/GRPO checkpoint。
#
# exec 会用 evaluate.sh 替换当前 shell 进程：退出码和 Ctrl+C 都能原样传回调用者。
exec "$ROOT/scripts/evaluate.sh" baseline
