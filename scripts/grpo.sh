#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 这是薄封装：所有参数原样交给 train_grpo.py。Python 入口只负责校验、设置环境变量
# 和拼 veRL/Hydra 命令；真正的 PPO/GRPO 训练循环位于 verl.trainer.main_ppo。
exec "$ROOT/.venv/bin/python" scripts/train_grpo.py "$@"
