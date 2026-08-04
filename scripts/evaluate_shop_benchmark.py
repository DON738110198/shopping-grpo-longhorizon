#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。

阅读这条评测主链时，把它看成一个“两服务、三层数据”的驱动器：

1. 模型服务接收 messages/tools，返回 assistant 文本或一个 tool call；
2. ShopSimulator 服务接收结构化购物动作，返回 observation、done 和 Reward v3；
3. ``collect_tasks`` 保存逐题 trajectory，``summarize_trajectories`` 再从原始轨迹
   计算固定分母指标。这个入口不加载模型权重，也不实现 Agent 循环本身。

关键调用链：
``main -> load_tasks -> OpenAIChatClient -> collect_tasks ->
OpenAIChatClient.complete / ShopAgentEnv.step -> summarize_trajectories``。
"""

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.summary import summarize_trajectories
from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_tasks, load_tasks


def parse_args():
    """定义冻结评测协议中允许从命令行改变的变量。

    公平比较 Base/SFT/GRPO 时，通常只改变正在服务的 checkpoint 和输出目录；
    temperature、top_p、max_steps、上下文预算应保持一致。
    """
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True, help="本地 vLLM 可传 EMPTY")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument("--context-window", type=int, default=24576)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action="store_true",
        help="上下文接近上限时压缩较早的交互；默认关闭。",
    )
    parser.add_argument("--observation-token-budget", type=int, default=1536)
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    return parser.parse_args()


def _read_jsonl(path):
    """重新读取落盘轨迹，而不是复用内存结果。

    这样 summary 总是可以由 ``trajectories.jsonl`` 单独复算；进程中断后再次运行时，
    ``collect_tasks`` 也能续跑缺失 task，而不会让已完成轨迹消失。
    """
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()

    # 阶段 0：在发出任何 HTTP 请求前拒绝不可能成立的预算。
    # 模型可见输入的理论上限是：
    # context_window - max_tokens - context_safety_margin。
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    # 阶段 1：每个 task 至少携带 task_id；task_id 同时是环境 reset 的主键，也是
    # 汇总时固定分母的依据。不要用“实际成功写出的轨迹数”充当分母。
    tasks = load_tasks(args.benchmark)

    # 阶段 2：这里只构造 API 客户端。它内部负责 chat template 之外的在线事务：
    # 消息历史、token 计数、可选上下文压缩，以及调用 /chat/completions。
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )
    # 阶段 3：真正的 ReAct/tool-use 循环位于 evaluation/rollout.py。
    # 一道题会反复执行“模型生成 -> 工具校验 -> 环境 step -> observation 入历史”，
    # 直到 Reward v3 终局、达到 35 步或出现明确错误。
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
    )
    # 阶段 4：从刚落盘的原始 JSONL 重新计算确定性指标。expected task_ids 被显式
    # 传入，因此缺失、崩溃或未完成的题不会从分母中悄悄消失。
    summary = summarize_trajectories(
        [task["task_id"] for task in tasks], _read_jsonl(args.output)
    )
    # protocol 与指标一起落盘，回答“这个数字是在什么采样/上下文设置下得到的”。
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "model": args.model,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction,
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
