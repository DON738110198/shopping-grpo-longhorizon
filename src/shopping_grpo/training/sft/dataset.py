"""将标准 OpenAI tool-calling messages 转为 LoRA SFT 所需的 labels。

训练时只计算 assistant token 的 loss；system、user 与 tool observation 都是上下文，
其标签固定为 ``IGNORE_INDEX``。边界完全交给目标模型的 chat template 决定，避免
手写 Qwen 特殊 token 或 tool-call 格式。
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

IGNORE_INDEX = -100


def _token_ids(tokenizer, text):
    """兼容 Hugging Face tokenizer 与测试用的最小 tokenizer。"""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _common_prefix_length(left, right):
    """返回两个 token 序列的最长公共前缀长度。"""
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def normalize_messages_for_chat_template(messages):
    """将 OpenAI 风格 tool-call 参数转成 Qwen3.5 模板可渲染的 mapping。

    原始 JSONL 保持 OpenAI 标准：``function.arguments`` 是 JSON 字符串。Qwen3.5
    的官方 template 则会遍历 arguments 的键值对，因此只在训练前复制并转换；
    无法解析为 object 的调用应被上层作为不可训练样本丢弃。
    """
    normalized = deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            function["arguments"] = parsed
    return normalized


def build_supervised_example(
    messages,
    tools,
    tokenizer,
    max_length=8192,
    chat_template=None,
    supervised_assistant_indices=None,
):
    """渲染一条轨迹，并只保留 assistant 回合对应的训练标签。

    每个 assistant 回合分别渲染「此前消息 + generation prompt」与「包含该回合的
    消息」，两者的 token 差即为该回合的可训练部分，其中自然包含 tool call。
    任何超长或模板边界不一致样本都会丢弃，不做可能截断工具调用的截断。

    返回的是单样本一维序列，形状均为 ``[L]``，且 ``L <= max_length``：

    - ``input_ids[t]``：第 t 个 token；
    - ``attention_mask[t]``：真实 token 恒为 1，batch padding 在 collator 中补 0；
    - ``labels[t]``：assistant token 等于 ``input_ids[t]``，其余位置为 -100。

    Transformers 的交叉熵会忽略 -100，因此目标等价于只最小化 assistant 区间：
    ``L_sft = -sum_{t in assistant} log p_theta(x_t | x_<t)``。工具 observation
    仍保留在 input_ids 中作为条件，却不会被模型当成需要模仿的输出。
    """
    # 原始消息保持 OpenAI 格式；只在这里为目标 chat template 做训练期转换。
    template = chat_template or tokenizer
    rendered_messages = normalize_messages_for_chat_template(messages)
    if rendered_messages is None:
        return None
    # assistant_indices 指向“消息级”位置；稍后还要用 chat template 把它映射成
    # token 级 [start, end) 区间。一个轨迹可以包含多个 assistant/tool 往返。
    all_assistant_indices = [
        index
        for index, message in enumerate(rendered_messages)
        if message.get("role") == "assistant"
    ]
    if supervised_assistant_indices is None:
        assistant_indices = all_assistant_indices
    else:
        try:
            requested_indices = {
                int(index) for index in supervised_assistant_indices
            }
        except (TypeError, ValueError):
            return None
        if (
            not requested_indices
            or not requested_indices.issubset(set(all_assistant_indices))
        ):
            return None
        assistant_indices = [
            index for index in all_assistant_indices if index in requested_indices
        ]
    if not assistant_indices:
        return None

    try:
        # full_text 是这条轨迹最终送入模型的唯一文本版本。先完整渲染，再做区间
        # 定位，可以保证训练 input 与推理时官方 chat template 的格式完全相同。
        full_text = template.apply_chat_template(
            rendered_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = _token_ids(tokenizer, full_text)
    except Exception:  # noqa: BLE001 - third-party templates raise varied errors.
        return None
    if len(input_ids) > int(max_length):
        return None

    # 先全部 mask，再逐个打开 assistant 回合；模型不会对用户指令或环境回复算 loss。
    labels = [IGNORE_INDEX] * len(input_ids)
    for index in assistant_indices:
        try:
            prefix_text = template.apply_chat_template(
                rendered_messages[:index],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            through_assistant_text = template.apply_chat_template(
                rendered_messages[: index + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_ids = _token_ids(tokenizer, prefix_text)
            through_assistant_ids = _token_ids(tokenizer, through_assistant_text)
        except Exception:  # noqa: BLE001 - third-party templates raise varied errors.
            return None

        # 部分 chat template 的 generation prompt 与实际 assistant 起始 token 会有
        # 极小差异（例如额外换行）。以公共前缀定位，避免把可用样本误判为损坏。
        # 例：prefix_ids=[system,user,<assistant-prefix>]，through_assistant_ids 还包含
        # assistant 内容/tool_call。最长公共前缀之后就是需要监督的第一个 token。
        start = _common_prefix_length(prefix_ids, through_assistant_ids)
        end = len(through_assistant_ids)
        if start >= end or end > len(input_ids):
            return None
        if input_ids[:end] != through_assistant_ids:
            return None
        labels[start:end] = input_ids[start:end]

    if not any(label != IGNORE_INDEX for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def load_supervised_examples(path, tokenizer, max_length=8192, chat_template=None):
    """读取本仓库生成的 SFT JSONL，并报告被模板拒绝的样本数。

    Tokenize 在训练开始前一次完成，返回 ``list[dict]``；这里故意记录 dropped，
    因为“JSON 能解析”不代表“目标模型模板能无损训练”。
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = lambda it, **kw: it

    examples = []
    stats = {"total": 0, "kept": 0, "dropped": 0}
    text = Path(path).read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    stats["total"] = len(lines)
    for line in _tqdm(lines, desc=f"  Tokenizing {Path(path).name}", unit=" samples"):
        try:
            row = json.loads(line)
            example = build_supervised_example(
                messages=row["messages"],
                tools=row.get("tools") or [],
                tokenizer=tokenizer,
                max_length=max_length,
                chat_template=chat_template,
                supervised_assistant_indices=row.get(
                    "supervised_assistant_indices"
                ),
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            example = None
        if example is None:
            stats["dropped"] += 1
            continue
        example["task_id"] = row.get("task_id")
        example["trajectory_id"] = row.get("trajectory_id")
        examples.append(example)
        stats["kept"] += 1
    return examples, stats


def split_rows_by_task(rows, validation_ratio=0.05, seed=42):
    """按 task_id 稳定划分 SFT 行，避免同题轨迹同时出现在训练和验证中。

    同一个 task 可能由 teacher 采集出多条 trajectory；若按行随机切分，模型会在
    validation 中看到训练题的近重复轨迹。这里先对 task_id 做稳定哈希，再整组切分。
    """
    ratio = float(validation_ratio)
    if not 0 <= ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    task_ids = {row.get("task_id") for row in rows}
    if ratio == 0 or len(task_ids) < 2:
        return list(rows), []

    def stable_key(task_id):
        value = f"{seed}:{task_id}".encode()
        return hashlib.sha256(value).hexdigest()

    ordered_ids = sorted(task_ids, key=stable_key)
    validation_count = max(1, round(len(ordered_ids) * ratio))
    validation_count = min(validation_count, len(ordered_ids) - 1)
    validation_ids = set(ordered_ids[:validation_count])
    validation_rows = [row for row in rows if row.get("task_id") in validation_ids]
    train_rows = [row for row in rows if row.get("task_id") not in validation_ids]
    return train_rows, validation_rows
