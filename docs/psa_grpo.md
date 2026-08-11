# PSA-GRPO 离线可行性审计

> 状态：方法原型与离线审计已实现，尚未进行 PSA-GRPO 训练。`PSA-GRPO`
> 是当前项目内的工作名（Pivotal-State Action Credit GRPO），不宣称学术首创。

## 问题

GRPO V2 的完整轨迹共享一个终局 advantage。已有 `11 gains / 13 losses` 表明模型
既获得了新成功，也破坏了 SFT 已会的局部决策；典型错误发生在已经打开候选、选择
规格或接近购买的后段。继续增加完整轨迹 GRPO 步数，不能回答“究竟是哪一个动作
导致后续成功或失败”。

PSA-GRPO 的目标不是给中间步骤手工加隐藏答案奖励，而是从一个可公开重放的中间
状态主动采样多个 suffix，再只对分叉后的 Assistant 决策 span 计算 loss。

## 已实现合同

### 1. 环境状态与模型输入分开标识

```text
replay_state_id = H(
  environment_manifest,
  task_id,
  public_query_hash,
  exact_accepted_action_prefix,
  public_observation_hash
)

branch_uid = H(
  replay_state_id,
  exact_actor_prompt_token_hash,
  tokenizer_contract_hash
)
```

相同页面不一定是相同决策状态：环境的重复、无进展和证据历史依赖已接受动作前缀；
相同环境状态也不一定是相同 GRPO prompt：Guard rejection、`think` 和上下文压缩会
改变模型实际看到的 token。因此只有 `branch_uid` 相同才能称为 exact prompt。

Replay ledger 与 `decision_trace.replay_parameters` 保存经过 JSON/schema 校验的完整
工具参数；`action_trace` 只保留有界展示字段。审计会从 exact preimage 重算 action
hash，并把已执行动作与 ledger 对照，因此长查询不会因 256 字符展示截断而碰撞。
任何审计记录失败只关闭 replay 合同，不改变原训练 reward 或动作执行。

### 2. Assistant turn 对齐

AgentLoop 为每次生成保留稳定 turn record，最终依据 veRL `response_mask` 的连续
`1-run / 0-run` 重建未 padding response 坐标：

```text
Assistant turn: [assistant_start, assistant_end)
Observation:    [assistant_end, next_assistant_start)
```

上下文压缩只删除完整的 Assistant + Observation 前缀组，并同步删除对应 turn record。
末轮若被 response 上限截断，只禁止该末轮 credit，不再清空之前已经对齐的 turns。
`think` 和未完成环境就输出最终回答也会进入 `decision_trace`，但不会进入环境 replay
ledger。

### 3. 逐步公开回放

`verify_pivotal_replay.py` 会在固定 Environment manifest 下执行：

1. `reset(task_id)`，要求 Observation v2；
2. 逐个执行 exact accepted action；
3. 每一步比较 before/after public observation SHA256；
4. 同时核对本地 manifest 和服务端 reset 返回的 manifest digest；
5. 在正常、环境异常和响应序列化异常路径释放环境租约；
6. 结果只写 hash、计数和异常类型，不保存服务端 raw payload。

隐藏 goal、目标 ASIN、正确规格和 Reward 细节不参与 state/branch identity，也不会写入
分叉审计。

当前租约仍有一个明确边界：若服务端已经成功 reset、但 HTTP 响应在传输中丢失，
客户端尚未拿到 `env_idx`，无法精确释放该 slot。正式主动分叉前需增加 lease request
token 与 TTL；在此之前 smoke 只在受控同 commit 服务进程上运行，并在前后检查 slot。

## 历史审计结果

输入是 GRPO V2 已保存的六份 `sampling_audit.jsonl`。旧日志没有 exact prompt、
Observation v2 完整性和 replay ledger 字段，因此下表只能统计观察性候选，不能直接
训练或证明首动作与终局存在因果关系。

| run | trajectories | valid | observational mixed states | unique tasks | task-capped states | exact-contract states |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| formal seed 3407 | 1,440 | 1,417 | 37 | 30 | 36 | 0 |
| formal seed 3408 | 1,432 | 1,409 | 50 | 38 | 50 | 0 |
| P0 25u | 392 | 383 | 8 | 8 | 8 | 0 |
| P1 25u | 320 | 317 | 11 | 10 | 11 | 0 |
| P2 25u | 560 | 552 | 44 | 24 | 35 | 0 |
| combined diagnostic | 4,152 | 4,086 | 150 | 83 | 118 | 0 |

合并口径只用于发现模式，不能把不同 seed、checkpoint 或 policy update 的样本拼成
一个 GRPO group。150 个观察性 mixed state 全部位于已打开候选的阶段，其中：

- 46 个涉及离开当前候选；
- 33 个已经选择过规格；
- 32 个发生在 Guard rejection 之后；
- 23 个下一动作包含购买决策。

最常见的动作差异是 `select_option` 与 `view_features`（37 组），其次是
`select_option` 与 `view_description/view_reviews`。在这些观察性候选中，`buy_now`
出现 28 次（23 次最终严格成功、5 次失败）；这说明 stopping boundary 值得主动
分叉验证，但不能据此直接给 `buy_now` 正奖励。

机器可读结果位于：

```text
outputs/experiments/psa_grpo_offline_20260811/
```

## 启动门槛

历史日志的 `exact-contract states = 0` 表示“旧 producer 没记录新合同”，不是证明
中间状态不存在。下一步先用固定 SFT policy 采集新审计轨迹，不更新权重：

1. 运行 1 update 工程 smoke，确认 optimizer 流程、turn span、decision trace、server
   manifest attestation 和 replay ledger 完整，且原 Reward/动态采样行为没有回归；
2. 使用 `lr=0` 的固定 SFT policy 收集 25 steps，避免把不同 policy 的状态混为一组；
3. 要求至少 50 个 exact-contract pivotal states，覆盖至少 40 个 task，每题最多计 2
   个 state；
4. 对候选执行 live reset + replay，逐步 public hash 匹配率必须为 100%；
5. 通过后才实现主动 suffix branching，每个状态采 `K>=4`，要求有效 mixed group 比例
   `>=40%`、真实基础设施无效率 `<=5%`；
6. 只有主动分叉产生的 suffix tensor 可以训练。历史 sampling-audit JSON 永远不能直接
   当作训练数据。

正式主动分叉还必须把 policy scope 绑定到 actor checkpoint/weight digest 与 decoding
config；当前 `run_manifest + global_step` 只适用于同一同步采样 run，不能跨 resume 或
异步 stale actor 合组。

在这些门槛通过前，报告中的 `training_ready` 固定为 `false`，也不运行 Final-200。
