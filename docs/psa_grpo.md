# PSA-GRPO 离线可行性审计

> 状态：方法原型、固定策略状态审计与主动 suffix 执行器已经实现；尚未进行
> PSA-GRPO 参数更新。`PSA-GRPO` 是当前项目内的工作名（Pivotal-State Action
> Credit GRPO），不宣称学术首创。

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

### 3. Outcome-blind 选择与逐步公开回放

`select_pivotal_states.py` 直接扫描未截断的 `sampling_audit.jsonl`，不读取
`strict`、`terminal_utility` 或 `policy_reward`。它用固定 seed 对 task 和裸
`branch_uid` 做哈希排序，跨 global step 去重，先为每题选一个 state、再选第二个，
每题最多两个。选择结果绑定 input SHA256，并保留 line、trajectory index 与 event
index；这里的 pivotal 标签可以使用 Actor 已采样的当前动作，但绝不使用终局结果。

```bash
.venv/bin/python scripts/select_pivotal_states.py \
  --input "$RUN/sampling_audit.jsonl" \
  --output "$RUN/pivotal_selection.json" \
  --seed 20260811 \
  --max-states 50

.venv/bin/python scripts/verify_pivotal_replay.py \
  --selection "$RUN/pivotal_selection.json" \
  --input "$RUN/sampling_audit.jsonl" \
  --environment-manifest data/environment.json \
  --output "$RUN/pivotal_live_replay.json"
```

`verify_pivotal_replay.py` 会在固定 Environment manifest 下执行：

1. 在创建任何环境前，完整核对 selection schema、input hash 与每个精确 locator；
2. `reset(task_id)`，要求 Observation v2；
3. 逐个执行 exact accepted action；
4. 每一步比较 before/after public observation SHA256；
5. 同时核对本地 manifest 和服务端 reset 返回的 manifest digest；
6. 在正常、环境异常和响应序列化异常路径释放环境租约；
7. 强制 `N selected -> N resolved -> N results`，任一 replay/release 失败都返回非零；
8. 结果只写 hash、计数和异常类型，不保存服务端 raw payload。

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

## 固定策略可行性结果

2026-08-11 在固定 SFT actor 上完成了 `lr=0`、25-step 的状态覆盖采集。该运行只验证
producer、动态采样和 replay 合同，不是性能实验：

| 指标 | 结果 |
| --- | ---: |
| generated / optimizer groups | 84 / 50 |
| optimizer group 利用率 | 59.52% |
| infrastructure-invalid groups | 1 / 84 (1.19%) |
| exact-contract pivotal states | 2,854 |
| unique tasks | 84 |
| task-cap@2 states | 168 |

唯一基础设施无效组来自 task 17349 的 `reward_unverifiable`。对 87 个 smoke 前缀执行
live reset + replay 时，87/87 的公开状态 hash 一致，且所有环境租约均释放。完整
25-step 审计未启用 actor prompt token capture，因此只证明“可找到并回放状态”，不能
直接用于主动分叉或训练；主动 suffix 必须从启用
`configs/agent_loop_active_capture.yaml` 的新采集生成。

## 启动门槛

历史日志的 `exact-contract states = 0` 表示“旧 producer 没记录新合同”，不是证明
中间状态不存在。下一步先用固定 SFT policy 采集新审计轨迹，不更新权重：

1. 已完成 1-update 工程 smoke 与 `lr=0` 的 25-step 固定策略状态采集；
2. 下一次 1-update 采集显式启用 actor prompt token capture，要求候选事件的 token、
   prompt hash、turn span 和 branch identity 全部一致；
3. 从该新产物按 outcome-blind 规则选择两个 state，并逐个执行 live reset + replay，
   public hash 匹配率必须为 100%；
4. 通过后运行 `2 states x 4 suffixes` 机械 smoke，要求 8/8 结果齐全、prompt echo 精确、
   tensor 对齐且所有环境租约释放；
5. 后续扩大主动 suffix branching 时，每个状态采 `K>=4`，要求有效 mixed group 比例
   `>=40%`、真实基础设施无效率 `<=5%`；
6. 只有主动分叉产生的 suffix tensor 可以训练。历史 sampling-audit JSON 永远不能直接
   当作训练数据。

主动分叉 runner 已把 policy scope 绑定到 actor checkpoint、live vLLM 元数据和完整
decoding config；尚未在远端真实 vLLM/Environment 上完成协议 smoke，所以仍不能把
本地 fake 测试当成可训练性证据。

### 2 x 4 机械采集命令

先启动加载 `$ACTOR` 的独立 vLLM。该进程必须显式带
`--generation-config vllm`，并把完整启动命令保存到 `$RUN/server_launch_command.txt`；
OpenAI API 不能证明这个启动参数，因此 summary 会保留
`generation_config_api_attested=false`。随后从 live `/version`、`/v1/models` 和实际 actor 目录
物化合同。materializer 默认读取
`configs/active_suffix_decoding_template.json`，不接受手填 backend digest：

```bash
.venv/bin/python scripts/select_pivotal_states.py \
  --input "$RUN/sampling_audit.jsonl" \
  --output "$RUN/pivotal_selection_2.json" \
  --seed 20260811 \
  --max-states 2 \
  --require-prompt-capture

.venv/bin/python scripts/materialize_active_suffix_contract.py \
  --actor-checkpoint "$ACTOR" \
  --served-model "$SERVED_MODEL" \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --backend-output "$RUN/sampling_backend_contract.json" \
  --decoding-output "$RUN/active_suffix_decoding.json"

.venv/bin/python scripts/build_active_branch_plan.py \
  --selection "$RUN/pivotal_selection_2.json" \
  --input "$RUN/sampling_audit.jsonl" \
  --decoding-config "$RUN/active_suffix_decoding.json" \
  --actor-checkpoint "$ACTOR" \
  --seed 20260811 \
  --suffixes-per-state 4 \
  --output "$RUN/active_branch_plan.json"

.venv/bin/python scripts/collect_active_suffixes.py \
  --plan "$RUN/active_branch_plan.json" \
  --selection "$RUN/pivotal_selection_2.json" \
  --input "$RUN/sampling_audit.jsonl" \
  --actor-checkpoint "$ACTOR" \
  --sampling-backend-contract "$RUN/sampling_backend_contract.json" \
  --served-model "$SERVED_MODEL" \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --environment-base-url http://127.0.0.1:5700 \
  --vllm-timeout 180 \
  --environment-timeout 60 \
  --expected-states 2 \
  --suffixes-per-state 4 \
  --output "$RUN/active_suffixes.jsonl" \
  --summary-output "$RUN/active_suffix_summary.json"
```

这一步只有 `8` 条 suffix rollout，不创建 optimizer，也不更新权重。每个 suffix 使用独立
environment lease；runner 恢复完整 prefix counter，并按 veRL 0.8 的剩余上下文公式动态计算
每轮 `max_tokens`。首轮使用 plan suffix seed，后续轮使用绑定到 suffix/turn 的 SHA256 seed。
任何 actor、prompt、replay hash、backend、token echo、tensor alignment 或 `K` 合同不一致都会
写入有界错误码；只要存在 infrastructure-invalid suffix，CLI 会在写完 artifact 后非零退出。

该逐轮 seed 方案是主动分叉实验的受控采样策略，不宣称复现 veRL 异步 worker 的随机数消费
顺序。vLLM 必须以 `--generation-config vllm` 启动并保存 PID、完整命令和日志；`/version`
与 `/v1/models` 只能证明 live 服务版本、模型别名和模型根目录，不能单独证明
generation-config。

当前 plan v1 是 `training_ready=false` 的机械 smoke：collection provenance 会显式记录环境
version、manifest SHA 集合和 policy-reward config SHA，但这些字段尚未进入 group UID。进入训练
前必须升级 plan v2，把 environment version 和 policy-reward SHA 纳入 plan/group identity。

## 2 x 4 机械验证结果

固定提交 `e3a7147` 上的真实 vLLM + ShopSimulator 机械验证已经完成：

```text
outputs/experiments/psa_grpo/active_suffix_smoke_sft_seed20260811_2x4_v1/
```

| 检查项 | 结果 |
| --- | ---: |
| exact prompt captures | 134 / 134 |
| selected state live replay | 2 / 2 |
| suffix cardinality | 2 x 4 = 8 / 8 |
| infrastructure-invalid suffixes | 0 / 8 |
| replay / tensor alignment | 8 / 8 |
| strict-success suffixes | 7 / 8 |
| action-credit eligible groups | 1 / 2 |

两个 group 中，一个 group 的四条 suffix 出现三种首动作且 strict outcome 有差异；另一个
group 的首动作和结果均相同，因此被正确排除。唯一失败 suffix 是模型提前结束，不是环境、
token 或 replay 故障。全部八条 suffix 都是 `optimizer_enabled=false`，产物也未包含 hidden
goal 字段。

这些数字只证明“同一精确状态和 prompt 下能够得到可校验的分叉对照”，不能作为模型
准确率或 PSA-GRPO 涨点。八条 suffix 中七条成功的比例没有统计意义，也没有运行
Final-200。

## 为什么 2 x 4 还不能训练

机械 smoke 中 task 10232 暴露了完整轨迹 credit 的同一个混淆：两条 suffix 的首个
规范化工具动作都是选择 `50cm`，但一条最终严格成功、另一条以 `-0.4` 结束；它们前
五个后续工具动作也相同，真正分叉发生在更晚的搜索与页面管理。若直接把终局 reward
贴到首动作，两个完全相同的工具动作会得到相反监督，或者被错误聚合成负 advantage。

因此 `mixed_strict` 和“首动作不同”现在都只作诊断，不能再作为训练资格。PSA 的训练
单位也明确为完整的首个 Assistant decision turn（推理 token + 工具调用），而不是只
截取工具名和参数 token。后者会切断前置推理 token 对动作概率的贡献，只能作为后续
有偏 surrogate 消融。

## Nested PSA 合同

新增的 nested collector 把一次对照拆成两个阶段：

1. Stage 1 从相同 exact branch 独立采样 `K=4` 个首 Assistant decision，并保存完整
   token、old logprob、规范化动作、proposal multiplicity 和内容 hash；结构不可重放的
   proposal 会按预注册规则排除整个 state，不单独丢样本，也不补采。
2. Stage 2 对每个 exact decision 使用新环境租约，重放原 prefix 并强制执行冻结的首
   decision；只有 post-action public state、actor prompt token 和 Harness snapshot 全部
   一致，才从该边界采 continuation。
3. 正式合同固定每个 decision 八条 continuation：索引 `0..3` 只用于估计训练目标，
   `4..7` 只用于 signal gate，gate fold 永不回填到 advantage。相同 state 和 slot 在
   不同 decision 间共用预注册随机种子，减少 continuation 噪声。
4. `assistant_final`、并行调用、未知/畸形工具、`think`、立即购买、Guard 上限和
   max-step 等模型行为均有显式路径；只有 token/span/logprob、环境、Reward 或 replay
   无法验证时才是无效样本。

训练侧先用 train fold 计算每个 decision 的均值与均值方差，再按同一 state 内的
proposal multiplicity 估计组均值。可见的 continuation 噪声会从 decision 间方差中
扣除，并用可靠度系数收缩到组均值；最后采用保持 proposal 加权零均值的有界
advantage，不再对很小的 reward 差强行做单位标准化。每个 decision 只产生一个训练
样本，continuation token 永远不进入该 decision 的 loss。

Estimator 的纯统计入口始终输出 `training_ready=false`。只有 artifact adapter 重新核对
actor、environment、plan、selection、Stage-1、backend、prompt/span、fresh lease、release、
Reward 计算和 outcome-blind exclusion 后，才可能放行 optimizer。正式信号门槛至少要求
32 个独立 task/state、state/task ESS 均不低于 32、无效率不超过 5%，并要求 gate fold
上的 top-vs-bottom decision 差值、排序一致性、task-cluster bootstrap 和置换检验同时
通过。未达到门槛时保留全部工件，但不进行参数更新。

在这些门槛通过前，报告中的 `training_ready` 固定为 `false`，也不运行 Final-200。

## Capture-only 真实验证

固定提交 `b1abb76` 上的一步 capture-only smoke 已在 GPU 1 完成：

```text
outputs/experiments/psa_grpo/psa_capture_only_smoke_sft_seed3407_1u_20260811_v2/
```

本次实际生成 `2 groups / 8 trajectories`，记录了 `98 / 98` 个精确 actor
prompt token capture。训练日志明确记录 `training/capture_only=1` 和
`training/optimizer_updated=0`，且没有产生 checkpoint、old/ref logprob、advantage 或
actor update 工件。这证明固定策略采集已不再借助 `lr=0` 模拟，而是从
Trainer 中真正断开了优化路径。

八条中有一条提前结束的模型负样本，Reward、prompt capture 和 replay 完整，
但末尾工具回合缺少 observation，因此 `turn_span_valid=false`。该轨迹仍作为轨迹
级负样本审计，但其 decision 不会进入局部 credit；不会为了满足样本数而补采。

## Nested 2 x 4 x 8 结果

在同一提交上，对已审计的两个 pivotal state 运行了真实 vLLM +
ShopSimulator nested mechanical smoke：

```text
outputs/experiments/psa_grpo/nested_suffix_smoke_sft_seed20260811_2x4x8_v1/
```

| 检查项 | 结果 |
| --- | ---: |
| state / Stage-1 proposal | 2 / 8 |
| distinct exact decision | 8 |
| continuation cardinality | 8 x 8 = 64 / 64 |
| fresh lease / verified release | 64 / 64 |
| post-action prompt + Harness parity | 64 / 64 |
| learning-valid continuation | 64 / 64 |
| infrastructure / reward invalid | 0 / 64 |
| strict success（仅诊断） | 45 / 64 |
| model failure（保留负奖励） | 7 / 64 |

Collector 的 cardinality、post-action parity、fresh lease 和 train/gate fold 检查全部通过。
Estimator 也完成了 artifact attestation，但按预注册门槛正确输出
`structural_ready=false`、`signal_ready=false`、`training_ready=false` 和
`optimizer_unlock_allowed=false`：当前只有 2 个 task/state，距离 32 个独立
task/state 及 ESS 门槛还很远。

另外 `scale_collection_ready=false` 仍是有意的阻断：当前 collector 没有可恢复的流式
journal，且会对每条 continuation 重新计算 actor tree hash。在这两项改为可中断
恢复的原子提交和起止完整 attestation 前，不启动 32-state 正式采集。

Gate fold 上当前观察到的 held-out delta 仅来自 1 个可识别 state，不具备统计
意义，不作为涨点证据。本轮没有调用 optimizer，没有评测 Final-200。
