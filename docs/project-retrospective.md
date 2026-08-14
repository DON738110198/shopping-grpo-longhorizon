# 项目复盘：长程购物 Agent 的可审计后训练与失败诊断

## 一句话定位

本项目不是“把 2B 模型通过 RL 做到更高分”的成功案例，而是一个把长程工具 Agent 的
数据、训练、环境重放、固定评测、失败诊断和 optimizer 放行条件连成闭环的工程与研究项目。

最终证据支持：

- 数据与运行合同可以被逐层重放、恢复和审计；
- 当前 GRPO、Recovery SFT、DAPO Clip-Higher 均没有可靠超过 SFT；
- 当前 outcome-blind Nested PSA 信用信号不稳定，不能解锁训练；
- 负结果能够定位下一步应改 credit estimator / intervention，而不是继续堆训练步数。

## STAR

### Situation：长程 Agent 的终局分数掩盖了局部失败

ShopSimulator 任务要求 Agent 在有状态网页环境中完成搜索、候选比较、详情核验、规格选择
和购买。模型可能在十几步后才失败，而失败前已经打开过正确商品、选中过正确规格，或者
只是在最后阶段继续探索、覆盖选项、循环和错过提交购买。

这与短问答 RL 有三个关键差异：

1. 一条样本包含多轮 Assistant、工具调用和长 Observation；
2. 环境槽、页面状态和 accepted action prefix 都会影响后续 Prompt；
3. 同一个终局 Reward 被广播到整条轨迹时，不能识别真正造成差异的局部决策。

### Task：先证明信号可信，再讨论训练收益

项目给自己设定了四条约束：

- 训练、tuning 与评测任务按 `task_id` 隔离；
- 基础设施错误不能伪装成模型负样本或有效零奖励；
- 每个性能结论都要有固定分母、配对轨迹和预先声明的晋级门；
- 局部 credit 只有在 fresh task、独立 continuation 和 held-out gate 上稳定时才允许更新参数。

### Action：从数据到信用门控逐层闭环

#### 1. 数据与监督

- 使用 DeepSeek V4 Flash 在真实 ShopSimulator 环境采集 616 个唯一任务轨迹；
- 逐动作执行并用 Reward v3 验收，只保留 428 条 `gold_purchase`；
- 删除教师私有推理，仅保留 Actor 可观察的工具动作；
- 以任务为单位划分 385 train / 43 validation，并固定文件 SHA-256；
- SFT 仅对 Assistant 动作 token 计算 Loss，Mask 用户与环境 Observation。

#### 2. 长程 Rollout 可靠性

- 在 veRL 0.8 上实现项目级多轮 AgentLoop、工具 parser 与 Observation projection；
- 用 tokenized lease v2、TTL、幂等 reset 和 generation guard 处理环境槽并发；
- 用 exact action prefix、public observation hash 和 replay state identity 验证中间状态；
- 捕获真实 actor prompt token 与 assistant turn span，核对 response mask、old logprob 和奖励；
- 增加 capture-only，使固定策略采集在 Trainer 内真正跳过 optimizer、checkpoint 与权重更新；
- 为 Stage 1 / Nested collection 增加 request-intent WAL、manifest、actor 起止 attestation 与无补采恢复。

#### 3. 固定评测与配对诊断

- 固定 200 题分母；失败、缺失与 `not_judged` 不从分母移除；
- 将 Environment Reward、Rubric 满足、Trajectory Judge 和 deterministic behavior 分成四个面板；
- Judge 输入移除 Reward、Gold、成功标签与 raw hidden observation；
- 对 SFT / GRPO 同 `task_id` 做 McNemar、paired bootstrap 与 gains/losses 轨迹审计。

#### 4. 从失败现象选择最小干预

项目没有在 GRPO 失败后直接换更复杂算法，而是按下面顺序验证假设：

| 问题 | 证据 | 假设 | 最小改动 | 结果与结论 |
|---|---|---|---|---|
| GRPO 有效组利用率低、模型失败被当 invalid | sampling audit 与 Reward/tensor 不一致 | 先修训练信号，性能才有可比性 | `shopping-policy-reward-v1`、动态有效组过滤、Reward 一致性校验 | 工程门通过；Final 仍 `115/200 < 117/200`，说明问题不只在基础设施 |
| GRPO losses 集中在后段 | 13 losses 中 8 个打开过 SFT 成功商品 | 模型缺少“正确候选后的恢复/收尾”监督 | 从 train rollout 构造 194 条 replay-gold Recovery SFT | `0.25x` 仅持平且 wrong purchase `5→7`；正向 late-action 标签会诱导过早购买 |
| Recovery SFT 不能表达失败负反馈 | wrong/partial purchase 上升 | 改优化规则可能比继续加正标签更合适 | 单变量 DAPO Clip-Higher 25-update pilot | `119/200 < 120/200`；clip fraction 太低，否决该窄消融，不外推到完整 DAPO |
| 终局 advantage 无法定位首决策 | 同规范动作的 suffix 可一正一负 | 需要固定决策后独立采 continuation | Nested PSA：`K=4` 首决策、每 decision `L=8`、train/gate 分离 | 三轮 structural pass、signal fail；optimizer 保持锁定 |

### Result：系统闭环成立，性能与信用假设被否决

| 结果层 | 量化证据 | 判定 |
|---|---|---|
| 数据 | 616 raw、428 strict gold、385/43；evaluation overlap 0 | 数据闭环通过 |
| GRPO V2 | two-seed 100 updates；Final `115/200` vs SFT `117/200`；11/13；`p=.839` | 无可靠提升 |
| Recovery SFT | full `113/200`；`0.25x=120/200` vs SFT `120/200` | 不晋级，不跑 Final |
| DAPO | `119/200` vs SFT `120/200`；mean `pg_clipfrac=0.215%` | 窄消融失败 |
| Query-only PSA | 120 fresh tasks、80 states、320 proposals、1,944 continuations、0 infra invalid | 工程/结构通过 |
| PSA signal | high-kappa `29.41%`、held-out delta `+0.0322`、consistency `46.81%`，六项门全失败 | 不训练，不跑 Final |

## 这项工作体现的思考

### 1. 把“代码跑通”与“方法有效”分开

环境 lease、exact replay、WAL 和 tensor contract 通过，只证明实验值得被相信；它们不能
替代性能结果。相反，只有工程合同通过以后，`115 < 117` 和 signal gate 失败才是可信负结果。

### 2. 先做配对 bad case，再选择算法

GRPO 没涨点后先审计 11 gains / 13 losses，发现主要问题是停止与规格覆盖，而非完全不会
搜索。Recovery SFT 和 DAPO 分别对应“补行为监督”和“改策略更新”两条不同假设，避免一次
同时改变数据、采样与优化器后无法解释结果。

### 3. 不根据结果放宽门槛

PSA 的 state 选择不读取 Reward、strict 或 hidden goal；train/gate continuation 分离；无效
slot 不按 outcome 补采。三轮 confirmation 未通过后停止继续切子集，不把 query-only 的某个
事后切片包装成通过。

### 4. 让错误类型决定是否进入学习

模型提前结束、循环、非法动作和错误购买是需要保留的负样本；环境超时、Reward 无法验证、
prompt/span 不一致则必须 fail closed。两类错误如果混在一个 `invalid` 中，既浪费训练数据，
也可能制造 reward hacking。

## 资源与显存解释

Qwen3.5-2B 的权重本身并不能解释 96 GB 配方。GRPO 还同时持有：

- 约 24K 的 prompt + response 序列预算；
- 每个 Prompt 四条多轮 Rollout；
- Actor 的梯度与优化器状态；
- 同卡 vLLM 的 KV Cache；
- 工具 Observation、response mask、old logprob 与对齐元数据。

因此它与单轮、短答案的 4B 医疗问答 QLoRA 不是同一内存合同。96 GB 是项目验证过的
工程配置，不是理论最低值。缩短上下文、降低 rollout 数、分离推理服务或只做离线信用审计，
都可能降低硬件门槛，但会改变吞吐或实验问题。

## 目前仍缺什么？

1. **新的未见测试集。** Final-200 已被多轮分析，只能继续作为已知回归集。
2. **属于当前 385/43 数据的 SFT 训练复现。** 上游 `60.5%` 不能算作当前数据结果。
3. **更强的局部干预或 credit estimator。** 当前 outcome-blind 首决策排序不稳定。
4. **实际 optimizer 证据。** 只有新的 signal gate 通过后，才允许 actor-only 一步 smoke 和正式训练。
5. **跨环境外推。** 当前证据只覆盖 ShopSimulator、Qwen3.5-2B 与单机运行合同。

在这些证据补齐前，最准确的项目定位是：

> 面向长程工具 Agent 的可审计后训练与失败诊断系统，而不是一个已经证明涨点的新 RL 算法。
