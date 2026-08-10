# GRPO V2 信号修复实验报告

## 结论

本轮修复了 GRPO 的训练信号和采样利用率，但没有在最终严格 Gold 指标上超过 SFT。
旧 Final-200 是已知回归集，不是盲测；最终结果必须记录为 **GRPO V2 未提升**。

| 模型 | Final-200 strict | guard | repeat loop | wrong purchase |
| --- | ---: | ---: | ---: | ---: |
| SFT | 117/200 | 69 | 27 | 4 |
| GRPO V1 | 114/200 | 92 | 28 | 6 |
| GRPO V2, seed 3408, step 50 | 115/200 | 89 | 28 | 4 |

GRPO V2 相对 SFT 的配对结果为 11 gains、13 losses，严格成功率差为 -1.0 个百分点。
精确 McNemar 检验 `p=0.8388197422`；固定种子 20260809、20,000 次配对 bootstrap
得到 95% 置信区间 `[-5.5, 3.5]` 个百分点。没有可靠提升证据。

## 实现与版本

- 实验分支：`experiment/grpo-v2-signal-fix`
- 正式实验固定 commit：`5ac6359b433b332fb0203e64528b4caba8b7df33`
- 核心信号修复 commit：`a45517c`
- Reward v3 测试更新：`199451f`
- Blackwell vLLM 兼容修复：`921eb23`
- float32 奖励一致性容差：`9db5bc0`
- 配对统计与 McNemar：`5ac6359`

训练侧新增 `shopping-policy-reward-v1`，保持 ShopSimulator Reward v3 和正式评测逻辑不变。
正常终局只加入有界 guard/repeat 惩罚；模型导致的提前结束、max steps、上下文上限、
并行工具调用和连续非法动作成为有效负样本；真正基础设施错误才标为 invalid。动态采样按
`policy_reward` 而不是原始 `terminal_utility` 判断组内方差，并校验张量奖励与元数据奖励一致。

## 数据与基线

- 切分种子：`20260809`
- GRPO train：850 题
- tuning：原 validation-50 加 150 题，共 200 题
- Final-200：与 train、tuning 均无重叠
- `train850.parquet` SHA256：`6869836564bd0f375eee56925ecd2a47a4669b6cee3288c8e336bfa41212744e`
- `tuning200.jsonl` SHA256：`3bfee3160e55019deeda4c871178f3c535c7a0ae46ef673f15a5a6ab3475d8c7`

SFT 在 tuning-200 上为 120/200，guard 86，repeat loop 35，wrong purchase 5；
在旧 Final-200 上为 117/200。

## Pilot

| 配方 | 有效组 | invalid 组 | tuning strict | mean Reward | guard | repeat | wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0, lr 1e-6, n=4 | 60.20% | 4.08% | 123/200 | 0.4939 | 64 | 27 | 8 |
| P1, lr 5e-7, n=4 | 70.00% | 1.25% | 118/200 | 0.4579 | 77 | 32 | 7 |
| P2, lr 5e-7, n=8 | 77.14% | 1.43% | 120/200 | 0.4680 | 61 | 29 | 8 |

P0 按 strict 指标胜出。P2 首次启动因环境只有 8 个 slot，而一次生成需要 16 条并行
trajectory，失败于环境资源不足；失败目录被保留。将 `SHOPSIM_ENV_SLOTS=16` 后，P2
原配方重跑成功。这个失败不是训练效果结果。

P0 相对 SFT 在 tuning-200 上有 8 gains、5 losses，但 95% 配对区间仍跨过 0；
同时 wrong purchase 从 5 增至 8，因此 pilot 的提升并不稳健。

## 正式实验

两个 seed 均使用 P0 配方运行 100 updates，并保存 25/50/75/100 checkpoint。

| seed | val strict at 25/50/75/100 | 选择 | tuning strict | gains/losses vs SFT |
| --- | --- | --- | ---: | ---: |
| 3407 | 32/28/26/31 | step 25 | 116/200 | 5/9 |
| 3408 | 30/32/28/32 | step 50 | 120/200 | 7/7 |

seed 3408 的 step 50 与 step 100 strict 相同；step 50 的 validation 平均 Reward
为 0.4767，高于 step 100 的 0.4393，因此选择 step 50。两个正式运行的有效组率分别为
60.00% 和 63.41%，invalid 组率分别为 2.78% 和 2.79%。四个验证点的 `actor/ppo_kl`
均低于 0.02，所以没有触发额外 KL 锚定实验。

## 验证判定

- 信号与采样门槛通过：有效组率大于 40%，invalid 组率小于 5%。
- tuning 门槛仅持平：正式最优候选为 120/200，与 SFT 相同。
- Final 最低门槛失败：115/200 未超过 SFT 的 117/200，且 losses 多于 gains。
- 可靠提升目标失败：未达到 121/200，guard 89 超过 69，repeat 28 超过 27。
- wrong purchase 为 4，单项达到目标，但不能抵消 strict 与合法性退化。

## 产物

远端根目录：`/home/dy/wh/shopping-grpo-longhorizon/outputs/experiments/grpo_v2/`

- smoke：`smoke_seed3407_20260810_v2/`
- SFT tuning 基线：`sft_tuning200_20260809_v1/`
- pilots：`p0_lr1e6_n4_seed3407_25u_20260810/`、
  `p1_lr5e7_n4_seed3407_25u_20260810/`、`p2_lr5e7_n8_seed3407_25u_20260810_v2/`
- 正式 seed 3407：`formal_p0_lr1e6_n4_seed3407_100u_20260810/`
- 正式 seed 3408：`formal_p0_lr1e6_n4_seed3408_100u_20260810/`
- 最终已知回归集：
  `formal_p0_lr1e6_n4_seed3408_100u_20260810/selected/final200_known_regression/`

每个训练目录保留 resolved config、run manifest、完整日志、sampling audit、rollout、
validation 和 checkpoint。最终目录包含 `summary.json`、`trajectories.jsonl` 和
`paired_vs_sft.json`。

## 下一步判断

这次实验说明主要的工程缺陷已经修好，但仅靠当前 outcome reward 和 GRPO 更新仍不能给
2B 模型带来稳健提升。validation-50 对 checkpoint 的排序也没有稳定迁移到 tuning-200。
下一轮不应继续盲目扩大 GRPO 步数；更有信息量的方向是从 11 gains、13 losses 以及
guard/repeat 轨迹中构造恢复型 SFT 数据，或在保持同一评测口径下尝试 DAPO，并把
wrong purchase 与合法性作为独立约束监控。
