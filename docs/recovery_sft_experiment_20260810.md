# 恢复型 SFT 实验报告

## 结论

恢复型 SFT V1 没有超过原 SFT 的严格成功率，不能作为当前正式模型。
完整恢复增量把 tuning-200 从 `120/200` 降到 `113/200`。它显著减少了循环、非法动作
和平均步骤，但把 `wrong_purchase` 从 5 增加到 15、`partial_alternative_purchase`
从 24 增加到 41，说明模型学会了更快收尾，却更容易在候选或规格尚未确认时购买。

对同一个 LoRA 增量做预先限定的强度线搜索后，`0.25x` 只能回到 `120/200`，没有净提升；
`0.5x` 为 `117/200`。因此停止继续调缩放系数，不运行 Final-200，继续保留原 SFT 作为
当前正式模型。

| 模型 | strict | mean Reward | guard | repeat | partial | wrong | 平均步骤 | gains / losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 SFT | 120/200 | 0.4559 | 86 | 35 | 24 | 5 | 11.88 | - |
| Recovery `0.25x` | 120/200 | 0.4761 | 80 | 30 | 22 | 7 | 11.68 | 8 / 8 |
| Recovery `0.5x` | 117/200 | 0.4418 | 79 | 32 | 24 | 8 | 11.79 | 9 / 12 |
| Recovery `1.0x` | 113/200 | 0.4490 | 54 | 13 | 41 | 15 | 7.09 | 9 / 16 |

## 数据构建

- 固定代码：训练与数据构建 commit `ba4226f2f3f475415b91a189da4f3582494d458b`。
- 来源：GRPO V2 正式实验 seed 3407/3408 的保存 rollout，共 1,600 条、400 个 prompt group。
- 其中有 239 个 mixed-success group、208 个 mixed-success task。
- 只允许无 guard、无重复动作、Reward v3 严格 Gold 的 sibling 作为示范，得到 194 条。
- 194/194 条通过 ShopSimulator 逐动作回放，全部再次得到严格 Gold。
- 与 tuning-200 和 Final-200 合计 400 个 held-out task 的重叠为 0。
- recovery 切分为 170 train + 24 validation；与原 SFT 混合后为 555 train + 67 validation。
- 实际 tokenize 后训练保留 553/555；丢弃的两条是原 SFT 中已有的超长样本，194 条恢复数据
  全部进入训练。
- 监督模式：58 条只监督购买提交，114 条监督规格选择与提交，22 条监督候选、规格与提交。

数据目录：
`outputs/experiments/recovery_sft/dataset_v1_20260810/`。其中 `metadata.json` 保存输入哈希、
Git SHA、切分、泄漏检查和文件哈希，`replay_results.jsonl` 保存逐题回放结果。

## 训练

训练从原 SFT merged checkpoint 出发，而不是从 GRPO V2 或 Final-200 结果出发：

- base：`outputs/models/sft-merged-dsv4flash-385-pro6000-20260806_020248`
- LoRA：`r=16`、`alpha=32`、dropout `0.05`
- `lr=2e-5`、1 epoch、gradient accumulation 8、BF16、SDPA
- 70 个 optimizer update，seed `20260810`
- `train_loss=0.1555`，Trainer `eval_loss=0.1966`
- 40.7 分钟；PyTorch peak allocated 72.78 GiB，外部 `nvidia-smi` 观察峰值约 95.8 GiB
- 186/186 个 LoRA-B 张量非零，最大绝对值 `5.2491e-4`

单步 smoke 因 warmup 后所有 LoRA-B 仍为零，被明确判为无效；两步、零 warmup 的 smoke
确认 186/186 个 LoRA-B 张量非零后，才启动正式训练。

训练目录：
`outputs/experiments/recovery_sft/recovery_lora_lr2e5_1ep_seed20260810_v1/`。

## 缩放实验

完整 adapter 的失败表现为明显的强度问题，因此没有重新训练，而是从同一个 adapter 做
LoRA delta 插值。合并满足：

`W_merged = W_sft + scale * (alpha / r) * B @ A`

`scripts/merge_lora_adapter.py` 在 commit
`e0e5e63703869230e2dde8f69c218c30503b974c` 增加 `--adapter-scale`，并在
`merge_manifest.json` 保存 scale 和被缩放的 186 个 adapter entry。远端完整测试为
181 tests，`OK (skipped=1)`。

三个模型都在同一 tuning-200、温度 0、35 环境步骤、相同上下文和观察预算下评测。
配对置信区间均跨过 0：`0.25x` 的 95% CI 为 `[-4.0, 4.0]` 个百分点；`0.5x` 为
`[-6.0, 3.0]`；`1.0x` 为 `[-8.5, 1.5]`。没有可靠的 strict 提升证据。

## 轨迹解释

完整 `1.0x` 的 16 个 strict loss 中，9 个属于购买选择退化；整体 wrong/partial purchase
同步上升。恢复样本只提供正动作标签，能够教模型在正确商品和规格后提交购买，却不能表达
“当前候选还没有满足全部约束，因此不要购买”。当这类 late-action 标签占比和更新强度过高
时，2B 模型把“结束探索并购买”泛化到了错误的第一候选。

`0.25x` 把这种偏置压低后回到 8 gains / 8 losses。它在一组任务上退出循环，同时在另一组
任务上增加循环和 guard，strict 恰好抵消。平均 Reward 更高不能替代严格 Gold，因此不会把
这一结果包装成提升。

完整证据位于各评测目录的 `summary.json`、`trajectories.jsonl`、`paired_vs_sft.json`、
`gain_loss_audit.json` 和 `gain_loss_audit.md`。

## 下一步判断

不再沿用“只增加正向 late-action 标签”的恢复 SFT V1，也不在 tuning-200 上继续细调缩放
系数。下一轮应从原 SFT 出发，做能显式利用失败负反馈的受控实验：优先测试小规模 DAPO
更新，将 Gold、partial、wrong、循环和非法终止继续映射到现有训练侧 policy reward，并把
错误购买作为有效负样本，而不是再用正向 SFT 间接矫正。先跑固定 25-update pilot；只有
tuning strict 超过 120 且 wrong purchase 不高于 5，才进入正式训练和 Final-200。
