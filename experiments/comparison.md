# Experiment Ledger and Claim Boundary

This file separates upstream historical numbers from experiments actually run
for this repository. Large checkpoints and full trajectories remain under the
Git-ignored `outputs/` tree; committed documents record configs, hashes,
paired statistics, failure analysis and promotion decisions.

## Current repository experiments

| Experiment | Dataset / scale | Primary result | Verdict | Evidence |
|---|---|---|---|---|
| Audited SFT collection | 616 unique raw trajectories | 428 replay-verified strict gold; 385 train / 43 validation; evaluation overlap 0 | Dataset promoted | [data collection](../docs/data-collection.md) |
| GRPO V2 | 2 seeds × 100 updates | Final-200 `115/200` vs SFT `117/200`; 11 gains / 13 losses; exact McNemar `p=.8388197422` | No reliable improvement | [GRPO V2 report](../docs/grpo_v2_experiment_20260810.md) |
| Recovery SFT | 194 replay-gold recovery rows | full `113/200`; `0.25x=120/200` vs SFT `120/200`; wrong purchase `5→7` | Not promoted; no Final evaluation | [Recovery report](../docs/recovery_sft_experiment_20260810.md) |
| DAPO Clip-Higher | 1 seed, 25 updates | tuning `119/200` vs SFT `120/200`; mean clip fraction `0.215%` | Narrow ablation rejected; no Final evaluation | [DAPO report](../docs/dapo.md) |
| General pivotal Nested PSA | 80 states, 2,128 continuations | structural gate passed; signal gate failed | Optimizer locked | [PSA report](../docs/psa_grpo.md#formal-n80-nested-psa-结果) |
| Search-decision confirmation | 80 fresh states, 2,200 continuations | 4/6 signal checks passed; ranking consistency failed | Optimizer locked | [PSA report](../docs/psa_grpo.md#search-decision-selector-v2-确认结果) |
| Query-only confirmation | 120 fresh tasks; 80 states; 1,944 continuations | structural ready; all six signal checks failed | `optimizer_unlock_allowed=false` | [PSA report](../docs/psa_grpo.md#query-only-独立确认结果) |

## Formal GRPO V2 comparison

Final-200 is now a **known regression set**, not a blind test.

| Model | Strict success | Guard rejections | Repeat loops | Wrong purchase |
|---|---:|---:|---:|---:|
| SFT | 117 / 200 | 69 | 27 | 4 |
| GRPO V1 | 114 / 200 | 92 | 28 | 6 |
| GRPO V2, selected seed/checkpoint | 115 / 200 | 89 | 28 | 4 |

GRPO V2 produced 11 gains and 13 losses relative to SFT. The strict-success
difference was `-1.0` percentage point. Exact McNemar testing gave `p=.8388`,
and a 20,000-sample paired bootstrap produced a 95% interval of
`[-5.5, +3.5]` percentage points. There is no reliable improvement evidence.

The paired trajectory audit found that 8 of the 13 lost tasks had opened the
product purchased by the successful SFT trajectory, and 7 had selected at
least one corresponding option. The regression was frequently about continued
exploration, option overwrite, loops or failure to commit the purchase rather
than a complete inability to retrieve the product.

## Upstream historical reference

The compact JSON summaries in [`baseline/`](baseline/), [`sft/`](sft/) and
[`grpo/`](grpo/) reproduce the table published by the upstream project:

| Model | Strict success | Purchase success | Mean reward |
|---|---:|---:|---:|
| Qwen3.5-2B baseline | 0.0% | 0.0% | -0.1105 |
| Upstream LoRA SFT | 60.5% | 60.5% | 0.4729 |
| Upstream GRPO step 100 | 62.0% | 62.5% | 0.5158 |

These are **historical upstream results**. They were trained with the upstream
448-row SFT recipe and are not a rerun of the current 385/43 audited dataset.
They must not be presented as the current repository owner's performance gain.

## Supported claims

- The repository provides an end-to-end, replayable and recoverable
  ShopSimulator post-training workflow.
- Data, Reward, rollout state, actor-visible tokens, local credit artifacts and
  evaluation denominators are explicitly audited.
- Controlled GRPO, Recovery SFT, DAPO and Nested PSA experiments were run and
  rejected when their preregistered gates failed.
- Negative results localize stopping, option-overwrite and continuation-noise
  failures that are hidden by a terminal score alone.

## Unsupported claims

- The current 385/43 SFT split reproduces the upstream 60.5% result.
- GRPO V2, Recovery SFT or DAPO improves over the repository's SFT policy.
- PSA-GRPO has been trained, improves performance or is an established novel
  algorithm. It remains a project working name for a credit-audit prototype.
- The 96 GB recipe is the theoretical minimum for a 2B model.
- Results generalize beyond ShopSimulator, Qwen3.5-2B or the tested single-node
  runtime.

The next performance claim requires a new unseen test set, a preregistered
training intervention that passes the signal gate and paired results from at
least two training seeds.
