# Documentation

## Start here

1. [Project retrospective](project-retrospective.md) explains the motivation,
   STAR narrative, failed interventions, measured outcomes and claim boundary.
2. [Experiment ledger](../experiments/comparison.md) separates current runs
   from upstream historical numbers.
3. [Evaluation](evaluation.md) defines the fixed denominator, input isolation,
   rubric and paired-comparison contract.

## Workflow guides

| Stage | Guide | Purpose |
|---|---|---|
| Data | [Data collection](data-collection.md) | Build and audit executable teacher trajectories |
| SFT | [Action-only LoRA SFT](sft.md) | Train only on Assistant action tokens |
| Reward | [Reward v3](reward-v3.md) | Specify deterministic terminal reward and invalidity |
| GRPO | [veRL GRPO](grpo.md) | Run online multi-turn policy optimization |
| Evaluation | [Fixed-denominator evaluation](evaluation.md) | Compare policies without dropping failures |

## Experiment reports

Read these in causal order rather than as independent feature lists:

1. [GRPO V2 signal repair and paired 11/13 audit](grpo_v2_experiment_20260810.md)
2. [Recovery SFT](recovery_sft_experiment_20260810.md)
3. [DAPO Clip-Higher pilot](dapo.md)
4. [Nested PSA credit audit and three confirmations](psa_grpo.md)

The sequence matters: GRPO failure was diagnosed before choosing Recovery SFT;
Recovery SFT failure motivated a controlled DAPO arm; terminal-credit ambiguity
then motivated Nested PSA. None of the three interventions passed its promotion
gate, so none should be described as a performance improvement.

## Visual artifacts

- [Interactive Final-200 dashboard](evaluation-dashboard.html)
- [`images/project-overview-pipeline.png`](images/project-overview-pipeline.png)
- [`images/shopsimulator-overview.png`](images/shopsimulator-overview.png)
- [`images/reward-v3-decision-rules.png`](images/reward-v3-decision-rules.png)

Final-200 has been inspected during repeated regression analysis and is no
longer a blind test. Use it only as a known regression set.
