# Experiments

This directory is a compact experiment index. Large checkpoints, complete
rollouts, manifests and logs are generated under the Git-ignored `outputs/`
tree and are not committed.

```text
baseline/      compact upstream baseline config and summary
sft/           compact upstream SFT config and summary
grpo/          compact upstream GRPO config and summary
comparison.md  current experiment ledger, provenance labels and claim boundary
```

The three compact stage directories contain **upstream historical artifacts**,
not a reproduction using the current 385/43 audited SFT split. Current
repository experiments are documented with their fixed configs, paired
statistics, failure analysis and promotion decisions in:

- [`../docs/grpo_v2_experiment_20260810.md`](../docs/grpo_v2_experiment_20260810.md)
- [`../docs/recovery_sft_experiment_20260810.md`](../docs/recovery_sft_experiment_20260810.md)
- [`../docs/dapo.md`](../docs/dapo.md)
- [`../docs/psa_grpo.md`](../docs/psa_grpo.md)

Start with [`comparison.md`](comparison.md). It is the authoritative map from
results to claims and explicitly records what did not pass.
