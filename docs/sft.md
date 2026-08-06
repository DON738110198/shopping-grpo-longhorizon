# LoRA SFT

## Purpose

The base model can speak naturally but does not reliably follow
ShopSimulator's action protocol. Supervised fine-tuning teaches the basic
policy: issue legal tool calls, use observations as evidence, select product
variants and terminate.

## Inputs

- Base model: `Qwen/Qwen3.5-2B`
- Train data: `data/sft/train.jsonl` (385 rows)
- Validation data: `data/sft/validation.jsonl` (43 rows)
- Target: assistant tokens only; user and tool-observation tokens are masked

The data provenance and hashes are recorded in
[`data/sft/metadata.json`](../data/sft/metadata.json).

## Run

After `bash scripts/setup.sh`:

```bash
bash scripts/sft.sh
```

The launcher trains a LoRA adapter and then merges it with the base model:

```text
outputs/models/sft-lora/
outputs/models/sft-merged/
```

Default recipe:

| Setting | Value |
|---|---|
| Maximum sequence length | 24,576 |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | `1e-4` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Gradient checkpointing | enabled |
| Attention implementation | SDPA |

The long context is intentional: a training example includes the complete
multi-turn interaction. Shortening it may truncate the terminal decision or the
evidence that supports it.

## Evaluate

```bash
bash scripts/serve_model.sh outputs/models/sft-merged
bash scripts/evaluate.sh sft
```

The upstream reported checkpoint completed 141 optimizer steps. Its validation loss was
0.3365 after epoch 1, 0.3189 after epoch 2 and 0.3147 after epoch 3. The frozen
result and reproduction config are in [`experiments/sft/`](../experiments/sft/).
Those metrics do not describe a new run trained on the promoted 385/43 split.

## Output contract

GRPO starts from the merged model, not directly from the adapter:

```text
GRPO_MODEL_PATH=outputs/models/sft-merged
```

This boundary keeps the GRPO launcher independent of the SFT trainer process.
