# Data collection

## Goal

The SFT stage needs complete examples of a shopping agent using tools correctly:
searching, opening products, inspecting evidence, choosing options and ending
with a valid purchase. The repository contains the accepted action-only
trajectories, not historical failed collection attempts.

## How the dataset was produced

The promoted collection used ShopSimulator Environment v2.1, Reward v3 and
`deepseek-v4-flash` as the teacher. One four-worker run produced 616 raw trajectories.
Every trajectory executed its actions in ShopSimulator during collection. The
saved result was accepted only when Environment v2.1 returned a valid Reward v3
gold purchase; no second model judged whether the trajectory succeeded.

Collection audit:

| Item | Value |
|---|---:|
| Raw trajectories | 616 |
| Unique task IDs | 616 |
| Accepted gold trajectories | 428 |
| Acceptance rate | 69.5% |
| Mean steps | 11.64 |
| Guard-rejected calls in raw audit | 327 |
| Multi-call truncations in raw audit | 28 |
| HTTP 400 responses | 2 |
| Collection errors | 2 |

The 428 accepted trajectories were split into 385 training and 43 validation
rows. Guard-rejected calls and their synthetic guard observations were removed
from the SFT rows, as were private reasoning fields and terminal reward details.
The accepted task IDs do not overlap the frozen evaluation set. They were drawn
from `data/grpo/train.jsonl`, so all 428 do overlap the GRPO training task pool.

## Frozen deliverables

| File | Rows | SHA-256 |
|---|---:|---|
| `data/sft/train.jsonl` | 385 | `f9485cf576dd1a40b6cd7d6652d7843b28ac433a83ba00698352671bb5b5a8e2` |
| `data/sft/validation.jsonl` | 43 | `4f5643fee15a1fd4128310b21acb5b71087f22d9d5576ab1469d59e41f826d53` |

The aggregate raw collection had SHA-256
`84c7c92d585bcb6f7ee921064ce06c586a7765d5a110f585c0cdc371510fb947`;
the accepted aggregate had SHA-256
`c2ae76920574bf0c8a77acaaf170da672435c5388b120740b9f5702cd07435a1`.
Raw teacher responses are intentionally not part of the beginner repository.

## Run a new collection

Start ShopSimulator, configure an OpenAI-compatible Teacher endpoint, and run:

```bash
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY=your-key

python scripts/collect_sft_data.py \
  --tasks data/grpo/train.jsonl \
  --output-dir outputs/sft-collection \
  --model deepseek-v4-flash \
  --target-accepted 428 \
  --workers 4
```

`raw.jsonl` is the resumable source of truth. Running the same command again
skips completed task attempts and rebuilds all derived files:

```text
outputs/sft-collection/
  raw.jsonl           complete Teacher responses and environment results
  accepted.jsonl      strict Reward v3 gold trajectories
  rejected.jsonl      task IDs and deterministic rejection reasons
  reject_stats.json   aggregate acceptance audit
  sft.jsonl           sanitized training rows before splitting
  train.jsonl         task-disjoint training split
  validation.jsonl    task-disjoint validation split
  metadata.json       row counts, configuration and SHA-256 hashes
```

The command removes all task IDs listed in `data/evaluation/tasks.jsonl` before
collection and checks again while building artifacts. It also keeps at most one
accepted trajectory per task. To rebuild the derived files without contacting
the Teacher or environment, run:

```bash
python scripts/collect_sft_data.py \
  --build-only \
  --output-dir outputs/sft-collection
```

Only copy `train.jsonl`, `validation.jsonl` and their metadata into `data/sft/`
after reviewing the collection audit. Raw Teacher responses remain in
`outputs/` and should not be committed.

## What a training row contains

Each JSONL row is a chat trajectory with:

- the shopping instruction;
- assistant tool calls;
- ShopSimulator tool observations;
- the final terminal action;
- metadata tying the row to Environment v2.1 and Reward v3.

During SFT, user and tool tokens are masked. Loss is computed only on assistant
actions. See [SFT](sft.md) for the exact training recipe.
