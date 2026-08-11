# DAPO Pilot

## Scope

DAPO is evaluated as a controlled optimization variant inside the repository's
existing GRPO stage. It starts from the original merged SFT checkpoint and uses
the same train split, ShopSimulator policy reward, rollout count, learning rate,
seed and evaluation protocol as GRPO V2 P0.

The project already had three DAPO ingredients before this pilot:

- reward-varying dynamic sampling;
- token-level policy-gradient loss aggregation (`token-mean`);
- no KL reward or actor KL loss.

[`configs/dapo.yaml`](../configs/dapo.yaml) adds the clipping differences used
by this first pilot:

- Clip-Higher: `clip_ratio_low=0.20`, `clip_ratio_high=0.28`;
- DAPO's dual-clip ceiling: `clip_ratio_c=10.0`.

This is intentionally named `dapo-clip-higher-pilot-v1`, not a reproduction of
the paper's Qwen2.5-32B math experiment. The paper's soft overlong penalty is
not enabled because veRL's multi-turn `response_length` also contains tool
observations. Penalizing that total would partly penalize ShopSimulator text
rather than only Actor-generated tokens. The existing bounded negative rewards
for `max_steps` and context hard limits remain unchanged.

Reward standard-deviation normalization also stays at the GRPO P0 value
(`false`). The Shopping reward deliberately contains small `0.02` and `0.03`
behavior differences; normalizing a low-variance group could magnify those
terms and would add a second causal change. It can be tested separately only if
Clip-Higher does not settle the question.

## Pilot

Run a one-update integration smoke first, then a 25-update pilot with:

| Setting | Value |
| --- | --- |
| Initial policy | original merged SFT |
| Train / validation | GRPO V2 train-850 / validation-50 |
| Prompt batch / rollouts | 2 / 4 |
| Learning rate | `1e-6` |
| Seed | `3407` |
| KL | disabled |
| Save / validation | step 25 |

The smoke must contain a real optimizer update, complete policy-reward audit,
no OOM/NaN and no infrastructure-invalid group explosion. The pilot proceeds to
tuning-200 only if its effective-group rate is at least 40% and true
sampling-invalid rate is at most 5%.

Promotion requires tuning strict success above the original SFT's `120/200`,
with `wrong_purchase <= 5` and no guard/repeat regression. Otherwise DAPO is
recorded as not improved and Final-200 is not run.
