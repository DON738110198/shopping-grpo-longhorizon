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

## Result (2026-08-11)

The controlled pilot ran on physical GPU 1 of `dy_10.191.245.63` from fixed
commit `e09c09e8b53d7f3a4911fd4d7872c9fcfff48ce9`.

The one-update smoke completed a real optimizer update with two effective
groups, eight rollouts and no sampling-invalid group. The 25-update run then
completed with exit code zero:

| Training diagnostic | Result |
| --- | ---: |
| Effective groups | 58 / 98 (59.18%) |
| Sampling-invalid groups | 4 / 98 (4.08%) |
| Skipped updates | 2 |
| Validation-50 strict at step 25 | 31 / 50 |

The promotion evaluation used the frozen tuning-200 protocol:

| Model | Strict | Mean Reward | Guard | Repeat | Wrong purchase |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original SFT | 120 / 200 | 0.4559 | 86 | 35 | 5 |
| GRPO V2 P0 | 123 / 200 | 0.4939 | 64 | 27 | 8 |
| DAPO Clip-Higher | 119 / 200 | 0.4725 | 74 | 28 | 6 |

Against SFT, DAPO produced 7 gains and 8 losses, for a strict-success delta of
`-0.5` percentage points. The paired 95% bootstrap interval was `[-4.5, 3.5]`
percentage points and the exact McNemar p-value was `1.0`.

The pilot therefore failed both promotion requirements: strict success did not
exceed 120 and wrong purchase exceeded 5. Final-200 was not run.

This negative result is mechanically plausible. Across the 25 updates,
`actor/pg_clipfrac` averaged only `0.215%` and peaked at `0.516%`; the dual-clip
lower fraction was always zero. The changed clipping bounds were rarely active,
so Clip-Higher could not repair the stopping and purchase-selection errors seen
in the previous trajectory audit. This experiment rejects this narrow pilot,
not the full DAPO recipe: response-mask-aware soft overlong shaping was not
implemented, and only one seed and 25 updates were evaluated.

Remote artifacts are under
`outputs/experiments/dapo_v1/d0_cliphigher_lr1e6_n4_seed3407_25u_20260811_v1/`.
They include the manifest, resolved config, complete log, sampling audit,
rollouts, step-25 checkpoint, exported adapter, merged model, validation-50,
tuning-200 trajectories, summary and paired comparison. The tuning trajectory
SHA256 is `55bc20140f3fad79767029cf0d99e134ef32491df2b37dee804f74f4ca61da7d`.
