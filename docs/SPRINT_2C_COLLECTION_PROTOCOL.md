# Sprint 2C Paired Recovery Corpus Protocol

Status: preregistered collection protocol. This phase collects data only and
does not train or implement a recovery selector.

## Frozen Policies

The base policy is the Sprint 1.5 H4 baseline: `Qwen/Qwen3-8B`,
`indexed_bounded_context_v1`, bounded history window 4, indexed admissible
actions, thinking disabled, greedy decoding, and 32 generated tokens.

The intervention is R2-two-stage-v2 at protocol commit
`1b84f922a39f58c3b6ed2ba31cb29d896486b0d0`. Stage 1 uses its frozen prompt
and 40-token budget. Stage 2 uses its frozen semantic prompt, 16-token budget,
and strict bare-ID contract `^A\d{3}$`. Each recovery branch receives one
Stage-1 call, one Stage-2 call, and at most one mapped environment action. No
retry, fallback, repair, memory, candidate ranking, or second intervention is
permitted.

## Source Pool

The source schedule is frozen in
`configs/corpus/sprint2c_h4_pool_schedule.json`. The initial pool contains 200
new `valid_seen` H4 episodes with nominal seeds beginning at 2000 and preserves
the recorded ALFWorld reset order. A single preregistered 100-episode extension
is executed only if the initial pool yields fewer than 10 successful episodes.
No other adaptive extension is allowed.

## Prefix Selection

Exactly 120 nonterminal states are selected by deterministic rules, with at
least eight environment actions remaining and no more than two states from one
source episode:

- 20 states whose frozen H4 continuation succeeds;
- 30 failed states with a past-only adjacent-repeat or two-cycle indicator;
- 30 failed, non-loop states at or above the within-episode nearest-rank Q75
  of the already-recorded H4 token entropy;
- 40 remaining failed, non-loop states sampled by deterministic family/stage
  round robin and SHA-256 rank.

If the non-loop failure entropy range is below `1e-8`, the high-uncertainty
stratum becomes a preregistered deterministic random fallback. Prefix state
fingerprints exclude nominal seed and previously consumed recovery-development
states are excluded. The manifest records the full selection rule, source
hashes, episode grouping, and decision-time feature schema.

The qualitative review order is frozen in the prefix manifest before recovery
inference. All beneficial and harmful cases are reviewed. Neutral review uses
the first 20 neutral states in that preregistered order.

## Paired Execution

Each branch point is reconstructed independently twice. Public observation,
task identity, ordered admissible actions, hidden-state evidence, and remaining
horizon must match. The continue branch must exactly reproduce the recorded H4
suffix. The recover branch executes the one R2 action and then resumes the same
H4 policy for the same remaining horizon.

Raw labels are `continue_return`, `recover_return`, and
`V_raw = recover_return - continue_return`. Intervention costs remain separate.
Sampling strata, hidden replay evidence, outcomes, future behavior, and
qualitative annotations are not selector inputs. Any later split must keep all
states sharing `episode_group_id` in the same partition.

## Gate

After exactly 120 pairs, protocol reliability and label density determine the
recommendation: at least 8 diverse beneficial examples permits selector
development; 3-7 calls for a second preregistered batch; and 0-2 indicates that
recovery opportunity is too sparse for the current selector study. The
stratified beneficial fraction is not an estimate of the natural on-policy
recovery rate.

For this gate, positive-label diversity requires at least four independent
source episodes and at least two task families. Eight or more beneficial cases
that fail this diversity check require a second preregistered batch rather than
selector development.
