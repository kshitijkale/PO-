# Mutation Beam Search

## The Problem

The reflection LLM is stochastic. Same input, different random seeds → wildly different proposed mutations. Some are great, some are garbage. GEPA bets everything on a single sample from this high-variance distribution. If the sample is bad, the entire iteration is wasted.

This is the most direct source of noise in GEPA, and no idea in the original folder addresses it. The original ideas all try to improve the *input* to reflection (better minibatches, better prompts, better selection). This idea improves the *output* by sampling more.

## The Fix

Sample K mutations from the same reflection, evaluate all K on the minibatch, keep the best.

```
1. Run reflection as normal → get reflective dataset
2. Call propose_new_texts() K times (K=3-5) with temperature > 0
   → get K candidate mutations
3. Evaluate all K on the same minibatch (the one used for reflection)
4. Keep the one with highest minibatch score
5. Run that one through the normal acceptance test + val gate
```

## Why This Works

Variance reduction, pure and simple.

If each mutation has a 30% chance of being "good" (would pass val gate), then:
- 1 sample: 30% success probability
- 3 samples: 1 - 0.7^3 = 66% success probability
- 5 samples: 1 - 0.7^5 = 83% success probability

You're dramatically increasing the acceptance rate per reflection cycle. Each reflection cycle's diagnostic work is amortized over K proposals instead of 1.

## The Cost

K extra proposal LLM calls + K extra minibatch evaluations.

For K=3 with minibatch size 3:
- 2 extra LLM proposal calls (~$0.02-0.20 total)
- 6 extra minibatch evaluations (~$0.006-0.06 total)
- Total extra cost: ~$0.03-0.26

Against a wasted valset evaluation (20+ examples, ~$0.02-0.20) on a bad mutation that would have been rejected, this is a bargain. You spend a little more per iteration to waste far less on bad candidates reaching val.

## Where It Intervenes

`ReflectiveMutationProposer.propose_new_texts()` in `src/gepa/proposer/reflective_mutation/reflective_mutation.py`.

Currently this calls `InstructionProposalSignature.run_with_metadata()` once per component. The change: call it K times per component, collect K proposals, evaluate all K, return the best.

```python
def propose_new_texts_beam(
    self,
    curr_prog: dict[str, str],
    reflective_dataset: dict[str, list],
    predictor_names: list[str],
    K: int = 3,
    eval_fn=None,  # evaluator for minibatch scoring
    minibatch=None,
) -> dict[str, str]:
    candidates = []
    for _ in range(K):
        new_texts = self.propose_new_texts(
            curr_prog, reflective_dataset, predictor_names
        )
        candidates.append(new_texts)

    if eval_fn is None or minibatch is None:
        # Fallback: return first candidate (no beam)
        return candidates[0]

    # Evaluate all K candidates on the reflection minibatch
    scores = []
    for cand in candidates:
        merged = {**curr_prog, **cand}
        result = eval_fn(minibatch, merged)
        scores.append(sum(result.scores))

    return candidates[scores.index(max(scores))]
```

The K proposals share the same reflective dataset — the expensive diagnosis work is done once. Only the proposal generation (the LLM call that writes new candidate text) is repeated.

## Key Insight: Proposals Are Embarrassingly Parallel

The K proposal calls are independent. They can be batched in a single API call (using `n=K` parameter in OpenAI-style APIs) or sent as parallel async requests. Wall-clock time increase is near zero if the API supports batching.

## Interaction with the Acceptance Test

The beam winner is selected by *minibatch* score. This biases toward candidates that do well on the specific reflection minibatch — potentially overfitting to it. But that's exactly what the val gate is for: the beam winner still faces the full val evaluation.

The expected interaction: beam search increases the probability that at least one mutation is genuinely good (not just lucky). The val gate still filters out candidates that overfit to the minibatch. The net effect: more candidates reach val, and more of those pass.

## Design Decisions

**How to generate diversity across the K proposals.** Options:
1. **Temperature sampling** (simplest): use temperature > 0 for the proposal LLM. Each call samples differently.
2. **Explicit diversity instruction**: add "propose a DIFFERENT approach than: {summary of previous proposals}" to the prompt for proposals 2-K.
3. **Component variation**: for multi-component candidates, vary which component each proposal focuses on.

Start with (1). Temperature sampling is the simplest and produces natural diversity without prompt engineering. If the proposals cluster too tightly, add (2).

**K value.** K=3 is the sweet spot. K=1 is current GEPA. K=5 gives diminishing returns (the best of 5 is only marginally better than the best of 3 in most distributions). K=3 triples the acceptance rate at 3x proposal cost.

**Should beam search apply to the merge proposer too?** Yes, if budget allows. The merge proposer (`src/gepa/proposer/merge.py`) constructs merged candidates from Pareto front pairs. Generating K merge variants and picking the best is equally valuable.

## What Can Go Wrong

**All K proposals are bad in the same way.** If the reflection diagnosis is wrong (misidentifies the root cause), all K mutations address the wrong thing. Beam search over a bad distribution doesn't help. This is why this idea composes well with Idea 1 (explanation accumulation) — fix the diagnosis quality AND sample more from the proposal.

**Minibatch overfitting in beam selection.** The beam winner is the one that scores highest on 3 examples. With K=3 candidates, you're picking the best of 3 on 3 examples — this could select a candidate that got lucky. Mitigation: if K > 3, use a slightly larger evaluation set for beam selection (e.g., 5-6 examples instead of the reflection minibatch of 3).

## Expected Outcome

- **Higher acceptance rate**: More proposals pass the val gate per reflection cycle.
- **Faster convergence**: Fewer wasted iterations mean faster progress toward the optimum.
- **Strictly better than baseline**: At worst (all K proposals identical), you get the same result as standard GEPA. Beam search cannot hurt — it only helps.

## Ablation Design

1. **Standard GEPA, K=1** (control)
2. **Beam search, K=2**
3. **Beam search, K=3** (recommended default)
4. **Beam search, K=5**
5. **Beam search, K=3 with diversity prompting** (explicit instruction to be different)
6. **Random K=3** (pick a random proposal instead of the best — controls for "is it just the extra evaluation budget?")

Compare (1) vs (3) for the headline. Compare (3) vs (6) to confirm that selection matters (it's not just more lottery tickets). Compare (3) vs (5) to test whether diversity prompting helps.
