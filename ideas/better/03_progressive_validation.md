# Progressive Validation (Successive Halving for Candidates)

## The Problem

GEPA's val evaluation is all-or-nothing. A candidate either gets evaluated on the full val set or not at all. The minibatch acceptance test (`new_sum > old_sum` on 3 examples) is supposed to filter bad candidates before the expensive val evaluation, but it's a terrible filter — 3 examples, sum comparison. Many mediocre candidates pass the minibatch test and waste full-valset budget.

The budget arithmetic makes this painful. With minibatch size 3 and valset size 20:
- Minibatch eval: 3 examples (parent) + 3 examples (child) = 6 metric calls
- Val eval: 20 metric calls
- Total if child passes minibatch: 26 metric calls
- Total if child fails minibatch: 6 metric calls

If 50% of children pass the minibatch test but only 30% of those improve on val, then 35% of iterations burn 20 val evaluations for nothing. Over a budget of 500 metric calls, that's ~70 wasted val evaluations — enough for 3-4 additional full iterations.

## The Fix: Evaluate Val Progressively, Early-Stop Losers

Instead of evaluating on the full val set at once, evaluate in chunks. Stop early if the candidate is clearly worse than the current best.

```
1. Candidate passes minibatch test
2. Shuffle val set, evaluate on first chunk (5 examples)
3. Compute running average score
4. If running_score < current_best - margin: REJECT EARLY (save remaining budget)
5. Evaluate next chunk (5 examples), update running score
6. If still below margin: REJECT EARLY
7. Continue until full val set evaluated or early-stopped
```

This is **successive halving** (Jamieson & Talbot, 2016) applied to candidate evaluation — the core idea behind Hyperband in hyperparameter optimization.

## The Budget Savings

Assume 60% of candidates that pass the minibatch test are clearly worse than the current best. With progressive validation:
- These 60% get rejected after ~5 val examples instead of 20
- Budget saved: 60% × 15 examples = 9 examples per iteration on average
- Over 20 iterations: ~180 saved metric calls
- That's enough for ~7 additional full iterations

More iterations = more chances to find improvements = better final result.

## Where It Intervenes

`GEPAEngine._run_full_eval_and_add()` in `src/gepa/core/engine.py` (line 146).

Currently this calls `state.cached_evaluate_full()` on the entire val set in one shot. The change: evaluate in chunks, check progress after each chunk.

```python
def _run_progressive_eval(
    self,
    candidate: dict[str, str],
    val_loader: DataLoader,
    current_best_score: float,
    chunk_size: int = 5,
    margin: float = 0.15,
) -> tuple[EvaluationBatch | None, bool]:
    """Evaluate candidate on val set progressively. Returns None if early-stopped."""
    val_ids = list(val_loader.all_ids())
    self.rng.shuffle(val_ids)

    all_scores = []
    for i in range(0, len(val_ids), chunk_size):
        chunk_ids = val_ids[i:i + chunk_size]
        chunk_batch = val_loader.get_batch(chunk_ids)
        chunk_result = self.adapter.evaluate(chunk_batch, candidate)
        all_scores.extend(chunk_result.scores)

        # Early stopping check
        running_avg = sum(all_scores) / len(all_scores)
        remaining = len(val_ids) - len(all_scores)
        # Even if all remaining score 1.0, can we beat current best?
        optimistic_bound = (sum(all_scores) + remaining) / len(val_ids)
        if optimistic_bound < current_best_score - margin:
            return None, False  # Impossible to beat best, stop early

    # Full evaluation complete
    return combine_chunks(all_scores), True
```

## The Margin

The margin controls the aggressiveness of early stopping.

- **Tight margin (0.05)**: Only reject candidates that are drastically worse. Saves budget only on clearly terrible candidates. Safe but modest savings.
- **Moderate margin (0.15)**: Reject candidates that are meaningfully worse. Good balance of savings vs. risk. Recommended default.
- **Loose margin (0.3)**: Aggressively reject anything below current best. Saves lots of budget but risks rejecting candidates that bring Pareto diversity.

An alternative to a fixed margin: use the **optimistic bound**. After evaluating k examples, the best possible final average (assuming all remaining examples score 1.0) is `(sum_so_far + (n - k)) / n`. If this optimistic bound is below the current best aggregate score, the candidate *cannot* beat the best even in the most favorable case. Reject with certainty.

The optimistic bound requires no margin hyperparameter and never produces false negatives for the aggregate score. It's conservative but free.

## Interaction with Pareto Front

A candidate that's mediocre on average but excellent on specific examples would be early-stopped (low running average) even though it might contribute to the Pareto front on those specific examples.

Mitigation: during progressive evaluation, track per-example scores. If the candidate achieves a new best on any evaluated example (beats all existing candidates on that example), don't early-stop — it has potential Pareto value even if the average is low.

```python
# In the early stopping check:
if any(score > state.pareto_front_valset.get(vid, 0)
       for vid, score in zip(chunk_ids, chunk_result.scores)):
    continue  # Potential Pareto contribution, don't stop
```

This preserves Pareto diversity while still early-stopping candidates that are worse everywhere.

## Interaction with Evaluation Cache

`EvaluationCache` in `GEPAState` caches `(candidate_hash, example_id) → score`. Progressive evaluation naturally interacts with this cache:
- If some val examples are already cached (from a previous partial evaluation of this candidate), those scores are free — count them in the running average.
- Even if a candidate is early-stopped, the scores computed so far are cached. If a similar candidate is proposed later, those cached scores accelerate its evaluation.

## Statistical Subtlety

Early evaluation on 5 examples is noisy. The running average has high variance. You'll sometimes reject good candidates (false negatives) and occasionally promote bad ones to full evaluation (false positives).

**False negatives are acceptable.** GEPA is iterative. A good mutation strategy will be rediscovered in a future iteration. The budget saved by rejecting bad candidates funds more iterations, which more than compensates for occasional false rejections.

**False positives waste budget but aren't harmful.** A bad candidate that survives progressive evaluation and completes full val evaluation just gets a low score and isn't added to the Pareto front. The cost is the extra val evaluations — same as current GEPA without progressive validation.

## Design Decisions

**Chunk size.** Smaller chunks = more frequent stopping decisions = more savings on bad candidates but more overhead (per-chunk evaluation setup). Chunk size 5 for val sets of 15-30 is a reasonable default. For larger val sets, use proportional chunks (e.g., 25% of val set per chunk).

**Evaluation order.** Shuffling the val set before progressive evaluation prevents bias from example ordering. Re-shuffle each time to avoid systematic effects.

**Minimum evaluation.** Always evaluate at least one full chunk before early stopping. One chunk (5 examples) gives a reasonable estimate; fewer is too noisy.

## Expected Outcome

- **~30-50% more iterations** for the same total budget, depending on the fraction of bad candidates reaching val evaluation.
- **No degradation in final quality**: early-stopped candidates were bad anyway. The budget saved funds better iterations.
- **Faster wall-clock time**: fewer total evaluations = faster completion.

## Ablation Design

1. **Standard GEPA** — full val evaluation for every candidate that passes minibatch test (control)
2. **Progressive validation, optimistic bound only** (no margin, never rejects a possibly-better candidate)
3. **Progressive validation, margin=0.10**
4. **Progressive validation, margin=0.20**
5. **Progressive validation with Pareto protection** (don't stop if candidate has per-example potential)
6. **Aggressive minibatch filter** (require `new_sum > old_sum + delta` on minibatch, tighter pre-filter)

Compare (1) vs (2) for the conservative headline. Compare (2) vs (3) vs (4) for the margin sensitivity. Compare (1) vs (5) for the recommended configuration. Track: total metric calls used, number of iterations completed, final val score, Pareto front diversity.
