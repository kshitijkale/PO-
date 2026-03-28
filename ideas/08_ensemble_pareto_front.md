# Ensembling the Pareto Front

## The AutoML Insight

Auto-sklearn (Feurer et al., 2015) demonstrated a simple but powerful idea: instead of returning the single best configuration found during search, return an **ensemble** of the top configurations weighted by their validation performance. The ensemble consistently outperforms the single best — often by a significant margin — because different configurations have complementary strengths.

This works because of a fundamental asymmetry: finding a single configuration that's best on *every* input is much harder than finding a set of configurations where *at least one* handles each input well, and routing to it.

GEPA already maintains exactly such a set: the Pareto front. Candidates on the front are there precisely because they're best at something. But GEPA discards this diversity at the end — `GEPAResult.best_candidate` returns a single candidate. All the specialized knowledge on the Pareto front is wasted.

## Proposed Intervention

At the end of optimization (or during, for anytime usage), construct an ensemble from Pareto front candidates and a routing mechanism:

### For Prompt Optimization (Generalization Mode)

Each candidate on the Pareto front is a prompt. Given a new test example:

1. **Route**: Determine which Pareto front candidate is most likely to succeed on this example
2. **Execute**: Run the selected candidate on the example
3. **(Optional) Verify**: If budget allows, run 2-3 candidates and take the best result

The routing mechanism can be:

**Option A: Score-Based Routing.** Use the per-example val scores stored in `prog_candidate_val_subscores`. For a new test example, find the most similar val examples (via embedding similarity), look up which candidate scored highest on those similar examples, and route to that candidate.

```
route(test_example):
  similar_val_ids = find_nearest(embed(test_example), val_embeddings, k=3)
  for each candidate on Pareto front:
    estimated_score = average(candidate's scores on similar_val_ids)
  return argmax(estimated_score)
```

**Option B: LLM-Based Routing.** Ask the LLM which candidate's strategy is best suited for the given example. This is more expensive but potentially more accurate.

**Option C: Feature-Based Routing.** Train a lightweight classifier on (example_features, best_candidate_id) pairs using the val set. At test time, predict which candidate to use. Features can be embeddings or LLM-extracted task characteristics.

**Option D: Majority Vote / Best-of-K.** Run the top K Pareto front candidates on the example, take the majority vote (for classification-like tasks) or the highest-scoring result (for optimization tasks). Simple, reliable, but K times more expensive per example.

### For Code / Single-Task Optimization

Ensembling is harder when the output is code (you can't majority-vote code). Options:

- **Best-of-K with re-evaluation**: Run top K candidates, evaluate each, return the best.
- **Component-level ensembling**: If the candidate has multiple components (e.g., `{system_prompt, code_template, verification_step}`), select the best version of each component independently. This is a recombination, not an ensemble — closer to the merge operator but informed by per-component performance data.

## Where This Intervenes in the Codebase

### Modified: GEPAResult
`src/gepa/core/result.py` — extend with ensemble capabilities:

```python
class GEPAResult:
    @property
    def pareto_ensemble(self) -> list[dict[str, str]]:
        """Return all candidates on the Pareto front."""
        pareto_indices = set()
        for prog_set in self._state.program_at_pareto_front_valset.values():
            pareto_indices.update(prog_set)
        return [self._state.program_candidates[i] for i in sorted(pareto_indices)]

    def route(self, example, embed_fn=None) -> dict[str, str]:
        """Select the best Pareto front candidate for a given example."""
        # Score-based routing using val set similarity
        ...
```

### New component: EnsembleRouter
A new class in `src/gepa/strategies/ensemble_router.py` that:
- Takes the Pareto front candidates and their per-example val scores
- Optionally takes an embedding function for similarity-based routing
- Provides `route(example) -> candidate` and `ensemble_predict(example, k=3) -> best_result`

### Integration with optimize_anything
The `optimize_anything` API returns a `GEPAResult`. We'd extend it to expose ensemble functionality:

```python
result = optimize_anything(...)
# Current: single best
best = result.best_candidate

# New: ensemble routing
best_for_example = result.route(example, embed_fn=embed)
# or: best-of-3
best_result = result.ensemble_predict(example, evaluator, k=3)
```

## Why This Is Almost Free

The Pareto front is already maintained. The per-example val scores are already stored in `prog_candidate_val_subscores`. Embedding-based routing requires one embed call per test example (cheap). The only meaningful cost is the implementation effort — there's no additional computational cost during optimization.

This is pure post-processing: it extracts more value from the same optimization run.

## When Ensembling Helps vs. Hurts

**Helps when:**
- Pareto front candidates have *complementary* strengths — each handles different example types well
- The test distribution has the same heterogeneity as the val distribution
- The routing mechanism correctly matches examples to candidates

**Hurts when:**
- Pareto front candidates are *redundant* — all do roughly the same thing, one is just slightly better
- Routing errors are common — wrong candidate is selected, performing worse than the single best
- The task requires a single coherent strategy (e.g., a system prompt that must handle all inputs — you can't switch prompts per input in some deployment scenarios)

The key diagnostic: look at the Pareto front diversity. If many candidates are on the front and they dominate on different val examples, ensembling will help. If one candidate dominates on most examples and the rest are marginal, the single best is sufficient.

## Connection to Mixture of Experts

This is structurally a **Mixture of Experts** (MoE) system where:
- Experts = Pareto front candidates
- Gating network = routing mechanism
- The experts were discovered by evolution, not trained end-to-end

Unlike standard MoE, the experts are *complete systems* (full prompts or code), not partial models. The gating operates at the input level (which expert handles which example), not at the feature level.

## Design Decisions

**How many candidates in the ensemble?** All Pareto front candidates, or top-K by aggregate score? Including all preserves maximum diversity but may include noisy candidates. Top-K (e.g., K=5-10) is more robust. Use the full front but weight by aggregate val score when routing.

**Online vs. offline ensembling.** Offline: ensemble is constructed after optimization completes. Online: use ensemble routing during optimization to improve evaluation quality (evaluate new candidates using the ensemble of existing ones as a comparison baseline). Start with offline — it's simpler and already provides value.

**Routing accuracy matters more than ensemble size.** A perfect router with 3 candidates beats a random router with 30. Invest in good similarity matching (high-quality embeddings, sufficient K for KNN) rather than maintaining more candidates.

## Expected Outcome

- **Improved test performance**: By routing to the best candidate per example, the ensemble should outperform any single candidate.
- **Better utilization of Pareto diversity**: Currently, Pareto diversity is only used for parent selection during optimization. Ensembling uses it at inference time.
- **Zero additional optimization cost**: Pure post-processing on existing optimization state.

## Ablation Design

1. **Single best candidate** (current behavior, control)
2. **Best-of-K (K=3)**: run top 3 candidates, pick highest score
3. **Embedding-based routing**: similarity to val examples
4. **Oracle routing**: route to the candidate that actually scores best (upper bound on routing quality)
5. **Uniform random routing**: select a random Pareto front candidate (lower bound)

The gap between (4) and (1) tells you the maximum possible gain from ensembling. The gap between (3) and (5) tells you how much routing quality matters.
