# History-Based Active Sampling

## Core Insight

Every iteration, GEPA evaluates the parent candidate on a minibatch of training examples. These scores are recorded in `state.full_program_trace`. Over the course of a run, this accumulates into a rich history of how the prompt's lineage has interacted with each training example — what it scored, whether it improved, how many times it failed, how long ago it was last seen.

This history is thrown away at sampling time. The `EpochShuffledBatchSampler` ignores it completely — it shuffles and iterates, blind to what happened before. Every question looks the same to the sampler regardless of whether the prompt aces it or fails on it repeatedly.

The proposal: use the (prompt, question) interaction history directly to guide sampling. No embeddings, no kernels, no spatial interpolation. Just the raw bookkeeping of what happened when the prompt met each question.

## The Dimensions of a Question Relative to a Prompt

Each question Q, relative to the current prompt P and its ancestry, has a measurable relationship state defined by four observable quantities:

1. **Improvement count** ($n_{\text{imp}}$): How many times has Q appeared in a minibatch where the prompt's score on Q improved between parent and child?
2. **Failure count** ($n_{\text{fail}}$): How many times has Q appeared in a minibatch where the prompt scored below threshold on Q?
3. **Recency** ($\Delta t$): How many iterations since Q was last evaluated? ($\infty$ if never)
4. **Interaction indicator**: Has Q been evaluated at all?

These four observables are not independent. They encode two latent quantities that together determine the value of including Q in the next minibatch.

## Latent Quantity 1: Expected Current Score ($\hat{s}_Q$)

"Would the prompt fail on Q if we evaluated it right now?"

We don't know this without evaluating. But we can estimate it. If Q was evaluated 2 iterations ago and scored 0.3, the prompt probably still fails on Q — prompts don't change drastically in 2 iterations. If Q was evaluated 40 iterations ago and scored 0.3, many mutations have occurred since — the prompt might handle Q now, or might still fail. If Q was never evaluated, we have no information.

The key modeling choice: **scores become stale as the prompt evolves.** The prompt at iteration 50 is not the prompt at iteration 10. An old score is evidence but not certainty. Model this as exponential decay toward a prior:

$$\hat{s}_Q = s_Q^{\text{last}} \cdot \lambda^{\Delta t} + \mu \cdot (1 - \lambda^{\Delta t})$$

where:
- $s_Q^{\text{last}}$ is the most recently observed score on Q
- $\Delta t$ is iterations since that observation
- $\lambda \in (0, 1)$ is a decay rate (e.g., 0.95 → half-life of ~14 iterations)
- $\mu$ is the prior — the expected score when we have no information

For questions never evaluated: $\hat{s}_Q = \mu$ (pure prior).

**What should $\mu$ be?** The running average score across all recently evaluated questions. This adapts as the prompt improves: early in optimization $\mu$ might be 0.3; late it might be 0.8. It represents "how well does this prompt handle a random question?"

**Why exponential decay?** Each iteration, the prompt undergoes one mutation. With probability $p$ (the acceptance rate), the mutation changes the prompt's behavior. Over $\Delta t$ iterations, the probability that the prompt is unchanged is approximately $(1-p)^{\Delta t}$, which decays exponentially. The decay rate $\lambda$ should approximate $(1-p)$, the probability of no change per iteration. In practice, acceptance rates are 10-30%, so $\lambda \in [0.7, 0.9]$ is reasonable. Using $\lambda = 0.95$ is conservative (trusts old scores longer).

This single quantity subsumes observables 3 and 4. Recency controls confidence; never-seen defaults to prior.

## Latent Quantity 2: Learnability ($\ell_Q$)

"If we include Q in the next minibatch, will the reflection LM produce a mutation that actually helps?"

Some questions are productive training signal. The reflection LM sees the failure, diagnoses the root cause, and proposes a fix that works. Other questions are persistently hard — the LM sees the failure repeatedly, proposes fixes, but they don't stick (rejected by val gate, or accepted but regressed on Q later).

Observable 1 (improvement count) and observable 2 (failure count) directly inform this.

Model learnability as a Bernoulli process with a Beta prior. Each time Q appears in a minibatch, it's a trial. "Success" means Q's score improved between parent and child. "Failure" means it didn't.

With a Beta(1, 1) uniform prior, the posterior after $n_{\text{imp}}$ successes and $n_{\text{fail}}$ failures is Beta($1 + n_{\text{imp}}$, $1 + n_{\text{fail}}$), with posterior mean:

$$\ell_Q = \frac{1 + n_{\text{imp}}}{2 + n_{\text{imp}} + n_{\text{fail}}}$$

This is Laplace-smoothed estimation. It has the right behavior:

| History | $\ell_Q$ | Interpretation |
|---|---|---|
| Never seen | 0.5 | Agnostic — could go either way |
| 3 improvements, 0 failures | 0.8 | Productive — the LM learns from Q |
| 0 improvements, 5 failures | 0.14 | Resistant — the LM can't crack Q |
| 2 improvements, 2 failures | 0.5 | Mixed — sometimes helps, sometimes not |
| 1 improvement, 10 failures | 0.15 | Nearly exhausted — tried hard, rarely works |

This is not an ad hoc formula. It is the exact Bayesian posterior mean under the Beta-Bernoulli model. The prior (Beta(1,1) = Uniform) expresses initial ignorance. The posterior concentrates as evidence accumulates.

**Important subtlety:** what counts as "improvement" vs "failure" for Q specifically? Q is part of a minibatch of 3 examples. The mutation was designed for all 3. Whether Q improved depends on the prompt, the other examples in the batch, and the reflection LM's focus. A failure on Q might mean Q is hard, or it might mean the LM focused on the other 2 examples. The signal is noisy per-trial, but the Beta posterior averages over many trials, and the noise washes out.

## The Sampling Weight

The value of including Q in the next minibatch:

$$w_Q = \underbrace{(1 - \hat{s}_Q)}_{\text{expected failure}} \times \underbrace{\ell_Q}_{\text{expected learnability}}$$

Two terms. One multiplication. That's the entire sampling policy.

Sample the minibatch of size $k$ without replacement with probabilities $P(Q) \propto w_Q$.

### Why This Decomposition Is Complete

There are only two reasons NOT to include Q in a minibatch:
1. The prompt already handles Q ($\hat{s}_Q$ is high → first term is small)
2. Including Q won't lead to improvement ($\ell_Q$ is low → second term is small)

If both terms are large, Q is a valuable training example. If either is small, Q is a waste of a minibatch slot. There is no third reason.

### Why No Explicit Exploration Term

The embedding-based version (Idea 7) needed a separate exploration bonus ($\gamma \cdot w^{\text{explore}}$) because spatial interpolation could confidently predict high scores for unseen questions near solved ones, suppressing exploration.

Here, there's no spatial interpolation. An unseen question gets:
- $\hat{s}_Q = \mu$ → failure probability $= 1 - \mu$ (moderate, not suppressed)
- $\ell_Q = 0.5$ (agnostic, not suppressed)
- $w_Q = (1 - \mu) \times 0.5$ (moderate weight)

Unseen questions automatically get moderate weight. They compete fairly with known failures. No exploration bonus needed.

And stale questions **automatically become exploration candidates** via the decay. A question that scored 0.9 twenty iterations ago now has $\hat{s}_Q$ decayed toward $\mu$, increasing its failure estimate. The sampler re-checks it. If the prompt still handles it, the score refreshes to 0.9 and the weight drops again. If the prompt regressed, the sampler caught it.

**The decay IS the exploration mechanism.** Staleness creates uncertainty. Uncertainty creates sampling weight. This is elegant because it handles catastrophic forgetting detection for free — no explicit mechanism needed, just the natural consequence of not trusting old scores.

## The Question State Machine

Each question transitions through states based on its interaction history:

```
                    ┌────────────────────────────────────────────┐
                    │          time passes (decay)               │
                    v                                            │
              ┌───────────┐                               ┌─────┴──────┐
              │  UNKNOWN   │────── evaluated ──────────────│  RECENTLY   │
              │            │                               │  EVALUATED  │
              │ ŝ = μ      │                               │             │
              │ ℓ = 0.5    │                               │ ŝ = fresh   │
              │ w = moderate│         ┌────────────────────│ ℓ = updated │
              └───────────┘          │                    └──────┬──────┘
                    ^                │                           │
                    │                │                      score high?
               extreme              │                     ┌─────┴─────┐
               staleness            │                    YES          NO
                    │                │                     │           │
              ┌─────┴──────┐        │              ┌──────v──┐  ┌────v───────┐
              │   STALE     │───────┘              │ SOLVED   │  │  FAILING   │
              │             │  re-evaluated         │          │  │            │
              │ ŝ → μ       │                      │ w = LOW  │  │ w = HIGH   │
              │ w = rising  │                      │          │  │ (if ℓ high)│
              └────────────┘                      └──────────┘  │ w = LOW    │
                                                                │ (if ℓ low) │
                                                                └────────────┘
                                                                      │
                                                                 many failures,
                                                                 ℓ drops
                                                                      │
                                                                ┌─────v──────┐
                                                                │ DIMINISHING │
                                                                │             │
                                                                │ w = LOW     │
                                                                │ (fail × !learn)
                                                                └─────────────┘
```

The weight function $w_Q = (1 - \hat{s}_Q) \times \ell_Q$ navigates this state machine automatically. No explicit state tracking — the two latent quantities encode the state implicitly.

## Where This Intervenes in the Codebase

### Existing Infrastructure

`state.full_program_trace` (in `src/gepa/core/state.py`) is a list of dicts, one per iteration. Each entry contains:
- `subsample_ids`: which train examples were in the minibatch
- `subsample_scores`: their scores (parent evaluation)
- `proposed_scores`: their scores (child evaluation, if available)
- `was_accepted`: whether the candidate was accepted

This is everything needed to compute $\hat{s}_Q$, $n_{\text{imp}}$, and $n_{\text{fail}}$ for every question.

### New: `HistoryBasedBatchSampler`

A drop-in replacement for `EpochShuffledBatchSampler` via the `BatchSampler` protocol in `src/gepa/strategies/batch_sampler.py`:

```python
class HistoryBasedBatchSampler(BatchSampler[DataId, DataInst]):
    """Samples minibatches weighted by expected failure × learnability,
    computed from the prompt's interaction history with each question."""

    def __init__(
        self,
        minibatch_size: int = 3,
        decay: float = 0.95,
        prior: float | None = None,  # None = adaptive (running average)
        rng: random.Random | None = None,
    ):
        self.minibatch_size = minibatch_size
        self.decay = decay
        self.fixed_prior = prior
        self.rng = rng or random.Random(0)

    def _build_performance_map(
        self, state: GEPAState
    ) -> dict[DataId, tuple[float, int, int, int]]:
        """Returns {data_id: (last_score, iters_ago, n_improve, n_fail)}."""
        perf: dict[DataId, tuple[float, int, int, int]] = {}
        current_iter = state.i

        for iter_idx, trace in enumerate(state.full_program_trace):
            ids = trace["subsample_ids"]
            scores = trace["subsample_scores"]
            proposed = trace.get("proposed_scores", {})

            for did, score in zip(ids, scores):
                prev = perf.get(did)
                n_imp = prev[2] if prev else 0
                n_fail = prev[3] if prev else 0

                # Did the child improve on this question?
                if did in proposed:
                    if proposed[did] > score:
                        n_imp += 1
                    else:
                        n_fail += 1

                iters_ago = current_iter - iter_idx
                perf[did] = (score, iters_ago, n_imp, n_fail)

        return perf

    def next_minibatch_ids(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState
    ) -> list[DataId]:
        all_ids = list(loader.all_ids())
        perf = self._build_performance_map(state)

        # Adaptive prior: average recent score
        if self.fixed_prior is not None:
            mu = self.fixed_prior
        else:
            recent_scores = [s for s, dt, _, _ in perf.values() if dt < 20]
            mu = sum(recent_scores) / len(recent_scores) if recent_scores else 0.5

        weights = []
        for did in all_ids:
            if did in perf:
                last_score, dt, n_imp, n_fail = perf[did]
                s_hat = last_score * (self.decay ** dt) + mu * (1 - self.decay ** dt)
                ell = (1 + n_imp) / (2 + n_imp + n_fail)
            else:
                s_hat = mu
                ell = 0.5

            w = (1 - s_hat) * ell
            weights.append(max(w, 1e-6))  # floor to avoid zero-probability

        # Sample without replacement, proportional to weights
        batch = []
        available = list(range(len(all_ids)))
        for _ in range(min(self.minibatch_size, len(all_ids))):
            w_available = [weights[i] for i in available]
            total = sum(w_available)
            probs = [w / total for w in w_available]
            chosen_idx = self.rng.choices(available, weights=probs, k=1)[0]
            batch.append(all_ids[chosen_idx])
            available.remove(chosen_idx)

        return batch
```

Three hyperparameters: `minibatch_size` (already exists), `decay` (one new parameter), and `prior` (optional, defaults to adaptive). No external dependencies. Reads directly from `state.full_program_trace`.

## Design Decisions

**Decay rate $\lambda$.** This is the only meaningfully new hyperparameter. It controls how quickly we lose trust in old scores.

$\lambda = 0.95$ gives a half-life of ~14 iterations. After 14 iterations, a score is halfway to the prior. After 30 iterations, it's ~80% prior. This is conservative — it trusts scores for a while before requiring re-evaluation.

$\lambda = 0.85$ gives a half-life of ~5 iterations. More aggressive — scores become stale quickly, forcing more re-evaluation. Better for tasks where the prompt changes rapidly (high acceptance rate).

The theoretically motivated choice: $\lambda \approx 1 - p_{\text{accept}}$ where $p_{\text{accept}}$ is the acceptance rate. If 20% of mutations are accepted, $\lambda = 0.8$. This reflects the rate at which the prompt actually changes.

**Adaptive vs. fixed prior.** The adaptive prior ($\mu$ = running average of recent scores) is preferred because it tracks the prompt's overall quality level. Early in optimization when the prompt is weak, $\mu$ is low, so unseen questions get high failure estimates → aggressive exploration. Late when the prompt is strong, $\mu$ is high, so unseen questions are assumed to be handled → exploration slows. This is the right behavior.

A fixed prior ($\mu$ = 0.5) works but doesn't adapt. It over-explores late in optimization (when most questions are solved) and under-explores early (when the prompt is terrible at everything).

**What counts as "improvement" for learnability?** The cleanest definition: Q's score improved between the parent candidate's evaluation and the child candidate's evaluation on the same minibatch. This is directly available from `subsample_scores` (parent) and `proposed_scores` (child) in the trace.

A stricter definition: only count as improvement if the child was accepted (passed the val gate). This conflates "Q improved" with "the whole candidate was good enough for val," which mixes Q-specific signal with global quality. The looser definition (Q improved regardless of acceptance) is more informative about Q specifically.

**Should learnability decay with time?** The improvement/failure counts are cumulative over the entire lineage. If Q was productive early but resistant late (or vice versa), the counts blend both periods. You could apply recency weighting to the counts (recent trials count more), but this adds complexity. Start without decay — the Beta posterior is already a reasonable summary of overall experience with Q. If empirical results show that learnability changes significantly over the run, add decay as a refinement.

**Minibatch diversity.** Weighted sampling might draw multiple questions from the same failure cluster (if several related questions all have high weight). The reflection LM would see redundant failures. Mitigation: after selecting each batch element, temporarily boost the weights of questions that are dissimilar to already-selected ones. But "dissimilar" requires a distance metric, which requires embeddings — defeating the purpose of this embedding-free approach.

A simpler mitigation: after selecting each batch element, temporarily reduce the weight of questions that have the exact same (last_score, n_imp, n_fail) profile. This is a rough proxy for "same type of question" without requiring content-level similarity. Or just accept the potential redundancy — with minibatch size 3, the probability of drawing 3 near-identical questions from a 50-example dataset is low.

## Comparison to Embedding-Based Sampling (Idea 7)

| | History-Based (This) | Embedding-Based (Idea 7) |
|---|---|---|
| **Signal for evaluated questions** | Direct observed scores | Direct observed scores |
| **Signal for unevaluated questions** | Prior only ($\mu$) | Spatial interpolation from neighbors |
| **External dependencies** | None | Embedding model |
| **Hyperparameters** | 1 (decay) | 4 (decay, kernel bandwidth, exploration weight, temperature) |
| **Assumptions** | Scores decay exponentially; improvements are Bernoulli | Same + embedding proximity ≈ difficulty similarity |
| **Best when** | Dataset is small-medium; full coverage within ~1 epoch | Dataset is large; many questions never evaluated |

For GEPA's typical use (50-200 examples, minibatch 3, 50+ iterations), the history-based approach builds direct observations for most questions within one epoch (~17 iterations for 50 examples). After that, it has actual scores for everything and the embedding interpolation adds no value.

The embedding approach is strictly more powerful (it has everything the history approach has, plus spatial predictions). The history approach is simpler, cheaper, has fewer assumptions, and captures most of the value.

**Recommendation:** Start with the history-based approach. It's the right default. If you're working with large datasets (500+ examples) where one-epoch coverage takes too long, add embedding interpolation as an enhancement.

## Expected Outcomes

1. **Fewer wasted iterations**: The reflection LM consistently sees failures it can learn from, not successes that provide no signal.
2. **Faster convergence**: Known failures get targeted first; the prompt builds capabilities in the order of what's most fixable.
3. **Automatic re-checking**: Staleness decay forces periodic re-evaluation of old successes, catching regressions without explicit catastrophic-forgetting detection.
4. **Natural schedule**: Early iterations explore broadly (everything is unknown); mid iterations target failures (some are identified); late iterations polish (few failures remain). No manual annealing.
5. **Preserved val independence**: The val set is never consulted. It remains a pure generalization gate.

## Ablation Design

1. **EpochShuffledBatchSampler** (current default, control)
2. **HistoryBasedBatchSampler, $\lambda = 0.95$, adaptive prior**
3. **HistoryBasedBatchSampler, $\lambda = 0.85$, adaptive prior** (faster decay)
4. **HistoryBasedBatchSampler, failure-only** ($w_Q = 1 - \hat{s}_Q$, no learnability term — tests whether learnability adds value)
5. **HistoryBasedBatchSampler, learnability-only** ($w_Q = \ell_Q$, no score estimate — tests whether failure targeting adds value)
6. **HistoryBasedBatchSampler, fixed prior $\mu = 0.5$** (tests adaptive prior)
7. **Embedding-based sampler (Idea 7)** (tests whether spatial interpolation adds value beyond history)

Compare (1) vs (2) for the headline result. Compare (4) vs (5) vs (2) to decompose the contribution of each term. Compare (2) vs (7) to measure the marginal value of embeddings over pure history.

Key metrics: val score over iterations (convergence speed), val gate acceptance rate (proposal quality), and final held-out test score (generalization).

---

## Afterthought: Thompson Sampling

The weight function $w_Q = (1 - \hat{s}_Q) \times \ell_Q$ uses point estimates. A more theoretically principled approach: use the full posterior distributions and select via Thompson Sampling.

For each question Q, maintain two Beta posteriors:

- **Score belief**: Beta($\alpha_s$, $\beta_s$) where $\alpha_s$ accumulates decayed "above threshold" evaluations and $\beta_s$ accumulates decayed "below threshold" evaluations.
- **Learnability belief**: Beta($1 + n_{\text{imp}}$, $1 + n_{\text{fail}}$)

To build a minibatch of size $k$:

```python
batch = []
for _ in range(k):
    best_q, best_v = None, -1
    for Q not in batch:
        s_sample = rng.betavariate(alpha_s[Q], beta_s[Q])
        l_sample = rng.betavariate(1 + n_imp[Q], 1 + n_fail[Q])
        v = (1 - s_sample) * l_sample
        if v > best_v:
            best_q, best_v = Q, v
    batch.append(best_q)
return batch
```

Each call samples from the posterior, so questions with high uncertainty occasionally draw extreme values and get selected — natural exploration without any exploration bonus. Thompson Sampling achieves $O(\sqrt{KT \log T})$ Bayesian regret for $K$ arms over $T$ rounds (Agrawal & Goyal, 2012), which is near-optimal.

In practice, the gap between Thompson Sampling and the point-estimate approach is small for problems at GEPA's scale (50-200 questions). The point estimate version is simpler, easier to debug, and produces nearly identical behavior. Thompson Sampling is worth noting for theoretical completeness — it's the "right" answer from a decision-theoretic perspective — but the simple version is what you should implement first.
