# Lesson-Based Minibatch Sampling

## The Problem with Current Sampling Strategies

Every proposed sampling strategy for GEPA — the default `EpochShuffledBatchSampler`, the history-based active sampler (Idea 8), the embedding-informed sampler (Idea 7) — treats examples as independent items with individual priorities. They ask: "which examples are most valuable to include?" and assign each example a weight. The minibatch is a bag of independently-prioritized items.

This misses the point. The minibatch isn't a bag of items. It's an **input to a reasoning process**. The reflection LLM reads the entire minibatch as a unit and reasons about it holistically. The value of a minibatch isn't the sum of the values of its elements — it's determined by how the elements interact to support diagnosis and fix proposal.

A minibatch of three unrelated failures gives the reflection LLM three independent problems. It can diagnose each one, but it can only propose one prompt edit. It either picks one to fix (ignoring the others) or makes a muddled edit that half-addresses all three. Either way, the resulting fix is shallow.

A minibatch of two related failures plus one contrasting success gives the reflection LLM a **coherent story** — a specific weakness illustrated by multiple examples, with a success that anchors what's already working. The LLM can make a deep, targeted diagnosis and propose a fix that addresses the root cause. That fix generalizes because it came from understanding, not from patching.

The unit of optimization isn't the example. It's the lesson.

## What Makes the Reflection LLM Learn

The reflection LLM is doing inductive reasoning: it sees specific cases and infers a general principle (a prompt modification). The quality of its inference depends on the quality of the evidence.

Three properties of a minibatch determine how well the reflection LLM can learn from it:

### 1. Failure Signal

The LLM needs failures to work with. A minibatch of all successes provides zero learning signal — there's nothing to diagnose, nothing to fix. The iteration is wasted.

This is the one thing all sampling proposals agree on, and it's the easiest to ensure.

### 2. Coherence

The failures in the batch should be **related** — they should stem from the same underlying weakness in the prompt. When two failures share a root cause, the LLM can identify the pattern with confidence: "both of these fail because the prompt doesn't instruct the solver to check units." One failure is anecdotal. Two related failures are a pattern.

When the failures are unrelated — one is about unit conversion, one is about geometric reasoning, one is about edge cases — the LLM sees three separate problems. It can't go deep on any of them. It spreads its diagnostic capacity across three independent issues and produces a shallow, scattered fix.

This is the property that no current sampling proposal addresses. Prioritizing failures (Idea 8) increases the number of failures in the batch but doesn't ensure they're related. Embedding-based sampling (Idea 7) prioritizes individual examples by expected failure but doesn't construct coherent groups.

### 3. Contrast

A success that's **related to the failures** is the most powerful diagnostic signal. If the prompt solves problem A but fails problem B, and A and B test similar skills, then the difference between A and B isolates exactly what's breaking. The LLM can reason: "these problems are similar, but B has larger numbers — the prompt's reasoning breaks down at scale."

Idea 9 (contrastive minimal pairs) captures this insight for the reflection prompt. But the same principle applies to sampling: the ideal minibatch includes a success that serves as a contrast anchor.

The contrast also serves a practical purpose: it keeps the **subsample filter meaningful**. In GEPA's flow (engine.py:540-542), after the reflection LLM proposes a child prompt, the child is evaluated on the same minibatch and must score strictly higher than the parent (`new_sum > old_sum`). If the minibatch is all failures, the child was designed to fix those failures — it will almost always beat the parent on them. The filter becomes vacuous: every child passes to the expensive valset evaluation. With a success in the batch, the child must also not regress on it, which screens out mutations that fix failures at the cost of breaking other things.

## Why History-Based Priority Alone Isn't Enough

Idea 8 proposes `w_Q = (1 - ŝ_Q) × ℓ_Q` — weight each example by its estimated failure probability times its learnability. This is a good individual priority, but it constructs the minibatch by sampling independently according to these weights. The resulting minibatch is a random draw from the failure pool, with no structure.

Three specific problems:

### The Learnability Signal Is Too Noisy

At GEPA's operating scale (50 training examples, ~50 iterations, minibatch size 3), each example is observed roughly 3 times over the entire run. The learnability estimate `ℓ_Q = (1 + n_imp) / (2 + n_imp + n_fail)` with 3 data points is barely distinguishable from the Beta(1,1) prior of 0.5. Furthermore, each observation is confounded: Q is part of a minibatch of 3, and whether Q improved depends on what the reflection LLM focused on, which depends on the other 2 examples. The per-example learnability signal is real but submerged in noise at this scale.

### The Subsample Filter Becomes Vacuous

With aggressive failure weighting, the minibatch is dominated by failures. The child prompt, designed to fix those failures, almost always beats the parent on the biased minibatch. The subsample filter (engine.py:542, `new_sum > old_sum`) stops screening. Every child proceeds to valset evaluation.

Budget impact: each iteration costs `2k` evaluations (parent + child on minibatch) plus `|valset|` if the child passes the filter. With minibatch size 3 and valset size 20, unfiltered iterations cost 26 evaluations vs. ~16 when the filter rejects ~50% of children. Over a budget of 500 metric calls, that's ~19 vs. ~31 iterations. Fewer iterations means fewer chances to find improvements, even if each iteration is individually better targeted.

### No Inter-Example Structure

Sampling examples independently, even with good per-example priorities, produces minibatches with no guaranteed relationship between the examples. Two failures drawn from a pool of 20 are likely to be unrelated, especially if the failures span multiple distinct skills (as they typically do in math benchmarks). The reflection LLM receives an incoherent set and cannot reason deeply about any single weakness.

## The Two-Role Minibatch

The minibatch serves two distinct roles simultaneously, and these roles have different requirements:

**Role 1: Reflection input.** The reflection LLM reads these examples to diagnose the prompt's weaknesses. For this role, the minibatch should contain related failures — a coherent story about one specific weakness. Including only failures maximizes signal density.

**Role 2: Subsample comparison.** The child prompt is compared to the parent on these same examples. For this role, the minibatch should be representative — it should test whether the child is genuinely better, not just better on the specific failures it was designed to fix. Including some successes provides calibration.

These roles are in tension. Optimizing purely for Role 1 (all failures) undermines Role 2 (filter becomes vacuous). Optimizing purely for Role 2 (representative sample) undermines Role 1 (not enough failures for the LLM).

The resolution: compose the minibatch deliberately. For a minibatch of size $k$:

- **$\lceil 2k/3 \rceil$ "lesson" slots**: failures, selected for coherence (related to each other)
- **$\lfloor k/3 \rfloor$ "anchor" slots**: successes, selected for relevance (related to the failures, providing contrast)

For $k = 3$: 2 failures + 1 success. The failures teach the LLM what's wrong. The success teaches it what's right. The success also gives the subsample filter a non-trivial anchor — if the child prompt breaks what the success tests, it won't pass.

## The Core Proposal: Structured Lesson Sampling

### Without Content Signals (History-Only)

At GEPA's scale, the interaction history is too sparse to reliably cluster failures by root cause. Each example has ~3 observations over 50 iterations. Co-improvement patterns (examples that improve together, suggesting shared root cause) require co-occurrence in minibatches, which is rare with minibatch size 3 from 50 examples.

Without content signals, the best we can do is **ensure the right composition and maximize diversity through rotation**:

1. **Maintain score estimates** $\hat{s}_Q$ using exponential decay (same as Idea 8):
   $$\hat{s}_Q = s_Q^{\text{last}} \cdot \lambda^{\Delta t} + \mu \cdot (1 - \lambda^{\Delta t})$$
   with adaptive prior $\mu$ = running average of recent scores.

2. **Partition examples** into a failure pool ($\hat{s}_Q < \theta$, e.g., $\theta = 0.5$) and a success pool ($\hat{s}_Q \geq \theta$). Unknown/never-evaluated examples go into the failure pool (should be evaluated promptly).

3. **Epoch-rotate within each pool.** Shuffle the failure pool; deal out failures in order across iterations. When all failures have been dealt, reshuffle. Same for the success pool (slower cycle — fewer slots per iteration).

4. **Compose each minibatch**: draw 2 from the failure epoch, 1 from the success epoch. If the failure pool is empty (prompt solves everything), draw 3 from the success pool and skip reflection (nothing to fix). If the success pool is empty (prompt fails everything), draw 3 from the failure pool (no contrast available, but the LLM still has the prompt text as implicit contrast of "what was intended").

The epoch rotation within each pool guarantees:
- **Coverage**: every failure is shown to the reflection LLM within one failure-epoch (~10 iterations for 20 failures with 2 per batch). Every success is used as an anchor within one success-epoch.
- **Diversity**: the LLM sees different failures each iteration, not the same ones repeatedly. Over 5 iterations, it has seen 10 different failures, likely hitting each major failure mode at least once.
- **No fixation**: unlike weighted sampling, epoch rotation guarantees every example gets exactly one turn per epoch, preventing any example from being over- or under-represented.

This approach doesn't achieve coherence (related failures) — that requires content signals. But it achieves the other two properties (failure signal and contrast) while guaranteeing coverage. It's a strict improvement over both the default epoch sampler (which doesn't prioritize failures) and the history-based sampler (which doesn't guarantee composition or coverage).

### With Content Signals (Failure-Mode Clustering)

If content signals are available — either from embeddings, evaluator feedback text, or a cheap classification LLM call — the minibatch can be constructed as a proper lesson:

1. **Classify each failure by failure mode.** Options, from cheapest to most expensive:
   - **Feedback clustering**: If the evaluator returns textual feedback (e.g., "incorrect, the geometric reasoning is flawed"), cluster failures by feedback similarity (TF-IDF, keyword matching). Nearly free.
   - **Embedding clustering**: Embed all training examples (one-time cost). Cluster failures by embedding proximity. Reuses infrastructure from Idea 7.
   - **LLM classification**: After each evaluation, ask a cheap model: "What type of error is this? One phrase." Group by response. ~$0.001 per classification.

2. **Each iteration, pick one failure-mode cluster to focus on.** Rotate across clusters over iterations (round-robin or weighted by cluster size).

3. **Draw 2 failures from the selected cluster.** The reflection LLM sees two examples of the same type of failure — a clear pattern it can diagnose deeply.

4. **Draw 1 success that's closest to the cluster** (by embedding distance, or from the same feedback category but with a high score). This is the contrastive anchor — "you solve this related problem but fail those. What's different?"

This achieves all three properties: failure signal (2 failures), coherence (same failure mode), and contrast (related success). The reflection LLM receives a proper lesson: "here's a specific type of problem you struggle with, here's what it looks like when you get it right, now figure out the gap."

This composes naturally with Idea 9 (contrastive pairs): the sampling constructs the lesson, and the reflection prompt frames it as a contrastive diagnosis task.

## Why This Should Improve Learning

### The Diagnosis Depth Argument

The reflection LLM's fix quality is limited by its diagnosis quality. Good diagnosis → general fix → good generalization. Bad diagnosis → specific patch → poor generalization.

Diagnosis depth depends on the evidence:
- **1 failure, 2 successes** (default sampler, late optimization): shallow diagnosis. One data point. The LLM might misattribute the failure to a surface quirk rather than a root cause. Fix is narrow.
- **3 unrelated failures** (history-based sampler): scattered diagnosis. Three independent problems. The LLM picks one or hedges across all three. Fix is either narrow (one problem) or shallow (all three).
- **2 related failures + 1 contrasting success** (lesson-based sampler): deep diagnosis. The pattern across the two failures isolates the root cause. The success confirms what's working. The LLM can make a targeted, generalizable fix.

### The Generalization Argument

Fixes generalize when they address root causes, not symptoms. A root cause is, by definition, something that affects multiple examples. The lesson-based sampler explicitly presents multiple examples of the same weakness, forcing the reflection LLM to find the commonality — the root cause. A fix derived from root cause analysis transfers to unseen examples that share the same root cause.

In contrast, a fix derived from a single failure might address a symptom specific to that example. It happens to work on the one example shown but doesn't transfer.

### The Budget Efficiency Argument

With 2 failures + 1 success, the subsample filter has a meaningful anchor. The child must beat the parent on the full batch, including the success. If the child's fix breaks the success (a regression), the filter catches it before the expensive valset evaluation. This preserves budget for more iterations without sacrificing the quality of each iteration's learning.

## Concrete Implementation

### `LessonBasedBatchSampler` (History-Only Version)

```python
class LessonBasedBatchSampler(BatchSampler[DataId, DataInst]):
    """Samples structured minibatches: failures for learning + successes for contrast.

    Maintains two epoch-rotating pools (failure, success) and composes each
    minibatch with a 2:1 failure-to-success ratio. Score estimates use
    exponential decay from interaction history.
    """

    def __init__(
        self,
        minibatch_size: int = 3,
        decay: float = 0.9,
        score_threshold: float = 0.5,
        rng: random.Random | None = None,
    ):
        self.minibatch_size = minibatch_size
        self.decay = decay
        self.score_threshold = score_threshold
        self.rng = rng or random.Random(0)

        # Epoch state for each pool
        self._failure_remaining: list[DataId] = []
        self._success_remaining: list[DataId] = []

    def _estimate_scores(
        self, all_ids: list[DataId], state: GEPAState
    ) -> dict[DataId, float]:
        """Estimate current score for each example using exponential decay."""
        current_iter = state.i
        last_score: dict[DataId, float] = {}
        last_seen: dict[DataId, int] = {}

        for iter_idx, trace in enumerate(state.full_program_trace):
            ids = trace.get("subsample_ids", [])
            scores = trace.get("subsample_scores", [])
            for did, score in zip(ids, scores):
                last_score[did] = score
                last_seen[did] = iter_idx

        # Adaptive prior: average of recent scores
        recent = [s for did, s in last_score.items()
                  if current_iter - last_seen.get(did, 0) < 20]
        mu = sum(recent) / len(recent) if recent else 0.5

        estimates = {}
        for did in all_ids:
            if did in last_score:
                dt = current_iter - last_seen[did]
                decay_factor = self.decay ** dt
                estimates[did] = last_score[did] * decay_factor + mu * (1 - decay_factor)
            else:
                # Never evaluated: assume failing (should be seen soon)
                estimates[did] = 0.0
        return estimates

    def _refill_pool(self, pool_ids: list[DataId]) -> list[DataId]:
        """Shuffle and return a new epoch for a pool."""
        shuffled = list(pool_ids)
        self.rng.shuffle(shuffled)
        return shuffled

    def _draw_from_pool(
        self, remaining: list[DataId], pool_ids: list[DataId], n: int
    ) -> tuple[list[DataId], list[DataId]]:
        """Draw n items from an epoch-rotating pool."""
        drawn = []
        for _ in range(n):
            if not remaining:
                remaining = self._refill_pool(pool_ids)
            if remaining:
                drawn.append(remaining.pop())
        return drawn, remaining

    def next_minibatch_ids(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState
    ) -> list[DataId]:
        all_ids = list(loader.all_ids())
        estimates = self._estimate_scores(all_ids, state)

        # Partition into pools
        failure_ids = [d for d in all_ids if estimates[d] < self.score_threshold]
        success_ids = [d for d in all_ids if estimates[d] >= self.score_threshold]

        # Determine composition
        k = min(self.minibatch_size, len(all_ids))
        n_failures = min(math.ceil(2 * k / 3), len(failure_ids)) if failure_ids else 0
        n_successes = min(k - n_failures, len(success_ids)) if success_ids else 0
        n_failures = k - n_successes  # fill remaining with failures

        # Draw from epoch-rotating pools
        # Filter remaining pools to only include current pool members
        self._failure_remaining = [d for d in self._failure_remaining if d in failure_ids]
        self._success_remaining = [d for d in self._success_remaining if d in success_ids]

        f_drawn, self._failure_remaining = self._draw_from_pool(
            self._failure_remaining, failure_ids, n_failures
        )
        s_drawn, self._success_remaining = self._draw_from_pool(
            self._success_remaining, success_ids, n_successes
        )

        batch = f_drawn + s_drawn

        # If we still need more (edge case: both pools too small)
        while len(batch) < k:
            remaining_ids = [d for d in all_ids if d not in batch]
            if not remaining_ids:
                break
            batch.append(self.rng.choice(remaining_ids))

        return batch
```

### Key Differences from Idea 8's `HistoryBasedBatchSampler`

| Property | Idea 8 (History-Based) | This (Lesson-Based) |
|---|---|---|
| Example selection | Independent weighted sampling | Structured composition (2 failures + 1 success) |
| Coverage guarantee | None (decay as proxy) | Epoch rotation within each pool |
| Learnability term | Yes ($\ell_Q$) | No (too noisy at scale) |
| Subsample filter | Effectively vacuous (all failures) | Meaningful (success provides anchor) |
| Score estimation | Same (exponential decay) | Same (exponential decay) |
| Coherence (related failures) | No | No (requires content signals) |
| Hyperparameters | 2 (decay, optional prior) | 2 (decay, score threshold) |

## The Real Lever: Coherence Through Failure-Mode Clustering

The history-only version above is a solid incremental improvement. But the analysis in this document points to a conclusion: **the biggest improvement in learning quality comes from failure-mode coherence, which requires content signals.**

The cheapest path to coherence is feedback clustering. Most GEPA evaluators already produce textual feedback (e.g., "incorrect — the candidate failed to account for negative numbers"). Simple keyword extraction or TF-IDF similarity on feedback strings can group failures by type. No embeddings, no extra LLM calls, nearly zero cost.

```python
def cluster_failures_by_feedback(
    failure_ids: list[DataId],
    feedback_map: dict[DataId, str],  # from evaluator side_info
    n_clusters: int = 5,
) -> list[list[DataId]]:
    """Group failures by feedback text similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    texts = [feedback_map.get(d, "") for d in failure_ids]
    if not texts or all(t == "" for t in texts):
        return [failure_ids]  # fallback: one big cluster

    tfidf = TfidfVectorizer(max_features=100, stop_words="english")
    X = tfidf.fit_transform(texts)
    n = min(n_clusters, len(failure_ids))
    labels = KMeans(n_clusters=n, random_state=0, n_init="auto").fit_predict(X)

    clusters: dict[int, list[DataId]] = {}
    for did, label in zip(failure_ids, labels):
        clusters.setdefault(label, []).append(did)
    return list(clusters.values())
```

With feedback clustering, the lesson construction becomes:

1. Cluster current failures by feedback similarity
2. Pick the next cluster in round-robin order
3. Draw 2 failures from that cluster
4. Draw 1 success (from the success pool, ideally the one with the most similar feedback to the cluster — or just the next in the success epoch)

The reflection LLM now sees two examples that failed for the same reason, plus a success for contrast. This is the "good lesson" structure described at the start of this document.

## Where This Doesn't Help

**Early optimization (prompt fails on everything).** No successes to use as anchors. No contrast signal. The failure pool is the entire training set. The lesson-based sampler degrades to "3 random failures" — same as any sampler. In this regime, the reflection LLM is just trying to make the prompt work at all; structured lessons don't add much.

**Near convergence (prompt solves almost everything).** Only 2-3 failures remain. The sampler keeps showing the same failures. If they're genuinely hard (unlearnable), the iterations are wasted regardless of how they're composed. In this regime, the prompt is near a local optimum; sampling can't find improvements that don't exist.

**Diverse failure landscape with no clusters.** If every failure is unique (completely different root causes), there are no related failures to pair. Feedback clustering produces singleton clusters. The lesson-based sampler degrades to "2 unrelated failures + 1 success" — slightly better than random (the success provides contrast) but no coherence benefit.

The sweet spot is the **middle regime**: the prompt solves 40-80% of examples, the remaining failures cluster into 3-6 distinct failure modes, and there are enough successes to provide meaningful contrast. This is exactly where most of the optimization budget is spent and where sampling quality matters most.

## Ablation Design

1. **EpochShuffledBatchSampler** — current default, control
2. **HistoryBasedBatchSampler** (Idea 8) — failure-weighted independent sampling
3. **LessonBasedBatchSampler, history-only** — 2 failures + 1 success, epoch rotation, no clustering
4. **LessonBasedBatchSampler, feedback-clustered** — 2 related failures + 1 success
5. **LessonBasedBatchSampler, failures-only (no anchor)** — 3 failures from same cluster, no success — isolates the value of the contrast anchor
6. **LessonBasedBatchSampler, no rotation (weighted)** — same composition but weighted sampling instead of epoch rotation — isolates the value of coverage guarantees

Compare (1) vs (3) for the headline: does structured composition beat random?
Compare (2) vs (3) to determine whether composition (2:1 ratio) or priority weighting matters more.
Compare (3) vs (4) to measure the value of failure-mode coherence.
Compare (4) vs (5) to measure the value of the contrast anchor.
Compare (3) vs (6) to measure the value of epoch-based coverage.

Key metrics: val score over iterations (learning speed), subsample filter pass rate (if near 100%, the filter is vacuous), final held-out test score (generalization), and — for the clustered variant — reflection LM fix specificity (does the LLM produce more targeted edits?).

## Recommendation

**Start with variant (3)**: the history-only lesson-based sampler. It's a drop-in `BatchSampler` replacement, zero external dependencies, two hyperparameters with obvious defaults (`decay=0.9`, `threshold=0.5`). It captures the two most impactful insights — structured composition and coverage guarantees — without requiring content analysis infrastructure.

If (3) shows improvement over (1) and (2), **add feedback clustering (variant 4)** as the next step. This is the cheapest path to failure-mode coherence and requires only that the evaluator returns textual feedback in the side_info — which most GEPA evaluators already do.

Save embedding-based clustering for when the dataset is large enough (200+ examples) that feedback text alone can't distinguish failure modes reliably.

## Summary

| Sampling Property | Default Epoch | Idea 8 (History) | This (Lesson-Based) |
|---|---|---|---|
| Prioritizes failures | No | Yes (weighted) | Yes (structured) |
| Guarantees coverage | Yes (epoch) | No (decay proxy) | Yes (epoch per pool) |
| Meaningful subsample filter | Yes (representative) | No (all failures) | Yes (success anchor) |
| Failure coherence | No | No | Optional (with clustering) |
| Contrast signal | Accidental | No | Yes (deliberate success) |
| Budget efficiency | Moderate | Low (filter vacuous) | High (filter active) |

The default sampler is good for generalization but bad for learning signal. Idea 8 is good for learning signal but bad for generalization and budget. Lesson-based sampling aims for both: structured minibatches that teach the reflection LLM well while maintaining the generalization safeguards that keep the optimization honest.
