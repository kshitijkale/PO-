# Failure-Directed Minibatch Sampling via Train-Side Embeddings

## Core Insight

A prompt's identity is shaped by its interaction history — the specific examples its lineage reflected on during evolution. With random minibatch sampling (the current `EpochShuffledBatchSampler`), the lineage's coverage of the training space is determined by chance. Some regions get visited early and often; others remain unexplored for many iterations. The prompt develops strategies for the regions it's seen and has blind spots where it hasn't.

Random sampling wastes iterations. If the prompt already handles a region well, drawing more examples from there gives the reflection LM nothing useful — it sees good scores, proposes minor tweaks, and the val gate probably rejects. Meanwhile, a different region has real failures that the reflection LM could fix if it saw them.

**The proposal: direct minibatch sampling toward the prompt's weaknesses using train-side information only. Embed training examples, track the prompt's performance across the training space, and sample from regions where it fails or has never been tested.**

Critically, the val set is never consulted for sampling decisions. It remains a pure, independent generalization gate. All directed sampling signal comes from the training side.

## Why Val-Free Matters

An earlier version of this idea used val failures to direct train sampling: embed val failures, find nearby train examples, sample those. This is wrong. It creates coupling between the val set and the training process — the val set becomes a training signal rather than an independent check. Over many iterations, the prompt gets sculpted to address val-specific weaknesses, which is overfitting to val disguised as generalization improvement.

The fix: use the train evaluations that GEPA already performs. Every iteration, `ReflectiveMutationProposer.propose()` evaluates the parent candidate on a train minibatch with `capture_traces=True` (line 210 in `reflective_mutation.py`). These scores accumulate in `state.full_program_trace`. Over many iterations, we have a sparse but growing map of how the current prompt (and its ancestors) performed across the training set.

This is train-side information. Using it to direct future train sampling doesn't compromise val independence.

## Mechanism

### Data Structures

1. **Embedding index**: Embed all training examples once at the start. Store as a matrix $E \in \mathbb{R}^{n \times d}$ where $n$ is the number of train examples and $d$ is the embedding dimension.

2. **Train-side performance map**: A dict `{DataId: (score, iteration)}` tracking the most recent score for each train example that the current prompt's lineage has been evaluated on. Updated every iteration from the minibatch evaluation. Stale entries (from many iterations ago) can be discounted or refreshed.

### Weight Computation

For each train example $x_i$ with embedding $e_i$, compute a sampling weight that combines **exploitation** (target known failures) with **exploration** (probe blind spots).

Let $H$ be the set of train examples previously evaluated by this prompt's lineage, with scores $\{s_j\}_{j \in H}$.

**Case 1: $x_i \in H$ (has been evaluated)**

The prompt was tested on this example. Use its score directly:

$$w_i^{\text{exploit}} = (1 - s_i)^\beta$$

where $\beta > 0$ controls how aggressively we target failures.
- $\beta = 1$: linear. Score 0.3 → weight 0.7. Score 0.9 → weight 0.1.
- $\beta = 2$: quadratic. Concentrates more on the worst failures. Score 0.3 → weight 0.49. Score 0.9 → weight 0.01.

**Case 2: $x_i \notin H$ (never evaluated)**

Predict its score by kernel-weighted interpolation from nearby evaluated examples:

$$\hat{s}_i = \frac{\sum_{j \in H} K(e_i, e_j) \cdot s_j}{\sum_{j \in H} K(e_i, e_j)}$$

where $K$ is a kernel, e.g., Gaussian: $K(e_i, e_j) = \exp(-\|e_i - e_j\|^2 / 2\sigma^2)$.

This is Nadaraya-Watson regression — a simple nonparametric estimator. If nearby evaluated examples all score high, $\hat{s}_i$ is high (probably solved). If they score low, $\hat{s}_i$ is low (probably failing). If no nearby examples have been evaluated, the denominator is small and $\hat{s}_i$ is unreliable — which is captured by the exploration term.

Use the predicted score for the exploitation weight:

$$w_i^{\text{exploit}} = (1 - \hat{s}_i)^\beta$$

**Exploration bonus for all examples:**

$$w_i^{\text{explore}} = \exp\left(-\max_{j \in H} \text{sim}(e_i, e_j) / \tau\right)$$

where $\text{sim}$ is cosine similarity and $\tau$ is a temperature. This is high when $x_i$ is far from any evaluated example (unexplored territory) and low when $x_i$ is close to evaluated examples (well-explored).

**Combined weight:**

$$w_i = w_i^{\text{exploit}} + \gamma \cdot w_i^{\text{explore}}$$

$\gamma$ controls the exploration-exploitation balance. The sampling probability for the minibatch is $P(x_i) \propto w_i$.

### Natural Annealing

No manual schedule is needed. The balance shifts automatically over the course of optimization:

- **Early** (few iterations, small $H$): most examples are unevaluated, $w^{\text{explore}}$ is high for most examples → broad exploration, similar to random sampling.
- **Mid** (moderate $H$, some clear failures): failure regions are identified, $w^{\text{exploit}}$ dominates for those regions → targeted failure fixing.
- **Late** (large $H$, most examples score well): both terms are small for well-solved examples, residual weight concentrates on the remaining hard cases → fine-grained polishing.

## Where This Intervenes in the Codebase

### New: `FailureDirectedBatchSampler`

A new `BatchSampler` implementation in `src/gepa/strategies/batch_sampler.py`:

```python
class FailureDirectedBatchSampler(BatchSampler[DataId, DataInst]):
    """Samples minibatches weighted toward failure regions and unexplored areas
    of the training space, using embeddings for spatial structure."""

    def __init__(
        self,
        embeddings: dict[DataId, np.ndarray],  # precomputed embeddings
        minibatch_size: int = 3,
        beta: float = 1.0,        # failure targeting aggressiveness
        gamma: float = 0.3,       # exploration bonus weight
        sigma: float = 0.5,       # kernel bandwidth for score interpolation
        tau: float = 0.3,         # temperature for exploration bonus
        rng: random.Random | None = None,
    ):
        ...

    def next_minibatch_ids(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState
    ) -> list[DataId]:
        # 1. Build/update train-side performance map from state.full_program_trace
        # 2. Compute w_exploit for all train examples (direct or interpolated)
        # 3. Compute w_explore for all train examples
        # 4. Combine: w = w_exploit + gamma * w_explore
        # 5. Sample minibatch_size examples without replacement, P(x_i) ∝ w_i
        ...
```

This is a drop-in replacement for `EpochShuffledBatchSampler`. The `BatchSampler` protocol (line 13-14 of `batch_sampler.py`) only requires `next_minibatch_ids(loader, state) -> list[DataId]`, so no changes are needed to the engine or proposer.

### Building the performance map from existing state

`state.full_program_trace` is a list of dicts, one per iteration. Each entry contains `subsample_ids` (which train examples were evaluated) and `subsample_scores` (their scores). This is all the information needed to build the performance map — no new tracking infrastructure required.

The key subtlety: the performance map should reflect the *current* prompt's abilities, not its distant ancestors'. A score from 30 iterations ago (on an ancestor candidate) may not reflect the current prompt's performance. Options:
- **Most recent only**: only use scores from the most recent evaluation of each example. Simple but sparse.
- **Decay-weighted**: weight scores by recency, e.g., $s_i^{\text{effective}} = s_i \cdot \lambda^{t - t_i}$ where $t$ is the current iteration and $t_i$ is when the score was recorded. Recent scores dominate; stale scores fade.
- **Lineage-aware**: only use scores from evaluations of the current prompt or its direct ancestors (following `parent_program_for_candidate` links). This is more accurate but requires tracing the lineage graph.

Start with decay-weighted for simplicity. The decay rate $\lambda$ (e.g., 0.95) controls how quickly old scores become irrelevant.

### Embedding computation

Embeddings are computed once at initialization. For the `optimize_anything` API, this happens in the `optimize_anything()` function before engine construction. For the lower-level `gepa.optimize()` API, the user passes embeddings (or an embedding function) to the sampler.

Cost: embedding 100 examples with a model like `text-embedding-3-small` costs ~$0.001 and takes <1 second. Negligible compared to any LLM call.

## Interaction with the Val Gate

The val gate is completely unaffected. The sampling changes which train examples the reflection LM sees. The val gate still independently evaluates every new candidate on the full val set.

The expected interaction: directed sampling produces mutations that target real weaknesses (as identified from train-side failures). These mutations are more likely to address weaknesses that also manifest on the val set (because train and val are drawn from the same distribution). The val gate acceptance rate should increase — not because the gate is weaker, but because the proposals are better targeted.

If directed sampling overfits to a train-specific failure pattern that doesn't exist in val, the val gate catches it and rejects. The mechanism is self-correcting: val rejections don't affect future sampling (no val leakage), but they do prevent overfitted mutations from entering the population.

## Relationship to Active Learning

This is closely related to **pool-based active learning** (Settles, 2009), where a learner selects which unlabeled examples to query next. The parallels:

| Active Learning | Failure-Directed Sampling |
|---|---|
| Unlabeled pool | Train examples not yet evaluated on current prompt |
| Label oracle | Evaluating the prompt on a train example |
| Model uncertainty | Predicted failure from embedding interpolation |
| Query strategy | Sampling weight function |
| Goal: label-efficient learning | Goal: iteration-efficient optimization |

The key difference: in active learning, each query is expensive (requires a human label). In GEPA, the query cost (evaluating the prompt on a train example) is already paid as part of the standard iteration — we're just choosing *which* examples to pay for more carefully.

The active learning literature provides theoretical backing for why directed sampling helps. The classic result: uncertainty sampling achieves the same accuracy as random sampling with exponentially fewer queries in certain problem classes (specifically, when the hypothesis class has bounded disagreement coefficient). The GEPA analog: directed sampling should achieve the same val score with fewer iterations.

## Relationship to Bayesian Optimization

The weight function is a simplified acquisition function. In Bayesian Optimization with Gaussian Processes:

- The GP posterior mean $\mu(x)$ corresponds to our predicted score $\hat{s}_i$
- The GP posterior variance $\sigma^2(x)$ corresponds to our exploration bonus $w_i^{\text{explore}}$
- The Expected Improvement acquisition function combines both, similar to our $w_i = w_i^{\text{exploit}} + \gamma \cdot w_i^{\text{explore}}$

The GP framing is more principled (it provides a proper posterior over the performance surface) but also heavier ($O(n^3)$ for $n$ observations). The kernel-weighted approach is $O(|H| \cdot n)$ per iteration — linear in the number of observations, which is much cheaper. For small-to-medium datasets (50-200 examples), both are fast. For larger datasets, the kernel approach scales better.

We don't use the full GP formalism because the embedding space is already an approximation of the "problem space" — adding a second approximation (the GP) on top has diminishing returns. The simple kernel-weighted approach captures the main effect (exploit failures, explore blind spots) without the machinery.

## Design Decisions

**Kernel choice.** Gaussian kernel on L2 distance, or cosine similarity? Cosine similarity is the standard for text embeddings (embedding models are typically trained with cosine objectives). Use cosine similarity: $K(e_i, e_j) = \exp(\text{cos\_sim}(e_i, e_j) / \sigma)$.

**Bandwidth $\sigma$.** Controls how broadly each evaluated example's score "radiates" to nearby unevaluated examples. Too small: only very close examples are informed, most predictions default to the prior. Too large: the entire space gets the same predicted score (global average), losing spatial structure. A reasonable default: set $\sigma$ so that the average example has ~5-10 neighbors with $K > 0.1$. This can be calibrated from the pairwise similarity distribution of the training embeddings.

**Minibatch diversity.** Pure weighted sampling might draw 3 examples from the same cluster (all near the same failure). This is redundant — the reflection LM gets three versions of the same problem. Adding a diversity term (DPP sampling, or simply rejection-sampling to enforce a minimum pairwise distance) ensures the minibatch covers distinct failure regions.

**Score staleness.** The performance map degrades over iterations as the prompt evolves. Decay-weighted scores ($\lambda = 0.95$) give a half-life of ~14 iterations, which means scores from >30 iterations ago are nearly zeroed out. This is reasonable — the prompt has likely changed significantly in 30 iterations.

**Handling the cold start.** In the first few iterations, $H$ is tiny (3-6 examples). The exploration term dominates, producing near-uniform sampling — essentially falling back to random sampling. This is correct behavior: with no performance information, you should explore broadly.

## Hyperparameter Defaults and Sensitivity

| Parameter | Default | Role | Sensitivity |
|---|---|---|---|
| $\beta$ | 1.0 | Failure targeting aggressiveness | Low — 1.0 vs 2.0 is a mild difference |
| $\gamma$ | 0.3 | Exploration vs exploitation | Medium — too high wastes iterations exploring solved regions |
| $\sigma$ | 0.5 | Kernel bandwidth | Medium — affects interpolation quality |
| $\tau$ | 0.3 | Exploration bonus temperature | Low — controls sharpness of exploration signal |
| $\lambda$ | 0.95 | Score decay rate | Low — anything in [0.9, 0.99] is reasonable |

The most important parameter is $\gamma$. Start with 0.3 (mostly exploit, some explore) and tune if needed.

## Expected Outcomes

1. **Faster convergence**: Each iteration addresses a real weakness instead of revisiting solved regions. The reflection LM gets high-signal minibatches.
2. **Higher val gate acceptance rate**: Proposals target genuine weaknesses that likely also manifest on val, so more proposals pass the independent val check.
3. **Better final generalization**: The prompt's lineage is forced to interact with the full breadth of the training space, not just the regions that random sampling happened to surface early.
4. **Natural exploration-to-exploitation annealing**: Early iterations explore broadly; later iterations focus on remaining hard cases. No manual schedule needed.

## Ablation Design

1. **EpochShuffledBatchSampler** (current default, control)
2. **FailureDirectedBatchSampler, $\gamma = 0$** (pure exploitation — only target known failures, no exploration)
3. **FailureDirectedBatchSampler, $\gamma = 0.3$** (balanced, recommended default)
4. **FailureDirectedBatchSampler, $\gamma = 1.0$** (heavy exploration — prioritize unseen regions)
5. **FailureDirectedBatchSampler with diversity-constrained minibatches**
6. **Random sampling with same evaluation budget** (controls for "is it just seeing more examples?")
7. **Val-directed sampling** (uses val failures to direct train sampling — the bad version, to confirm it overfits)

Compare (1) vs (3) for the headline result. Compare (2) vs (3) vs (4) to validate the exploration-exploitation balance. Compare (3) vs (7) to confirm that val-free sampling generalizes better than val-directed. Track: val score over iterations (convergence speed), val gate acceptance rate, train-side performance map coverage, and final test-set score (the true generalization measure).
