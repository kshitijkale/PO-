# Embedding-Informed Evolutionary Search

## The Observation

GEPA operates in *opaque string space*. Candidates are `dict[str, str]`, mutations are LLM generations, and the only signal about relationships between candidates is the genealogy tree (parent/child) and per-example scores. The optimization has no geometric awareness — no sense of distance between candidates, no concept of search coverage, no way to recognize that two textually different candidates embody the same strategy, or that a proposed mutation is semantically identical to one tried three iterations ago.

Meanwhile, modern embedding models map text to dense vectors where semantic similarity corresponds to vector proximity. An embedding model call is 100-1000x cheaper than a generation call. This is an untapped resource: GEPA already pays for expensive LLM generation and evaluation, but does no cheap geometric reasoning about its own search space.

The question is where embedding-based geometry actually buys something that raw strings and scores cannot provide. Not everywhere — some uses would be superficial. The valuable uses are those where GEPA currently either (a) wastes budget rediscovering things it already knows, or (b) makes decisions that would be better informed by semantic relationships.

## Where Embeddings Are Genuinely Useful

### 1. Retrieval-Augmented Reflection (the strongest use case)

**The problem it solves:** As GEPA runs, it accumulates a history of (candidate, evaluation_feedback, mutation, accepted/rejected). This history contains rich signal — what failures look like, what fixes work, what doesn't transfer. But the reflection LM sees none of this. Each reflection call starts fresh: here's the candidate, here's the minibatch feedback, propose something better.

Meta-reflection (Idea 1) addresses this by periodically summarizing the history. But summarization is lossy — a summary of 50 iterations can't preserve the specific details that matter for the current failure.

**The embedding solution:** Embed every piece of evaluation feedback (the side_info / ASI) as the run progresses. When the reflection LM is about to propose a mutation for the current candidate:

1. Embed the current evaluation feedback (the reflective dataset for this iteration).
2. Retrieve the K most similar past evaluation feedbacks from the history.
3. For each retrieved feedback, also retrieve what mutation was proposed and whether it was accepted.
4. Inject these precedents into the reflection prompt: "Here are similar failures from earlier in this run, and what was tried."

This is **case-based reasoning** powered by embedding retrieval. The reflection LM gets targeted, relevant history — not a generic summary, but specific cases where similar things went wrong.

**Why this works better than full-history meta-reflection:** The context window is finite. You can't show the LLM 50 iterations of history. But you can show it 3-5 *relevant* precedents retrieved by embedding similarity. The retrieval acts as an attention mechanism over the optimization history.

**Concrete implementation:**

```python
class ReflectionMemory:
    """Embedding-indexed memory of past reflection episodes."""

    def __init__(self, embed_fn, k=5):
        self.embed_fn = embed_fn  # text -> vector
        self.episodes = []  # list of {feedback_text, feedback_embedding, mutation, accepted, score_delta}
        self.k = k

    def add(self, feedback_text, mutation, accepted, score_delta):
        emb = self.embed_fn(feedback_text)
        self.episodes.append({
            "feedback_text": feedback_text,
            "feedback_embedding": emb,
            "mutation_summary": mutation,
            "accepted": accepted,
            "score_delta": score_delta,
        })

    def retrieve(self, current_feedback_text):
        query_emb = self.embed_fn(current_feedback_text)
        # cosine similarity ranking
        similarities = [cosine_sim(query_emb, ep["feedback_embedding"]) for ep in self.episodes]
        top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:self.k]
        return [self.episodes[i] for i in top_k_indices]
```

This slots into `ReflectiveMutationProposer.propose()` between building the reflective dataset (line 279) and calling `propose_new_texts()` (line 310). The retrieved precedents are appended to the reflective dataset or injected into the reflection prompt template as a new section.

**What gets embedded:** The textual representation of the reflective dataset — the formatted side_info for each example in the minibatch. This is exactly the `<side_info>` content that gets plugged into the reflection prompt template in `InstructionProposalSignature.prompt_renderer()`.

### 2. Redundancy Detection (saving evaluation budget)

**The problem it solves:** GEPA sometimes proposes candidates that are semantically near-identical to previously evaluated candidates. The text might differ (reworded, reorganized) but the strategy is the same. These candidates waste evaluation budget — they'll get similar scores to their near-duplicate.

The evaluation cache (`EvaluationCache` in `src/gepa/core/state.py`) catches *exact* duplicates (via sha256 hash of the candidate dict). But semantic near-duplicates — same strategy, different wording — pass through.

**The embedding solution:** Before evaluating a newly proposed candidate on the valset, embed it and check distance to all previously evaluated candidates. If the minimum distance is below a threshold, skip evaluation and reuse the nearest neighbor's scores as an estimate.

```python
def should_skip_evaluation(new_candidate_text, candidate_embeddings, threshold=0.05):
    new_emb = embed(new_candidate_text)
    min_dist = min(1 - cosine_sim(new_emb, prev_emb) for prev_emb in candidate_embeddings)
    return min_dist < threshold
```

This is aggressive — it might skip a candidate that has a small but important difference. So use it as a soft signal: instead of hard skipping, reduce the priority of near-duplicate candidates, or flag them to the reflection LM ("your proposal is very similar to candidate #7 which scored 0.65 — try something more different").

**Where it intervenes:** In `GEPAEngine._run_full_eval_and_add()` (line 146 of `engine.py`), before the expensive valset evaluation.

### 3. Diversity-Aware Candidate Selection

**The problem it solves:** Idea 5 (Quality-Diversity) proposed using the LLM to extract behavior characterizations — an expensive operation (one LLM call per candidate). Embeddings offer a cheaper continuous proxy.

Two candidates with similar embeddings likely use similar strategies. Two candidates with distant embeddings likely use different strategies. This isn't as semantically precise as LLM-extracted behavior labels, but it's 100x cheaper and supports continuous distance metrics rather than discrete categories.

**The embedding solution:** Embed all candidates on the Pareto front. When selecting a parent for mutation, prefer candidates that are distant from recently mutated parents — i.e., encourage the search to explore different regions of the embedding space rather than repeatedly refining the same neighborhood.

```python
class DiversityAwareParetoCandidateSelector(CandidateSelector):
    def __init__(self, embed_fn, recency_window=5, diversity_weight=0.3, rng=None):
        self.embed_fn = embed_fn
        self.recent_parents = []  # last N selected parent embeddings
        self.diversity_weight = diversity_weight
        self.rng = rng or random.Random(0)

    def select_candidate_idx(self, state):
        pareto_programs = get_pareto_front_programs(state)

        if not self.recent_parents:
            # No history yet, fall back to standard Pareto selection
            return standard_pareto_select(state, self.rng)

        # Score each Pareto candidate by: (1-w)*pareto_score + w*distance_from_recent
        scores = []
        recent_centroid = mean(self.recent_parents)
        for pid in pareto_programs:
            emb = self.embed_fn(candidate_to_text(state.program_candidates[pid]))
            dist = 1 - cosine_sim(emb, recent_centroid)
            pareto_score = state.per_program_tracked_scores[pid]
            combined = (1 - self.diversity_weight) * pareto_score + self.diversity_weight * dist
            scores.append((pid, combined))

        # Weighted random selection
        return weighted_sample(scores, self.rng)
```

This naturally alternates between exploiting the best-known region and exploring distant alternatives, without the overhead of LLM-based behavior characterization.

### 4. Informed Minibatch Sampling

**The problem it solves:** `EpochShuffledBatchSampler` samples minibatches randomly (shuffled epoch order). This means some minibatches are redundant — three similar examples that exercise the same aspect of the candidate — while other minibatches happen to be maximally diverse. The reflection LM's diagnosis quality depends heavily on minibatch composition: diverse minibatches produce richer, more general diagnoses.

**The embedding solution:** Embed all training examples. Sample minibatches that maximize coverage of the embedding space:

- **Diverse minibatches:** Select examples that are far apart in embedding space. This gives the reflection LM a broad view of the problem landscape in each minibatch, encouraging general rather than specific diagnoses.
- **Targeted minibatches:** When the Pareto front reveals persistent weak spots (val examples where all candidates score poorly), select training examples that are nearest to those hard val examples in embedding space. This focuses reflection on the areas where improvement is most needed.

For diverse sampling, use a greedy farthest-point heuristic:
```python
def sample_diverse_minibatch(example_embeddings, k):
    # Start with a random example
    selected = [random.choice(range(len(example_embeddings)))]
    for _ in range(k - 1):
        # Add the example farthest from all currently selected
        dists = [min(distance(example_embeddings[i], example_embeddings[j])
                     for j in selected)
                 for i in range(len(example_embeddings)) if i not in selected]
        next_idx = max(range(len(dists)), key=lambda i: dists[i])
        selected.append(next_idx)
    return selected
```

This replaces `EpochShuffledBatchSampler` with `DiverseMinibatchSampler` in `src/gepa/strategies/batch_sampler.py`.

## Where Embeddings Are NOT Useful (Avoiding False Leads)

**Candidate interpolation / averaging in embedding space.** You can't meaningfully "average" two prompt embeddings and decode back to text. Embedding space doesn't support this — the mapping is many-to-one and non-invertible. The merge proposer's LLM-based approach (combining strategies from two candidates using LLM reasoning) is fundamentally better for combining candidates.

**Replacing LLM-based reflection with embedding-based optimization.** You might think: embed candidates, compute gradient in embedding space toward high-scoring regions, decode to text. But (a) the embedding-to-text mapping is non-invertible, (b) the score landscape in embedding space is not smooth (similar prompts can have very different scores), and (c) the whole point of GEPA is that LLMs can reason about *why* things fail, which embedding geometry cannot capture. Embeddings should *inform* LLM reasoning, not replace it.

**Measuring candidate "complexity" via embedding norm or variance.** Embedding norms don't correlate well with semantic complexity. A long, verbose prompt might have a similar embedding norm to a short, precise one. Complexity in the relevant sense (how many specific heuristics vs. general principles) is a semantic property that embeddings don't capture well.

## The Unifying Principle

Embeddings give GEPA **spatial awareness** of its own search process. Without embeddings, GEPA knows:
- What candidates it has tried (exact text)
- How they scored (exact numbers)
- Who descended from whom (genealogy)

With embeddings, GEPA additionally knows:
- How *similar* candidates are to each other (continuous distance)
- How *similar* current failures are to past failures (retrieval)
- How *spread out* the search has been (coverage)
- How *diverse* a minibatch is (geometric span)

This is the difference between a blind search (propose, evaluate, accept/reject) and an informed search (propose in under-explored directions, retrieve relevant past experience, sample diverse training signal). The embedding model serves as a cheap geometric oracle over the text space.

## Cost Analysis

Embedding calls are very cheap relative to GEPA's existing LLM costs:

| Operation | Model | Approx. cost per call |
|---|---|---|
| Reflection LLM call | GPT-4.1/5 | $0.01 - $0.10 |
| Evaluation (if LLM-based) | GPT-4.1-mini | $0.001 - $0.01 |
| Embedding | text-embedding-3-small | $0.00002 (0.002 cents) |

At ~1000x cheaper than a reflection call, embedding every candidate and every piece of feedback across a 200-iteration run costs on the order of cents. The compute cost is dominated by the vector similarity search, which for a few hundred vectors is trivially fast (naive numpy cosine similarity suffices — no need for approximate nearest neighbor indices).

## Implementation Plan

### Phase 1: Retrieval-Augmented Reflection (highest value, moderate effort)
- New class `ReflectionMemory` in `src/gepa/strategies/reflection_memory.py`
- Integrates into `ReflectiveMutationProposer` — stores feedback after each iteration, retrieves before each reflection
- Requires: an embedding function (default to litellm embeddings, matching the existing litellm dependency)
- New section in reflection prompt template: "Similar failures from earlier in this optimization, and what was tried:"

### Phase 2: Redundancy Detection (high value, low effort)
- Embed each new candidate before valset evaluation
- Skip or deprioritize near-duplicates
- Integrates into `GEPAEngine._run_full_eval_and_add()`

### Phase 3: Diversity-Aware Selection (moderate value, moderate effort)
- New `DiversityAwareParetoCandidateSelector` in `src/gepa/strategies/candidate_selector.py`
- Tracks recent parent embeddings, biases selection toward distant candidates

### Phase 4: Informed Minibatch Sampling (moderate value, moderate effort)
- New `DiverseMinibatchSampler` and `TargetedMinibatchSampler` in `src/gepa/strategies/batch_sampler.py`
- Precompute training example embeddings once, use for all iterations

## Interaction With Other Ideas

**With Meta-Reflection (Idea 1):** Meta-reflection summarizes full history into principles. Retrieval-augmented reflection (above) retrieves specific relevant precedents. These are complementary — principles provide general guidance, retrieved precedents provide specific examples. Use both: principles as a prompt section, retrieved cases as another.

**With Stratified Reflection (Idea 2):** Embedding-based clustering of past failure diagnostics could automate the "abstraction" stage. Instead of asking the LLM to abstract, cluster past diagnostics by embedding similarity and label each cluster — the cluster structure reveals failure classes automatically.

**With Quality-Diversity (Idea 5):** Embeddings provide a cheap behavior characterization for the QD archive. Use LLM-based BC for the primary behavior dimensions (interpretable, strategy-level) and embedding distance as a secondary diversity signal (continuous, cheap). The archive cells could be defined by LLM-extracted labels, but candidate similarity within cells is measured by embedding distance.

**With Adversarial Curriculum (Idea 3):** Embed adversarial examples alongside existing training examples to ensure the adversary generates examples in under-represented regions of the embedding space, avoiding redundancy in the generated curriculum.

## Ablation Design

1. **Base GEPA** (no embeddings, control)
2. **Retrieval-augmented reflection only** (K=3 retrieved precedents)
3. **Retrieval-augmented reflection + redundancy detection**
4. **Diversity-aware selection only**
5. **Informed minibatch sampling only**
6. **All four embedding-informed components together**

For (2) vs (1): does showing relevant precedents improve proposal acceptance rate and final val score?
For (4) vs (1): does diversity-aware selection improve exploration and final val score?
For (6) vs (1): do the components compose, or does the combined effect plateau?

Also measure: proposals skipped by redundancy detection (budget saved), embedding space coverage over time (search diversity), and retrieval precision (are the retrieved precedents actually relevant, as judged by the LLM or by human inspection).
