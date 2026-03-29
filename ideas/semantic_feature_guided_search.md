# Semantic Feature-Guided Search

## Core Thesis

GEPA's current reflection loop is memoryless — each iteration sees a random minibatch and reflects from scratch. The accumulated evaluation history is a rich dataset that goes mostly unused.

The idea: **decompose the monolithic LLM reflection into a hybrid pipeline where LLMs extract structured features, classical ML/math finds patterns across iterations, and LLMs render the patterns back into targeted guidance for the proposer.**

LLMs can create features that classical ML can't (semantic mistake types, strategy identification, causal reasoning). Classical ML can find patterns that LLMs can't (statistical trends across hundreds of evaluations, correlation structures, optimal sampling). The composition is strictly more powerful than either alone.

## General Architecture

```
  Eval traces (candidate, input, output, score, side_info)
       |
       v
  +-------------------+
  | LLM Feature       |  text -> structured features
  | Extraction        |  (classify, extract, embed, compare, decompose)
  +--------+----------+
           |
           v
  +-------------------+
  | Structured Store   |  grows across iterations
  | (dataframe / DB)   |  rows = evaluations, columns = features
  +--------+----------+
           |
           v
  +-------------------+
  | Classical ML/Math  |  pattern discovery
  | (cluster, regress, |  (operates on what LLMs can't see in-context)
  |  sample, optimize) |
  +--------+----------+
           |
           v
  +-------------------+
  | LLM Rendering      |  patterns -> natural language guidance
  +--------+----------+
           |
           v
  Injection into GEPA
  (minibatch selection, reflection prompt, candidate seeding,
   module selection, merge selection, stopping decisions)
```

## LLM Operators (text <-> structure)

These are the atomic operations that convert between unstructured evaluation traces and structured features:

**Classify**: trace -> category
- "This failure is an: arithmetic error"
- "The candidate uses strategy: chain-of-thought with self-correction"
- "This input is type: multi-step word problem"

**Extract**: trace -> key-value pairs
- {strategy: "CoT", has_examples: true, tone: "formal", length: 347}
- {failure_location: "step 3", failure_severity: "wrong_answer" vs "partial_credit"}

**Embed**: text -> dense vector
- Candidate embedding, input embedding, output embedding
- Difference embeddings (parent - child)

**Compare**: (trace_a, trace_b) -> structured diff
- "Parent -> child: added few-shot examples, removed verbose preamble"
- "Candidate A solves geometry by coordinates; candidate B uses synthetic geometry"

**Decompose**: candidate -> sub-components with roles
- "This prompt has: persona section, instruction section, format constraints, examples"
- "This code has: input parsing, core algorithm, output formatting"

**Attribute**: (candidate, input, output) -> causal links
- "The phrase 'think step by step' caused the model to show work on line 3"
- "The missing edge case in the prompt caused failure on inputs with negative numbers"

## Classical Operators (pattern discovery on structured data)

**Cluster**: group similar items
- Failure mode clustering
- Candidate strategy clustering
- Input difficulty clustering

**Regress/Classify**: predict outcomes from features
- features -> score (surrogate model)
- features -> pass/fail
- feature importances (which features matter most)

**Correlate/Associate**: find feature-outcome relationships
- "Feature X predicts +0.3 score"
- "Candidates with property A AND property B score 2x higher"
- Association rules: {CoT, examples} -> high_score (support=0.7, confidence=0.85)

**Sample/Select**: choose informative subsets
- Active learning: pick examples that maximize information gain
- Curriculum: order by difficulty
- Diversity sampling: cover the feature space

**Surrogate-Optimize**: search in feature space
- Fit GP/RF on (features -> score), use acquisition function to pick next target
- Bayesian optimization in the LLM-extracted feature space

**Dimensionality Reduction**: find effective search dimensions
- PCA/UMAP on candidate embeddings
- Discover "the 3 axes that matter" in the candidate space

**Graph Algorithms**: on candidate lineage + features
- Find productive lineage paths in the tree
- Detect cycles (oscillating between two strategies)

**Frequent Pattern Mining**: across evaluation database
- "Failure mode X co-occurs with input property Y in 80% of cases"
- "Successful candidates all share sub-pattern Z"

---

## Instantiations

### 1. Failure Mode Concentration

**The problem**: Random minibatches mix failure modes. If a prompt fails on arithmetic 40% of the time and reading comprehension 10%, a random minibatch of 3 might show 1 arithmetic, 1 reading comp, 1 success — the signal is diluted.

**The pipeline**:
```
LLM.classify(candidate, input, output, score) -> mistake_type
  for each evaluation across all iterations
     |
     v
accumulate: {mistake_type: [(candidate, input, output, score), ...]}
     |
     v
rank mistake types by (frequency x severity)
     |
     v
sample ALL examples with the dominant uncorrected failure mode
     |
     v
reflect with concentrated signal:
  "Here are 12 examples all showing the same failure: [type].
   The current candidate fails this way because..."
```

**Injection point**: Minibatch selection (replace random with failure-mode-concentrated sampling)

**Classical technique**: Counting/ranking, possibly hierarchical clustering of mistake types

**Why it works**: Forces the proposer to fix one thing well instead of trying to address 3 different issues with a single edit. The reflection LLM can see the pattern clearly when all examples share the same failure mode.

**Variants**:
- **Rotating focus**: Fix the top failure mode, then re-classify, fix the next one (curriculum over failure modes)
- **Stratified sampling**: Instead of pure concentration, sample proportional to failure mode frequency but with a minimum per stratum
- **Diminishing returns detection**: If the same failure mode persists after K attempts to fix it, deprioritize it and move to the next one (maybe it's inherent to the task, not fixable by the candidate)

**Cost**: 1 LLM classification call per evaluation. For expensive evaluators (code execution, agent runs), this is negligible. For cheap evaluators, consider classifying in batches or using a small/fast classifier model.

### 2. Change Attribution / Diff Analysis

**The problem**: GEPA tracks parent-child lineage but doesn't systematically analyze *what types of changes* were productive vs. unproductive.

**The pipeline**:
```
for each (parent -> child) transition in the lineage tree:
  LLM.compare(parent_candidate, child_candidate) -> change_description
  record: (change_type, delta_score, parent_score, child_score)
     |
     v
accumulate across all iterations
     |
     v
cluster change types, compute statistics:
  - avg delta_score per change type
  - success rate per change type
  - interaction effects (change type A helps when parent has property B)
     |
     v
inject into reflection prompt:
  "Historical analysis of what works:
   - Adding worked examples: +0.18 avg improvement (8 instances)
   - Making instructions more specific: +0.12 avg (5 instances)
   - Shortening the prompt: -0.05 avg (3 instances, generally hurts)
   - Changing tone to formal: no significant effect"
```

**Injection point**: Reflection prompt augmentation

**Classical technique**: Clustering of change descriptions, per-cluster statistics, possibly simple regression (change_features -> delta_score)

**Why it works**: Extracts a policy gradient signal — which *types* of edits move the score in which direction. The proposer gets a prior over what to try, rather than starting from scratch each time.

**Variants**:
- **Conditional attribution**: "Adding examples helps for math problems but hurts for creative writing" — condition on input features
- **Interaction effects**: "CoT + examples together work, but CoT alone doesn't" — look at feature combinations
- **Recency weighting**: Recent changes are more relevant than early ones (the candidate has evolved, so what helped at score 0.3 may not help at score 0.8)

### 3. Candidate Feature -> Score Surrogate Model

**The problem**: The LLM proposer has no model of what makes a good candidate. It proposes based on local reflection, without a global picture of the search landscape.

**The pipeline**:
```
LLM.extract(candidate) -> feature_vector
  {has_cot: true, num_examples: 3, specificity: "high",
   uses_formatting: true, length: 450, ...}
     |
     v
accumulate (feature_vector, score) across all candidates
     |
     v
fit surrogate model: features -> score
  (random forest, GP, or even linear regression)
     |
     v
two uses:
  a) Feature importance -> "The 3 features most predictive of score are..."
  b) Acquisition function -> "The most promising unexplored region is:
     high specificity + few-shot examples + short length"
     |
     v
inject into reflection prompt:
  "Based on analysis of all 47 candidates explored so far:
   - Candidates with few-shot examples score 0.2 higher on average
   - Length beyond 400 tokens shows diminishing returns
   - Formal tone and high specificity interact positively
   Consider these patterns when proposing the next candidate."
```

**Injection point**: Reflection prompt augmentation + candidate seeding

**Classical technique**: Random forest / GP regression, SHAP values for feature importance, Bayesian optimization acquisition functions (EI, UCB)

**Why it works**: Gives the proposer a global view of the search landscape derived from all candidates, not just the current parent.

**Caveats**: LLM-extracted features may not be smooth enough for GP surrogates. The feature space may be high-dimensional and sparse. Start with simple models (linear, RF) before going to GPs.

### 4. Input-Aware Active Sampling (Curriculum Learning)

**The problem**: Default minibatch sampling is random or epoch-shuffled. Some inputs are easy (already solved by most candidates), some are hard (unsolved by any). Random sampling wastes evaluations on already-solved inputs.

**The pipeline**:
```
LLM.classify(input) -> {topic, difficulty_features}
  or just: use score history per input as a difficulty signal
     |
     v
track: score_matrix[candidate_idx, input_idx]
       input_features[input_idx]
     |
     v
identify the hard frontier:
  - inputs where best_candidate_score < threshold
  - inputs where variance across candidates is highest (uncertain)
  - inputs in underexplored topic clusters
     |
     v
bias minibatch sampling toward the hard frontier
```

**Injection point**: Minibatch sampler (replace EpochShuffledBatchSampler)

**Classical technique**: Active learning acquisition functions, multi-armed bandits over input clusters, uncertainty sampling

**Why it works**: Focuses evaluation budget on the inputs that matter. No point re-evaluating inputs where every candidate scores 1.0.

**Variants**:
- **Pure hard-case focus**: Only sample inputs where max score across all candidates < 0.8
- **Uncertainty sampling**: Sample inputs where candidate scores have highest variance (the search hasn't converged on what works for these)
- **Topic rotation**: Cluster inputs by topic, round-robin across topic clusters to ensure coverage
- **Hybrid**: Mix hard-frontier sampling with random for exploration

**Note**: This doesn't require LLM feature extraction if you only use score history. The LLM adds value when you want to understand *why* certain inputs are hard (topic classification, difficulty decomposition).

### 5. Embedding-Space Search Structure

**The problem**: Candidates are text, and the search operates on text directly. But candidates that score similarly may cluster in embedding space, and high-scoring regions may have geometric structure.

**The pipeline**:
```
embed all candidates -> R^d (via LLM embedding API)
embed all inputs -> R^d
     |
     v
dimensionality reduction (PCA, UMAP, t-SNE)
     |
     v
discover structure:
  - high-scoring candidates cluster in region C
  - the "improvement direction" from seed to best is along axis A
  - input difficulty varies along axis B
     |
     v
guide search:
  - "Generate a candidate that would embed near point P"
    (described via nearest-neighbor candidates in the embedding space)
  - "The best candidates share similarity with [X, Y, Z] -
    generate something in their intersection"
```

**Injection point**: Candidate seeding, reflection prompt augmentation

**Classical technique**: PCA, UMAP, k-NN, kernel methods, manifold optimization

**Why it works**: May reveal structure that's invisible at the text level. "These 5 top candidates are all near each other in embedding space" is a signal.

**Caveats**: Embedding spaces from LLMs may not have the right inductive bias for optimization. This is the most speculative instantiation. Worth trying, but not the first thing to implement.

### 6. Cross-Candidate Failure Correlation

**The problem**: Different candidates fail on different inputs. Understanding which failures co-occur reveals the structure of the problem space.

**The pipeline**:
```
build binary matrix: M[candidate_idx, input_idx] = 1 if score > threshold
     |
     v
factor analysis / NMF on M:
  discover latent "skills" — groups of inputs that tend to be
  solved together by the same candidates
     |
     v
identify skill gaps: "Candidate 7 has skill A and B but lacks skill C"
     |
     v
LLM describes the skills in natural language by examining
the inputs in each skill cluster
     |
     v
inject: "The current candidate solves problems requiring [skill A]
         and [skill B] but consistently fails on problems requiring
         [skill C]. Skill C appears to involve [description].
         Here are 5 examples requiring skill C: ..."
```

**Injection point**: Reflection prompt augmentation + minibatch selection

**Classical technique**: Matrix factorization (NMF, SVD), factor analysis, biclustering

**Why it works**: Decomposes the problem into latent skills. Instead of "improve the prompt", the signal becomes "acquire skill C." This is more actionable for the reflector.

**Variants**:
- **Skill transfer detection**: "Candidates that have skill A tend to also have skill B — they transfer. But skill C is independent." This tells the reflector which improvements are likely to come as packages.
- **Skill prerequisite ordering**: "Skill B requires skill A" — curriculum over skills

### 7. Temporal Pattern Detection (Search Dynamics)

**The problem**: The search may be stuck in a plateau, oscillating, or still making progress. The current system doesn't detect this.

**The pipeline**:
```
track: score trajectory, feature evolution, failure mode distribution
  across iterations
     |
     v
detect patterns:
  - plateau: score hasn't improved in K iterations
  - oscillation: alternating between two strategies
  - regression: a previously-solved failure mode reappeared
  - convergence: candidates becoming more similar over time
     |
     v
adapt strategy:
  - plateau -> increase exploration (mutate more aggressively,
    try different component, widen minibatch)
  - oscillation -> detect the two attractors, try to merge them
  - regression -> inject constraint: "do not lose capability X"
  - convergence -> inject diversity: seed from a different region
```

**Injection point**: Meta-level control (adjust exploration/exploitation, module selection, merge triggering)

**Classical technique**: Time series analysis, change point detection, autocorrelation

**Why it works**: Makes the search process self-aware. Instead of blindly iterating, it can diagnose its own dynamics and adapt.

### 8. Module Selection via Feature Attribution

**The problem**: Current module selector is round-robin or random. But at any given point, some components of the candidate matter more than others.

**The pipeline**:
```
LLM.attribute(candidate, input, output) ->
  {component_name: responsibility_score}
  "The persona section contributed 20% to the score,
   the instruction section 60%, the format section 20%"
     |
     v
accumulate attributions across evaluations
     |
     v
identify: which component has the most room for improvement?
  - component with highest variance in responsibility
  - component most correlated with failures
  - component least recently improved successfully
     |
     v
bias module_selector toward that component
```

**Injection point**: Module selector (replace round_robin)

**Classical technique**: Attribution analysis, variance decomposition, multi-armed bandits

**Why it works**: Don't waste iterations mutating a component that's already good. Focus on the bottleneck.

### 9. Merge Candidate Selection via Complementarity Analysis

**The problem**: Current merge proposer looks at score overlap on the validation set to find complementary candidates. But this is a coarse signal — two candidates may score similarly overall but excel for different *reasons*.

**The pipeline**:
```
for each candidate on the Pareto front:
  LLM.extract(candidate) -> strategy_description, feature_vector
     |
     v
cluster Pareto candidates by strategy
identify complementary pairs: different strategy, different success inputs
     |
     v
when triggering merge, select the pair with maximum complementarity
in feature space (not just score space)
     |
     v
LLM merge prompt includes:
  "Candidate A uses strategy X and excels at [inputs].
   Candidate B uses strategy Y and excels at [inputs].
   Combine their strengths."
```

**Injection point**: Merge candidate selection + merge prompt

**Classical technique**: Diversity metrics in feature space, set cover (which pair covers the most uncovered inputs)

### 10. Constraint Learning from Failure Patterns

**The problem**: Some constraints are implicit — the evaluator penalizes certain behaviors, but the reflection prompt doesn't state these constraints explicitly.

**The pipeline**:
```
LLM.classify(low_scoring_outputs) -> violation_type
accumulate violations across iterations
     |
     v
frequent pattern mining: which violations always lead to low scores?
     |
     v
infer constraints:
  "Outputs that exceed 500 tokens always score < 0.3"
  "Outputs that don't show intermediate steps always score < 0.5"
  "Outputs that use informal language score 0.2 lower on average"
     |
     v
inject as hard/soft constraints in the reflection prompt:
  "Learned constraints (from N evaluations):
   - MUST show intermediate steps (violation -> 0.4 avg score penalty)
   - SHOULD keep output under 500 tokens
   - AVOID informal language"
```

**Injection point**: Reflection prompt augmentation (as a "learned constraints" section)

**Classical technique**: Frequent pattern mining, association rules, decision tree on (features -> low_score)

**Why it works**: Makes implicit evaluator preferences explicit. The proposer doesn't have to rediscover these constraints each iteration.

### 11. Population Diversity Maintenance

**The problem**: Evolutionary search can lose diversity — all candidates converge to the same local optimum. The Pareto frontier helps, but only along the score dimension.

**The pipeline**:
```
embed or featurize all candidates on the Pareto front
     |
     v
measure diversity: average pairwise distance in feature space
detect: diversity is dropping across iterations
     |
     v
when diversity is low:
  - inject candidates from underrepresented regions
  - bias the proposer toward novel strategies:
    "All current candidates use CoT. Try a fundamentally
     different approach."
  - increase mutation magnitude
```

**Injection point**: Candidate seeding, reflection prompt augmentation, proposer temperature

**Classical technique**: Diversity metrics (pairwise distance, entropy), novelty search

### 12. Evaluation Efficiency via Score Prediction

**The problem**: Full evaluation is expensive. Some candidates are obviously bad and don't need full evaluation.

**The pipeline**:
```
LLM.extract(candidate) -> feature_vector
     |
     v
surrogate model predicts: estimated_score
     |
     v
if estimated_score < threshold (e.g., below current Pareto front minimum):
  skip full evaluation, save the budget
else:
  proceed with full evaluation
     |
     v
update surrogate with actual score
```

**Injection point**: Evaluation policy (before calling the evaluator)

**Classical technique**: Surrogate-assisted evolutionary optimization, early stopping with predicted bounds

**Why it works**: Saves evaluation budget for promising candidates. Particularly valuable when evaluation is expensive.

**Risk**: False negatives — the surrogate may incorrectly reject a good candidate. Use conservative thresholds.

---

## Implementation Considerations

### Cost-Benefit

Each LLM extraction call has a cost (tokens, latency). The framework pays off when:
- Evaluators are expensive (code execution, API calls, agent runs) — saving even 1 evaluation justifies many classification calls
- The dataset has clear structure (distinct failure modes, learnable patterns) — unstructured noise won't benefit
- The search runs for many iterations (>50) — need enough data for classical techniques to find patterns

For cheap evaluators with few iterations, the overhead may not be worth it. Consider:
- Using a small/fast model for classification (haiku-class)
- Batching classifications
- Only triggering analysis every K iterations rather than every evaluation
- Lazy extraction: only classify failures, not successes

### Feature Schema Discovery

The hardest design problem: what features to extract. Options:
1. **Hardcoded task-specific schema**: Works best, requires domain knowledge per task
2. **LLM-generated schema**: Ask an LLM to propose a feature taxonomy given the objective and a few examples, then use it consistently
3. **Emergent schema**: Start with free-text descriptions, cluster them to discover natural categories, then refine into a fixed taxonomy
4. **Hierarchical**: Coarse categories first (mistake type), then fine-grained as data accumulates

Option 3 (emergent) is the most general but hardest to get right. Option 2 (LLM-generated) is probably the practical sweet spot.

### Where to Start

Ranked by (likely impact x implementation ease):

1. **Failure Mode Concentration** (#1) — highest leverage, simplest to implement. Just add a classifier after each eval and a biased sampler.
2. **Input-Aware Active Sampling** (#4) — doesn't even require LLM extraction if using score history alone. Swap the batch sampler.
3. **Change Attribution** (#2) — moderate complexity. Requires analyzing the lineage tree post-hoc.
4. **Constraint Learning** (#10) — moderate complexity but high value for tasks with implicit constraints.
5. **Module Selection via Attribution** (#8) — moderate. Requires per-component attribution.
6. Everything else — research bets. Worth exploring but not guaranteed to pay off.

### Integration with GEPA

The natural integration points in the current architecture:
- **BatchSampler**: Replace `EpochShuffledBatchSampler` with a feature-aware sampler (instantiations 1, 4)
- **ReflectiveMutationProposer.propose_new_texts()**: Augment the reflection prompt with accumulated analysis (instantiations 2, 3, 6, 10)
- **ReflectionComponentSelector**: Replace `round_robin` / `all` with attribution-based selection (instantiation 8)
- **MergeProposer**: Use feature-based complementarity for merge pair selection (instantiation 9)
- **GEPAEngine main loop**: Add meta-level pattern detection and strategy adaptation (instantiation 7)
- **EvaluationPolicy**: Add surrogate-based early rejection (instantiation 12)
- **New component: FeatureStore**: A structured store that accumulates features across iterations, queryable by classical ML techniques. This is the shared substrate that all instantiations write to and read from.
