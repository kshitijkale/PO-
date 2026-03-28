# Quality-Diversity Search for Text Artifact Evolution

## Problem

GEPA's Pareto frontier preserves candidates that are "best at something" — specifically, best on some val example or some objective metric. But "best on val example 7" is a narrow notion of diversity. Two candidates might dominate different val examples but use essentially the same *strategy* (e.g., both emphasize chain-of-thought but one happens to get example 7 right while the other gets example 12 right). The population is diverse in *what it solves* but not in *how it solves things*.

This matters for generalization: if all Pareto-front candidates use the same strategy family, mutations can only refine within that family. When a new problem requires a fundamentally different approach, no candidate on the front provides a useful starting point.

## Background: Quality-Diversity Algorithms

Quality-Diversity (QD) algorithms (Pugh et al., 2016) maintain a population that is simultaneously high-performing *and* behaviorally diverse. The most well-known QD algorithm is **MAP-Elites** (Mouret & Clune, 2015):

1. Define a low-dimensional **behavior characterization** (BC) space — features that describe *how* a solution behaves, not just *how well* it performs.
2. Discretize the BC space into cells (a grid).
3. For each cell, store the highest-performing solution found so far.
4. To generate new candidates, select a random occupied cell and mutate its solution.

The result is an *archive* of diverse, high-quality solutions spanning the behavior space. QD has produced strong results in robotics (diverse locomotion gaits), game design (diverse game levels), and optimization (diverse solutions to multi-modal problems).

## Proposed Intervention: LLM-Characterized Quality-Diversity for GEPA

Apply QD to text artifact evolution, using the LLM itself to define and compute behavior characterizations.

### The Key Challenge: Defining Behavior Space for Text Artifacts

In robotics, behavior characterizations are physical (final position, gait frequency, energy usage). For text artifacts (prompts, code, configs), there's no obvious physical behavior space. But there *is* a strategy space — the set of approaches, heuristics, and patterns a text artifact employs.

**Use the LLM to extract behavior features.** Given a candidate, ask the LLM:

```
Analyze this system component and describe its strategy along the following dimensions.
For each dimension, provide a short label (1-3 words).

Component:
```
{candidate_text}
```

Dimensions:
1. Primary reasoning approach (e.g., "step-by-step", "pattern-matching", "case-analysis")
2. Verification strategy (e.g., "none", "re-check", "alternative-method")
3. Error handling approach (e.g., "none", "explicit-checks", "fallback-strategies")
4. Specificity level (e.g., "general-principles", "domain-heuristics", "example-specific-rules")
```

This produces a behavior characterization like ("step-by-step", "re-check", "explicit-checks", "domain-heuristics"). Each unique combination defines a cell in the behavior space.

### Algorithm

```
Initialize:
  - archive: dict[BehaviorCell, (candidate, score)] = {}
  - Compute BC for seed candidate, add to archive

For each iteration:
  1. Select a random occupied cell from the archive
  2. Select the candidate from that cell
  3. [Standard GEPA reflection] Sample minibatch, evaluate with traces,
     reflect, propose new candidate
  4. Evaluate new candidate on valset -> score
  5. Compute BC for new candidate
  6. If the new candidate's cell is empty OR new candidate's score > existing
     candidate's score in that cell:
     - Add/replace in archive
  7. Parent tracking: the candidate came from cell X and landed in cell Y
     (possibly different — exploring new behavioral niches)
```

### Integration with Existing Pareto Front

QD doesn't replace Pareto selection — it augments it. The archive provides a diverse set of high-quality candidates. Parent selection can draw from:
- The Pareto front (for exploitation — refining what works)
- Under-explored archive cells (for exploration — trying new strategies)

A mixing parameter alpha controls the exploration-exploitation balance:
- With probability alpha, select from the archive (uniform over occupied cells)
- With probability (1-alpha), select from the Pareto front (standard GEPA selection)

alpha could start high (explore diverse strategies) and anneal toward 0 (exploit best strategies) as the optimization progresses.

## Where This Intervenes in the Codebase

### New component: BehaviorCharacterizer
A new class in `src/gepa/strategies/behavior_characterizer.py` that:
- Takes a candidate `dict[str, str]` and an LLM callable
- Returns a behavior characterization (tuple of labels)
- Caches characterizations to avoid redundant LLM calls

### New component: QDArchive
A new class in `src/gepa/strategies/qd_archive.py` that:
- Maintains the behavior-space grid
- Supports `add(candidate, score, behavior_cell)` and `sample()` operations
- Tracks archive coverage statistics (how many cells are filled, diversity metrics)

### Modified: CandidateSelector
A new `QDCandidateSelector` in `src/gepa/strategies/candidate_selector.py`:
```python
class QDCandidateSelector(CandidateSelector):
    def __init__(self, archive, pareto_selector, alpha=0.3, rng=None):
        self.archive = archive
        self.pareto_selector = pareto_selector
        self.alpha = alpha
        self.rng = rng or random.Random(0)

    def select_candidate_idx(self, state):
        if self.rng.random() < self.alpha and self.archive.num_occupied() > 1:
            # Explore: sample from archive
            return self.archive.sample_random_cell(self.rng)
        else:
            # Exploit: standard Pareto selection
            return self.pareto_selector.select_candidate_idx(state)
```

### Modified: GEPAEngine.run()
After a new candidate is accepted and added to the state, compute its behavior characterization and update the archive.

## Design Decisions

**Behavior dimensions: how many, which ones?** Too few dimensions = too coarse (everything in one cell). Too many = too fine (every candidate in its own cell, no competition within cells). 3-4 dimensions with 3-5 values each gives 27-625 cells — enough for diversity without being so fine-grained that cells are never competed over.

The *choice* of dimensions matters more than the number. Good dimensions should capture *strategy-level* differences (how the candidate approaches the task), not surface-level differences (wording, length). The prompt design for the behavior characterizer is critical.

**LLM-based BC vs. embedding-based BC.** An alternative to LLM-extracted features: embed candidate texts and cluster in embedding space. This is cheaper but captures surface similarity, not strategic similarity. Two prompts with different wording but the same strategy would land in different clusters. LLM-based BC is more expensive but captures the right abstraction level.

**Dynamic behavior dimensions.** The behavior dimensions could be fixed a priori or discovered during the run. For a first implementation, use fixed dimensions relevant to the task (e.g., for prompt optimization: reasoning approach, verification, specificity, structure). Future work: use the LLM to propose relevant behavior dimensions based on the diversity of candidates seen so far.

**Archive vs. Pareto front: complementary, not competing.** The archive stores the best candidate per behavior cell. The Pareto front stores candidates that are best per val example. A candidate can be in both. The archive adds the "diversity in strategy" dimension that the Pareto front lacks.

**Computational cost.** Each behavior characterization requires one LLM call. This adds ~1 LLM call per iteration (for the proposed candidate). Since reflection already uses 1+ LLM calls, the overhead is moderate. Cache the BC for candidates that aren't modified.

## What This Brings That Pareto Alone Does Not

Consider a scenario where GEPA discovers two strategies for math problems:
- Strategy A: detailed chain-of-thought (scores 0.7 average)
- Strategy B: problem decomposition into sub-problems (scores 0.65 average)

Under standard Pareto selection, Strategy A dominates on most examples. Strategy B survives on a few examples where decomposition happens to work better. But Strategy B's Pareto slot is fragile — if a slightly better chain-of-thought candidate takes over those examples, Strategy B is evicted entirely.

Under QD, Strategy A and Strategy B occupy different behavior cells. Strategy B is preserved as the best candidate in the "decomposition" cell, regardless of how well chain-of-thought does on specific examples. When a new problem requires decomposition, Strategy B is available as a parent for mutation — even if it's not on the Pareto front.

## Expected Outcome

- **Greater strategic diversity**: The archive maintains candidates using fundamentally different approaches.
- **Better exploration of the solution space**: Mutations from diverse parents explore more of the landscape than mutations from a homogeneous Pareto front.
- **More robust generalization**: When the test distribution differs from the val distribution, having diverse strategies increases the chance that at least one strategy transfers.
- **Richer optimization history**: The archive provides a map of "what strategies have been tried and how well they worked" — useful for meta-reflection (Idea 1) and for post-hoc analysis.

## Ablation Design

1. **Base GEPA with Pareto selection** (control)
2. **QD with LLM-based behavior characterization, alpha=0.3**
3. **QD with embedding-based behavior characterization, alpha=0.3** (cheaper BC, controls for the QD framework vs. the BC quality)
4. **QD with alpha=0.5** (more exploration)
5. **QD with alpha=0.1** (more exploitation)
6. **QD with annealing alpha** (0.5 -> 0.05 over the run)

Measure: val score (primary), archive coverage (how many cells are filled), strategic diversity (number of distinct strategies on the Pareto front, as judged by LLM), and generalization gap (train score - val score).
