# Research Ideas: Improving Generalization in LLM-Guided Text Evolution

## Core Thesis

GEPA's fundamental tension: **specificity** is needed for reflection (the LLM must diagnose concrete failures), but **generality** is needed for the output (the evolved artifact must work on unseen examples). Current GEPA addresses this partially by gating acceptance on a held-out val set. These ideas attack the remaining gaps — in the mutation operator, the selection mechanism, the training distribution, and the evaluation protocol.

## The Ideas

### Mutation Quality
1. **[Meta-Reflection](01_meta_reflection.md)** — Learn the mutation operator from optimization history. Periodically analyze which types of mutations were accepted vs. rejected, distill principles, and inject them into future reflection prompts. The mutation operator becomes adaptive rather than fixed.

2. **[Stratified Reflection](02_stratified_reflection.md)** — Force abstraction via a multi-stage reflection pipeline (diagnose -> abstract -> propose). The abstraction stage acts as an information bottleneck, filtering out example-specific noise before the proposal stage.

### Training Signal
3. **[Adversarial Curriculum](03_adversarial_curriculum.md)** — Co-evolve the training set alongside candidates. After each accepted candidate, generate adversarial examples that target its weaknesses. This creates a minimax game where the candidate must develop robust strategies.

### Selection Mechanism
4. **[Robust Pareto Selection](04_robust_pareto_selection.md)** — Replace per-example Pareto optimality with per-group optimality. Candidates must demonstrate consistent performance across groups of examples, not just spike on individuals. Grounded in distributionally robust optimization.

5. **[Quality-Diversity Search](05_quality_diversity.md)** — Augment the Pareto front with a MAP-Elites-style archive that preserves diversity in *strategy space* (how candidates solve problems), not just *score space* (which examples they solve). Uses the LLM to extract behavior characterizations.

### Geometric Awareness (Cross-Cutting)
6. **[Embedding-Informed Search](07_embedding_informed_search.md)** — Give GEPA spatial awareness of its own search by embedding candidates, feedback, and examples. Enables retrieval-augmented reflection (show the LLM relevant precedents from earlier iterations), redundancy detection (skip near-duplicate proposals), diversity-aware parent selection, and informed minibatch sampling. Embeddings are 1000x cheaper than LLM generation calls — this adds geometric structure to the search at negligible cost.

## Recommended Focus for a Paper

Two possible paper framings depending on the story you want to tell:

### Framing A: "Improving the Mutation Operator" (tighter, more focused)

**Primary contribution:** Ideas 1 + 2 (meta-reflection + stratified reflection). Conceptually unified as "making the mutation operator adaptive and structured." Theoretical backing: learned optimizers + information bottleneck.

**Supporting contribution:** Idea 4 (robust Pareto selection). Improves the selection side. Clean DRO connection.

**Ablation matrix:** 2x2 of (base vs. improved mutation) x (base vs. robust selection).

### Framing B: "Spatial Awareness for Text Evolution" (broader, more novel)

**Primary contribution:** Idea 6 (embedding-informed search), specifically retrieval-augmented reflection. The core claim: GEPA's search is blind — it has no memory of past failures and no geometric structure. Embeddings add spatial awareness at negligible cost, enabling the optimizer to learn from its own history via retrieval.

**Supporting contributions:** Idea 1 (meta-reflection) as a non-embedding alternative for history utilization (ablation comparison: summarization vs. retrieval). Idea 4 or diversity-aware selection from Idea 6 for the selection improvement.

**What makes this NeurIPS-worthy:** The insight that embedding models (cheap) and generation models (expensive) play complementary roles in text optimization. Generation models mutate; embedding models provide the geometric scaffolding (distance, retrieval, coverage) that makes the mutations efficient. This is a general principle applicable beyond GEPA.

**Ablation matrix:** (base, +retrieval-augmented reflection, +redundancy detection, +diversity selection, +all) across 3+ benchmarks.

### Framing recommendation

Framing B is more novel — nobody has studied the role of embedding geometry in LLM-guided optimization. Framing A is safer — mutation operator improvement is a well-understood research direction. For NeurIPS specifically, novelty matters more. I'd lean toward B with elements of A as ablation comparisons.

**Benchmarks to cover for either framing:** Prompt optimization (AIME, HotpotQA), code optimization (CUDA kernels or SWE-bench tasks), and a creative/structural task (SVG optimization or agent architecture). Three qualitatively different domains demonstrate generality.

## Relationship Between Ideas

```
                    ┌──────────────────────┐
                    │   Mutation Quality    │
                    │                      │
                    │  1. Meta-Reflection   │──── learns what mutations
                    │  2. Stratified Refl.  │──── work, forces abstraction
                    └──────────┬───────────┘
                               │
                    proposes better candidates
                               │
                               v
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│ Training Signal  │   │   GEPA Engine    │   │ Selection Mechanism │
│                  │──>│                  │<──│                     │
│ 3. Adversarial   │   │  evaluate,       │   │ 4. Robust Pareto    │
│    Curriculum    │   │  accept/reject   │   │ 5. Quality-Diversity│
└─────────────────┘   └──────────────────┘   └─────────────────────┘
   harder examples            │                  better parents
                              │
         ┌────────────────────v───────────────────────┐
         │  6. Embedding-Informed Search (cross-cutting) │
         │                                              │
         │  Provides geometric scaffolding to ALL above: │
         │  - Retrieval memory for mutation (1, 2)      │
         │  - Diversity signal for selection (4, 5)     │
         │  - Informed sampling for training (3)        │
         └──────────────────────────────────────────────┘
```

Idea 6 is cross-cutting — it provides infrastructure (embedding-based geometry) that makes the other ideas cheaper or more effective. Retrieval-augmented reflection (6) is a lightweight alternative to full meta-reflection (1). Embedding diversity (6) is a cheap proxy for LLM-based behavior characterization (5). Embedding-based minibatch sampling (6) targets training signal quality alongside adversarial curriculum (3).

Ideas 1-5 are independent interventions on specific components. Idea 6 is a substrate that improves all of them.
