# Contrastive Minimal Pairs Reflection

## Core Insight

Every existing reflection strategy treats training examples as *independent*: sample some, reflect on failures, propose a fix. The reflection LM sees isolated failure traces and must *infer* why the candidate failed. But the most diagnostic signal about overfitting isn't "here's an example that failed." It's **"here's a structurally identical example that succeeded, and here's one that failed — what changed?"**

This is the contrastive signal. GEPA currently throws it away entirely.

When a candidate solves Problem A but fails Problem B, and A ≅ B (same structure, similar difficulty), the failure is *by construction* due to surface features — phrasing, number magnitudes, variable names, problem ordering. The candidate is responding to something that varies between A and B but shouldn't matter. That is the exact definition of overfitting to the training distribution. And the reflection LM can read the differential directly, without having to guess.

Standard reflection on B alone yields narrow patches ("the candidate forgot to carry when adding"). Contrastive reflection on `(A, B)` yields structural diagnoses ("the candidate handles small-integer examples but breaks at scale — it's applying a mental shortcut that doesn't generalize"). The contrast hands the LM the overfitting fingerprint on a silver platter.

## The Mechanism

### Contrastive Minibatch Construction

Instead of sampling individual examples for the reflection minibatch, construct **minimal pairs**: `(A, B)` where:
- The current candidate **solves A** (score ≥ threshold)
- The current candidate **fails B** (score < threshold)
- A and B are **semantically similar** (high cosine similarity in embedding space)

The tighter the pair (higher similarity, larger score gap), the more diagnostic the contrast: the surface differential between A and B is small, so the failure is almost certainly due to surface sensitivity, not conceptual difficulty.

### Pair-Aware Reflection Prompt

The reflection prompt changes from:

> "Here are examples where the candidate failed. Diagnose the failures and propose improvements."

To:

> "Here are contrastive pairs. For each pair, the candidate solved Problem A but failed Problem B. A and B have similar structure. Identify what the candidate did on A that it failed to do on B, and what surface feature of B triggered the failure. Propose a change that removes this surface dependency."

This explicitly directs the LM to reason about *invariance*: what should be the same between A and B that the candidate is treating differently?

### Algorithm

```
Build contrastive minibatch for candidate C, training set T:

1. Score C on all examples in T (or use cached scores from state.full_program_trace)
2. Partition: successes S = {i | score(C, T[i]) >= threshold}
              failures  F = {i | score(C, T[i]) <  threshold}
3. If no embedding index exists, embed T (one-time cost, reuse across iterations)
4. For each failure f in F:
     Find best_match(f) = argmax_{s in S} cosine_similarity(embed(T[f]), embed(T[s]))
     Record pair (best_match(f), f) with similarity sim(f)
5. Sort pairs by sim(f) descending — tightest pairs first
6. Take top K pairs for the minibatch (fill remainder with solo failures if |pairs| < K)
7. Pass pairs to reflection LM with pair-aware prompt
```

Steps 1 and 3 reuse work already done by the engine: `state.full_program_trace` has accumulated scores, and embeddings computed for failure-directed sampling (Idea 7) can be shared directly.

## Why This Attacks Generalization Specifically

### The Overfitting Fingerprint

If A ≅ B but the candidate fails B and not A, the failure is localized to the differential `B \ A`. That differential — whatever surface feature distinguishes B from A in the candidate's representation — is the overfitting artifact. Standard reflection can only approach this differential asymptotically (by seeing many failures and abstracting across them). Contrastive reflection hands the differential to the LM directly in a single call.

### Connection to Stratified Reflection (Idea 2)

Stratified reflection forces abstraction by interposing an information bottleneck between specific traces and the proposal. Contrastive reflection achieves similar abstraction via a different mechanism: the LM *cannot* propose a narrow patch to B because the fix must also preserve the behavior on A. Any change that fixes B at the expense of A is immediately visible as a regression. The pair structure acts as an implicit regularizer on the mutation.

These two ideas compose cleanly: run contrastive pairs through the stratified pipeline. Stage 1 (diagnose) now produces per-pair differentials instead of per-example failures. Stage 2 (abstract) identifies the *class* of surface dependencies the candidate exhibits. Stage 3 (propose) removes them.

### Connection to Failure-Directed Sampling (Ideas 7, 8)

Failure-directed sampling decides *which* failures to show the LM. Contrastive pairs changes *what the LM sees about those failures*. They operate at different levels and compose without conflict: use failure-directed sampling to select which failures matter most, then find contrastive pairs for those failures.

## Theoretical Backing

This is the **contrastive learning** principle applied to LLM reflection. In representation learning (SimCLR, CLIP), point-wise loss on individual examples encourages feature memorization; pair-wise contrastive loss forces the model to learn *invariances* — representations that don't change across semantically equivalent inputs. The generalization gain comes from this invariance pressure.

The same principle applies to text optimization: reflection on isolated failures patches narrow behaviors; reflection on contrastive pairs penalizes surface sensitivity, forcing the proposed mutation to be robust to the features that distinguish A from B.

More formally: let `d(A, B)` be the semantic distance between A and B, and `Δscore(A, B) = score(A) - score(B)` be the score gap. A minimal pair maximizes `Δscore / d` — large score gap per unit of semantic distance. This ratio is a direct measure of the candidate's *sensitivity* to semantic-irrelevant surface variation. Contrastive reflection is a targeted attack on the examples where this ratio is highest.

## Failure Modes and Mitigations

**No tight pairs exist (training set is small or homogeneous)**
If all successes are semantically distant from all failures, no tight pairs can be formed. In this regime, the candidate's failures are genuinely hard (different structure from its successes), not just surface overfitting. Fall back to standard solo-failure reflection — the same behavior as current GEPA.

**All examples are either all-success or all-failure**
Early in the run, the seed candidate may fail most examples (no successes to pair against). Late in the run, a strong candidate may succeed on most. In both cases, fall back to standard reflection. Contrastive pairs are most useful in the middle regime where the candidate is partially generalized — exactly when overfitting is the dominant failure mode.

**Pairs are semantically similar but structurally different**
Embedding similarity doesn't guarantee structural isomorphism. Two problems may have similar vocabulary but different mathematical structures. The reflection LM will still see the differential and can flag "these problems are not as similar as they appear — the structure differs in [way X]." This is itself useful signal: it means the candidate's failure is not surface overfitting but genuine conceptual gap, which should change the direction of the proposed mutation.

## Implementation Notes

- **Embedding reuse**: If Ideas 6 or 7 are implemented, the embedding index already exists. Contrastive pair construction is a new use of an existing artifact — no new infrastructure required.
- **Score reuse**: `state.full_program_trace` already stores per-example scores for every candidate the engine has evaluated. Cached scores can be used in step 1 rather than re-evaluating from scratch.
- **Threshold sensitivity**: The success/failure partition depends on a score threshold. Using a relative threshold (top/bottom quantile of current training scores) is more robust than an absolute value, especially in tasks where raw scores drift as the candidate improves.
- **Pair budget**: K pairs take the same LLM context budget as 2K solo examples. Either reduce K (fewer but more informative examples) or accept slightly larger prompts. In practice, K=4 pairs (8 examples equivalent) is a reasonable starting point.
