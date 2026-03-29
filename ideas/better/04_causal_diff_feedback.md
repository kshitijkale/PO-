# Causal Diff Feedback

## The Problem

The reflection LLM sees *what* the candidate does wrong, but not *what kinds of changes help*. It has no information about the effects of previous mutations. Each reflection starts from zero context about the optimization history.

Meta-reflection (Idea 1 in the original folder) tries to fix this by periodically summarizing mutation history into principles. But summaries are lossy and generic — "adding verification steps helps" doesn't tell you *how much*, *on which examples*, or *at what cost*.

## The Fix: Show the LLM Causal Evidence

After each mutation (accepted or rejected), compute the actual text diff and the actual score changes. Show the last 3-5 diffs to the reflection LLM as concrete causal evidence.

```
After each mutation attempt:
  Store: {
    change_summary: "Added 'verify your answer by substituting back'",
    score_changes: {ex_1: +0.3, ex_5: -0.1, ex_9: +0.2},
    net_effect: +0.4,
    accepted: true,
    val_delta: +0.08  (if evaluated on val)
  }

Before next reflection, inject into prompt:
  "Recent changes to this candidate and their effects:

   1. ACCEPTED (+0.08 val): Added verification step.
      Minibatch: ex_1 +0.3, ex_5 -0.1, ex_9 +0.2

   2. REJECTED: Rewrote intro to be more domain-specific.
      Minibatch: ex_3 +0.4, ex_7 +0.1, ex_12 -0.0
      Failed val gate — specificity didn't generalize.

   3. REJECTED: Removed bullet point formatting.
      Minibatch: ex_2 -0.1, ex_8 +0.0, ex_4 -0.2
      Made things worse even on minibatch."
```

## Why This Is Different from Meta-Reflection

Meta-reflection asks the LLM to analyze history and extract principles ("what types of changes work?"). This is a meta-reasoning step that's itself noisy and lossy.

Causal diff feedback just *shows* the raw history directly. The reflection LLM does its own pattern-finding in context. No separate meta-reflection step, no accumulated principles, no prompt bloat from stale guidance.

The diffs are natural experiments. "I added X, and scores on examples like Y went up while Z went down." This is exactly the information a human optimizer would use. The reflection LLM can learn patterns directly from the evidence.

## What Gets Stored

Each diff record contains:

```python
@dataclass
class MutationDiffRecord:
    iteration: int
    change_summary: str          # 1-sentence description of what changed
    component_changed: str       # which component was mutated
    minibatch_ids: list[DataId]  # which examples were in the reflection minibatch
    parent_scores: list[float]   # parent's scores on minibatch
    child_scores: list[float]    # child's scores on minibatch
    accepted: bool               # did it pass the val gate?
    val_delta: float | None      # change in aggregate val score (if evaluated)
```

## How to Compute the Change Summary

Full Unix-style text diffs are noisy and hard to read. The better approach: ask the LLM to summarize the change in one sentence *when it proposes the mutation*.

Add to the proposal prompt:

```
After writing the improved component, briefly describe in ONE sentence
what you changed and why.
```

This is almost free — a few extra tokens in the proposal output. The summary is stored as the `change_summary` field.

If the proposal LLM doesn't cooperate, fall back to automated diff: compute the character-level edit distance and describe the change structurally ("added 2 sentences at the end", "replaced paragraph 3").

## Where It Intervenes

### Storage

In `GEPAEngine.run()`, after `_run_full_eval_and_add()` or after a minibatch rejection. The data is already available:
- `state.full_program_trace` stores minibatch IDs and scores
- The proposal output contains the change summary
- The val delta is computed in `_run_full_eval_and_add()`

Store diff records as a new field on `GEPAState`:

```python
# In GEPAState.__init__
self.mutation_diffs: list[MutationDiffRecord] = []
```

### Injection

In `InstructionProposalSignature.prompt_renderer()` in `src/gepa/strategies/instruction_proposal.py`. Before the existing `<curr_param>` and `<side_info>` sections, add a new section:

```
Recent changes and their effects:
{formatted_diff_records}

Use this history to inform your proposal. Avoid repeating changes that were
rejected. Build on changes that were accepted.
```

### Integration Point

In `ReflectiveMutationProposer.propose()`, between building the reflective dataset (line 279) and calling `propose_new_texts()` (line 310). Retrieve the last N diff records from `state.mutation_diffs` and pass them to the prompt renderer.

## Cost

Essentially zero incremental cost:
- Storage: a few KB per iteration (text summaries + score vectors)
- Prompt context: ~200-400 tokens for 3-5 diff records
- No extra LLM calls (the change summary is extracted from the existing proposal output)
- No extra evaluations

This is pure information recycling — making the reflection LLM aware of what it already did.

## Design Decisions

**How many diffs to show.** Show the last 3-5. More recent diffs are more relevant (they describe changes to the current candidate). Older diffs may describe changes to a different ancestor. 5 diffs = ~300 tokens of context, well within budget.

**Should rejected mutations be shown?** Yes — they're arguably *more* valuable than accepted ones. A rejected mutation says "this direction doesn't work." The reflection LLM should avoid repeating it. The prompt should explicitly mark each diff as ACCEPTED or REJECTED.

**Should diffs from different candidates be shown?** Only from the current candidate's lineage. If parent selection switches to a different Pareto front candidate, the diff history resets — those diffs describe a different prompt's behavior.

**Interaction with the refiner.** In `optimize_anything`, the refiner_prompt is a co-evolved component. Diff records should include which component was changed (system_prompt vs refiner_prompt) so the reflection LLM knows what it's building on.

## What Can Go Wrong

**The reflection LLM anchors too heavily on the diffs.** Instead of diagnosing the current failure, it just tries the opposite of the last rejected change or repeats the last accepted one. Mitigation: keep the diff section brief and position it before (not after) the main reflective dataset, so the current failure information gets more attention weight.

**Diff summaries are misleading.** If the change summary is inaccurate ("added verification step" when it actually rewrote the entire prompt), the causal attribution is wrong. Mitigation: use the actual text diff as a fallback/supplement. Show both the summary and the key changed lines.

**Candidates may not have been evaluated on the same examples.** Diff record from iteration 10 shows score changes on examples {1, 5, 9}. Diff record from iteration 11 shows changes on {3, 7, 12}. The reflection LLM can't directly compare because the examples differ. This is inherent to minibatch-based evaluation. Mitigation: focus the diff description on the *direction* of change ("improved math examples, hurt reasoning examples") rather than specific score numbers.

## Expected Outcome

- **Fewer repeated mistakes**: The reflection LLM sees what was already tried and rejected, avoiding redundant proposals.
- **Faster convergence**: The LLM builds on what worked (accepted changes) and avoids what didn't.
- **Zero extra cost**: Pure information recycling from data already collected.
- **Composes with everything**: This is an information augmentation — it doesn't conflict with any other optimization change.

## Ablation Design

1. **Standard GEPA** — no diff feedback (control)
2. **Causal diff feedback, last 3 diffs**
3. **Causal diff feedback, last 5 diffs**
4. **Only accepted diffs** (hide rejections — tests whether negative signal adds value)
5. **Only rejected diffs** (hide acceptances — tests whether positive signal adds value)
6. **Raw text diffs instead of summaries** (tests whether LLM summaries are better than mechanical diffs)

Compare (1) vs (2) for the headline. Compare (4) vs (5) to determine whether positive or negative signal is more valuable. Compare (2) vs (6) to test diff representation.

Key metric: proposal redundancy rate (how often does the reflection LLM propose something semantically similar to a recently rejected mutation). This should drop sharply with diff feedback.
