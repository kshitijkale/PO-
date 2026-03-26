# Stratified Reflection: Forcing Abstraction via a Diagnostic Bottleneck

## Problem

When the reflection LM receives evaluation traces, it tends to jump directly from specific failures to specific fixes. For example:

- Sees: "Example 3: expected answer 42, got 37. The model forgot to add the carry."
- Proposes: "When adding multi-digit numbers, always check for carries in each column."

This fix is fine *for that example*, but the general issue might be "the prompt doesn't encourage systematic verification of intermediate steps." The specific fix (check carries) is a narrow patch; the general principle (verify intermediate steps) would help across many problem types.

The root cause: the reflection LM's input mixes *what went wrong* (specific) with *why it went wrong* (general) with *how to fix it* (specific again). The LM takes the shortest path from observation to fix, skipping the abstraction step.

## Proposed Intervention

Decompose the single reflection call into a multi-stage pipeline with an explicit **abstraction bottleneck**:

### Stage 1: Diagnose (Specific)
Input: current candidate + evaluation traces for the minibatch.
Task: For each example in the minibatch, identify what went wrong and what went right. Produce a per-example diagnostic.
Output: A structured diagnostic — e.g., for each example: `{example_id, score, what_worked, what_failed, failure_type}`.

### Stage 2: Abstract (General)
Input: The per-example diagnostics from Stage 1. Crucially, **not** the raw evaluation traces and **not** the current candidate text.
Task: Identify *classes* of failures across examples. What general principles or strategies would address these failure classes? What successful patterns should be preserved?
Output: A set of abstract principles — e.g., "The candidate lacks systematic verification of intermediate computations" or "The candidate handles simple cases well but fails when the problem requires multi-step reasoning."

### Stage 3: Propose (Targeted)
Input: The current candidate text + the abstract principles from Stage 2.
Task: Modify the candidate to incorporate the general principles while preserving what works.
Output: The new candidate text.

The key is Stage 2: it acts as an **information bottleneck**. The abstraction stage can only see diagnostics, not raw traces, so it cannot propagate example-specific details into the proposal. It is forced to identify patterns.

## Why This Should Work: Information-Theoretic Argument

Consider three random variables:
- X: the specific training examples in the minibatch
- Z: the abstraction (Stage 2 output)
- Y: performance on the validation set (what we care about)

The standard reflection pipeline implicitly computes a mutation that has high mutual information I(mutation; X) — it's heavily influenced by the specific examples seen. But what we want is high I(mutation; Y) — mutations that correlate with val performance.

The abstraction bottleneck forces Z to compress information about X. By the data processing inequality, I(Z; X) <= I(diagnostics; X). If the bottleneck is well-designed, it preserves I(Z; Y) (task-relevant information) while reducing I(Z; X \ Y) (example-specific noise).

This is structurally identical to the Information Bottleneck method (Tishby et al., 2000): find the representation Z that minimizes I(Z; X) subject to preserving I(Z; Y). The difference is that here, the "compression" is performed by an LLM reasoning about its diagnostics, not by an optimization algorithm over a parametric encoder.

We don't have a formal guarantee that the LLM will find the optimal bottleneck, but the structural constraint (Stage 2 cannot see raw traces) mechanically limits how much example-specific information leaks through.

## Where This Intervenes in the Codebase

Currently, the full pipeline lives in `ReflectiveMutationProposer.propose()`:

1. `adapter.evaluate(minibatch, curr_prog, capture_traces=True)` — produces evaluation batch with trajectories
2. `adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)` — formats traces into a reflective dataset
3. `propose_new_texts(curr_prog, reflective_dataset, predictor_names_to_update)` — renders the prompt template and calls the LM

The intervention replaces step 3 with a three-stage pipeline. `propose_new_texts()` currently calls `InstructionProposalSignature.run_with_metadata()` which renders the prompt and calls the LM once. The new version would make three LM calls:

```
# Stage 1: Diagnose
diagnosis_prompt = render_diagnosis_prompt(curr_param, reflective_dataset)
diagnostics = reflection_lm(diagnosis_prompt)

# Stage 2: Abstract (bottleneck — no access to curr_param or raw traces)
abstraction_prompt = render_abstraction_prompt(diagnostics)
principles = reflection_lm(abstraction_prompt)

# Stage 3: Propose
proposal_prompt = render_proposal_prompt(curr_param, principles)
new_candidate = reflection_lm(proposal_prompt)
```

This triples the LLM cost per proposal. Mitigations:
- Use a cheaper/faster model for Stage 1 (diagnosis is easier than abstraction)
- Cache diagnostics if the same candidate is evaluated on the same examples again
- Amortize: run Stage 2 over accumulated diagnostics from multiple iterations, not just the current minibatch

## Prompt Design for Each Stage

### Stage 1 (Diagnose)
```
You are analyzing evaluation results for a system component.

Current component:
```
<curr_param>
```

Evaluation results:
```
<side_info>
```

For each example, provide a structured diagnostic:
1. What specific aspects of the component's behavior succeeded?
2. What specific aspects failed, and what was the immediate cause?
3. Classify the failure type (e.g., "missing verification", "incorrect heuristic", "incomplete coverage", "wrong strategy")

Output ONLY the diagnostics, one per example. Do not propose fixes.
```

### Stage 2 (Abstract)
```
You are a research scientist analyzing patterns in system diagnostics.

The following are per-example diagnostics from evaluating a system component:
```
<diagnostics>
```

Identify:
1. What general classes of failures appear across examples? (Not specific to any one example)
2. What underlying principles or strategies would address each failure class?
3. What successful behaviors should be preserved?

Be abstract — describe principles, not patches. Focus on WHY things fail, not WHAT specifically fails.
```

### Stage 3 (Propose)
```
You are improving a system component based on high-level principles.

Current component:
```
<curr_param>
```

Principles to incorporate (from analysis of evaluation patterns):
<principles>

Modify the component to incorporate these principles while preserving what works.
Provide the improved component within ``` blocks.
```

## Design Decisions

**Should Stage 2 see the current candidate?** No. This is critical. If Stage 2 sees the candidate, it can leak specific information about the candidate's text (which examples it might handle well based on surface features). The abstraction should be about the *task structure*, not the candidate.

**What if the minibatch is too small for meaningful abstraction?** With reflection_minibatch_size=3, there may not be enough examples to find patterns. Two options:
1. Accumulate diagnostics across iterations and run Stage 2 over the accumulated set (e.g., last 9-12 diagnostics from the last 3-4 iterations).
2. Increase the minibatch size for stratified reflection (but this costs more evaluation budget).
Option 1 is more budget-efficient and also helps with temporal consistency — the principles evolve smoothly rather than jumping based on each minibatch.

**How to prevent the LLM from ignoring the structure?** The LLM might still smuggle specific information through Stage 2 by being overly specific in its "abstract principles." Mitigation: in Stage 2, explicitly instruct "your output should be applicable to any problem of this type, not just the examples shown" and "do not reference specific inputs, outputs, or values from the diagnostics."

## Connection to Other Fields

**Curriculum Learning.** Stratified reflection is related to curriculum learning (Bengio et al., 2009) in that it structures the learning signal. But instead of ordering examples by difficulty, it structures the *reasoning process* by forcing abstraction before proposal.

**Abstraction in Program Synthesis.** The DreamCoder system (Ellis et al., 2021) alternates between solving synthesis tasks and *abstracting* reusable library functions from successful solutions. Stage 2 of stratified reflection plays an analogous role: it extracts reusable principles from successful diagnoses.

**Dual Process Theory.** The two-system model from cognitive science (Kahneman, 2011): System 1 (fast, pattern-matching) and System 2 (slow, deliberate reasoning). Standard single-shot reflection is System 1 — the LM pattern-matches from traces to fixes. Stratified reflection forces System 2 — deliberate abstraction before action.

## Expected Outcome

- **Better generalization**: Mutations based on general principles should transfer to unseen examples better than mutations based on specific fixes.
- **More stable optimization**: Reducing example-specific noise in the mutation signal should reduce the variance of proposed candidates.
- **Higher LLM cost per iteration, but fewer iterations needed**: If each proposal is better targeted, fewer proposals should be needed to reach the same performance.

## Ablation Design

1. **Base GEPA** — single-shot reflection (control)
2. **Two-stage** — diagnose + propose (no abstraction bottleneck)
3. **Three-stage** — diagnose + abstract + propose (full stratified reflection)
4. **Three-stage with accumulated diagnostics** — Stage 2 runs over last 3 iterations of diagnostics
5. **Three-stage with cheap Stage 1** — use a smaller model for diagnosis

Compare (2) vs (3) to isolate the effect of the abstraction bottleneck specifically. Compare (1) vs (2) to see if multi-stage reasoning alone helps (without the bottleneck). The key prediction: (3) should show the largest gap on valset performance relative to train performance — i.e., less overfitting.
