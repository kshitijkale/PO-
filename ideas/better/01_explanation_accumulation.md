# Explanation Accumulation with Periodic Rewrite

## The Core Problem with GEPA's Loop

Each iteration does diagnosis + mutation as a single step. The reflection LLM sees 3 examples, diagnoses the failure, and immediately proposes a fix. The diagnosis is based on 3 data points — inevitably noisy. And the diagnosis is thrown away after the mutation. Next iteration starts from scratch.

The reflection LLM never builds up understanding. Iteration 1 learns something. Iteration 2 learns something different. Neither informs the other. Over 50 iterations, GEPA generates 50 independent diagnoses from 150 examples — but each diagnosis only uses 3.

## The Fix: Separate Diagnosis from Synthesis

Stop trying to mutate every iteration. Instead, accumulate diagnostic signal and periodically synthesize.

```
For each iteration:
  1. Evaluate candidate on minibatch with traces (same as now)
  2. Ask the reflection LLM: "What went wrong and why?"
     Output: a failure explanation (2-3 sentences). NOT a mutation.
  3. Append explanation to a running knowledge base.

Every K iterations (e.g., K=5):
  4. Hand the LLM the current candidate + ALL accumulated explanations
  5. "Rewrite this candidate to address all these failure modes."
  6. Evaluate the rewrite on val. Accept/reject.
  7. If accepted, clear the explanation buffer (or keep as context).
```

## Why This Is Fundamentally Better

Each individual explanation is noisy (3 examples). But after 5 iterations, you have 15 examples' worth of diagnostic signal accumulated into 5 explanations. The synthesis step sees a rich, multi-faceted picture of what's wrong. It can find the common thread across 5 independent diagnoses — that common thread is the real signal, and the per-diagnosis noise cancels out.

This is the law of large numbers applied to LLM reasoning. One diagnosis from 3 examples: noisy. Five diagnoses from 15 examples: much less noisy. The synthesis LLM does the averaging.

## Where It Intervenes in the Codebase

`ReflectiveMutationProposer.propose()` in `src/gepa/proposer/reflective_mutation/reflective_mutation.py`. Instead of calling `propose_new_texts()` every iteration, call a cheaper `explain_failures()` most iterations, and call `synthesize_candidate()` every K iterations.

The explanation step uses a simpler prompt than the full mutation proposal:

```
You are diagnosing evaluation failures for a system component.

Current component:
```
{candidate_text}
```

Evaluation results:
{side_info}

In 2-3 sentences, explain what went wrong and why. Focus on the root cause,
not the surface symptom. Do NOT propose a fix — just diagnose.
```

The synthesis step sees the full picture:

```
You are rewriting a system component based on accumulated failure analysis.

Current component:
```
{candidate_text}
```

Failure analyses from recent evaluations:
1. {explanation_1}
2. {explanation_2}
3. {explanation_3}
4. {explanation_4}
5. {explanation_5}

Rewrite the component to address these failure modes while preserving what works.
Provide the improved component within ``` blocks.
```

## The Cost Math

Standard GEPA: 1 reflection + 1 proposal LLM call per iteration.
This approach: 1 explanation call (cheaper — shorter output, no candidate generation) per iteration + 1 full synthesis every K iterations.

Net LLM cost is comparable or lower. But the synthesis step is much better informed.

## What Can Go Wrong

**Stale explanations.** If the candidate changes significantly between explanation-gathering iterations (from a merge, or from a previous synthesis), old explanations become stale — they describe failures of a different candidate. Mitigation: tag each explanation with the candidate version (hash). After a candidate change, discard explanations from the old version. You lose accumulated signal but avoid stale diagnoses.

**Explanation quality varies.** Some explanations are insightful ("the candidate fails because it doesn't verify units"), others are vague ("the candidate got the wrong answer"). The synthesis LLM must distinguish signal from noise in the explanation set. Mitigation: include the minibatch scores alongside each explanation so the synthesis LLM can weight explanations by failure severity.

**K is a new hyperparameter.** Too small (K=2): not enough explanations for synthesis, barely better than standard GEPA. Too large (K=10): too long between mutations, slow progress. K=5 is a reasonable default — 15 examples of signal, and the candidate is only re-evaluated on val every 5 iterations.

## Design Decisions

**Should diagnosis iterations evaluate on val?** No. Diagnosis iterations only evaluate on the minibatch (to get traces). Val evaluation only happens at synthesis time. This saves val budget for when you have a real candidate to test.

**What about the acceptance test?** The minibatch acceptance test (`new_sum > old_sum`) only runs at synthesis time. During diagnosis iterations, there's no new candidate to test — you're just gathering information. This means 4 out of every 5 iterations skip the acceptance test entirely, saving evaluation budget.

**Should explanations accumulate indefinitely?** No. Keep a sliding window (last 3-5 synthesis cycles worth of explanations). Old explanations from 30 iterations ago are about a very different candidate and a potentially different failure landscape. But within a window, accumulation is valuable.

## How This Differs from Meta-Reflection (Idea 1 in the Original Folder)

Meta-reflection periodically summarizes mutation history into principles ("adding verification helps"). This is lossy — the summary can't preserve the specific details of individual failures.

Explanation accumulation stores the raw diagnoses and lets the synthesis LLM do its own pattern-finding. No information is lost to summarization. The synthesis step has access to the full detail of each diagnosis.

Meta-reflection modifies the reflection *prompt* (prepends principles). Explanation accumulation changes the *loop structure* (separate diagnosis from synthesis). The loop change is more fundamental.

## Expected Outcome

- **Higher quality mutations**: Each synthesis is informed by 15+ examples of signal, not 3.
- **Fewer wasted iterations**: Diagnosis iterations are cheap (no val eval, shorter LLM calls). Budget is concentrated on synthesis iterations where mutations are well-informed.
- **More stable optimization**: The noise of individual minibatches is averaged out across the explanation buffer. Mutations target real patterns, not random minibatch artifacts.

## Ablation Design

1. **Standard GEPA** — reflect + mutate every iteration (control)
2. **Explanation accumulation, K=3** — synthesize every 3 iterations
3. **Explanation accumulation, K=5** — synthesize every 5 iterations
4. **Explanation accumulation, K=10** — synthesize every 10 iterations
5. **Accumulation without explanation** — just store raw traces, show all at synthesis time (tests whether the explanation step adds value over raw data)
6. **Standard GEPA with 5x minibatch** — increase minibatch to 15 (matches signal quantity but not structure)

Compare (1) vs (3) for the headline. Compare (3) vs (6) to test whether structured accumulation beats brute-force larger minibatches.
