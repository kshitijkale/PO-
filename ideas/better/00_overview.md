# Better Ideas: Attacking GEPA's Noise Problem at Its Root

## Core Thesis

GEPA wastes information. It throws away traces after one use, discards rejected mutations, evaluates wastefully, and samples only once from a noisy proposal distribution. The original idea folder (sampling improvements, selection mechanisms, reflection restructuring) treats symptoms. These ideas fix the root cause: **make GEPA stop throwing things away, and make it spend its budget more wisely.**

## The Five Ideas

### Signal Accumulation
1. **[Explanation Accumulation](01_explanation_accumulation.md)** — Separate diagnosis from synthesis. Accumulate failure explanations across iterations and periodically rewrite the candidate from accumulated understanding. Each individual diagnosis is noisy (3 examples); five accumulated diagnoses from 15 examples are much less noisy. Changes the loop structure itself.

### Variance Reduction
2. **[Mutation Beam Search](02_mutation_beam_search.md)** — Propose K diverse mutations from each reflection, evaluate all, keep the best. Simple variance reduction for the mutation operator. If each mutation has 30% chance of being good, K=3 gives 66% chance that at least one is good. Embarrassingly parallel, near-zero wall-clock overhead.

### Budget Efficiency
3. **[Progressive Validation](03_progressive_validation.md)** — Evaluate val set in chunks, early-stop candidates that can't possibly beat the current best. Saves 30-50% of val budget on bad candidates, funding more iterations. Successive halving applied to GEPA.

### Information Recycling
4. **[Causal Diff Feedback](04_causal_diff_feedback.md)** — Show the reflection LLM what specific text changes were tried and what score effects they caused. Concrete causal evidence from the optimization's own history, injected directly into the reflection prompt. Zero extra cost — pure information recycling.

5. **[Evaluation Recycling](05_evaluation_recycling.md)** — Store evaluation traces (the ASI) across iterations. At reflection time, retrieve failure traces from recent iterations to augment the current minibatch. Get the diagnostic depth of an 8-example minibatch at the evaluation cost of a 3-example minibatch.

## Unifying Principle

These ideas share a single thesis: **GEPA's noise problem is an information utilization problem, not a signal generation problem.** GEPA already generates rich diagnostic information (traces, mutation outcomes, score histories). It just doesn't use most of it. Fix the utilization and the noise drops.

## Recommended Build Order

```
Step 1: Mutation Beam Search (Idea 2)
  → Easiest (~20 lines), biggest immediate impact
  → Reduces mutation noise by ~sqrt(K)

Step 2: Evaluation Recycling (Idea 5)
  → Show reflection LLM 6-8 traces instead of 3, at zero evaluation cost
  → Doubles effective signal per reflection

Step 3: Progressive Validation (Idea 3)
  → Early-stop bad candidates on val
  → Save ~40% of val budget, fund more iterations

Step 4: Causal Diff Feedback (Idea 4)
  → Nearly free (information augmentation)
  → Composes with everything

Step 5: Explanation Accumulation (Idea 1)
  → Most architecturally novel, changes the loop structure
  → The paper-worthy idea, build after quick wins are validated
```

## How These Compose

All five ideas are orthogonal. They intervene at different points in the GEPA loop:

```
┌─────────────────────────────────────────────────────────┐
│                    GEPA Iteration                        │
│                                                         │
│  SELECT parent ──────────────────────────────────────   │
│       │                                                 │
│  SAMPLE minibatch ──────────────────────────────────    │
│       │                                                 │
│  EVALUATE with traces ─── [5] Store traces in DB ──     │
│       │                                                 │
│  REFLECT ─── [4] Inject causal diffs ──                 │
│       │      [5] Retrieve historical traces ──          │
│       │      [1] Accumulate explanation (skip mutation)  │
│       │                                                 │
│  PROPOSE mutation(s) ─── [2] Generate K proposals ──    │
│       │                  [2] Evaluate all, pick best     │
│       │                                                 │
│  ACCEPTANCE TEST ────────────────────────────────────   │
│       │                                                 │
│  VAL EVALUATION ─── [3] Progressive, early-stop ──      │
│       │              [4] Store diff record               │
│       │                                                 │
│  UPDATE state ──────────────────────────────────────    │
│       │                                                 │
│  [1] Every K iterations: SYNTHESIZE from explanations   │
└─────────────────────────────────────────────────────────┘
```

No conflicts. Each idea can be enabled independently or in combination.
