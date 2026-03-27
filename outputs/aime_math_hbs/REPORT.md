# History-Based Active Sampling: Experiment Report

## 1. Executive Summary

We implemented and tested **History-Based Batch Sampling (HBS)**, a new minibatch selection strategy for GEPA that weights training examples by `(1 - expected_score) * learnability` rather than cycling uniformly. The sampler was evaluated on AIME math competition problems using GPT-4.1 Mini as the task LM and GPT-5 as the reflection LM.

**Result**: The sampler mechanism works as designed -- it targets failures (61% vs 39% baseline), eliminates wasted iterations, and triples the acceptance rate (91% vs 31%). However, **the best candidate val score is identical between runs (55.6%)**, and the optimized prompt **regressed on the held-out test set** (44.0% vs 52.7% unoptimized baseline). The sampler successfully solves the problem it was designed for (wasted minibatch slots), but the generalization bottleneck lies elsewhere: in the reflection prompt, which causes memorization of specific problem solutions rather than learning transferable strategies.

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Task LM | GPT-4.1 Mini (temperature=1) |
| Reflection LM | GPT-5 (temperature=1) |
| Dataset (train/val) | AIME 2022-2024, 90 problems split 50/50 |
| Test set | AIME 2025, 30 problems x 5 repeats = 150 |
| Minibatch size | 3 |
| Budget | 500 metric calls |
| Sampler (HBS) | decay=0.95, adaptive prior |
| Sampler (baseline) | Epoch-shuffled (uniform cycling) |
| Metric | Exact integer match with rich feedback (full solutions) |
| Seed instruction | "Solve the problem and provide the answer in the correct format." (63 chars) |

---

## 3. Iteration-by-Iteration Results

### HBS Run (11 iterations, budget exhausted at 531 calls)

| Iter | Minibatch IDs | Parent Scores | Child Scores | Val% | Best? | Accepted |
|------|--------------|---------------|--------------|------|-------|----------|
| 1 | [34, 18, 11] | [0, 1, 1] | [1, 1, 1] | 48.9 | Yes | Yes |
| 2 | [24, 41, 36] | [0, 0, 0] | [1, 1, 1] | 44.4 | | Yes |
| 3 | [17, 40, 9] | [0, 0, 1] | [1, 0, 1] | 48.9 | | Yes |
| 4 | [15, 35, 29] | [0, 0, 1] | [1, 1, 1] | 48.9 | | Yes |
| 5 | [44, 24, 34] | [1, 0, 1] | [1, 1, 1] | 44.4 | | Yes |
| 6 | [15, 35, 40] | [1, 1, 0] | [1, 1, 1] | 33.3 | | Yes |
| 7 | [24, 4, 21] | [1, 0, 0] | [1, 1, 0] | 40.0 | | Yes |
| 8 | [30, 7, 25] | [0, 1, 1] | [1, 1, 1] | 55.6 | Yes | Yes |
| 9 | [40, 12, 36] | [0, 0, 0] | [1, 1, 0] | 53.3 | | Yes |
| 10 | [41, 4, 38] | [0, 1, 1] | [0, 1, 1] | - | | No |
| 11 | [39, 30, 0] | [0, 0, 0] | [1, 1, 1] | 53.3 | | Yes |

### Baseline Run (13 iterations, epoch-shuffled sampler)

| Iter | Minibatch IDs | Parent Scores | Child Scores | Val% | Best? | Accepted |
|------|--------------|---------------|--------------|------|-------|----------|
| 1 | [1, 27, 35] | [0, 1, 1] | - | 55.6 | Yes | Yes |
| 2 | [0, 23, 37] | [0, 0, 1] | - | - | | No |
| 3 | [14, 12, 7] | [0, 1, 1] | - | - | | No |
| 4 | [44, 42, 34] | [1, 1, 1] | - | - | | No |
| 5 | [21, 5, 6] | [0, 0, 0] | - | 42.2 | | Yes |
| 6 | [11, 20, 15] | [1, 1, 1] | - | - | | No |
| 7 | [10, 43, 29] | [0, 0, 1] | - | 46.7 | | Yes |
| 8 | [9, 4, 28] | [1, 1, 1] | - | - | | No |
| 9 | [36, 17, 40] | [0, 0, 1] | - | - | | No |
| 10 | [39, 38, 3] | [0, 1, 1] | - | - | | No |
| 11 | [24, 33, 18] | [0, 1, 1] | - | 53.3 | | Yes |
| 12 | [8, 41, 13] | [0, 1, 1] | - | - | | No |
| 13 | [22, 30, 19] | [1, 1, 1] | - | - | | No |

Note: Baseline iterations 4, 6, 8, and 13 had **zero parent failures** -- the entire minibatch was already solved, giving the reflection LM no useful signal. This never happens in HBS.

---

## 4. Head-to-Head Comparison

| Metric | Baseline (Epoch) | HBS | Delta |
|--------|-----------------|-----|-------|
| Iterations completed | 13 | 11 | -2 |
| **Acceptance rate** | **31% (4/13)** | **91% (10/11)** | **+60pp** |
| **Failure targeting** | **39% (14/36 slots)** | **61% (20/33 slots)** | **+22pp** |
| Wasted iterations (0 parent fails) | 4 | 0 | -4 |
| Unique train examples seen | 39/45 (87%) | 22/45 (49%) | -38pp |
| Pareto front size | 5 | 11 | +6 |
| **Pareto union coverage (val)** | **30/45 (67%)** | **40/45 (89%)** | **+22pp** |
| Seed val score | 46.7% | 46.7% | 0 |
| **Best single-candidate val** | **55.6%** | **55.6%** | **0** |
| Train parent accuracy | 61.1% | 39.4% | -21.7pp |
| Train child accuracy | 66.7% | 87.9% | +21.2pp |
| **Child improvement over parent** | **+5.6pp** | **+48.5pp** | **+42.9pp** |
| Test baseline (unoptimized) | 46.7% | 52.7% | +6.0pp* |
| Test optimized (best candidate) | N/A | 44.0% | - |
| Instruction length (best) | ~8,726 chars | 8,543 chars | ~same |
| Metric calls used | ~531 | 531 | ~same |

*Test baseline difference is due to LLM stochasticity at temperature=1, not a meaningful signal.

---

## 5. What the Sampler Got Right

### 5.1 Failure Targeting Works

HBS directed 61% of minibatch slots to parent failures vs 39% in the baseline. This means the reflection LM consistently received examples where the current prompt failed -- the most informative signal for proposing improvements.

The baseline's epoch-shuffled sampler blindly cycles through all examples. In 4 out of 13 iterations (31%), the entire minibatch was already solved, giving the reflection LM nothing to fix. HBS eliminated this entirely: every iteration contained at least one failure.

### 5.2 Acceptance Rate Tripled

With more failures in each minibatch, the child candidate has room to improve. HBS achieved 91% acceptance (10/11) vs 31% (4/13) baseline. This is a direct consequence of failure targeting -- when the parent scores 0/3 or 1/3, the child needs only modest improvement to beat it.

### 5.3 Larger and More Diverse Pareto Front

HBS produced 11 candidates on the Pareto front (vs 5 baseline). Together, these candidates collectively solve 40/45 val examples (89% coverage) vs 30/45 (67%) for the baseline. The Pareto front represents a richer "portfolio" of prompts with complementary strengths.

### 5.4 Massive Child Improvement Signal

The child-vs-parent improvement on minibatches was +48.5pp for HBS vs +5.6pp baseline. This shows the reflection LM is doing substantial work on each minibatch when given failures to fix.

---

## 6. What Went Wrong

### 6.1 Best Val Score Is Identical

Despite all the sampler improvements, both runs peaked at 55.6% val. The sampler made the optimization loop more efficient (higher acceptance, more candidates) but didn't produce a better final prompt. The bottleneck is not minibatch selection.

### 6.2 Test Set Regression

The optimized prompt scored **44.0% on the test set vs 52.7% unoptimized** -- an 8.7pp regression. The optimization made the prompt worse on unseen problems. This is textbook overfitting.

### 6.3 Instruction Memorization

The seed instruction was 63 characters. The best candidate (Candidate 8) grew to **8,543 characters** containing:
- 36 specific three-digit numbers (e.g., 247, 719, 540, 150)
- 12 domain-specific strategy sections (A through L), each encoding the complete solution to a specific AIME problem
- Exact answers embedded: "m+n = 125", "m+n = 247", "Total = 81", "m+n = 150"

The prompt is not a general math-solving strategy -- it's a lookup table of ~12 memorized solutions. When the test set contains different problems, the lookup table is useless and the bloated instruction actively hurts by consuming context and confusing the task LM.

### 6.4 Coverage vs Targeting Tradeoff

HBS only saw 22/45 training examples (49%) vs 39/45 (87%) for epoch-shuffled. By biasing toward known failures, it kept re-sampling the same ~7 examples (IDs 24, 40, 36, 41, 34, 15, 35 appeared 2-3 times each) while 23 examples were never evaluated. This means the sampler has a narrow view of the training distribution.

### 6.5 Cold Start Problem

With batch_size=3 and 45 training examples, the sampler accumulates history slowly. After iteration 1, only 3 examples have differentiated weights; the other 42 all get identical weight `(1-mu)*0.5`. Even at iteration 11, the unseen pool still commands **52% of total sampling weight**, making the sampler partially random.

---

## 7. Root Cause Analysis

The overfitting is not caused by the sampler. It's caused by the **reflection prompt template** in `src/gepa/strategies/instruction_proposal.py` (lines 13-29), which explicitly instructs the reflection LM to:

> "Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future."

Combined with `metric_with_feedback` providing **full step-by-step solutions** for every problem, this causes the reflection LM to copy specific AIME solutions into the instruction. The child is then evaluated on the **same minibatch** it was trained on, so it always scores near-perfect -- but the memorized details don't transfer.

The `optimize_anything` API uses a different reflection prompt that asks for "failure patterns", "success patterns", and "root causes" -- focused on pattern analysis rather than fact extraction. This is a more generalization-friendly design.

### Why HBS amplifies the memorization

HBS makes the problem slightly worse in one way: by targeting failures, every minibatch contains unsolved problems with high-value solutions. The reflection LM sees these solutions in the feedback and dutifully memorizes them. With epoch-shuffled, some minibatches are all-solved (no new solutions to memorize), which accidentally acts as a weak regularizer.

---

## 8. Val Score Trajectory

```
HBS:      49* 44  49  49  44  33  40  56* 53  rej 53
Baseline: 56* rej rej rej 42      rej 47  rej rej rej 53  rej rej
          (only accepted candidates get val scores in baseline)
```

HBS val scores oscillate wildly (33% to 56%), showing high variance. No monotonic improvement -- the optimization is not converging, it's exploring randomly around the 45-55% range.

---

## 9. Instruction Evolution

| Candidate | Chars | Lines | Val% | Parent | Notes |
|-----------|-------|-------|------|--------|-------|
| 0 (seed) | 63 | 1 | 46.7 | - | One-liner |
| 1 | 4,459 | 67 | 48.9 | 0 | First expansion: adds task/output structure |
| 2 | 6,910 | 95 | 44.4 | 0 | Adds domain strategies |
| 3 | 6,793 | 87 | 48.9 | 0 | Variant |
| 4 | 6,604 | 91 | 48.9 | 0 | Variant |
| 5 | 5,580 | 90 | 44.4 | - | Slightly shorter |
| 6 | 6,649 | 88 | 33.3 | - | Worst performer |
| 7 | 6,864 | 95 | 40.0 | - | Below seed |
| **8** | **8,543** | **124** | **55.6** | - | **Best: 12 memorized strategies** |
| 9 | 10,596 | 144 | 53.3 | - | Longest, but not best |
| 10 | 8,225 | 90 | rej | - | Only rejection |

Instructions plateau at 6,000-8,000 chars after iteration 1. Length and val score are weakly correlated. Candidate 6 (6,649 chars) scored worst at 33.3%, while candidate 8 (8,543 chars) scored best at 55.6%. The specific content matters more than length.

---

## 10. Key Findings

1. **History-based sampling works mechanically**: Failure targeting (61%), zero wasted iterations, 91% acceptance rate. The sampler does exactly what it's designed to do.

2. **The bottleneck is reflection quality, not minibatch selection**: Both runs reach the same 55.6% best val. The sampler makes the loop more efficient but can't fix what the reflection LM produces.

3. **The optimized prompt overfits severely**: Test accuracy drops from 52.7% (unoptimized) to 44.0% (optimized). The "best" prompt is a memorization artifact, not a genuine improvement.

4. **The default reflection prompt template is the root cause**: It explicitly instructs fact memorization. The `optimize_anything` API's reflection prompt does not have this issue.

5. **Coverage matters**: Epoch-shuffled's guarantee of seeing every example (87% coverage in 13 iters) provides broader signal than HBS's 49% coverage. For short runs, this breadth may outweigh the depth of targeted failure analysis.

6. **Pareto diversity is a hidden win for HBS**: 40/45 val problems are solved by at least one candidate (vs 30/45 baseline). If a Pareto-aware ensemble or selection method were used at test time, HBS's richer front would be more valuable.

---

## 11. Recommendations

### Immediate (fix the bottleneck)
- **Replace the reflection prompt template** with an abstraction-forcing variant that asks for general strategies and failure patterns, not domain-specific facts. Model after the `optimize_anything` template.
- **Strip solutions from feedback**: The `metric_with_feedback` function provides full step-by-step solutions. These should be summarized or omitted to prevent the reflection LM from copy-pasting them.

### Sampler improvements
- **Hybrid warm-start**: Use epoch-shuffled for the first N iterations (to build coverage), then switch to HBS once every example has been seen at least once.
- **Increase minibatch size**: batch_size=5 instead of 3 would see 55 slots in 11 iters (covering all 45 examples), eliminating the cold-start problem.
- **Exploration bonus for unseen examples**: Give unseen examples a higher prior weight than `(1-mu)*0.5` to ensure coverage.

### Longer-term
- **Implement abstraction-forcing reflection (Idea 2)**: Stratified reflection that forces the LM to abstract across failures before proposing changes.
- **Contrastive pairs (Idea 9)**: Show the reflection LM paired (success, failure) examples to diagnose surface sensitivity rather than memorize solutions.
- **Instruction length regularization**: Penalize prompt growth in the Pareto objective to discourage memorization.
