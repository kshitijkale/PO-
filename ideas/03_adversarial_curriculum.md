# Adversarial Curriculum Co-evolution

## Problem

GEPA optimizes candidates against a fixed training set. The reflection LM sees the same examples repeatedly (in different minibatch combinations), learns to handle them, and eventually the candidate achieves high scores on the training distribution. But the training set has finite coverage — there are regions of the problem space it doesn't probe. Candidates can develop strategies that happen to work on the training examples without being truly general.

This is the classic problem of distribution mismatch between train and test. In supervised ML, data augmentation partially addresses it. But in GEPA's setting, the "data" is task instances (math problems, code inputs, queries), and meaningful augmentation requires understanding the task structure — something an LLM can do.

## Proposed Intervention

Co-evolve the training set alongside the candidates. After each accepted candidate, use an LLM to generate **adversarial examples** — new task instances that the current best candidate is likely to fail on. Add these to the training pool.

### Algorithm

```
Initialize: trainset T, valset V, seed candidate c_0
For each iteration i:
    1. [Standard GEPA] Select candidate from Pareto front, sample minibatch from T,
       reflect, propose c_new, evaluate on V, accept/reject
    2. [Adversarial step] If c_new was accepted:
       a. Generate K adversarial examples using an LLM:
          - Input: c_new (the accepted candidate), task description, a few examples from T
          - Prompt: "Generate K new task instances that this candidate is likely to fail on.
                     Target the candidate's weaknesses based on its strategy."
       b. Evaluate c_new on the adversarial examples to confirm they're actually hard
       c. Add confirmed-hard examples to T (those where c_new scores below threshold)
```

### Why Not Just Generate Random Examples?

Random generation expands the training set but doesn't target weaknesses. The LLM can read the candidate's strategy and reason about what it would fail on — e.g., "this prompt emphasizes algebraic manipulation but never mentions geometric reasoning, so generate problems requiring coordinate geometry." This directed generation is far more sample-efficient than random expansion.

## Game-Theoretic Framing

This is a two-player game:

- **Player 1 (Protagonist)**: the candidate, trying to maximize score across all task instances
- **Player 2 (Adversary)**: the example generator, trying to find instances where the candidate fails

The equilibrium of this game (if it exists) is a candidate that is robust to the hardest instances the adversary can construct — i.e., a *minimax-optimal* candidate.

Formally, let C be the space of candidates and E be the space of task instances. The candidate wants to maximize, and the adversary wants to minimize:

```
max_c min_e Score(c, e)
```

In practice, we don't solve this exactly. Instead, we alternate between:
- Improving the candidate given the current training set (GEPA's standard loop)
- Expanding the training set with adversarial examples given the current candidate

This is **iterative best response**, which converges to approximate minimax equilibria in many game classes. The convergence is not guaranteed for arbitrary games, but in practice the training set grows monotonically (we only add examples, never remove), so the candidate must handle an increasingly comprehensive set of challenges.

## Where This Intervenes in the Codebase

The main loop is in `GEPAEngine.run()` (`src/gepa/core/engine.py`). After a candidate is accepted (the `_run_full_eval_and_add` call returns, and the candidate is on the Pareto front), we insert the adversarial generation step.

The training set is stored as a `DataLoader` in `ReflectiveMutationProposer.trainset`. For in-memory datasets (the common case), this wraps a list. We'd need to make this list mutable — currently `ensure_loader()` wraps a list in a `ListDataLoader` which we'd extend with an `add()` method.

The adversarial example generator would be a new component, likely living in `src/gepa/strategies/adversarial_generator.py`. It takes:
- The current best candidate (from `state.program_candidates`)
- The task description (from `objective` and `background`)
- A few existing examples (for format reference)
- An LLM callable (could reuse `reflection_lm` or use a separate one)

And returns a list of new task instances.

## Design Decisions

**How many adversarial examples per iteration?** Too many dilutes the training set with potentially noisy examples. Too few doesn't provide enough signal. Start with K=2-3 per accepted candidate. The training set grows at the rate of acceptance (not every iteration), so growth is bounded.

**Quality control.** Not all LLM-generated examples are valid or useful. Filter by:
1. **Validity**: the example must be well-formed (e.g., a math problem must have a correct answer). Verify by evaluating with a known-good solver if available, or by checking basic structural constraints.
2. **Difficulty**: the current best candidate must actually struggle with it (score below a threshold). If it already handles the adversarial example well, the example isn't targeting a real weakness.
3. **Diversity**: adversarial examples shouldn't all target the same weakness. Track which failure types have been generated and promote diversity.

**Should adversarial examples affect valset?** No. The valset should remain fixed to provide a stable generalization benchmark. Adversarial examples only enter the training pool. This maintains the clean separation: train set evolves (getting harder), val set stays fixed (providing an honest signal).

**When to stop generating adversarial examples?** When the candidate consistently scores above threshold on generated examples — i.e., the adversary can no longer find weaknesses. This is a natural stopping criterion that complements max_metric_calls.

**Risk of training set explosion.** If the training set grows unboundedly, minibatch sampling becomes inefficient (most examples are easy for mature candidates). Mitigation: periodically prune the training set by removing examples that all Pareto-front candidates handle perfectly. These examples no longer provide useful signal.

## Prompt Design for the Adversary

```
You are generating challenging test cases for a system that is being optimized.

The system uses the following strategy:
```
{candidate_text}
```

The system is designed to solve tasks like these:
{objective}

Here are some existing task instances for reference (to match the format):
{example_1}
{example_2}

Your goal: generate {K} new task instances that this system is likely to FAIL on.

Think about:
- What assumptions does the system's strategy make that could be violated?
- What edge cases or boundary conditions does the strategy not address?
- What types of inputs would cause the strategy to produce incorrect results?

Each generated instance must:
1. Be a valid task instance (same format as the reference examples)
2. Have a clear correct answer
3. Target a specific weakness in the system's strategy

For each instance, briefly explain which weakness you're targeting (this explanation is for analysis only, not shown to the system).
```

## Relationship to Self-Play

This is conceptually similar to **self-play** in game AI (Silver et al., 2017), where an agent plays against itself to discover weaknesses. The key difference: in self-play, the adversary IS the protagonist (same model, different role). Here, the adversary is an LLM generating examples, and the protagonist is a text artifact being evolved. The adversary doesn't need to be the same system — it just needs to understand the protagonist's strategy well enough to exploit it.

Also related to **Generative Adversarial Networks** (Goodfellow et al., 2014), but the analogy is loose. The more precise connection is to **adversarial training** in robust optimization (Madry et al., 2018), where the training procedure includes a maximization step (find worst-case perturbations) inside the minimization step (optimize parameters).

## Interaction with Other Ideas

**With meta-reflection (Idea 1):** Meta-reflection learns what mutation types work. Adversarial curriculum learning ensures the training signal is always challenging. Together, they create a system that both learns *how* to mutate effectively and always has hard examples to mutate *on*.

**With stratified reflection (Idea 2):** The adversarial examples are specifically designed to challenge the candidate's strategy. When these examples appear in a minibatch, the abstraction stage (Stage 2) should produce richer principles because the failures are more targeted and diverse.

## Ablation Design

1. **Base GEPA** — fixed training set (control)
2. **Random augmentation** — generate new examples without targeting candidate weaknesses
3. **Adversarial augmentation** — generate examples targeting current candidate's weaknesses
4. **Adversarial + pruning** — also prune examples that all Pareto-front candidates solve
5. **Adversarial with quality filter** — only add examples that the candidate actually fails on

Compare (2) vs (3) to isolate the value of *targeted* generation over random expansion. Compare (3) vs (4) to measure the effect of keeping the training set focused.
