# Meta-Reflection: Learning the Mutation Operator from Optimization History

## Problem

GEPA's mutation operator is *fixed* throughout a run. Every iteration uses the same reflection prompt template and the same LLM to propose candidates. The reflection LM receives the current candidate text and a minibatch of evaluation traces, and proposes an improved version. But it has no knowledge of what kinds of mutations have worked or failed in *previous* iterations of this same run.

This means GEPA repeats mistakes. If a certain type of mutation (e.g., adding verbose chain-of-thought instructions) consistently gets rejected on valset, the reflection LM has no way to know this and will keep trying it. Conversely, if a certain mutation pattern (e.g., adding verification steps) consistently leads to accepted candidates, the LM has no way to exploit this pattern more aggressively.

In standard optimization terms: GEPA uses a fixed step-size, fixed-direction gradient descent. The optimizer never adapts to the loss landscape it's navigating.

## Proposed Intervention

After every K iterations (e.g., K=5 or K=10), pause the main loop and run a **meta-reflection** step:

1. **Collect mutation history.** From `GEPAState.full_program_trace`, extract for each iteration:
   - The parent candidate and its val score
   - The reflective dataset (what the LM saw)
   - The proposed mutation (what the LM changed)
   - Whether the candidate was accepted (improved valset Pareto front) or rejected
   - The delta in val score

2. **Analyze patterns.** Prompt the reflection LM (or a separate meta-LM) with a summary of recent accepted vs. rejected mutations. Ask it to identify:
   - What types of changes consistently improve generalization?
   - What types of changes look good on the minibatch but fail on valset?
   - What failure modes remain unaddressed?
   - What strategies have been tried but don't work for this task?

3. **Distill mutation principles.** The meta-reflection produces a set of natural-language principles (e.g., "Adding explicit verification steps improves accuracy without hurting generalization," or "Avoid adding example-specific rules; prefer general heuristics").

4. **Inject into future reflections.** Prepend these principles to the reflection prompt template for subsequent iterations. The reflection LM now operates with accumulated knowledge about what works.

## Where This Intervenes in the Codebase

The main intervention point is `ReflectiveMutationProposer.propose()` in `src/gepa/proposer/reflective_mutation/reflective_mutation.py`. Currently this method:

1. Selects a candidate (line 154)
2. Samples a minibatch (line 178)
3. Evaluates with traces (line 210)
4. Builds reflective dataset (line 279)
5. Calls `propose_new_texts()` which renders the reflection prompt and calls the LM (line 310)

The meta-reflection step would:
- Run periodically (checked at the top of `propose()`)
- Read from `state.full_program_trace` to collect history
- Generate mutation principles via an LM call
- Modify `self.reflection_prompt_template` by prepending the principles

The reflection prompt template is used in `InstructionProposalSignature.prompt_renderer()` (`src/gepa/strategies/instruction_proposal.py`), where `<curr_param>` and `<side_info>` are filled in. The meta-reflection principles would be injected as a new section before `<curr_param>`, giving the LM context about what has and hasn't worked.

## Design Decisions

**How often to run meta-reflection?** Too often wastes LLM budget on meta-analysis instead of actual proposals. Too rarely and the principles become stale. A reasonable default: every 5-10 iterations, or when the acceptance rate drops below a threshold (e.g., < 20% of recent proposals accepted).

**Cumulative vs. sliding window?** Should meta-reflection analyze the full history or only the last N iterations? A sliding window (e.g., last 15-20 iterations) keeps the analysis focused on recent dynamics and avoids the LM being overwhelmed by a long history. But early successes might contain important signal. A compromise: always include a summary of the first few iterations alongside the recent window.

**Same LM or separate LM?** Using the same reflection LM for meta-reflection has the advantage of being aware of its own reasoning patterns. Using a separate, potentially larger LM could provide higher-quality analysis. Start with the same LM for simplicity.

**Replacing vs. augmenting the prompt?** The principles should *augment* the existing reflection prompt, not replace it. The core prompt template still provides the structure (here's the candidate, here's the evaluation data). The principles add context about what to try and what to avoid.

## Connection to Learned Optimizers

In the ML literature, **learning to optimize** (L2O / L2L) replaces hand-designed optimizers with learned ones. The seminal work by Andrychowicz et al. (2016) trains an LSTM to output update steps, learning from the optimization trajectory itself. More recent work (e.g., VeLO by Metz et al. 2022) trains optimizers across tasks.

Meta-reflection is the natural-language analog. Instead of learning optimizer parameters (step sizes, momentum coefficients) from gradients of the training loss, we learn optimizer *heuristics* (mutation principles) from the optimization history via LLM reasoning. The key structural parallel:

| Numerical L2O | Meta-Reflection |
|---|---|
| LSTM/Transformer reads gradient history | LLM reads mutation accept/reject history |
| Outputs learned update rule (step size, direction) | Outputs learned mutation principles (what to try, what to avoid) |
| Adapts to loss landscape curvature | Adapts to task-specific generalization patterns |
| Trained across tasks via meta-learning | Learned within a single run via in-context learning |

The within-run learning is actually an advantage for deployment: no meta-training phase is needed. The LLM's in-context learning serves as the adaptation mechanism.

## Theoretical Angle: Regret Bounds for Adaptive Mutation

Consider the mutation operator as a bandit problem. At each iteration, the optimizer chooses a "mutation strategy" (implicitly, via the reflection prompt) and observes a reward (accepted or rejected, with some score delta). A fixed mutation operator has linear regret — it cannot adapt to the reward distribution.

Meta-reflection implements a form of **Exp3-style exploration-exploitation**: by analyzing which strategies have worked, it shifts probability mass toward successful strategies. In the bandit framework, this gives sub-linear regret O(sqrt(T log K)) where T is the number of iterations and K is the effective number of mutation strategies.

This isn't a formal proof (the strategy space is continuous and the rewards are non-stationary), but it provides theoretical motivation: adaptive mutation should converge faster than fixed mutation, and the gap grows with run length.

## Expected Outcome

- **Faster convergence**: Fewer wasted iterations on mutation types that don't work for this task.
- **Better generalization**: The meta-analysis explicitly asks "what changes pass valset?" — it directly optimizes for generalization.
- **Reduced LLM cost**: Fewer rejected proposals means fewer wasted evaluation calls.

## Ablation Design

1. **Base GEPA** — no meta-reflection (control)
2. **Meta-reflection every 5 iterations** — principles prepended to reflection prompt
3. **Meta-reflection every 10 iterations** — less frequent adaptation
4. **Meta-reflection with sliding window (last 15)** vs. **full history**
5. **Oracle ablation** — manually write the "correct" mutation principles for a task and inject them from the start. This measures the ceiling: how much could perfect meta-reflection help?

The gap between (1) and (5) tells you the maximum headroom. The gap between (1) and (2/3/4) tells you how much meta-reflection actually captures.
