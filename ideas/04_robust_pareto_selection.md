# Robust Pareto Selection

## Problem

GEPA tracks a Pareto frontier over validation examples: a candidate survives on the frontier if it achieves the highest score on *any* single validation example. This is tracked in `GEPAState.program_at_pareto_front_valset`, which maps each `val_id` to the set of program indices that are best on it.

The problem: a candidate that scores 1.0 on one val example and 0.0 on the other nine is Pareto-optimal (it dominates on that one example). It gets selected as a parent for future mutations. But this candidate is almost certainly overfitting — it happened to produce the right answer on one example, possibly by accident or via a strategy so narrow it doesn't generalize.

This matters because parent selection drives the entire search. If overfitted candidates occupy Pareto slots, they get selected as parents, and their mutations inherit their narrow strategies. The Pareto front — meant to preserve diversity — becomes a shelter for overfitted specialists.

The `frontier_type="instance"` setting (default in `gepa.optimize()`) is most vulnerable. `frontier_type="hybrid"` (default in `optimize_anything()`) also tracks per-objective frontiers, which helps somewhat but doesn't solve the fundamental per-example overfitting issue.

## Proposed Intervention: Group-Robust Pareto Selection

Instead of tracking Pareto optimality per individual val example, track it per **group** of val examples. A candidate must be the best on a *group* of examples, not just one.

### Mechanism

1. Partition the validation set into G groups of size m (e.g., G=|valset|/3, m=3). The partition can be random or stratified if example metadata is available.
2. For each group, a candidate's score is the *average* (or *minimum*) over the group members.
3. Pareto optimality is defined over group scores, not individual scores.
4. Periodically re-randomize the grouping to avoid overfitting to a specific partition.

### Why Grouping Helps

A candidate that scores 1.0 on one example and 0.0 on the rest will have a group-average near 0.33 (if groups have 3 members) — unlikely to be Pareto-optimal on any group. Only candidates with *consistent* performance across multiple examples can dominate a group.

Using the *minimum* over the group is even more aggressive: the candidate must handle ALL examples in the group well. This is a form of **distributionally robust optimization (DRO)** — optimizing for worst-case performance over subsets of the validation set.

### Formal Connection to DRO

In distributionally robust optimization (Ben-Tal et al., 2009; Duchi et al., 2021), instead of minimizing expected loss, we minimize the worst-case expected loss over an uncertainty set of distributions:

```
min_c  max_{Q in U}  E_Q[loss(c, x)]
```

where U is a set of distributions "close to" the empirical distribution.

Group-robust Pareto selection implements a discrete version: the "uncertainty set" is the set of all size-m subsets of the validation set. By requiring Pareto optimality over groups, we ensure that selected candidates perform well under perturbations of the validation distribution — which is exactly the generalization guarantee we want.

The group size m controls the robustness level:
- m=1: standard per-example Pareto (no robustness)
- m=|valset|: single aggregate score (maximum robustness, but no diversity)
- m=3-5: sweet spot — requires local consistency without collapsing to a single score

## Where This Intervenes in the Codebase

The Pareto front is maintained in `GEPAState` (`src/gepa/core/state.py`). The key data structures:

```python
pareto_front_valset: dict[DataId, float]                     # best score per val example
program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]] # who's best per val example
```

These are updated in `GEPAState.update_state_with_new_program()`. The `frontier_type` parameter controls which frontiers are active.

For group-robust selection, we'd add a new frontier type (e.g., `frontier_type="group_robust"`) with:

```python
# Groups are tuples of DataIds
pareto_front_groups: dict[tuple[DataId, ...], float]
program_at_pareto_front_groups: dict[tuple[DataId, ...], set[ProgramIdx]]
```

The group assignment would be managed by a new component (e.g., `GroupAssigner`) that:
- Assigns val examples to groups at initialization
- Re-randomizes every R iterations (configurable)
- Computes group scores as mean or min of member scores

Parent selection in `ParetoCandidateSelector` (`src/gepa/strategies/candidate_selector.py`) calls `state.get_pareto_front_mapping()`. This returns the frontier dict, which `select_program_candidate_from_pareto_front` uses to pick a parent. The group-robust version would return the group-based frontier instead — no changes needed downstream.

## Alternative: Consistency-Weighted Pareto Selection

Instead of grouping, an alternative that requires no partition:

For each candidate on the Pareto front, compute a **consistency score**: the fraction of val examples where it scores above median (or above some threshold). Weight the candidate's probability of being selected as parent by its consistency score.

```python
class ConsistencyWeightedParetoCandidateSelector(CandidateSelector):
    def select_candidate_idx(self, state: GEPAState) -> int:
        pareto_mapping = state.get_pareto_front_mapping()
        # Collect all programs on the front
        front_programs = set()
        for prog_set in pareto_mapping.values():
            front_programs.update(prog_set)

        # Compute consistency: fraction of val examples above median
        median_scores = {vid: sorted(state.prog_candidate_val_subscores[pid].get(vid, 0)
                                     for pid in range(len(state.program_candidates)))
                         [len(state.program_candidates) // 2]
                         for vid in state.pareto_front_valset}

        consistency = {}
        for pid in front_programs:
            subscores = state.prog_candidate_val_subscores[pid]
            above_median = sum(1 for vid, ms in median_scores.items()
                              if subscores.get(vid, 0) >= ms)
            consistency[pid] = above_median / len(median_scores)

        # Sample proportional to consistency
        programs = list(consistency.keys())
        weights = [consistency[p] for p in programs]
        return self.rng.choices(programs, weights=weights, k=1)[0]
```

This is softer than group-based Pareto: inconsistent candidates still have *some* chance of being selected, but consistent candidates are strongly preferred.

## Design Decisions

**Group size m.** This is the critical hyperparameter.
- m=2: very mild robustness. A candidate just needs to be good on pairs of examples.
- m=3: moderate. Empirically, groups of 3 are large enough to filter out lucky single-example dominance while still allowing meaningful specialization.
- m=5: strong. Candidates must demonstrate broader competence.
- Good default: m = max(2, |valset| // 10), so groups are ~10% of the val set.

**Re-randomization frequency.** If groups are fixed, candidates can overfit to the specific partition. Re-randomizing every 5-10 iterations prevents this. But too-frequent re-randomization makes the frontier noisy (a candidate that was Pareto-optimal under one partition might not be under another).

**Mean vs. min aggregation.** Mean is standard and smooth. Min is more robust (a candidate can't compensate for one bad example with another good one) but also more aggressive — it might eliminate candidates with one outlier failure even if they're generally strong. Start with mean; try min as an ablation.

**Interaction with hybrid frontier.** When `frontier_type="hybrid"`, both per-example and per-objective frontiers are tracked. Group-robust selection would replace the per-example frontier while keeping the per-objective frontier unchanged. This gives robustness on the example dimension while preserving multi-objective diversity.

## Expected Outcome

- **Fewer overfitted candidates on the Pareto front**: Candidates must demonstrate consistent performance to survive.
- **Better parent candidates for mutation**: Since parents are drawn from the front, higher-quality parents lead to higher-quality proposals.
- **Modest computational overhead**: The only extra cost is computing group scores during Pareto updates, which is O(|groups| * m) — negligible compared to evaluation cost.

## Ablation Design

1. **Per-example Pareto** (control, current `frontier_type="instance"`)
2. **Group-robust Pareto, m=3, mean aggregation**
3. **Group-robust Pareto, m=5, mean aggregation**
4. **Group-robust Pareto, m=3, min aggregation**
5. **Consistency-weighted selection** (no grouping, but weight by consistency)
6. **Group-robust with re-randomization every 5 iterations**
7. **Group-robust with fixed groups** (no re-randomization)

Compare (1) vs (2) for the basic effect. Compare (2) vs (4) for mean vs. min. Compare (6) vs (7) for the effect of re-randomization. Look at the *diversity* of the Pareto front (how many distinct candidates survive) as well as the val score — we want both high quality and meaningful diversity.
