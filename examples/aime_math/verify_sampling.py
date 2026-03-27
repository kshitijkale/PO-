"""Verify HistoryBasedBatchSampler with a small fixed scenario.

Creates a 10-question mock training set with known scores and traces,
then prints the weights and sampling distribution at each simulated iteration.

Run:  cd gepa && uv run python examples/aime_math/verify_sampling.py
"""

import random
from collections import Counter
from types import SimpleNamespace

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler, HistoryBasedBatchSampler

# ---------------------------------------------------------------------------
# Setup: 10 questions with known difficulty profiles
# ---------------------------------------------------------------------------

LABELS = {
    0: "easy_A",
    1: "easy_B",
    2: "easy_C",
    3: "medium_A",
    4: "medium_B",
    5: "hard_A",
    6: "hard_B",
    7: "hard_C",
    8: "impossible",  # never improves
    9: "unseen",  # never evaluated
}

loader = ListDataLoader([f"q{i}" for i in range(10)])

# Simulated trace: 8 iterations of history
# Easy questions (0-2): always score 1.0
# Medium questions (3-4): score 0.0 initially, improve to 1.0
# Hard questions (5-7): score 0.0, sometimes improve
# Impossible (8): score 0.0, never improves despite many attempts
# Unseen (9): never appears in any minibatch
TRACE = [
    # iter 0: batch [0, 3, 5] — easy pass, medium fail, hard fail
    {
        "i": 0,
        "subsample_ids": [0, 3, 5],
        "subsample_scores": [1.0, 0.0, 0.0],
        "new_subsample_scores": [1.0, 1.0, 0.0],  # medium improved, hard didn't
    },
    # iter 1: batch [1, 4, 6] — easy pass, medium fail, hard fail
    {
        "i": 1,
        "subsample_ids": [1, 4, 6],
        "subsample_scores": [1.0, 0.0, 0.0],
        "new_subsample_scores": [1.0, 0.0, 1.0],  # hard improved this time
    },
    # iter 2: batch [2, 8, 7] — easy pass, impossible fail, hard fail
    {
        "i": 2,
        "subsample_ids": [2, 8, 7],
        "subsample_scores": [1.0, 0.0, 0.0],
        "new_subsample_scores": [1.0, 0.0, 0.0],  # neither improved
    },
    # iter 3: batch [0, 8, 5] — easy pass, impossible fail again, hard fail
    {
        "i": 3,
        "subsample_ids": [0, 8, 5],
        "subsample_scores": [1.0, 0.0, 0.0],
        "new_subsample_scores": [1.0, 0.0, 0.0],
    },
    # iter 4: batch [1, 8, 6]
    {
        "i": 4,
        "subsample_ids": [1, 8, 6],
        "subsample_scores": [1.0, 0.0, 0.0],
        "new_subsample_scores": [1.0, 0.0, 0.0],
    },
    # iter 5: batch [2, 8, 7] — impossible fails yet again
    {
        "i": 5,
        "subsample_ids": [2, 8, 7],
        "subsample_scores": [1.0, 0.0, 0.0],
        "new_subsample_scores": [1.0, 0.0, 1.0],  # hard_C finally improved
    },
    # iter 6: batch [3, 8, 5] — medium now passes, impossible fails
    {
        "i": 6,
        "subsample_ids": [3, 8, 5],
        "subsample_scores": [1.0, 0.0, 0.0],
        "new_subsample_scores": [1.0, 0.0, 1.0],  # hard_A improved
    },
    # iter 7: batch [4, 8, 6]
    {
        "i": 7,
        "subsample_ids": [4, 8, 6],
        "subsample_scores": [0.0, 0.0, 0.0],
        "new_subsample_scores": [1.0, 0.0, 0.0],  # medium_B improved
    },
]


def print_weights(sampler, state, title):
    """Print per-example weights and sampling distribution."""
    stats = sampler._build_question_stats(state)

    # Compute prior
    recent = [s for s, dt, _, _ in stats.values() if dt < 20]
    mu = sum(recent) / len(recent) if recent else 0.5

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"  Adaptive prior (mu): {mu:.3f}")
    print(f"{'='*80}")
    print(f"{'ID':>3}  {'Label':>12}  {'last_sc':>7}  {'dt':>3}  {'n_imp':>5}  {'n_fail':>6}  {'s_hat':>6}  {'ell':>6}  {'weight':>8}")
    print("-" * 80)

    all_ids = list(loader.all_ids())
    weights = []
    for did in all_ids:
        label = LABELS[did]
        if did in stats:
            last_score, dt, n_imp, n_fail = stats[did]
            s_hat = last_score * (sampler.decay**dt) + mu * (1 - sampler.decay**dt)
            ell = (1 + n_imp) / (2 + n_imp + n_fail)
        else:
            last_score, dt, n_imp, n_fail = None, None, 0, 0
            s_hat = mu
            ell = 0.5

        w = max((1 - s_hat) * ell, 1e-6)
        weights.append(w)
        ls = f"{last_score:.1f}" if last_score is not None else "  -"
        dt_s = f"{dt:>3}" if dt is not None else "  -"
        print(f"{did:>3}  {label:>12}  {ls:>7}  {dt_s}  {n_imp:>5}  {n_fail:>6}  {s_hat:>6.3f}  {ell:>6.3f}  {w:>8.5f}")

    # Normalize to probabilities
    total = sum(weights)
    print("-" * 80)
    print(f"{'':>3}  {'Sampling %':>12}", end="")
    print(f"{'':>7}  {'':>3}  {'':>5}  {'':>6}  {'':>6}  {'':>6}", end="")
    print()
    for did in all_ids:
        pct = weights[did] / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {did}: {pct:>5.1f}%  {bar}")

    # Sample 3000 times and show empirical distribution
    counts = Counter()
    for _ in range(3000):
        batch = sampler.next_minibatch_ids(loader, state)
        for bid in batch:
            counts[bid] += 1

    print(f"\n  Empirical distribution (3000 draws, batch_size=3):")
    for did in all_ids:
        pct = counts[did] / 3000 * 100
        bar = "#" * int(pct / 2)
        print(f"  {did} ({LABELS[did]:>12}): {counts[did]:>4} ({pct:>5.1f}%)  {bar}")


def main():
    sampler_hb = HistoryBasedBatchSampler(minibatch_size=3, decay=0.95, rng=random.Random(42))
    sampler_ep = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(42))

    # --- Scenario 1: No history (iteration 0) ---
    state = SimpleNamespace(i=0, full_program_trace=[{"i": 0}])
    print_weights(sampler_hb, state, "Scenario 1: No history (iter 0) — should be uniform")

    # --- Scenario 2: After 4 iterations of history ---
    state = SimpleNamespace(i=4, full_program_trace=TRACE[:4] + [{"i": 4}])
    print_weights(sampler_hb, state, "Scenario 2: After 4 iters — failures should dominate, impossible still has learnability")

    # --- Scenario 3: After all 8 iterations ---
    state = SimpleNamespace(i=8, full_program_trace=TRACE[:8] + [{"i": 8}])
    print_weights(sampler_hb, state, "Scenario 3: After 8 iters — impossible's learnability should be crushed")

    # --- Scenario 4: Compare epoch sampler on same state ---
    print(f"\n{'='*80}")
    print(f"  Scenario 4: EpochShuffled comparison (iter 8)")
    print(f"{'='*80}")
    state = SimpleNamespace(i=8)
    counts = Counter()
    for trial in range(1000):
        state.i = trial
        batch = sampler_ep.next_minibatch_ids(loader, state)
        for bid in batch:
            counts[bid] += 1
    print(f"\n  EpochShuffled distribution (1000 iters, batch_size=3):")
    for did in range(10):
        pct = counts[did] / 1000 * 100
        bar = "#" * int(pct / 2)
        print(f"  {did} ({LABELS[did]:>12}): {counts[did]:>4} ({pct:>5.1f}%)  {bar}")
    print("\n  Note: EpochShuffled is ~uniform — it gives solved examples equal weight.")

    # --- Expected behavior summary ---
    print(f"\n{'='*80}")
    print("  EXPECTED BEHAVIOR CHECKLIST")
    print(f"{'='*80}")
    print("  [1] Scenario 1: all weights equal (~10% each)          — no history, uniform")
    print("  [2] Scenario 2: easy_A/B/C near 0%, hard/impossible high — solved examples deprioritized")
    print("  [3] Scenario 3: impossible lower than hard_A/B/C       — learnability penalizes dead ends")
    print("  [4] Scenario 3: unseen (ID 9) has moderate weight      — never-seen gets prior")
    print("  [5] Scenario 4: epoch sampler ~uniform across all      — blind to history")


if __name__ == "__main__":
    main()
