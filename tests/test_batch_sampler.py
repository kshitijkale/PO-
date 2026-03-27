import random
from types import SimpleNamespace

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler, HistoryBasedBatchSampler


def test_epoch_sampler_refreshes_when_loader_expands():
    loader = ListDataLoader(["a", "b", "c", "d"])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    first_batch = sampler.next_minibatch_ids(loader, state)
    assert len(first_batch) == 2
    assert len(sampler.shuffled_ids) == 4
    assert sampler.last_trainset_size == 4

    state.i += 1
    loader.add_items(["e", "f"])

    second_batch = sampler.next_minibatch_ids(loader, state)
    assert len(second_batch) == 2
    assert sampler.last_trainset_size == 6
    assert len(sampler.shuffled_ids) == 6
    assert {4, 5}.issubset(set(sampler.shuffled_ids))


def test_epoch_sampler_errors_when_loader_empty():
    loader = ListDataLoader([])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    with pytest.raises(ValueError):
        sampler.next_minibatch_ids(loader, state)


# ---------------------------------------------------------------------------
# HistoryBasedBatchSampler tests
# ---------------------------------------------------------------------------


def test_history_sampler_empty_trace_returns_valid_batch():
    """First iteration: no history, should return a valid batch of the right size."""
    loader = ListDataLoader(["a", "b", "c", "d", "e"])
    sampler = HistoryBasedBatchSampler(minibatch_size=3, rng=random.Random(42))
    state = SimpleNamespace(i=0, full_program_trace=[{"i": 0}])

    batch = sampler.next_minibatch_ids(loader, state)
    assert len(batch) == 3
    assert len(set(batch)) == 3  # no duplicates
    assert all(bid in range(5) for bid in batch)


def test_history_sampler_empty_trace_uniform_distribution():
    """With no history, all examples should be sampled roughly equally."""
    loader = ListDataLoader(["a", "b", "c", "d"])
    sampler = HistoryBasedBatchSampler(minibatch_size=1, rng=random.Random(0))

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for trial in range(400):
        state = SimpleNamespace(i=trial, full_program_trace=[{"i": trial}])
        batch = sampler.next_minibatch_ids(loader, state)
        counts[batch[0]] += 1

    # Each ID should appear roughly 100 times out of 400; allow wide margin
    for cid, count in counts.items():
        assert count > 50, f"ID {cid} appeared only {count} times, expected ~100"


def test_history_sampler_failures_preferred_over_successes():
    """Examples with low scores should be sampled much more than high-score examples."""
    loader = ListDataLoader(["fail_q", "pass_q"])
    sampler = HistoryBasedBatchSampler(minibatch_size=1, decay=0.95, rng=random.Random(0))

    trace = [
        {"i": 0, "subsample_ids": [0, 1], "subsample_scores": [0.0, 1.0], "new_subsample_scores": [0.0, 1.0]},
        {"i": 1},  # current incomplete entry
    ]
    state = SimpleNamespace(i=1, full_program_trace=trace)

    counts = {0: 0, 1: 0}
    for _ in range(500):
        batch = sampler.next_minibatch_ids(loader, state)
        counts[batch[0]] += 1

    # ID 0 (failure) should dominate
    assert counts[0] > counts[1] * 3, f"Expected failures to dominate: {counts}"


def test_history_sampler_decay_reduces_stale_confidence():
    """A failure observed many iterations ago should have lower weight than a fresh failure."""
    loader = ListDataLoader(["stale_fail", "fresh_fail"])
    # Use fast decay so staleness matters more
    sampler = HistoryBasedBatchSampler(minibatch_size=1, decay=0.8, prior=0.5, rng=random.Random(0))

    trace = [
        # Stale failure: 50 iterations ago
        {"i": 0, "subsample_ids": [0], "subsample_scores": [0.0]},
    ]
    # Add 49 empty entries to create staleness
    for t in range(1, 50):
        trace.append({"i": t})
    # Fresh failure: 1 iteration ago
    trace.append({"i": 50, "subsample_ids": [1], "subsample_scores": [0.0]})
    trace.append({"i": 51})  # current incomplete
    state = SimpleNamespace(i=51, full_program_trace=trace)

    counts = {0: 0, 1: 0}
    for _ in range(500):
        batch = sampler.next_minibatch_ids(loader, state)
        counts[batch[0]] += 1

    # Fresh failure (ID 1) should be preferred because stale failure's score decayed toward prior
    assert counts[1] > counts[0], f"Expected fresh failure to be preferred: {counts}"


def test_history_sampler_learnability_penalizes_dead_ends():
    """Examples that never improve should get lower weight than those that sometimes improve."""
    loader = ListDataLoader(["dead_end", "learnable"])
    sampler = HistoryBasedBatchSampler(minibatch_size=1, decay=0.99, prior=0.0, rng=random.Random(0))

    trace = []
    # Both examples consistently fail (score=0), but ID 1 sometimes improves
    for t in range(10):
        trace.append({
            "i": t,
            "subsample_ids": [0, 1],
            "subsample_scores": [0.0, 0.0],
            # ID 0 never improves; ID 1 improves on even iterations
            "new_subsample_scores": [0.0, 1.0 if t % 5 == 0 else 0.0],
        })
    trace.append({"i": 10})  # current
    state = SimpleNamespace(i=10, full_program_trace=trace)

    counts = {0: 0, 1: 0}
    for _ in range(500):
        batch = sampler.next_minibatch_ids(loader, state)
        counts[batch[0]] += 1

    # ID 1 (learnable) should be preferred over ID 0 (dead end)
    assert counts[1] > counts[0], f"Expected learnable to be preferred: {counts}"


def test_history_sampler_all_solved_no_crash():
    """When all examples are solved, sampling should still work (via weight floor)."""
    loader = ListDataLoader(["a", "b", "c"])
    sampler = HistoryBasedBatchSampler(minibatch_size=2, rng=random.Random(0))

    trace = [
        {"i": 0, "subsample_ids": [0, 1, 2], "subsample_scores": [1.0, 1.0, 1.0]},
        {"i": 1},
    ]
    state = SimpleNamespace(i=1, full_program_trace=trace)

    batch = sampler.next_minibatch_ids(loader, state)
    assert len(batch) == 2
    assert len(set(batch)) == 2


def test_history_sampler_dynamic_dataset():
    """New examples added to the loader should be included in sampling.

    Uses a fixed prior so that unseen examples get moderate weight while
    solved examples get near-zero weight.
    """
    loader = ListDataLoader(["a", "b"])
    sampler = HistoryBasedBatchSampler(minibatch_size=2, prior=0.5, rng=random.Random(0))

    trace = [
        {"i": 0, "subsample_ids": [0, 1], "subsample_scores": [1.0, 1.0]},
        {"i": 1},
    ]
    state = SimpleNamespace(i=1, full_program_trace=trace)

    # Add new examples
    loader.add_items(["c", "d"])
    batch = sampler.next_minibatch_ids(loader, state)
    assert len(batch) == 2

    # New IDs (2, 3) should appear frequently since old ones are solved
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(400):
        batch = sampler.next_minibatch_ids(loader, state)
        for bid in batch:
            counts[bid] += 1

    # New IDs should dominate since old ones are solved and prior=0.5 gives them real weight
    assert counts[2] + counts[3] > counts[0] + counts[1], f"New IDs should be preferred: {counts}"


def test_history_sampler_missing_new_scores():
    """Trace entries without new_subsample_scores should update last_score but not learnability."""
    loader = ListDataLoader(["a", "b"])
    sampler = HistoryBasedBatchSampler(minibatch_size=1, decay=0.99, rng=random.Random(0))

    trace = [
        # Entry with child scores: ID 0 improves
        {"i": 0, "subsample_ids": [0], "subsample_scores": [0.0], "new_subsample_scores": [1.0]},
        # Entry without child scores (no proposal generated): ID 1 observed but no improvement data
        {"i": 1, "subsample_ids": [1], "subsample_scores": [0.0]},
        {"i": 2},  # current
    ]
    state = SimpleNamespace(i=2, full_program_trace=trace)

    # Both have same last_score (0.0) and similar recency
    # ID 0 has 1 improvement, 0 failures -> ell = 2/3 = 0.67
    # ID 1 has 0 improvements, 0 failures -> ell = 1/2 = 0.5 (prior)
    # Both should be sampled, but ID 0 slightly more due to higher learnability
    batch = sampler.next_minibatch_ids(loader, state)
    assert len(batch) == 1
    assert batch[0] in [0, 1]


def test_history_sampler_errors_when_loader_empty():
    loader = ListDataLoader([])
    sampler = HistoryBasedBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0, full_program_trace=[{"i": 0}])

    with pytest.raises(ValueError):
        sampler.next_minibatch_ids(loader, state)
