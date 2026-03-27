# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter
from typing import Protocol

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState


class BatchSampler(Protocol[DataId, DataInst]):
    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None):
        self.minibatch_size = minibatch_size
        self.shuffled_ids: list[DataId] = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.last_trainset_size = 0
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]):
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size

        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return

        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        base_idx = state.i * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)

        needs_refresh = not self.shuffled_ids or trainset_size != self.last_trainset_size or curr_epoch > self.epoch
        if needs_refresh:
            self.epoch = curr_epoch
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]


class HistoryBasedBatchSampler(BatchSampler[DataId, DataInst]):
    """Samples minibatches weighted by expected failure * learnability,
    computed from the prompt's interaction history with each training example.

    Weight per example Q:
        w_Q = (1 - s_hat_Q) * ell_Q

    where s_hat_Q is the score estimate (decayed toward a prior) and ell_Q is the
    Beta-Bernoulli posterior mean of improvement probability.

    See ideas/08_history_based_sampling.md for the full derivation.
    """

    def __init__(
        self,
        minibatch_size: int = 3,
        decay: float = 0.95,
        prior: float | None = None,
        rng: random.Random | None = None,
    ):
        self.minibatch_size = minibatch_size
        self.decay = decay
        self.fixed_prior = prior
        self.rng = rng or random.Random(0)

    def _build_question_stats(
        self, state: GEPAState
    ) -> dict[DataId, tuple[float, int, int, int]]:
        """Scan completed trace entries and return per-example statistics.

        Returns {data_id: (last_score, delta_t, n_improvements, n_failures)}.
        """
        stats: dict[DataId, tuple[float, int, int, int]] = {}
        current_iter: int = state.i

        # Skip the last entry — it's the current iteration's incomplete dict
        completed_traces = state.full_program_trace[:-1] if state.full_program_trace else []

        for entry in completed_traces:
            ids = entry.get("subsample_ids")
            scores = entry.get("subsample_scores")
            if ids is None or scores is None:
                continue

            iter_num: int = entry["i"]
            new_scores: list[float] | None = entry.get("new_subsample_scores")

            for j, (did, score) in enumerate(zip(ids, scores, strict=False)):
                prev = stats.get(did)
                n_imp = prev[2] if prev else 0
                n_fail = prev[3] if prev else 0

                if new_scores is not None and j < len(new_scores):
                    if new_scores[j] > score:
                        n_imp += 1
                    else:
                        n_fail += 1

                delta_t = current_iter - iter_num
                stats[did] = (score, delta_t, n_imp, n_fail)

        return stats

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if len(all_ids) == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        stats = self._build_question_stats(state)

        # Adaptive prior: mean of recently-observed scores
        if self.fixed_prior is not None:
            mu = self.fixed_prior
        else:
            recent_scores = [s for s, dt, _, _ in stats.values() if dt < 20]
            mu = sum(recent_scores) / len(recent_scores) if recent_scores else 0.5

        # Compute per-example weights
        weights: list[float] = []
        for did in all_ids:
            if did in stats:
                last_score, dt, n_imp, n_fail = stats[did]
                s_hat = last_score * (self.decay**dt) + mu * (1 - self.decay**dt)
                ell = (1 + n_imp) / (2 + n_imp + n_fail)
            else:
                s_hat = mu
                ell = 0.5

            w = (1 - s_hat) * ell
            weights.append(max(w, 1e-6))

        # Sample without replacement, proportional to weights
        k = min(self.minibatch_size, len(all_ids))
        batch: list[DataId] = []
        available = list(range(len(all_ids)))
        for _ in range(k):
            w_available = [weights[i] for i in available]
            chosen_idx = self.rng.choices(available, weights=w_available, k=1)[0]
            batch.append(all_ids[chosen_idx])
            available.remove(chosen_idx)

        return batch
