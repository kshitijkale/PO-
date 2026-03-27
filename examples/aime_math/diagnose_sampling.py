"""Retroactive analysis: how wasteful is EpochShuffledBatchSampler on the AIME run?

Replays the trace from outputs/aime_math/run_log.json and computes what
HistoryBasedBatchSampler would have assigned as weights to each example.
Reports how often the epoch sampler wasted slots on already-solved examples.

Run:  cd gepa && uv run python examples/aime_math/diagnose_sampling.py
"""

import json
import os

# ---------------------------------------------------------------------------
# Weight computation (standalone, mirrors HistoryBasedBatchSampler logic)
# ---------------------------------------------------------------------------


def compute_weights(
    all_ids: list[int],
    completed_traces: list[dict],
    current_iter: int,
    decay: float = 0.95,
    prior: float | None = None,
) -> dict[int, dict]:
    """Compute per-ID sampling weights from trace history.

    Returns {id: {"s_hat": ..., "ell": ..., "weight": ..., "last_score": ..., "n_imp": ..., "n_fail": ...}}.
    """
    stats: dict[int, tuple[float, int, int, int]] = {}  # (last_score, delta_t, n_imp, n_fail)

    for entry in completed_traces:
        ids = entry.get("subsample_ids")
        scores = entry.get("subsample_scores")
        if ids is None or scores is None:
            continue

        iter_num = entry["i"]
        new_scores = entry.get("new_subsample_scores")

        for j, (did, score) in enumerate(zip(ids, scores)):
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

    # Adaptive prior
    if prior is not None:
        mu = prior
    else:
        recent = [s for s, dt, _, _ in stats.values() if dt < 20]
        mu = sum(recent) / len(recent) if recent else 0.5

    result = {}
    for did in all_ids:
        if did in stats:
            last_score, dt, n_imp, n_fail = stats[did]
            s_hat = last_score * (decay**dt) + mu * (1 - decay**dt)
            ell = (1 + n_imp) / (2 + n_imp + n_fail)
        else:
            last_score, n_imp, n_fail = None, 0, 0
            s_hat = mu
            ell = 0.5

        w = max((1 - s_hat) * ell, 1e-6)
        result[did] = {
            "s_hat": s_hat,
            "ell": ell,
            "weight": w,
            "last_score": last_score,
            "n_imp": n_imp,
            "n_fail": n_fail,
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    trace_path = os.path.join(os.path.dirname(__file__), "../../outputs/aime_math/run_log.json")
    trace_path = os.path.normpath(trace_path)

    if not os.path.exists(trace_path):
        print(f"Trace file not found: {trace_path}")
        return

    with open(trace_path) as f:
        trace = json.load(f)

    # Infer training set size from max ID seen
    all_seen_ids = set()
    for entry in trace:
        ids = entry.get("subsample_ids", [])
        all_seen_ids.update(ids)
    num_train = max(all_seen_ids) + 1 if all_seen_ids else 0
    all_ids = list(range(num_train))

    print(f"Trace: {len(trace)} iterations, {num_train} training examples")
    print(f"{'='*90}")

    total_slots = 0
    wasted_slots = 0
    total_actual_weight = 0.0
    total_top_weight = 0.0

    print(
        f"{'Iter':>4}  {'Batch IDs':>12}  {'Batch scores':>14}  {'Improved?':>9}"
        f"  {'Wasted':>6}  {'Avg w(actual)':>13}  {'Avg w(top-3)':>12}"
    )
    print("-" * 90)

    for idx, entry in enumerate(trace):
        iter_num = entry["i"]
        batch_ids = entry.get("subsample_ids", [])
        batch_scores = entry.get("subsample_scores", [])
        new_scores = entry.get("new_subsample_scores")

        if not batch_ids:
            continue

        # Compute weights using trace entries BEFORE this iteration
        completed = [e for e in trace[:idx]]
        weights_info = compute_weights(all_ids, completed, iter_num)

        # Actual batch weights
        actual_weights = [weights_info[bid]["weight"] for bid in batch_ids]
        avg_actual = sum(actual_weights) / len(actual_weights) if actual_weights else 0

        # Top-k weights (what history-based would prefer)
        sorted_by_weight = sorted(weights_info.items(), key=lambda x: -x[1]["weight"])
        top_k = sorted_by_weight[: len(batch_ids)]
        avg_top = sum(w["weight"] for _, w in top_k) / len(top_k) if top_k else 0

        # Wasted slots: examples the candidate already solves
        iter_wasted = sum(1 for s in batch_scores if s >= 1.0)
        wasted_slots += iter_wasted
        total_slots += len(batch_ids)
        total_actual_weight += avg_actual
        total_top_weight += avg_top

        improved = "YES" if new_scores and sum(new_scores) > sum(batch_scores) else "no"

        print(
            f"{iter_num:>4}  {str(batch_ids):>12}  {str(batch_scores):>14}  {improved:>9}"
            f"  {iter_wasted:>3}/{len(batch_ids)}   {avg_actual:>12.4f}  {avg_top:>12.4f}"
        )

    print("-" * 90)
    print(f"\nSummary:")
    print(f"  Total minibatch slots:   {total_slots}")
    print(f"  Wasted on solved:        {wasted_slots} ({100*wasted_slots/total_slots:.1f}%)")
    print(f"  Avg weight (actual):     {total_actual_weight/len(trace):.4f}")
    print(f"  Avg weight (top-k):      {total_top_weight/len(trace):.4f}")
    print(f"  Weight efficiency:       {total_actual_weight/total_top_weight:.2%}" if total_top_weight > 0 else "")


if __name__ == "__main__":
    main()
