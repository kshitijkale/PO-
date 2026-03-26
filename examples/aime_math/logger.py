import json
import time
from difflib import unified_diff
from pathlib import Path
from typing import Any


def _trunc(s: str, n: int = 200) -> str:
    """Truncate string to n chars, adding '...' if trimmed."""
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


def _mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def _text_diff(old: str, new: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    return "".join(unified_diff(old_lines, new_lines, fromfile="parent", tofile="child"))


class ResearchLogger:
    """GEPACallback that writes structured JSONL and prints verbose progress.

    Every callback event is both:
      1. Written as structured JSON to ``{log_dir}/iterations.jsonl`` or ``run.jsonl``
      2. Printed to stdout with full detail

    Load with:
        import pandas as pd
        df = pd.read_json("iterations.jsonl", lines=True)
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._iter_file = open(self.log_dir / "iterations.jsonl", "a")
        self._run_file = open(self.log_dir / "run.jsonl", "a")
        self._cur: dict[str, Any] = {}
        self._t0: float = 0.0
        self._iter_t0: float = 0.0

        # running stats
        self._total_iters: int = 0
        self._accepted_iters: int = 0
        self._best_val: float = 0.0
        self._seed_val: float = 0.0
        self._val_history: list[float] = []

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _write_iter(self, record: dict[str, Any]) -> None:
        self._iter_file.write(json.dumps(record, default=str) + "\n")
        self._iter_file.flush()

    def _write_run(self, record: dict[str, Any]) -> None:
        self._run_file.write(json.dumps(record, default=str) + "\n")
        self._run_file.flush()

    def close(self) -> None:
        self._iter_file.close()
        self._run_file.close()

    # ==================================================================
    # OPTIMIZATION LIFECYCLE
    # ==================================================================

    def on_optimization_start(self, event) -> None:
        self._t0 = time.time()
        self._write_run({
            "type": "optimization_start",
            "timestamp": self._t0,
            "seed_candidate": event["seed_candidate"],
            "trainset_size": event["trainset_size"],
            "valset_size": event["valset_size"],
            "config": event["config"],
        })

        seed_text = list(event["seed_candidate"].values())[0]
        print("\n" + "=" * 80)
        print("GEPA OPTIMIZATION START")
        print("=" * 80)
        print(f"  Train: {event['trainset_size']}  |  Val: {event['valset_size']}")
        print(f"  Components: {list(event['seed_candidate'].keys())}")
        print(f"  Seed prompt ({len(seed_text)} chars):")
        for line in seed_text.splitlines():
            print(f"    {line}")
        print("=" * 80)

    def on_optimization_end(self, event) -> None:
        state = event["final_state"]
        wall_time = time.time() - self._t0
        best_candidate = state.program_candidates[event["best_candidate_idx"]]
        acc_rate = self._accepted_iters / max(self._total_iters, 1)

        self._write_run({
            "type": "optimization_end",
            "timestamp": time.time(),
            "wall_time_seconds": wall_time,
            "best_candidate_idx": event["best_candidate_idx"],
            "best_candidate": best_candidate,
            "total_iterations": event["total_iterations"],
            "total_metric_calls": event["total_metric_calls"],
            "total_candidates": len(state.program_candidates),
            "final_acceptance_rate": acc_rate,
            "seed_val_score": self._seed_val,
            "best_val_score": self._best_val,
            "val_improvement": self._best_val - self._seed_val,
            "val_score_history": self._val_history,
        })

        best_text = list(best_candidate.values())[0]
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"  Wall time:        {wall_time:.1f}s ({wall_time / 60:.1f}m)")
        print(f"  Iterations:       {event['total_iterations']}")
        print(f"  Metric calls:     {event['total_metric_calls']}")
        print(f"  Candidates:       {len(state.program_candidates)}")
        print(f"  Acceptance rate:  {acc_rate:.1%} ({self._accepted_iters}/{self._total_iters})")
        print(f"  Seed val score:   {self._seed_val:.4f}")
        print(f"  Best val score:   {self._best_val:.4f} ({self._best_val - self._seed_val:+.4f})")
        print(f"  Best candidate:   #{event['best_candidate_idx']}")
        print(f"  Val history:      {['%.3f' % v for v in self._val_history]}")
        print(f"  Best prompt ({len(best_text)} chars):")
        for line in best_text.splitlines():
            print(f"    {line}")
        print("=" * 80 + "\n")
        self.close()

    # ==================================================================
    # ITERATION LIFECYCLE
    # ==================================================================

    def on_iteration_start(self, event) -> None:
        self._iter_t0 = time.time()
        self._cur = {
            "iteration": event["iteration"],
            "timestamp": time.time(),
            "wall_time_from_start": time.time() - self._t0,
        }
        print(f"\n--- Iteration {event['iteration']} " + "-" * (80 - 18 - len(str(event['iteration']))))

    def on_iteration_end(self, event) -> None:
        accepted = event["proposal_accepted"]
        iter_time = time.time() - self._iter_t0

        self._total_iters += 1
        if accepted:
            self._accepted_iters += 1
        acc_rate = self._accepted_iters / self._total_iters

        # compute score delta
        eval_data = self._cur.get("evaluation", {})
        parent_scores = eval_data.get("parent_scores", [])
        child_scores = eval_data.get("child_scores", [])
        parent_mean = _mean(parent_scores)
        child_mean = _mean(child_scores)
        score_delta = child_mean - parent_mean if parent_scores and child_scores else None

        self._cur["accepted"] = accepted
        self._cur["iteration_wall_time"] = iter_time
        self._cur["running_acceptance_rate"] = acc_rate
        self._cur["running_best_val"] = self._best_val
        if score_delta is not None:
            self._cur["score_delta"] = score_delta
            self._cur["parent_mean_score"] = parent_mean
            self._cur["child_mean_score"] = child_mean

        self._write_iter(self._cur)

        # summary line
        budget = self._cur.get("budget", {})
        used = budget.get("used", "?")
        remaining = budget.get("remaining", "?")
        total = f"{used}/{used + remaining}" if isinstance(used, int) and isinstance(remaining, int) else "?"
        print(f"\n  {'=' * 70}")
        status = "ACCEPTED" if accepted else "REJECTED"
        print(f"  Iteration {self._cur['iteration']} summary: {status}")
        if score_delta is not None:
            print(f"    Minibatch:  parent={parent_mean:.3f}  child={child_mean:.3f}  delta={score_delta:+.3f}")
        print(f"    Best val:   {self._best_val:.4f}  |  Acc rate: {acc_rate:.0%} ({self._accepted_iters}/{self._total_iters})")
        print(f"    Budget:     {total}  |  Wall: {iter_time:.1f}s  |  Total: {time.time() - self._t0:.1f}s")
        print(f"  {'=' * 70}")
        self._cur = {}

    # ==================================================================
    # CANDIDATE SELECTION & SAMPLING
    # ==================================================================

    def on_candidate_selected(self, event) -> None:
        self._cur["parent"] = {
            "idx": event["candidate_idx"],
            "score": event["score"],
            "candidate": event["candidate"],
        }
        print(f"  [PARENT] Candidate #{event['candidate_idx']}  (val score: {event['score']:.4f})")
        for comp_name, comp_text in event["candidate"].items():
            print(f"    {comp_name} ({len(comp_text)} chars): {_trunc(comp_text, 120)}")

    def on_minibatch_sampled(self, event) -> None:
        self._cur["minibatch"] = {
            "ids": event["minibatch_ids"],
            "trainset_size": event["trainset_size"],
        }
        print(f"  [MINIBATCH] ids={event['minibatch_ids']}  ({len(event['minibatch_ids'])} of {event['trainset_size']} train)")

    # ==================================================================
    # EVALUATION EVENTS
    # ==================================================================

    def on_evaluation_start(self, event) -> None:
        role = "PARENT" if event["candidate_idx"] is not None else "CHILD"
        idx_str = f"#{event['candidate_idx']}" if event["candidate_idx"] is not None else "(new)"
        traces_str = " +traces" if event["capture_traces"] else ""
        print(f"  [EVAL {role} START] candidate={idx_str}  batch_size={event['batch_size']}{traces_str}")

        self._cur.setdefault("evaluation_starts", []).append({
            "candidate_idx": event["candidate_idx"],
            "batch_size": event["batch_size"],
            "capture_traces": event["capture_traces"],
            "is_seed_candidate": event["is_seed_candidate"],
        })

    def on_evaluation_end(self, event) -> None:
        is_parent = event["candidate_idx"] is not None
        role = "PARENT" if is_parent else "CHILD"
        idx_str = f"#{event['candidate_idx']}" if is_parent else "(new)"

        key = "parent_scores" if is_parent else "child_scores"
        self._cur.setdefault("evaluation", {})[key] = event["scores"]

        mean_score = _mean(event["scores"])
        scores_str = ", ".join(f"{s:.1f}" for s in event["scores"])
        print(f"  [EVAL {role} END] candidate={idx_str}  scores=[{scores_str}]  mean={mean_score:.3f}")

        if event["objective_scores"]:
            self._cur.setdefault("evaluation", {})[f"{key}_objectives"] = event["objective_scores"]
            print(f"    Objective scores: {event['objective_scores']}")

    def on_evaluation_skipped(self, event) -> None:
        self._cur["skipped"] = {
            "reason": event["reason"],
            "scores": event["scores"],
        }
        scores_str = ""
        if event["scores"]:
            scores_str = f"  scores=[{', '.join(f'{s:.1f}' for s in event['scores'])}]"
        print(f"  [EVAL SKIPPED] reason={event['reason']}{scores_str}")

    # ==================================================================
    # REFLECTION EVENTS
    # ==================================================================

    def on_reflective_dataset_built(self, event) -> None:
        self._cur["reflection"] = {
            "components": event["components"],
            "dataset": event["dataset"],
        }
        print(f"  [REFLECTIVE DATASET] components={event['components']}")
        for comp_name, examples in event["dataset"].items():
            print(f"    Component '{comp_name}': {len(examples)} examples")
            for i, ex in enumerate(examples):
                inputs_str = _trunc(str(ex.get("Inputs", "")), 100)
                outputs_str = _trunc(str(ex.get("Generated Outputs", "")), 100)
                feedback_str = _trunc(str(ex.get("Feedback", "")), 150)
                print(f"      [{i}] Input:    {inputs_str}")
                print(f"          Output:   {outputs_str}")
                print(f"          Feedback: {feedback_str}")

    def on_proposal_start(self, event) -> None:
        self._cur["proposal_start"] = {
            "components": event["components"],
        }
        print(f"  [PROPOSAL START] Proposing new text for: {event['components']}")

    def on_proposal_end(self, event) -> None:
        r = self._cur.get("reflection", {})
        r["proposed_text"] = event["new_instructions"]
        r["prompt"] = event["prompts"]
        r["raw_lm_output"] = event["raw_lm_outputs"]

        # compute diff
        parent_candidate = self._cur.get("parent", {}).get("candidate", {})
        if parent_candidate and event["new_instructions"]:
            diffs = {}
            for component, new_text in event["new_instructions"].items():
                old_text = parent_candidate.get(component, "")
                diffs[component] = {
                    "old_len": len(old_text),
                    "new_len": len(new_text),
                    "len_delta": len(new_text) - len(old_text),
                    "diff": _text_diff(old_text, new_text),
                }
            r["text_diff"] = diffs
        self._cur["reflection"] = r

        print(f"  [PROPOSAL END]")
        for comp_name, new_text in event["new_instructions"].items():
            old_text = parent_candidate.get(comp_name, "") if parent_candidate else ""
            old_len = len(old_text)
            new_len = len(new_text)
            delta = new_len - old_len
            print(f"    Component '{comp_name}': {old_len} -> {new_len} chars ({delta:+d})")
            print(f"    New text:")
            for line in new_text.splitlines():
                print(f"      {line}")
            if old_text and old_text != new_text:
                diff = _text_diff(old_text, new_text)
                if diff:
                    print(f"    Diff:")
                    for line in diff.splitlines():
                        print(f"      {line}")

        # print prompts/raw outputs if available (not empty when using default proposer)
        if event["prompts"]:
            for comp_name, prompt in event["prompts"].items():
                prompt_str = str(prompt)
                print(f"    Reflection prompt for '{comp_name}' ({len(prompt_str)} chars): {_trunc(prompt_str, 300)}")
        if event["raw_lm_outputs"]:
            for comp_name, raw in event["raw_lm_outputs"].items():
                print(f"    Raw LM output for '{comp_name}' ({len(str(raw))} chars): {_trunc(str(raw), 300)}")

    # ==================================================================
    # ACCEPTANCE / REJECTION
    # ==================================================================

    def on_candidate_accepted(self, event) -> None:
        self._cur["acceptance"] = {
            "new_candidate_idx": event["new_candidate_idx"],
            "subsample_score": event["new_score"],
            "parent_ids": list(event["parent_ids"]),
        }
        print(f"  [ACCEPTED] New candidate #{event['new_candidate_idx']}  "
              f"subsample_score={event['new_score']:.3f}  parents={list(event['parent_ids'])}")

    def on_candidate_rejected(self, event) -> None:
        self._cur["rejection"] = {
            "old_score": event["old_score"],
            "new_score": event["new_score"],
            "reason": event["reason"],
        }
        print(f"  [REJECTED] old={event['old_score']:.3f}  new={event['new_score']:.3f}  reason={event['reason']}")

    # ==================================================================
    # VALSET EVALUATION
    # ==================================================================

    def on_valset_evaluated(self, event) -> None:
        scores_dict = {str(k): v for k, v in event["scores_by_val_id"].items()}
        record = {
            "candidate_idx": event["candidate_idx"],
            "average_score": event["average_score"],
            "is_best": event["is_best_program"],
            "num_evaluated": event["num_examples_evaluated"],
            "total_valset_size": event["total_valset_size"],
            "scores_by_val_id": scores_dict,
        }

        if event["is_best_program"]:
            self._best_val = event["average_score"]
            self._val_history.append(self._best_val)

        n_correct = sum(1 for v in event["scores_by_val_id"].values() if v >= 1.0)
        best_marker = "  NEW BEST" if event["is_best_program"] else ""

        if event["iteration"] == 0:
            self._seed_val = event["average_score"]
            self._best_val = event["average_score"]
            self._val_history.append(self._best_val)
            self._write_run({"type": "seed_valset", "timestamp": time.time(), **record})
            print(f"\n  [SEED VALSET] score={event['average_score']:.4f}  "
                  f"({n_correct}/{event['total_valset_size']} correct)")
            # print per-example breakdown
            for val_id, score in sorted(event["scores_by_val_id"].items(), key=lambda x: str(x[0])):
                mark = "+" if score >= 1.0 else "-"
                print(f"    {mark} val[{val_id}] = {score:.1f}")
        else:
            self._cur["valset"] = record
            print(f"  [VALSET] candidate=#{event['candidate_idx']}  score={event['average_score']:.4f}  "
                  f"({n_correct}/{event['total_valset_size']}){best_marker}")
            # show which examples changed vs previous if there's a parent
            prev_scores = self._cur.get("_prev_val_scores", {})
            changed = []
            for val_id, score in event["scores_by_val_id"].items():
                prev = prev_scores.get(str(val_id))
                if prev is not None and prev != score:
                    changed.append((val_id, prev, score))
            if changed:
                print(f"    Changed examples:")
                for val_id, prev, cur in changed:
                    direction = "+" if cur > prev else "-"
                    print(f"      {direction} val[{val_id}]: {prev:.1f} -> {cur:.1f}")

        # store for next comparison
        self._cur["_prev_val_scores"] = scores_dict

    # ==================================================================
    # PARETO FRONT
    # ==================================================================

    def on_pareto_front_updated(self, event) -> None:
        self._cur["pareto_front"] = {
            "members": event["new_front"],
            "displaced": event["displaced_candidates"],
            "front_size": len(event["new_front"]) if isinstance(event["new_front"], (list, dict)) else None,
        }
        front = event["new_front"]
        displaced = event["displaced_candidates"]
        front_size = len(front) if isinstance(front, (list, dict)) else "?"
        print(f"  [PARETO] front_size={front_size}  members={front}")
        if displaced:
            print(f"    Displaced: {displaced}")

    # ==================================================================
    # STATE SAVED
    # ==================================================================

    def on_state_saved(self, event) -> None:
        self._cur["state_saved"] = {
            "run_dir": event["run_dir"],
        }
        print(f"  [STATE SAVED] run_dir={event['run_dir']}")

    # ==================================================================
    # BUDGET
    # ==================================================================

    def on_budget_updated(self, event) -> None:
        self._cur["budget"] = {
            "used": event["metric_calls_used"],
            "delta": event["metric_calls_delta"],
            "remaining": event["metric_calls_remaining"],
        }
        total = event["metric_calls_used"] + event["metric_calls_remaining"] if event["metric_calls_remaining"] is not None else "?"
        print(f"  [BUDGET] {event['metric_calls_used']}/{total}  (+{event['metric_calls_delta']})  "
              f"remaining={event['metric_calls_remaining']}")

    # ==================================================================
    # MERGE EVENTS
    # ==================================================================

    def on_merge_attempted(self, event) -> None:
        self._cur["merge_attempted"] = {
            "parent_ids": list(event["parent_ids"]),
            "merged_candidate": event["merged_candidate"],
        }
        print(f"  [MERGE ATTEMPTED] parents={list(event['parent_ids'])}")
        for comp, text in event["merged_candidate"].items():
            print(f"    {comp}: {_trunc(text, 150)}")

    def on_merge_accepted(self, event) -> None:
        self._cur["merge_accepted"] = {
            "new_candidate_idx": event["new_candidate_idx"],
            "parent_ids": list(event["parent_ids"]),
        }
        print(f"  [MERGE ACCEPTED] candidate=#{event['new_candidate_idx']}  parents={list(event['parent_ids'])}")

    def on_merge_rejected(self, event) -> None:
        self._cur["merge_rejected"] = {
            "parent_ids": list(event["parent_ids"]),
            "reason": event["reason"],
        }
        print(f"  [MERGE REJECTED] parents={list(event['parent_ids'])}  reason={event['reason']}")

    # ==================================================================
    # ERRORS
    # ==================================================================

    def on_error(self, event) -> None:
        self._cur["error"] = {
            "exception": str(event["exception"]),
            "will_continue": event["will_continue"],
        }
        print(f"  [ERROR] {event['exception']}")
        print(f"    Will continue: {event['will_continue']}")
