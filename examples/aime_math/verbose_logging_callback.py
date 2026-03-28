"""Verbose logging callback for GEPA optimization.

Prints and logs detailed information about every stage of the optimization loop:
- Which candidate/prompt was selected
- Every LLM evaluation: question, attempted answer, reasoning, correct answer, feedback
- The reflective dataset fed to the reflection LLM
- The full prompt sent to the reflection LLM and its raw output
- The new proposed prompt
- Accept/reject decisions with per-example score breakdowns
- Validation set scores per example
- Timestamps for every event
"""

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _sep(title: str, char: str = "=", width: int = 100) -> str:
    return f"\n{char * width}\n  [{_ts()}] {title}\n{char * width}"


class VerboseLoggingCallback:
    """Callback that prints and logs everything going in and out of every LLM call during GEPA optimization.

    Args:
        log_file: Path to write a full log file. If None, only prints to stdout.
                  The log file is never truncated — it gets the complete output.
        truncate_stdout: Max chars for long fields in stdout. Set 0 to disable truncation.
                         The log file always gets the full untruncated output.
    """

    def __init__(self, log_file: str | Path | None = None, truncate_stdout: int = 5000):
        self._log_fh = None
        self._truncate = truncate_stdout
        self._iter_start_times: dict[int, datetime] = {}
        self._optimization_start_time: datetime | None = None
        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(path, "w")

    def _print(self, text: str, truncate: bool = False) -> None:
        """Print to stdout (optionally truncated) and always write full text to log file."""
        if self._log_fh is not None:
            self._log_fh.write(text + "\n")
            self._log_fh.flush()
        if truncate and self._truncate > 0 and len(text) > self._truncate:
            print(text[: self._truncate] + f"\n... [truncated at {self._truncate} chars, full output in log file]")
        else:
            print(text)
        sys.stdout.flush()

    def on_optimization_start(self, event):
        self._optimization_start_time = datetime.now()
        self._print(_sep("OPTIMIZATION START"))
        self._print(f"  Trainset size: {event['trainset_size']}")
        self._print(f"  Valset size:   {event['valset_size']}")
        self._print(f"  Seed candidate:")
        for name, text in event["seed_candidate"].items():
            self._print(f"    [{name}]: {text}")

    def on_optimization_end(self, event):
        self._print(_sep("OPTIMIZATION END"))
        self._print(f"  Best candidate idx:  {event['best_candidate_idx']}")
        self._print(f"  Total iterations:    {event['total_iterations']}")
        self._print(f"  Total metric calls:  {event['total_metric_calls']}")
        if self._optimization_start_time:
            elapsed = datetime.now() - self._optimization_start_time
            self._print(f"  Total wall time:     {elapsed}")

        # Print the final best candidate text
        final_state = event.get("final_state")
        if final_state is not None:
            best_idx = event["best_candidate_idx"]
            best_candidate = final_state.program_candidates[best_idx]
            self._print(f"\n  FINAL BEST CANDIDATE (idx={best_idx}):")
            for name, text in best_candidate.items():
                self._print(f"    [{name}]:")
                self._print(textwrap.indent(text, "      "), truncate=True)

    def on_iteration_start(self, event):
        iteration = event["iteration"]
        self._iter_start_times[iteration] = datetime.now()
        self._print(_sep(f"ITERATION {iteration} START", char="-"))

    def on_iteration_end(self, event):
        iteration = event["iteration"]
        accepted = "ACCEPTED" if event["proposal_accepted"] else "REJECTED"
        elapsed = ""
        if iteration in self._iter_start_times:
            dt = datetime.now() - self._iter_start_times.pop(iteration)
            elapsed = f", took {dt.total_seconds():.1f}s"
        self._print(f"\n--- [{_ts()}] ITERATION {iteration} END ({accepted}{elapsed}) ---")

    def on_candidate_selected(self, event):
        self._print(f"\n>> [{_ts()}] Candidate selected: idx={event['candidate_idx']}, score={event['score']:.4f}")
        for name, text in event["candidate"].items():
            self._print(f"   [{name}]:")
            self._print(textwrap.indent(text, "     "), truncate=True)

    def on_minibatch_sampled(self, event):
        self._print(f"\n>> Minibatch: {len(event['minibatch_ids'])} examples from {event['trainset_size']} total")
        self._print(f"   IDs: {event['minibatch_ids']}")

    def on_evaluation_start(self, event):
        role = "PARENT" if event["capture_traces"] else "NEW CANDIDATE"
        idx_str = f"idx={event['candidate_idx']}" if event["candidate_idx"] is not None else "new"
        self._print(f"\n>> [{_ts()}] Evaluation START ({role}, {idx_str}, batch_size={event['batch_size']})")

    def on_evaluation_end(self, event):
        role = "PARENT" if event["has_trajectories"] else "NEW CANDIDATE"
        idx_str = f"idx={event['candidate_idx']}" if event["candidate_idx"] is not None else "new"
        scores = event["scores"]
        avg = sum(scores) / len(scores) if scores else 0.0
        total = sum(scores)
        self._print(f"\n>> [{_ts()}] Evaluation END ({role}, {idx_str})")
        self._print(f"   Scores: {scores}  (sum={total:.1f}, avg={avg:.4f})")

        # Print per-example details from outputs/trajectories
        outputs = event.get("outputs") or []
        trajectories = event.get("trajectories") or []

        # For PARENT eval: trajectories are the side_info dicts
        # For NEW CANDIDATE eval: no trajectories, outputs are (score, candidate, side_info) tuples
        if trajectories:
            for i, traj in enumerate(trajectories):
                score_str = f"{scores[i]:.1f}" if i < len(scores) else "?"
                self._print(f"\n   --- Example {i + 1} (score={score_str}) ---")
                if isinstance(traj, dict):
                    self._print_side_info(traj, indent=6)
                else:
                    self._print(f"      trajectory: {traj}", truncate=True)
        elif outputs:
            for i, out in enumerate(outputs):
                score_str = f"{scores[i]:.1f}" if i < len(scores) else "?"
                self._print(f"\n   --- Example {i + 1} (score={score_str}) ---")
                if isinstance(out, tuple) and len(out) >= 3:
                    _, candidate_used, side_info = out[0], out[1], out[2]
                    if isinstance(side_info, dict):
                        self._print_side_info(side_info, indent=6)
                    else:
                        self._print(f"      output: {out}", truncate=True)
                else:
                    self._print(f"      output: {out}", truncate=True)

    def on_evaluation_skipped(self, event):
        self._print(f"\n>> Evaluation SKIPPED: idx={event['candidate_idx']}, reason={event['reason']}")
        if event.get("scores"):
            self._print(f"   Scores were: {event['scores']}")

    def on_reflective_dataset_built(self, event):
        self._print(_sep(f"REFLECTIVE DATASET (iter {event['iteration']})", char="."))
        self._print(f"  Candidate idx: {event['candidate_idx']}")
        self._print(f"  Components:    {event['components']}")
        for comp_name, entries in event["dataset"].items():
            self._print(f"\n  Component: [{comp_name}] ({len(entries)} examples)")
            for j, entry in enumerate(entries):
                self._print(f"    -- Example {j + 1} --")
                self._print_side_info(entry, indent=6)

    def on_proposal_start(self, event):
        self._print(_sep(f"PROPOSAL START (iter {event['iteration']})", char="."))
        self._print(f"  Components to update: {event['components']}")
        self._print(f"  Parent candidate:")
        for name, text in event["parent_candidate"].items():
            self._print(f"    [{name}]:")
            self._print(textwrap.indent(text, "      "), truncate=True)

    def on_proposal_end(self, event):
        self._print(_sep(f"PROPOSAL END (iter {event['iteration']})", char="."))

        # Print the full prompt sent to the reflection LM
        for comp_name, prompt in event.get("prompts", {}).items():
            self._print(f"\n  >> Reflection LM INPUT for [{comp_name}]:")
            if isinstance(prompt, str):
                self._print(textwrap.indent(prompt, "     "), truncate=True)
            elif isinstance(prompt, list):
                self._print(textwrap.indent(json.dumps(prompt, indent=2, default=str), "     "), truncate=True)

        # Print the full raw LM output
        for comp_name, raw_out in event.get("raw_lm_outputs", {}).items():
            self._print(f"\n  >> Reflection LM OUTPUT for [{comp_name}]:")
            self._print(textwrap.indent(str(raw_out), "     "), truncate=True)

        # Print the extracted new instructions
        for comp_name, new_text in event.get("new_instructions", {}).items():
            self._print(f"\n  >> NEW PROMPT for [{comp_name}]:")
            self._print(textwrap.indent(new_text, "     "), truncate=True)

    def on_candidate_accepted(self, event):
        self._print(f"\n>> [{_ts()}] CANDIDATE ACCEPTED: new_idx={event['new_candidate_idx']}, "
                     f"subsample_score={event['new_score']:.4f}, parents={list(event['parent_ids'])}")

    def on_candidate_rejected(self, event):
        self._print(f"\n>> [{_ts()}] CANDIDATE REJECTED: "
                     f"old_subsample_sum={event['old_score']:.4f}, "
                     f"new_subsample_sum={event['new_score']:.4f}, "
                     f"reason={event['reason']}")

    def on_valset_evaluated(self, event):
        is_best = " ** NEW BEST **" if event["is_best_program"] else ""
        self._print(f"\n>> [{_ts()}] VALSET EVAL: idx={event['candidate_idx']}, "
                     f"avg_score={event['average_score']:.4f}, "
                     f"evaluated={event['num_examples_evaluated']}/{event['total_valset_size']}, "
                     f"parents={list(event.get('parent_ids', []))}"
                     f"{is_best}")

        # Print the full candidate that was evaluated
        candidate = event.get("candidate")
        if candidate:
            for name, text in candidate.items():
                self._print(f"   [{name}]:")
                self._print(textwrap.indent(text, "     "), truncate=True)

        # Print per-example val scores
        scores_by_id = event.get("scores_by_val_id", {})
        if scores_by_id:
            score_vals = list(scores_by_id.values())
            correct = sum(1 for s in score_vals if s >= 1.0)
            self._print(f"   {correct}/{len(score_vals)} correct")
            for val_id, score in scores_by_id.items():
                marker = "+" if score >= 1.0 else "-"
                self._print(f"     [{marker}] val_id={val_id}: {score:.4f}")

    def on_pareto_front_updated(self, event):
        self._print(f"\n>> Pareto front updated: {event['new_front']}")
        if event["displaced_candidates"]:
            self._print(f"   Displaced: {event['displaced_candidates']}")

    def on_budget_updated(self, event):
        remaining = event["metric_calls_remaining"]
        remaining_str = str(remaining) if remaining is not None else "unlimited"
        self._print(f"   [Budget] used={event['metric_calls_used']} "
                     f"(+{event['metric_calls_delta']}), remaining={remaining_str}")

    def on_state_saved(self, event):
        if event["run_dir"]:
            self._print(f"   [State saved to {event['run_dir']}]")

    def on_error(self, event):
        self._print(f"\n!! [{_ts()}] ERROR at iteration {event['iteration']}: {event['exception']}")
        self._print(f"   Will continue: {event['will_continue']}")

    def _print_side_info(self, info: dict, indent: int = 4) -> None:
        """Pretty-print a side_info dict with labeled fields."""
        prefix = " " * indent
        for key, value in info.items():
            if key.endswith("_specific_info"):
                continue
            val_str = str(value)
            if "\n" in val_str:
                self._print(f"{prefix}{key}:")
                self._print(textwrap.indent(val_str, prefix + "  "), truncate=True)
            else:
                self._print(f"{prefix}{key}: {val_str}", truncate=True)
