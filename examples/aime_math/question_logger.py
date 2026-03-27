"""Per-question logging callback for AIME experiments.

Logs every LLM evaluation (train, val, test) with full context:
problem, reasoning, answer, correct answer, and score.  Also validates
feedback in reflective datasets to catch bugs like the ScoreWithFeedback
class-attribute issue early.

Output: ``{log_dir}/questions.jsonl``
"""

import json
import time
from pathlib import Path
from typing import Any, Callable


class QuestionLogger:
    """GEPA callback that logs per-question LLM inputs/outputs.

    Hooks into:
      - on_evaluation_start / on_evaluation_end  — train evals (parent & child)
      - on_valset_evaluated                      — validation evaluations
      - on_reflective_dataset_built              — feedback validation
      - log_test_results() / log_captured_test()  — test evaluations (manual)
    """

    def __init__(
        self,
        log_dir: str,
        valset: list,
        extract_fn: Callable[[str], int] | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._valset = valset
        self._extract_fn = extract_fn
        self._pending_inputs: dict[tuple[int, int | None], list] = {}
        self._file = open(self.log_dir / "questions.jsonl", "a")
        self._test_capture: list[tuple[Any, Any, float]] = []

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_answer(self, raw: str) -> int | None:
        if self._extract_fn is None:
            return None
        try:
            return self._extract_fn(str(raw))
        except (ValueError, TypeError):
            return None

    def _pred_fields(self, output: Any) -> dict[str, Any]:
        """Extract reasoning/answer from a dspy.Prediction or similar."""
        if output is None:
            return {"llm_reasoning": None, "llm_answer_raw": None, "llm_answer_parsed": None}
        reasoning = getattr(output, "reasoning", None)
        answer_raw = getattr(output, "answer", None)
        return {
            "llm_reasoning": str(reasoning) if reasoning is not None else None,
            "llm_answer_raw": str(answer_raw) if answer_raw is not None else None,
            "llm_answer_parsed": self._parse_answer(answer_raw) if answer_raw is not None else None,
        }

    def _example_fields(self, example: Any) -> dict[str, Any]:
        """Extract problem/answer/solution from a dspy.Example."""
        problem = getattr(example, "problem", None)
        answer = getattr(example, "answer", None)
        solution = getattr(example, "solution", None)
        return {
            "problem": str(problem) if problem is not None else None,
            "correct_answer": str(answer) if answer is not None else None,
            "correct_solution": str(solution) if solution is not None else None,
        }

    # ------------------------------------------------------------------
    # Test capture helper
    # ------------------------------------------------------------------

    def make_capturing_metric(self, metric_fn: Callable) -> Callable:
        """Wrap a metric to capture (example, prediction, score) for test logging.

        Usage::

            capturing_metric = question_logger.make_capturing_metric(metric)
            evaluate = dspy.Evaluate(devset=..., metric=capturing_metric, ...)
            evaluate(program)
            question_logger.log_captured_test("test_baseline")
        """
        self._test_capture = []

        def wrapper(example, prediction, trace=None, **kwargs):
            score = metric_fn(example, prediction, trace=trace, **kwargs)
            self._test_capture.append((example, prediction, float(score)))
            return score

        return wrapper

    def log_captured_test(self, label: str) -> None:
        """Log and clear results captured by make_capturing_metric."""
        self.log_test_results(self._test_capture, label=label)
        self._test_capture = []

    # ------------------------------------------------------------------
    # Train evaluation callbacks
    # ------------------------------------------------------------------

    def on_evaluation_start(self, event) -> None:
        key = (event["iteration"], event["candidate_idx"])
        self._pending_inputs[key] = event["inputs"]

    def on_evaluation_end(self, event) -> None:
        key = (event["iteration"], event["candidate_idx"])
        inputs = self._pending_inputs.pop(key, None)
        if inputs is None:
            print(f"  [QUESTION_LOG] WARNING: No pending inputs for key={key}")
            return

        is_parent = event["candidate_idx"] is not None
        phase = "train_parent" if is_parent else "train_child"
        iteration = event["iteration"]
        candidate_idx = event["candidate_idx"]

        for i, (inp, out, score) in enumerate(zip(inputs, event["outputs"], event["scores"])):
            ex = self._example_fields(inp)
            pred = self._pred_fields(out)
            self._write({
                "timestamp": time.time(),
                "phase": phase,
                "iteration": iteration,
                "candidate_idx": candidate_idx,
                "question_idx": i,
                **ex,
                **pred,
                "score": score,
            })
            tag = phase.upper()
            print(f"  [{tag} Q#{i} iter={iteration}] score={score:.0f}"
                  f" | llm={pred['llm_answer_parsed']} correct={ex['correct_answer']}")
            if out is None:
                print(f"    WARNING: output is None for Q#{i}!")

    # ------------------------------------------------------------------
    # Validation evaluation callback
    # ------------------------------------------------------------------

    def on_valset_evaluated(self, event) -> None:
        iteration = event["iteration"]
        candidate_idx = event["candidate_idx"]
        scores_by_val_id = event["scores_by_val_id"]
        outputs_by_val_id = event.get("outputs_by_val_id")

        n_correct = 0
        n_total = 0

        for val_id, score in sorted(scores_by_val_id.items(), key=lambda x: str(x[0])):
            n_total += 1
            if score >= 1.0:
                n_correct += 1

            ex: dict[str, Any] = {}
            if isinstance(val_id, int) and 0 <= val_id < len(self._valset):
                ex = self._example_fields(self._valset[val_id])

            pred = {"llm_reasoning": None, "llm_answer_raw": None, "llm_answer_parsed": None}
            if outputs_by_val_id is not None and val_id in outputs_by_val_id:
                pred = self._pred_fields(outputs_by_val_id[val_id])

            self._write({
                "timestamp": time.time(),
                "phase": "val",
                "iteration": iteration,
                "candidate_idx": candidate_idx,
                "val_id": val_id,
                **ex,
                **pred,
                "score": score,
            })

        avg = event["average_score"]
        print(f"  [VAL QUESTIONS iter={iteration} cand=#{candidate_idx}] "
              f"{n_correct}/{n_total} correct ({avg:.4f}) -- {n_total} questions logged")

    # ------------------------------------------------------------------
    # Reflective dataset validation
    # ------------------------------------------------------------------

    def on_reflective_dataset_built(self, event) -> None:
        iteration = event["iteration"]
        dataset = event["dataset"]
        any_none = False

        for comp_name, examples in dataset.items():
            for i, ex in enumerate(examples):
                feedback = ex.get("Feedback")
                if feedback is None:
                    any_none = True
                    print(f"\n  WARNING: Feedback is None for example #{i} in '{comp_name}'!")
                    print(f"    The reflection LM will not see why the answer was wrong.")
                    print(f"    Check predictor_feedback -- ScoreWithFeedback bug?\n")

                self._write({
                    "timestamp": time.time(),
                    "phase": "reflective_dataset",
                    "iteration": iteration,
                    "candidate_idx": event["candidate_idx"],
                    "component": comp_name,
                    "example_idx": i,
                    "feedback": str(feedback) if feedback is not None else None,
                    "feedback_is_none": feedback is None,
                    "inputs": str(ex.get("Inputs", ""))[:500],
                    "outputs": str(ex.get("Generated Outputs", ""))[:500],
                })

        if any_none:
            print("  CRITICAL: Some feedback values are None!")
            print("  Optimization will likely not improve without feedback.")

    # ------------------------------------------------------------------
    # Test evaluation (manual)
    # ------------------------------------------------------------------

    def log_test_results(self, results: list, label: str = "test") -> None:
        """Log test results.

        Args:
            results: List of (example, prediction, score) tuples.
            label: e.g. "test_baseline" or "test_optimized".
        """
        n_correct = 0
        for i, (example, prediction, score) in enumerate(results):
            ex = self._example_fields(example)
            pred = self._pred_fields(prediction)
            self._write({
                "timestamp": time.time(),
                "phase": label,
                "iteration": None,
                "candidate_idx": None,
                "question_idx": i,
                **ex,
                **pred,
                "score": float(score),
            })
            if score >= 1.0:
                n_correct += 1
            tag = label.upper()
            print(f"  [{tag} Q#{i}] score={score:.0f}"
                  f" | llm={pred['llm_answer_parsed']} correct={ex['correct_answer']}")

        n_total = len(results)
        pct = n_correct / n_total if n_total else 0
        print(f"\n  [{label.upper()} SUMMARY] {n_correct}/{n_total} correct ({pct:.4f})")
