"""
GEPA for AIME (Math)

Optimizes GPT-4.1 Mini's Chain of Thought (dspy.ChainOfThought) for solving
math problems (AIME) using the GEPA optimizer.

Dataset:
  - Train/Val: AIME 2022-2024 (90 problems, split 50/50)
  - Test: AIME 2025 (30 problems × 5 repeats for stability)
"""

import os
import random
import re

import dspy
from datasets import load_dataset
from dotenv import load_dotenv

import gepa
from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter
from examples.aime_math.logger import ResearchLogger
from examples.aime_math.question_logger import QuestionLogger

load_dotenv()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def init_dataset():
    train_split = load_dataset("AI-MO/aimo-validation-aime")["train"]
    train_split = [
        dspy.Example({
            "problem": x["problem"],
            "solution": x["solution"],
            "answer": x["answer"],
        }).with_inputs("problem")
        for x in train_split
    ]
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)

    test_split = load_dataset("MathArena/aime_2025")["train"]
    test_split = [
        dspy.Example({
            "problem": x["problem"],
            "answer": x["answer"],
        }).with_inputs("problem")
        for x in test_split
    ]

    train_set = train_split[: int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]
    test_set = test_split * 5

    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""

    problem = dspy.InputField()
    answer = dspy.OutputField()


program = dspy.ChainOfThought(GenerateResponse)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def extract_integer(raw: str) -> int:
    """Extract an integer from an LLM answer, stripping LaTeX and extra text.

    Handles: \\boxed{123}, $123$, "123 minutes", "The answer is 123", etc.
    """
    s = str(raw).strip()
    # Strip \boxed{...} (possibly nested in \[ \] or $ $)
    m = re.search(r"\\boxed\{([^}]+)\}", s)
    if m:
        s = m.group(1).strip()
    # Strip surrounding $ or \( \) or \[ \]
    s = re.sub(r"^[\$\\(\[]+|[\$\\)\]]+$", "", s).strip()
    # Try direct parse first
    try:
        return int(s)
    except ValueError:
        pass
    # Extract the first integer-like token (optional leading minus)
    m = re.search(r"-?\d+", s)
    if m:
        return int(m.group())
    raise ValueError(f"Could not extract integer from: {raw!r}")


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Simple exact-match metric for evaluation."""
    correct_answer = int(example["answer"])
    try:
        llm_answer = extract_integer(prediction.answer)
    except (ValueError, TypeError):
        return 0
    return int(correct_answer == llm_answer)


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Rich feedback metric for GEPA optimization.

    Returns dspy.Prediction(score=..., feedback=...) so the reflection LM
    can see *why* the answer was wrong, including full solutions when available.
    """
    correct_answer = int(example["answer"])
    written_solution = example.get("solution", "")

    try:
        llm_answer = extract_integer(prediction.answer)
    except (ValueError, TypeError):
        feedback_text = (
            f"The final answer must be a valid integer and nothing else. "
            f"You responded with '{prediction.answer}', which couldn't be parsed as a python integer. "
            f"Please ensure your answer is a valid integer without any additional text or formatting."
            f" The correct answer is '{correct_answer}'."
        )
        if written_solution:
            feedback_text += (
                f" Here's the full step-by-step solution:\n{written_solution}\n\n"
                "Think about what takeaways you can learn from this solution to improve your "
                "future answers and approach to similar problems and ensure your final answer "
                "is a valid integer."
            )
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)

    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    if written_solution:
        feedback_text += (
            f" Here's the full step-by-step solution:\n{written_solution}\n\n"
            "Think about what takeaways you can learn from this solution to improve your "
            "future answers and approach to similar problems."
        )

    return dspy.Prediction(score=score, feedback=feedback_text)


def predictor_feedback(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
    """Per-predictor feedback bridge for DspyAdapter.

    Returns a dict (not ScoreWithFeedback) because ScoreWithFeedback has a
    class-attribute bug: ``feedback: str | None = None`` shadows Prediction's
    internal _store, so .feedback always returns None regardless of the value passed.
    """
    result = metric_with_feedback(module_inputs, module_outputs)
    return {"score": result.score, "feedback": result.feedback}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ["OPENAI_API_KEY"]
    lm = dspy.LM("openai/gpt-4.1-mini", temperature=1, api_key=api_key, max_tokens=32000)
    dspy.configure(lm=lm)

    train_set, val_set, test_set = init_dataset()
    print(f"Dataset: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

    question_logger = QuestionLogger(
        log_dir="outputs/aime_math_baseline/research_logs",
        valset=val_set,
        extract_fn=extract_integer,
    )

    # --- Baseline evaluation ---
    print("\nEvaluating unoptimized Chain Of Thought...")
    capturing_metric = question_logger.make_capturing_metric(metric)
    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=capturing_metric,
        num_threads=32,
        display_table=True,
        display_progress=True,
    )
    baseline_result = evaluate(program)
    question_logger.log_captured_test("test_baseline")

    # --- Optimize with GEPA ---
    seed_candidate = {"predict": program.predict.signature.instructions}

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=metric_with_feedback,
        feedback_map={"predict": predictor_feedback},
        num_threads=32,
        reflection_minibatch_size=3,
        reflection_lm=dspy.LM(
            model="openai/gpt-5",
            temperature=1.0,
            max_tokens=32000,
            api_key=api_key,
        ),
    )

    logger = ResearchLogger("outputs/aime_math_baseline/research_logs")

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_set,
        valset=val_set,
        adapter=adapter,
        reflection_minibatch_size=3,
        max_metric_calls=500,
        run_dir="outputs/aime_math_baseline",
        callbacks=[logger, question_logger],
        cache_evaluation=True,
        track_best_outputs=True,
        use_cloudpickle=True,
    )

    # --- Print optimized prompt ---
    optimized_program = adapter.build_program(result.best_candidate)
    print("\nOptimized instruction:")
    print(optimized_program.predict.signature.instructions)

    # --- Evaluate optimized program ---
    print("\nEvaluating optimized Chain Of Thought...")
    evaluate_opt = dspy.Evaluate(
        devset=test_set,
        metric=question_logger.make_capturing_metric(metric),
        num_threads=32,
        display_table=True,
        display_progress=True,
    )
    optimized_result = evaluate_opt(optimized_program)
    question_logger.log_captured_test("test_optimized")

    # Handle both float and object return types from dspy.Evaluate
    baseline_score = baseline_result if isinstance(baseline_result, (int, float)) else baseline_result.score
    optimized_score = optimized_result if isinstance(optimized_result, (int, float)) else optimized_result.score

    print(f"\nBaseline Score: {baseline_score:.1f}%")
    print(f"Optimized Score: {optimized_score:.1f}%")
    print(f"Improvement: {optimized_score - baseline_score:+.1f}%")

    question_logger.close()


if __name__ == "__main__":
    main()
