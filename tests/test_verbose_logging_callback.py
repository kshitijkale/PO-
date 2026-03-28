"""Tests for VerboseLoggingCallback with optimize_anything.

Verifies that the verbose logging callback is invoked at all expected stages
and prints detailed output during GEPA optimization, using mock LLMs.
"""

import sys
from io import StringIO
from unittest.mock import Mock

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)

# Inline the callback here so the test is self-contained and doesn't depend on examples/ path
sys.path.insert(0, "examples/aime_math")
from verbose_logging_callback import VerboseLoggingCallback


def _make_mock_evaluator():
    """Create a mock evaluator that returns deterministic scores and side_info."""
    call_count = {"n": 0}

    def evaluator(candidate: str, example) -> tuple[float, SideInfo]:
        call_count["n"] += 1
        question = example.get("question", "unknown question")
        answer = example.get("answer", "0")

        # Simulate the LLM "trying" to answer
        attempted = str(int(answer) + 1) if call_count["n"] % 3 != 0 else answer
        score = 1.0 if attempted == answer else 0.0
        reasoning = f"I think the answer is {attempted} because I computed it step by step."
        if score == 0.0:
            feedback = f"Incorrect. You said {attempted}, correct answer is {answer}."
        else:
            feedback = f"Correct! The answer is {answer}."

        side_info = {
            "question": question,
            "prompt_used": candidate,
            "attempted_answer": attempted,
            "correct_answer": answer,
            "reasoning": reasoning,
            "feedback": feedback,
        }
        return score, side_info

    return evaluator


def _make_mock_reflection_lm():
    """Create a mock reflection LM that returns a plausible 'improved' prompt."""
    call_count = {"n": 0}

    def reflection_lm(prompt):
        call_count["n"] += 1
        return (
            "```\n"
            f"You are an expert math solver (revision {call_count['n']}). "
            "Read the problem carefully. Break it into steps. "
            "Double-check your arithmetic. "
            "Give only the final integer answer.\n"
            "```"
        )

    return reflection_lm


MOCK_DATASET = [
    {"question": "What is 2 + 3?", "answer": "5"},
    {"question": "What is 7 * 8?", "answer": "56"},
    {"question": "What is 100 / 4?", "answer": "25"},
    {"question": "What is 15 - 9?", "answer": "6"},
    {"question": "What is 3^3?", "answer": "27"},
    {"question": "What is 12 + 19?", "answer": "31"},
]


def test_verbose_logging_callback_fires_all_events():
    """Test that VerboseLoggingCallback prints output for all key optimization events."""
    callback = VerboseLoggingCallback()
    mock_evaluator = _make_mock_evaluator()
    mock_reflection_lm = _make_mock_reflection_lm()

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=30,
            parallel=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=mock_reflection_lm,
            reflection_minibatch_size=2,
        ),
        callbacks=[callback],
    )

    # Capture stdout to verify logging output
    captured = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    try:
        result = optimize_anything(
            seed_candidate="Solve the math problem carefully. Give the final answer as a number.",
            evaluator=mock_evaluator,
            dataset=MOCK_DATASET[:4],
            valset=MOCK_DATASET[4:],
            config=config,
        )
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()

    # Print a summary of what was captured (visible in pytest -s)
    print(f"\n--- Captured {len(output)} chars of verbose logging output ---")
    print(output[:5000])
    if len(output) > 5000:
        print(f"... [{len(output)} chars total]")

    # Verify key sections appear in output
    assert "OPTIMIZATION START" in output, "Should log optimization start"
    assert "OPTIMIZATION END" in output, "Should log optimization end"
    assert "Candidate selected" in output, "Should log candidate selection"
    assert "Evaluation START" in output, "Should log evaluation start"
    assert "Evaluation END" in output, "Should log evaluation end"
    assert "REFLECTIVE DATASET" in output, "Should log reflective dataset"
    assert "PROPOSAL START" in output, "Should log proposal start"
    assert "PROPOSAL END" in output, "Should log proposal end"
    assert "Reflection LM INPUT" in output, "Should log the prompt sent to reflection LM"
    assert "Reflection LM OUTPUT" in output, "Should log the raw LM output"
    assert "NEW PROMPT" in output, "Should log the extracted new prompt"
    assert "VALSET EVAL" in output, "Should log validation set evaluation"

    # Verify evaluator side_info details appear
    assert "question" in output.lower(), "Should include question in evaluation details"
    assert "feedback" in output.lower(), "Should include feedback in evaluation details"

    # Verify the result is valid
    assert result.best_candidate is not None


def test_verbose_logging_callback_with_print():
    """Run optimization with verbose logging and print all output (use pytest -s to see)."""
    callback = VerboseLoggingCallback()
    mock_evaluator = _make_mock_evaluator()
    mock_reflection_lm = _make_mock_reflection_lm()

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=18,
            parallel=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=mock_reflection_lm,
            reflection_minibatch_size=2,
        ),
        callbacks=[callback],
    )

    result = optimize_anything(
        seed_candidate="Solve the math problem. Give the answer as a number.",
        evaluator=mock_evaluator,
        dataset=MOCK_DATASET[:3],
        valset=MOCK_DATASET[3:],
        config=config,
    )

    assert result.best_candidate is not None
    print(f"\n\nFinal best candidate:\n{result.best_candidate}")


def test_verbose_logging_callback_log_file(tmp_path):
    """Test that log file captures full untruncated output."""
    log_file = tmp_path / "test_verbose.log"
    callback = VerboseLoggingCallback(log_file=log_file, truncate_stdout=200)
    mock_evaluator = _make_mock_evaluator()
    mock_reflection_lm = _make_mock_reflection_lm()

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=18,
            parallel=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=mock_reflection_lm,
            reflection_minibatch_size=2,
        ),
        callbacks=[callback],
    )

    result = optimize_anything(
        seed_candidate="Solve the math problem. Give the answer as a number.",
        evaluator=mock_evaluator,
        dataset=MOCK_DATASET[:3],
        valset=MOCK_DATASET[3:],
        config=config,
    )

    assert result.best_candidate is not None

    # Verify log file was written and has full content
    log_content = log_file.read_text()
    assert len(log_content) > 0, "Log file should not be empty"
    assert "OPTIMIZATION START" in log_content
    assert "OPTIMIZATION END" in log_content
    assert "FINAL BEST CANDIDATE" in log_content, "Should print the final best candidate text"
    assert "Reflection LM INPUT" in log_content
    assert "Reflection LM OUTPUT" in log_content
    assert "val_id=" in log_content, "Should print per-example val scores"
    assert "Total wall time" in log_content, "Should print elapsed time"
