"""Evaluate all candidate prompts from a GEPA run on the AIME 2025 test set.

Evaluates each candidate sequentially (one at a time), using 32 threads
for parallel evaluation across the 30 test problems. Temperature 0 for
deterministic results. DSPy cache disabled.

Usage:
    uv run python -m examples.aime_math.eval_all_candidates
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import dspy
from dotenv import load_dotenv

from examples.aime_math.utils import load_math_dataset, math_metric

load_dotenv(Path(__file__).parent / ".env")


def evaluate_single_problem(prompt: str, example, problem_idx: int) -> dict:
    """Evaluate a prompt on a single problem. Runs inside a thread."""
    predictor = dspy.ChainOfThought("input -> answer")
    predictor.predict.signature.instructions = prompt

    try:
        prediction = predictor(input=example.input)
        answer_str = str(prediction.answer)
        reasoning = getattr(prediction, "reasoning", "")
        score, feedback = math_metric(example, prediction)
    except Exception as e:
        answer_str = f"ERROR: {e}"
        reasoning = ""
        score = 0.0
        feedback = str(e)

    return {
        "problem_idx": problem_idx,
        "question": example.input,
        "correct_answer": str(example.answer),
        "model_answer": answer_str,
        "reasoning": reasoning,
        "score": score,
        "feedback": feedback,
    }


def evaluate_candidate(prompt: str, testset, candidate_idx: int) -> dict:
    """Evaluate a single candidate prompt on the full test set using 32 threads."""
    print(f"\n{'='*80}")
    print(f"Evaluating Candidate {candidate_idx}")
    print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"{'='*80}")

    start = datetime.now()
    per_problem = [None] * len(testset)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {}
        for i, example in enumerate(testset):
            future = executor.submit(evaluate_single_problem, prompt, example, i)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            per_problem[idx] = result
            mark = "+" if result["score"] >= 1.0 else "-"
            print(f"  [{mark}] Problem {idx}: model={result['model_answer']}, correct={result['correct_answer']}")

    elapsed = datetime.now() - start
    correct = sum(1 for p in per_problem if p["score"] >= 1.0)
    score = correct / len(testset)

    print(f"\n  => Candidate {candidate_idx}: {correct}/{len(testset)} ({score:.1%}) in {elapsed.total_seconds():.1f}s")

    return {
        "candidate_idx": candidate_idx,
        "prompt": prompt,
        "score": score,
        "correct": correct,
        "total": len(testset),
        "eval_time_seconds": elapsed.total_seconds(),
        "per_problem": per_problem,
    }


def main():
    # Load candidates
    candidates_path = Path("outputs/aime_math/candidates.json")
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found. Run the optimization first.")
        sys.exit(1)

    candidates = json.load(open(candidates_path))
    print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    # Load test set
    _, _, testset = load_math_dataset()
    print(f"Test set: {len(testset)} problems (AIME 2025)")

    # Configure LM: temperature 0, cache OFF
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    solver_lm = dspy.LM(
        "gpt-4.1-mini",
        api_key=api_key,
        temperature=0.0,
        max_tokens=32000,
        cache=False,
    )
    dspy.configure(lm=solver_lm)

    # Evaluate each candidate sequentially
    all_results = []
    start_time = datetime.now()

    for i, candidate in enumerate(candidates):
        prompt = candidate["current_candidate"]
        result = evaluate_candidate(prompt, testset, i)
        all_results.append(result)

    elapsed = datetime.now() - start_time

    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"RESULTS SUMMARY — AIME 2025 Test Set ({len(testset)} problems)")
    print(f"Model: gpt-4.1-mini, temperature=0, cache=off")
    print(f"Total time: {elapsed}")
    print(f"{'='*80}")
    print(f"\n{'Idx':<5} {'Score':<12} {'Correct':<10} {'Prompt'}")
    print(f"{'-'*5} {'-'*12} {'-'*10} {'-'*60}")

    best_idx = max(range(len(all_results)), key=lambda j: all_results[j]["score"])
    for r in all_results:
        marker = " <-- BEST" if r["candidate_idx"] == best_idx else ""
        prompt_snippet = r["prompt"][:60] + "..." if len(r["prompt"]) > 60 else r["prompt"]
        print(f"{r['candidate_idx']:<5} {r['score']:<12.1%} {r['correct']}/{r['total']:<8} {prompt_snippet}{marker}")

    # Per-problem breakdown
    print(f"\n\nPER-PROBLEM BREAKDOWN (+ = correct, - = wrong):")
    header = f"{'Prob':<6}"
    for i in range(len(all_results)):
        header += f"{'C' + str(i):<5}"
    header += f"  {'Ans':<8} Question"
    print(header)
    print("-" * len(header))

    for prob_idx in range(len(testset)):
        row = f"{prob_idx:<6}"
        for r in all_results:
            p = r["per_problem"][prob_idx]
            mark = "+" if p["score"] >= 1.0 else "-"
            row += f"{mark:<5}"
        first = all_results[0]["per_problem"][prob_idx]
        row += f"  {first['correct_answer']:<8} {first['question'][:50]}"
        print(row)

    # Save full results
    output_path = Path("outputs/aime_math/eval_all_candidates.json")
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "cache": False,
        "test_set": "MathArena/aime_2025",
        "test_set_size": len(testset),
        "num_candidates": len(candidates),
        "total_time_seconds": elapsed.total_seconds(),
        "results": all_results,
    }
    json.dump(output, open(output_path, "w"), indent=2, default=str)
    print(f"\n\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
