"""Evaluate a single candidate prompt N times on the AIME 2025 test set.

Usage:
    uv run python -m examples.aime_math.eval_single_candidate --candidate 7 --runs 2
"""

import argparse
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


def evaluate_once(prompt: str, testset, run_idx: int) -> dict:
    print(f"\n{'='*80}")
    print(f"Run {run_idx + 1}")
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
    print(f"\n  => Run {run_idx + 1}: {correct}/{len(testset)} ({score:.1%}) in {elapsed.total_seconds():.1f}s")

    return {
        "run_idx": run_idx,
        "score": score,
        "correct": correct,
        "total": len(testset),
        "eval_time_seconds": elapsed.total_seconds(),
        "per_problem": per_problem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    candidates = json.load(open("outputs/aime_math/candidates.json"))
    if args.candidate >= len(candidates):
        print(f"ERROR: candidate {args.candidate} not found (max {len(candidates) - 1})")
        sys.exit(1)

    prompt = candidates[args.candidate]["current_candidate"]
    print(f"Candidate {args.candidate}:")
    print(f"  {prompt[:120]}...")

    _, _, testset = load_math_dataset()
    print(f"Test set: {len(testset)} problems")

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

    all_runs = []
    start_time = datetime.now()
    for r in range(args.runs):
        result = evaluate_once(prompt, testset, r)
        all_runs.append(result)

    elapsed = datetime.now() - start_time

    # Summary
    scores = [r["correct"] for r in all_runs]
    print(f"\n\n{'='*80}")
    print(f"SUMMARY — Candidate {args.candidate}, {args.runs} runs")
    print(f"Model: gpt-4.1-mini, temperature=0, cache=off")
    print(f"Total time: {elapsed}")
    print(f"{'='*80}")
    for r in all_runs:
        print(f"  Run {r['run_idx'] + 1}: {r['correct']}/{r['total']} ({r['score']:.1%})")
    print(f"  Mean: {sum(scores)/len(scores):.1f}/{all_runs[0]['total']}")

    # Per-problem agreement
    print(f"\nPER-PROBLEM AGREEMENT:")
    print(f"{'Prob':<6}", end="")
    for r in range(args.runs):
        print(f"{'R' + str(r+1):<5}", end="")
    print(f"  {'Ans':<8}")
    print("-" * (6 + 5 * args.runs + 10))

    for prob_idx in range(len(testset)):
        row = f"{prob_idx:<6}"
        for r in all_runs:
            p = r["per_problem"][prob_idx]
            mark = "+" if p["score"] >= 1.0 else "-"
            row += f"{mark:<5}"
        ans = all_runs[0]["per_problem"][prob_idx]["correct_answer"]
        row += f"  {ans:<8}"
        print(row)

    # Save
    output_path = Path(f"outputs/aime_math/eval_candidate_{args.candidate}_{args.runs}runs.json")
    output = {
        "timestamp": datetime.now().isoformat(),
        "candidate_idx": args.candidate,
        "prompt": prompt,
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "cache": False,
        "num_runs": args.runs,
        "total_time_seconds": elapsed.total_seconds(),
        "runs": all_runs,
    }
    json.dump(output, open(output_path, "w"), indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
