"""
Pull LangSmith evaluator feedback for a benchmark run.

Use this after running benchmark.py. It finds root benchmark traces by run ID,
pulls any feedback/evaluator results attached to those runs, and prints a JSON
summary by evaluator key and by benchmark case.

Run:
  python eval_results.py --run-id june-10-latency-v1
"""
import argparse
import json
import os
from collections import Counter, defaultdict

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

PROJECT = os.getenv("LANGCHAIN_PROJECT")


def _root_benchmark_runs(client: Client, run_id: str) -> list:
    return list(
        client.list_runs(
            project_name=PROJECT,
            filter=f'and(has(tags, "benchmark"), has(tags, "run-{run_id}"))',
            is_root=True,
        )
    )


def _score_value(score):
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if isinstance(score, (int, float)):
        return float(score)
    return None


def build_eval_summary(run_id: str) -> dict:
    client = Client()
    runs = _root_benchmark_runs(client, run_id)
    run_ids = [run.id for run in runs]
    feedback = list(client.list_feedback(run_ids=run_ids)) if run_ids else []

    run_by_id = {str(run.id): run for run in runs}
    by_key = defaultdict(list)
    per_case = {}

    for fb in feedback:
        run = run_by_id.get(str(fb.run_id))
        if not run:
            continue

        case_id = run.metadata.get("case_id")
        entry = per_case.setdefault(
            str(case_id),
            {
                "case_id": case_id,
                "expected_route": run.metadata.get("expected_route"),
                "workflow": run.metadata.get("workflow"),
                "feedback": [],
            },
        )
        item = {
            "key": fb.key,
            "score": fb.score,
            "value": fb.value,
            "comment": fb.comment,
        }
        entry["feedback"].append(item)
        by_key[fb.key].append(item)

    summary_by_key = {}
    for key, items in sorted(by_key.items()):
        numeric = [_score_value(item["score"]) for item in items]
        numeric = [value for value in numeric if value is not None]
        values = Counter(str(item["value"]) for item in items if item["value"] is not None)
        summary_by_key[key] = {
            "count": len(items),
            "avg_score": round(sum(numeric) / len(numeric), 3) if numeric else None,
            "values": dict(values),
        }

    missing_feedback = [
        {
            "case_id": run.metadata.get("case_id"),
            "workflow": run.metadata.get("workflow"),
            "expected_route": run.metadata.get("expected_route"),
        }
        for run in runs
        if str(run.metadata.get("case_id")) not in per_case
    ]

    return {
        "run_id": run_id,
        "project": PROJECT,
        "root_runs": len(runs),
        "feedback_items": len(feedback),
        "summary_by_key": summary_by_key,
        "missing_feedback": sorted(missing_feedback, key=lambda row: row["case_id"] or 0),
        "per_case": [per_case[key] for key in sorted(per_case, key=lambda value: int(value))],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize LangSmith evaluator feedback for a benchmark run.")
    parser.add_argument("--run-id", required=True, help="Benchmark run ID, e.g. june-10-latency-v1")
    args = parser.parse_args()

    print(json.dumps(build_eval_summary(args.run_id), indent=2, default=str))
