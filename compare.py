"""
vAIcation comparison report — pulls baseline (ReAct) and routed (LangGraph)
runs straight out of LangSmith and produces a single before/after report.

Both `baseline.py` and `benchmark.py` tag every run with a `run_type`
("baseline" vs. "benchmark") and a `case_id` matching the shared BENCHMARK
dataset's `id`. LangSmith's tracer already recorded, for each of those runs,
every individual LLM call and tool call — which one, how many times, and how
long each took — with no custom instrumentation needed on our end (see the
note in baseline.py about why we don't hand-count this ourselves).

This script just asks LangSmith for that data, lines up matching case_ids
from both run types, and prints one consolidated report — instead of opening
34 individual traces in the UI and comparing them by eye.

Run *after* both `baseline.py` and `benchmark.py` (or `/eval/run`) have
executed at least once, so their traces exist in the project.

Run: `python compare.py`
"""
import json
import argparse
import os
from collections import Counter

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

PROJECT = os.getenv("LANGCHAIN_PROJECT")


def _root_runs_by_case(client: Client, run_type: str, benchmark_run_id: str) -> dict:
    """Top-level runs for a given run_type, keyed by the case_id we tagged them with."""
    runs = client.list_runs(
        project_name=PROJECT,
        filter=f'and(has(tags, "{run_type}"), has(tags, "run-{benchmark_run_id}"))',
        is_root=True,
    )
    return {
        run.metadata["case_id"]: run
        for run in runs
        if run.metadata.get("case_id") is not None
    }


def _trace_breakdown(client: Client, run) -> dict:
    """Walk a run's full trace tree, tallying LLM calls and tool calls by name."""
    llm_calls = 0
    tool_calls = Counter()
    for child in client.list_runs(project_name=PROJECT, trace_id=run.trace_id):
        if child.run_type == "llm":
            llm_calls += 1
        elif child.run_type == "tool":
            tool_calls[child.name] += 1

    latency_ms = None
    if run.end_time and run.start_time:
        latency_ms = round((run.end_time - run.start_time).total_seconds() * 1000)

    return {
        "llm_calls": llm_calls,
        "tool_calls": dict(tool_calls),
        "total_tool_calls": sum(tool_calls.values()),
        "latency_ms": latency_ms,
    }


def _avg(rows: list, picker) -> float | None:
    values = [picker(r) for r in rows if picker(r) is not None]
    return round(sum(values) / len(values), 2) if values else None


def _avg_pair(rows: list) -> dict:
    return {
        "matched_cases": len(rows),
        "baseline_avg": {
            "llm_calls": _avg(rows, lambda m: m["baseline"]["llm_calls"]),
            "tool_calls": _avg(rows, lambda m: m["baseline"]["total_tool_calls"]),
            "latency_ms": _avg(rows, lambda m: m["baseline"]["latency_ms"]),
        },
        "routed_avg": {
            "llm_calls": _avg(rows, lambda m: m["routed"]["llm_calls"]),
            "tool_calls": _avg(rows, lambda m: m["routed"]["total_tool_calls"]),
            "latency_ms": _avg(rows, lambda m: m["routed"]["latency_ms"]),
        },
    }


def _by_expected_route(rows: list) -> dict:
    return {
        route: _avg_pair([row for row in rows if row["expected_route"] == route])
        for route in ("simple", "research", "deep")
    }


def build_comparison(
    run_id: str,
) -> dict:
    client = Client()

    baseline_runs = _root_runs_by_case(client, "baseline", run_id)
    routed_runs = _root_runs_by_case(client, "benchmark", run_id)

    matched = []
    for case_id in sorted(set(baseline_runs) & set(routed_runs)):
        baseline_run = baseline_runs[case_id]
        routed_run = routed_runs[case_id]
        matched.append({
            "case_id": case_id,
            "expected_route": routed_run.metadata.get("expected_route"),
            "workflow": routed_run.metadata.get("workflow"),
            "evaluation_focus": routed_run.metadata.get("evaluation_focus"),
            "baseline": _trace_breakdown(client, baseline_run),
            "routed": _trace_breakdown(client, routed_run),
        })

    return {
        "run_id": run_id,
        **_avg_pair(matched),
        "stats_by_expected_complexity": _by_expected_route(matched),
        "matched": matched,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline and routed traces from LangSmith.")
    parser.add_argument(
        "--run-id",
        default=os.getenv("VAICATION_RUN_ID"),
        required=os.getenv("VAICATION_RUN_ID") is None,
        help="Shared comparison id used for both baseline.py and benchmark.py.",
    )
    args = parser.parse_args()

    print(json.dumps(build_comparison(args.run_id), indent=2))
