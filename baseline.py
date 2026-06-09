"""
vAIcation baseline — runs a representative slice of the benchmark set
against the *original* ReAct agent (app.py) to establish "before" numbers
for the "22+ second latency, redundant tool calls" claim the routed
rebuild fixes.

The ReAct agent has no route classifier and doesn't deterministically choose
a retrieval depth, so it is NOT scored against `expected_route` — that field
just rides along so the post-router benchmark (benchmark.py / /eval/run) can
be compared against these numbers later.

This script measures only whether each case completed or errored. Latency,
LLM-call counts, and tool-call breakdowns are recorded automatically by
LangSmith's tracer for every tagged run — compare.py pulls that data
directly from LangSmith rather than re-measuring it locally.

Every run here is tagged run_type="baseline" with a matching case_id, so
`compare.py` can pull the LLM/tool-call breakdown for these runs straight
from LangSmith and line them up against the routed system's runs
(run_type="benchmark") for the same queries.

Runs a 9-case sample (3 per tier, spanning both workflows) rather than the
full 25: at ~22s+ per query with no tool-call discipline, the full set would
take ~10 minutes and cost meaningfully more for a number that's purely a
"before" reference point.

Run: `python baseline.py`
"""
import asyncio
import argparse
import json
import os
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app import WebAgent
from benchmark import BENCHMARK

load_dotenv()

BASELINE_SAMPLE_IDS = {1, 3, 5, 9, 13, 17, 18, 20, 23}
BASELINE_CASES = [case for case in BENCHMARK if case["id"] in BASELINE_SAMPLE_IDS]


def _new_run_id() -> str:
    """Shared comparison key for LangSmith traces from this baseline session."""
    return str(uuid.uuid4())


def _build_baseline_agent():
    llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"))
    return WebAgent().build_graph(
        api_key=os.getenv("TAVILY_API_KEY"),
        llm=llm,
        prompt="You are a helpful web research assistant.",
    )


async def run_baseline(agent, benchmark_run_id: str | None = None) -> dict:
    """Drive each sampled case through the baseline agent, tagged for LangSmith.

    No metrics are computed here. LangSmith already records, for every tagged
    run: latency (start_time/end_time), completion status (error), output
    (outputs), and the full LLM/tool-call trace. Re-measuring any of that
    locally would just be a worse copy of what the tracer already has —
    compare.py pulls the real thing straight from LangSmith by run_type and
    case_id. This function's only job is to run the queries and tag them
    correctly so they're findable there.
    """
    benchmark_run_id = benchmark_run_id or _new_run_id()
    errors = []
    for case in BASELINE_CASES:
        try:
            await agent.ainvoke(
                {"messages": [("user", case["query"])]},
                config={
                    "tags": ["baseline", case["workflow"], case["expected_route"], f"run-{benchmark_run_id}"],
                    "metadata": {
                        "run_type": "baseline",
                        "benchmark_version": "v1",
                        "benchmark_run_id": benchmark_run_id,
                        "case_id": case["id"],
                        "workflow": case["workflow"],
                        "expected_route": case["expected_route"],
                        "evaluation_focus": case["evaluation_focus"],
                    },
                },
            )
            print(f"[{case['id']}] done — {case['query'][:70]}")
        except Exception as exc:
            errors.append({"id": case["id"], "query": case["query"], "error": str(exc)})
            print(f"[{case['id']}] ERROR — {exc}")

    return {
        "benchmark_run_id": benchmark_run_id,
        "total_cases": len(BASELINE_CASES),
        "errors": errors,
        "next_step": ('Run `python compare.py` — it pulls latency, LLM-call, and '
                      'tool-call breakdowns for these runs (run_type="baseline") '
                      'directly from LangSmith and lines them up against the '
                      'routed benchmark by matching benchmark_run_id and case_id.'),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline ReAct traces for comparison.")
    parser.add_argument(
        "--run-id",
        default=os.getenv("VAICATION_RUN_ID"),
        help="Shared comparison id. Reuse this exact id for benchmark.py and compare.py.",
    )
    args = parser.parse_args()

    agent = _build_baseline_agent()
    summary = asyncio.run(run_baseline(agent, benchmark_run_id=args.run_id))
    print(json.dumps(summary, indent=2))
