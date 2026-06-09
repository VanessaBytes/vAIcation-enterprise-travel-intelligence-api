"""
vAIcation benchmark — runs a fixed set of travel queries through the routing
graph and checks whether each one took the retrieval path it should have.

The benchmark intentionally spans simple, research, and deep travel
intelligence tasks to evaluate whether vAIcation routes each query to the
least expensive retrieval path that can still produce a useful,
source-grounded enterprise answer — i.e. routing efficiency is the thing
under test, not just answer quality.

Each case carries an `expected_route` (simple / research / deep), a
`workflow` (brief / readiness), and an `evaluation_focus` note describing
what it probes — so results are gradable and self-documenting rather than
just logged. `run_benchmark` is shared by two callers:

  - CLI:  `python benchmark.py`        (prints a JSON summary)
  - HTTP: `POST /eval/run`             (see app_v2.py)

It takes the compiled graph as an argument rather than importing it, so this
module has no import-time dependency on app_v2 — app_v2 imports from here,
not the other way around.
"""
import asyncio
import argparse
import json
import os
import uuid

# expected_route: simple = search only, research = search+extract,
# deep = search+extract+crawl. Distribution (8/9/8 simple/research/deep,
# 10/15 brief/readiness) is deliberate: enough simple cases to prove the
# cost/latency win, weighted toward readiness since that's the
# differentiated workflow, without ballooning eval-run cost on crawl calls.
BENCHMARK = [
    # --- simple: one search call should be sufficient ---
    {"id": 1, "workflow": "brief", "expected_route": "simple",
     "evaluation_focus": "static factual lookup",
     "query": "What currency is used in Japan?"},
    {"id": 2, "workflow": "brief", "expected_route": "simple",
     "evaluation_focus": "simple travel fact",
     "query": "What is the typical nonstop flight time from New York to London?"},
    {"id": 3, "workflow": "brief", "expected_route": "simple",
     "evaluation_focus": "entry requirement lookup",
     "query": "Do US citizens need a visa for short business travel to Portugal?"},
    {"id": 4, "workflow": "brief", "expected_route": "simple",
     "evaluation_focus": "traveler logistics fact",
     "query": "What power plug type is used in Singapore?"},
    {"id": 5, "workflow": "readiness", "expected_route": "simple",
     "evaluation_focus": "current condition lookup",
     "query": "Are there current weather alerts for Paris today?"},
    {"id": 6, "workflow": "readiness", "expected_route": "simple",
     "evaluation_focus": "single disruption lookup",
     "query": "Is there a rail strike in the UK this week?"},
    {"id": 7, "workflow": "readiness", "expected_route": "simple",
     "evaluation_focus": "current advisory lookup",
     "query": "Are there active travel advisories for Mexico right now?"},
    {"id": 8, "workflow": "readiness", "expected_route": "simple",
     "evaluation_focus": "airport operations lookup",
     "query": "What is the current TSA wait time at Austin-Bergstrom airport?"},

    # --- research: comparison/synthesis needing search + extract ---
    {"id": 9, "workflow": "brief", "expected_route": "research",
     "evaluation_focus": "destination comparison",
     "query": "Compare Lisbon and Barcelona for a 3-day sales offsite in September, "
              "focusing on flight access, hotel availability, and business traveler logistics."},
    {"id": 10, "workflow": "brief", "expected_route": "research",
     "evaluation_focus": "business traveler logistics comparison",
     "query": "Which London airport is better for an executive arriving next week: "
              "Heathrow, Gatwick, or London City?"},
    {"id": 11, "workflow": "brief", "expected_route": "research",
     "evaluation_focus": "event-driven demand risk",
     "query": "What major conferences or citywide events could affect hotel availability in Austin in October?"},
    {"id": 12, "workflow": "brief", "expected_route": "research",
     "evaluation_focus": "flight option comparison",
     "query": "Compare business class options from New York to Tokyo in September "
              "for schedule reliability and alliance coverage."},
    {"id": 13, "workflow": "readiness", "expected_route": "research",
     "evaluation_focus": "airport disruption synthesis",
     "query": "What disruptions are affecting Heathrow airport this week?"},
    {"id": 14, "workflow": "readiness", "expected_route": "research",
     "evaluation_focus": "business traveler safety synthesis",
     "query": "Summarize current safety conditions in Nairobi for business travelers."},
    {"id": 15, "workflow": "readiness", "expected_route": "research",
     "evaluation_focus": "civil disruption synthesis",
     "query": "Are there major protests or demonstrations planned in Paris next week "
              "that could affect business travelers?"},
    {"id": 16, "workflow": "readiness", "expected_route": "research",
     "evaluation_focus": "airport operations synthesis",
     "query": "What are the current flight cancellation or delay issues affecting JFK airport?"},
    {"id": 17, "workflow": "readiness", "expected_route": "research",
     "evaluation_focus": "hotel demand intelligence",
     "query": "What hotel demand spikes are expected in Austin in October because of "
              "conferences, festivals, or major events?"},

    # --- deep: full itineraries / multi-domain readiness needing search + extract + crawl ---
    {"id": 18, "workflow": "brief", "expected_route": "deep",
     "evaluation_focus": "multi-city business travel planning",
     "query": "Build a business travel brief for a 5-day investor roadshow across London, Paris, "
              "and Frankfurt, including logistics, major events, safety considerations, and travel friction."},
    {"id": 19, "workflow": "brief", "expected_route": "deep",
     "evaluation_focus": "multi-destination enterprise planning",
     "query": "Compare Singapore, Dubai, and London as locations for a 4-day executive strategy retreat, "
              "including flight access, hotel demand, safety, visa friction, and local event conflicts."},
    {"id": 20, "workflow": "readiness", "expected_route": "deep",
     "evaluation_focus": "comprehensive trip readiness",
     "query": "Monitor all disruption risks for a New York to London business trip "
              "departing June 15 and returning June 20."},
    {"id": 21, "workflow": "readiness", "expected_route": "deep",
     "evaluation_focus": "country-level business travel readiness",
     "query": "Analyze all travel risks and logistics changes affecting a consulting team traveling to Brazil this month."},
    {"id": 22, "workflow": "readiness", "expected_route": "deep",
     "evaluation_focus": "regional change detection",
     "query": "What has changed in the last 30 days affecting corporate travel to the Middle East?"},
    {"id": 23, "workflow": "readiness", "expected_route": "deep",
     "evaluation_focus": "multi-domain executive travel readiness",
     "query": "Assess travel readiness for an executive trip to Tokyo next week, including airport "
              "operations, weather, transit reliability, safety, and entry requirements."},
    {"id": 24, "workflow": "readiness", "expected_route": "deep",
     "evaluation_focus": "go/no-go enterprise travel assessment",
     "query": "Evaluate whether a legal team should proceed with planned travel to Johannesburg next month, "
              "considering safety, transportation, health, and business continuity risks."},
    {"id": 25, "workflow": "readiness", "expected_route": "deep",
     "evaluation_focus": "post-booking change detection",
     "query": "What operational travel risks have emerged since booking a June 15-20 Austin to London trip "
              "for a senior executive?"},
]


def _new_run_id() -> str:
    """Shared comparison key for LangSmith traces from this benchmark session."""
    return str(uuid.uuid4())


async def run_benchmark(graph, benchmark_run_id: str | None = None) -> dict:
    """Run every benchmark case through `graph` and grade its routing decision.

    Routing agreement rate (`expected_route` vs. `actual_route`) is the one piece of
    analysis that's genuinely domain-specific — it requires comparing against
    labels from OUR dataset, so it has to be computed here. Everything else
    about how each run executed (latency, LLM/tool-call counts and breakdowns,
    status, output) is recorded automatically by LangSmith for every tagged
    run; compare.py pulls that straight from the trace store by run_type and
    case_id rather than re-measuring it locally.
    """
    benchmark_run_id = benchmark_run_id or _new_run_id()
    results = []
    for case in BENCHMARK:
        try:
            outcome = await graph.ainvoke(
                {"query": case["query"], "workflow": case["workflow"]},
                config={
                    "tags": [
                        "benchmark",
                        case["workflow"],
                        case["expected_route"],
                        f"run-{benchmark_run_id}",
                    ],
                    "metadata": {
                        "run_type": "benchmark",
                        "benchmark_version": "v1",
                        "benchmark_run_id": benchmark_run_id,
                        "case_id": case["id"],
                        "workflow": case["workflow"],
                        "expected_route": case["expected_route"],
                        "evaluation_focus": case["evaluation_focus"],
                    },
                },
            )
            actual_route = outcome["query_type"]
            results.append({
                "id": case["id"],
                "workflow": case["workflow"],
                "query": case["query"],
                "evaluation_focus": case["evaluation_focus"],
                "expected_route": case["expected_route"],
                "actual_route": actual_route,
                "status": "completed",
                "passed": actual_route == case["expected_route"],
            })
        except Exception as exc:
            results.append({
                "id": case["id"],
                "workflow": case["workflow"],
                "query": case["query"],
                "evaluation_focus": case["evaluation_focus"],
                "expected_route": case["expected_route"],
                "actual_route": None,
                "status": "error",
                "passed": False,
                "error": str(exc),
            })

    completed = [r for r in results if r["status"] == "completed"]
    passed = sum(r["passed"] for r in completed)
    return {
        "benchmark_run_id": benchmark_run_id,
        "total_cases": len(results),
        "completed": len(completed),
        "errored": len(results) - len(completed),
        "passed": passed,
        "failed": len(completed) - passed,
        "routing_agreement_rate": round(passed / len(completed), 2) if completed else 0,
        "next_step": ('Run `python compare.py` — it pulls latency, LLM-call, and '
                      'tool-call breakdowns for these runs (run_type="benchmark") '
                      'directly from LangSmith and lines them up against the '
                      'baseline runs by matching benchmark_run_id and case_id.'),
        "results": results,
    }


if __name__ == "__main__":
    from app_v2 import travel_graph

    parser = argparse.ArgumentParser(description="Run routed LangGraph benchmark traces.")
    parser.add_argument(
        "--run-id",
        default=os.getenv("VAICATION_RUN_ID"),
        help="Shared comparison id. Reuse this exact id for baseline.py and compare.py.",
    )
    args = parser.parse_args()

    summary = asyncio.run(run_benchmark(travel_graph, benchmark_run_id=args.run_id))
    print(json.dumps(summary, indent=2))
