"""
vAIcation — Enterprise Travel Intelligence API

Replaces the open-ended ReAct agent with a deterministic LangGraph pipeline:
a classifier decides up front how much live-web retrieval a query needs
(simple / research / deep), and the graph runs only the tools that tier
requires — each at most once. No redundant tool calls, no open-ended loops.
"""
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")


# ---------------------------------------------------------------------------
# Tools and LLM — built once at startup and shared across requests
# ---------------------------------------------------------------------------
search_tool = TavilySearch(api_key=TAVILY_API_KEY, search_depth="advanced")
extract_tool = TavilyExtract(api_key=TAVILY_API_KEY)
crawl_tool = TavilyCrawl(api_key=TAVILY_API_KEY, limit=5)  # capped for cost control

llm = ChatOpenAI(model="gpt-5-mini", api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
class QueryClassification(BaseModel):
    """Routing decision: how much live retrieval a query needs."""

    query_type: Literal["simple", "research", "deep"] = Field(
        description=(
            "Choose the LEAST expensive retrieval tier that can still produce "
            "a complete, source-grounded answer:\n\n"
            "'simple': one search call is enough. Use for narrow factual lookups, "
            "single current-condition checks (weather, strikes, advisories), or "
            "anything answerable from search snippets alone.\n\n"
            "'research': search + extract. Use for comparisons between 2-3 options, "
            "synthesis across multiple sources, or planning tasks that need full "
            "page content from specific URLs.\n\n"
            "'deep': search + extract + crawl. Use ONLY when the query requires "
            "monitoring across MULTIPLE risk domains simultaneously (safety AND "
            "transport AND weather AND entry requirements), OR when the user needs "
            "a comprehensive go/no-go assessment, OR when a full site needs to be "
            "mapped for complete coverage. Single-topic queries never need deep "
            "even if they sound complex."
        )
    )


class TravelIntelligenceReport(BaseModel):
    """Structured report returned to enterprise callers."""

    summary: str = Field(description="Concise executive summary of the findings")
    key_findings: list[str] = Field(description="Bullet-point facts surfaced by retrieval")
    risks: list[str] = Field(description="Operational risks or disruptions identified")
    recommendations: list[str] = Field(description="Concrete actions the traveler/org should take")
    sources: list[str] = Field(description="URLs the findings are drawn from")


classifier_llm = llm.with_structured_output(QueryClassification)
reporter_llm = llm.with_structured_output(TravelIntelligenceReport)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class TravelState(TypedDict):
    query: str
    workflow: Literal["brief", "readiness"]
    query_type: str
    search_results: dict
    extract_results: dict
    crawl_results: dict
    report: dict


WORKFLOW_FOCUS = {
    "brief": (
        "You are producing a trip-planning brief for a business traveler. "
        "Focus on what they need to know to plan the trip with confidence."
    ),
    "readiness": (
        "You are producing a travel-readiness report. Focus on what has CHANGED "
        "since the trip was likely booked, and what risks or disruptions could "
        "affect the traveler before or during the trip."
    ),
}


# ---------------------------------------------------------------------------
# Nodes — each tool runs at most once, only on the path that needs it
# ---------------------------------------------------------------------------
def classify(state: TravelState) -> dict:
    """Decide, once and up front, how much retrieval this query warrants."""
    classification = classifier_llm.invoke(
        f"Workflow: {state['workflow']}\n"
        f"Classify this travel query for routing purposes:\n\n{state['query']}"
    )
    return {"query_type": classification.query_type}


def run_search(state: TravelState) -> dict:
    """Every path runs exactly one search — the cheapest read on 'what's true now'."""
    return {"search_results": search_tool.invoke({"query": state["query"]})}


def _result_urls(state: TravelState) -> list[str]:
    """URLs from the search step, skipping any results missing one."""
    return [u for r in state["search_results"].get("results", []) if (u := r.get("url"))]


def run_extract(state: TravelState) -> dict:
    """Pull full page content for the top search hits (research and deep paths)."""
    urls = _result_urls(state)[:3]
    return {"extract_results": extract_tool.invoke({"urls": urls}) if urls else {}}


def run_crawl(state: TravelState) -> dict:
    """Crawl the most relevant site for comprehensive coverage (deep path only)."""
    urls = _result_urls(state)
    return {"crawl_results": crawl_tool.invoke({"url": urls[0]}) if urls else {}}


def _truncate(content: str, max_chars: int = 15000) -> str:
    return content[:max_chars] if len(content) > max_chars else content


def synthesize(state: TravelState) -> dict:
    """Turn whatever evidence was gathered into the final structured report."""
    evidence = "\n\n".join(
        part for part in (
            f"SEARCH RESULTS:\n{_truncate(str(state.get('search_results', '')))}",
            f"EXTRACTED PAGE CONTENT:\n{_truncate(str(state['extract_results']))}" if state.get("extract_results") else "",
            f"CRAWLED SITE CONTENT:\n{_truncate(str(state['crawl_results']))}" if state.get("crawl_results") else "",
        )
        if part
    )

    report = reporter_llm.invoke(
        f"{WORKFLOW_FOCUS[state['workflow']]}\n\n"
        f"Traveler query: {state['query']}\n\n"
        f"Evidence gathered from live web retrieval:\n{evidence}\n\n"
        "Produce a structured report grounded only in this evidence."
    )
    return {"report": report.model_dump()}


# ---------------------------------------------------------------------------
# Graph wiring — deterministic routing, no LLM-driven loops
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(TravelState)
    graph.add_node("classify", classify)
    graph.add_node("search", run_search)
    graph.add_node("extract", run_extract)
    graph.add_node("crawl", run_crawl)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "search")

    graph.add_conditional_edges(
        "search",
        lambda s: "extract" if s["query_type"] in ("research", "deep") else "synthesize",
        {"extract": "extract", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "extract",
        lambda s: "crawl" if s["query_type"] == "deep" else "synthesize",
        {"crawl": "crawl", "synthesize": "synthesize"},
    )
    graph.add_edge("crawl", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


travel_graph = build_graph()


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="vAIcation Travel Intelligence API")


@app.get("/health")
async def health():
    """Liveness check — confirms the service is up and configuration loaded."""
    return {"status": "ok"}


class TravelQuery(BaseModel):
    query: str


async def _run(query: str, workflow: Literal["brief", "readiness"]) -> dict:
    try:
        result = await travel_graph.ainvoke({"query": query, "workflow": workflow})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query": query,
        "workflow": workflow,
        "classification": result["query_type"],
        "report": result["report"],
    }


@app.post("/travel/brief")
async def travel_brief(request: TravelQuery):
    """Trip-planning mode — help an org plan a trip on current, grounded information."""
    return await _run(request.query, "brief")


@app.post("/travel/readiness")
async def travel_readiness(request: TravelQuery):
    """Trip-monitoring mode — surface what changed since booking and what to do about it."""
    return await _run(request.query, "readiness")


@app.post("/eval/run")
async def eval_run():
    """Run the benchmark suite against the live graph and report routing accuracy.

    Synchronous and slow by design (26 LLM + Tavily round trips) — this is a
    development/regression tool, not a high-traffic endpoint. Each case is
    tagged in LangSmith under run_type="benchmark" for trace inspection.
    """
    from benchmark import run_benchmark

    return await run_benchmark(travel_graph)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
