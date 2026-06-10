"""
Unit tests for the vAIcation routing graph.

These tests do NOT call OpenAI or Tavily. They mock the classifier, the
reporter, and the three Tavily tools, then run the real compiled graph and
assert on its *mechanics*:

  1. Per-tier tool discipline — each tier runs exactly the tools it should,
     each at most once. This is the core claim the whole project rests on.
  2. Dynamic search scoping — readiness searches get the right Tavily params
     computed per request (topic/news, since-booking start_date, carrier
     domains), and brief searches stay plain.
  3. Trip context reaches synthesis — the booked trip is handed to the model
     as a baseline so the answer assesses change since booking.

Run:  pytest test_routing.py -v
"""
import os

# Satisfy app_v2's import-time key guard before importing it. No network calls
# are made by constructing the tools/LLM — these dummy values are never used.
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

import app_v2


# ---------------------------------------------------------------------------
# Fakes — stand in for the LLM calls and Tavily tools
# ---------------------------------------------------------------------------
class _Classification:
    def __init__(self, query_type):
        self.query_type = query_type


class FakeClassifier:
    """Returns a fixed routing decision so each tier can be tested in isolation."""

    def __init__(self, query_type):
        self._query_type = query_type
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return _Classification(self._query_type)


class _Report:
    def model_dump(self):
        return {
            "summary": "",
            "key_findings": [],
            "risks": [],
            "recommendations": [],
            "sources": [],
        }


class FakeReporter:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return _Report()


class FakeTool:
    """Stands in for a Tavily tool. Records the params each .invoke() got.

    We replace the whole tool object rather than patch its .invoke method,
    because the real tools are Pydantic models that intercept attribute
    assignment. The graph nodes look up `search_tool` etc. as module globals
    at call time, so swapping the global is enough.
    """

    def __init__(self, return_value):
        self._return_value = return_value
        self.calls = []

    def invoke(self, arg):
        self.calls.append(arg)
        return self._return_value

    @property
    def count(self):
        return len(self.calls)


SEARCH_RESULTS = {"results": [{"url": "https://a.example"}, {"url": "https://b.example"}]}


def _wire(monkeypatch, query_type):
    """Swap the graph's LLMs and tools for fakes; return them for assertions."""
    classifier = FakeClassifier(query_type)
    reporter = FakeReporter()
    search = FakeTool(SEARCH_RESULTS)
    extract = FakeTool({"pages": "extracted"})
    crawl = FakeTool({"site": "crawled"})

    monkeypatch.setattr(app_v2, "classifier_llm", classifier)
    monkeypatch.setattr(app_v2, "reporter_llm", reporter)
    monkeypatch.setattr(app_v2, "search_tool", search)
    monkeypatch.setattr(app_v2, "extract_tool", extract)
    monkeypatch.setattr(app_v2, "crawl_tool", crawl)
    return classifier, reporter, search, extract, crawl


SAMPLE_TRIP = {
    "booking_ref": "PNR-7QF4K2",
    "traveler": "J. Okafor",
    "origin": "JFK",
    "destination": "LHR",
    "depart_date": "2026-06-15",
    "return_date": "2026-06-20",
    "carriers": ["British Airways", "American Airlines"],
    "booked_at": "2026-05-20",
}


# ---------------------------------------------------------------------------
# 1. Per-tier tool discipline
# ---------------------------------------------------------------------------
def test_simple_tier_runs_search_only(monkeypatch):
    _, _, search, extract, crawl = _wire(monkeypatch, "simple")
    app_v2.travel_graph.invoke({"query": "q", "workflow": "brief", "trip_context": None})
    assert search.count == 1
    assert extract.count == 0
    assert crawl.count == 0


def test_research_tier_runs_search_and_extract(monkeypatch):
    _, _, search, extract, crawl = _wire(monkeypatch, "research")
    app_v2.travel_graph.invoke({"query": "q", "workflow": "brief", "trip_context": None})
    assert search.count == 1
    assert extract.count == 1
    assert crawl.count == 0


def test_deep_tier_runs_all_three_once(monkeypatch):
    _, _, search, extract, crawl = _wire(monkeypatch, "deep")
    app_v2.travel_graph.invoke({"query": "q", "workflow": "readiness", "trip_context": None})
    assert search.count == 1
    assert extract.count == 1
    assert crawl.count == 1


# ---------------------------------------------------------------------------
# 2. Dynamic search scoping (computed per request, not baked into the tool)
# ---------------------------------------------------------------------------
def test_readiness_with_trip_scopes_search(monkeypatch):
    _, _, search, _, _ = _wire(monkeypatch, "simple")
    app_v2.travel_graph.invoke(
        {"query": "q", "workflow": "readiness", "trip_context": SAMPLE_TRIP}
    )
    params = search.calls[0]
    assert params["topic"] == "news"
    assert params["start_date"] == "2026-05-20"          # exact since-booking window
    assert "britishairways.com" in params["include_domains"]
    assert "aa.com" in params["include_domains"]


def test_readiness_without_trip_falls_back_to_recent_window(monkeypatch):
    _, _, search, _, _ = _wire(monkeypatch, "simple")
    app_v2.travel_graph.invoke(
        {"query": "q", "workflow": "readiness", "trip_context": None}
    )
    params = search.calls[0]
    assert params["topic"] == "news"
    assert params["time_range"] == "week"
    assert "start_date" not in params


def test_brief_search_has_no_readiness_params(monkeypatch):
    _, _, search, _, _ = _wire(monkeypatch, "research")
    app_v2.travel_graph.invoke(
        {"query": "q", "workflow": "brief", "trip_context": None}
    )
    params = search.calls[0]
    assert params == {"query": "q"}  # plain query, no recency/domain scoping


def test_deep_readiness_stays_broad_not_domain_filtered(monkeypatch):
    # Deep covers many risk domains at once; a hard domain filter starves
    # coverage and was the root cause of the hallucinated claims. Deep should
    # widen results instead of restricting domains.
    _, _, search, _, _ = _wire(monkeypatch, "deep")
    app_v2.travel_graph.invoke(
        {"query": "q", "workflow": "readiness", "trip_context": SAMPLE_TRIP}
    )
    params = search.calls[0]
    assert params["topic"] == "news"
    assert params["start_date"] == "2026-05-20"      # recency still applies
    assert "include_domains" not in params           # NOT hard-filtered on deep
    # max_results is NOT passed per-call (this tool rejects it at invocation);
    # breadth comes from the instantiation-level max_results instead.
    assert "max_results" not in params


def test_unknown_carrier_is_skipped(monkeypatch):
    _, _, search, _, _ = _wire(monkeypatch, "simple")
    trip = {**SAMPLE_TRIP, "carriers": ["British Airways", "Some Regional Air"]}
    app_v2.travel_graph.invoke(
        {"query": "q", "workflow": "readiness", "trip_context": trip}
    )
    domains = search.calls[0]["include_domains"]
    assert "britishairways.com" in domains
    assert all("regional" not in d for d in domains)  # unmapped carrier dropped, no crash


# ---------------------------------------------------------------------------
# 3. Trip context reaches synthesis as a baseline
# ---------------------------------------------------------------------------
def test_trip_context_is_handed_to_synthesis(monkeypatch):
    _, reporter, *_ = _wire(monkeypatch, "simple")
    app_v2.travel_graph.invoke(
        {"query": "q", "workflow": "readiness", "trip_context": SAMPLE_TRIP}
    )
    prompt = reporter.prompts[0]
    assert "PNR-7QF4K2" in prompt           # the booked trip is in the prompt
    assert "2026-05-20" in prompt           # ...and the diff baseline date


def test_no_trip_means_no_baseline_block(monkeypatch):
    _, reporter, *_ = _wire(monkeypatch, "simple")
    app_v2.travel_graph.invoke(
        {"query": "q", "workflow": "readiness", "trip_context": None}
    )
    assert "BOOKED TRIP" not in reporter.prompts[0]


def test_synthesis_keeps_policy_fields_out_of_llm_output(monkeypatch):
    _, reporter, *_ = _wire(monkeypatch, "deep")
    app_v2.travel_graph.invoke(
        {"query": "q", "workflow": "readiness", "trip_context": SAMPLE_TRIP}
    )
    prompt = reporter.prompts[0]
    assert "evidence_url" in prompt
    assert "downstream deterministic policy layer" in prompt
    assert "Do not invent business policy fields" in prompt


def test_report_schema_uses_evidence_linked_operational_objects():
    report = app_v2.TravelIntelligenceReport(
        summary="One current transport risk found.",
        key_findings=["A source reports disruption affecting the booked route."],
        risks=[
            {
                "category": "transport",
                "severity": "medium",
                "summary": "Disruption may affect the outbound leg.",
                "affected_leg": "JFK-LHR",
                "evidence_url": "https://example.com/status",
                "confidence": "medium",
            }
        ],
        recommendations=[
            {
                "action": "Monitor the carrier status page before departure.",
                "rationale": "The cited source reports a route-relevant disruption.",
                "related_risk": "Disruption may affect the outbound leg.",
                "evidence_url": "https://example.com/status",
                "confidence": "medium",
            }
        ],
        sources=["https://example.com/status"],
    )

    dumped = report.model_dump()
    assert dumped["risks"][0]["evidence_url"] == "https://example.com/status"
    assert dumped["risks"][0]["category"] == "transport"
    assert "owner" not in dumped["recommendations"][0]
    assert "deadline" not in dumped["recommendations"][0]
