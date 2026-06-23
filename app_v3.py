"""
vAIcation — Enterprise Travel Intelligence API

Replaces the open-ended ReAct agent with a deterministic LangGraph pipeline:
a classifier decides up front how much live-web retrieval a query needs
(simple / research / deep), and the graph runs only the tools that tier
requires — each at most once. No redundant tool calls, no open-ended loops.

Readiness queries can also carry a TripContext: the booked trip as a
structured record. When present, the search is scoped to the trip — only
sources published since the booking date, biased toward the trip's own
carriers — so the answer is "what changed since you booked" rather than a
fresh generic search. In production, TripContext is populated from the
customer's TMC / booking system (Concur, Navan, Amex GBT, a GDS feed); here
it's a mock standing in for that upstream source.
"""
import json
import os
import re
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
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not NEBIUS_API_KEY:
    raise RuntimeError("NEBIUS_API_KEY not set")


# ---------------------------------------------------------------------------
# Tools and LLM — built once at startup and shared across requests.
# NOTE: only request-invariant settings go on the constructor (api_key,
# search_depth). Anything that varies per query — time window, domains,
# topic — is passed at .invoke() time instead, because a value set here would
# override the per-call value and freeze it for every request.
# ---------------------------------------------------------------------------
# max_results is set here (instantiation) because this tool rejects it per-call.
# 10 (vs the default 5) gives the deep tier enough sources to cover its many
# risk domains; it's harmless for narrow tiers, which use far fewer.
search_tool = TavilySearch(api_key=TAVILY_API_KEY, search_depth="advanced", max_results=10)
extract_tool = TavilyExtract(api_key=TAVILY_API_KEY)
crawl_tool = TavilyCrawl(api_key=TAVILY_API_KEY, limit=5)  # capped for cost control

# Synthesis stays on OpenAI GPT; classification moves to Nebius Token Factory
# so the hybrid version isolates classifier latency/cost while leaving answer
# quality on the same report model as app_v2.py.
SYNTH_MODEL = os.getenv("VAICATION_SYNTH_MODEL", "gpt-5-mini")
CLASSIFIER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Traces showed synthesis is the main latency cost, not Tavily retrieval.
# Keep the OpenAI report model on low reasoning effort; Nebius does not support
# reasoning_effort, so the classifier omits it.
llm = ChatOpenAI(model=SYNTH_MODEL, api_key=OPENAI_API_KEY, reasoning_effort="low")
classifier_base = ChatOpenAI(
    model=CLASSIFIER_MODEL,
    api_key=NEBIUS_API_KEY,
    base_url=NEBIUS_BASE_URL,
)


# ---------------------------------------------------------------------------
# Source maps for scoped readiness search
# ---------------------------------------------------------------------------
# Major wire services carry breaking travel-disruption news (strikes, closures,
# weather events). Safe to restrict to because they're high-volume — a domain
# filter on these won't starve results the way a niche-site filter would.
AUTHORITATIVE_NEWS_DOMAINS = ["reuters.com", "apnews.com", "bbc.com"]

# Carrier name -> official domain. Lets a readiness search prefer the airline's
# own status/operations pages for the trip actually booked.
CARRIER_DOMAINS = {
    "american airlines": "aa.com",
    "british airways": "britishairways.com",
    "delta": "delta.com",
    "united": "united.com",
    "lufthansa": "lufthansa.com",
    "air france": "airfrance.com",
    "klm": "klm.com",
    "emirates": "emirates.com",
    "singapore airlines": "singaporeair.com",
}

# NOTE: government travel advisories (travel.state.gov, gov.uk/foreign-travel-
# advice) are static reference pages, not "news", so they surface better via a
# targeted extract against a known advisory URL than via news search. Left as a
# follow-up for the extract node rather than forced into include_domains here.


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


class TripContext(BaseModel):
    """A booked trip, as a structured record.

    In production this is populated from the customer's travel management
    system (Concur/SAP, Navan, Amex GBT) or a GDS booking feed — vAIcation
    defines the shape it consumes; it does not own the booking data. `booked_at`
    is the baseline a readiness check diffs current conditions against.
    """

    booking_ref: str = Field(description="PNR / booking reference from the system of record")
    traveler: str = Field(description="Traveler name or employee identifier")
    origin: str = Field(description="Origin airport/city (IATA or name)")
    destination: str = Field(description="Destination airport/city (IATA or name)")
    depart_date: str = Field(description="Departure date, YYYY-MM-DD")
    return_date: str | None = Field(default=None, description="Return date, YYYY-MM-DD")
    carriers: list[str] = Field(default_factory=list, description="Booked airline(s)")
    hotel: str | None = Field(default=None, description="Booked accommodation, if any")
    booked_at: str = Field(description="Date the trip was booked, YYYY-MM-DD — the diff baseline")


class EvidenceRisk(BaseModel):
    """A grounded operational risk extracted from retrieved evidence."""

    category: Literal["transport", "safety", "weather", "health", "entry", "lodging", "event", "other"] = Field(
        description="Risk domain supported by the retrieved source"
    )
    severity: Literal["low", "medium", "high", "unknown"] = Field(
        description="Impact level supported by the evidence; use unknown if the source does not justify a level"
    )
    summary: str = Field(description="Concise risk statement grounded in the cited evidence")
    affected_leg: str | None = Field(
        default=None,
        description="Trip leg, carrier, location, or date affected; null if not established by the evidence"
    )
    evidence_url: str = Field(description="Source URL that supports this risk")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence based on source specificity, recency, and relevance"
    )


class EvidenceRecommendation(BaseModel):
    """A grounded recommendation, not an org-policy action assignment."""

    action: str = Field(description="Concrete traveler or travel-team action supported by the evidence")
    rationale: str = Field(description="Why this action follows from the cited risk/evidence")
    related_risk: str = Field(description="Short reference to the risk this recommendation addresses")
    evidence_url: str = Field(description="Source URL that supports the recommendation")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence based on source specificity, recency, and relevance"
    )


class TravelIntelligenceReport(BaseModel):
    """Structured report returned to enterprise callers.

    The LLM emits only evidence-derived facts and recommendations. Customer-
    specific policy fields such as owner, SLA, alert priority, and final trip
    status belong in a deterministic downstream policy layer.
    """

    summary: str = Field(description="Concise executive summary of the findings")
    key_findings: list[str] = Field(description="Bullet-point facts surfaced by retrieval")
    risks: list[EvidenceRisk] = Field(description="Evidence-linked operational risks identified")
    recommendations: list[EvidenceRecommendation] = Field(
        description="Evidence-linked actions, excluding org-specific ownership, priority, and SLA fields"
    )
    sources: list[str] = Field(description="URLs the findings are drawn from")


classifier_llm = classifier_base.with_structured_output(QueryClassification)
reporter_llm = llm.with_structured_output(TravelIntelligenceReport)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class TravelState(TypedDict, total=False):
    query: str
    workflow: Literal["brief", "readiness"]
    trip_context: dict | None  # booked-trip record (readiness only); None if not supplied
    query_type: str
    search_results: dict
    extract_results: dict
    crawl_results: dict
    report: dict


WORKFLOW_FOCUS = {
    "simple": (
        "You are a precise travel intelligence assistant. "
        "Answer the query directly and factually based only on the retrieved evidence. "
        "Be concise. Do not speculate beyond what the sources support. "
        "If the information is time-sensitive, note when it was last confirmed."
    ),
    "research_brief": (
        "You are producing a business travel planning brief. "
        "The traveler or travel manager needs to make a decision — destination choice, "
        "flight selection, hotel strategy, or logistics planning. "
        "Structure your findings to support that decision directly. "
        "Emphasize practical differences, cost signals, and friction points. "
        "Every recommendation must be specific and actionable, not general advice."
    ),
    "research_readiness": (
        "You are producing a travel disruption assessment for a business traveler. "
        "Focus on current conditions that could affect an upcoming trip. "
        "Identify specific disruptions — strikes, delays, cancellations, safety incidents, "
        "weather events — and assess their likelihood and impact. "
        "Prioritize recency: a disruption happening today matters more than a historical pattern. "
        "Every risk must have a corresponding recommended action."
    ),
    "deep_brief": (
        "You are producing an enterprise travel brief for a senior stakeholder. "
        "This covers a complex trip — multiple destinations, multiple travelers, "
        "or a high-stakes business event. "
        "Structure your findings by destination or leg of the trip. "
        "Cover logistics, safety, major events, visa friction, and travel timing. "
        "The output should be ready to share with an executive or travel manager "
        "without further editing. Be thorough but not verbose."
    ),
    "deep_readiness": (
        "You are producing an enterprise travel risk assessment for a trip that has already been booked. "
        "Your sole focus is what has changed or could change that affects whether this trip "
        "should proceed as planned. "
        "Assess risk across all relevant domains: transport disruptions, safety and security, "
        "weather, entry requirements, and operational logistics. "
        "Assign a severity level — low, medium, or high — to each identified risk. "
        "Weight recent developments more heavily than historical patterns. "
        "Surface the evidence a downstream policy layer would need to make a go/no-go decision, "
        "but do not assign final trip status yourself."
    ),
}


# ---------------------------------------------------------------------------
# Per-trip search scoping
# ---------------------------------------------------------------------------
def _carrier_domains(carriers: list[str]) -> list[str]:
    """Map booked carrier names to their official domains, skipping unknowns."""
    return [d for c in carriers if (d := CARRIER_DOMAINS.get(c.strip().lower()))]


def _readiness_search_params(state: TravelState) -> dict:
    """Build dynamic Tavily params for a readiness search, computed per request.

    Readiness is about "what's new", so we always bias toward recent news.
    Recency is set from the booking date when a trip is supplied; the *domain*
    strategy depends on the tier:

      - deep tier covers MANY risk domains at once (transport / safety / weather
        / health / entry). A hard include_domains filter there starves
        cross-domain coverage, which leaves gaps the model backfills from
        training knowledge — i.e. hallucinations. So deep stays broad and widens
        max_results instead of restricting domains.
      - narrow tiers (simple / research) are single-topic, so a domain filter
        cuts noise without starving the answer. There we bias to authoritative
        wires plus the trip's own carriers.
    """
    params: dict = {"topic": "news"}  # readiness == current events / disruptions
    trip = state.get("trip_context")
    is_deep = state.get("query_type") == "deep"

    # Recency: exact "since booking" window when we have a trip, else recent week.
    if trip and trip.get("booked_at"):
        params["start_date"] = trip["booked_at"]  # YYYY-MM-DD
    elif not trip:
        params["time_range"] = "week"

    # Deep covers many risk domains at once, so it stays broad — NO domain filter,
    # relying on the wider max_results (set at instantiation) for coverage. Narrow
    # tiers on a known trip filter to authoritative wires + the trip's carriers to
    # cut noise without starving a single-topic answer.
    if not is_deep and trip:
        domains = AUTHORITATIVE_NEWS_DOMAINS + _carrier_domains(trip.get("carriers", []))
        if domains:
            params["include_domains"] = domains

    return params


# ---------------------------------------------------------------------------
# Trip-aware query construction
# ---------------------------------------------------------------------------
# An under-specified readiness query ("what changed since we booked?") carries
# no route, carrier, or date terms, so search returns generic travel articles
# even though trip_context holds the specifics. We append the trip's own terms
# to the query text — but only the ones the user didn't already say — and broad
# risk-domain terms only on the deep tier (multi-domain by definition; narrow
# tiers stay scoped to the user's single topic).
#
# Deliberately kept basic: deterministic templating, no extra LLM call, in
# keeping with the routing layer's cost rationale. Two known, accepted
# simplifications, left as the documented "make it great" follow-up:
#   - dates go in as ISO ("2026-06-15"); news phrases them as "June 15", so the
#     date terms retrieve less than the route/carrier terms.
#   - `_missing` is a substring check, so "American" already in the query won't
#     suppress appending "American Airlines" — it may add a near-duplicate term.
def _missing(query: str, term: str | None) -> bool:
    """True if `term` is set and not already present in the query (case-insensitive)."""
    return bool(term) and term.lower() not in query.lower()


def _enriched_search_query(state: TravelState) -> str:
    """Append the booked trip's identity terms (and, on deep, risk domains) to a
    readiness query — only the terms the user hasn't already mentioned.

    No-op for the brief workflow or when no trip is supplied, so every non-trip
    path (including the entire benchmark set) searches exactly as before.
    """
    query = state["query"]
    trip = state.get("trip_context")
    if state["workflow"] != "readiness" or not trip:
        return query

    identity_terms = [
        trip.get("origin"),
        trip.get("destination"),
        trip.get("depart_date"),
        trip.get("return_date"),
        trip.get("hotel"),
        *trip.get("carriers", []),
    ]
    additions = [t for t in identity_terms if _missing(query, t)]

    # Broad risk-domain terms only on deep, which assesses many domains at once.
    if state.get("query_type") == "deep":
        additions.append(
            "travel disruptions airport delays cancellations strikes weather "
            "safety entry requirements"
        )

    return " ".join([query, *additions])


# Worked examples that anchor the slippery boundaries — especially research vs
# deep, where the classifier was flip-flopping. Giving the model a few labeled
# cases to pattern-match against makes borderline calls far more consistent than
# letting it decide cold each time.
#
# IMPORTANT: these are deliberately DISJOINT from the benchmark eval set. Reusing
# eval queries here would leak the test answers into the classifier and inflate
# the routing-agreement rate — the few-shot anchors and the eval cases must stay
# separate for the benchmark to mean anything.
CLASSIFIER_FEWSHOT = """\
Worked examples (query -> route, and why):

- "What time zone is Tokyo in?" -> simple
  One static fact; a single search answers it.
- "Is Gatwick airport operating normally today?" -> simple
  One place, one current-condition check; snippets are enough.
- "What are travelers saying about the new business lounge at Munich airport?" -> research
  One topic, but needs synthesis across several pages (search + extract).
- "Compare Marriott vs Hilton for a business stay in Berlin on price and location." -> research
  A 2-3 option comparison needing page content, but a single topic. Not deep.
- "What disruptions are affecting Frankfurt airport this week?" -> research
  One airport, one synthesis pass. Sounds urgent, but it's single-domain.
- "Monitor every risk for a round-trip to Frankfurt next week: weather, strikes, safety, entry rules." -> deep
  Several risk domains at once on a whole trip (search + extract + crawl).
- "Should the team proceed with the Mumbai offsite given current conditions?" -> deep
  A go/no-go decision spanning safety, transport, and logistics together.

Decisive rule: research vs deep is about SCOPE, not how serious it sounds. One
topic or one place stays research; several risk domains together, or an explicit
go/no-go on a whole trip, is deep."""


# ---------------------------------------------------------------------------
# Nodes — each tool runs at most once, only on the path that needs it
# ---------------------------------------------------------------------------
def classify(state: TravelState) -> dict:
    """Decide, once and up front, how much retrieval this query warrants.

    If a route was forced (state already carries query_type), honor it and skip
    the LLM call. This makes the route deterministic for testing and clean
    before/after comparisons, instead of fighting classifier nondeterminism on
    boundary queries.
    """
    if state.get("query_type"):
        return {"query_type": state["query_type"]}

    trip = state.get("trip_context")
    trip_hint = (
        f"\nBooked trip: {trip['origin']} -> {trip['destination']}, "
        f"carriers {trip.get('carriers') or 'n/a'}, booked {trip.get('booked_at')}."
        if trip else ""
    )
    classification = classifier_llm.invoke(
        f"{CLASSIFIER_FEWSHOT}\n\n"
        f"Now classify this query using the same rule.\n"
        f"Workflow: {state['workflow']}{trip_hint}\n"
        f"Query:\n\n{state['query']}"
    )
    return {"query_type": classification.query_type}


def run_search(state: TravelState) -> dict:
    """Every path runs exactly one search — the cheapest read on 'what's true now'.

    Params are assembled at call time (not baked into the tool) so each request
    gets the scoping its workflow and trip warrant. The query text itself is
    enriched from the trip on readiness so an under-specified question still
    retrieves trip-specific sources (see _enriched_search_query).
    """
    params = {"query": _enriched_search_query(state)}
    if state["workflow"] == "readiness":
        params.update(_readiness_search_params(state))
    return {"search_results": search_tool.invoke(params)}


_STOPWORDS = {
    "about", "across", "affect", "affecting", "after", "against", "airport",
    "already", "also", "business", "brief", "build", "check", "conditions",
    "considering", "could", "current", "does", "every", "executive", "focus",
    "from", "given", "have", "including", "into", "major", "monitor", "next",
    "planned", "readiness", "right", "short", "should", "since", "team",
    "this", "travel", "traveler", "travelers", "trip", "week", "what", "when",
    "where", "with", "would",
}

_RISK_TERMS = {
    "advisory", "advisories", "cancellation", "cancellations", "delay",
    "delays", "demand", "entry", "health", "hotel", "logistics", "protest",
    "protests", "safety", "security", "strike", "strikes", "transport",
    "visa", "weather",
}

_LOCATION_ALIASES = {
    "new york": ["new york", "nyc", "jfk", "lga", "ewr"],
    "jfk": ["jfk", "new york", "nyc"],
    "lga": ["lga", "new york", "nyc"],
    "ewr": ["ewr", "newark", "new york", "nyc"],
    "london": ["london", "lhr", "heathrow", "gatwick", "city airport", "uk", "united kingdom"],
    "lhr": ["lhr", "heathrow", "london", "uk", "united kingdom"],
    "lgw": ["lgw", "gatwick", "london", "uk", "united kingdom"],
    "austin": ["austin", "aus", "austin-bergstrom"],
    "aus": ["aus", "austin", "austin-bergstrom"],
    "tokyo": ["tokyo", "hnd", "haneda", "nrt", "narita"],
    "hnd": ["hnd", "haneda", "tokyo"],
    "nrt": ["nrt", "narita", "tokyo"],
    "paris": ["paris", "cdg", "orly"],
    "cdg": ["cdg", "paris"],
}

_MONTHS = {
    "01": "january", "02": "february", "03": "march", "04": "april",
    "05": "may", "06": "june", "07": "july", "08": "august",
    "09": "september", "10": "october", "11": "november", "12": "december",
}


def _normalize(text: str) -> str:
    """Lowercase text with punctuation collapsed so cheap matching is stable."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _term_variants(term: str | None) -> list[str]:
    """Return simple aliases for a route/location term."""
    if not term:
        return []
    normalized = _normalize(term)
    variants = [normalized]
    variants.extend(_LOCATION_ALIASES.get(normalized, []))
    return list(dict.fromkeys(v for v in variants if v))


def _date_terms(date_value: str | None) -> list[str]:
    """Return ISO and month-name forms for a YYYY-MM-DD-ish date string."""
    if not date_value:
        return []
    terms = [date_value]
    parts = date_value.split("-")
    if len(parts) >= 2 and (month := _MONTHS.get(parts[1])):
        terms.append(month)
    return terms


def _relevance_terms(state: TravelState) -> dict[str, list[str]]:
    """Build deterministic signals used to rank retrieved evidence.

    This is intentionally lexical rather than LLM-based: the goal is to prefer
    sources that mention the trip/query surface area before synthesis, without
    adding another variable-cost reasoning step.
    """
    query = state.get("query", "")
    query_norm = _normalize(query)
    trip = state.get("trip_context") or {}

    location_terms: list[str] = []
    for field in ("origin", "destination", "hotel"):
        location_terms.extend(_term_variants(trip.get(field)))

    carrier_terms: list[str] = []
    for carrier in trip.get("carriers", []):
        carrier_terms.extend(_term_variants(carrier))

    date_terms: list[str] = []
    for field in ("depart_date", "return_date", "booked_at"):
        date_terms.extend(_date_terms(trip.get(field)))

    risk_terms = [term for term in _RISK_TERMS if term in query_norm]
    if state.get("query_type") == "deep" and state.get("workflow") == "readiness":
        risk_terms.extend(["delay", "cancellation", "strike", "weather", "safety", "entry"])

    query_terms = [
        word for word in re.findall(r"[a-z0-9]+", query_norm)
        if len(word) > 3 and word not in _STOPWORDS
    ]

    return {
        "location": list(dict.fromkeys(location_terms)),
        "carrier": list(dict.fromkeys(carrier_terms)),
        "date": list(dict.fromkeys(_normalize(t) for t in date_terms)),
        "risk": list(dict.fromkeys(risk_terms)),
        "query": list(dict.fromkeys(query_terms)),
    }


def _text_for_relevance(item) -> str:
    """Flatten the fields worth scoring from a Tavily result or evidence block."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item.get(field, ""))
        for field in ("url", "title", "content", "raw_content")
        if item.get(field)
    )


def _score_relevance(item, terms: dict[str, list[str]]) -> int:
    """Weighted lexical score: trip identity beats generic query overlap."""
    text = _normalize(_text_for_relevance(item))
    if not text:
        return 0

    score = 0
    score += 4 * sum(1 for term in terms["location"] if term in text)
    score += 4 * sum(1 for term in terms["carrier"] if term in text)
    score += 2 * sum(1 for term in terms["date"] if term in text)
    score += 2 * sum(1 for term in terms["risk"] if term in text)
    score += min(6, sum(1 for term in terms["query"] if term in text))
    return score


def _rank_results(results, state: TravelState):
    """Prefer source items with stronger query/trip overlap; keep weak broad sets."""
    if isinstance(results, str):
        results = [results]
    items = list(results or [])
    if not items:
        return []

    terms = _relevance_terms(state)
    scored = [(_score_relevance(item, terms), index, item) for index, item in enumerate(items)]
    if not any(score for score, _, _ in scored):
        return items
    return [item for score, _, item in sorted(scored, key=lambda row: (-row[0], row[1])) if score > 0]


def _result_urls(state: TravelState) -> list[str]:
    """URLs from the ranked search step, skipping any results missing one."""
    results = _rank_results(state["search_results"].get("results", []), state)
    return [u for r in results if isinstance(r, dict) and (u := r.get("url"))]


def run_extract(state: TravelState) -> dict:
    """Pull full page content for the top search hits (research and deep paths)."""
    urls = _result_urls(state)[:3]
    return {"extract_results": extract_tool.invoke({"urls": urls}) if urls else {}}


def run_crawl(state: TravelState) -> dict:
    """Crawl the most relevant site for comprehensive coverage (deep path only).

    Note: currently targets the top-ranked search result URL. A production
    implementation would apply source quality scoring before selecting
    the crawl target.
    """
    urls = _result_urls(state)
    return {"crawl_results": crawl_tool.invoke({"url": urls[0]}) if urls else {}}


# Evidence hygiene caps. Search "content" is a clean snippet, so it needs little.
# Extract/crawl "raw_content" is the full page including nav bars, related-story
# sidebars, and ads — capping per page keeps the article body and drops most of
# that trailing junk, which was both polluting context and tripping the
# hallucination judge.
_SNIPPET_CAP = 800       # per clean search snippet
_PAGE_CAP = 2500         # per extracted/crawled page
_MAX_PAGES = 3           # extract/crawl pages fed to synthesis
_EVIDENCE_CAP = 12000    # overall ceiling on the evidence block


def _result_items(payload):
    """Return result items from a Tavily payload, preserving string evidence."""
    if isinstance(payload, dict):
        results = payload.get("results")
        if results is None:
            return [payload]
        return results if isinstance(results, list) else [results]
    return payload


def _format_results(results, cap: int, limit: int | None = None, state: TravelState | None = None) -> list[str]:
    """Pull only (url, main text) from each result, capped — never the raw dict.

    Prefers the clean `content` snippet; falls back to `raw_content` (full page)
    when that's all a tool returns, trimmed to `cap` to shed sidebar/nav noise.
    """
    if state is not None:
        results = _rank_results(results, state)
    elif isinstance(results, str):
        results = [results]

    blocks = []
    for r in (results or [])[: limit or len(results or [])]:
        if isinstance(r, str):
            text = r.strip()
            if text:
                blocks.append(f"[no url]\n{text[:cap]}")
            continue
        if not isinstance(r, dict):
            continue
        url = r.get("url", "")
        text = (r.get("content") or r.get("raw_content") or "").strip()
        if text:
            blocks.append(f"[{url}]\n{text[:cap]}")
    return blocks


def synthesize(state: TravelState) -> dict:
    """Turn whatever evidence was gathered into the final structured report."""
    parts: list[str] = []
    parts += _format_results(_result_items(state.get("search_results", {})), _SNIPPET_CAP, state=state)
    parts += _format_results(_result_items(state.get("extract_results", {})), _PAGE_CAP, _MAX_PAGES, state)
    parts += _format_results(_result_items(state.get("crawl_results", {})), _PAGE_CAP, _MAX_PAGES, state)
    evidence = "\n\n".join(parts)[:_EVIDENCE_CAP]

    # When a booked trip is present, hand the model the baseline explicitly so it
    # assesses *change since booking* rather than reporting generic conditions.
    trip = state.get("trip_context")
    trip_block = ""
    if trip:
        trip_block = (
            "\n\nBOOKED TRIP (the baseline to assess against):\n"
            f"{json.dumps(trip, indent=2)}\n"
            f"Assess what has changed or newly emerged since this trip was booked "
            f"({trip.get('booked_at')}) that affects whether it should proceed as planned."
        )

    focus_key = state["query_type"] if state["query_type"] == "simple" else f"{state['query_type']}_{state['workflow']}"
    report = reporter_llm.invoke(
        f"{WORKFLOW_FOCUS[focus_key]}\n\n"
        f"Traveler query: {state['query']}{trip_block}\n\n"
        f"Evidence gathered from live web retrieval:\n{evidence}\n\n"
        "Produce a structured report grounded only in this evidence.\n"
        "GROUNDING RULES — follow strictly:\n"
        "- Use only evidence that directly matches the requested location, route, "
        "carrier, date, or risk domain. Ignore unrelated travel sources, even if "
        "they are otherwise factual. Only include risks and recommendations tied "
        "to that matching evidence. If no relevant evidence is retrieved, say so "
        "clearly rather than filling the report with generic or off-topic risks.\n"
        "- Every claim, risk, and severity rating must be traceable to a specific "
        "item in the evidence above. Do not add facts from prior knowledge.\n"
        "- For every risk and recommendation object, set evidence_url to the exact "
        "URL in the evidence block that supports it. If no URL supports the object, "
        "do not include the object.\n"
        "- Fill only evidence-derived operational fields. Do not invent business "
        "policy fields such as owner, SLA, alert priority, escalation path, or "
        "final trip status; those belong to a downstream deterministic policy layer.\n"
        "- Do not generalize beyond what a source states (e.g. if a source "
        "describes screening in one country, do not write 'some countries').\n"
        "- Do not infer cause and effect. If a source reports two facts, do not "
        "claim one caused, triggered, or prompted the other unless the source "
        "says so explicitly.\n"
        "- Only assign severity low, medium, or high when the source explicitly "
        "describes impact, disruption, risk level, cancellations, delays, "
        "restrictions, violence, weather warnings, or operational consequences. "
        "If the evidence is indirect, generic, historical, trend-based, or merely "
        "adjacent to the trip, either omit the risk or set severity to unknown.\n"
        "- Do not turn general travel trends into trip-specific risks. Do not "
        "claim event overlap, hotel-demand pressure, safety spillover, border "
        "delays, or transport disruption unless the source explicitly states "
        "that impact for the relevant place, route, date, or carrier.\n"
        "- Recommendations may use only the booked trip details provided and the "
        "evidence above. Do not assume traveler count, party composition, employer "
        "or agency procedures, or any logistic detail that was not given.\n"
        "- If the evidence does not cover a relevant risk domain (transport, "
        "safety, weather, health, entry requirements), say so explicitly — e.g. "
        "'No retrieved source addresses weather for this period' — rather than "
        "inferring or supplying general expectations.\n"
        "- Use cautious wording: 'retrieved evidence reports...' or 'no retrieved "
        "source establishes...' instead of unsupported phrases such as likely, "
        "could cause, may affect, or expected to.\n"
        "- A shorter report that omits weak claims is better than a complete-"
        "looking one that upgrades weak evidence into operational conclusions.\n"
        "- Be concise: at most 3 key findings, 3 risks, and 3 recommendations. "
        "State each point once; do not pad or repeat."
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
# Sample trip — a mock standing in for a record the customer's TMC/booking
# system would supply in production. Used for the /travel/readiness demo.
# ---------------------------------------------------------------------------
SAMPLE_TRIP = {
    "booking_ref": "PNR-7QF4K2",
    "traveler": "J. Okafor (emp #44218)",
    "origin": "JFK",
    "destination": "LHR",
    "depart_date": "2026-06-15",
    "return_date": "2026-06-20",
    "carriers": ["British Airways", "American Airlines"],
    "hotel": "Sofitel London Heathrow",
    "booked_at": "2026-05-20",
}


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
    trip: TripContext | None = None  # optional booked-trip context (readiness)


async def _run(
    query: str,
    workflow: Literal["brief", "readiness"],
    trip: TripContext | None = None,
) -> dict:
    try:
        result = await travel_graph.ainvoke({
            "query": query,
            "workflow": workflow,
            "trip_context": trip.model_dump() if trip else None,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query": query,
        "workflow": workflow,
        "trip_context": trip.model_dump() if trip else None,
        "classification": result["query_type"],
        "report": result["report"],
    }


@app.post("/travel/brief")
async def travel_brief(request: TravelQuery):
    """Trip-planning mode — help an org plan a trip on current, grounded information."""
    return await _run(request.query, "brief")


@app.post("/travel/readiness")
async def travel_readiness(request: TravelQuery):
    """Trip-monitoring mode — surface what changed since booking and what to do about it.

    Pass `trip` to scope the check to a booked itinerary (search is filtered to
    sources published since `booked_at` and biased to the trip's carriers). Omit
    it for a generic readiness check on a recent window.
    """
    return await _run(request.query, "readiness", request.trip)


@app.post("/eval/run")
async def eval_run():
    """Run the benchmark suite against the live graph and report routing agreement rate.

    Synchronous and slow by design (25 LLM + Tavily round trips) — this is a
    development/regression tool, not a high-traffic endpoint. Each case is
    tagged in LangSmith under run_type="benchmark" for trace inspection.
    """
    from benchmark import run_benchmark

    return await run_benchmark(travel_graph)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
