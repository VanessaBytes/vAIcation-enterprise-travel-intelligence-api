# vAIcation — Enterprise Travel Intelligence API

vAIcation is an enterprise travel intelligence API that uses Tavily-powered
live web retrieval and LangSmith observability to help organizations plan,
validate, and continuously monitor business travel.

Unlike consumer itinerary planners, vAIcation focuses on **travel
readiness**: identifying what changed after a trip was booked, surfacing
operational risks, and recommending actions before disruptions affect
employees. It is not a vacation planner — it's a travel intelligence layer
for companies that need to know whether a trip is still viable.

## Technical Statement

See [TECHNICAL_STATEMENT.md](./TECHNICAL_STATEMENT.md) for the full
writeup — approach, design decisions, benchmark results, and production
considerations.

## Project Structure

```text
app.py                  # Original base app (ReAct agent) — baseline reference only
app_v2.py               # Improved routing architecture — main entry point
baseline.py             # Runs base app against benchmark sample, tags traces in LangSmith
benchmark.py            # Runs routing graph against full 25-case eval set
compare.py              # Pulls and compares baseline vs routed traces from LangSmith
.env.example            # Environment variable template — copy to .env and fill in keys
TECHNICAL_STATEMENT.md  # Full writeup — approach, decisions, results, limitations
```

## Approach

This system was built benchmark-first. Before writing any routing logic,
LangSmith tracing was added to the base app to establish what was actually
happening. The first trace showed a simple factual query triggering three
LLM calls, two search calls, and 22 seconds of latency — the agent looping
with no exit condition.

That measurement drove every subsequent decision: the routing architecture
was built to fix a measured problem, not a hypothesized one. A 25-case
evaluation dataset was created before the routing logic was finalized, so
improvement could be demonstrated rather than claimed.

Routing agreement rate improved from 71% on the first pass to 84% after classifier
prompt refinement. Context engineering reduced the problem further by giving
the synthesis step explicit success criteria per query type. LLM as a judge
evaluators running automatically on every trace validate output quality
independently of routing efficiency.

## Architecture

Incoming queries are routed through a deterministic [LangGraph](https://github.com/langchain-ai/langgraph)
pipeline rather than an open-ended ReAct agent. A classifier decides, once
and up front, how much live retrieval a query needs — then the graph runs
only the tools that tier requires, each at most once:

```
START → classify → search ──┬─→ synthesize                      (simple)
                            └─→ extract ──┬─→ synthesize         (research)
                                          └─→ crawl → synthesize (deep)
```

| Tier       | Tools run                  | Example                                              |
|------------|----------------------------|------------------------------------------------------|
| `simple`   | search                     | "What currency is used in Japan?"                    |
| `research` | search → extract           | "Compare business class options NYC → Tokyo"         |
| `deep`     | search → extract → crawl   | "Monitor disruption risks for a NYC → London trip"   |

The line between `research` and `deep` is *scope*, not topic — "What
disruptions are affecting Heathrow airport this week?" stays `research`
(one airport, one synthesis pass), while "Monitor all disruption risks for
a NYC → London trip" goes `deep` (a full round-trip's worth of
weather, transit, safety, and operational risk needs the broader crawl
pass to ground a comprehensive answer).

This replaces an earlier ReAct-agent version where the LLM decided which
tools to call turn by turn, producing redundant tool calls and 22+ second
latency even on simple factual queries. Routing the decision into graph
edges instead of LLM reasoning makes execution predictable, auditable, and
cheap to run at scale.

## Why Tavily

Travel intelligence has a freshness problem. Flight cancellations, rail
strikes, visa rule changes, travel advisories — these happen today, not six
months ago when an LLM was trained. A static model cannot answer "is my
trip still viable?" because it doesn't know what changed this week.

Tavily is purpose-built for this gap. Unlike generic search APIs that return
raw HTML full of ads, navigation menus, and noise, Tavily returns clean,
structured, LLM-ready content optimized for retrieval-augmented generation.
There is no parsing pipeline to maintain, no scraping infrastructure to
manage — just grounded, current web intelligence ready for an LLM to reason
over.

Tavily also provides three distinct retrieval modes that map directly to
vAIcation's routing tiers:

- **TavilySearch** — fast, broad web search. Ideal for current-condition
  lookups and single-topic queries. Called on every path.
- **TavilyExtract** — full page content from specific URLs. Used when search
  snippets are insufficient and deeper synthesis is needed.
- **TavilyCrawl** — maps and retrieves an entire site's content. Reserved
  for comprehensive multi-domain risk assessments where a single page is not
  enough.

This three-tool architecture is why the simple/research/deep routing tiers
exist. Each tier uses exactly the Tavily tools the query warrants — no more,
no less. A simple factual lookup never triggers a crawl. A multi-domain
risk assessment gets the full retrieval depth it needs.

At enterprise scale this matters economically. Routing tool selection
efficiently keeps cost per query predictable and low. Without routing, the
agent makes unpredictable tool selections — often over-retrieving on simple
queries and under-retrieving on complex ones. At 10,000 queries per day,
unpredictable retrieval depth means unpredictable cost — which at enterprise
scale is the difference between a viable and an unviable product.

## Observability — LangSmith

Every request is fully traced through LangSmith with zero additional code —
configured via environment variables only. Each trace captures every node
in the graph, every Tavily tool call, every LLM inference, latency at each
step, token usage, and cost.

LangSmith serves three roles in vAIcation:

- **Tracing** — full execution trace for every query, showing exactly which
  path the graph took and how long each step took
- **Observability** — aggregate dashboard showing latency distributions,
  error rates, token usage, and cost across all queries
- **Evaluation host** — runs the LLM as a judge evaluators automatically
  on every trace after the request completes

This is what makes the before/after comparison in `compare.py` possible —
LangSmith already recorded everything, so the comparison script just pulls
the data rather than re-measuring it.

## Endpoints

| Method | Path                | Purpose                                                          |
|--------|---------------------|------------------------------------------------------------------|
| POST   | `/travel/brief`     | Trip-planning mode — research and plan a trip before booking     |
| POST   | `/travel/readiness` | Trip-monitoring mode — what's changed since booking, what to do  |
| POST   | `/eval/run`         | Run the benchmark suite and report routing agreement rate        |
| GET    | `/health`           | Liveness check                                                   |

Both travel endpoints share the same routing graph; `workflow` only changes
the lens the final synthesis step applies (plan-the-trip vs.
what-changed-since-booking) and is passed to the classifier so routing can
account for it too.

### Roadmap: `/travel/monitor`

A continuous-surveillance endpoint — "keep watching this trip and tell me
when something changes," as opposed to `/travel/readiness`'s one-time
snapshot check. This needs trip identity and persisted state to diff against
on each recurring check, which is meaningfully more than a routing change —
left for a follow-up iteration once that design is settled.

## Setup

Requires **Python 3.10+** (the codebase uses union type syntax and built-in
generic type hints introduced in 3.10).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=vAIcation-enterprise-travel-intelligence-api
```

## Running

```bash
python app_v2.py
```

The API starts on `http://localhost:8000`.

## Usage

```bash
curl -X POST "http://localhost:8000/travel/brief" \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare business class options from New York to Tokyo in September"}'

curl -X POST "http://localhost:8000/travel/readiness" \
  -H "Content-Type: application/json" \
  -d '{"query": "Monitor all disruption risks for a New York to London trip departing June 15"}'
```

Both endpoints return a structured JSON report:

```json
{
  "query": "Is there a rail strike in the UK this week?",
  "workflow": "readiness",
  "classification": "simple",
  "report": {
    "summary": "No — retrieved sources report no major UK rail strikes this week.",
    "key_findings": [
      "StrikeTracker (June 2026): no major rail strikes reported in the UK today.",
      "Trainline (April 2026): no plans for industrial action currently."
    ],
    "risks": [
      "Local engineering works can still cause disruption even without a strike.",
      "Planned action could be announced after the date of these sources."
    ],
    "recommendations": [
      "Check National Rail Enquiries before travel for live status.",
      "Allow extra time and monitor operator messages for cancellations."
    ],
    "sources": [
      "https://striketracker.app/strikes-in-united-kingdom",
      "https://www.thetrainline.com/trains/great-britain/industrial-action"
    ]
  }
}
```

## Context Engineering

The synthesis step uses tier-aware and workflow-aware system prompts — five
distinct prompt configurations based on query complexity and workflow type
(`simple`, `research_brief`, `research_readiness`, `deep_brief`, `deep_readiness`).

Rather than a single generic instruction, each configuration gives the LLM
explicit success criteria for that specific type of query. A deep readiness
query gets a prompt that demands severity-rated risks and an explicit go/no-go
recommendation. A research planning query gets a prompt anchored to
decision-making between options. This is context engineering — deliberately
shaping what the model receives to improve output quality without changing
the model or adding latency.

## Baseline & Comparison

Run all three scripts with the same run ID to get a clean paired comparison.
Run IDs are arbitrary strings you choose — use something meaningful like `june-08-v1`.

```bash
python baseline.py --run-id june-08-v1
python benchmark.py --run-id june-08-v1
python compare.py --run-id june-08-v1
```

**baseline.py** replays a representative 9-case sample (3 per tier, both
workflows) through the original ReAct agent (`app.py`), tagging each run
`run_type="baseline"` with a matching `case_id`. This establishes the
"before" reference point.

**benchmark.py** runs the full 25-case evaluation set through the routed
graph, tagging each run `run_type="benchmark"` with a matching `case_id`.

**compare.py** pulls both sets of traces from LangSmith, matches them by
shared run ID and `case_id`, and prints one consolidated before/after report
covering the 9 matched pairs — latency, LLM-call counts, and tool-call
breakdowns.

Results from the june-08-v2 run:

| Metric          | Baseline (ReAct) | Routed (vAIcation) | Change  |
|-----------------|------------------|--------------------|---------|
| Avg LLM calls   | 3.0              | 2.0                | -33%    |
| Avg tool calls  | 3.22             | 2.0                | -38%    |
| Avg latency     | 31.4s            | 26.8s              | -15%    |

Research tier specifically: tool calls reduced by 63%, latency reduced by 31%.

## Benchmark

`benchmark.py` holds a 25-case evaluation set (8 simple / 9 research / 8 deep,
split 10 brief / 15 readiness) spanning all three routing tiers and both
workflows. Each case carries an `expected_route` and an `evaluation_focus`
note, so results are gradable and self-documenting rather than just logged.

Run it standalone:

```bash
python benchmark.py --run-id june-08-v1
```

Or trigger it over HTTP:

```bash
curl -X POST "http://localhost:8000/eval/run"
```

Either path scores **routing agreement rate** — the percentage of queries where
the classifier's routing decision matched the expected route label in the
benchmark dataset — and tags every run in LangSmith under
`run_type="benchmark"` with a matching `case_id`.

**Note:** the HTTP endpoint auto-generates a run ID returned in the response
JSON. Use that ID with `compare.py --run-id` to pull the matching traces.
Only the CLI path lets you pre-specify a run ID for clean pairing with a
baseline run.

Routing agreement rate across runs:

| Run         | Agreement Rate | Notes                                                        |
|-------------|----------------|--------------------------------------------------------------|
| june-08-v1  | 71%            | Baseline classifier                                          |
| june-08-v2  | 84%            | Improved classifier prompt                                   |
| june-08-v3  | 80%            | Context engineering added — within margin of LLM variability |

## Evaluation Framework — LLM as a Judge

Beyond routing agreement rate, vAIcation measures output quality through two
LangSmith LLM-as-a-judge evaluators running automatically on every trace:

- **Hallucination** — checks whether every claim in the output is supported
  by the Tavily evidence retrieved. Catches cases where the LLM fabricates
  information not present in the sources.
- **Answer Relevance** — checks whether the output actually addressed what
  was asked. Catches cases where the system produces a grounded response
  that answers a different question than the one posed.

Both evaluators are configured directly in LangSmith's evaluator UI using
gpt-5.5 as the judge model. They run asynchronously against every tagged
trace after each request completes — no additional code required, zero
latency added to the serving path. At enterprise scale, sampling rate can
be reduced from 100% to control evaluation cost while maintaining quality
visibility.

## Known Limitations

The current implementation is intentionally focused on routing discipline and
measurement, not a full enterprise travel platform. Three important production
gaps remain:

- The response schema uses prose lists for `risks` and `recommendations`; a
  production API should expose machine-readable `trip_status`, severity-rated
  issues, evidence, source URLs, and action ownership.
- The crawl node currently targets the first Tavily search result URL; a
  production implementation should score source quality and domain relevance
  before selecting a crawl target.
- `/travel/readiness` is a one-time readiness check. Continuous monitoring
  would require persisted trip state, scheduled re-checks, and change detection.

See [TECHNICAL_STATEMENT.md](./TECHNICAL_STATEMENT.md) for the fuller production
evolution path.
