# vAIcation — Enterprise Travel Intelligence API

vAIcation is an enterprise travel intelligence API that uses Tavily-powered
live web retrieval and LangSmith observability to help organizations plan,
validate, and continuously monitor business travel.

Unlike consumer itinerary planners, vAIcation focuses on **travel
readiness**: identifying what changed after a trip was booked, surfacing
operational risks, and recommending actions before disruptions affect
employees. It is not a vacation planner — it's a travel intelligence layer
for companies that need to know whether a trip is still viable.

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
tools to call turn by turn — producing redundant tool calls and 22+ second
latency even on simple factual queries. Routing the decision into graph
edges instead of LLM reasoning makes execution predictable, auditable, and
cheap to run at scale.

Every step is traced automatically through LangSmith (configured via
environment variables — no extra code needed).

## Endpoints

| Method | Path                | Purpose                                                          |
|--------|---------------------|------------------------------------------------------------------|
| POST   | `/travel/brief`     | Trip-planning mode — research and plan a trip before booking     |
| POST   | `/travel/readiness` | Trip-monitoring mode — what's changed since booking, what to do  |
| POST   | `/eval/run`         | Run the benchmark suite and report routing accuracy              |
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

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with:

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

Both return a structured JSON report (`summary`, `key_findings`, `risks`,
`recommendations`, `sources`) alongside the routing decision the classifier
made for that query.

## Benchmark

`benchmark.py` holds a 25-case evaluation set (8 simple / 9 research / 8 deep,
split 10 brief / 15 readiness) spanning all three routing tiers and both
workflows. Each case carries an `expected_route` and an `evaluation_focus`
note, so results are gradable and self-documenting rather than just logged.

Run it standalone:

```bash
python benchmark.py
```

…or trigger it over HTTP (same underlying runner, same graph):

```bash
curl -X POST "http://localhost:8000/eval/run"
```

Either path scores **routing accuracy** — the one judgment that's genuinely
domain-specific, since it means comparing the graph's `actual_route` against
labels from our own dataset — and tags every run in LangSmith under
`run_type="benchmark"` with a matching `case_id`.

Routing accuracy is the *only* metric computed locally. Everything about how
each run actually executed — latency, LLM-call and tool-call counts and
breakdowns, completion status, output — is recorded automatically by
LangSmith's tracer for every tagged run. Re-measuring any of that by hand
would just be a worse copy of what the observability platform already owns,
so the benchmark and baseline runners don't try; they only drive the queries
and tag them correctly.

### Baseline comparison

`baseline.py` replays a representative 9-case sample (3 per tier, both
workflows) of the same benchmark dataset through the *original* ReAct agent
(`app.py`), tagging each run `run_type="baseline"` with a matching `case_id`.
This is what substantiates the "22+ second latency, redundant tool calls"
claim that motivated the routed rebuild — it's a fixed "before" reference
point, not something worth running at full scale.

```bash
python baseline.py
```

### Consolidated comparison report

`compare.py` pulls both the baseline (`run_type="baseline"`) and routed
(`run_type="benchmark"`) traces straight out of LangSmith, matches them by
`case_id`, and prints one consolidated before/after report — latency,
LLM-call counts, and tool-call breakdowns for each matched pair plus
aggregate averages — instead of opening dozens of individual traces in the
LangSmith UI and comparing them by eye.

```bash
python compare.py
```

Run it after both `baseline.py` and `benchmark.py` (or `/eval/run`) have
populated traces in the project.
