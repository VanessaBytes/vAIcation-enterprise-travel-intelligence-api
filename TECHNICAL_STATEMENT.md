# vAIcation — Technical Statement

**The problem with the base app wasn't the tools — it was the absence of measurement.**

The starter application had all three Tavily capabilities available — Search, Extract, and Crawl — but no way to know which ones fired, when, why, or at what cost. Before building anything, that had to be fixed. You can't improve what you can't observe.

The first change was instrumentation. I wired LangSmith tracing into the application via environment variables — zero code changes, full visibility into every LLM call, every tool invocation, latency at each step, and token usage per query. The first trace told me everything I needed to know: a simple factual question — "Who is the Snowflake CEO?" — triggered three LLM calls, two search calls, and took 22 seconds. The agent was looping because nothing told it to stop.

**The domain choice was deliberate.**

I chose travel intelligence because it's one of the few domains where the value of live web retrieval is non-negotiable. Flights get cancelled. Strikes happen. Visa rules change. Travel advisories get issued. A static LLM cannot answer "is my trip still viable?" — but Tavily can, because it retrieves what's true right now, not what was true at training time. Travel is where Tavily's core capabilities shine.

I also framed it as an enterprise product rather than a consumer tool. The real problem enterprise mobility teams face isn't planning trips — it's knowing what changed after they booked. vAIcation's `/travel/readiness` endpoint is built around that specific workflow: not "plan me a trip" but "tell me what's different since I committed to this trip."

**The architectural improvement was routing discipline.**

The base app used a ReAct agent — a pattern where the LLM decides which tools to call at every step, looping until it feels confident enough to answer. That's fine for exploration but wrong for production. It produces unpredictable cost, unpredictable latency, and no auditability.

I replaced it with a deterministic LangGraph pipeline. A classifier makes one routing decision upfront — simple, research, or deep — based on what the query actually needs. The graph then executes exactly that path, each tool firing at most once. No loops. No redundant calls. No LLM deciding mid-execution to search again because it's not sure.

The classifier uses structured output with a Literal type constraint — the LLM cannot return anything except one of three valid routing decisions. That's not just prompt engineering, it's a hard architectural guarantee.

**The benchmark told the real story.**

I built a 25-case evaluation dataset spanning both workflows and all three routing tiers, with expected routes and evaluation focus notes on every case. I ran the original ReAct agent against a representative sample first to establish a baseline, then ran the routed system against the full set.

First pass: 71% routing accuracy. The classifier was under-routing complex readiness queries — treating multi-domain risk assessments as single-topic lookups. I diagnosed the failure pattern, tightened the classifier prompt with explicit trigger conditions for each tier, and reran.

Second pass: 84% routing accuracy, 25/25 completed, 0 errors.

The before/after comparison from LangSmith showed:

- Average LLM calls reduced by 33% (3.0 → 2.0)
- Average tool calls reduced by 38% (3.22 → 2.0)
- Average latency reduced by 15% (31.4s → 26.8s)
- Research tier specifically: tool calls reduced by 63%, latency reduced by 31%

The most striking individual case: "What disruptions are affecting Heathrow airport this week?" — the ReAct agent used 9 LLM calls and 8 search calls. The routed system used 2 LLM calls and 2 tool calls. Same question, a fraction of the cost, faster answer, structured output.

**A multi-dimensional evaluation framework.**

Routing accuracy alone doesn't tell the full story. A system can route correctly and still produce a hallucinated or irrelevant answer. To measure output quality independently of routing efficiency, I added two LLM-as-a-judge evaluators running automatically in LangSmith on every trace — Hallucination and Answer Relevance. Both use gpt-5.5 as the judge model, operating asynchronously after each request so evaluation adds zero latency to the serving path.

Hallucination checks whether every claim in the output is supported by the Tavily evidence that was retrieved — catching cases where the LLM fabricates information not present in the sources. Answer Relevance checks whether the output actually addressed what was asked — catching cases where the system produces a technically grounded response that answers a different question than the one posed.

Together with routing accuracy and the LangSmith efficiency traces, vAIcation's evaluation framework measures four independent dimensions of system quality: did it take the right path, did it run without waste, did it tell the truth, and did it answer the question? That combination — routing efficiency, cost control, hallucination detection, and answer relevance — reflects how enterprise AI teams evaluate production systems at scale, not just whether a demo works.

**What this is and what it isn't.**

vAIcation is not a vacation planner. It is a travel intelligence layer for organizations that need to know whether a trip is still viable. The structured JSON output — summary, key findings, risks, recommendations, sources — is designed to be consumed by downstream enterprise systems, not read by end users.

The system has known limitations. The classifier achieves 84% routing accuracy; the remaining misroutes are genuine boundary cases where the distinction between tiers is ambiguous. A production system would address this through few-shot examples or a confidence threshold that escalates uncertain classifications. The synthesize node currently passes truncated retrieval content to the LLM — the production solution is dynamic filtering, letting the model write query-specific filter programs at runtime so only relevant content enters context. Tavily's own benchmarks show this pattern reduces token usage by 3.5x. The evaluation framework also lacks idempotency guarantees — a production implementation would add run-level locking before writing results.

The goal of this submission wasn't to build everything. It was to measure a real problem, implement a focused solution, and demonstrate improvement through evidence. That's how production AI systems get built — not in one pass, but iteratively, with data driving every decision.
