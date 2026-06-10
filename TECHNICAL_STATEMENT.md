# vAIcation — Technical Statement

## Starting with measurement

The base app gave the model Tavily Search, Extract, and Crawl and let a ReAct
loop decide which to call. The first thing I did was turn on LangSmith tracing
(environment variables only, no code changes) and run a few queries to see what
it actually did.

The traces were enough to point me at the problem. A plain factual question
routed through three LLM calls and two searches and took about 22 seconds,
because the loop had no real reason to stop — it kept deciding it could do a
little better with one more search. That's fine for open-ended research, but it
means cost and latency are unpredictable per query, and at volume unpredictable
cost is the thing you actually have to manage.

## Picking a use case

I went with travel intelligence because it's a case where live retrieval is the
whole point. Flights get cancelled, strikes get called, advisories get issued,
visa rules change. A model trained months ago can't tell you whether a trip you
booked last week is still a good idea; a current web read can.

I framed it for an enterprise buyer rather than a consumer. The interesting
problem for a corporate travel team isn't planning a trip — it's knowing what
changed after they booked it. That's the `/travel/readiness` workflow, as
opposed to `/travel/brief` for up-front planning.

## Routing instead of looping

I replaced the ReAct agent with a LangGraph pipeline that makes one routing
decision up front and then runs a fixed path. A classifier labels each query
`simple`, `research`, or `deep`, and the graph runs exactly the tools that tier
needs, each at most once: simple is search; research is search then extract;
deep is search, extract, then crawl.

The classifier returns a `Literal`, so it can only ever produce one of the three
valid labels — the routing decision can't drift into free text. The point isn't
that routing is clever. It's that it makes execution predictable: you can look
at a query type and know what it will cost before you run it.

## Trip context

A readiness check is more useful when it knows the trip. The `/travel/readiness`
endpoint accepts an optional structured booked-trip record — route, dates,
carriers, and the booking date. vAIcation defines the shape it consumes; in a
real deployment that record comes from the customer's travel-management or
booking system, not from vAIcation.

When a trip is present, the booking date becomes the baseline. The search is
scoped to what was published since `booked_at`, and on the narrower routes it's
biased toward authoritative news sources and the trip's own carriers. This is
what makes the readiness check a "since booking" diff rather than a fresh
generic search.

## What the benchmark showed

I wrote a 25-case set covering all three tiers and both workflows, each case
labeled with the route I expected. I ran a sample of those queries through the
original ReAct agent first to get a "before," then ran the routed system against
the full set.

The routed system used noticeably fewer calls — average LLM calls and tool
calls both dropped by roughly a third. That's the result I care about most,
because call count drives cost at volume. Latency improved less, and the routed
system was still slow in absolute terms; most of that time is the Tavily reads
and the synthesis call, not the loop, so routing helps the bill more than the
clock.

Two honest caveats on these numbers. The set is only 25 cases, so the
routing-agreement percentages move by a few points when a single case flips —
I read them as directional, not precise. And both the benchmark and the live
path hit the real web, so the exact figures aren't reproducible run to run. The
specific numbers in the README are labeled as a snapshot from the original
routing build; they predate the schema and grounding changes described below,
and `baseline.py` / `benchmark.py` / `compare.py` regenerate current ones.

## The output contract

The response is structured so downstream systems can consume it, but it stops
short of making business decisions. The model returns evidence-linked facts:
each risk carries a category, severity, summary, affected leg, an evidence URL,
and a confidence level, and each recommendation carries its own rationale and
evidence URL. Severity includes an explicit `unknown` value so the model has an
honest option when the evidence doesn't justify a level.

What the model does not produce is customer policy — owner, SLA, alert priority,
escalation path, and final trip status. Those depend on an organization's risk
tolerance, not on anything in the retrieved evidence, so they belong in a
deterministic policy layer the customer configures. Keeping them out of the
model's output also keeps them out of the hallucination surface: the model can't
fabricate a deadline or an owner if it's never asked to produce one. The short
version is that the model gives grounded travel risks, and a separate policy
layer decides what the company does about them.

## Measuring quality, and what it caught

Routing the right way doesn't guarantee a good answer, so I added two
LLM-as-judge evaluators in LangSmith — one for hallucination (is every claim
backed by retrieved evidence) and one for answer relevance (did it answer the
question that was asked). They run against tagged traces after the request
finishes, so they add no latency to serving.

The hallucination evaluator earned its place by failing a live readiness query.
On a booked JFK–LHR trip, the report asserted that an event "prompted" a
screening rule the source never connected causally, generalized a single
country's measures into "some countries," and rated generic travel concerns as
trip-specific "Medium" risks. The judge flagged it, and the trace let me work
backwards through why.

There turned out to be several layers, not one bug:

- The synthesis prompt was under-constrained, so the model filled gaps with
  general knowledge and inferred causation the sources didn't state.
- The extracted evidence was noisy — full pages including navigation and
  related-story sidebars — which both bloated context and gave the judge
  unsupported material to flag.
- I'd introduced a real bug: I set `max_results` on the search call, which this
  tool only accepts at instantiation. The search call failed, the failure wasn't
  surfaced clearly, and the system still produced a confident report from no
  evidence at all.
- The search query was just the raw user sentence, with no route, carrier, or
  date terms. Even once the bug was fixed, that meant the deep tier retrieved
  generic travel-trend articles rather than anything specific to this trip.

I fixed what was worth fixing: cleaned the evidence down to main content with
per-source caps, added explicit grounding rules (no prior knowledge, no inferred
causation, no over-generalization, and say so plainly when a risk domain isn't
covered), added the `unknown` severity option, added a few disjoint few-shot
examples to steady the classifier on boundary cases, and moved `max_results` to
instantiation.

The most useful thing I learned came out of this loop: fixing the hallucination
exposed a separate problem. Once the model stopped inventing, it produced an
honest report grounded in the wrong sources — relevant-looking but generic,
because retrieval was generic. "Grounded" and "relevant" are different axes,
which is exactly why two evaluators are worth having: one caught the fabrication,
the other caught the off-target sourcing. The remaining fix is to build the
search query from the trip, so retrieval targets the actual route and carriers.
I left that as documented future work rather than adding more retrieval
complexity to a project whose main point is cost control.

## What I'd do next

- **Trip-aware query construction.** Under-specified readiness queries retrieve
  generic sources. Rewriting the search query from the trip (destination,
  carriers, dates, risk domains) is the highest-value retrieval improvement.
- **The policy layer.** The schema emits grounded facts; a production
  integration needs the deterministic layer that maps them to owner, priority,
  SLA, and trip status.
- **Crawl source selection.** The crawl node currently takes the first search
  result. It should score source quality first and prefer authoritative pages
  (government advisories, carrier and airport sites).
- **Context filtering.** Synthesis currently truncates retrieved content.
  Dynamic, query-specific filtering would let only relevant content into the
  prompt — Tavily's own write-up on dynamic filtering is the reference point
  here ([Dynamic Filtering](https://www.tavily.com/blog/dynamic-filtering-let-the-model-program-its-own-search-filters)).
- **Evaluators in code.** The two judges live in the LangSmith UI today. Moving
  them into the repo would make the quality numbers reproducible and
  version-controlled, and would let me sharpen the hallucination rubric beyond a
  single pass/flag.
- **Surfacing tool failures.** The `max_results` incident showed the graph will
  quietly turn a failed retrieval into a confident answer. A production version
  should detect a tool error and report "couldn't assess" rather than proceed.

## What this was

The goal wasn't to build the whole platform. It was to find a real problem by
measuring it, fix that problem, and use the same measurement to catch the next
one — including the ones I introduced myself. The honest version of this work is
a loop: instrument, measure, route, evaluate, find a failure, trace it, fix it,
and know when the remaining gap is better documented than chased.
