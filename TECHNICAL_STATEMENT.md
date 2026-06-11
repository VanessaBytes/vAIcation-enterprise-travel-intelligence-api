# vAIcation — Technical Statement

## How I approached it

What I appreciated about this assignment was starting with a plain app that had
real limitations. It was simple enough to understand, but open enough that I
could make it my own: choose a domain, decide what a useful enterprise workflow
would look like, and then measure whether the changes actually helped.

I chose travel readiness because it made the context problem concrete. A model
can only be as useful as the information and boundaries it is given. Throughout
the project, two themes kept recurring. One was context engineering, the
question of what evidence I feed the model and what boundaries I give it. The
other was evaluation, meaning how I check whether the answer actually met the
standard I expected. Those two themes drove the design choices behind the
system, mainly what the model should decide and what should stay deterministic
in code. Tavily's Search, Extract, and Crawl tools were useful because they
exposed that whole chain: live retrieval, source selection, evidence quality,
and final synthesis.

## Starting with measurement

The base app gave the model Tavily Search, Extract, and Crawl and let a ReAct
loop decide which to call. The first thing I did was turn on LangSmith tracing
(environment variables only, no code changes) and run a few queries to see what
it actually did.

The traces were enough to point me at the problem. A plain factual question
routed through three LLM calls and two searches and took about 22 seconds,
because the loop had no real reason to stop. It kept deciding it could do a
little better with one more search. That's fine for open-ended research, but it
means cost and latency are unpredictable per query, and at volume unpredictable
cost is the thing you actually have to manage.

## Picking a use case

I went with travel intelligence because it's a case where live retrieval is the
whole point. Flights get cancelled, strikes get called, advisories get issued,
visa rules change. A model trained months ago can't tell you whether a trip you
booked last week is still a good idea; a current web read can.

I framed it for an enterprise buyer rather than a consumer. For a corporate
travel team, planning the trip is the easy part. The harder and more valuable question
is what changed after they booked it. That's the `/travel/readiness` workflow, as
opposed to `/travel/brief` for up-front planning.

## Routing instead of looping

I replaced the ReAct agent with a LangGraph pipeline that makes one routing
decision up front and then runs a fixed path. A classifier labels each query
`simple`, `research`, or `deep`, and the graph runs exactly the tools that tier
needs, each at most once: simple is search; research is search then extract;
deep is search, extract, then crawl.

The classifier returns a `Literal`, so it can only ever produce one of the three
valid labels. The routing decision can't drift into free text. The point isn't
that routing is clever. It's that it makes execution predictable: you can look
at a query type and know what it will cost before you run it.

Another way to say it is that the classifier decides how much Tavily evidence
to gather. Search runs once for every request because every answer should be
grounded in current web information; it gives the app URLs and snippets to work
from. Extract and Crawl are added only when the route needs fuller page content
or broader site coverage.

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

I also added deterministic query enrichment for under-specified readiness
questions. If the user asks "what changed since booking?" and supplies a trip,
the graph appends the missing route, date, hotel, and carrier terms before
search. On the deep route it also adds broad risk-domain terms, because that
path is explicitly for multi-domain readiness.

## What the benchmark showed

I wrote a 25-case set covering all three tiers and both workflows, each case
labeled with the route I expected. I ran a sample of those queries through the
original ReAct agent first to get a "before," then ran the routed system against
the full set.

The routed system used fewer calls and, after one fix, less wall-clock time. On
a matched nine-case before/after sample (`june-10-latency-v1`), average LLM
calls dropped from 3.2 to 2.0 and tool calls from 2.7 to 2.0. Call count is the
result I care about most, because it drives cost at volume.

Latency took an extra step to get right. An early routed run was actually
*slower* than the baseline, and the traces showed why: the final synthesis call
was about 90% of each request, while Tavily returned in two to three seconds.
The synthesis model was a reasoning model spending most of its time thinking. I
kept the architecture the same and lowered reasoning effort for the bounded
tasks (`low` for synthesis, `minimal` for the three-way classifier), which
preserved the fixed-call design while taking pressure off the slowest span.
After that change, routed latency fell to about 20s against 32s for the baseline
on the same sample, so the routed graph now wins on all three axes:

| Metric (9-case matched sample) | Baseline ReAct | Routed graph | Change |
|--------------------------------|----------------|--------------|--------|
| Avg LLM calls                  | 3.2            | 2.0          | −38%   |
| Avg tool calls                 | 2.7            | 2.0          | −25%   |
| Avg latency                    | 32.4s          | 20.1s        | −38%   |

The per-case data keeps this honest. The routed graph is faster on most cases,
and dramatically so where the baseline's ReAct loop spiralled. One case ran ten
LLM calls and nine searches at 63s, where the routed graph answered the same
query in 12s. The only cases where the baseline is faster are the ones where it
skipped retrieval entirely and answered from memory, which is exactly the
ungrounded behaviour the routed design exists to prevent.

The routing classifier also improved through iteration. The early runs were not
meant to be final scores; they were a development trail showing where the
classifier was missing boundary cases and how prompt/context changes affected
it:

| Run        | Routing agreement | What changed                  |
|------------|-------------------|-------------------------------|
| june-08-v1 | 71%               | Initial classifier            |
| june-08-v2 | 84%               | Classifier prompt tightened   |
| june-08-v3 | 80%               | Context-engineering pass      |
| quality-check-v4 | 96%           | Current routed graph          |

Two honest caveats on these numbers. The set is only 25 cases, so the
routing-agreement percentages move by a few points when a single case flips.
I read them as directional, not precise. And both the benchmark and the live
path hit the real web, so the exact figures aren't reproducible run to run. The
call and latency numbers regenerate from `baseline.py` / `benchmark.py` /
`compare.py` against a shared run ID, and `eval_results.py` pulls the evaluator
scores, so every figure here can be reproduced from a tagged run.

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

Routing the right way doesn't guarantee a good answer. It only guarantees that
the app took the path I expected. To check answer quality, I added two
LLM-as-judge evaluators in LangSmith:

- **Hallucination:** is every claim backed by the retrieved evidence?
- **Answer relevance:** did the response answer the question that was asked?

They run against tagged traces after the request finishes, so they add no
latency to serving.

The hallucination evaluator earned its place by failing a live readiness query.
On a booked JFK–LHR trip, the report asserted that an event "prompted" a
screening rule the source never connected causally, generalized a single
country's measures into "some countries," and rated generic travel concerns as
trip-specific "Medium" risks. The judge flagged it, and the trace let me work
backwards through why.

There turned out to be several layers, not one bug:

- The synthesis prompt was under-constrained, so the model filled gaps with
  general knowledge and inferred causation the sources didn't state.
- The extracted evidence was noisy: full pages including navigation and
  related-story sidebars, which both bloated context and gave the judge
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
covered), added stricter evidence-strength rules for severity, added the
`unknown` severity option, added a few disjoint few-shot examples to steady the
classifier on boundary cases, and moved `max_results` to instantiation.

The most useful thing I learned came out of this loop: fixing the hallucination
exposed a separate problem. Once the model stopped inventing, it produced an
honest report grounded in the wrong sources, relevant-looking but generic,
because retrieval was generic. "Grounded" and "relevant" are different axes,
which is exactly why two evaluators are worth having: one caught the fabrication,
the other caught the off-target sourcing.

That led to the final retrieval hygiene pass. I added a deterministic relevance
ranker before extract, crawl, and synthesis. It scores retrieved items by simple
signals: route/location matches, carrier and hotel matches, date/month matches,
risk-domain terms, and meaningful query terms. It is intentionally not an LLM
reranker. The point is to prefer obviously trip-relevant evidence without adding
another variable-cost reasoning step.

The last full benchmark run completed all 25 cases with 96% routing agreement.
After the final retrieval hygiene pass, the LangSmith evaluators moved in the
right direction:
answer relevance went from 0.68 to 0.72, and hallucination flag rate went from
0.68 to 0.574. Across the broader quality loop, hallucination flag rate fell
from 0.75 to 0.574. The hallucination number is a flag rate, so lower is better.
The absolute numbers are still not something I'd oversell (the set is small,
the web changes, and the judges live in LangSmith), but the direction matched
the failure we were trying to fix.

The aggregate flag rate also needs context, and splitting it by route is where
it gets useful. The simple tier was mostly clean: 2 of 8 cases were flagged, and
both were minor attribution/over-specificity issues rather than broad
fabrication. The research tier was mixed. The deep tier was the real problem:
all 8 deep cases were flagged. That tells me the remaining risk is not the whole
system; it concentrates on the multi-source, multi-domain synthesis path. The
failure mode was consistent too: noisy extract/crawl content made it into
synthesis, and the model sometimes promoted adjacent evidence into trip-specific
claims, for example applying Schengen/EU Entry-Exit System evidence to a
UK/London trip in two separate deep cases. That points straight at the next
production improvement: stronger source ranking and paragraph-level context
filtering before synthesis, rather than a blanket attempt to lower the aggregate.

## What I'd do next

- **The policy layer.** The schema emits grounded facts; a production
  integration needs the deterministic layer that maps them to owner, priority,
  SLA, and trip status.
- **Stronger source ranking.** The current relevance ranker is deliberately
  simple and rule-based. A production version should use richer place aliases,
  source-quality signals, and preferred authoritative sources before deciding
  what to extract or crawl.
- **A lighter simple-route contract.** Simple factual/current checks still emit
  the same full report schema as deep readiness assessments. A production API
  should probably return a smaller shape for simple answers.
- **Context filtering.** The current ranker chooses which sources look most
  relevant before extraction, crawl, and synthesis. A production version should
  go one level deeper and keep only the most relevant paragraphs from those
  sources. That would reduce page boilerplate, related-story noise, and
  adjacent-but-irrelevant facts before the model writes the final report.
  Tavily's write-up on dynamic filtering is the reference point here
  ([Dynamic Filtering](https://www.tavily.com/blog/dynamic-filtering-let-the-model-program-its-own-search-filters)).
- **Evaluators in code.** The two judges live in the LangSmith UI today. Moving
  them into the repo would make the quality numbers reproducible and
  version-controlled, and would let me sharpen the hallucination rubric beyond a
  single pass/flag.
- **Surfacing tool failures.** The `max_results` incident showed the graph will
  quietly turn a failed retrieval into a confident answer. A production version
  should detect a tool error and report "couldn't assess" rather than proceed.

## What this was

The goal wasn't to build the whole platform. It was to take a small app, give it
a realistic enterprise use case, and then keep tightening the parts that made
the answer useful or not useful. The honest version of this work is a loop:
instrument, measure, fix what's worth fixing, and know when the remaining gap is
better documented than chased.

That is also what made the assignment interesting to me. Wiring up Tavily and
the graph was the straightforward part. The hard part was deciding what context
the model should see, what decisions should stay in code, and how to tell when a
polished answer was actually grounded.

It was honestly hard to stop iterating, because each measurement made the next
improvement more obvious. Routing exposed latency and cost. The hallucination
judge exposed weak grounding. Fixing grounding exposed retrieval relevance.
Improving retrieval made the remaining product gaps clearer. That was the most
useful part of the assignment for me: seeing how much better the system got when
I treated it as a measured loop instead of a one-time implementation.
