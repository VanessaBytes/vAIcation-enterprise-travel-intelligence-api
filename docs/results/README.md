# Result Summary

This directory contains the checked-in metric snapshots used in the README and
technical statement. Filenames describe what the artifact is; the JSON files
also include the original LangSmith `run_id`.

| File | What it Shows | LangSmith Run | Key Numbers |
| --- | --- | --- | --- |
| `baseline-sample.json` | Original ReAct agent on the 9-case sample. | `june-10-latency-v1` | 3.22 avg LLM calls, 2.67 avg tool calls, 32.4s avg latency. |
| `matched-comparison.json` | Baseline vs. routed graph on the same 9 cases. | `june-10-latency-v1` | Routed graph: 2.0 avg LLM calls, 2.0 avg tool calls, 20.1s avg latency. |
| `full-benchmark.json` | Final routed graph on all 25 benchmark cases. | `quality-check-v4` | 24/25 routes matched, 96% routing agreement. |
| `evaluator-results.json` | LangSmith evaluator scores for the final 25-case run. | `quality-check-v4` | 0.72 answer relevance, 0.574 hallucination flag rate. |

Earlier routing checkpoints are summarized in the technical statement: `june-08-v1`
at 71%, `june-08-v2` at 84%, and `june-08-v3` at 80%. They are included there as
development context, not as separate checked-in JSON artifacts.
