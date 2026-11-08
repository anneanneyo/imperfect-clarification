# E027: visible-input executors under fixed information masks

2026-11-08. Fixed before E027 executor outputs or outcome scores. All 12 E025 tasks are already-inspected development data. No new test claim, training or task replacement.

## Scope

Use the frozen E025 public menus, initial user message and implicit preference statements. Evaluate against both unchanged source-best/source-acceptable labels and the E026 reviewed T/F/U table, including its explicit Rental/Rentals correction. Preserve unresolved semantics. Do not read the official test or E023 reserve.

Factorize the executor by aspect: give it the full public trip request, the publicly known target aspect, that aspect's 18 visible options, and only the revealed preference statements. It chooses one option ID. This is a fixed-menu modification, without UserBench search, routing, simulator or multi-aspect planning. An aspect's choice is assumed unaffected by replies about other aspects; state this recombination assumption.

Four masks: no additional statements, both statements, source-index 0 only, source-index 1 only. Full/partial replies are verbatim source implicit statements, not newly generated user conversations. No-information reuses the no-additional-statements input/output exactly; a natural-language “I don't know” interaction is not tested. There are at most 192 unique aspect inputs (12 tasks × 4 aspects × 4 masks), deduplicated by exact message hash.

## Executors fixed before scoring

1. Uniform: chooses each displayed option with probability 1/18, ignoring the reply. This is a noninformative floor, distinct from E026's oracle symbolic filter.
2. Visible lexical/price heuristic: maximize IDF-weighted token overlap with revealed text; break ties by overlap with the initial public request, then lower displayed base price (hotel: minimum listed room price; restaurant: Cheap < Normal < Expensive). Final ties are uniform. IDF uses only the 18 visible option descriptions; source labels and criteria never enter. Tokens, stopwords, price conventions and code are frozen before outputs. No claim that this heuristic must outperform uniform.
3. Local Qwen3-4B-Instruct-2507, existing 4-bit checkpoint revision 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b. Temperature 0, seed 20261108, at most 64 output tokens, one attempt per unique input. Give a fixed general instruction to respect stated requirements, avoid inventing unstated ones, and favor lower cost when otherwise suitable. Request exactly one JSON object with only an id key. Invalid/extra-key/duplicate-key/out-of-menu output fails with zero score; no repair, retry or best-of selection.

The model process loads only prepared public inputs and the pinned model/runtime. It never reads preference catalogues, private gold files, prior error reports or counterfactual outcomes. Preparation may access hidden statements solely to construct each revealed answer; the scoring process remains separate. Freeze input hashes, model-file hashes, code, prompt template and decoding settings before generation.

Local budget: 3600 seconds maximum generation-loop time; no paid API, downloads or training. Log every started attempt, raw output, timing, token counts and failures. If incomplete, retain all attempts and report coverage; do not summarize a subset as a complete run or regenerate completed outputs.

## Outcomes and comparisons

Record validity, source-best, source-acceptable, and provisional preference-conjunction bounds. For a fixed executor distribution p, bound its criterion score by treating T=1, F=0, U in [0,1]. Invalid outputs have all-zero choice mass and zero score.

For one aspect question and mask, recombine that aspect's post-reply choice with the other three no-information choices. All one-question comparisons use the same executor and one-question budget. Keep the all-four-aspects-informed executor diagnostic explicitly separate: it is not a one-question agent result.

Compute coupled gain bounds with coefficients (p_after - p_before)/4 on the same unknown option outcomes. Negative gains are allowed. For provisional semantics, use the E026 strict robust-winner definition: lower(q) > upper(other) for every other question. A reversal requires different unique robust winners for full and partial; report first_only and second_only separately and retain unresolved cases.

For source-acceptable labels also report a tie-controlled mismatch: find the entire set of full-mask maximizers, then subtract the best partial-mask gain within that set from the global best partial-mask gain. Thus an arbitrary full-mask tie-break cannot create a positive mismatch.

Report all four constant-question choices, rather than presenting a post hoc best constant as a validated learned policy. Oracle rankings use unavailable outcomes and are diagnostic only. No inferred user-response distribution, expected value over real replies, population significance or publication claim.

## Verification and interpretation

Behavioral tests must cover public-input boundaries, malformed outputs, tie handling, negative gains and coupled U bounds. Independently rescore saved outputs and recombined choices without importing the main analyzer. Check all frozen E025/E026 artifacts remain unchanged.

The key decision is whether the E026 pattern persists beyond ideal symbolic filtering and a uniform action executor. A null/negative result, answer-induced errors or label/semantic disagreement is a valid outcome. Semantics remain provisional, both previous reversal tasks share a flight template, and no independent evaluation split has been established.
