# E026: explicit preference criteria and information-mask diagnostic

2026-11-08. Fixed before calculating E026 option truth tables or information-mask outcomes. This is development on the same 12 E025 tasks, already inspected; not a fresh confirmatory experiment.

## Inputs and scope

Use only frozen E025 `public_cases.json`, `private_cases.json`, `preference_inventory.json`, `selection.json` and manifest. Preserve all 12 tasks, 48 aspects, 96 preferences and 864 options; do not replace or silently drop difficult examples. No new upstream split, model generation, paid API or E023 reserve inspection.

Build an explicit criterion for each preference, with status `operational`, `ambiguous`, or `unsupported`. An operational criterion means testable under the stated dataset-field convention, not independently validated human intent. Keep the implicit statement, generation instruction, interpretation rationale and field-level assumptions. Source best/correct labels must not determine rules. Freeze and hash the catalogue before applying rules to options.

## Semantics

Use three-valued evaluation: T (satisfies the operational criterion), F (contradicts it), U (insufficient evidence or unresolved criterion). Missing required fields give U, not F. An explicit null service fee means unavailable under the source schema. A present amenity/tag list is treated as the source's complete list for that field; state this closed-list convention. Normalize case, whitespace and hyphens, not arbitrary synonyms. Preserve nested room types. A numeric threshold may be used when explicitly supplied by the implicit statement; vague longer/reasonable/minimum, causal claims about comfort/rewards, or underspecified alternatives remain unresolved unless their operational interpretation is explicitly limited.

Score preference conjunction only: F if either preference is F, T if both T, otherwise U. Keep source-best/source-correct alongside the new values. Do not claim this validates travel feasibility, real menus, real reward eligibility or total monetary cost. No automatic repair of source labels. Report ambiguous/unsupported criteria and per-option U separately from a user's `I don't know` reply.

## Secondary diagnostic: can information masks change the best question?

This is an oracle symbolic filter with a specified random-choice executor, not a learned agent or an estimated user response model. Four fixed public questions ask about requirements for flight, lodging, rental car and restaurant; lodging follows the publicly known hotel/apartment aspect. Do not expose active preference categories in the question menu. One question is asked in every compared condition, with the same cost; monetary/time cost is not estimated.

No-information executor: uniform choice over the 18 displayed options per aspect. Given a revealed criterion, discard only options evaluated F; keep T and U. Choose uniformly from the remaining options. If the candidate set is empty, flag it and fall back to the original uniform distribution; report this explicitly.

For each question, evaluate four deterministic masks: reveal both preferences (`full`), reveal source preference index 0 (`first_only`), reveal index 1 (`second_only`), reveal nothing (`no_information`). The ordering is a storage convention, not a realistic or estimated partial-response mechanism. Natural-language parsing, search and response probabilities are absent.

The target is mean conjunction satisfaction across the four aspects, under executor randomness. Give lower/upper bounds: T=1, F=0, U in [0,1]. For a change in policy, compute coupled bounds on delta using each option's coefficient p_after - p_before; never subtract unrelated endpoint bounds. This interval allows independent completions of unresolved option truth and may be conservative for shared uncertain semantics. It is not a confidence interval.

A robust winner q has delta_lower(q) > delta_upper(r) for every other question r. Count a robust ranking reversal only when full and a partial mask each have a unique robust winner and the two differ. Retain ties, overlapping intervals and unknown outcomes. Report first_only and second_only separately; do not select the more favorable mask as the main result. No hypothesis significance test or publication claim from this development screen.

## Checks and decision

Check three-valued logic, explicit zero vs null fee, missing fields, numeric boundaries, list normalization and uniform filtering. Independently recompute outcomes with exact rational arithmetic, including exhaustive completions where needed to check coupled bounds. Check question menus do not contain hidden active categories. Retain hashes, rule catalogue, per-option evaluations, every mask outcome and limitations. An independent computational verifier is not an independent semantic annotator.

If the rules remain partly unresolved, publish coverage and bounded diagnostics rather than claiming a validated benchmark. Even robust rank reversals would show only structural opportunity under these explicit assumptions. Next training requires justified response/goal distributions, independently reviewed semantics, a leakage-controlled split, and budget-matched no-ask/constant/heuristic/direct/factorized comparisons.
