# E025: pinned UserBench train audit

Status at creation: protocol fixed before reading train rows or source task contents. 2026-11-08.

## Scope and selection

Use SalesforceAIResearch/UserBench commit `80506d2ab484cab843e60a2401ff3e0290d05b87`. Configuration: `data/travel2222_multiturn/train.parquet`, Git blob `0eaea0fa79df05a7bdd165b323a79b10fb9e15c7`. Obtain its task IDs from `reward_model.id` first. Rank unique IDs by SHA256 of UTF-8 `E025|` + task ID, then ID; select the first 12 (or all if fewer). No replacement after inspection.

Task source: `travelgym/data/travelgym_data_2222.json`, Git blob `974c4457d1ace709ea3f2652ddd2dbb405db9052`. This upstream file mixes splits: mechanically parse it but expose and analyze only the selected train IDs. Do not download the official test parquet or inspect/evaluate nonselected tasks. Raw files stay in the external cache. Record hashes, selected IDs and derived evidence in Git.

## Questions and outputs

1. Record train index schema, ID uniqueness, visible initial messages and source-to-index matching for the selected tasks. Check direct hidden preference/gold leakage and explain the limits of automated substring checks. Distinguish public trip constraints from private user preferences.
2. Inspect source preference records, option schema, `best_id`/`correct_ids`, option uniqueness and labels. Check what information survives the upstream option sanitation. Inspect identifier/order patterns as potential shortcuts. Do not silently repair source annotations.
3. Read all selected preference texts and a deterministic subset of options for semantic consistency. Record specific evidence and unresolved concerns. This is a single-assistant development audit, not independent human validation or a benchmark score.
4. Implement an isolated, explicit final-choice evaluator: one submitted option per required aspect; best and acceptable membership scored separately; invalid/missing choices fail; no intermediate elicitation reward; completion after one choice per aspect. Report exact-best rate, acceptable rate, and whole-task success. These are source-label metrics, not a newly validated preference-satisfaction metric. Keep any actual criterion-level semantic audit separate.
5. Freeze a visible/hidden adapter boundary. Options may include natural-language constraints but must not contain source `type`, `reason`, grouping, best/correct IDs or hidden preferences. Any local option remapping or fixed-menu choice mode must be labeled as a modification, not original UserBench execution.
6. Decide whether this source is ready for question-selection experiments, needs an adaptation, or should stop. Passing structural checks alone does not prove useful imperfect-answer diversity or novelty. No training/model generations in E025; do not invent gain estimates. If a source-label shortcut is found, quantify it on this fixed sample and specify its effect on the next protocol.

## Integrity and continuation

Save the preinspection protocol hash, source hashes, deterministic selection, audit script/results and verification. Do not change E014–E020/E022–E024 or open E023 reserve outcomes. Any follow-up experiment needs a separate protocol. A future budget-matched comparison should include no clarification, a constant/heuristic question, direct value prediction, and factorized response/decision prediction; the original topic remains decision-focused clarification under imperfect user replies.
