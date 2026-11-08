# Research workspace

Read `research/PROJECT_STATUS.md` and `research/NEXT_DECISIVE_EXPERIMENT.md` before continuing. Explain progress to the user in clear Russian. Work in this repository; branch names start with `codex/`.

Keep frozen experiment protocols, scripts and results byte-stable. E014–E020, E022–E025 are completed; E021 has an inventory but no reply replay. E025 includes its choice adapter, tests and independent verifier; develop a new adapter version separately. Add new diagnostics separately. Never rerun model generations just to improve a reported result. E016 test has already been inspected; E023's 217 reserved groups have not had outcomes scored. Do not select a more favorable screen after its gate failed.

Keep oracle results distinct from trained-agent results, synthetic data distinct from human evidence, and source-code contract probes distinct from benchmark reproduction. Report negative findings and limitations. Publication-level novelty is not established.

Use `REPRODUCE.md` for current verification. Large language model checkpoints, packages, downloaded external source repositories and credentials belong outside Git. Small fitted estimator matrices are retained with experiment results. Two large historical CSVs are tracked as gzip with hashes; use the restoration script when necessary. Preserve the current GitHub README curator correction.

Do not contact supervisors or upstream authors, submit papers, or run paid API jobs unless explicitly authorized by the user. Ordinary local research, repository updates, and checks are within the active task.
