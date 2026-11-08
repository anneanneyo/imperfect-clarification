"""Score frozen E027 outputs; hidden labels are used only in this process."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from executors_e027 import lexical_distribution, parse_choice, score_bounds, full_tie_gap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/userbench_executor_e027"
POLICIES = ["uniform", "lexical_price", "qwen3_4b"]
MASKS = ["no_information", "full", "first_only", "second_only"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def average_metrics(records):
    n = len(records)
    return {k: str(sum(F(r[k]) for r in records) / n) for k in ["valid", "source_best", "source_acceptable", "semantic_lower", "semantic_upper"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    read = lambda name: json.loads((OUT / name).read_text())
    manifest = read("input_manifest.json")
    for group in ["input_sha256", "code_sha256"]:
        for path, expected in manifest[group].items():
            assert sha(ROOT / path) == expected, path
    run_status = read("run_status.json")
    if not run_status["complete"]:
        print(json.dumps({"status": "incomplete_run_no_complete_comparison", "coverage": run_status}))
        return
    generations = [json.loads(line) for line in (OUT / "generations.jsonl").read_text().splitlines()]
    starts = [json.loads(line) for line in (OUT / "started.jsonl").read_text().splitlines()]
    jobs = {r["id"]: r for r in [json.loads(line) for line in (OUT / "public_inputs.jsonl").read_text().splitlines()]}
    outputs = {r["id"]: r for r in generations}
    assert len(generations) == len(outputs) == len(jobs) == len(starts)
    assert {r["id"] for r in starts} == set(jobs) == set(outputs)
    schedule = read("schedule.json")
    e26 = ROOT / "results/userbench_semantics_e026"
    review_manifest = json.loads((e26 / "review_manifest.json").read_text())
    assert sha(e26 / "reviewed_option_evaluations.json") == review_manifest["artifact_sha256"]["reviewed_option_evaluations.json"]
    truth = json.loads((e26 / "reviewed_option_evaluations.json").read_text())
    by_group = {}
    for row in truth:
        by_group.setdefault((row["case_id"], row["aspect"]), {})[row["option_id"]] = row
    choices, scores, heuristic_traces = [], [], {}
    lookup = {}
    for entry in schedule:
        job = jobs[entry["id"]]
        labels = [o["id"] for o in job["payload"]["options"]]
        gold = by_group[(entry["case_id"], entry["aspect"])]
        assert set(labels) == set(gold)
        heuristic, trace = lexical_distribution(job["payload"])
        heuristic_traces[entry["id"]] = trace
        generation = outputs[entry["id"]]
        chosen = parse_choice(generation["raw_output"], labels) if generation["error"] is None else None
        distributions = {
            "uniform": {i: str(F(1, len(labels))) for i in labels},
            "lexical_price": heuristic,
            "qwen3_4b": {i: str(int(i == chosen)) for i in labels},
        }
        for policy, distribution in distributions.items():
            bounds = score_bounds(distribution, {i: g["conjunction"] for i, g in gold.items()})
            score = {"valid": str(sum(F(v) for v in distribution.values())),
                     "source_best": str(sum(F(distribution[i]) for i, g in gold.items() if g["source_best"])),
                     "source_acceptable": str(sum(F(distribution[i]) for i, g in gold.items() if g["source_acceptable"])),
                     "semantic_lower": bounds["lower"], "semantic_upper": bounds["upper"]}
            item = dict(entry, policy=policy, distribution=distribution,
                        model_chosen_id=chosen if policy == "qwen3_4b" else None)
            choices.append(item)
            scores.append(dict(entry, policy=policy, **score))
            lookup[(entry["case_id"], entry["aspect"], entry["mask"], policy)] = (distribution, score)
    cases = sorted({s["case_id"] for s in schedule})
    questions, ranks = [], []
    for case_id in cases:
        aspects = sorted({s["aspect"] for s in schedule if s["case_id"] == case_id})
        assert len(aspects) == 4
        for policy in POLICIES:
            before = {a: lookup[(case_id, a, "no_information", policy)] for a in aspects}
            baseline = average_metrics([r[1] for r in before.values()])
            local = []
            for aspect in aspects:
                q = "lodging" if aspect in {"hotel", "apartment"} else aspect
                gold = by_group[(case_id, aspect)]
                for mode in MASKS:
                    after_distribution, after_score = lookup[(case_id, aspect, mode, policy)]
                    before_distribution, before_score = before[aspect]
                    difference = {i: str(F(after_distribution[i]) - F(before_distribution[i])) for i in gold}
                    gain = score_bounds(difference, {i: g["conjunction"] for i, g in gold.items()}, F(1, 4))
                    after_task = average_metrics([after_score if a == aspect else before[a][1] for a in aspects])
                    item = {"case_id": case_id, "policy": policy, "question": q, "aspect": aspect, "mask": mode,
                            "baseline": baseline, "after_one_question": after_task, "semantic_gain": gain,
                            "source_best_gain": str((F(after_score["source_best"]) - F(before_score["source_best"])) / 4),
                            "source_acceptable_gain": str((F(after_score["source_acceptable"]) - F(before_score["source_acceptable"])) / 4)}
                    questions.append(item); local.append(item)
            winners = {}
            for mode in MASKS:
                candidates = [r for r in local if r["mask"] == mode]
                robust = [r["question"] for r in candidates if all(
                    F(r["semantic_gain"]["lower"]) > F(other["semantic_gain"]["upper"])
                    for other in candidates if r["question"] != other["question"])]
                assert len(robust) <= 1
                winners[mode] = robust[0] if robust else None
            full = {r["question"]: r["source_acceptable_gain"] for r in local if r["mask"] == "full"}
            gaps = {mode: full_tie_gap(full, {r["question"]: r["source_acceptable_gain"] for r in local if r["mask"] == mode}) for mode in ["first_only", "second_only"]}
            ranks.append({"case_id": case_id, "policy": policy, "robust_semantic_winners": winners,
                          "robust_semantic_reversals": {mode: bool(winners["full"] and winners[mode] and winners["full"] != winners[mode]) for mode in ["first_only", "second_only"]},
                          "source_acceptable_tie_controlled_gaps": gaps})
    all_aspects = {p: {m: average_metrics([r for r in scores if r["policy"] == p and r["mask"] == m]) for m in MASKS} for p in POLICIES}
    constants = {p: {m: {q: average_metrics([r["after_one_question"] for r in questions if r["policy"] == p and r["mask"] == m and r["question"] == q]) for q in ["flight", "lodging", "rental_car", "restaurant"]} for m in ["full", "first_only", "second_only"]} for p in POLICIES}
    ranking_summary = {}
    for policy in POLICIES:
        rs = [r for r in ranks if r["policy"] == policy]
        ranking_summary[policy] = {
            "robust_semantic_reversals": {m: [r["case_id"] for r in rs if r["robust_semantic_reversals"][m]] for m in ["first_only", "second_only"]},
            "source_acceptable_positive_strict_gaps": {m: [r["case_id"] for r in rs if F(r["source_acceptable_tie_controlled_gaps"][m]["gap"]) > 0] for m in ["first_only", "second_only"]},
            "source_acceptable_mean_strict_gap": {m: str(sum(F(r["source_acceptable_tie_controlled_gaps"][m]["gap"]) for r in rs) / len(rs)) for m in ["first_only", "second_only"]},
        }
    transitions = {p: {} for p in POLICIES}
    for policy in POLICIES:
        for mode in ["full", "first_only", "second_only"]:
            qs = [r for r in questions if r["policy"] == policy and r["mask"] == mode]
            transitions[policy][mode] = {
                "source_acceptable_harms": sum(F(r["source_acceptable_gain"]) < 0 for r in qs),
                "source_acceptable_helps": sum(F(r["source_acceptable_gain"]) > 0 for r in qs),
                "semantic_certified_harms": sum(F(r["semantic_gain"]["upper"]) < 0 for r in qs),
                "semantic_certified_helps": sum(F(r["semantic_gain"]["lower"]) > 0 for r in qs),
                "aspect_transitions": len(qs)}
    summary = {"scope": "same 12 development tasks; fixed-mask replay; no learned question selector",
               "model": manifest["model"], "revision": manifest["revision"], "unique_model_calls": len(generations),
               "model_invalid_outputs": sum(g["error"] is not None or parse_choice(g["raw_output"], [o["id"] for o in jobs[j]["payload"]["options"]]) is None for j, g in outputs.items()),
               "model_invalid_scheduled_choices": sum(r["model_chosen_id"] is None for r in choices if r["policy"] == "qwen3_4b"),
               "model_runtime_errors": sum(r["error"] is not None for r in generations),
               "model_length_finishes": sum(r["finish_reason"] == "length" for r in generations),
               "model_prompt_tokens": sum(r["prompt_tokens"] for r in generations),
               "model_generated_tokens": sum(r["generated_tokens"] for r in generations),
               "peak_memory_gb": max(r["peak_memory_gb"] or 0 for r in generations),
               "elapsed_seconds": run_status["elapsed_seconds"],
               "all_aspects_same_mask_diagnostic": all_aspects,
               "all_aspects_diagnostic_caveat": "full/partial applied to all four aspects is a four-question diagnostic, not the one-question comparison",
               "constant_one_question": constants, "ranking_diagnostics": ranking_summary,
               "information_transitions": transitions,
               "official_test_tasks_evaluated": 0, "paid_api_calls": 0,
               "limitations": ["provisional semantics", "known development tasks with template overlap", "synthetic information masks, no response probabilities", "aspect-factorized choices", "oracle question ranks use hidden outcomes"]}
    artifacts = {"executor_choices.json": choices, "executor_scores.json": scores,
                 "heuristic_traces.json": heuristic_traces, "question_outcomes.json": questions,
                 "ranking_details.json": ranks, "summary.json": summary}
    encoded = {name: encode(value) for name, value in artifacts.items()}
    result_manifest = {"input_manifest_sha256": sha(OUT / "input_manifest.json"),
        "reviewed_truth_sha256": sha(e26 / "reviewed_option_evaluations.json"),
        "review_manifest_sha256": sha(e26 / "review_manifest.json"),
        "run_sha256": {name: sha(OUT / name) for name in ["generations.jsonl", "started.jsonl", "run_status.json"]},
        "artifact_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()}}
    encoded["result_manifest.json"] = encode(result_manifest)
    for name, content in encoded.items():
        if args.verify:
            assert (OUT / name).read_bytes() == content, name
        else:
            with (OUT / name).open("xb") as f:
                f.write(content)
    print(json.dumps({k: v for k, v in summary.items() if k != "constant_one_question"}, indent=2))


if __name__ == "__main__":
    main()
