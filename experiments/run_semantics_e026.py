"""E026 symbolic, bounded information-mask diagnostic; no model calls."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from semantics_e026 import evaluate, conjunction

ROOT = Path(__file__).resolve().parents[1]
E025 = ROOT / "results/userbench_train_e025"
OUT = ROOT / "results/userbench_semantics_e026"
MASKS = {"full": [0, 1], "first_only": [0], "second_only": [1], "no_information": []}
QUESTIONS = {
    "flight": "What requirements matter to you when choosing the flight?",
    "lodging": "What requirements matter to you when choosing the accommodation?",
    "rental_car": "What requirements matter to you when choosing the rental car?",
    "restaurant": "What requirements matter to you when choosing the restaurant?",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def bounds(coefficients, outcomes):
    lower = upper = F(0)
    for coefficient, outcome in zip(coefficients, outcomes):
        if outcome == "T":
            lower += coefficient; upper += coefficient
        elif outcome == "U":
            lower += min(F(0), coefficient); upper += max(F(0), coefficient)
    return {"lower": str(lower), "upper": str(upper)}


def run():
    read = lambda path: json.loads(path.read_text())
    lock = read(OUT / "criteria_lock.json")
    assert sha((OUT / "criteria.json").read_bytes()) == lock["criteria_sha256"]
    assert sha((ROOT / "experiments/build_criteria_e026.py").read_bytes()) == lock["builder_sha256"]
    assert sha((ROOT / "research/E026_PROTOCOL.md").read_bytes()) == read(OUT / "preinspection.json")["protocol_sha256"]
    parent_manifest = read(E025 / "manifest.json")
    for name in ["public_cases.json", "private_cases.json", "preference_inventory.json"]:
        assert sha((E025 / name).read_bytes()) == parent_manifest["artifact_sha256"][name]
    criteria = read(OUT / "criteria.json")
    by_key = {(c["case_id"], c["aspect"], c["index"]): c for c in criteria}
    public, hidden = read(E025 / "public_cases.json"), read(E025 / "private_cases.json")
    rows, probes, cases = [], [], []
    for case, gold in zip(public, hidden):
        case_id = case["case_id"]
        assert case_id == gold["case_id"]
        q_aspect = {q: q if q != "lodging" else next(d for d in case["aspects"] if d in {"hotel", "apartment"}) for q in QUESTIONS}
        aspect_rows = {}
        baseline_lower = baseline_upper = F(0)
        for aspect in case["aspects"]:
            pairs = [by_key[(case_id, aspect, i)] for i in range(2)]
            group = []
            inverse = {v: k for k, v in gold["aspects"][aspect]["source_to_visible"].items()}
            for option in case["options"][aspect]:
                values = [evaluate(c, option) for c in pairs]
                row = {"case_id": case_id, "aspect": aspect, "option_id": option["id"],
                       "source_id": inverse[option["id"]], "criteria": [c["criterion_id"] for c in pairs],
                       "values": values, "conjunction": conjunction(values),
                       "source_best": option["id"] == gold["aspects"][aspect]["best_id"],
                       "source_acceptable": option["id"] in gold["aspects"][aspect]["acceptable_ids"]}
                group.append(row); rows.append(row)
            aspect_rows[aspect] = group
            n = len(group)
            base = bounds([F(1, n)] * n, [r["conjunction"] for r in group])
            baseline_lower += F(base["lower"]) / len(case["aspects"])
            baseline_upper += F(base["upper"]) / len(case["aspects"])
        case_probes = []
        for q, aspect in q_aspect.items():
            group = aspect_rows[aspect]
            n, n_aspects = len(group), len(case["aspects"])
            for mode, indices in MASKS.items():
                kept = [j for j, row in enumerate(group) if all(row["values"][i] != "F" for i in indices)]
                empty = not kept
                if empty:
                    kept = list(range(n))
                coefficients = [(F(1, len(kept)) if j in kept else F(0)) - F(1, n) for j in range(n)]
                gain = bounds([c / n_aspects for c in coefficients], [r["conjunction"] for r in group])
                item = {"case_id": case_id, "question_id": q, "aspect": aspect, "mask": mode,
                        "revealed_indices": indices, "kept_option_ids": [group[j]["option_id"] for j in kept],
                        "fallback_empty_set": empty, "delta_bounds": gain}
                probes.append(item); case_probes.append(item)
        winners = {}
        for mode in MASKS:
            candidates = [r for r in case_probes if r["mask"] == mode]
            robust = [r["question_id"] for r in candidates if all(
                F(r["delta_bounds"]["lower"]) > F(other["delta_bounds"]["upper"])
                for other in candidates if other["question_id"] != r["question_id"])]
            assert len(robust) <= 1
            winners[mode] = robust[0] if robust else None
        cases.append({"case_id": case_id, "question_aspects": q_aspect,
                      "baseline_satisfaction_bounds": {"lower": str(baseline_lower), "upper": str(baseline_upper)},
                      "robust_winners": winners,
                      "robust_reversals": {mode: bool(winners["full"] and winners[mode] and winners["full"] != winners[mode]) for mode in ["first_only", "second_only"]}})
    status = Counter(c["status"] for c in criteria)
    summary = {
        "scope": "development; provisional semantics; oracle symbolic filter and uniform executor; no learned policy",
        "tasks": len(cases), "aspects": sum(len(c["aspects"]) for c in public), "criteria": len(criteria),
        "criterion_status_counts": dict(status), "option_count": len(rows),
        "predicate_evaluation_counts": dict(Counter(v for r in rows for v in r["values"])),
        "conjunction_counts": dict(Counter(r["conjunction"] for r in rows)),
        "source_best_conjunction_counts": dict(Counter(r["conjunction"] for r in rows if r["source_best"])),
        "source_acceptable_conjunction_counts": dict(Counter(r["conjunction"] for r in rows if r["source_acceptable"])),
        "mask_probes": len(probes), "empty_filter_fallbacks": sum(r["fallback_empty_set"] for r in probes),
        "robust_winner_counts": {mode: dict(Counter(c["robust_winners"][mode] or "unresolved_or_tied" for c in cases)) for mode in MASKS},
        "robust_rank_reversals": {mode: [c["case_id"] for c in cases if c["robust_reversals"][mode]] for mode in ["first_only", "second_only"]},
        "all_no_information_deltas_zero": all(F(r["delta_bounds"]["lower"]) == F(r["delta_bounds"]["upper"]) == 0 for r in probes if r["mask"] == "no_information"),
        "mean_no_information_satisfaction_bounds": {key: str(sum(F(c["baseline_satisfaction_bounds"][key]) for c in cases) / len(cases)) for key in ["lower", "upper"]},
        "model_calls": 0, "official_test_tasks_evaluated": 0,
        "limitations": ["No independently validated semantic ground truth", "Preference conjunction only, not trip feasibility or cost", "No user response probabilities or latent-goal prior", "Masks based on source ordering, not a realistic response model", "U intervals are completion bounds, not confidence intervals", "Source-best/source-correct kept unchanged"]}
    artifacts = {"option_evaluations.json": rows, "mask_probes.json": probes,
                 "case_diagnostics.json": cases, "summary.json": summary, "question_menu.json": QUESTIONS}
    output = {k: encode(v) for k, v in artifacts.items()}
    input_paths = [E025 / name for name in ["public_cases.json", "private_cases.json", "preference_inventory.json", "selection.json", "manifest.json"]] + [OUT / name for name in ["criteria.json", "criteria_lock.json", "preinspection.json"]] + [ROOT / "research/E026_PROTOCOL.md"]
    manifest = {"input_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in input_paths},
                "code_sha256": {name: sha((ROOT / name).read_bytes()) for name in ["experiments/build_criteria_e026.py", "experiments/semantics_e026.py", "experiments/run_semantics_e026.py"]},
                "artifact_sha256": {name: sha(value) for name, value in output.items()}}
    output["manifest.json"] = encode(manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = run()
    for name, content in output.items():
        path = OUT / name
        if args.verify:
            assert path.read_bytes() == content, name
        else:
            with path.open("xb") as f:
                f.write(content)
    print(json.dumps({"status": "verified" if args.verify else "created", "summary": json.loads(output["summary.json"])}, indent=2))


if __name__ == "__main__":
    main()
