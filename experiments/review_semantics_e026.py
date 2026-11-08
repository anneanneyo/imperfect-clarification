"""Post-run, explicitly separated name-normalization sensitivity for E026.

Never changes the locked catalogue, primary truth table or original replay.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from semantics_e026 import normalize, phrase_in, conjunction
from verify_semantics_e026 import replay

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/userbench_semantics_e026"
TARGET = "E025-07/apartment/0"


def main():
    read = lambda name: json.loads((OUT / name).read_text())
    rows = read("option_evaluations.json")
    updated = copy.deepcopy(rows)
    public = json.loads((ROOT / "results/userbench_train_e025/public_cases.json").read_text())
    case = next(c for c in public if c["case_id"] == "E025-07")
    options = {o["id"]: o for o in case["options"]["apartment"]}
    changes = []
    for old, row in zip(rows, updated):
        if TARGET not in row["criteria"]:
            continue
        name = options[row["option_id"]].get("name")
        value = "U" if not isinstance(name, str) else "T" if any(phrase_in(name, alias) for alias in ["Zillow Rental", "Zillow Rentals"]) else "F"
        i = row["criteria"].index(TARGET)
        row["values"][i] = value
        row["conjunction"] = conjunction(row["values"])
        if row != old:
            changes.append({"case_id": row["case_id"], "aspect": row["aspect"], "source_id": row["source_id"],
                            "option_id": row["option_id"], "name": name, "before": old["values"],
                            "after": row["values"], "conjunction_before": old["conjunction"], "conjunction_after": row["conjunction"]})
    probes, winners, reversals, fallbacks, nc = replay(updated)
    serialized_probes = [dict(case_id=k[0], question_id=k[1], mask=k[2], **v) for k, v in sorted(probes.items())]
    output = {"reviewed_option_evaluations.json": updated, "reviewed_mask_probes.json": serialized_probes}
    summary = {
        "status": "post-run sensitivity; primary files unchanged",
        "reason": "The literal-name rule falsely rejected the plural Zillow Rentals in displayed listing names.",
        "change": {"criterion_id": TARGET, "accepted_explicit_name_forms": ["Zillow Rental", "Zillow Rentals"]},
        "changes": changes, "option_change_count": len(changes),
        "conjunction_counts": dict(Counter(r["conjunction"] for r in updated)),
        "source_best_conjunction_counts": dict(Counter(r["conjunction"] for r in updated if r["source_best"])),
        "source_acceptable_conjunction_counts": dict(Counter(r["conjunction"] for r in updated if r["source_acceptable"])),
        "robust_rank_reversals": reversals, "robust_winners": winners, "fallbacks": fallbacks,
        "unknown_success_count_completions_checked": nc,
        "limitations": "This repairs one demonstrated name mismatch. It is not independent semantic validation or a general alias policy. U from unresolved daily cleaning remains U.",
    }
    output["review_summary.json"] = summary
    encoded = {name: (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode() for name, obj in output.items()}
    manifest = {"primary_manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["experiments/review_semantics_e026.py", "experiments/verify_semantics_e026.py"]},
                "artifact_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in encoded.items()}}
    encoded["review_manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    for name, content in encoded.items():
        path = OUT / name
        if path.exists():
            assert path.read_bytes() == content, name
        else:
            path.write_bytes(content)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
