"""Independent exact replay verification conditional on the saved truth table.

Does not import the rule evaluator or diagnostic runner. It verifies computation,
not human meaning of the provisional criteria.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/userbench_semantics_e026"


def replay(rows):
    """Independent count-based recomputation, including all unknown-success counts."""
    groups = defaultdict(list)
    for row in rows:
        assert row["conjunction"] == ("F" if "F" in row["values"] else "T" if row["values"] == ["T", "T"] else "U")
        groups[(row["case_id"], row["aspect"])].append(row)
    probes, winners, fallbacks = {}, {}, []
    collapsed_completions = 0
    for case_id in sorted({key[0] for key in groups}):
        aspects = [key[1] for key in groups if key[0] == case_id]
        by_mode = {}
        for mode, mask in [("full", (0, 1)), ("first_only", (0,)), ("second_only", (1,)), ("no_information", ())]:
            values = {}
            for aspect in aspects:
                group = groups[(case_id, aspect)]
                selected = [r for r in group if not any(r["values"][i] == "F" for i in mask)]
                empty = not selected
                if empty:
                    selected = group
                    fallbacks.append((case_id, aspect, mode))
                # Every T/U conjunction survives filtering; removed options are F.
                assert {r["option_id"] for r in group if r["conjunction"] != "F"}.issubset({r["option_id"] for r in selected})
                nt = sum(r["conjunction"] == "T" for r in group)
                nu = sum(r["conjunction"] == "U" for r in group)
                # With uniform sampling, all binary completions collapse by number
                # of successful unknown options, so j=0..nu exhausts their values.
                outcomes = [Fraction(nt + j, len(selected)) / len(aspects) - Fraction(nt + j, len(group)) / len(aspects) for j in range(nu + 1)]
                collapsed_completions += len(outcomes)
                result = {"lower": str(min(outcomes)), "upper": str(max(outcomes))}
                q = "lodging" if aspect in {"hotel", "apartment"} else aspect
                values[q] = result
                probes[(case_id, q, mode)] = {"delta_bounds": result,
                    "kept_option_ids": [r["option_id"] for r in selected], "fallback_empty_set": empty}
            robust = []
            for q, value in values.items():
                if all(Fraction(value["lower"]) > Fraction(v["upper"]) for other, v in values.items() if other != q):
                    robust.append(q)
            assert len(robust) <= 1
            by_mode[mode] = robust[0] if robust else None
        winners[case_id] = by_mode
    reversals = {mode: [case_id for case_id, winner in winners.items() if winner["full"] and winner[mode] and winner["full"] != winner[mode]] for mode in ["first_only", "second_only"]}
    return probes, winners, reversals, fallbacks, collapsed_completions


def main():
    read = lambda name: json.loads((OUT / name).read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = read("manifest.json")
    for category in ["input_sha256", "code_sha256"]:
        for name, expected in manifest[category].items():
            assert digest(ROOT / name) == expected, name
    for name, expected in manifest["artifact_sha256"].items():
        assert digest(OUT / name) == expected, name
    rows = read("option_evaluations.json")
    assert len(rows) == len({(r["case_id"], r["aspect"], r["option_id"]) for r in rows}) == 864
    probes, winners, reversals, fallbacks, nc = replay(rows)
    for stored in read("mask_probes.json"):
        expected = probes[(stored["case_id"], stored["question_id"], stored["mask"])]
        for name in expected:
            assert stored[name] == expected[name], (stored["case_id"], name)
    assert len(probes) == len(read("mask_probes.json")) == 192
    for case in read("case_diagnostics.json"):
        assert case["robust_winners"] == winners[case["case_id"]]
    summary = read("summary.json")
    assert summary["robust_rank_reversals"] == reversals
    assert summary["empty_filter_fallbacks"] == len(fallbacks)
    for flag, key in [("source_best", "source_best_conjunction_counts"), ("source_acceptable", "source_acceptable_conjunction_counts")]:
        assert dict(Counter(r["conjunction"] for r in rows if r[flag])) == summary[key]
    result = {"status": "passed", "scope": "independent arithmetic replay conditional on saved provisional truth table",
              "predicate_semantics_independently_validated": False,
              "hashes_checked": True, "options": len(rows), "mask_probes": len(probes),
              "unknown_success_count_completions_checked": nc,
              "robust_rank_reversals": reversals, "fallbacks": fallbacks,
              "verifier_sha256": digest(Path(__file__)), "manifest_sha256": digest(OUT / "manifest.json")}
    path = OUT / "verification.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
