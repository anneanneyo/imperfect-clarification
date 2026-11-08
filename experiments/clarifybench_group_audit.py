"""Structural dependence audit; no models or third-party code are executed.

This does not define a final test split. It produces candidate grouping IDs so
that later eligible tasks can be split without separating exact scenario copies.
Only IDs, hashes and counts are saved, not upstream task content.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "a85d4f9df1fc87d05c713fd408f8a57d72bcc348"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def normalized_text(text):
    return " ".join(text.casefold().split())


def components(rows, keys):
    parent = {r["id"]: r["id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for r in rows:
        for key in keys:
            token = (key, r[key])
            if token in seen:
                a, b = find(r["id"]), find(seen[token])
                parent[max(a, b)] = min(a, b)
            else:
                seen[token] = r["id"]
    groups = defaultdict(list)
    for r in rows:
        groups[find(r["id"])].append(r["id"])
    return sorted((sorted(v) for v in groups.values()), key=lambda v: v[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/clarifybench_group_audit"))
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    assert commit == COMMIT, (commit, COMMIT)
    rows, empty = [], []
    for subset in ("A", "E", "I"):
        for p in sorted((args.repo / "ClarifyBench" / ("ClarifyBench_" + subset)).glob("*.json")):
            d = json.loads(p.read_text())
            if not d:
                empty.append(str(p.relative_to(args.repo)))
                continue
            domain, calls = d["primary_api"], d["ground_truth_tool_calls"]
            # Ignore annotations, preserving call order, argument names and values.
            semantic_calls = [{"tool_name": c["tool_name"], "parameters": c.get("parameters", {})}
                              for c in calls]
            row = {
                "id": subset + "/" + p.stem, "subset": subset, "domain": domain,
                "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "environment": digest([domain, d["initial_config"]]),
                "goal": digest([domain, calls]),
                "semantic_goal": digest([domain, semantic_calls]),
                "environment_goal": digest([domain, d["initial_config"], calls]),
                "initial_query": digest([domain, normalized_text(d["user_query"])]),
                "visible_initial_context": digest([domain, d["initial_config"], normalized_text(d["user_query"])]),
                "offline_full_scenario_context": digest([domain, d["initial_config"], normalized_text(d["user_query"]),
                                                [normalized_text(x) for x in d.get("potential_follow_ups", [])]]),
                "call_count": len(calls),
            }
            rows.append(row)
    schemes = {
        "exact_environment_and_goal": ["environment_goal"],
        "same_environment": ["environment"],
        "conservative_connected": ["environment", "semantic_goal", "initial_query"],
    }
    groupings = {}
    summary = {"source_commit": commit, "task_count": len(rows), "empty_paths": empty,
               "model_calls": 0, "status": "structural_audit_only_not_final_test", "schemes": {}}
    for name, keys in schemes.items():
        groups = components(rows, keys)
        groupings[name] = [{"group_id": digest(g), "members": g} for g in groups]
        summary["schemes"][name] = {"group_count": len(groups), "max_size": max(map(len, groups)),
                                      "size_histogram": dict(sorted(Counter(map(len, groups)).items()))}
        member_to_group = {member: digest(g) for g in groups for member in g}
        # Every declared equality edge must stay within a component.
        for key in keys:
            ids = defaultdict(set)
            for row in rows:
                ids[row[key]].add(member_to_group[row["id"]])
            assert all(len(v) == 1 for v in ids.values())
    summary["by_domain"] = {}
    for domain in sorted({r["domain"] for r in rows}):
        subset = [r for r in rows if r["domain"] == domain]
        summary["by_domain"][domain] = {"rows": len(subset), "types": dict(Counter(r["subset"] for r in subset)),
            **{key + "_count": len({r[key] for r in subset}) for key in ("environment", "semantic_goal", "environment_goal")},
            "conservative_group_count": len(components(subset, schemes["conservative_connected"]))}
    summary["cross_subset_overlap"] = {}
    for key in ("environment", "semantic_goal", "environment_goal", "initial_query"):
        by_subset = {s: {r[key] for r in rows if r["subset"] == s} for s in "AEI"}
        summary["cross_subset_overlap"][key] = {
            a + "_" + b: len(by_subset[a] & by_subset[b]) for a, b in (("A", "E"), ("A", "I"), ("E", "I"))}
    # Multiple annotated goals under identical initial context do not by
    # themselves identify a response distribution or a preference prior.
    summary["context_goal_multiplicity"] = {}
    # The second key includes future follow-ups: it is offline metadata only.
    # Neither key implies that the full initial DB is exposed to the agent.
    for key in ("visible_initial_context", "offline_full_scenario_context"):
        by_context = defaultdict(list)
        for row in rows:
            by_context[row[key]].append(row)
        ambiguous = [v for v in by_context.values() if len({r["semantic_goal"] for r in v}) > 1]
        summary["context_goal_multiplicity"][key] = {
            "context_count": len(by_context), "contexts_with_distinct_call_sequences": len(ambiguous),
            "affected_rows": sum(map(len, ambiguous)), "groups": [[r["id"] for r in v] for v in ambiguous]}
    sample_hashes = [hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((args.repo / "ClarifyBench" / "sample").glob("*.json"))]
    main_hashes = {r["file_sha256"] for r in rows}
    summary["sample"] = {"count": len(sample_hashes), "byte_identical_to_main": sum(h in main_hashes for h in sample_hashes)}
    summary["rows_manifest_sha256"] = digest(rows)
    summary["checks"] = {"unique_record_ids": len({r["id"] for r in rows}) == len(rows),
                          "all_declared_group_edges_preserved": True,
                          "no_task_text_saved": True, "no_split_claim": True}
    assert all(summary["checks"].values())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("records.json", rows), ("groups.json", groupings)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
