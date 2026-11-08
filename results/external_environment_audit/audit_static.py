"""Read-only structural audit. Does not import or execute benchmark source code."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inventory(root):
    files = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "tracked_file_count": len(files),
        "license_files": [p for p in files if any(x in Path(p).name.lower() for x in ("license", "licence", "copying", "notice"))],
        "tracked_manifest_sha256": digest([(p, hashlib.sha256((root / p).read_bytes()).hexdigest()) for p in sorted(files)]),
        "files": files,
    }


def clarify_stats(root):
    groups = {}
    for directory in sorted((root / "ClarifyBench").iterdir()):
        if not directory.is_dir():
            continue
        rows = [(p, json.loads(p.read_text())) for p in sorted(directory.glob("*.json"))]
        valid = [(p, row) for p, row in rows if isinstance(row, dict) and all(k in row for k in ("user_query", "ground_truth_tool_calls", "primary_api"))]
        repeated = []
        repeated_distinct = []
        for p, row in valid:
            by_name = {}
            for call in row["ground_truth_tool_calls"]:
                by_name.setdefault(call["tool_name"], []).append(call.get("parameters", {}))
            if any(len(v) > 1 for v in by_name.values()):
                repeated.append(p.stem)
            if any(len({digest(x) for x in v}) > 1 for v in by_name.values()):
                repeated_distinct.append(p.stem)
        groups[directory.name] = {
            "json_files": len(rows),
            "task_records": len(valid),
            "invalid_or_empty_paths": [str(p.relative_to(root)) for p, row in rows if not row or "user_query" not in row],
            "domains": dict(sorted(Counter(row["primary_api"] for _, row in valid).items())),
            "tool_calls": sum(len(row["ground_truth_tool_calls"]) for _, row in valid),
            "records_with_missing_turn_annotation": sum(any("turn" not in c for c in row["ground_truth_tool_calls"]) for _, row in valid),
            "repeated_tool_name_record_ids": repeated,
            "repeated_tool_name_distinct_params_record_ids": repeated_distinct,
            "ids": [p.stem for p, _ in valid],
            "file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in rows},
            "environment_goal_hashes": {p.stem: digest({k: row.get(k) for k in ("primary_api", "initial_config", "ground_truth_tool_calls")}) for p, row in valid},
        }
    overlaps = {}
    names = sorted(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = groups[left], groups[right]
            overlaps[left + "__" + right] = {
                "record_ids": sorted(set(a["ids"]) & set(b["ids"])),
                "environment_goal_hash_overlap_count": len(set(a["environment_goal_hashes"].values()) & set(b["environment_goal_hashes"].values())),
            }
    return {"subsets": groups, "overlaps": overlaps}


def tau_counts(root):
    out = {}
    for path in sorted((root / "tau_bench" / "envs").glob("*/tasks_*.py")):
        tree = ast.parse(path.read_text())
        counts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                for name in node.targets:
                    if isinstance(name, ast.Name) and name.id.startswith("TASKS"):
                        counts[name.id] = len(node.value.elts)
        out[str(path.relative_to(root))] = counts
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clarify", type=Path, required=True)
    parser.add_argument("--ig", type=Path, required=True)
    parser.add_argument("--tau", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "audit_date": "2026-10-27",
        "method": "Own stdlib reader of tracked files, JSON and Python AST; no benchmark code imported or executed; no model calls.",
        "model_calls": 0,
        "clarifybench": {**inventory(args.clarify), **clarify_stats(args.clarify)},
        "ig_clarifier": inventory(args.ig),
        "original_tau_bench": {**inventory(args.tau), "task_counts": tau_counts(args.tau)},
    }
    assert result["clarifybench"]["commit"] == "a85d4f9df1fc87d05c713fd408f8a57d72bcc348"
    assert result["ig_clarifier"]["commit"] == "4071af1140d434e713767a0d837e31918b63adec"
    assert result["original_tau_bench"]["commit"] == "59a200c6d575d595120f1cb70fea53cef0632f6b"
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "static_audit_complete", "model_calls": 0,
        "clarifybench_tasks": {k: v["task_records"] for k, v in result["clarifybench"]["subsets"].items()},
        "tau_task_counts": result["original_tau_bench"]["task_counts"]}, indent=2))


if __name__ == "__main__":
    main()
