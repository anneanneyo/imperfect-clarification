"""Reproduce the fixed E025 train-only source audit (no model calls)."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from userbench_choice_adapter import prepare_case, score_choices

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "80506d2ab484cab843e60a2401ff3e0290d05b87"
FILES = {
    "data/travel2222_multiturn/train.parquet": "0eaea0fa79df05a7bdd165b323a79b10fb9e15c7",
    "travelgym/data/travelgym_data_2222.json": "974c4457d1ace709ea3f2652ddd2dbb405db9052",
}
PROTOCOL_SHA = "456e6be5d2c01f3a76cee243f751e4883a21209d0e1e8c97edbd5cbb13a921bd"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def checked_file(cache, name, download=False):
    path = cache / name
    if not path.exists() and download:
        url = "https://raw.githubusercontent.com/SalesforceAIResearch/UserBench/" + COMMIT + "/" + name
        data = urllib.request.urlopen(url, timeout=60).read()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == FILES[name]
    return data


def audit(cache, download=False):
    import pyarrow.parquet as pq
    import pyarrow
    import io
    assert sha((ROOT / "research/E025_PROTOCOL.md").read_bytes()) == PROTOCOL_SHA
    source = {name: checked_file(cache, name, download) for name in FILES}
    table = pq.read_table(io.BytesIO(source["data/travel2222_multiturn/train.parquet"]))
    rows = table.to_pylist()
    ids = [str(r["reward_model"]["id"]) for r in rows]
    assert len(ids) == len(set(ids))
    assert all(r["extra_info"]["split"] == "train" for r in rows)
    selected = sorted(ids, key=lambda i: (sha(("E025|" + i).encode()), i))[:12]
    assert selected == json.loads((ROOT / "results/userbench_train_e025/selection.json").read_text())["selected_ids"]
    # Parse the shared upstream container; never iterate over nonselected tasks.
    container = json.loads(source["travelgym/data/travelgym_data_2222.json"])
    tasks = {i: container[i] for i in selected}
    del container
    by_id = {str(r["reward_model"]["id"]): r for r in rows}
    public, private, checks, restaurant, preference_inventory = [], [], [], [], []
    exact_hidden_hits, gold_id_hits, metadata_mismatches = [], [], []
    costs = {"Cheap": 0, "Normal": 1, "Expensive": 2}
    for index, task_id in enumerate(selected):
        task = tasks[task_id]
        row = by_id[task_id]
        messages = row["prompt"]
        assert [m["role"] for m in messages] == ["system", "user"]
        user = messages[1]["content"]
        full_prompt = "\n".join(m["content"] for m in messages)
        case_id = "E025-%02d" % (index + 1)
        assert task["initial_description"] in user
        source_best = sorted(task[d]["best_id"] for d in task["dimensions"])
        assert sorted(ast.literal_eval(row["reward_model"]["ground_truth"])) == source_best
        pub, gold = prepare_case(task, user, case_id)
        public.append(pub)
        private.append(gold)
        for d in task["dimensions"]:
            data = task[d]
            options = [o for group in data["options"].values() for o in group]
            option_ids = [o["id"] for o in options]
            assert len(option_ids) == len(set(option_ids))
            missing_metadata_ids = sorted(set(data["all_ids"]) - set(option_ids))
            assert set(option_ids).issubset(data["all_ids"])
            group_mismatches = {}
            for typ in ["correct", "wrong", "noise"]:
                actual = {o["id"] for o in data["options"][typ]}
                declared = set(data[typ + "_ids"])
                assert actual.issubset(declared)
                if actual != declared:
                    group_mismatches[typ] = sorted(declared - actual)
            if missing_metadata_ids or group_mismatches:
                metadata_mismatches.append({"case_id": case_id, "aspect": d,
                    "all_ids_without_option": missing_metadata_ids,
                    "group_ids_without_option": group_mismatches})
            hits = [i for i in option_ids if re.search(r"\b" + re.escape(i) + r"\b", full_prompt)]
            gold_id_hits.extend({"case_id": case_id, "id": i} for i in hits)
            for j, pref in enumerate(data["preferences"]):
                preference_inventory.append({"case_id": case_id, "task_id": task_id,
                    "aspect": d, "index": j, "category": pref[1],
                    "preference": pref[2], "implicit_statement": pref[3],
                    "generation_criterion": pref[4]})
                for field in [2, 3, 4]:
                    if pref[field].casefold() in full_prompt.casefold():
                        exact_hidden_hits.append({"case_id": case_id, "aspect": d,
                                                  "preference_index": j, "field": field})
            checks.append({"case_id": case_id, "task_id": task_id, "aspect": d,
                "option_count": len(options), "preference_count": len(data["preferences"]),
                "option_group_counts": {k: len(v) for k, v in data["options"].items()},
                "best_id": data["best_id"], "correct_ids": data["correct_ids"],
                "smallest_original_id": min(option_ids, key=lambda v: int(v[1:])),
                "first_raw_correct_id": data["options"]["correct"][0]["id"]})
        d = task["restaurant"]
        correct = d["options"]["correct"]
        tiers = [costs[o["expectation"]] for o in correct]
        cheapest = [o["id"] for o in correct if costs[o["expectation"]] == min(tiers)]
        restaurant.append({"case_id": case_id, "task_id": task_id,
            "source_best_id": d["best_id"], "cheapest_tier_ids": cheapest,
            "numeric_cost_present": any(isinstance(o.get("cost"), (int, float)) for o in correct),
            "best_rationale": d["best_id_thought"],
            "implicit_statements": [p[3] for p in d["preferences"]],
            "criteria": [p[4] for p in d["preferences"]],
            "correct_options": correct})
    components = Counter(c for i in ids for c in i.split("|"))
    selected_components = Counter(c for i in selected for c in i.split("|"))
    # Connected components of train task IDs joined by a shared template ID.
    unseen, graph_sizes = set(ids), []
    while unseen:
        pending, found = [min(unseen)], set()
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            parts = set(current.split("|"))
            pending.extend(i for i in unseen - found if parts.intersection(i.split("|")))
        unseen -= found
        graph_sizes.append(len(found))
    baseline_results = []
    for pub, gold, task_id in zip(public, private, selected):
        task = tasks[task_id]
        choices = []
        for d in pub["aspects"]:
            ids_here = gold["aspects"][d]["source_to_visible"]
            source_id = min(ids_here, key=lambda i: int(i[1:]))
            choices.append((d, ids_here[source_id]))
        baseline_results.append(score_choices(gold, choices))
    summary = {
        "scope": "fixed train-only source audit; not benchmark evaluation or learned agent",
        "upstream_commit": COMMIT, "configuration": "travel2222_multiturn",
        "train_rows": len(rows), "unique_train_ids": len(ids),
        "selected_tasks": len(selected), "selected_aspects": len(checks),
        "selected_preferences": len(preference_inventory),
        "selected_options": sum(c["option_count"] for c in checks),
        "source_index_best_id_agreement": len(selected),
        "all_selected_best_and_correct_membership_checks_passed": True,
        "metadata_mismatches": metadata_mismatches,
        "exact_hidden_text_hits_in_initial_prompt": exact_hidden_hits,
        "option_id_hits_in_initial_prompt": gold_id_hits,
        "prompt_check_limit": "exact full-string/ID matches only; semantic absence of leakage not proved",
        "restaurant_minimum_tier_ties": sum(len(r["cheapest_tier_ids"]) > 1 for r in restaurant),
        "restaurant_cases": len(restaurant),
        "restaurant_correct_options_with_numeric_cost": sum(isinstance(o.get("cost"), (int, float)) for r in restaurant for o in r["correct_options"]),
        "restaurant_best_in_minimum_tier": sum(r["source_best_id"] in r["cheapest_tier_ids"] for r in restaurant),
        "train_component_id_count": sum(components.values()),
        "train_unique_component_ids": len(components),
        "train_reused_component_id_types": sum(n > 1 for n in components.values()),
        "train_task_connected_component_sizes": sorted(graph_sizes, reverse=True),
        "selected_unique_component_ids": len(selected_components),
        "selected_reused_component_ids": {k: v for k, v in sorted(selected_components.items()) if v > 1},
        "component_caveat": "component IDs denote preference templates; instantiated statements/options can differ",
        "smallest_original_id_diagnostic": {
            "best_count": sum(r["best_count"] for r in baseline_results),
            "acceptable_count": sum(r["acceptable_count"] for r in baseline_results),
            "aspect_count": len(checks),
            "caveat": "exploratory ID shortcut check, not a prespecified model comparison"},
        "raw_correct_first_diagnostic": {
            "best_count": sum(c["first_raw_correct_id"] == c["best_id"] for c in checks),
            "acceptable_count": len(checks), "aspect_count": len(checks),
            "caveat": "uses hidden category ordering; invalid policy; upstream loader shuffles"},
        "model_calls": 0, "official_test_tasks_evaluated": 0,
    }
    artifacts = {"summary.json": summary, "aspect_checks.json": checks,
                 "preference_inventory.json": preference_inventory,
                 "restaurant_evidence.json": restaurant,
                 "public_cases.json": public, "private_cases.json": private}
    output = {name: encoded(value) for name, value in artifacts.items()}
    manifest = {
        "protocol_sha256": PROTOCOL_SHA, "upstream_commit": COMMIT,
        "pyarrow_version": pyarrow.__version__,
        "source_files": {name: {"git_blob": FILES[name], "sha256": sha(value), "bytes": len(value)} for name, value in source.items()},
        "artifact_sha256": {name: sha(value) for name, value in output.items()},
        "code_sha256": {name: sha((ROOT / name).read_bytes()) for name in [
            "experiments/audit_userbench_train_e025.py", "experiments/userbench_choice_adapter.py"]},
    }
    output["manifest.json"] = encoded(manifest)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path.home() / ".cache/clarification-research/userbench-80506d2")
    p.add_argument("--output", type=Path, default=ROOT / "results/userbench_train_e025")
    p.add_argument("--download", action="store_true", help="Download only the two pinned source files if absent")
    p.add_argument("--verify", action="store_true", help="Recompute and compare; never rewrite results")
    args = p.parse_args()
    output = audit(args.cache, args.download)
    for name, data in output.items():
        dest = args.output / name
        if args.verify:
            assert dest.read_bytes() == data, "Artifact differs: " + name
        else:
            if dest.exists():
                raise FileExistsError("Refusing to overwrite " + str(dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    print(json.dumps({"status": "verified" if args.verify else "created", "files": len(output),
                      "summary": json.loads(output["summary.json"])}, indent=2))


if __name__ == "__main__":
    main()
