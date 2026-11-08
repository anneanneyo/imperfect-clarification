"""Pinned, offline domain-component replay. NOT an official tau benchmark run.

Run with Python 3.12 and Pydantic 2; no other non-stdlib dependencies.
Upstream source and tasks remain in a separate checkout. No LLM calls.
"""
import argparse
import ast
from collections import Counter
import copy
from enum import Enum
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import typing

import pydantic
from pydantic import BaseModel, ConfigDict, Field

COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
HASHES = {
    "src/tau2/domains/retail/data_model.py": "f5ad09590953855cc49d2ab2ad5c0594be408cae74d1500fb278e5a6bf2c50d6",
    "src/tau2/domains/retail/tools.py": "34f4fe9702d36f0dd4697700fbcd63d0cf5386e6eb8e7499aaa336243bad8750",
    "src/tau2/environment/toolkit.py": "3e897891e8878c3399aec72e42eb0eb5546642420c0c902b225421fc936b924a",
    "src/tau2/environment/db.py": "49e12f900a8aabe62d6137a9baa1e7d471f9b3e516cf7c85fb61ecde8a65154c",
    "src/tau2/utils/pydantic_utils.py": "618dfc39b619bb0f623151682962ddc52bf2ea70bc2562dded884f118c84b78a",
    "data/tau2/domains/retail/db.json": "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765",
    "data/tau2/domains/retail/tasks.json": "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8",
    "data/tau2/domains/retail/split_tasks.json": "ed0580ec52575b63fbf76568af42490da6ee7783ecb4aa81af46961291358f20",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def load_domain(repo):
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == COMMIT
    for filename, expected in HASHES.items():
        assert hashlib.sha256((repo / filename).read_bytes()).hexdigest() == expected, filename
    # Definitions are selected from the reviewed pinned files. All package-level
    # imports, startup paths and demo blocks are excluded. Method bodies are intact.
    ns = {name: getattr(typing, name) for name in ("Any", "Dict", "List", "Literal", "Optional", "Union", "Callable", "TypeVar")}
    ns.update({"__name__": __name__, "BaseModel": BaseModel, "ConfigDict": ConfigDict,
               "Field": Field, "Enum": Enum, "json": json})
    selections = [
        ("src/tau2/utils/pydantic_utils.py", {"BaseModelNoExtra"}),
        ("src/tau2/environment/db.py", {"DB"}),
        ("src/tau2/domains/retail/data_model.py", None),
        ("src/tau2/environment/toolkit.py", {"ToolKitType", "ToolType", "is_tool", "ToolKitBase"}),
        ("src/tau2/domains/retail/tools.py", {"RetailTools"}),
    ]
    toolkit_methods = {"__init__", "tools", "use_tool", "tool_type", "tool_mutates_state"}
    extracted = []
    for filename, names in selections:
        tree = ast.parse((repo / filename).read_text(), filename)
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and (names is None or filename.endswith("toolkit.py")):
                nodes.append(node)
            elif isinstance(node, ast.ClassDef) and (names is None or node.name in names):
                if node.name == "ToolKitBase":
                    # Only domain dispatch is needed; omit prompt/schema helpers.
                    node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in toolkit_methods]
                nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and names is not None and node.name in names:
                nodes.append(node)
        extracted.append({"file": filename, "ast_sha256": hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[])).encode()).hexdigest()})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(repo / filename), "exec"), ns)
    return ns["RetailDB"], ns["RetailTools"], extracted


def replay(db_class, tool_class, raw_db, actions):
    db = db_class.model_validate(copy.deepcopy(raw_db))
    tools = tool_class(db)
    errors, metadata_mismatches = [], []
    for i, action in enumerate(actions):
        before_items = {}
        if action["name"] == "modify_pending_order_items":
            before_order = db.orders.get(action["arguments"].get("order_id"))
            if before_order is not None:
                before_items = {
                    id(item): (item.item_id, item.price, copy.deepcopy(item.options))
                    for item in before_order.items
                }
        try:
            tools.use_tool(action["name"], **copy.deepcopy(action["arguments"]))
            if action["name"] == "modify_pending_order_items":
                order = db.orders[action["arguments"]["order_id"]]
                # Review repair: inspect the objects actually changed by this call.
                # Searching only by new_id can pick an untouched pre-existing
                # duplicate whose historical price legitimately differs.
                for item in order.items:
                    current = (item.item_id, item.price, item.options)
                    if before_items.get(id(item)) == current:
                        continue
                    variant = db.products[item.product_id].variants[item.item_id]
                    if item.price != variant.price or item.options != variant.options:
                        metadata_mismatches.append({"action_index": i, "item_hash": digest(item.item_id),
                                                   "price_mismatch": item.price != variant.price,
                                                   "options_mismatch": item.options != variant.options})
        except Exception as e:
            errors.append({"action_index": i, "tool": action["name"], "type": type(e).__name__, "message": str(e)})
    return {"state_hash": digest(db.model_dump()), "errors": errors,
            "all_calls_succeeded": not errors, "changed_item_metadata_mismatches": metadata_mismatches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/tau3_retail_state_probe"))
    args = parser.parse_args()
    db_class, tool_class, extracted = load_domain(args.repo)
    data_dir = args.repo / "data/tau2/domains/retail"
    raw_db = json.loads((data_dir / "db.json").read_text())
    tasks = json.loads((data_dir / "tasks.json").read_text())
    split = json.loads((data_dir / "split_tasks.json").read_text())
    assert all(t.get("initial_state") is None for t in tasks)
    assert len(tasks) == 114
    assert set(split["train"]).isdisjoint(split["test"])
    proto = tool_class(db_class.model_validate(copy.deepcopy(raw_db)))
    def writes(actions):
        return [i for i, a in enumerate(actions) if proto.tool_mutates_state(a["name"])]
    fixtures = sorted((t for t in tasks if t["id"] in split["train"] and writes(t["evaluation_criteria"]["actions"])),
                      key=lambda t: hashlib.sha256(("E007-v1:" + t["id"]).encode()).hexdigest())[:12]
    args.output.mkdir(parents=True, exist_ok=True)
    fixture_manifest = {"selected_before_execution": True, "seed_string": "E007-v1:",
                        "train_task_ids": [t["id"] for t in fixtures], "source_commit": COMMIT}
    (args.output / "fixtures.json").write_text(json.dumps(fixture_manifest, indent=2) + "\n")
    initial_hash = digest(proto.db.model_dump())
    rows, mutants = [], []
    for task in tasks:
        calls = task["evaluation_criteria"]["actions"]
        result = replay(db_class, tool_class, raw_db, calls)
        repeated = replay(db_class, tool_class, raw_db, calls)
        assert result == repeated, task["id"]
        rows.append({"task_id": task["id"], "split": "train" if task["id"] in split["train"] else "test",
                     "call_count": len(calls), "write_count": len(writes(calls)),
                     "db_changed": result["state_hash"] != initial_hash, **result})
        if task not in fixtures:
            continue
        first_write = writes(calls)[0]
        wrong = copy.deepcopy(calls)
        id_key = "order_id" if "order_id" in wrong[first_write]["arguments"] else "user_id"
        assert id_key in wrong[first_write]["arguments"]
        wrong[first_write]["arguments"][id_key] = "__audit_nonexistent__"
        user_id = sorted(raw_db["users"])[0]
        address = copy.deepcopy(raw_db["users"][user_id]["address"])
        address["address1"] = "__audit_unrequested_change__"
        extra_write = {"name": "modify_user_address", "arguments": {"user_id": user_id, **address}}
        variants = {
            "no_action": [],
            "write_only": [copy.deepcopy(calls[i]) for i in writes(calls)],
            "drop_first_write": [copy.deepcopy(a) for i, a in enumerate(calls) if i != first_write],
            "wrong_write_id": wrong,
            "extra_read": copy.deepcopy(calls) + [{"name": "list_all_product_types", "arguments": {}}],
            "extra_unrequested_write": copy.deepcopy(calls) + [extra_write],
            "reverse": list(reversed(copy.deepcopy(calls))),
            "duplicate_first_write": copy.deepcopy(calls[:first_write + 1]) + [copy.deepcopy(calls[first_write])] + copy.deepcopy(calls[first_write + 1:]),
        }
        for label, actions in variants.items():
            mutated = replay(db_class, tool_class, raw_db, actions)
            mutants.append({"task_id": task["id"], "variant": label,
                            "gold_all_calls_succeeded": result["all_calls_succeeded"],
                            "db_matches_reference": mutated["state_hash"] == result["state_hash"], **mutated})
    summary = {
        "experiment": "E007", "source_commit": COMMIT, "source_tag": "v1.0.1",
        "scope": "offline_retail_domain_component_not_official_task_success",
        "model_calls": 0, "planned_gold_tasks": len(tasks), "completed_gold_tasks": len(rows),
        "gold_all_calls_succeeded": sum(r["all_calls_succeeded"] for r in rows),
        "gold_with_call_errors": sum(not r["all_calls_succeeded"] for r in rows),
        "gold_db_changed": sum(r["db_changed"] for r in rows),
        "gold_tasks_with_changed_item_metadata_mismatches": sum(bool(r["changed_item_metadata_mismatches"]) for r in rows),
        "gold_changed_item_metadata_mismatch_count": sum(len(r["changed_item_metadata_mismatches"]) for r in rows),
        "all_gold_replay_identical_twice": True, "planned_mutations": 12 * 8, "completed_mutations": len(mutants),
        "reward_basis": dict(Counter("+".join(t["evaluation_criteria"]["reward_basis"]) for t in tasks)),
        "nonempty_nl_assertions": sum(bool(t["evaluation_criteria"].get("nl_assertions")) for t in tasks),
        "mutations": {}, "source_hashes": HASHES, "extracted_definitions": extracted,
        "runtime": {"python": sys.version.split()[0], "pydantic": pydantic.__version__},
        "initial_state_hash": initial_hash,
    }
    for variant in variants:
        rr = [r for r in mutants if r["variant"] == variant]
        summary["mutations"][variant] = {"n": len(rr), "db_matches": sum(r["db_matches_reference"] for r in rr),
                                            "all_calls_succeeded": sum(r["all_calls_succeeded"] for r in rr)}
    assert len(mutants) == 96
    assert all(r["db_matches_reference"] for r in mutants if r["variant"] == "extra_read")
    assert not any(r["db_matches_reference"] for r in mutants if r["variant"] == "extra_unrequested_write")
    for filename, value in (("summary.json", summary), ("gold.json", rows), ("mutations.json", mutants)):
        (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_hashes", "extracted_definitions")}, indent=2))


if __name__ == "__main__":
    main()
