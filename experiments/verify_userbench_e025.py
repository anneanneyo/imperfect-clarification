"""Independent source-to-artifact checks for E025; does not call its generator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/clarification-research/userbench-80506d2")
    parser.add_argument("--output", type=Path, help="Write a new verification record; never overwrite")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "results/userbench_train_e025"
    read = lambda name: json.loads((folder / name).read_text())
    sha = lambda data: hashlib.sha256(data).hexdigest()
    manifest = read("manifest.json")
    assert sha((root / "research/E025_PROTOCOL.md").read_bytes()) == manifest["protocol_sha256"]
    assert read("preinspection.json")["protocol_sha256"] == manifest["protocol_sha256"]
    for path, expected in manifest["code_sha256"].items():
        assert sha((root / path).read_bytes()) == expected
    for name, expected in manifest["artifact_sha256"].items():
        assert sha((folder / name).read_bytes()) == expected
    for name, expected in manifest["source_files"].items():
        data = (args.cache / name).read_bytes()
        assert sha(data) == expected["sha256"]
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == expected["git_blob"]
    import pyarrow.parquet as pq
    rows = pq.read_table(args.cache / "data/travel2222_multiturn/train.parquet").to_pylist()
    ids = [str(row["reward_model"]["id"]) for row in rows]
    assert len(ids) == len(set(ids)) == 96
    selected = sorted(ids, key=lambda i: (sha(("E025|" + i).encode()), i))[:12]
    assert selected == read("selection.json")["selected_ids"]
    all_data = json.loads((args.cache / "travelgym/data/travelgym_data_2222.json").read_text())
    data = [all_data[i] for i in selected]
    del all_data
    public, private = read("public_cases.json"), read("private_cases.json")
    tiers = {"Cheap": 0, "Normal": 1, "Expensive": 2}
    n_options = n_prefs = n_aspects = dangling = ties = 0
    for source, visible, hidden in zip(data, public, private):
        assert set(visible) == {"case_id", "user_message", "aspects", "options"}
        assert visible["case_id"] == hidden["case_id"]
        assert set(visible["aspects"]) == set(source["dimensions"])
        for aspect in source["dimensions"]:
            src, gold = source[aspect], hidden["aspects"][aspect]
            originals = {o["id"]: o for group in src["options"].values() for o in group}
            aliases = gold["source_to_visible"]
            assert len(aliases) == len(set(aliases.values())) == len(originals)
            displayed = {o["id"]: o for o in visible["options"][aspect]}
            assert set(displayed) == set(aliases.values()) == set(gold["valid_ids"])
            for original_id, alias in aliases.items():
                expected = {k: v for k, v in originals[original_id].items() if k not in {"type", "reason", "id"}}
                observed = {k: v for k, v in displayed[alias].items() if k != "id"}
                assert observed == expected
            assert gold["best_id"] == aliases[src["best_id"]]
            assert set(gold["acceptable_ids"]) == {aliases[i] for i in src["correct_ids"]}
            assert gold["preferences"] == src["preferences"]
            assert set(src["noise_ids"]) - set(originals) == set(src["all_ids"]) - set(originals)
            dangling += len(set(src["all_ids"]) - set(originals))
            n_options += len(originals)
            n_prefs += len(src["preferences"])
            n_aspects += 1
        restaurants = source["restaurant"]["options"]["correct"]
        counts = Counter(tiers[o["expectation"]] for o in restaurants)
        ties += counts[min(counts)] > 1
        assert all("cost" not in o for o in restaurants)
    # Verify the two concrete source-semantic counterexamples, without changing labels.
    mismatch_cases = []
    for index, phrase in [(2, "ribeye"), (5, "steak")]:
        src = data[index]["restaurant"]
        assert phrase in src["preferences"][0][3].lower()
        best = next(o for o in src["options"]["correct"] if o["id"] == src["best_id"])
        assert best["cuisine"] == "Seafood"
        assert any(o["cuisine"] == "Steakhouse" and o["expectation"] == best["expectation"] for o in src["options"]["correct"])
        mismatch_cases.append({"case_id": public[index]["case_id"], "best_id": src["best_id"]})
    # Independent union-find instead of generator's graph traversal.
    parents = list(range(len(ids)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    owners = {}
    for index, task_id in enumerate(ids):
        for part in task_id.split("|"):
            if part in owners:
                parents[find(index)] = find(owners[part])
            else:
                owners[part] = index
    graph_sizes = sorted(Counter(find(i) for i in range(len(ids))).values(), reverse=True)
    summary = read("summary.json")
    assert (n_aspects, n_prefs, n_options, dangling, ties) == (48, 96, 864, 192, 6)
    assert summary["restaurant_minimum_tier_ties"] == ties
    assert summary["train_task_connected_component_sizes"] == graph_sizes == [96]
    assert summary["train_unique_component_ids"] == len(owners) == 241
    result = {"status": "passed", "protocol_and_artifact_hashes": "verified",
              "selected_train_ids": "independently_recomputed",
              "public_private_boundary": "all_864_options_checked",
              "source_label_mapping": "all_48_aspects_checked",
              "dangling_noise_metadata_ids": dangling, "restaurant_minimum_tier_ties": ties,
              "source_semantic_counterexamples": mismatch_cases,
              "train_template_graph_components": graph_sizes,
              "model_calls": 0, "official_test_tasks_evaluated": 0,
              "verifier_sha256": sha(Path(__file__).read_bytes()),
              "manifest_sha256": sha((folder / "manifest.json").read_bytes())}
    if args.output:
        with args.output.open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
