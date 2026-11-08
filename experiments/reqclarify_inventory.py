"""Read-only inventory of the published ReqClarifyBench Zenodo archive."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/reqclarify_inventory.json"))
    args = parser.parse_args()
    groups = {}
    for d in sorted((args.root / "data/reconstruction").iterdir()):
        if not d.is_dir():
            continue
        files = {name: sorted(d.rglob(name)) for name in ("atomic.json", "intent.md", "requirements.md")}
        parents = [{p.parent.relative_to(d).as_posix() for p in ff} for ff in files.values()]
        groups[d.name] = {name: len(ff) for name, ff in files.items()}
        groups[d.name]["matching_task_directories"] = parents[0] == parents[1] == parents[2]
        groups[d.name]["all_atomic_valid_json"] = all(isinstance(json.loads(p.read_text()), list) for p in files["atomic.json"])
    manifest = {p.relative_to(args.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.root.rglob("*")) if p.is_file()}
    tree_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert tree_hash == "4661e96e431664a4b82c410751ce15f4710b13472579acdc8206f0c9e583d59b", "Archive tree changed"
    report = {
        "source": "https://zenodo.org/records/23031774", "access_date": "2026-10-27",
        "archive_url": "https://zenodo.org/api/records/23031774/files/ReqClarifyBench-main.zip/content",
        "archive_md5_verified_at_download": "421b963803d9d9913d02468af9f0cf57",
        "archive_sha256_at_download": "c105f6dab1a6b79631169b0d94085532d41a7a1e09179314aa2fb5c291e670a3",
        "archive_bytes": 3962434, "file_count": len(manifest), "tree_manifest_sha256": tree_hash, "groups": groups,
        "task_count": sum(g["intent.md"] for g in groups.values()),
        "license_file": "LICENSE", "license_label": "MIT (archive and Zenodo metadata)",
        "evaluation_files": [k for k in manifest if k.startswith("evaluation/")],
        "result_files": [k for k in manifest if k.startswith("results/")],
        "executed_upstream_code": False, "model_calls": 0, "file_sha256": manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
