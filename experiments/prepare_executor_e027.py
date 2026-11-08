"""Freeze public-only model inputs. Never evaluate an executor outcome here."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from executors_e027 import payload_for, messages_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/userbench_executor_e027"
MODEL = Path.home() / ".cache/clarification-research/models/qwen3-4b-instruct-2507-4bit"
MASKS = {"no_information": [], "full": [0, 1], "first_only": [0], "second_only": [1]}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def save(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2); f.write("\n")


def main():
    assert not (OUT / "input_manifest.json").exists()
    read = lambda path: json.loads(path.read_text())
    assert sha(ROOT / "research/E027_PROTOCOL.md") == read(OUT / "preinspection.json")["protocol_sha256"]
    parent = ROOT / "results/userbench_train_e025"
    manifest = read(parent / "manifest.json")
    for name in ["public_cases.json", "private_cases.json"]:
        assert sha(parent / name) == manifest["artifact_sha256"][name]
    previous = read(ROOT / "results/evidence_facts_e020/manifest.json")
    for name, expected in previous["model_files"].items():
        assert sha(MODEL / name) == expected, name
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=False, local_files_only=True)
    public, private = read(parent / "public_cases.json"), read(parent / "private_cases.json")
    jobs, schedule = {}, []
    rendered = {}
    for case, hidden in zip(public, private):
        assert case["case_id"] == hidden["case_id"]
        for aspect in case["aspects"]:
            statements = [p[3] for p in hidden["aspects"][aspect]["preferences"]]
            assert len(statements) == 2
            for mode, indices in MASKS.items():
                payload = payload_for(case, aspect, [statements[i] for i in indices])
                messages = messages_for(payload)
                job_id = digest(messages)
                jobs[job_id] = {"id": job_id, "payload": payload, "messages": messages}
                schedule.append({"case_id": case["case_id"], "aspect": aspect, "mask": mode, "id": job_id})
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                assert len(ids) <= 8192, len(ids)
                rendered[job_id] = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                    "input_token_ids_sha256": digest(ids), "input_token_count": len(ids)}
    # Fixed hash order, unrelated to source labels or observed successes.
    jobs = [jobs[k] for k in sorted(jobs)]
    assert len(schedule) == 192
    with (OUT / "public_inputs.jsonl").open("x") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    save(OUT / "schedule.json", schedule)
    save(OUT / "rendering_hashes.json", rendered)
    (OUT / "requirements.lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    paths = [ROOT / "research/E027_PROTOCOL.md", OUT / "preinspection.json", parent / "public_cases.json",
             parent / "private_cases.json"] + [OUT / n for n in ["public_inputs.jsonl", "schedule.json", "rendering_hashes.json", "requirements.lock.txt"]]
    code = ["experiments/executors_e027.py", "experiments/prepare_executor_e027.py", "experiments/run_executor_e027.py", "experiments/analyze_executor_e027.py"]
    save(OUT / "input_manifest.json", {"input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
         "code_sha256": {p: sha(ROOT / p) for p in code}, "model": previous["model"], "revision": previous["revision"],
         "model_files": previous["model_files"], "runtime": {k: metadata.version(k) for k in ["mlx", "mlx-lm", "transformers"]},
         "python": sys.version, "platform": platform.platform(), "seed": 20261108, "temperature": 0,
         "max_tokens": 64, "budget_seconds": 3600, "unique_inputs": len(jobs),
         "max_input_tokens": max(r["input_token_count"] for r in rendered.values())})
    print(json.dumps({"inputs_frozen": len(jobs), "max_input_tokens": max(r["input_token_count"] for r in rendered.values())}))


if __name__ == "__main__":
    main()
