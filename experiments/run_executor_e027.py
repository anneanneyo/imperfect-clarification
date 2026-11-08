"""One local inference attempt per frozen public input; never load hidden gold."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/userbench_executor_e027"
MODEL = Path.home() / ".cache/clarification-research/models/qwen3-4b-instruct-2507-4bit"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def main():
    assert not (OUT / "started.jsonl").exists(), "No implicit repeats of started attempts"
    assert not (OUT / "generations.jsonl").exists()
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    # Deliberately validate only public runner inputs; never read hidden paths
    # listed in the preparer's broader provenance manifest.
    for name in ["public_inputs.jsonl", "rendering_hashes.json"]:
        path = OUT / name
        assert sha(path) == manifest["input_sha256"][str(path.relative_to(ROOT))]
    assert sha(Path(__file__)) == manifest["code_sha256"]["experiments/run_executor_e027.py"]
    for name, expected in manifest["model_files"].items():
        assert sha(MODEL / name) == expected
    jobs = [json.loads(line) for line in (OUT / "public_inputs.jsonl").read_text().splitlines()]
    rendering = json.loads((OUT / "rendering_hashes.json").read_text())
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    start = time.monotonic()
    mx.random.seed(manifest["seed"])
    model, tokenizer = load(str(MODEL), trust_remote_code=False)
    completed = 0
    with (OUT / "started.jsonl").open("x") as started, (OUT / "generations.jsonl").open("x") as output:
        for index, job in enumerate(jobs):
            if time.monotonic() - start >= manifest["budget_seconds"]:
                break
            prompt = tokenizer.apply_chat_template(job["messages"], tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            assert hashlib.sha256(prompt.encode()).hexdigest() == rendering[job["id"]]["prompt_sha256"]
            assert digest(ids) == rendering[job["id"]]["input_token_ids_sha256"]
            started.write(json.dumps({"id": job["id"], "index": index, "elapsed_seconds": time.monotonic() - start}) + "\n")
            started.flush()
            before = time.monotonic()
            chunks, last, error = [], None, None
            try:
                for part in stream_generate(model, tokenizer, prompt=ids, max_tokens=manifest["max_tokens"], sampler=make_sampler(temp=0.0)):
                    chunks.append(part.text); last = part
                    if time.monotonic() - start >= manifest["budget_seconds"] and part.finish_reason is None:
                        raise TimeoutError("Frozen overall generation budget exhausted")
                if last is None or last.finish_reason not in ["stop", "length"]:
                    raise RuntimeError("Missing final generation status")
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            record = {"id": job["id"], "raw_output": "".join(chunks), "error": error,
                      "elapsed_seconds": time.monotonic() - before,
                      "finish_reason": last.finish_reason if last else None,
                      "prompt_tokens": last.prompt_tokens if last else len(ids),
                      "generated_tokens": last.generation_tokens if last else 0,
                      "peak_memory_gb": last.peak_memory if last else None}
            output.write(json.dumps(record, ensure_ascii=False) + "\n"); output.flush()
            completed += 1
            if completed % 8 == 0 or error:
                print(json.dumps({"completed": completed, "total": len(jobs), "elapsed_seconds": round(time.monotonic() - start, 1), "error": error}), flush=True)
            mx.clear_cache()
            if error and error["type"] == "TimeoutError":
                break
    result = {"completed": completed, "total": len(jobs), "complete": completed == len(jobs),
              "elapsed_seconds": time.monotonic() - start}
    with (OUT / "run_status.json").open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
