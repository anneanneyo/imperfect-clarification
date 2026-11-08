"""Restore the two large historical CSVs without replacing differing files."""
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for entry in json.loads((root/'results/large_results_manifest.json').read_text()):
    path = root/entry['path']
    raw = gzip.decompress((root/entry['archive']).read_bytes())
    assert len(raw) == entry['bytes']
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']
    if path.exists():
        assert path.read_bytes() == raw, f'Existing file differs: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    print('Verified:', entry['path'])
