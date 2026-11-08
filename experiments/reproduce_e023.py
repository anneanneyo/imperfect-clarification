"""Run frozen E023 verifier with only its historical source path relocated."""
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fresh_screen_e023'
manifest = json.loads((OUT/'manifest.json').read_text())
verifier = ROOT/'experiments/verify_fresh_e023.py'
assert hashlib.sha256(verifier.read_bytes()).hexdigest() == '26a2ab6c54639afce2fa9ec43e7796f6c9edfaf4b10e57a65aee40dc6313bc5e'
source = Path(manifest['source_path'])
if not source.exists():
    source = Path.home()/'.cache/clarification-research/e023-portable/banking_support.jsonl'
    if not source.exists():
        url = 'https://huggingface.co/datasets/moganai/lavoir-dialogues/resolve/d2c66dc86c782fadccc14748d8fa073ca16a6090/data/train/banking_support.jsonl'
        raw = urlopen(url, timeout=60).read()
        assert hashlib.sha256(raw).hexdigest() == manifest['source_sha256']
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(raw)
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']

class SourcePathOnly(ast.NodeTransformer):
    count = 0
    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == 'Path' and len(node.args) == 1
            and isinstance(node.args[0], ast.Subscript)
            and ast.dump(node.args[0]) == ast.dump(ast.parse("m['source_path']", mode='eval').body)):
            self.count += 1
            node.args = [ast.Name(id='_relocated_source', ctx=ast.Load())]
        return node

adapter = SourcePathOnly()
tree = adapter.visit(ast.parse(verifier.read_text()))
assert adapter.count == 2
ast.fix_missing_locations(tree)
exec(compile(tree, str(verifier), 'exec'), {'__name__': '__main__', '__file__': str(verifier), '_relocated_source': str(source)})
