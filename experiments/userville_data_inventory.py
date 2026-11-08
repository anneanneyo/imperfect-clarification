"""Inspect public UserVille task inputs without running its code or any models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pyarrow
import pyarrow.parquet as pq

EXPECTED_SHA = '8a71c2278151bf508593d4629b28bc864b3822087bdb1f85548b46685ba13d11'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    assert source_sha == EXPECTED_SHA, 'Public Drive snapshot changed'
    table = pq.read_table(args.data)
    rows = table.to_pylist()
    roles = Counter(tuple(m['role'] for m in r['prompt']) for r in rows)
    extra_roles = Counter(tuple(m['role'] for m in r['extra_info']['prompt']) for r in rows)
    answers = Counter(r['answer'] for r in rows)
    # Retain counts/schema only; no source issue text, patch, or person data.
    result = {
        'repository': 'https://github.com/sunnweiwei/PPP-Agent',
        'commit': 'abbb7b1594f90eb94cf9543878f88996b7cd21ac',
        'source_url': 'https://drive.google.com/uc?export=download&id=1w4Zjne7mwu6iWtDH2KQI0UkTlNPz5WMW',
        'source_file': 'test_ood.parquet', 'source_sha256': source_sha,
        'source_bytes': args.data.stat().st_size, 'pyarrow_version': pyarrow.__version__,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rows': len(rows), 'columns': table.column_names,
        'schema': str(table.schema.remove_metadata()),
        'prompt_role_patterns': [{'roles': k, 'n': v} for k,v in roles.items()],
        'extra_prompt_role_patterns': [{'roles': k, 'n': v} for k,v in extra_roles.items()],
        'answer_unique_values': len(answers),
        'answer_placeholder_count': answers['There is no answer'],
        'null_reward_ground_truth': sum(r['reward_model']['ground_truth'] is None for r in rows),
        'is_vague_counts': dict(Counter(str(r['extra_info']['is_vague']) for r in rows)),
        'preference_counts': dict(Counter(r['extra_info']['preference']['preference_name'] for r in rows)),
        'unique_config_indices': len({r['extra_info']['index'] for r in rows}),
        'full_trajectories_observed': 0 if set(roles) == {('system','user')} and set(extra_roles) == {('system','user')} else None,
        'interpretation': 'Checked OOD input file contains initial prompts and preferences, with placeholder answers; not observed user-response/outcome trajectories. Other two Drive files not downloaded or audited.',
        'source_code_executed': False, 'model_calls': 0, 'raw_data_exported': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['rows','answer_placeholder_count','null_reward_ground_truth','full_trajectories_observed']},indent=2))


if __name__ == '__main__':
    main()
