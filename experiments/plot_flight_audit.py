"""Plot the offline audit; requires matplotlib, does not call models."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parents[1] / 'results' / 'flight_audit'
data = json.loads((base / 'audit.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True, sharey=True)
for ax, contrast, title in zip(axes,
        ['VOI minus NoQuestion:0', 'VOI minus VOI_own_turn0'],
        ['Published baseline (different prompt)', 'VOI trajectory before any questions']):
    rows = [r for r in data['rows'] if r.get('contrast') == contrast]
    y = [r['mean'] for r in rows]
    errors = [[r['mean']-r['ci95'][0] for r in rows], [r['ci95'][1]-r['mean'] for r in rows]]
    ax.errorbar(range(3), y, yerr=errors, fmt='o', color='#226b9c', capsize=6)
    ax.axhline(0, color='#777777', linestyle='--', linewidth=1)
    ax.set_xticks(range(3), ['0.01', '0.05', '0.10'])
    ax.set_xlabel('Cost per question')
    ax.set_ylabel('VOI minus no-question utility')
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=.2)
    ax.set_ylim(-.08, .34)
fig.suptitle('Author-log reanalysis: 100 episodes, 64 source records\n95% cluster-bootstrap intervals; no new model runs', fontsize=12)
fig.savefig(base / 'utility_intervals.png', dpi=180)
fig.savefig(base / 'utility_intervals.svg')
plt.close(fig)
