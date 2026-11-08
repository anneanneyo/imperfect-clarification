"""Plot E001a descriptive gaps; matplotlib is optional for the evaluator."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
fig, axes=plt.subplots(1,2,figsize=(10,4.6),constrained_layout=True)
for ax, folder, title in zip(axes,['channel_shift_pilot','channel_shift_pilot_scalar'],
                            ['5 questions (airlines bundled)','8 questions (one coefficient each)']):
    data=json.loads((root/'results'/folder/'summary.json').read_text())
    for method, label, color in [('evsi_source','No adaptation','#b35b52'),
                                  ('evsi_pooled','Source + new counts','#a48420'),
                                  ('evsi_adapt','New counts only','#246d9b')]:
        rows=sorted([r for r in data['summary'] if r['regime']=='selective' and r['cost']==0 and r['method']==method],key=lambda r:r['budget'])
        ax.plot([r['budget'] for r in rows],[r['gap'] for r in rows],marker='o',label=label,color=color)
        ax.fill_between([r['budget'] for r in rows],[r['gap_seed_minmax'][0] for r in rows],
                        [r['gap_seed_minmax'][1] for r in rows],color=color,alpha=.1)
    ax.set_title(title,fontsize=11)
    ax.set_xlabel('New answer/no-answer observations')
    ax.set_ylabel('Excess expected loss above known-rate reference')
    ax.set_xticks([0,50,200]);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Synthetic channel, exact executor, 89 external catalogs\nMean over 30 calibration seeds; shading = seed range (not CI)',fontsize=11)
fig.savefig(root/'results/channel_shift_pilot/adaptation.png',dpi=180)
fig.savefig(root/'results/channel_shift_pilot/adaptation.svg')
plt.close(fig)
