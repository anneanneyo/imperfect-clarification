"""Standalone development figure for E012; bands are SD, not confidence intervals."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/retail_small_feedback'
s=json.loads((OUT/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,ax=plt.subplots(figsize=(8.5,5.3),layout='constrained')
budgets=[0,16,64,256,1024]
for method,label,color in [('learned_contextual','Выбор вопроса для текущих заказов','#2166ac'),
                            ('learned_fixed','Один выученный вопрос для всех','#b35806')]:
    rows=[next(r for r in s['aggregate'] if r['budget']==b and r['method']==method) for b in budgets]
    means=[100*r['accuracy']['mean'] for r in rows]
    sd=[100*r['accuracy']['sd'] for r in rows]
    ax.plot(range(len(budgets)),means,'o-',color=color,label=label,lw=2)
    ax.fill_between(range(len(budgets)),[m-d for m,d in zip(means,sd)],[m+d for m,d in zip(means,sd)],color=color,alpha=.12)
ax.axhline(100*s['controls']['oracle']['accuracy'],color='#343a40',ls='--',lw=1.4,label='Oracle: механизм ответа известен')
ax.axhline(100*s['controls']['always_contents']['accuracy'],color='#777777',ls=':',lw=1.3,label='Всегда спрашивать состав')
ax.set_xticks(range(len(budgets)),labels=budgets)
ax.set_xlabel('Число наблюдённых ответов (отдельные бюджеты)')
ax.set_ylabel('Ожидаемая вероятность верного выбора, %')
ax.set_ylim(83,95.5);ax.grid(axis='y',alpha=.2)
ax.spines[['top','right']].set_visible(False)
ax.set_title('E012 · Простой обучаемый метод близок к oracle',loc='left',fontweight='bold',pad=15)
ax.legend(loc='lower right',frameon=False,fontsize=9)
fig.supxlabel('20 development-семей · искусственный канал ответов\nПолосы: ±1 SD по 30 журналам обучения, не доверительные интервалы',fontsize=9,color='#555555')
fig.savefig(OUT/'learning_curve.png',dpi=180)
fig.savefig(OUT/'learning_curve.svg')
plt.close(fig)
print(OUT/'learning_curve.png')
