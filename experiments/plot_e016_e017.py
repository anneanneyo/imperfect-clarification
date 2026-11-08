"""Publication-style descriptive figure from independently verified summaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'results/heldout_value_e016';B=ROOT/'results/encoder_value_e017'
def read(p):return json.loads(p.read_text())
if __name__=='__main__':
    assert read(A/'verification.json')['verified'] and read(B/'verification.json')['verified']
    a=read(A/'summary.json');b=read(B/'summary.json');m=a['methods'];control=read(ROOT/'results/e016_e017_stopping_control/summary.json')['methods']['lexical']
    labels=['Без уточнения','Всегда «Какой продукт?»','Всегда «Что случилось?»','Один вопрос + те же остановки†','TF–IDF + ridge','Языковые признаки + ridge','Post-hoc ориентир*']
    vals=[m['no_ask']['accuracy'],m['train_constant']['accuracy'],m['issue']['accuracy'],control['metrics']['matched_stop_constant']['accuracy'],m['ridge']['accuracy'],b['accuracy'],m['component_posthoc_oracle']['accuracy']]
    colors=['#84929c']*4+['#2471a3','#117864','#b99c6b']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.2,5.7),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
    y=np.arange(len(vals));ax.barh(y,np.array(vals)*100,color=colors,height=.56);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,106)
    for yy,v in zip(y,vals):ax.text(v*100+1,yy,f'{v*100:.1f}%',va='center',fontsize=10)
    ax.set_xlabel('Доля правильных решений, %');ax.set_title('48 отложенных ситуаций',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    comparisons=[('TF–IDF − постоянный',a['paired_utility_comparisons']['train_constant']),('TF–IDF − те же остановки†',control['paired']['matched_stop_constant']),
        ('Языковые − TF–IDF',b['paired_comparisons']['ridge']),('Языковые − постоянный',b['paired_comparisons']['train_constant'])]
    for i,(label,p) in enumerate(comparisons):
        d=p.get('difference',p.get('utility_difference'));lo,hi=p['percentile95_interval'];color='#2471a3' if i<2 else '#117864'
        ax2.errorbar(d*100,i,xerr=np.array([[d-lo],[hi-d]])*100,fmt='o',color=color,capsize=4,markersize=5)
    ax2.set_yticks(range(len(comparisons)),[label for label,_ in comparisons]);ax2.invert_yaxis();ax2.axvline(0,color='#555',linestyle='--',linewidth=1)
    ax2.set_title('Разница с учётом цены вопроса',loc='left',fontweight='bold');ax2.set_xlabel('Разница utility × 100; 95% bootstrap интервал');ax2.grid(axis='x',alpha=.15)
    fig.suptitle('E016–E017 · Выбор уточнения по исходному тексту',fontsize=15,fontweight='bold')
    fig.supxlabel('Один синтетический workflow; цена вопроса 0.05. Интервалы по ситуациям, не по новым ответам.\n†Диагностический контроль: тот же выбор «не спрашивать», иначе вопрос о продукте.\n*Ориентир использует тестовые исходы и недоступен при реальном выборе.',fontsize=9)
    out=ROOT/'results/e016_e017_diagnostics';out.mkdir(exist_ok=True)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
    print(out/'comparison.png')
