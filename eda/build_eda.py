# -*- coding: utf-8 -*-
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# EDA: уточняющие вопросы при неполных ответах

Анна Губарева, 8 ноября 2026.

Исследуем, какой вопрос помогает выбрать подходящий вариант, если пользователь отвечает не полностью. Здесь проверяем данные для эксперимента: пропуски, разметку, распределения и первые результаты исполнителей.

В наборе 12 задач UserBench, 48 отдельных выборов и 864 варианта. Задачи синтетические и уже использовались при разработке. Официальный test не оценивался.

Источник: SalesforceAIResearch/UserBench, версия `80506d2ab484cab843e60a2401ff3e0290d05b87`, лицензия Apache-2.0. Из 96 train-задач выбрали 12 по заранее заданному хешу. Подробнее — в [описании источников](THIRD_PARTY.md).''')
code('''from pathlib import Path
import json, hashlib, re, sys, copy
from fractions import Fraction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
ROOT=Path.cwd()
if not (ROOT/'results').exists(): ROOT=ROOT.parent
load=lambda p:json.loads((ROOT/p).read_text())
pub=load('results/userbench_train_e025/public_cases.json')
gold=load('results/userbench_train_e025/private_cases.json')
truth=load('results/userbench_semantics_e026/reviewed_option_evaluations.json')
criteria=load('results/userbench_semantics_e026/criteria.json')
manifest=load('results/userbench_train_e025/manifest.json')
for name,sha in manifest['artifact_sha256'].items():
 assert hashlib.sha256((ROOT/'results/userbench_train_e025'/name).read_bytes()).hexdigest()==sha
records=[]
for case in pub:
 for aspect,options in case['options'].items():
  for o in options:
   row=next(r for r in truth if (r['case_id'],r['aspect'],r['option_id'])==(case['case_id'],aspect,o['id']))
   costs=o.get('cost'); price=min(costs) if isinstance(costs,list) and costs else costs
   records.append(dict(case_id=case['case_id'],aspect=aspect,option_id=o['id'],source_acceptable=int(row['source_acceptable']),source_best=int(row['source_best']),semantic=row['conjunction'],base_cost=price,rating=o.get('rating'),text_chars=len(json.dumps(o,ensure_ascii=False)),attribute_count=len(o),raw=o))
df=pd.DataFrame(records)
assert len(pub)==12 and len(df)==864
print({'tasks':len(pub),'aspect_decisions':df.groupby(['case_id','aspect']).ngroups,'options':len(df),'preferences':len(criteria)})
display(df.drop(columns='raw').describe(include='all'))''')
md('''## 1. Состав данных и метки

Одна строка — один вариант перелёта, жилья, машины или ресторана. Внутри задачи строки связаны между собой.

В исходной разметке есть две метки: `source_acceptable` — допустимый вариант, `source_best` — единственный лучший. Дополнительно проверяем два предпочтения пользователя по собственным правилам: T — выполнены оба, F — нарушено хотя бы одно, U — сведений недостаточно. Эти правила пока не проверены независимым разметчиком. U относится к проверке варианта, а не к ответу пользователя «не знаю».

Посмотрим на баланс меток и повторы шаблонов.''')
code('''display(df.groupby('aspect').agg(options=('option_id','size'),acceptable=('source_acceptable','sum'),best=('source_best','sum')))
fig,ax=plt.subplots(1,2,figsize=(10,3.5))
pd.crosstab(df.aspect,df.source_acceptable).plot.bar(stacked=True,ax=ax[0],color=['#bec9d2','#16846c']);ax[0].set_title('Source labels by aspect');ax[0].set_ylabel('Options')
df.semantic.value_counts().reindex(['T','F','U']).plot.bar(ax=ax[1],color=['#16846c','#c45b4f','#b3a260']);ax[1].set_title('Provisional criterion conjunction');ax[1].set_ylabel('Options')
plt.tight_layout();plt.show()
s=load('results/userbench_train_e025/summary.json')
print('Train template IDs:',s['train_unique_component_ids'],'unique /',s['train_component_id_count'],'occurrences')
print('Connected components of train tasks:',s['train_task_connected_component_sizes'])''')
md('''В каждом выборе 3 допустимых варианта из 18 и один лучший. Всего 144 положительные метки acceptable и 48 best. Баланс задан авторами набора.

Все 96 train-задач связаны через повторяющиеся шаблоны. Случайное разбиение строк на обучение и тест даст пересечения. Для проверки на новых шаблонах придётся исключить часть задач или собрать другой набор. Два случая смены лучшего вопроса в предыдущем эксперименте тоже имеют общий шаблон перелёта.''')
md('''## 2. Пропуски

У разных частей поездки разные поля. Например, у ресторана есть ценовая категория, но нет числовой цены. Заполнять её средней ценой нельзя.

Для услуг различаем три случая: отсутствующий ключ — нет сведений; явный null — услуга недоступна по соглашению источника; ноль — услуга бесплатна. Ниже считаем их отдельно.''')
code('''display(df.groupby('aspect')[['base_cost','rating']].agg(lambda x: x.isna().mean()).rename(columns=lambda c:c+'_missing_share'))
miss=[]
for aspect,group in df.groupby('aspect'):
 flat=pd.json_normalize(group.raw.tolist())
 table=pd.DataFrame({'missing_count':flat.isna().sum(),'missing_share':flat.isna().mean()})
 print(aspect, 'within-schema missing-cell fraction:',round(float(flat.isna().to_numpy().mean()),3))
 display(table[table.missing_count>0])
 for parent in ['service','insurance']:
  keys=sorted({k for o in group.raw for k in o.get(parent,{})})
  for key in keys:
   vals=[o.get(parent,{}) for o in group.raw]
   miss.append(dict(aspect=aspect,field=parent+'.'+key,absent=sum(key not in v for v in vals),explicit_null=sum(key in v and v[key] is None for v in vals),numeric_zero=sum(key in v and v[key] is not None and v[key]==0 for v in vals),n=len(vals)))
missing=pd.DataFrame(miss)
for c in ['absent','explicit_null','numeric_zero']:missing[c+'_share']=missing[c]/missing.n
display(missing)''')
md('''Неприменимые числовые поля оставляем NaN и сравниваем варианты внутри одной части поездки. Смысл null берём из схемы и инструкций генерации UserBench. Если заменить неизвестную стоимость нулём, неизвестная услуга станет бесплатной.''')
md('''## 3. Ошибки и выбросы

Проверим отрицательные цены, рейтинги вне шкалы, длины текстов и повторяющиеся ID. Значения за границами IQR отмечаем для просмотра. В меню намеренно добавлены шумовые варианты, поэтому удалять все выбросы нельзя.''')
code('''print('Negative numeric prices:',int((df.base_cost.dropna()<0).sum()))
print('Ratings outside 0..10:',int(((df.rating<0)|(df.rating>10)).sum()))
flags=[]
for aspect,g in df.groupby('aspect'):
 for feature in ['base_cost','rating','text_chars']:
  x=g[feature].dropna()
  if len(x):
   q1,q3=x.quantile([.25,.75]);iqr=q3-q1
   flags.append({'aspect':aspect,'feature':feature,'n':len(x),'iqr_flags':int(((x<q1-1.5*iqr)|(x>q3+1.5*iqr)).sum()),'min':x.min(),'max':x.max()})
display(pd.DataFrame(flags))
print('Repeated IDs inside a menu:',int(df.duplicated(['case_id','aspect','option_id']).sum()))
display(pd.DataFrame(criteria).groupby('status').size().rename('criteria'))
print('Source-correct vs provisional interpretation:')
display(pd.crosstab(df.source_acceptable,df.semantic))
pref=pd.DataFrame(criteria)
pref['statement_chars']=pref.implicit_statement.str.len()
display(pref.groupby('aspect').statement_chars.describe())
fig,ax=plt.subplots(1,2,figsize=(10,3.5));sns.histplot(data=df,x='text_chars',hue='aspect',element='step',ax=ax[0]);sns.boxplot(data=pref,x='aspect',y='statement_chars',ax=ax[1]);plt.tight_layout();plt.show()''')
md('''Найдены 2 отрицательные цены и 14 рейтингов вне диапазона 0–10. Для логарифма цены отрицательные значения заменяем на NaN; для корреляций исключаем рейтинги вне шкалы. Исходные записи сохраняем. IQR-выбросы без дополнительных оснований не удаляем.

Обнаружены содержательные проблемы эталонов: в двух ресторанных задачах текст конкретизирует стейк, а лучший вариант имеет категорию Seafood; в одной задаче с указанным Hong Kong лучший рейс проходит через Los Angeles. У шести ресторанных задач минимум ценовой категории делят несколько вариантов, но source-best один. В all_ids сохранены 192 ссылки на отсутствующие шумовые варианты. Все best/correct ID при этом доступны.

Наш проверяющий код тоже не безошибочен: буквальное Zillow Rental не узнавало Zillow Rentals. Поправка сохранена отдельно, исходный прогон не переписан. После неё 6 из 144 source-acceptable вариантов противоречат хотя бы одному нашему критерию, 33 остаются U. Это согласованность с предварительными правилами, не доказанная частота ошибок всего корпуса. Реальное меню ресторанов, удобство машин и частота уборки этими полями полностью не проверяются.''')
md('''## 4. Связи между признаками

Считаем корреляции отдельно для перелётов, жилья и машин: цены у них имеют разный смысл. Для ресторанов числовой цены нет.

Варианты созданы под заданные предпочтения, а строки внутри задачи зависимы. Графики могут отражать способ генерации данных; по ним нельзя установить причинную связь.''')
code('''display(df.groupby(['aspect','source_acceptable'])[['base_cost','rating','text_chars','attribute_count']].median())
fig,axs=plt.subplots(1,3,figsize=(13,3.5))
for ax,aspect in zip(axs,['flight','hotel','rental_car']):
 g=df[df.aspect==aspect].copy();g['rating']=g.rating.where(g.rating.between(0,10));g['log_price']=np.log1p(g.base_cost.where(g.base_cost>=0))
 sns.heatmap(g[[c for c in ['log_price','rating','text_chars','attribute_count','source_acceptable'] if g[c].nunique()>1]].corr(method='spearman'),vmin=-1,vmax=1,cmap='vlag',ax=ax);ax.set_title(aspect)
plt.tight_layout();plt.show()
g=df[df.aspect=='rental_car'].copy();g['log_price']=np.log1p(g.base_cost.where(g.base_cost>=0))
sns.pairplot(g[['log_price','text_chars','source_acceptable']],hue='source_acceptable',diag_kind='hist',height=2);plt.show()''')
md('''Постоянные признаки исключены из корреляционных таблиц. Если корреляция не определена, это не означает отсутствие связи.

Цену и длину текста можно использовать как признаки. Метки качества, скрытые предпочтения и наши правила проверки доступны только при оценке. Для обучения выбора вопроса ещё потребуется задать возможные ответы пользователя и их вероятности.''')
md('''## 5. Первые модели

Сравниваем случайный выбор, эвристику по совпадению слов и цене и Qwen3-4B-Instruct-2507 в 4-битной версии. Модель использована без дообучения. Каждый из 192 входов запущен один раз; скрытые предпочтения и метки в запрос не включались.

Проверяем четыре режима: без дополнительных сведений, оба предпочтения, только первое, только второе. В каждом режиме сведения раскрываются для всех четырёх частей поездки. Это проверка исполнителя; выбор одного вопроса оценивается отдельно.''')
code('''summary=load('results/userbench_executor_e027/summary.json')
rows=[]
for policy,modes in summary['all_aspects_same_mask_diagnostic'].items():
 for mode,metrics in modes.items():rows.append({'executor':policy,'mask':mode,**{k:100*float(Fraction(v)) for k,v in metrics.items()}})
display(pd.DataFrame(rows).round(2))
# Independent direct check of model acceptable choices against E025 source labels.
jobs={r['id']:r for r in map(json.loads,(ROOT/'results/userbench_executor_e027/public_inputs.jsonl').read_text().splitlines())}
gens={r['id']:r for r in map(json.loads,(ROOT/'results/userbench_executor_e027/generations.jsonl').read_text().splitlines())}
schedule=load('results/userbench_executor_e027/schedule.json')
hidden={c['case_id']:c for c in gold}; counts={m:0 for m in ['full','no_information','first_only','second_only']}
for e in schedule:
 obj=json.loads(gens[e['id']]['raw_output']);assert set(obj)=={'id'}
 counts[e['mask']]+=obj['id'] in hidden[e['case_id']]['aspects'][e['aspect']]['acceptable_ids']
for mode,n in counts.items():assert Fraction(n,48)==Fraction(summary['all_aspects_same_mask_diagnostic']['qwen3_4b'][mode]['source_acceptable'])
print('Direct source-label recount (48 aspect choices per condition):',counts)
print('Strict semantic reversals for the language executor:',summary['ranking_diagnostics']['qwen3_4b']['robust_semantic_reversals'])''')
md('''Qwen выбрала допустимый по исходной разметке вариант в 10/48 случаев без дополнительных сведений и в 31/48 при раскрытии обоих предпочтений. У эвристики — 19,44% и 45,83%. Числа для Qwen выше пересчитаны напрямую из сохранённых ответов.

При сравнении отдельных вопросов у Qwen не найдено строгой смены единственного лучшего вопроса по нашим правилам проверки с учётом неизвестных значений. По исходным меткам есть один случай расхождения между полным и частичным ответом с учётом ничьих. Пока этого недостаточно, чтобы говорить о преимуществе метода выбора вопроса.''')
md('''## 6. Аугментации

Проверим, меняет ли эвристика выбор при перестановке вариантов и изменении пробелов. Для обеих операций смысл данных сохраняется. Qwen в этой проверке не запускаем. Это две проверки устойчивости к преобразованиям входа, а не эксперимент по улучшению качества после обучения с аугментациями.

Удаление части предпочтений меняет доступную информацию, поэтому считаем его отдельным условием эксперимента. Перефразирование пока не используем: сначала нужно проверить, сохраняет ли оно предпочтения. Перестановка меню пригодна, если варианты сохраняют свои ID: она позволяет проверять зависимость от позиции. Изменение пробелов подходит для проверки устойчивости к форматированию. Для эвристики обе операции ожидаемо сохраняют результат: она сопоставляет слова и возвращает распределение по ID. Перенос этого вывода на языковую модель не проверен; для Qwen потребуется отдельный запуск на преобразованных входах.''')
code('''sys.path.insert(0,str(ROOT/'experiments'))
from executors_e027 import lexical_distribution
checks={'option_order':0,'whitespace':0}
for job in jobs.values():
 p=job['payload'];reference=lexical_distribution(p)[0]
 perm=copy.deepcopy(p);perm['options'].reverse();checks['option_order']+=lexical_distribution(perm)[0]==reference
 spaced=copy.deepcopy(p);spaced['trip_request']='  '.join(spaced['trip_request'].split());spaced['additional_statements']=['  '.join(t.split()) for t in spaced['additional_statements']]
 checks['whitespace']+=lexical_distribution(spaced)[0]==reference
assert checks=={'option_order':192,'whitespace':192}
print(checks)''')
md('''## 7. Выводы

- В данных есть ошибки числовых полей и спорные эталоны. Одного совпадения с source-best недостаточно для оценки решения.
- Пропуски обозначают разные вещи: отсутствие сведений, недоступность услуги или неприменимость поля. Заполнять их одним значением нельзя.
- Дополнительные сведения улучшают выбор Qwen на этом наборе. Строгую смену лучшего вопроса при неполном ответе по нашим правилам пока не нашли.
- Перестановка вариантов и изменение пробелов не изменили распределение выбора эвристики на всех 192 входах.

Отдельный пересчёт 576 оценок выбора, 576 результатов вопросов и 36 ранжирований завершён без расхождений. Разобраны 14 нарушений предпочтений при полном ответе ([отчёт](research/E027_ERROR_ANALYSIS.md)). Эта проверка подтверждает арифметику, но использует готовые предварительные семантические правила. Ближайший шаг — глубже изучить тему: разобрать ключевые статьи, расчёт пользы вопроса и ограничения проведённых экспериментов. После обсуждения с руководителем выбрать следующий эксперимент. Независимая проверка правил и новые задачи с контролем повторяющихся шаблонов остаются необходимыми для дальнейшей оценки метода.

Ограничения: 12 уже просмотренных синтетических задач, предварительные правила проверки, отсутствие ответов реальных пользователей. 864 варианта не являются независимыми задачами. На UserBench оценщик пользы вопроса пока не обучен. Простые оценщики в ранних постановках описаны в [E005](research/E005_RESULTS.md) и [E016–E017](research/E016_E017_RESULTS.md).

[Практическая ценность проекта](output/pdf/practical_value.pdf).''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
NotebookClient(n,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nb.write(n,ROOT/'EDA.ipynb')
from reportlab.lib import colors
from matplotlib.font_manager import findfont, FontProperties

pdfmetrics.registerFont(TTFont('Body', findfont(FontProperties(family='DejaVu Sans'))))
pdfmetrics.registerFont(TTFont('Bold', findfont(FontProperties(family='DejaVu Sans', weight='bold'))))
out = ROOT / 'output/pdf/practical_value.pdf'
out.parent.mkdir(parents=True, exist_ok=True)

ink = colors.HexColor('#17212D')
muted = colors.HexColor('#4E5C6B')
accent = colors.HexColor('#125B50')
title = ParagraphStyle('Title', fontName='Bold', fontSize=24, leading=29,
                       textColor=ink, spaceAfter=9)
meta = ParagraphStyle('Meta', fontName='Body', fontSize=10, leading=15,
                      textColor=muted, spaceAfter=23)
body = ParagraphStyle('Body', fontName='Body', fontSize=11, leading=17,
                      textColor=ink)
section = ParagraphStyle('Section', fontName='Bold', fontSize=12, leading=17,
                         textColor=accent, spaceBefore=16, spaceAfter=5)

story = [
    Paragraph('Когда стоит переспросить?', title),
    Paragraph('Анна Губарева · Годовой проект · 8 ноября 2026', meta),
    Paragraph(
        'Человек просит подобрать жильё для поездки. У помощника есть каталог, '
        'но он пока не знает, нужна ли кухня и насколько важна близость к метро. '
        'Если можно задать один вопрос, какой из ответов сильнее повлияет на выбор?', body),
    Paragraph('Кому это поможет', section),
    Paragraph(
        'Сервису подбора жилья и его пользователям. Например, ответ «кухня нужна, '
        'а насчёт района пока не знаю» уже позволяет убрать часть вариантов. '
        'Если кухня есть во всех доступных номерах, вопрос о ней мало что даст. '
        'Мы хотим проверить, можно ли оценить такую разницу до ответа пользователя.', body),
    Paragraph('Что можно сделать сейчас', section),
    Paragraph(
        'Первую версию можно проверить на каталоге с ценами, удобствами и коротким '
        'списком вопросов. Самое трудное в реальном сервисе - понять, какие ответы '
        'обычно дают люди, и вовремя обновлять каталог. Если сведения о кухне '
        'пропущены или устарели, даже удачное уточнение может привести к плохой '
        'рекомендации. Пока у нас есть только пробные результаты на синтетических задачах.', body),
    Paragraph('Как проверить пользу', section),
    Paragraph(
        'При одном вопросе на задачу сравнить два способа: всегда задавать один '
        'и тот же вопрос или выбирать его по запросу и вариантам в каталоге. '
        'Основная метрика - доля рекомендаций, которые удовлетворяют обязательным '
        'условиям пользователя. Отдельно сравнить с подбором без уточнений. '
        'В тесте с людьми можно также измерить, как часто они бросают подбор после вопроса.', body),
]
doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=62, rightMargin=62,
                        topMargin=65, bottomMargin=54,
                        title='Когда стоит переспросить?', author='Анна Губарева')
doc.build(story)
print('Created executed EDA.ipynb and', out)
