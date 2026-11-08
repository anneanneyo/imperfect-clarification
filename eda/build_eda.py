# -*- coding: utf-8 -*-
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('# EDA: уточняющие вопросы при неполных ответах\n\nАнна Губарева, 8 ноября 2026.\n\nИсследуем, какой вопрос помогает выбрать подходящий вариант, если пользователь отвечает не полностью. Здесь проверяем данные для эксперимента: пропуски, разметку, распределения и первые результаты исполнителей.\n\nВ наборе 12 задач UserBench, 48 отдельных выборов и 864 варианта. Задачи синтетические и уже использовались при разработке. Официальный test не оценивался.\n\nИсточник: SalesforceAIResearch/UserBench, версия `80506d2ab484cab843e60a2401ff3e0290d05b87`, лицензия Apache-2.0. Из 96 train-задач выбрали 12 по заранее заданному хешу. Подробнее — в [описании источников](THIRD_PARTY.md).')
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
md('## 1. Состав данных и метки\n\nОдна строка — один вариант перелёта, жилья, машины или ресторана. Внутри задачи строки связаны между собой.\n\nВ исходной разметке есть две метки: `source_acceptable` — допустимый вариант, `source_best` — единственный лучший. Дополнительно проверяем два предпочтения пользователя по собственным правилам: T — выполнены оба, F — нарушено хотя бы одно, U — сведений недостаточно. Эти правила пока не проверены независимым разметчиком. U относится к проверке варианта, а не к ответу пользователя «не знаю».\n\nПосмотрим на баланс меток и повторы шаблонов.')
code('''display(df.groupby('aspect').agg(options=('option_id','size'),acceptable=('source_acceptable','sum'),best=('source_best','sum')))
fig,ax=plt.subplots(1,2,figsize=(10,3.5))
pd.crosstab(df.aspect,df.source_acceptable).plot.bar(stacked=True,ax=ax[0],color=['#bec9d2','#16846c']);ax[0].set_title('Source labels by aspect');ax[0].set_ylabel('Options')
df.semantic.value_counts().reindex(['T','F','U']).plot.bar(ax=ax[1],color=['#16846c','#c45b4f','#b3a260']);ax[1].set_title('Provisional criterion conjunction');ax[1].set_ylabel('Options')
plt.tight_layout();plt.show()
s=load('results/userbench_train_e025/summary.json')
print('Train template IDs:',s['train_unique_component_ids'],'unique /',s['train_component_id_count'],'occurrences')
print('Connected components of train tasks:',s['train_task_connected_component_sizes'])''')
md('В каждом выборе 3 допустимых варианта из 18 и один лучший. Всего 144 положительные метки acceptable и 48 best. Баланс задан авторами набора.\n\nВсе 96 train-задач связаны через повторяющиеся шаблоны. Случайное разбиение строк на обучение и тест даст пересечения. Для проверки на новых шаблонах придётся исключить часть задач или собрать другой набор. Два случая смены лучшего вопроса в предыдущем эксперименте тоже имеют общий шаблон перелёта.')
md('## 2. Пропуски\n\nУ разных частей поездки разные поля. Например, у ресторана есть ценовая категория, но нет числовой цены. Заполнять её средней ценой нельзя.\n\nДля услуг различаем три случая: отсутствующий ключ — нет сведений; явный null — услуга недоступна по соглашению источника; ноль — услуга бесплатна. Ниже считаем их отдельно.')
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
md('Неприменимые числовые поля оставляем NaN и сравниваем варианты внутри одной части поездки. Смысл null берём из схемы и инструкций генерации UserBench. Если заменить неизвестную стоимость нулём, неизвестная услуга станет бесплатной.')
md('## 3. Ошибки и выбросы\n\nПроверим отрицательные цены, рейтинги вне шкалы, длины текстов и повторяющиеся ID. Значения за границами IQR отмечаем для просмотра. В меню намеренно добавлены шумовые варианты, поэтому удалять все выбросы нельзя.')
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
md('## 4. Связи между признаками\n\nСчитаем корреляции отдельно для перелётов, жилья и машин: цены у них имеют разный смысл. Для ресторанов числовой цены нет.\n\nВарианты созданы под заданные предпочтения, а строки внутри задачи зависимы. Графики могут отражать способ генерации данных; по ним нельзя установить причинную связь.')
code('''display(df.groupby(['aspect','source_acceptable'])[['base_cost','rating','text_chars','attribute_count']].median())
fig,axs=plt.subplots(1,3,figsize=(13,3.5))
for ax,aspect in zip(axs,['flight','hotel','rental_car']):
 g=df[df.aspect==aspect].copy();g['rating']=g.rating.where(g.rating.between(0,10));g['log_price']=np.log1p(g.base_cost.where(g.base_cost>=0))
 sns.heatmap(g[[c for c in ['log_price','rating','text_chars','attribute_count','source_acceptable'] if g[c].nunique()>1]].corr(method='spearman'),vmin=-1,vmax=1,cmap='vlag',ax=ax);ax.set_title(aspect)
plt.tight_layout();plt.show()
g=df[df.aspect=='rental_car'].copy();g['log_price']=np.log1p(g.base_cost.where(g.base_cost>=0))
sns.pairplot(g[['log_price','text_chars','source_acceptable']],hue='source_acceptable',diag_kind='hist',height=2);plt.show()''')
md('Постоянные признаки исключены из корреляционных таблиц. Если корреляция не определена, это не означает отсутствие связи.\n\nЦену и длину текста можно использовать как признаки. Метки качества, скрытые предпочтения и наши правила проверки доступны только при оценке. Для обучения выбора вопроса ещё потребуется задать возможные ответы пользователя и их вероятности.')
md('## 5. Первые модели\n\nСравниваем случайный выбор, эвристику по совпадению слов и цене и Qwen3-4B-Instruct-2507 в 4-битной версии. Модель использована без дообучения. Каждый из 192 входов запущен один раз; скрытые предпочтения и метки в запрос не включались.\n\nПроверяем четыре режима: без дополнительных сведений, оба предпочтения, только первое, только второе. В каждом режиме сведения раскрываются для всех четырёх частей поездки. Это проверка исполнителя; выбор одного вопроса оценивается отдельно.')
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
md('Qwen выбрала допустимый по исходной разметке вариант в 10/48 случаев без дополнительных сведений и в 31/48 при раскрытии обоих предпочтений. У эвристики — 19,44% и 45,83%. Числа для Qwen выше пересчитаны напрямую из сохранённых ответов.\n\nПри сравнении отдельных вопросов у Qwen не найдено строгой смены единственного лучшего вопроса по нашим правилам проверки с учётом неизвестных значений. По исходным меткам есть один случай расхождения между полным и частичным ответом с учётом ничьих. Пока этого недостаточно, чтобы говорить о преимуществе метода выбора вопроса.')
md('## 6. Аугментации\n\nПроверим, меняет ли эвристика выбор при перестановке вариантов и изменении пробелов. Для обеих операций смысл данных сохраняется. Qwen в этой проверке не запускаем.\n\nУдаление части предпочтений меняет доступную информацию, поэтому считаем его отдельным условием эксперимента. Перефразирование пока не используем: сначала нужно проверить, сохраняет ли оно предпочтения.')
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
md('## 7. Выводы\n\n- В данных есть ошибки числовых полей и спорные эталоны. Одного совпадения с source-best недостаточно для оценки решения.\n- Пропуски обозначают разные вещи: отсутствие сведений, недоступность услуги или неприменимость поля. Заполнять их одним значением нельзя.\n- Дополнительные сведения улучшают выбор Qwen на этом наборе. Строгую смену лучшего вопроса при неполном ответе по нашим правилам пока не нашли.\n- Перестановка вариантов и изменение пробелов не изменили распределение выбора эвристики на всех 192 входах.\n\nДальше нужно независимо пересчитать ранжирования вопросов и разобрать ошибки модели. Затем — проверить результат на новых задачах с контролем повторяющихся шаблонов.\n\nОграничения: 12 уже просмотренных синтетических задач, предварительные правила проверки, отсутствие ответов реальных пользователей. 864 варианта не являются независимыми задачами. Обученного оценщика пользы вопроса пока нет.\n\n[Практическая ценность проекта](output/pdf/practical_value.pdf).')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
NotebookClient(n,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nb.write(n,ROOT/'EDA.ipynb')
from matplotlib.font_manager import findfont, FontProperties
pdfmetrics.registerFont(TTFont('Arial',findfont(FontProperties(family='DejaVu Sans'))))
pdfmetrics.registerFont(TTFont('ArialBold',findfont(FontProperties(family='DejaVu Sans',weight='bold'))))
out=ROOT/'output/pdf/practical_value.pdf';out.parent.mkdir(parents=True,exist_ok=True)
style=ParagraphStyle('Body',fontName='Arial',fontSize=10.5,leading=15,spaceAfter=10)
heading=ParagraphStyle('Title',fontName='ArialBold',fontSize=16,leading=21,spaceAfter=18)
paras=[('Практическая ценность проекта',heading),('Выбор уточняющего вопроса при неполном ответе пользователя. Анна Губарева, 8 ноября 2026.',style),
('Задача. Помощник при выборе услуги или товара часто получает неполный запрос. Уточнения тратят время, а ответы бывают частичными. Мы исследуем, можно ли выбирать вопрос по ожидаемому улучшению конечной рекомендации, учитывая, что ответ может не раскрыть все необходимые сведения.',style),
('Кому это полезно. Возможные пользователи - сервисы подбора поездок и товаров, службы поддержки и внутренние помощники. Например, вопрос о номере с кухней может оказаться полезнее вопроса о виде из окна, если он сильнее меняет доступный набор подходящих гостиниц. Польза для человека - меньше лишних вопросов и меньше рекомендаций, противоречащих его условиям.',style),
('Что можно реализовать. Прототип получает запрос, историю, варианты решения и доступные уточнения; возвращает вопрос либо предлагает действовать. Начать можно с ограниченного каталога и явных условий. Сейчас готов экспериментальный протокол и локальная проверка исполнителя. Обученного оценщика пользы вопроса и проверки в реальном сервисе пока нет.',style),
('Основные ограничения. Синтетический пользователь не заменяет ответы людей. В данных встречаются спорные эталоны и повторяющиеся шаблоны; 12 разобранных задач недостаточно для оценки обобщения. Реальные цены и доступность услуг меняются. Текст может не содержать сведений, нужных для проверки рекомендации. Личные предпочтения следует хранить минимально необходимое время и не передавать внешним сервисам без согласия. Для решений с существенными последствиями потребуется подтверждение человеком.',style),
('Как оценивать результат. При одинаковом бюджете вопросов сравнивать долю подходящих конечных решений, нарушения обязательных условий, число уточнений и отказы пользователя. Дополнительно измерять время ответа и вычислительные затраты. После локального этапа нужна проверка на новых задачах и добровольных пользователях. Сравнения: без уточнения, один постоянный вопрос, простая эвристика и обученный выбор вопроса.',style),
('Исследовательская ценность. Даже при отсутствии улучшения важно установить, когда уточнение бесполезно из-за неполного ответа, ошибок понимания или неверной метрики. Эти причины требуют разных решений. Публикационный вклад и преимущество предлагаемого подхода пока не установлены.',style)]
SimpleDocTemplate(str(out),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=35).build([Paragraph(t,s) for t,s in paras])
print('Created executed EDA.ipynb and',out)
