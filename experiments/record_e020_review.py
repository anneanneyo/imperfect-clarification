"""Record one assistant's unblinded review of all 90 remaining E020 disagreements.

The judgments below were assigned after reading all 67 complete messages and their
90 mismatched fields. They are qualitative judgments, not independent annotations.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/e020_semantic_review';SOURCE=ROOT/'results/evidence_facts_e020'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(SOURCE/'verification.json')['passed']
for n,h in read(SOURCE/'score_manifest.json')['files'].items():assert sha(ROOT/n)==h
assert not OUT.exists()
rows=read(ROOT/'results/e020_diagnostics/remaining_mismatches.json');allcases=read(SOURCE/'details.json');byid={c['case_id']:c for c in allcases}
gaps={('00115','product'),('00455','product'),('00234','product'),('00447','product'),('00281','product'),('00084','product')}
ambiguous={
 ('00127','money_gone'):'Просьба вернуть сумму на счёт естественно предполагает списание, но не задаёт явно posted/pending статус; source оставляет оба варианта.',
 ('00131','issue'):'Ошибка входа не входит ни в одну из пяти issue-категорий; набор категорий не покрывает описанную проблему.',
 ('00334','compensation'):'Вопрос о возможности компенсации допускает чтение как запрос или как предварительное выяснение права.',
 ('00340','issue'):'Текст одновременно упоминает сомнительную операцию и вопрос о комиссии; граница двух категорий спорна.',
 ('00455','issue'):'Текст явно описывает и потерю карты, и неавторизованную операцию; однокатегорийный эталон не описывает оба события.',
 ('00403','compensation'):'Фраза I was wondering if I am asking for money back неоднозначна и не является обычной прямой просьбой.',
 ('00116','issue'):'Ошибка входа не входит в пять перечисленных категорий; failed_payment не следует из неё, но и source full-domain не обозначает out-of-ontology.',
 ('00031','compensation'):'Пользователь спрашивает о возможности и праве на компенсацию; граница inquiry/request спорна.',
 ('00484','issue'):'Вопрос о комиссии сочетается с неожиданным списанием; uncertain charge и unrecognized charge разграничены неявно.'}
missed={('00491','product'):'Карта явно названа, но raw модель сохранила весь домен даже с подходящей цитатой.',('00315','compensation'):'Явно написано not a request for monetary compensation or a refund; модель оставила оба значения.'}
default_reasons={
 'issue':'Общая просьба, упоминание продукта или факт списания не устанавливают выбранный тип проблемы и не исключают остальные категории.',
 'product':'Текст не называет выбранный тип продукта; generic account/charge/payment не устанавливают card/current_account и не исключают loan.',
 'money_gone':'Согласие или отказ от компенсации и общая просьба не устанавливают, списаны ли уже деньги.',
 'compensation':'Просьба о помощи или проверке операции не устанавливает просьбу о денежной компенсации.'}
labels=[]
for r in rows:
 key=(r['case_id'].rsplit('/',1)[-1],r['field']);d=byid[r['case_id']];k=r['field']
 if r['quote_check']['reset_to_unknown']:
  assert d['raw'][k]==d['facts'][k]
  cat='gate_discarded_source_correct_value';why='Raw совпал с эталоном, но механическая проверка отвергла цитату из-за длины или несовпадения строки; это отдельный механизм потери факта.'
  if key==('00176','compensation'):why+=' Сам вопрос о возможности компенсации также допускает смысловую неоднозначность.'
 elif key in gaps:cat='model_supported_annotation_incomplete';why='Потерянная карта прямо названа в тексте; product=card поддержан, хотя исходная аннотация оставляет продукт неизвестным.'
 elif key in ambiguous:cat='ambiguous_text_or_ontology';why=ambiguous[key]
 elif key in missed:cat='missed_explicit_fact';why=missed[key]
 else:
  cat='model_not_supported_by_text';why=default_reasons[k]
  assert r['quote_check']['quote_valid'] and r['quote_check']['narrowed']
 labels.append({**r,'assessment':cat,'rationale':why})
assert len(labels)==90 and len({r['case_id'] for r in labels})==67
summary={'reviewer':'single assistant, unblinded post-outcome; all remaining disagreements; no independent human validation','cases':67,'fields':90,'assessments':dict(Counter(r['assessment'] for r in labels)),'primary_metrics_unchanged':True,'by_dataset':{}}
for dataset in ['E014_dev','E016_train']:
 rs=[r for r in labels if r['dataset']==dataset];counts=dict(Counter(r['assessment'] for r in rs));s=read(SOURCE/'summary.json')[dataset]['gated']
 # An optimistic accounting exercise, not a corrected performance estimate.
 credited=sum(r['assessment']in ['gate_discarded_source_correct_value','model_supported_annotation_incomplete','ambiguous_text_or_ontology'] for r in rs)
 summary['by_dataset'][dataset]={'cases':len({r['case_id'] for r in rs}),'fields':len(rs),'assessments':counts,'optimistic_credit_sensitivity':{'original_exact':s['slot_exact'],'additional_credited_disagreements':credited,'exact_if_all_these_credited':s['slot_exact']+credited,'total':s['slot_total'],'interpretation':'Post hoc sensitivity only; assume every mechanical/source/ambiguous disagreement correct; not a revised score or semantic accuracy bound.'}}
resets=[]
for d in allcases:
 for k,c in d['checks'].items():
  if not c['reset_to_unknown']:continue
  q=d['quotes'][k]['evidence'];reason='null' if q is None else 'empty' if not q else 'length>160' if len(q)>160 else 'not_substring'
  effect='helps' if d['gated'][k]==d['facts'][k] else 'hurts' if d['raw'][k]==d['facts'][k] else 'neutral'
  resets.append(dict(case_id=d['case_id'],dataset=d['dataset'],field=k,reason=reason,effect=effect,quote=q,raw=d['raw'][k],gated=d['gated'][k],reference=d['facts'][k]))
attribution={'resets':len(resets),'effect_counts':dict(Counter(r['effect'] for r in resets)),'by_dataset':{dataset:{reason:dict(Counter(r['effect'] for r in resets if r['dataset']==dataset and r['reason']==reason)) for reason in sorted({r['reason'] for r in resets if r['dataset']==dataset})} for dataset in ['E014_dev','E016_train']}}
OUT.mkdir()
for name,x in [('labels.json',labels),('summary.json',summary),('quote_gate_details.json',resets),('quote_gate_summary.json',attribution)]: (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
paths=[Path(__file__).resolve(),ROOT/'results/e020_diagnostics/remaining_mismatches.json',SOURCE/'details.json',SOURCE/'verification.json']+[OUT/n for n in ['labels.json','summary.json','quote_gate_details.json','quote_gate_summary.json']]
(OUT/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}},indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False));print(json.dumps(attribution))
