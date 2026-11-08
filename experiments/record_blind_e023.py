"""Record manually assigned text interpretations before joining E023 source references.

All 36 full inputs in blind_audit.json were read. No audit_references.json or
E023 branch outcomes were displayed to the reviewing assistant before this file.
This is one assistant's review, not independent human annotation.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fresh_screen_e023'
D={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
# Explicitly reported fields; omitted fields are unknown, never inferred negative.
assignments=[
 ({},'Общая просьба; четыре решающих факта не сообщены.'),
 ({'issue':['failed_payment']},'Прямо указан failed payment; списание не сообщено.'),
 ({},'Account без типа; общая просьба.'),
 ({'money_gone':['no']},'Деньги ещё не ушли; раздражение и processing times не задают тип проблемы.'),
 ({},'Общая неопределённая просьба; looked off не задаёт категорию.'),
 ({},'Feeling lost означает растерянность, не lost card. Тип продукта не сообщён.'),
 ({'product':['loan'],'money_gone':['yes']},'Loan и факт ухода денег явно названы; тип проблемы и компенсация нет.'),
 ({'issue':['change_request'],'product':['savings'],'money_gone':['no'],'compensation':['no']},'Все четыре факта явно сообщены.'),
 ({'issue':['fee_question'],'compensation':['no']},'Явный вопрос о комиссии и отказ от компенсации; account не задаёт продукт, posted status не сообщён.'),
 ({'issue':['fee_question'],'product':['card'],'money_gone':['yes'],'compensation':['no']},'Все четыре факта явно сообщены.'),
 ({'issue':['fee_question'],'product':['savings'],'money_gone':['yes'],'compensation':['no']},'Явный вопрос о комиссии savings; unexpected charge не добавляет однозначно другую категорию.'),
 ({'product':['loan']},'Назван только loan.'),
 ({'issue':['fee_question'],'product':['loan'],'money_gone':['yes'],'compensation':['yes']},'Явный fee question, loan, списание и просьба о компенсации.'),
 ({'product':['current_account'],'compensation':['no']},'Продукт и отказ названы; остальные факты не заданы.'),
 ({'issue':['fee_question'],'money_gone':['yes'],'compensation':['yes']},'Названы комиссия, списание, компенсация; продукт неизвестен.'),
 ({'compensation':['no']},'Явный отказ от компенсации; общая просьба не задаёт issue.'),
 ({'compensation':['no']},'Отказ от компенсации; эмоции не задают тип проблемы.'),
 ({'product':['card'],'money_gone':['yes'],'compensation':['yes']},'Карта, уход денег и компенсация явно названы; тип проблемы нет.'),
 ({},'Полностью общая просьба.'),
 ({'product':['card']},'Карта названа, её потеря не сообщена.'),
 ({'issue':['unrecognized_charge']},'Прямо unrecognized charge; generic statement не задаёт продукт или posted status.'),
 ({'product':['loan']},'Назван loan, но не характер проблемы, списание или компенсация.'),
 ({'issue':['unrecognized_charge'],'product':['current_account'],'money_gone':['yes'],'compensation':['yes']},'Все четыре факта явно названы.'),
 ({'issue':['unrecognized_charge'],'product':['current_account'],'money_gone':['no']},'Непризнанная операция и current account названы, деньги ещё не ушли; компенсация не обсуждается.'),
 ({},'Неясность смысла вопроса не означает согласие или отказ от компенсации.'),
 ({'product':['loan'],'compensation':['no']},'Сохраняется loan и отказ от компенсации; ответ money unknown.'),
 ({'issue':['unrecognized_charge'],'compensation':['no']},'Ответ добавляет unrecognized charge и явный отказ от refund.'),
 ({'issue':['unrecognized_charge']},'Issue сохраняется; явная неопределённость компенсации не является отказом.'),
 ({'product':['loan']},'Loan остаётся известным; ответ на вопрос о компенсации неинформативен.'),
 ({'issue':['fee_question'],'product':['card'],'compensation':['no']},'Ответ называет карту и повторяет отказ; деньги не сообщены.'),
 ({'issue':['fee_question'],'product':['savings'],'money_gone':['yes'],'compensation':['no']},'Ответ повторяет уже известное списание, остальные явные факты сохраняются.'),
 ({'issue':['lost_card'],'product':['card'],'money_gone':['yes']},'Ответ добавляет lost card; исходные карта и уход денег сохраняются.'),
 ({'issue':['change_request'],'money_gone':['no']},'Ответ сообщает change request и отсутствие списания; тип продукта не назван.'),
 ({'product':['current_account'],'money_gone':['yes'],'compensation':['no']},'Ответ повторяет current account, остальные исходные факты сохраняются.'),
 ({'product':['current_account']},'Ответ оставляет списание неизвестным; current account известен ранее.'),
 ({'product':['current_account','savings']},'Прямо предложены savings/current account как две альтернативы; буквальная исчерпывающая трактовка является контрактом набора, не универсальным смыслом неуверенной речи.')]
blind=json.loads((OUT/'blind_audit.json').read_text());assert len(blind)==len(assignments)==36
assert not (OUT/'blind_interpretations.json').exists() and not (OUT/'summary.json').exists() and not (OUT/'audit_comparison.json').exists()
labels=[]
for u,(explicit,why) in zip(blind,assignments):
 fs={k:[v for v in dom if v in explicit.get(k,dom)] for k,dom in D.items()};assert all(fs.values())
 labels.append({'unit_id':u['unit_id'],'facts':fs,'ambiguous_fields':['product'] if u['unit_id']=='R-f89480751a6befc9' else [],'rationale':why})
(OUT/'blind_interpretations.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n')
paths=[Path(__file__).resolve(),OUT/'manifest.json',OUT/'blind_audit.json',OUT/'blind_interpretations.json']
manifest={'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'reference_comparison_existed_at_freeze':False,'screen_outcomes_existed_at_freeze':False,'reviewer':'single assistant; source reference fields/kinds/gold not displayed before interpretation; general ontology known; not independent human validation'}
(OUT/'audit_freeze.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'frozen_interpretations':36,'fields':144,'ambiguous_units':1}))
