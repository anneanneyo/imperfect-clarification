"""Join frozen blind text interpretations to source labels without changing either."""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fresh_screen_e023'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for mn in ['manifest.json','audit_freeze.json']:
 for n,h in read(OUT/mn)['files'].items():assert sha(ROOT/n)==h,n
assert not (OUT/'audit_comparison.json').exists()
labels=read(OUT/'blind_interpretations.json');refs={r['unit_id']:r for r in read(OUT/'audit_references.json')};blind={r['unit_id']:r for r in read(OUT/'blind_audit.json')};assert len(labels)==len(refs)==len(blind)==36
sizes={'issue':5,'product':4,'money_gone':2,'compensation':2};joined=[]
for l in labels:
 r=refs[l['unit_id']];diff=[k for k in sizes if set(l['facts'][k])!=set(r['facts'][k])]
 joined.append({**r,'messages':blind[l['unit_id']]['messages'],'blind_facts':l['facts'],'blind_ambiguous_fields':l['ambiguous_fields'],'blind_rationale':l['rationale'],'disagreeing_fields':diff})
summary={'reviewer':'single assistant; interpretations frozen before reference join; not independent human annotation','source_labels_changed':False,'groups':len({r['group_id'] for r in joined}),'type_results':{}}
for typ in ['initial','reply']:
 rs=[r for r in joined if r['type']==typ];correct=known=known_ok=unknown=unknown_ok=0
 for r in rs:
  for k,z in sizes.items():
   ok=k not in r['disagreeing_fields'];correct+=ok
   if len(r['facts'][k])<z:known+=1;known_ok+=ok
   else:unknown+=1;unknown_ok+=ok
 summary['type_results'][typ]={'units':len(rs),'fields':4*len(rs),'exact_fields':correct,'exact_units':sum(not r['disagreeing_fields'] for r in rs),'known_fields':known,'known_exact':known_ok,'unknown_fields':unknown,'unknown_exact':unknown_ok,'blind_ambiguous_fields':sum(len(r['blind_ambiguous_fields']) for r in rs)}
summary['reply_kinds']=dict(Counter(r['kind'] for r in joined if r['type']=='reply'))
for name,x in [('audit_comparison.json',joined),('audit_summary.json',summary)]: (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
paths=[Path(__file__).resolve(),OUT/'audit_freeze.json',OUT/'audit_references.json',OUT/'audit_comparison.json',OUT/'audit_summary.json'];(OUT/'audit_score_manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}},indent=2)+'\n');print(json.dumps(summary))
for r in joined:
 if r['disagreeing_fields']:print(json.dumps(r,ensure_ascii=False))
