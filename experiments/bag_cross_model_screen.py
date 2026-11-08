"""E009: pinned, outcome-blind feasibility screen of six public BAG configurations."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REV = '58b2215ce80c74731d6d8dd3a843d2767943f569'
PROTOCOL_SHA = 'f08464cea103454672c28f63e2e3f42fc741a37b2a128474966bc964951b670f'
MODELS = ['gemini-2.5-flash','olmo2-13b-instruct','olmo2-7b-instruct','olmo3-7b-instruct','qwen3-14b','qwen3-8b']
BASE = 'https://huggingface.co/datasets/jsbaan/bag-paper-data'
MARKER = re.compile(r"\bi\s+(?:don't\s+know|do\s+not\s+know|am\s+not\s+sure|cannot\s+answer|can't\s+answer|am\s+unable\s+to\s+answer)\b|\bi'm\s+not\s+sure\b", re.I)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def analyze_annotations(out, pack, mapping, records):
    path=out/'annotations_lead.json'
    if not path.exists(): return
    frozen=json.loads((out/'annotation_freeze.json').read_text())
    assert frozen['annotations_lead_sha256']==digest(path.read_bytes())
    assert frozen['blind_pack_sha256']==digest((out/'blind_pack.json').read_bytes())
    labels=json.loads(path.read_text()); byid={r['blind_id']:r for r in labels}
    assert len(byid)==len(labels)==len(pack) and set(byid)=={r['blind_id'] for r in pack}
    classes=['apparently_complete','partial','ambiguous','explicit_unknown_or_refusal','nonresponsive','unclear']
    for r in labels:
        assert r['surface_class'] in classes and r['annotation_status']=='assistant_provisional'
        assert type(r['has_explicit_unknown']) is bool
        assert r['has_useful_constraint_on_surface'] in ['yes','no','unclear'] and r['rationale']
    recordmap={(r['model'],r['source_id']):r for r in records}
    cells=Counter((recordmap[(m['model'],m['source_id'])]['lexical_unknown_marker'],byid[m['blind_id']]['has_explicit_unknown']) for m in mapping)
    analysis={
      'overall_sample_counts':dict(Counter(r['surface_class'] for r in labels)),
      'per_model':[{'model':model,'n':sum(m['model']==model for m in mapping),
                   'counts':dict(Counter(byid[m['blind_id']]['surface_class'] for m in mapping if m['model']==model))} for model in MODELS],
      'sample_lexical_vs_assistant_unknown':[{'lexical_marker':a,'assistant_unknown':b,'n':n} for (a,b),n in cells.items()],
      'excluding_previously_seen_counts':dict(Counter(byid[m['blind_id']]['surface_class'] for m in mapping if not m['previous_E008_seen'])),
      'limitations':'One provisional assistant review; 20/model, not pooled-population prevalence; no human or interrater validation. Some E008 examples previously seen. No outcome analysis.',
      'annotation_freeze':frozen}
    earlier_path=ROOT/'results/bag_response_audit/blind_mapping_internal.json'
    if earlier_path.exists():
        earlier_ids={r['source_id'] for r in json.loads(earlier_path.read_text())}
        analysis['E008_source_id_overlap']={
            'sample_records_on_previously_examined_source_ids':sum(m['source_id'] in earlier_ids for m in mapping),
            'sample_unique_previously_examined_source_ids':len({m['source_id'] for m in mapping}&earlier_ids),
            'mapping_sha256':digest(earlier_path.read_bytes()),
            'note':'Same source problem can have a different clarification under a new model. Exposure to earlier source problems limits complete blindness, even when this response and outcome were unseen.'}
    write(out/'annotation_analysis.json',analysis)


def fetch(data):
    listing_url = f'https://huggingface.co/api/datasets/jsbaan/bag-paper-data/tree/{REV}/dev?limit=1000'
    with urllib.request.urlopen(listing_url,timeout=45) as r:
        listing = json.load(r)
    lookup={x['path']:x for x in listing}
    def download(model):
        rel=f'dev/user_{model}_gemini-2.5-flash_50_belief.jsonl'
        meta=lookup[rel]; p=data/rel; p.parent.mkdir(parents=True,exist_ok=True)
        url=f'{BASE}/resolve/{REV}/{rel}'
        if p.exists(): b=p.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=60) as r: b=r.read()
            p.write_bytes(b)
        assert len(b)==meta['size'],rel
        git_sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if meta.get('lfs'): assert digest(b)==meta['lfs']['oid'],rel
        else: assert git_sha==meta['oid'],rel
        return {'model':model,'path':rel,'url':url,'bytes':len(b),'sha256':digest(b),'repository_entry':meta}
    with ThreadPoolExecutor(max_workers=6) as pool: files=list(pool.map(download,MODELS))
    write(data/'manifest.json',{'revision':REV,'listing_url':listing_url,'files':files})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/bag_cross_model_screen')
    parser.add_argument('--fetch',action='store_true'); args=parser.parse_args()
    assert digest((ROOT/'research/E009_PROTOCOL.md').read_bytes())==PROTOCOL_SHA
    args.data.mkdir(parents=True,exist_ok=True); args.output.mkdir(parents=True,exist_ok=True)
    if args.fetch: fetch(args.data)
    manifest=json.loads((args.data/'manifest.json').read_text()); assert manifest['revision']==REV
    assert [x['model'] for x in manifest['files']]==MODELS
    inventories=[]; selected=[]; idsets={}; active_sets={}; marker_sets={}; configs={}
    response_records=[]
    for spec in manifest['files']:
        b=(args.data/spec['path']).read_bytes(); assert digest(b)==spec['sha256']
        rows=[json.loads(s) for s in b.splitlines() if s.strip()]
        assert all(isinstance(r.get('id'),str) for r in rows)
        ids=[r['id'] for r in rows]; assert len(ids)==len(set(ids)),spec['path']
        active=[]; markers=[]; missing=[]; unusable=[]
        for r in rows:
            g=r.get('generations'); text=g.get('response') if isinstance(g,dict) else None
            if not isinstance(text,str) or not text.strip():
                if g is not None:
                    unusable.append({'source_id':r['id'],'reason':'missing_or_nonstring_response' if not isinstance(text,str) else 'empty_response',
                                     'generation_keys':list(g) if isinstance(g,dict) else None})
                continue
            ctx=r.get('context') or {}
            if not all(isinstance(ctx.get(k),str) and ctx[k].strip() for k in ['question','clarification_q']):
                missing.append(r['id']); continue
            active.append(r)
            marked=bool(MARKER.search(text.replace('’',"'")))
            if marked: markers.append(r['id'])
            response_records.append({'model':spec['model'],'source_id':r['id'],'lexical_unknown_marker':marked})
        model=spec['model']; idsets[model]=set(ids);active_sets[model]={r['id'] for r in active};marker_sets[model]=set(markers)
        config_values={digest(json.dumps(r.get('generation_config'),sort_keys=True).encode()):r.get('generation_config') for r in rows}
        configs[model]=config_values
        inventories.append({'model':model,'rows':len(rows),'unique_ids':len(set(ids)),'nonnull_generations':sum(r.get('generations') is not None for r in rows),
                            'usable_nonempty_replies':len(active),'missing_context_ids':missing,'lexical_unknown_marker_replies':len(markers),
                            'unusable_nonnull_replies':unusable,
                            'explicit_disambiguation_replies':sum(r['context'].get('disambiguation') is not None for r in active),
                            'generation_config_count':len(config_values)})
        ranked=sorted(active,key=lambda r:digest(f"E009-v1:{spec['path']}:{r['id']}".encode()))
        selected.extend((model,r) for r in ranked[:20])
    random.Random(20261027).shuffle(selected)
    pack=[];mapping=[]
    for n,(model,r) in enumerate(selected,1):
        bid=f'L{n:03d}'
        pack.append({'blind_id':bid,'original_query':r['context']['question'],'clarification_question':r['context']['clarification_q'],'user_answer':r['generations']['response']})
        mapping.append({'blind_id':bid,'source_id':r['id'],'model':model,'previous_E008_seen':model=='qwen3-8b'})
    packbytes=(json.dumps(pack,ensure_ascii=False,indent=2)+'\n').encode()
    if (args.output/'blind_pack.json').exists(): assert (args.output/'blind_pack.json').read_bytes()==packbytes
    (args.output/'blind_pack.json').write_bytes(packbytes)
    (args.output/'blind_pack.txt').write_text('\n\n'.join(f"{r['blind_id']}\nOriginal: {r['original_query']}\nQuestion: {r['clarification_question']}\nReply: {r['user_answer']}" for r in pack)+'\n')
    write(args.output/'mapping_internal.json',mapping);write(args.output/'response_records.json',response_records)
    write(args.output/'source_manifest.json',{**manifest,'license':'CC BY-SA 3.0 per jsbaan/BAG card, inherited from AmbigQA; credit jsbaan/BAG and Min et al. (2020).',
          'protocol_sha256':PROTOCOL_SHA,'code_sha256':digest(Path(__file__).read_bytes()),'generation_configs':configs,
          'blind_pack_sha256':digest(packbytes),'raw_logs_redistributed':False})
    summary={'configurations':inventories,'all_id_sets_equal':all(s==idsets[MODELS[0]] for s in idsets.values()),
             'unique_active_source_ids':len(set().union(*active_sets.values())),
             'active_response_records':len(response_records),'unique_lexical_marker_source_ids':len(set().union(*marker_sets.values())),
             'pairwise_active_overlap':[{'a':a,'b':b,'n':len(active_sets[a]&active_sets[b])} for i,a in enumerate(MODELS) for b in MODELS[i+1:]],
             'blind_sample_records':len(pack),'blind_sample_unique_source_ids':len({x['source_id'] for x in mapping}),
             'previous_E008_seen_sample_records':sum(x['previous_E008_seen'] for x in mapping),
             'interpretation':'Lexical feasibility screen on policy-selected simulated replies. No causal inference or human prevalence; same task can occur across configurations.',
             'new_external_model_calls':0,'outcomes_loaded':False}
    write(args.output/'summary.json',summary)
    analyze_annotations(args.output,pack,mapping,response_records)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
