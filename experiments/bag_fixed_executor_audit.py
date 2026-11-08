"""E010: staged, outcome-blind preparation and descriptive paired-log analysis."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import re
import string
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REV='58b2215ce80c74731d6d8dd3a843d2767943f569'
PROTOCOL_SHA='630d9a296d057d2bfcf87dd8fade3a65fd6551381cdaeb1daad61d62a4cbb471'
VARIANTS=['belief','prompt','belief6','belief7']


def sha(b): return hashlib.sha256(b).hexdigest()
def digest(x): return sha(json.dumps(x,sort_keys=True,ensure_ascii=False).encode())
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def normalized(s):
    s=''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())


def fetch(data,availability):
    tasks=[(v,stage,spec) for v,stages in availability['configurations'].items() for stage,spec in stages.items()]
    def one(task):
        v,stage,spec=task; assert spec['available']; rel=spec['path'];meta=spec['metadata']
        path=data/rel;path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/datasets/jsbaan/bag-paper-data/resolve/{REV}/{rel}'
        if path.exists(): b=path.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=60) as r: b=r.read()
            path.write_bytes(b)
        assert len(b)==meta['size']
        if meta.get('lfs'): assert sha(b)==meta['lfs']['oid']
        else: assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==meta['oid']
        return {'variant':v,'stage':stage,'path':rel,'url':url,'sha256':sha(b),'bytes':len(b)}
    with ThreadPoolExecutor(max_workers=6) as pool: files=list(pool.map(one,tasks))
    write(data/'manifest.json',{'revision':REV,'files':files})


def read_rows(path):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    assert all(isinstance(r.get('id'),str) for r in rows)
    byid={r['id']:r for r in rows}; assert len(byid)==len(rows),f'Duplicate ID: {path}'
    return byid


def load(data,stage):
    manifest=json.loads((data/'manifest.json').read_text());assert manifest['revision']==REV
    logs={v:{} for v in VARIANTS}
    for spec in manifest['files']:
        p=data/spec['path'];assert sha(p.read_bytes())==spec['sha256']
        if stage=='prepare' and spec['stage']=='judge': continue
        logs[spec['variant']][spec['stage']]=read_rows(p)
    assert len(manifest['files'])==12
    return logs,manifest


def executor_config(cfg,strict=False):
    out={k:cfg.get(k) for k in ['model_name','assistant_model','mode','direct_prompt','final_prompt','use_8bit','sampling']}
    params=dict(cfg.get('resolved_generation_params') or {})
    if not strict: params.pop('batch_size',None)
    out['resolved_generation_params']=params
    return out


def prepare(logs,out,manifest):
    branches={}; stage_audits={};config_values={}; all_source_sets={}
    for v in VARIANTS:
        users=logs[v]['user'];finals=logs[v]['final'];active=0; records=[]
        all_source_sets[v]=set(users)
        assert set(users)==set(finals),f'ID mismatch: {v}'
        for sid,u in users.items():
            g=u.get('generations') or {}; text=g.get('response') if isinstance(g,dict) else None
            if not isinstance(text,str) or not text.strip():continue
            active+=1;ctx=u.get('context') or {};f=finals[sid];fc=f.get('context') or {};cfg=f.get('generation_config') or {}
            goal=ctx.get('disambiguation');refs=ctx.get('reference');why=[]
            if not all(isinstance(ctx.get(k),str) and ctx[k].strip() for k in ['question','clarification_q']):why.append('missing_query_or_question')
            valid_goal=isinstance(goal,dict) and isinstance(goal.get('question'),str) and bool(goal['question'].strip()) and isinstance(goal.get('answer'),list) and bool(goal['answer']) and all(isinstance(a,str) and a.strip() for a in goal['answer'])
            if not valid_goal:why.append('missing_or_invalid_explicit_goal')
            elif refs != [goal['answer']]:why.append('reference_not_selected_goal')
            for key in ['question','clarification_q','reference','disambiguation']:
                if ctx.get(key)!=fc.get(key):why.append('final_context_mismatch_'+key)
            if text!=fc.get('user_answer'):why.append('final_context_mismatch_user_answer')
            prompt=[{'role':'user','content':ctx.get('question')},{'role':'assistant','content':ctx.get('clarification_q')},{'role':'user','content':text}]
            if f.get('prompt')!=prompt:why.append('unexpected_final_prompt')
            expected=f"user\n{ctx.get('question')}\nassistant\n{ctx.get('clarification_q')}\nuser\n{text}\nassistant\n<think>\n\n</think>"
            if f.get('processed_prompt')!=expected:why.append('processed_prompt_not_verified')
            if cfg.get('model_name')!='qwen3-8b' or cfg.get('assistant_model')!='qwen3-8b':why.append('unexpected_executor_model')
            if cfg.get('final_prompt')!='prompt' or cfg.get('direct_prompt')!='vanilla':why.append('unexpected_final_prompt_mode')
            ec=executor_config(cfg);strict=executor_config(cfg,True)
            config_values[digest(ec)]=ec
            key=f'{v}:{sid}'
            record={'key':key,'variant':v,'source_id':sid,'eligible_branch':not why,'exclusion_reasons':why,
                    'query_hash':digest(ctx.get('question')),'question_hash':digest(ctx.get('clarification_q')),
                    'answer_hash':digest(text),'goal_hash':digest(goal),'reference_hash':digest(refs),
                    'prompt_hash':digest(prompt),'executor_config_hash':digest(ec),'strict_executor_config_hash':digest(strict),
                    'batch_size':(cfg.get('resolved_generation_params') or {}).get('batch_size')}
            branches[key]=record;records.append(record)
        stage_audits[v]={'rows':len(users),'active_replies':active,'eligible_branches':sum(r['eligible_branch'] for r in records),
                         'exclusion_reasons':dict(Counter(x for r in records for x in r['exclusion_reasons'])),
                         'batch_sizes':dict(Counter(str(r['batch_size']) for r in records))}
    pairs=[];pair_summaries=[]
    for a,b in combinations(VARIANTS,2):
        aa={r['source_id']:r for r in branches.values() if r['variant']==a};bb={r['source_id']:r for r in branches.values() if r['variant']==b}
        records=[]
        for sid in sorted(set(aa)&set(bb)):
            ra,rb=aa[sid],bb[sid];why=[]
            if not ra['eligible_branch']:why.append('ineligible_a')
            if not rb['eligible_branch']:why.append('ineligible_b')
            for k in ['query_hash','goal_hash','reference_hash','executor_config_hash']:
                if ra[k]!=rb[k]:why.append('mismatch_'+k)
            record={'a':a,'b':b,'source_id':sid,'a_key':ra['key'],'b_key':rb['key'],'eligible':not why,
                    'exclusion_reasons':why,'different_question':ra['question_hash']!=rb['question_hash'],
                    'identical_prompt':ra['prompt_hash']==rb['prompt_hash'],
                    'same_question_different_answer':ra['question_hash']==rb['question_hash'] and ra['answer_hash']!=rb['answer_hash'],
                    'strict_config_match':ra['strict_executor_config_hash']==rb['strict_executor_config_hash']}
            pairs.append(record);records.append(record)
        pair_summaries.append({'a':a,'b':b,'active_union':len(set(aa)|set(bb)),'active_intersection':len(records),
                               'eligible':sum(r['eligible'] for r in records),'different_question':sum(r['eligible'] and r['different_question'] for r in records),
                               'identical_prompt':sum(r['eligible'] and r['identical_prompt'] for r in records),
                               'same_question_different_answer':sum(r['eligible'] and r['same_question_different_answer'] for r in records),
                               'strict_config_match':sum(r['eligible'] and r['strict_config_match'] for r in records),
                               'exclusions':dict(Counter(x for r in records for x in r['exclusion_reasons']))})
    source_ids={p['source_id'] for p in pairs if p['eligible'] and p['different_question']}
    chosen=sorted(source_ids,key=lambda sid:sha(('E010-v1:'+sid).encode()))[:24]
    keys=sorted({k for p in pairs if p['eligible'] and p['source_id'] in chosen for k in [p['a_key'],p['b_key']]})
    random.Random(20261027).shuffle(keys);pack=[];mapping=[]
    for i,key in enumerate(keys,1):
        r=branches[key];u=logs[r['variant']]['user'][r['source_id']];ctx=u['context'];bid=f'P{i:03d}'
        pack.append({'blind_id':bid,'original_query':ctx['question'],'clarification_question':ctx['clarification_q'],
                     'user_answer':u['generations']['response'],'selected_disambiguation':ctx['disambiguation']})
        mapping.append({'blind_id':bid,'key':key,'variant':r['variant'],'source_id':r['source_id']})
    artifacts={'branches.json':list(branches.values()),'pairs.json':pairs,'blind_pack.json':pack,'mapping_internal.json':mapping}
    for name,value in artifacts.items():
        encoded=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();path=out/name
        if path.exists():assert path.read_bytes()==encoded,f'Frozen preparation changed: {name}'
        else:path.write_bytes(encoded)
    summary={'variants':stage_audits,'all_source_id_sets_equal':all(x==all_source_sets['belief'] for x in all_source_sets.values()),
             'pairs':pair_summaries,'unique_eligible_source_ids':len({p['source_id'] for p in pairs if p['eligible']}),
             'unique_different_question_source_ids':len(source_ids),'annotation_sample_source_ids':len(chosen),'annotation_sample_branches':len(pack),
             'executor_configs':config_values,'new_outcomes_inspected':False,'new_external_model_calls':0}
    write(out/'preparation_summary.json',summary)
    write(out/'preparation_freeze.json',{'protocol_sha256':PROTOCOL_SHA,'source_manifest_sha256':digest(manifest),
                                       'artifact_sha256':{name:sha((out/name).read_bytes()) for name in artifacts},
                                       'note':'Eligibility and sample frozen without inspecting new predictions or judge outcomes. Baseline belief previously analyzed in E008.'})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def verify_freezes(out,manifest=None):
    frozen=json.loads((out/'preparation_freeze.json').read_text())
    assert frozen['protocol_sha256']==PROTOCOL_SHA
    if manifest is not None:assert frozen['source_manifest_sha256']==digest(manifest)
    for name,h in frozen['artifact_sha256'].items():assert sha((out/name).read_bytes())==h,name
    af=json.loads((out/'annotation_freeze.json').read_text())
    assert af['preparation_freeze_sha256']==sha((out/'preparation_freeze.json').read_bytes())
    assert af['annotations_lead_sha256']==sha((out/'annotations_lead.json').read_bytes())
    assert af['blind_pack_sha256']==sha((out/'blind_pack.json').read_bytes())


def pair_table(pairs,outcomes):
    result=[]
    for a,b in combinations(VARIANTS,2):
        base=[p for p in pairs if p['eligible'] and p['a']==a and p['b']==b]
        for subset in ['all_compatible','different_question','identical_prompt','same_question_different_answer','strict_config_match']:
            selected=[p for p in base if subset=='all_compatible' or p[subset]]
            complete=[p for p in selected if outcomes[p['a_key']]['valid_target'] and outcomes[p['b_key']]['valid_target']]
            cells=Counter((outcomes[p['a_key']]['target'],outcomes[p['b_key']]['target']) for p in complete)
            total=len(selected);n=len(complete);difference=cells[(1,0)]-cells[(0,1)]
            result.append({'a':a,'b':b,'subset':subset,'eligible':total,'valid_pairs':n,'missing_or_invalid_pairs':total-n,
                           'both_positive':cells[(1,1)],'both_negative':cells[(0,0)],'a_positive_b_negative':cells[(1,0)],'a_negative_b_positive':cells[(0,1)],
                           'mean_a_minus_b_valid':difference/n if n else None,
                           'conservative_missing_bounds_over_eligible':[(difference-(total-n))/total,(difference+(total-n))/total] if total else None})
    return result


def analyze(logs,out,manifest):
    verify_freezes(out,manifest)
    branches=json.loads((out/'branches.json').read_text());pairs=json.loads((out/'pairs.json').read_text())
    labels=json.loads((out/'annotations_lead.json').read_text());mapping=json.loads((out/'mapping_internal.json').read_text())
    labelmap={r['blind_id']:r for r in labels}; assert len(labelmap)==len(labels)==len(mapping)
    assert set(labelmap)=={m['blind_id'] for m in mapping}
    for r in labels:
        assert r['surface_class'] in ['apparently_complete','partial','ambiguous','explicit_unknown_or_refusal','nonresponsive','unclear']
        assert r['asks_user_for_final_fact'] in ['yes','no','mixed','unclear']
        assert r['intent_alignment'] in ['aligned','contradictory','unclear']
        assert r['annotation_status']=='assistant_provisional' and r['rationale']
        assert r['unresolved_explicit_goal_detail'] is None or isinstance(r['unresolved_explicit_goal_detail'],str)
    for v in VARIANTS:assert set(logs[v]['user'])==set(logs[v]['final'])==set(logs[v]['judge']),v
    outcomes={};samples=[]
    for r in branches:
        if not r['eligible_branch']:continue
        v,sid=r['variant'],r['source_id'];f=logs[v]['final'][sid];j=logs[v]['judge'][sid];u=logs[v]['user'][sid]
        goal=u['context']['disambiguation'];jg=j.get('generations') or {};jc=j.get('context') or {};fg=f.get('generations');why=[]
        prediction=fg[0] if isinstance(fg,list) and len(fg)==1 and isinstance(fg[0],str) and fg[0].strip() else None
        if prediction is None:why.append('missing_single_final_prediction')
        if jc.get('question')!=u['context']['question']:why.append('judge_question_mismatch')
        if prediction is None or jc.get('generation_candidate')!=prediction:why.append('judge_candidate_mismatch')
        if jc.get('final_ref')!=goal['answer']:why.append('judge_target_reference_mismatch')
        target=(jg.get('final') or {}).get('verdict')
        if type(target) is not int or target not in [0,1]:why.append('missing_or_invalid_target_verdict')
        refs=[normalized(s) for s in goal['answer'] if normalized(s)]
        em=normalized(prediction) in refs if prediction is not None and refs else None
        outcomes[r['key']]={'key':r['key'],'variant':v,'source_id':sid,'valid_target':not why,'issues':why,
                            'target':target,'normalized_exact_match':em,'prediction_hash':digest(prediction),
                            'reference_hash':digest(goal['answer'])}
    byvariant=[]
    for v in VARIANTS:
        selected=[r for r in outcomes.values() if r['variant']==v];valid=[r for r in selected if r['valid_target']]
        byvariant.append({'variant':v,'eligible_branches':len(selected),'valid_target':len(valid),
                          'target_positive':sum(r['target']==1 for r in valid),
                          'normalized_em_true':sum(r['normalized_exact_match'] is True for r in selected),
                          'judge_em_cells':[{'judge':a,'em':b,'n':n} for (a,b),n in Counter((r['target'],r['normalized_exact_match']) for r in valid).items()],
                          'issues':dict(Counter(x for r in selected for x in r['issues']))})
    sample_rows=[]
    for m in mapping:
        r=labelmap[m['blind_id']];o=outcomes[m['key']];u=logs[m['variant']]['user'][m['source_id']];f=logs[m['variant']]['final'][m['source_id']]
        sample_rows.append({**m,**r,'valid_target':o['valid_target'],'target':o['target'],'normalized_exact_match':o['normalized_exact_match']})
        samples.append({'blind_id':m['blind_id'],'variant':m['variant'],'source_id':m['source_id'],
                        'original_query':u['context']['question'],'clarification_question':u['context']['clarification_q'],
                        'user_answer':u['generations']['response'],'selected_disambiguation':u['context']['disambiguation'],
                        'final_prediction':f['generations'][0] if isinstance(f.get('generations'),list) and f['generations'] else None,
                        'target_verdict':o['target'],'normalized_exact_match':o['normalized_exact_match']})
    sample_summary={field:dict(Counter(r[field] for r in sample_rows)) for field in ['surface_class','asks_user_for_final_fact','intent_alignment']}
    sample_summary['unresolved_goal_detail_flagged']=sum(r['unresolved_explicit_goal_detail'] is not None for r in sample_rows)
    sample_summary['target_by_surface']=[{'surface_class':label,'n':len(rs),'target_positive':sum(r['valid_target'] and r['target']==1 for r in rs)}
                                        for label in sorted({r['surface_class'] for r in sample_rows}) for rs in [[r for r in sample_rows if r['surface_class']==label]]]
    table=pair_table(pairs,outcomes)
    write(out/'outcomes.json',list(outcomes.values()));write(out/'paired_results.json',table)
    write(out/'sample_analysis.json',{'summary':sample_summary,'rows':sample_rows,'caution':'Provisional single-assistant labels; diagnostic sample, not causal mechanism or population prevalence.'})
    write(out/'sample_outcomes.json',samples)
    summary={'by_variant':byvariant,'paired_primary':[r for r in table if r['subset']=='all_compatible'],
             'semantic_sample_summary':sample_summary,'protocol_sha256':PROTOCOL_SHA,
             'preparation_freeze_sha256':sha((out/'preparation_freeze.json').read_bytes()),
             'annotation_freeze_sha256':sha((out/'annotation_freeze.json').read_bytes()),
             'new_external_model_calls':0,'stage':'descriptive_outcome_analysis_after_freezes',
             'limitations':['Same recorded model/decoding/batch config, not verified identical weights snapshot or RNG.',
                           'Policy-selected intersecting cases and one stochastic continuation per branch; no propensities, no causal claim.',
                           'Target judge unvalidated as ground truth; exact match is a restricted diagnostic.',
                           'No verified no-question branch. No new trained selector or H3 comparison.']}
    write(out/'outcome_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/bag_fixed_executor')
    parser.add_argument('--stage',choices=['prepare','analyze'],default='prepare');parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args();assert sha((ROOT/'research/E010_PROTOCOL.md').read_bytes())==PROTOCOL_SHA
    args.data.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    avail_path=ROOT/'results/fixed_executor_feasibility/availability.json';avail=json.loads(avail_path.read_text())
    if args.fetch:fetch(args.data,avail)
    if args.stage=='analyze':verify_freezes(args.output)
    logs,manifest=load(args.data,args.stage)
    write(args.output/'source_manifest.json',{**manifest,'availability_sha256':sha(avail_path.read_bytes()),
          'protocol_sha256':PROTOCOL_SHA,'code_sha256':sha(Path(__file__).read_bytes()),
          'license':'CC BY-SA 3.0 per source card; credit jsbaan/BAG and AmbigQA (Min et al., 2020). Raw full logs remain outside the project.'})
    if args.stage=='prepare':prepare(logs,args.output,manifest)
    else:analyze(logs,args.output,manifest)


if __name__=='__main__':main()
