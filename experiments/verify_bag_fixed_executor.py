"""Independent E010 count/hash checks and post-hoc core-reference review.

Human validation and external factual verification are not performed here.
The explicit per-case judgements below were written by the lead assistant after
reading all 48 prediction texts, before printing individual judge labels.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/bag_fixed_executor'

NOTES = {
1: '60% for 2023–24, not the reference slightly below 50 for 2016.',
2: '2010 redesign; does not give the reference 2004 series date.',
3: 'Describes the show and asks for a season; no target winner.',
4: 'Names actresses instead of the requested film language Hindi.',
5: 'Affirms Woody as the singing character; composer/voice claims are not validated.',
6: 'Affirms Dan Marino for NFL 1984; additional league answers and statistics are not validated.',
7: 'Attributes the film version to Tony Bennett, not Billie Holiday.',
8: 'Does not identify Sonus; discusses a custom theme and a different band.',
9: 'Names teams rather than one of the target players.',
10: 'Gives 1908 for men and 2004 for women; mention of 1996 is in the wrong relation.',
11: 'Names Ronaldo Nazario, not Ronaldinho, and describes a specific tournament.',
12: 'Names The Dragon, not Texas SkyScreamer.',
13: 'Claims 15 players, not 2; numbered list is not an answer of 2.',
14: 'Names Sally Field, not Elizabeth Perkins.',
15: 'Names The Wombats, not Sonus.',
16: 'Names West Bromwich Albion and Norwich, not the reference three teams.',
17: 'Names England and discusses a different year, not Australia in 2013.',
18: 'Affirms University of Texas at Austin; accompanying career facts are not validated.',
19: 'Names a different season and winners, not the season-15 target.',
20: 'Names Elizabeth Taylor, not Elizabeth Perkins.',
21: 'Denies that Durant attended college, contrary to the supplied reference.',
22: 'Includes Los Angeles as filming location but omits Marfa from the composite reference.',
23: 'Gives New Zealand locations rather than the specified church.',
24: 'Names actresses instead of the requested film language Hindi.',
25: 'Describes Orwell and the fictional text but not the committee in the reference.',
26: 'Names teams, not one of the target players.',
27: 'Attributes the book to OBrien alone, not to a committee including him.',
28: 'Gives 1908, not 1996 for womens soccer.',
29: 'Names Washington and other single-season records, not Tampa Bay.',
30: 'Names Space Spiral, not Texas SkyScreamer.',
31: 'Names The Move, not Gwen McCrae.',
32: 'Names Shivam Raj, not Ansh Bagri.',
33: 'Names England, not Australia.',
34: 'Gives July 1 2017, not January 7 2018.',
35: 'Gives a different set of relegated teams.',
36: 'Affirms Dan Marino for 1984; numerical and historical additions are not validated.',
37: 'Names The Move, not Gwen McCrae.',
38: 'Gives 1928/2003/2010, not the 2004 series-date reference.',
39: 'Discusses fictional settings rather than the supplied filming locations.',
40: 'Claims 13 players, not 2; numbered list is not an answer of 2.',
41: 'Names Tony Bennett, not Billie Holiday.',
42: 'Denies episode 122 was released, not the supplied release date.',
43: 'Affirms Woody as the character; actor/singing claims are not validated.',
44: 'Names New England for the current season, not the Tampa Bay reference.',
45: 'Mentions Ronaldinho only among other scorers; asserts Ronaldo as the answer.',
46: 'Gives Verona locations, not the specified church.',
47: 'Does not name Ansh Bagri in the saved, visibly cut-off answer.',
48: 'Gives 14.3% for a differently defined Canadian population, not the supplied reference.',
}
YES = {5, 6, 18, 36, 43}
PARTIAL = {22, 27}

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    sample = read(OUT / 'sample_outcomes.json')
    assert [x['blind_id'] for x in sample] == [f'P{i:03d}' for i in range(1, 49)]
    review = {'status': 'post_hoc_single_assistant_provisional',
              'scope': 'Core agreement with supplied reference only; not whole-answer factuality.',
              'aggregate_judge_outcomes_previously_seen': True,
              'sample_sha256': sha(OUT / 'sample_outcomes.json'),
              'rows': [{'blind_id': f'P{i:03d}', 'core_reference_supported': i in YES,
                        'partial_core_overlap': i in PARTIAL, 'rationale': NOTES[i]}
                       for i in range(1, 49)]}
    dump(OUT / 'outcome_core_review.json', review)
    frozen = read(OUT / 'preparation_freeze.json')
    assert sha(ROOT / 'research/E010_PROTOCOL.md') == frozen['protocol_sha256']
    for name, h in frozen['artifact_sha256'].items():
        assert sha(OUT / name) == h, name
    af = read(OUT / 'annotation_freeze.json')
    assert sha(OUT / 'annotations_lead.json') == af['annotations_lead_sha256']
    assert sha(OUT / 'blind_pack.json') == af['blind_pack_sha256']
    assert sha(OUT / 'preparation_freeze.json') == af['preparation_freeze_sha256']
    outcomes = {x['key']: x for x in read(OUT / 'outcomes.json')}
    assert len(outcomes) == 1004
    pairs = [p for p in read(OUT / 'pairs.json') if p['eligible']]
    assert len(pairs) == 452
    recount = []
    for row in read(OUT / 'outcome_summary.json')['paired_primary']:
        selected = [p for p in pairs if (p['a'], p['b']) == (row['a'], row['b'])]
        cells = Counter((outcomes[p['a_key']]['target'], outcomes[p['b_key']]['target']) for p in selected)
        assert all(outcomes[p[k]]['valid_target'] for p in selected for k in ('a_key', 'b_key'))
        assert len(selected) == row['eligible'] == row['valid_pairs']
        for cell, field in [((1,1),'both_positive'),((0,0),'both_negative'),
                            ((1,0),'a_positive_b_negative'),((0,1),'a_negative_b_positive')]:
            assert cells[cell] == row[field]
        recount.append({'a': row['a'], 'b': row['b'], 'n':len(selected),
                        'cells': {str(k):v for k,v in cells.items()}})
    # Reload raw sources without importing the original analyzer.
    raw = Path('/private/tmp/bag-fixed-executor-20261027')
    mf = read(raw / 'manifest.json')
    logs = {}
    for spec in mf['files']:
        path = raw / spec['path']
        assert sha(path) == spec['sha256']
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        index = {r['id']:r for r in rows}
        assert len(index) == len(rows) == 1832
        logs[(spec['variant'],spec['stage'])] = index
    for o in outcomes.values():
        v, sid = o['variant'], o['source_id']
        j = logs[v,'judge'][sid]; f = logs[v,'final'][sid]; u = logs[v,'user'][sid]
        assert j['context']['generation_candidate'] == f['generations'][0]
        assert j['context']['final_ref'] == u['context']['disambiguation']['answer']
        assert j['generations']['final']['verdict'] == o['target']
    disagreements = []
    cells = Counter()
    for s, r in zip(sample, review['rows']):
        cells[int(r['core_reference_supported']), s['target_verdict']] += 1
        if int(r['core_reference_supported']) != s['target_verdict']:
            disagreements.append({'blind_id':s['blind_id'], 'target_verdict':s['target_verdict'],
                                  'core_reference_supported':r['core_reference_supported'],
                                  'rationale':r['rationale'], 'source_id':s['source_id'], 'variant':s['variant']})
    report = {'source_revision':mf['revision'], 'source_files_verified':len(mf['files']),
              'valid_outcome_links':len(outcomes), 'eligible_pair_records':len(pairs),
              'eligible_unique_tasks':len({outcomes[p['a_key']]['source_id'] for p in pairs}),
              'freezes_unchanged':True, 'pair_recount':recount,
              'posthoc_core_review_n':len(sample), 'core_reference_supported':len(YES),
              'partial_core_overlap':len(PARTIAL),
              'core_review_judge_cells':[{'core':a,'judge':b,'n':n} for (a,b),n in sorted(cells.items())],
              'disagreements':disagreements, 'core_review_sha256':sha(OUT/'outcome_core_review.json'),
              'verification_script_sha256':sha(Path(__file__)), 'no_new_external_model_calls':True}
    dump(OUT / 'verification.json', report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
