"""Aggregate complete runs without score filtering; align IDs and seeds explicitly."""
import argparse,csv,json
from pathlib import Path
from revision.config import dump,load_config
from revision.metrics import summarize,paired_runs,paired_sample_test,group_bootstrap,four_task_mean

def read(path):return json.loads(path.read_text(encoding='utf8'))
def predictions(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return {r['sample_id']:r for r in csv.DictReader(f)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--variant',default='main');p.add_argument('--compare');p.add_argument('--output',required=True);p.add_argument('--split',choices=['validation','test'],default='validation');a=p.parse_args()
    c=load_config();root=Path(a.root);results={};missing=[];per_seed={s:{} for s in c['seeds']}
    for task in c['tasks']:
        values={};other={};sample_tests={};all_metrics={}
        for seed in c['seeds']:
            folder=root/a.variant/'student'/task/str(seed)
            if not (folder/'status.json').exists() or read(folder/'status.json')['status']!='COMPLETED':missing.append(str(folder));continue
            if read(folder/'status.json').get('smoke_not_paper'):raise ValueError('Cannot summarize smoke runs as paper results')
            if read(folder/'status.json').get('stage13_only'):raise ValueError('Use the stage13 search report, not formal paper aggregation')
            if a.split=='validation':score=read(folder/'status.json')['best_val_f1'];predfile=folder/'validation_predictions.csv'
            else:score=read(folder/'final_test/metrics.json')['f1'];predfile=folder/'final_test/predictions.csv'
            if a.split=='validation':
                best_epoch=read(folder/'status.json')['best_epoch'];metric_row=next(r['val'] for r in read(folder/'history.json') if r['epoch']==best_epoch)
            else:metric_row=read(folder/'final_test/metrics.json')
            for name,value in metric_row.items():
                if value is not None and name not in ['n','tn','fp','fn','tp']:all_metrics.setdefault(name,[]).append(value)
            values[seed]=score;per_seed[seed][task]=score
            if a.compare:
                rival=root/a.compare/'student'/task/str(seed)
                if not (rival/'status.json').exists() or read(rival/'status.json')['status']!='COMPLETED':missing.append(str(rival));continue
                if read(rival/'status.json').get('smoke_not_paper'):raise ValueError('Smoke comparison forbidden')
                if read(rival/'status.json').get('stage13_only'):raise ValueError('Stage13 results are not formal runs')
                other[seed]=read(rival/'status.json')['best_val_f1'] if a.split=='validation' else read(rival/'final_test/metrics.json')['f1']
                pa=predictions(predfile);pb=predictions(rival/('validation_predictions.csv' if a.split=='validation' else 'final_test/predictions.csv'))
                if pa.keys()!=pb.keys():raise ValueError('Compared methods have different coverage')
                ids=sorted(pa)
                if any(pa[i]['label']!=pb[i]['label'] or pa[i]['split_group']!=pb[i]['split_group'] for i in ids):raise ValueError('Prediction labels/groups disagree')
                y=[int(pa[i]['label']) for i in ids];x=[int(pa[i]['prediction']) for i in ids];z=[int(pb[i]['prediction']) for i in ids]
                sample_tests[seed]=dict(mcnemar=paired_sample_test(y,x,z),cluster_bootstrap=group_bootstrap(y,x,z,[pa[i]['split_group'] for i in ids]))
        if values:results[task]=dict(f1=summarize(list(values.values())),metrics={k:summarize(v) for k,v in all_metrics.items()},by_seed=values,complete=len(values)==len(c['seeds']))
        if a.compare and values.keys()==other.keys() and values:results[task].update(paired=paired_runs(values,other),per_seed_sample_tests=sample_tests)
    complete=[four_task_mean(v) for v in per_seed.values() if len(v)==4]
    dump(a.output,dict(split=a.split,results=results,missing_runs=missing,four_task_equal_mean_f1=summarize(complete) if complete else None,
                      note='Mean F1 is mean of per-run F1, not F1 reconstructed from mean precision/recall. P-values are unadjusted exploratory comparisons; apply a prespecified multiplicity policy before significance claims.'))

if __name__=='__main__':main()
