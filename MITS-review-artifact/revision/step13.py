"""Manual stage-13 search and timing. Never evaluates held-out test data."""
import argparse,copy,csv,json,re,time,traceback
from pathlib import Path
import yaml
from revision.config import ROOT,dump,code_hashes
from revision.step13_common import local_config,session,forbid_test_reads,read
from revision.experiments import configure
from revision.train import run

def write_rows(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def validate_plan(plan):
    if plan['search']['max_session_minutes']<=0:raise ValueError('Session budget must be positive')
    for profile in ['cpu','server']:
        setting=plan['search'][profile]
        if any(setting[k]<1 for k in ['epochs','patience','batch_size','max_run_seconds']):raise ValueError('Invalid search budget')
        if setting['train_per_class'] is not None and setting['train_per_class']<1:raise ValueError('Invalid pilot sample count')
    ids=[]
    for t in plan['search']['trials']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',t['id']):raise ValueError('Trial IDs must be safe, distinct directory names')
        if not 0<float(t['temperature'])<float('inf') or not 0<=float(t['beta'])<float('inf'):raise ValueError('Invalid T or beta')
        ids.append(t['id'])
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('Empty or duplicate trial IDs')

def require_checks(path):
    if path is None:
        candidates=sorted((ROOT/'step13_results').glob('*_checks_*/summary.json'))
        if not candidates:raise ValueError('Run python test_revision_pipeline.py first; no stage-13 check result exists')
        path=candidates[-1]
    result=read(path)
    if result.get('status')!='PASS' or len(result.get('tasks_passed',[]))!=4:raise ValueError('All four task checks must pass before search/benchmark')
    checked={k.replace('\\','/'):v for k,v in read(Path(path).parent/'code_hashes.json').items()}
    current={k.replace('\\','/'):v for k,v in code_hashes().items()}
    for name in ['revision/model.py','revision/data.py','revision/train.py','revision/distillation.py']:
        if checked.get(name)!=current.get(name):raise ValueError('Core code changed after checks; rerun checks: '+name)
    return str(path)

def search(c,plan,out,tasks,profile):
    setting=plan['search'][profile];base_config=copy.deepcopy(c);c=copy.deepcopy(c)
    c['training'].update(epochs=setting['epochs'],patience=setting['patience'],batch_size=setting['batch_size'])
    c['stage13']={'mode':'search','train_per_class':setting['train_per_class'],'validation_per_class':None,
                  'small_graphs':False,'audit_updates':False,'max_run_seconds':setting['max_run_seconds']}
    seed=plan['seed'];rows=[];teachers={};start=time.perf_counter();limit=plan['search']['max_session_minutes']*60
    # All choices, budgets and task scope are saved before the first batch.
    dump(out/'search_protocol.json',{'plan':plan,'tasks':tasks,'profile':profile,'seed':seed,'config':c,
        'selection':'equal-weight four-task best validation F1; ties retain earlier declared trial',
        'cpu_caution':'CPU subset + 3 epochs is screening only, not final hyperparameter selection'})
    with forbid_test_reads(c) as attempts:
        for task in tasks:
            if time.perf_counter()-start>=limit:break
            cfg=configure(c,'main');cfg['output_root']=str(out/'teachers')
            try:
                folder=run(cfg,task,seed,'teacher');teachers[task]=folder/'best.pt'
                rows.append({'kind':'teacher','task':task,'trial':'shared','status':'COMPLETED','folder':str(folder),**read(folder/'status.json')})
            except Exception:rows.append({'kind':'teacher','task':task,'trial':'shared','status':'FAILED','error':traceback.format_exc()})
            write_rows(out/'trials.csv',rows)
        for trial in plan['search']['trials']:
            for task in tasks:
                record={'kind':'student','task':task,'trial':trial['id'],'T':trial['temperature'],'beta':trial['beta']}
                if task not in teachers:record.update(status='NOT_RUN',error='teacher unavailable')
                elif time.perf_counter()-start>=limit:record.update(status='NOT_RUN',error='session budget reached')
                else:
                    cfg=configure(c,'main');cfg['output_root']=str(out/'trials'/trial['id']);cfg['teacher_checkpoint']=str(teachers[task])
                    cfg['distillation'].update(temperature=float(trial['temperature']),beta=float(trial['beta']),loss_type='none' if trial['beta']==0 else 'feature_kl')
                    try:
                        folder=run(cfg,task,seed,'student');record.update(folder=str(folder),**read(folder/'status.json'))
                    except Exception:record.update(status='FAILED',error=traceback.format_exc())
                rows.append(record);write_rows(out/'trials.csv',rows)
        scores=[]
        for trial in plan['search']['trials']:
            rr=[r for r in rows if r['kind']=='student' and r['trial']==trial['id']]
            complete=len(rr)==4 and set(tasks)==set(c['tasks']) and all(r['status']=='COMPLETED' for r in rr)
            scores.append({'trial':trial['id'],'T':trial['temperature'],'beta':trial['beta'],'complete_four_tasks':complete,
                           'mean_val_f1':sum(r['best_val_f1'] for r in rr)/4 if complete else None})
        write_rows(out/'validation_ranking.csv',scores)
        all_complete=all(r['complete_four_tasks'] for r in scores)
        best=max(scores,key=lambda r:r['mean_val_f1']) if all_complete else None
        dump(out/'summary.json',{'status':'COMPLETED' if all_complete else 'INCOMPLETE','test_open_attempts':attempts,
            'profile':profile,'best_validation_trial':best,'final_configuration_frozen':False,
            'elapsed_seconds':time.perf_counter()-start,'warning':'CPU results only screen candidates. Preserve every run; inspect convergence, then confirm on server/full training data before freezing.'})
        if best and profile=='server':
            candidate=base_config;candidate['distillation'].update(temperature=best['T'],beta=best['beta'],loss_type='none' if best['beta']==0 else 'feature_kl')
            (out/'candidate_revision_NOT_FROZEN.yaml').write_text(yaml.safe_dump(candidate,allow_unicode=True,sort_keys=False),encoding='utf8')

def benchmark(c,plan,out,tasks):
    c=copy.deepcopy(c);c['training'].update(epochs=1,patience=1)
    c['stage13']={'mode':'benchmark','train_per_class':None,'validation_per_class':None,'audit_updates':False,'max_run_seconds':plan['search']['server']['max_run_seconds']}
    c['output_root']=str(out/'runs');rows=[];timings={};seed=plan['seed']
    with forbid_test_reads(c) as attempts:
        for task in tasks:
            for role in ['teacher','student']:
                try:
                    folder=run(configure(c,'main'),task,seed,role);status=read(folder/'status.json');history=read(folder/'history.json');provenance=read(folder/'provenance.json')
                    row={'task':task,'role':role,'status':'COMPLETED','device':provenance['device'],
                         'train_count':provenance['train_count'],'validation_count':provenance['validation_count'],
                         'one_epoch_wall_seconds':status['elapsed_seconds'],'train_seconds':history[0]['train_seconds'],
                         'validation_seconds':history[0]['validation_seconds'],'folder':str(folder)}
                    timings[(task,role)]=row
                except Exception:row={'task':task,'role':role,'status':'FAILED','error':traceback.format_exc()}
                rows.append(row);write_rows(out/'timings.csv',rows)
    estimates=[];lo,hi=plan['budget']['illustrative_epoch_range']
    for stage in plan['budget']['stages']:
        for task in tasks:
            measured=timings.get((task,stage['timing_proxy']))
            if measured:
                count=plan['budget']['final_seeds'];seconds=measured['one_epoch_wall_seconds']
                estimates.append({'stage':stage['name'],'task':task,'runs':count,'device':measured['device'],
                    'low_hours':seconds*lo*count/3600,'high_hours':seconds*hi*count/3600,'epochs_low':lo,'epochs_high':hi,
                    'assumption':'linear same-machine one-epoch proxy; includes initial overhead each epoch; not a measured variant runtime'})
    write_rows(out/'core_budget_estimate.csv',estimates)
    dump(out/'summary.json',{'status':'COMPLETED' if len(timings)==2*len(tasks) else 'INCOMPLETE','test_open_attempts':attempts,
        'core_only_low_hours':sum(r['low_hours'] for r in estimates),'core_only_high_hours':sum(r['high_hours'] for r in estimates),
        'scope':'five configured core stages only; excludes remaining ablations, baselines, API latency, preprocessing and retries',
        'caution':'CPU estimates apply only to this CPU. Repeat benchmark on server; do not convert CPU hours to GPU hours. Epoch range is an illustrative scenario, not a guaranteed finish time.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['search','benchmark']);p.add_argument('--config');p.add_argument('--plan',default=str(ROOT/'step13_plan.yaml'))
    p.add_argument('--profile',choices=['cpu','server'],default='cpu');p.add_argument('--device',choices=['cpu','cuda','auto']);p.add_argument('--task');p.add_argument('--checks-report');a=p.parse_args()
    c=local_config(a.config);plan=yaml.safe_load(Path(a.plan).read_text(encoding='utf-8-sig'));validate_plan(plan)
    checked=require_checks(a.checks_report);tasks=[a.task] if a.task else c['tasks']
    if any(t not in c['tasks'] for t in tasks):p.error('Unknown task')
    c['training'].update(threads=plan['threads'],device=a.device or ('cpu' if a.profile=='cpu' else 'cuda'))
    out=session(a.mode,c);dump(out/'plan.json',plan);dump(out/'checks_reference.json',{'summary':checked});print('Results:',out,flush=True)
    try:
        if a.mode=='search':search(c,plan,out,tasks,a.profile)
        else:benchmark(c,plan,out,tasks)
    except BaseException:
        dump(out/'fatal.json',{'error':traceback.format_exc()});raise
    print('Finished. Send this folder:',out,flush=True)

if __name__=='__main__':main()
