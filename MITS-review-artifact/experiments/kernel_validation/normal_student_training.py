"""Train/validation only. Test is a separate command after the protocol is frozen."""
import argparse,copy,csv,json,os,platform,random,time,traceback
from pathlib import Path
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from revision.config import ROOT,load_config,dump,digest,code_hashes,data_identity
from revision.data import RevisionDataset,collate
from revision.model import MITS,parameter_report,validate_node_matching
from revision.distillation import loss
from revision.metrics import metrics
from revision.experiments import configure,VARIANTS

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)

def move(batch,device):
    return {k:v.to(device) if hasattr(v,'to') else v for k,v in batch.items()}

def forward(model,b):return model(b['graph'],b['tokens'],b['source_vector'] if 'source' in model.branches else None)

def predict(model,loader,device,threshold):
    model.eval(); rows=[];bces=[]
    with torch.no_grad():
        for batch in loader:
            b=move(batch,device);out=forward(model,b);p=out['logits'].sigmoid().cpu().tolist()
            bces.extend(torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],b['label'],reduction='none').cpu().tolist())
            for sid,group,y,prob,logit in zip(b['sample_id'],b['split_group'],b['label'].cpu().tolist(),p,out['logits'].cpu().tolist()):
                rows.append(dict(sample_id=sid,split_group=group,label=int(y),probability=prob,logit=logit,prediction=int(prob>=threshold)))
    result=metrics([r['label'] for r in rows],[r['probability'] for r in rows],threshold);result['bce']=float(np.mean(bces))
    return result,rows

def save_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def save_checkpoint(path,state):
    # Older PyTorch Windows builds cannot reliably open Unicode paths themselves.
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(state,f)
    os.replace(tmp,path)

def parameter_hash(model):
    return __import__('hashlib').sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()

def validate_teacher_checkpoint(ckpt,c,task,seed,data_hashes):
    if ckpt['role']!='teacher' or ckpt['task']!=task or ckpt['seed']!=seed:raise ValueError('Teacher role/task/seed mismatch')
    if ckpt['config']['data_version']!=c['data_version'] or ckpt.get('data_hashes')!=data_hashes:raise ValueError('Teacher dataset identity mismatch')
    for key in ['training','source_coverage','teacher_type','source_strategy','shuffle_instructions','coverage_seed','source_encoder_local','stage13']:
        if ckpt['config'].get(key)!=c.get(key):raise ValueError('Teacher protocol mismatch: '+key)
    expected=copy.deepcopy(c['kernel_validation']['frozen_teacher_model']);saved=copy.deepcopy(ckpt['config']['model']);expected.pop('branches');saved.pop('branches')
    if expected!=saved:raise ValueError('Teacher architecture mismatch')
    current=code_hashes()
    for module in ['revision/model.py','revision/data.py','revision/train.py','utils/bytecode_to_token.py']:
        native=str(Path(module))
        if ckpt.get('code_hashes',{}).get(native)!=current.get(native):raise ValueError('Teacher implementation changed: '+module)

def limit_pilot_rows(dataset,per_class,small_graphs=False):
    if per_class is None:return
    selected=[]
    for label in [0,1]:
        rows=[r for r in dataset.rows if int(r['label'])==label]
        rows.sort(key=(lambda r:(int(r.get('graph_nodes') or 0),r['sample_id'])) if small_graphs else (lambda r:r['sample_id']))
        if not rows:raise ValueError('Pilot subset has no examples of one class')
        selected.extend(rows[:per_class])
    dataset.rows=selected

def run(c,task,seed,role,smoke=False):
    c=copy.deepcopy(c);c['run_seed']=seed;t=c['training'];torch.set_num_threads(t['threads'])
    if c['variant'].startswith('node_'):
        c['node_matching']=validate_node_matching(c['model'],c['node_matching'])
    seed_all(seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if t['device']=='auto' else t['device'])
    base=Path(c['output_root']);base=base if base.is_absolute() else ROOT/base
    if smoke:base=base/'SMOKE_NOT_PAPER'
    folder=base/c['variant']/role/task/str(seed);folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'config.json',c)
    dump(folder/'status.json',{'status':'RUNNING','smoke_not_paper':smoke,'stage13_only':bool(c.get('stage13'))})
    start=time.perf_counter();process=psutil.Process();peak_rss=process.memory_info().rss
    try:
        dataset_hash=digest(Path(c['report_root'])/'final_manifest.csv');data_hashes=data_identity(c)
        teacher_type=c['teacher_type'];need_source=(role=='teacher' and teacher_type=='multimodal') or (role=='student' and c['distillation']['beta']>0 and teacher_type=='multimodal')
        train=RevisionDataset(c,task,'train',need_source=need_source,teacher_subset=role=='teacher' and teacher_type=='multimodal')
        val=RevisionDataset(c,task,'validation',need_source=role=='teacher' and teacher_type=='multimodal')
        pilot=c.get('stage13',{})
        if pilot:
            limit_pilot_rows(train,pilot.get('train_per_class'),pilot.get('small_graphs',False))
            limit_pilot_rows(val,pilot.get('validation_per_class'),pilot.get('small_graphs',False))
        elif smoke:train.rows=train.rows[:4];val.rows=val.rows[:4]
        dump(folder/'used_ids.json',{'train':[r['sample_id'] for r in train.rows],'validation':[r['sample_id'] for r in val.rows]})
        generator=torch.Generator().manual_seed(seed+1000)
        labels=torch.tensor([int(r['label']) for r in train.rows]);counts=torch.bincount(labels,minlength=2)
        if not smoke and (counts==0).any():raise ValueError('Training subset lacks a class')
        sampler=WeightedRandomSampler((1/counts.clamp_min(1).float())[labels],len(labels),replacement=True,generator=generator) if t['weighted_sampling'] else None
        loader=DataLoader(train,batch_size=t['batch_size'],sampler=sampler,shuffle=sampler is None,collate_fn=collate,generator=generator)
        valloader=DataLoader(val,batch_size=t['batch_size'],collate_fn=collate)
        model=MITS(c['model'],role,teacher_type).to(device);teacher=None;teacher_hash=None
        initial_hash=__import__('hashlib').sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        if role=='student' and c['distillation']['beta']>0 and c['distillation']['loss_type']!='none':
            path=Path(c['teacher_checkpoint']) if c.get('teacher_checkpoint') else base/c['teacher_variant']/'teacher'/task/str(seed)/'best.pt'
            ckpt=torch.load(path,map_location='cpu')
            validate_teacher_checkpoint(ckpt,c,task,seed,data_hashes)
            if ckpt['dataset_hash']!=dataset_hash or ckpt['seed']!=seed or ckpt['task']!=task:raise ValueError('Teacher identity mismatch')
            if ckpt['config']['source_coverage']!=c['source_coverage'] or ckpt['config']['teacher_type']!=teacher_type:raise ValueError('Teacher coverage/modality mismatch')
            for key in ['training','source_strategy','shuffle_instructions','coverage_seed','source_encoder_local']:
                if ckpt['config'].get(key)!=c.get(key):raise ValueError('Teacher protocol mismatch: '+key)
            expected_model=copy.deepcopy(c['kernel_validation']['frozen_teacher_model']);saved_model=copy.deepcopy(ckpt['config']['model'])
            expected_model.pop('branches');saved_model.pop('branches')
            if expected_model!=saved_model:raise ValueError('Teacher architecture mismatch')
            if ckpt.get('data_hashes')!=data_hashes:raise ValueError('Teacher split/feature manifest changed')
            current_hashes=code_hashes()
            for module in ['revision/model.py','revision/data.py','revision/train.py','utils/bytecode_to_token.py']:
                native=str(Path(module))
                if ckpt.get('code_hashes',{}).get(native)!=current_hashes.get(native):raise ValueError('Teacher implementation changed: '+module)
            teacher=MITS(ckpt['config']['model'],'teacher',teacher_type).to(device);teacher.load_state_dict(ckpt['model']);teacher.eval();teacher.requires_grad_(False);teacher_hash=digest(path)
            # Teacher construction must not change student dropout streams vs beta=0.
            seed_all(seed+2000)
        else:seed_all(seed+2000)
        if t.get('optimizer','adam')!='adam' or t.get('scheduler','none')!='none':raise ValueError('Unsupported optimizer/scheduler')
        optimizer=torch.optim.Adam(model.parameters(),lr=t['learning_rate'],weight_decay=t.get('weight_decay',0.0))
        teacher_before=parameter_hash(teacher) if teacher is not None and pilot.get('audit_updates') else None
        checked_graphs=0
        hashes=code_hashes();dump(folder/'config.json',c)
        dump(folder/'provenance.json',dict(dataset_hash=dataset_hash,data_hashes=data_hashes,code_hashes=hashes,seed=seed,teacher_hash=teacher_hash,
             initial_parameter_hash=initial_hash,parameter_report=parameter_report(model),torch=torch.__version__,python=platform.python_version(),
             device=str(device),hardware=platform.platform(),cpu=platform.processor(),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
             train_count=len(train),validation_count=len(val),coverage_policy='train groups masked; teacher validation uses paired validation source',
             source_available_train=sum(task+'/'+r['sample_id'] in train.available for r in train.rows)))
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        best=float('-inf');stale=0;history=[];best_epoch=None
        epochs=1 if smoke else t['epochs']
        for epoch in range(1,epochs+1):
            epoch_start=time.perf_counter()
            model.train();total=0.;n=0;bce_sum=0.;kd_sum=0.;train_y=[];train_p=[]
            for batch in loader:
                if pilot.get('max_run_seconds') and time.perf_counter()-start>pilot['max_run_seconds']:
                    raise TimeoutError('Stage 13 per-run time budget reached; this run is not a completed trial')
                b=move(batch,device)
                graph_before=(b['graph'].x.clone(),b['graph'].edge_index.clone()) if pilot.get('audit_updates') else None
                optimizer.zero_grad(set_to_none=True);s=forward(model,b);teacher_out=None
                paired=b['source_available'] if teacher_type=='multimodal' else torch.ones_like(b['source_available'])
                if teacher is not None:
                    with torch.no_grad():teacher_out=forward(teacher,b)
                args=c['distillation'] if role=='student' else {'beta':0,'loss_type':'none'}
                objective,parts=loss(s,b['label'],teacher_out,paired,**args);objective.backward();optimizer.step()
                if graph_before is not None:
                    if not torch.equal(graph_before[0],b['graph'].x) or not torch.equal(graph_before[1],b['graph'].edge_index):raise AssertionError('Forward mutated graph input')
                    checked_graphs+=1
                size=len(b['label']);total+=objective.item()*size;n+=size;bce_sum+=parts['bce']*size;kd_sum+=parts['kd']*size
                train_y.extend(b['label'].detach().cpu().tolist());train_p.extend(s['logits'].detach().sigmoid().cpu().tolist())
                peak_rss=max(peak_rss,process.memory_info().rss)
            train_seconds=time.perf_counter()-epoch_start;val_start=time.perf_counter()
            scores,predictions=predict(model,valloader,device,t['threshold'])
            val_seconds=time.perf_counter()-val_start
            improved=scores['f1']>best
            if improved:best=scores['f1'];best_epoch=epoch;stale=0
            else:stale+=1
            record=dict(epoch=epoch,train_loss=total/n,train_bce=bce_sum/n,train_kd_weighted_batch_mean=kd_sum/n,
                        sampled_train_metrics=metrics(train_y,train_p,t['threshold']),val=scores,train_seconds=train_seconds,validation_seconds=val_seconds,elapsed_seconds=time.perf_counter()-start)
            history.append(record);dump(folder/'history.json',history)
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict() if not pilot else None,config=c,task=task,seed=seed,role=role,epoch=epoch,best_epoch=best_epoch,
                       dataset_hash=dataset_hash,data_hashes=data_hashes,teacher_hash=teacher_hash,code_hashes=hashes,smoke_not_paper=smoke,stage13_only=bool(pilot),
                       rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       sampler_rng=generator.get_state(),rng_numpy=np.random.get_state(),rng_python=random.getstate())
            save_checkpoint(folder/'last.pt',state)
            if improved:save_checkpoint(folder/'best.pt',state);save_csv(folder/'validation_predictions.csv',predictions)
            print(json.dumps(dict(task=task,seed=seed,role=role,epoch=epoch,val_f1=scores['f1']),ensure_ascii=False),flush=True)
            if stale>=t['patience']:break
        if device.type=='cuda':torch.cuda.synchronize(device)
        audit={'student_or_teacher_updated':parameter_hash(model)!=initial_hash,'teacher_unchanged':parameter_hash(teacher)==teacher_before if teacher_before else None,'graph_batches_unchanged':checked_graphs}
        if pilot.get('audit_updates'):
            if not audit['student_or_teacher_updated'] or audit['teacher_unchanged'] is False:raise AssertionError('Parameter update/freeze check failed')
        dump(folder/'update_audit.json',audit)
        dump(folder/'status.json',dict(status='COMPLETED',smoke_not_paper=smoke,stage13_only=bool(pilot),best_epoch=best_epoch,best_val_f1=best,
             elapsed_seconds=time.perf_counter()-start,peak_sampled_rss_bytes=peak_rss,
             peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
        return folder
    except BaseException:
        dump(folder/'status.json',dict(status='FAILED',stage13_only=bool(c.get('stage13')),elapsed_seconds=time.perf_counter()-start,error=traceback.format_exc()));raise

def main(default_role=None,default_variant=None):
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--role',choices=['teacher','student'],default=default_role or 'student')
    p.add_argument('--variant',choices=VARIANTS,default=default_variant or 'main');p.add_argument('--task');p.add_argument('--seed',type=int);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();c=configure(load_config(a.config),a.variant)
    if a.task and a.task not in c['tasks']:p.error('Unknown task')
    for task in ([a.task] if a.task else c['tasks']):
        for seed in ([a.seed] if a.seed is not None else c['seeds']):run(c,task,seed,a.role,a.smoke)

if __name__=='__main__':main()
