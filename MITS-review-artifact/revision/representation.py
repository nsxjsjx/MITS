"""Aligned CKA/probes from fixed checkpoints; train/validation by default."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from revision.model import MITS
from revision.data import RevisionDataset,collate
from revision.config import dump,digest,data_identity
from revision.probe import linear_cka,fit_probe
from revision.metrics import metrics

def load_checkpoint(path,role,allow_stage13=False):
    ck=torch.load(path,map_location='cpu')
    if ck['role']!=role or ck.get('smoke_not_paper'):raise ValueError('Wrong role or smoke checkpoint')
    if ck.get('stage13_only') and not allow_stage13:raise ValueError('Stage13 checkpoint is not a formal result')
    c=ck['config']
    if ck.get('data_hashes')!=data_identity(c):raise ValueError('Dataset/split identity changed')
    if ck['dataset_hash']!=digest(Path(c['report_root'])/'final_manifest.csv'):raise ValueError('Manifest changed')
    model=MITS(c['model'],role,c['teacher_type']);model.load_state_dict(ck['model']);model.eval()
    return ck,model

def collect(ck,model,split,layer,device):
    c=ck['config'];ds=RevisionDataset(c,ck['task'],split,need_source='source' in model.branches)
    loader=DataLoader(ds,batch_size=c['training']['batch_size'],collate_fn=collate)
    model.to(device);ids=[];groups=[];labels=[];features=[];paired=[]
    semantic='source' if 'source' in model.branches else 'derived'
    if layer=='semantic' and semantic not in model.branches:raise ValueError('Selected semantic branch missing')
    with torch.no_grad():
        for batch in loader:
            graph=batch['graph'].to(device);tokens=batch['tokens'].to(device)
            src=batch['source_vector'].to(device) if 'source' in model.branches else None
            out=model(graph,tokens,src,return_branches=True)
            value=out['distill_features'] if layer=='fused' else out['branches'][semantic]
            features.append(value.cpu().numpy());ids.extend(batch['sample_id']);groups.extend(batch['split_group']);labels.extend(batch['label'].tolist())
            paired.extend(batch['source_available'].tolist() if 'source' in model.branches else [True]*len(batch['sample_id']))
    order=np.argsort(ids)
    return dict(ids=np.asarray(ids)[order],groups=np.asarray(groups)[order],labels=np.asarray(labels,dtype=int)[order],features=np.concatenate(features)[order],paired=np.asarray(paired,dtype=bool)[order])

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--student',required=True)
    p.add_argument('--reference-student',help='Optional identical beta0 student, same task/seed/data')
    p.add_argument('--layer',choices=['semantic','fused'],default='semantic');p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True);p.add_argument('--test',action='store_true');p.add_argument('--protocol-frozen',action='store_true')
    p.add_argument('--allow-stage13',action='store_true');a=p.parse_args()
    if a.test and (not a.protocol_frozen or a.allow_stage13):p.error('Test requires frozen formal protocol, never stage13')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    dump(out/'status.json',{'status':'RUNNING'})
    try:
        paths={'teacher':a.teacher,'student':a.student}
        if a.reference_student:paths['reference_student']=a.reference_student
        loaded={name:load_checkpoint(path,'teacher' if name=='teacher' else 'student',a.allow_stage13) for name,path in paths.items()}
        base=loaded['student'][0]
        for name,(ck,_) in loaded.items():
            if any(ck[k]!=base[k] for k in ['task','seed','dataset_hash','data_hashes']):raise ValueError('Checkpoint task/seed/data mismatch: '+name)
        if 'reference_student' in loaded and loaded['reference_student'][0]['config']['distillation']['beta']!=0:
            raise ValueError('Reference must be beta0 for the prespecified transfer comparison')
        torch.set_num_threads(base['config']['training']['threads'])
        splits=['train','validation']+(['test'] if a.test else [])
        exported={}
        for name,(ck,model) in loaded.items():
            exported[name]={}
            for split in splits:
                values=collect(ck,model,split,a.layer,a.device);exported[name][split]=values
                np.savez_compressed(out/(name+'_'+split+'.npz'),**values)
        result={'layer':a.layer,'task':base['task'],'seed':base['seed'],'checkpoint_hashes':{n:digest(Path(f)) for n,f in paths.items()},'stage13_only':a.allow_stage13,'cka':{},'probes':{},'warning':'Auxiliary association evidence, not causal proof of semantic transfer. eval disables Dropout; fused is the same aligned tensor position but deterministic at evaluation.'}
        for split in splits:
            teacher=exported['teacher'][split]
            for name in paths:
                vals=exported[name][split]
                if not np.array_equal(teacher['ids'],vals['ids']) or not np.array_equal(teacher['labels'],vals['labels']) or not np.array_equal(teacher['groups'],vals['groups']):raise ValueError('IDs/labels/groups not aligned')
                if name=='teacher':continue
                mask=teacher['paired']
                result['cka'][name+'_'+split]={'n':int(mask.sum()),'linear_cka':linear_cka(teacher['features'][mask],vals['features'][mask]) if mask.sum()>=2 else None}
        # Same paired training rows for every probe, regularization selected on validation only.
        mask=exported['teacher']['train']['paired'];vmask=exported['teacher']['validation']['paired']
        for name in paths:
            train=exported[name]['train'];val=exported[name]['validation']
            if len(np.unique(train['labels'][mask]))<2:raise ValueError('Probe training needs both classes')
            fitted,selection=fit_probe(train['features'][mask],train['labels'][mask],val['features'][vmask],val['labels'][vmask],base['seed'])
            result['probes'][name]={'selection':selection,'train_n':int(mask.sum()),'validation_n':int(vmask.sum()),'metrics':{}}
            for split in splits[1:]:
                values=exported[name][split];m=exported['teacher'][split]['paired'];prob=fitted.predict_proba(values['features'][m])[:,list(fitted.classes_).index(1)]
                result['probes'][name]['metrics'][split]=metrics(values['labels'][m],prob)
            np.savez_compressed(out/(name+'_probe_coefficients.npz'),coef=fitted.coef_,intercept=fitted.intercept_,classes=fitted.classes_,C=fitted.C)
        dump(out/'results.json',result);dump(out/'status.json',{'status':'COMPLETED','test_evaluated':a.test,'stage13_only':a.allow_stage13})
        print('Representation analysis saved:',out)
    except Exception:
        import traceback
        dump(out/'status.json',{'status':'FAILED','error':traceback.format_exc()});raise

if __name__=='__main__':main()
