"""Explicit final-test command, never imported by the training loop."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from revision.data import RevisionDataset,collate
from revision.model import MITS
from revision.config import dump,digest,data_identity
from revision.train import predict,save_csv

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--protocol-frozen',action='store_true');a=p.parse_args()
    if not a.protocol_frozen:p.error('Confirm all validation choices are frozen with --protocol-frozen')
    checkpoint=Path(a.checkpoint);ck=torch.load(checkpoint,map_location='cpu');c=ck['config']
    if ck.get('smoke_not_paper'):raise ValueError('Smoke checkpoints cannot produce paper test results')
    if ck.get('stage13_only'):raise ValueError('Stage 13 checkpoints are for checks/tuning only; retrain the frozen final configuration')
    if digest(Path(c['report_root'])/'final_manifest.csv')!=ck['dataset_hash']:raise ValueError('Dataset changed')
    if ck.get('data_hashes')!=data_identity(c):raise ValueError('Split or feature manifest changed')
    torch.set_num_threads(c['training']['threads'])
    model=MITS(c['model'],ck['role'],c['teacher_type']);model.load_state_dict(ck['model'])
    data=RevisionDataset(c,ck['task'],'test',need_source='source' in model.branches)
    out=checkpoint.parent/'final_test';out.mkdir(exist_ok=False)
    score,rows=predict(model,DataLoader(data,batch_size=c['training']['batch_size'],collate_fn=collate),'cpu',c['training']['threshold'])
    save_csv(out/'predictions.csv',rows);dump(out/'metrics.json',score);dump(out/'provenance.json',{'checkpoint_sha256':digest(checkpoint),'dataset_hash':ck['dataset_hash']})

if __name__=='__main__':main()
