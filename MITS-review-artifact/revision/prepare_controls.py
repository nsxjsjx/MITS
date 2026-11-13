"""Optional feature controls; write separate caches, never alter frozen inputs."""
import argparse,hashlib,multiprocessing as mp,queue,time,traceback
from pathlib import Path
import torch
from revision.config import load_config,dump,digest
from revision.data import read_manifest
from revision.train import save_checkpoint

def shuffle_worker(c,r,seed,path,q):
    try:
        from revision.features import graph_from_bytecode
        source=Path(c['data_root'])/r['task']/'bytecode_pure'/(r['sample_id']+'.sol')
        rng=int(hashlib.sha256((str(seed)+r['sample_id']).encode()).hexdigest()[:8],16)
        graph=graph_from_bytecode(source.read_text().strip(),c,shuffle_seed=rng)
        original=torch.load(Path(c['data_root'])/r['task']/'cfg_pyg_data'/(r['sample_id']+'.pt'),map_location='cpu')
        if len(graph.x)!=len(original.x) or sorted(map(tuple,graph.edge_index.t().tolist()))!=sorted(map(tuple,original.edge_index.t().tolist())):
            raise ValueError('Regenerated CFG differs from fixed graph; review extractor/version, do not silently replace')
        if not torch.equal(graph.x.sort(dim=1).values,original.x.sort(dim=1).values):
            raise ValueError('Shuffle changed visible tokens; the control must preserve node token multisets')
        save_checkpoint(path,dict(graph=graph,seed=seed,source_bytecode_hash=digest(source),cache_version='visible-instructions-v2',
                                 rule='shuffle complete visible instructions with immediates; clipped boundary pinned; per-node token multiset unchanged'))
        q.put({'status':'SUCCESS'})
    except Exception:q.put({'status':'FAILED','error':traceback.format_exc()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--kind',choices=['shuffle','window_mean'],required=True)
    p.add_argument('--task');p.add_argument('--seed',type=int,default=17);p.add_argument('--limit',type=int);p.add_argument('--timeout',type=int,default=60)
    p.add_argument('--device',default='cpu');p.add_argument('--source-check',help='Passed consistency.json from revision.source_controls')
    p.add_argument('--include-test',action='store_true',help='Offline feature preparation only; no test predictions or tuning')
    p.add_argument('--sample-id',help='Optional one-sample diagnostic; requires --task')
    a=p.parse_args()
    c=load_config(a.config)
    if a.task and a.task not in c['tasks']:p.error('Unknown task')
    rows=[r for r in read_manifest(c) if (not a.task or r['task']==a.task) and (a.include_test or r['split']!='test')];rows=rows[:a.limit] if a.limit else rows
    if a.sample_id:
        if not a.task:p.error('--sample-id requires --task')
        rows=[r for r in rows if r['sample_id']==a.sample_id]
    if not rows:p.error('No matching samples')
    root=Path(c['control_root'])/a.kind;results=[]
    if a.kind=='window_mean':
        from transformers import RobertaTokenizer,RobertaModel
        from revision.features import source_vector,encoder_identity
        import json
        if not a.source_check:p.error('Window control requires --source-check consistency.json')
        checked=json.loads(Path(a.source_check).read_text(encoding='utf-8'))
        identity=encoder_identity(c['source_encoder_local'])
        if not checked.get('all_pass') or checked['encoder_identity']!=identity or checked['dataset_manifest_sha256']!=digest(Path(c['report_root'])/'final_manifest.csv'):
            raise ValueError('Source consistency proof absent, changed or for another dataset')
        tok=RobertaTokenizer.from_pretrained(c['source_encoder_local'],local_files_only=True)
        encoder=RobertaModel.from_pretrained(c['source_encoder_local'],local_files_only=True).to(a.device);encoder.eval();encoder.requires_grad_(False);torch.set_num_threads(c['training']['threads'])
        root.mkdir(parents=True,exist_ok=True)
        identity_file=root/'encoder_identity.json'
        if identity_file.exists() and json.loads(identity_file.read_text(encoding='utf-8'))!=identity:raise ValueError('Existing cache encoder differs')
        if not identity_file.exists():dump(identity_file,identity)
    for r in rows:
        path=root/(str(a.seed) if a.kind=='shuffle' else '')/r['task']/(r['sample_id']+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('Refuse to silently reuse unverified cache: '+str(path))
        start=time.perf_counter()
        if a.kind=='shuffle':
            ctx=mp.get_context('spawn');q=ctx.Queue();proc=ctx.Process(target=shuffle_worker,args=(c,r,a.seed,path,q));proc.start()
            try:result=q.get(timeout=a.timeout)
            except queue.Empty:result={'status':'TIMEOUT' if proc.is_alive() else 'CRASH'}
            finally:
                proc.join(1)
                if proc.is_alive():proc.terminate();proc.join()
                q.close()
        else:
            try:
                source=Path(c['data_root'])/r['task']/'sourcecode'/(r['sample_id']+'.sol')
                vector=source_vector(source.read_text(encoding='utf-8-sig'),tok,encoder,'window_mean')
                save_checkpoint(path,dict(vector=vector.detach().clone().contiguous(),source_hash=digest(source),encoder=c['source_encoder'],encoder_identity=identity,cache_version='source-window-v2',strategy='mean_of_510_token_window_CLS'))
                result={'status':'SUCCESS'}
            except Exception:result={'status':'FAILED','error':traceback.format_exc()}
        results.append(dict(task=r['task'],sample_id=r['sample_id'],elapsed_seconds=time.perf_counter()-start,**result))
        dump(root/f'preparation_seed{a.seed}.json',results)
    print(f'{sum(r["status"]=="SUCCESS" for r in results)}/{len(results)} prepared')

if __name__=='__main__':main()
