"""Raw creation bytecode -> CFG/tokens -> inference, with bounded workers."""
import argparse,csv,json,multiprocessing as mp,queue,time,traceback,logging,platform
from pathlib import Path
import numpy as np
import torch
from revision.config import dump,digest,data_identity
from revision.data import read_manifest
from revision.model import MITS,parameter_report
from revision.baselines import command_adapter

def worker(checkpoint,device,requests,responses,warmup):
    try:
        start=time.perf_counter();ck=torch.load(checkpoint,map_location='cpu');c=ck['config'];torch.set_num_threads(c['training']['threads'])
        model=MITS(c['model'],'student',c['teacher_type']);model.load_state_dict(ck['model']);model.to(device);model.eval()
        from torch_geometric.data import Data,Batch
        from revision.features import extract
        if device.startswith('cuda'):torch.cuda.synchronize(device)
        loaded=time.perf_counter();peak=0
        if warmup:
            g=Data(x=torch.tensor([[1,2]+[0]*(c['node_max_length']-2)]),edge_index=torch.empty((2,0),dtype=torch.long))
            tokens=torch.tensor([[1,2]+[0]*(c['sequence_max_length']-2)],device=device)
            with torch.no_grad():model(Batch.from_data_list([g]).to(device),tokens)
            if device.startswith('cuda'):torch.cuda.synchronize(device)
        responses.put({'status':'READY','load_seconds':loaded-start,'warmup_seconds':time.perf_counter()-loaded,'parameters':parameter_report(model),'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters())})
        while True:
            job=requests.get()
            if job is None:return
            begin=time.perf_counter()
            stage='extraction';warnings=[]
            class WarningCapture(logging.Handler):
                def emit(self,record):
                    if record.levelno>=logging.WARNING:warnings.append(record.getMessage())
            capture=WarningCapture();logging.getLogger().addHandler(capture)
            try:
                if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
                graph,tokens=extract(job['bytecode'],c);extracted=time.perf_counter()
                stage='inference';graph=Batch.from_data_list([graph]).to(device);tokens=tokens.unsqueeze(0).to(device)
                with torch.no_grad():prob=float(model(graph,tokens)['logits'].sigmoid().item())
                if device.startswith('cuda'):torch.cuda.synchronize(device)
                responses.put({'status':'SUCCESS','sample_id':job['sample_id'],'probability':prob,'prediction':int(prob>=c['training']['threshold']),
                    'extraction_seconds':extracted-begin,'transfer_and_inference_seconds':time.perf_counter()-extracted,'pipeline_seconds':time.perf_counter()-begin,
                    'extraction_warning_count':len(warnings),'extraction_warnings':warnings[:100],
                    'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None})
            except Exception:responses.put({'status':'EXTRACTION_FAILED' if stage=='extraction' else 'INFERENCE_FAILED','sample_id':job['sample_id'],'error':traceback.format_exc(),'pipeline_seconds':time.perf_counter()-begin,'extraction_warning_count':len(warnings),'extraction_warnings':warnings[:100]})
            finally:logging.getLogger().removeHandler(capture)
    except Exception:responses.put({'status':'STARTUP_FAILED','error':traceback.format_exc()})

class Session:
    def __init__(self,path,device,timeout,warmup):
        ctx=mp.get_context('spawn');self.req=ctx.Queue();self.res=ctx.Queue();self.proc=ctx.Process(target=worker,args=(str(path),device,self.req,self.res,warmup));self.timeout=timeout
        begin=time.perf_counter();self.proc.start()
        try:self.startup=self.res.get(timeout=timeout)
        except queue.Empty:self.close();raise TimeoutError('Model process startup timed out')
        self.startup['process_startup_wall_seconds']=time.perf_counter()-begin
        if self.startup['status']!='READY':self.close();raise RuntimeError(str(self.startup))
    def run(self,job):
        begin=time.perf_counter();self.req.put(job)
        try:r=self.res.get(timeout=self.timeout)
        except queue.Empty:
            r={'status':'TIMEOUT' if self.proc.is_alive() else 'CRASH','sample_id':job['sample_id']}
            self.close()
        r['request_wall_seconds']=time.perf_counter()-begin
        return r
    def close(self):
        if self.proc.is_alive():self.proc.terminate()
        self.proc.join(5);self.req.close();self.res.close()

def latency_summary(rows):
    ok=[r for r in rows if r['status']=='SUCCESS'];values=[r['total_wall_seconds'] for r in ok];alltime=[r['total_wall_seconds'] for r in rows]
    return dict(total=len(rows),success=len(ok),coverage=len(ok)/len(rows) if rows else None,status_counts=dict(__import__('collections').Counter(r['status'] for r in rows)),contracts_with_extraction_warnings=sum(r.get('extraction_warning_count',0)>0 for r in rows),
        successful_mean_seconds=float(np.mean(values)) if values else None,successful_p95_seconds=float(np.percentile(values,95)) if values else None,
        all_attempts_mean_seconds=float(np.mean(alltime)) if alltime else None,all_attempts_p95_seconds=float(np.percentile(alltime,95)) if alltime else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['train','validation','test'],default='validation');p.add_argument('--protocol-frozen',action='store_true')
    p.add_argument('--allow-stage13',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--limit',type=int);p.add_argument('--mode',choices=['warm','cold','both'],default='both')
    p.add_argument('--baseline-spec',help='JSON list: name/version/supported_tasks/argv template; real wrappers required')
    a=p.parse_args()
    if a.split=='test' and (not a.protocol_frozen or a.allow_stage13):p.error('Test requires frozen formal checkpoint/protocol')
    path=Path(a.checkpoint);ck=torch.load(path,map_location='cpu');c=ck['config']
    if ck['role']!='student' or ck.get('smoke_not_paper'):p.error('Student nonsmoke checkpoint required')
    if ck.get('stage13_only') and not a.allow_stage13:p.error('Stage13 requires explicit diagnostic flag')
    if ck['dataset_hash']!=digest(Path(c['report_root'])/'final_manifest.csv') or ck['data_hashes']!=data_identity(c):p.error('Checkpoint data identity changed')
    rows=sorted([r for r in read_manifest(c) if r['task']==ck['task'] and r['split']==a.split],key=lambda r:r['sample_id'])
    if a.limit is not None:
        if a.limit<1:p.error('Positive limit required')
        rows=rows[:a.limit]
    if not rows:p.error('No samples')
    specs=json.loads(Path(a.baseline_spec).read_text(encoding='utf-8-sig')) if a.baseline_spec else []
    for spec in specs:
        if not all(k in spec for k in ['name','version','supported_tasks','argv']):p.error('Incomplete real baseline specification')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[];startups=[];session=None
    modes=['warm','cold'] if a.mode=='both' else [a.mode]
    dump(out/'protocol.json',dict(checkpoint_sha256=digest(path),task=ck['task'],split=a.split,ids=[r['sample_id'] for r in rows],device=a.device,mode=a.mode,timeout=a.timeout,
        stage13_only=a.allow_stage13,checkpoint_size_bytes=path.stat().st_size,baseline_specs=specs,python=platform.python_version(),torch=torch.__version__,platform=platform.platform(),threads=c['training']['threads'],
        timing_scope='Both include raw-bytecode disk read, extraction, transfer/inference and IPC. Cold additionally includes process/import/model load. Warm model load and synthetic warmup reported separately. External commands include their own startup. No cached CFG/token input.',
        caution='Validation default; test timing/predictions must not guide tuning. Compare matched hardware and coverage, distinguish cold from warm. Labels never supplied to tools.'))
    dump(out/'status.json',{'status':'RUNNING'})
    try:
        for row in rows:
            d=Path(c['data_root'])/row['task'];sid=row['sample_id'];bcfile=d/'bytecode_pure'/(sid+'.sol');srcfile=d/'sourcecode'/(sid+'.sol')
            for mode in modes:
                begin=time.perf_counter()
                try:
                    raw=bcfile.read_text();job={'sample_id':sid,'bytecode':raw}
                    if mode=='cold' or session is None:
                        active=Session(path,a.device,a.timeout,mode=='warm');startups.append(dict(sample_id=sid,mode=mode,**active.startup))
                        if mode=='warm':session=active
                    else:active=session
                    result=active.run(job)
                    if mode=='cold':active.close()
                    elif result['status'] in ['TIMEOUT','CRASH']:session=None
                    elapsed=time.perf_counter()-begin
                    # First warm request includes startup in wall time; report steady time separately.
                    startup=startups[-1]['process_startup_wall_seconds'] if startups and startups[-1]['sample_id']==sid and startups[-1]['mode']==mode else 0
                    result.update(method='MITS',mode=mode,total_wall_seconds=elapsed-startup if mode=='warm' else elapsed,startup_included_seconds=startup if mode=='cold' else 0)
                except Exception:result={'sample_id':sid,'method':'MITS','mode':mode,'status':'FAILED','error':traceback.format_exc(),'total_wall_seconds':time.perf_counter()-begin}
                records.append(result);dump(out/'records.json',records)
            for spec in specs:
                begin=time.perf_counter()
                if not spec['argv']:r={'status':'MISSING_MATERIALS','sample_id':sid,'error_reason':'No real wrapper configured'}
                elif row['task'] not in spec['supported_tasks']:r={'status':'UNSUPPORTED','sample_id':sid}
                else:
                    rawsource=Path(c.get('baseline_source_root',str(Path(c['data_root']).parents[1]/'baseline_inputs'/c['data_version'])))/row['task']/'sourcecode'/(sid+'.sol')
                    fields={'sample_id':sid,'task':row['task'],'bytecode_file':str(bcfile),'source_file':str(srcfile),'original_source_file':str(rawsource),'target_contract':row['target_contract'],'compiler_version':row.get('compiler_version','')}
                    r=command_adapter(sid,[arg.format(**fields) for arg in spec['argv']],a.timeout)
                records.append(dict(r,method=spec['name'],version=spec['version'],mode='external_command',total_wall_seconds=time.perf_counter()-begin));dump(out/'records.json',records)
        summary={}
        for method,mode in sorted({(r['method'],r['mode']) for r in records}):summary[method+'/'+mode]=latency_summary([r for r in records if r['method']==method and r['mode']==mode])
        dump(out/'summary.json',summary);dump(out/'startups.json',startups)
        with (out/'latency.csv').open('w',encoding='utf-8-sig',newline='') as f:
            keys=['sample_id','method','mode','status','total_wall_seconds','extraction_seconds','transfer_and_inference_seconds','peak_cuda_bytes'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(records)
        dump(out/'status.json',{'status':'COMPLETED','stage13_only':a.allow_stage13,'samples_attempted':len(rows)})
        print('Efficiency results saved:',out)
    except Exception:dump(out/'status.json',{'status':'FAILED','error':traceback.format_exc()});raise
    finally:
        if session is not None:session.close()

if __name__=='__main__':main()
