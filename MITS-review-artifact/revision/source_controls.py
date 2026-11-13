"""Train-only encoder consistency check and source-length report, without training."""
import argparse,csv,json
from pathlib import Path
import torch
from revision.config import load_config,dump,digest
from revision.data import read_manifest
from revision.features import source_vector,encoder_identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--atol',type=float,default=1e-4);p.add_argument('--rtol',type=float,default=1e-4)
    a=p.parse_args();c=load_config(a.config);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    dump(out/'status.json',{'status':'RUNNING'})
    try:
        from transformers import RobertaTokenizer,RobertaModel
        tok=RobertaTokenizer.from_pretrained(c['source_encoder_local'],local_files_only=True)
        identity=encoder_identity(c['source_encoder_local']);rows=read_manifest(c);lengths=[]
        for r in rows:
            # No test feature/source reads during this preparatory analysis.
            if r['split']=='test':continue
            path=Path(c['data_root'])/r['task']/'sourcecode'/(r['sample_id']+'.sol')
            n=len(tok.encode(path.read_text(encoding='utf-8-sig'),add_special_tokens=False,truncation=False))
            lengths.append(dict(task=r['task'],sample_id=r['sample_id'],split=r['split'],origin=r['origin'],content_tokens=n,total_tokens=n+2,truncated=n>510))
        with (out/'source_lengths_train_validation.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(lengths[0]));w.writeheader();w.writerows(lengths)
        encoder=RobertaModel.from_pretrained(c['source_encoder_local'],local_files_only=True).to(a.device);encoder.eval();encoder.requires_grad_(False);torch.set_num_threads(c['training']['threads'])
        # Prespecified shortest+longest in each task/origin training stratum; no outcome-based choice.
        picked=[]
        for task in c['tasks']:
            for origin in ['old','new']:
                pool=sorted([r for r in lengths if r['task']==task and r['origin']==origin and r['split']=='train'],key=lambda r:(r['content_tokens'],r['sample_id']))
                if pool:
                    picked.append(pool[0])
                    if pool[-1]['sample_id']!=pool[0]['sample_id']:picked.append(pool[-1])
        checks=[]
        for r in picked:
            d=Path(c['data_root'])/r['task'];sid=r['sample_id'];code=(d/'sourcecode'/(sid+'.sol')).read_text(encoding='utf-8-sig')
            baseline=source_vector(code,tok,encoder,'truncate');saved=torch.load(d/'source_vectors'/(sid+'.pt'),map_location='cpu').reshape(-1)
            matches=baseline.shape==saved.shape and torch.allclose(baseline,saved,atol=a.atol,rtol=a.rtol)
            window=source_vector(code,tok,encoder,'window_mean') if r['content_tokens']<=510 else None
            checks.append(dict(**r,max_abs_difference=float((baseline-saved).abs().max()),baseline_matches=bool(matches),short_window_equals_truncate=bool(torch.equal(window,baseline)) if window is not None else None))
        passed=bool(checks) and all(r['baseline_matches'] and r['short_window_equals_truncate'] is not False for r in checks)
        dump(out/'consistency.json',dict(encoder_identity=identity,dataset_manifest_sha256=digest(Path(c['report_root'])/'final_manifest.csv'),atol=a.atol,rtol=a.rtol,checks=checks,all_pass=passed,scope='Stratified representative inputs, not every source vector'))
        if not passed:raise ValueError('Encoder baseline mismatch; do not run window control until resolved')
        dump(out/'status.json',{'status':'PASS','samples_checked':len(checks),'test_sources_read':False})
        print('Source baseline consistency PASS:',out)
    except Exception:
        import traceback
        dump(out/'status.json',{'status':'FAILED','error':traceback.format_exc()});raise

if __name__=='__main__':main()
