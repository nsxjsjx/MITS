import argparse,json,time,traceback,multiprocessing as mp
from pathlib import Path
import torch
from torch_geometric.data import Batch
from revision.model import MITS
from revision.config import dump

def worker(checkpoint,bytecode,task,input_type,queue):
    start=time.perf_counter()
    try:
        ck=torch.load(checkpoint,map_location='cpu');c=ck['config'];torch.set_num_threads(c['training']['threads'])
        if ck['role']!='student' or task!=ck['task']:raise ValueError('Task/role mismatch')
        if input_type!=c['bytecode_type']:raise ValueError('Input bytecode domain mismatch')
        from revision.features import extract
        graph,tokens=extract(bytecode,c);extracted=time.perf_counter()
        model=MITS(c['model'],'student');model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():prob=float(model(Batch.from_data_list([graph]),tokens.unsqueeze(0))['logits'].sigmoid().item())
        queue.put(dict(status='SUCCESS',task=task,input_type=input_type,probability=prob,prediction=int(prob>=c['training']['threshold']),
                       extraction_seconds=extracted-start,elapsed_seconds=time.perf_counter()-start))
    except Exception:queue.put(dict(status='FAILED',error_reason=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start))

def predict(checkpoint,bytecode,task,input_type='creation',timeout=60):
    ctx=mp.get_context('spawn');q=ctx.Queue();p=ctx.Process(target=worker,args=(str(checkpoint),bytecode,task,input_type,q));p.start()
    try:
        result=q.get(timeout=timeout);p.join(5)
    except __import__('queue').Empty:result=dict(status='TIMEOUT' if p.is_alive() else 'CRASH',error_reason='Extraction/inference worker did not return',elapsed_seconds=timeout)
    finally:
        if p.is_alive():p.terminate();p.join()
        q.close()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--bytecode-file',required=True);p.add_argument('--task',required=True)
    p.add_argument('--input-type',choices=['creation','runtime'],required=True);p.add_argument('--output',required=True);p.add_argument('--timeout',type=int,default=60);a=p.parse_args()
    result=predict(a.checkpoint,Path(a.bytecode_file).read_text(),a.task,a.input_type,a.timeout);dump(a.output,result);print(json.dumps(result))

if __name__=='__main__':main()
