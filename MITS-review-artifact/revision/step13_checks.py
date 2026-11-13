import argparse,copy,traceback
from pathlib import Path
import torch
from revision.config import dump,data_identity
from revision.step13_common import local_config,session,forbid_test_reads,read
from revision.experiments import configure
from revision.train import run,save_checkpoint,validate_teacher_checkpoint,limit_pilot_rows
from revision.model import MITS
from revision.distillation import loss
from revision.data import RevisionDataset,collate
from revision.predict_bytecode import predict

def require(condition,message):
    if not condition:raise AssertionError(message)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--threads',type=int,default=4);a=parser.parse_args()
    c=local_config(a.config);c['training'].update(device='cpu',batch_size=1,epochs=1,patience=1,threads=a.threads)
    c['stage13']={'mode':'checks','train_per_class':2,'validation_per_class':2,'small_graphs':True,'audit_updates':True,'max_run_seconds':900}
    out=session('checks',c);c['output_root']=str(out/'runs');seed=17;results={};errors={}
    print('Results:',out,flush=True)
    try:
        with forbid_test_reads(c) as attempts:
            for task in c['tasks']:
                try:
                    cfg=configure(c,'main');teacher=run(cfg,task,seed,'teacher');student=run(cfg,task,seed,'student')
                    audit=read(student/'update_audit.json')
                    require(audit['teacher_unchanged'] is True,'Teacher changed during student training')
                    require(audit['student_or_teacher_updated'],'Student did not update')
                    require(audit['graph_batches_unchanged']>0,'No graph mutation checks executed')
                    checkpoint=torch.load(teacher/'best.pt',map_location='cpu')
                    validate_teacher_checkpoint(checkpoint,cfg,task,seed,data_identity(cfg))
                    for key in ['task','seed','data_version']:
                        bad=copy.copy(checkpoint)
                        if key=='data_version':bad['config']=copy.deepcopy(checkpoint['config']);bad['config']['data_version']='WRONG_VERSION'
                        else:bad[key]='WRONG_TASK' if key=='task' else seed+1
                        rejected=False
                        try:validate_teacher_checkpoint(bad,cfg,task,seed,data_identity(cfg))
                        except ValueError:rejected=True
                        require(rejected,'Checkpoint mismatch accepted: '+key)
                    # Same architecture, weights, labels and dropout RNG: compare loss AND gradients.
                    data=RevisionDataset(cfg,task,'train',need_source=False);limit_pilot_rows(data,1,True);batch=collate([data[0]])
                    first=MITS(cfg['model']);second=copy.deepcopy(first);first.train();second.train()
                    torch.manual_seed(991);one=first(batch['graph'],batch['tokens'])
                    value,_=loss(one,batch['label'],beta=0);value.backward()
                    torch.manual_seed(991);two=second(batch['graph'],batch['tokens'])
                    bce=torch.nn.functional.binary_cross_entropy_with_logits(two['logits'],batch['label']);bce.backward()
                    require(torch.equal(value,bce),'beta=0 loss differs from BCE')
                    require(all(torch.equal(x.grad,y.grad) for x,y in zip(first.parameters(),second.parameters())),'beta=0 gradients differ')
                    for folder in [teacher,student]:
                        for filename in ['config.json','used_ids.json','provenance.json','history.json','best.pt','last.pt','validation_predictions.csv','status.json','update_audit.json']:
                            require((folder/filename).is_file(),'Missing log: '+filename)
                    results[task]={'status':'PASS','batch_size':1,'parameter_audit':audit,'checkpoint_mismatch_tests':'PASS','beta0_loss_and_gradients':'PASS','logs':'PASS'}
                except Exception:errors[task]=traceback.format_exc()
                dump(out/'checks.json',{'tasks':results,'errors':errors,'test_open_attempts':attempts})
            if results:
                task=next(iter(results));checkpoint=torch.load(out/'runs/main/student'/task/str(seed)/'best.pt',map_location='cpu')
                isolated=out/'bytecode_only';isolated.mkdir()
                checkpoint['config']['data_root']=str(isolated/'NO_DATASET');checkpoint['config']['source_encoder_local']=str(isolated/'NO_SOURCE_ENCODER')
                save_checkpoint(isolated/'student.pt',checkpoint)
                prediction=predict(isolated/'student.pt','600160020100',task,timeout=120)
                require(prediction['status']=='SUCCESS',str(prediction));dump(out/'no_source_inference.json',prediction)
            else:raise AssertionError('No task passed training checks')
            require(not attempts,'Test data access attempted')
        dump(out/'summary.json',{'status':'PASS' if not errors and len(results)==4 else 'FAILED','tasks_passed':list(results),'errors':errors,
            'test_data':'not opened; frozen manifest metadata used for overlap checks only','source_free_inference':'PASS',
            'purpose':'CPU code checks only; small graphs and 4 examples/split, not accuracy or server timing'})
    except BaseException:
        dump(out/'summary.json',{'status':'FAILED','errors':errors,'fatal':traceback.format_exc()});raise
    print('Finished. Send the entire folder:',out,flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
