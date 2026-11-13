"""One-batch integration checks. Outputs are explicitly excluded from paper results."""
import argparse,copy,sys
from pathlib import Path
import torch
from revision.config import load_config,dump,ROOT
from revision.experiments import configure,VARIANTS
from revision.model import MITS,parameter_report
from revision.train import run
from revision.baselines import command_adapter
from revision.predict_bytecode import predict

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',default='revision_checks_runs');parser.add_argument('--seed',type=int,default=43);args=parser.parse_args()
    c=load_config();seed=args.seed;task='reentrancy';results={}
    c['output_root']=args.output_root
    teacher=run(configure(c,'main'),task,seed,'teacher',smoke=True)
    student=run(configure(c,'main'),task,seed,'student',smoke=True)
    beta0=run(configure(c,'beta0'),task,seed,'student',smoke=True)
    import json
    a=json.loads((student/'provenance.json').read_text());b=json.loads((beta0/'provenance.json').read_text())
    assert a['initial_parameter_hash']==b['initial_parameter_hash']
    results['same_initialization_main_beta0']='PASS'
    ck=torch.load(student/'best.pt',map_location='cpu');assert ck['smoke_not_paper']
    results['teacher_student_beta0_one_epoch']='PASS (4 train/4 validation only)'
    bc=(Path(c['data_root'])/task/'bytecode_pure/old_3.sol').read_text().strip()
    result=predict(student/'best.pt',bc,task,timeout=60);assert result['status']=='SUCCESS',result
    results['raw_bytecode_inference']=result
    invalid=predict(student/'best.pt',bc,task,input_type='runtime',timeout=60);assert invalid['status']=='FAILED'
    results['runtime_mismatch_rejected']='PASS'
    result=command_adapter('fixture',[sys.executable,'-c','import json;print(json.dumps(dict(status="SUCCESS",prediction=0,score=.2)))'])
    assert result['status']=='SUCCESS';results['baseline_schema_fixture']='PASS; not a real baseline reproduction'
    fail=command_adapter('fixture',['THIS_TOOL_DOES_NOT_EXIST_20261011']);assert fail['status']=='MISSING_MATERIALS'
    results['missing_baseline_not_negative']='PASS'
    registry=[]
    for variant in VARIANTS:
        v=configure(c,variant);model=MITS(v['model']);record=parameter_report(model)
        record.update(variant=variant,teacher_variant=v['teacher_variant'],kd=v['distillation'],node_matching=v.get('node_matching'),source_coverage=v['source_coverage'])
        if v.get('node_matching'):
            assert v['node_matching']['matched'];assert record['modules']['node']==v['node_matching']['actual']
        registry.append(record)
    dump(ROOT/'revision_reports/experiment_registry.json',registry)
    dump(ROOT/'revision_reports/integration_checks.json',results);print(results)

if __name__=='__main__':main()
