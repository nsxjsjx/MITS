"""Validation-only kernel sensitivity under retained task-specific configurations."""
import argparse, copy, csv, gc, json, os, shutil, statistics, tarfile, time, traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

from common import *
from model_factory import natural_parent, state_hash

OVERFLOW_RETAINED_SHA256 = {
    17: '886f91e8edc13aa5b82e37609a23d8b8fe856c10aca2b182f104c7c380a69f41',
    29: '0293e6cd82f32ae5e4130750b330a04b7033e36f3d2a915d97b6b313acdb67d3',
    43: '424027927d99af35632ae86c99e84dd88277f0fbcb7d88a4c78431a813a61bdd',
    71: '8a0c87230c23900e23ce1b480c277ef464089233dbbb10410a62ee85fc2ed608',
    101: '8e37d963a2a16cfbae2fb808815aa13877c279fbbc44683794762784547dd8b7',
}


def archive(output):
    destination = Path(str(output) + '_reports.tar.gz')
    with tarfile.open(destination, 'w:gz') as tar:
        for file in output.rglob('*'):
            if file.is_file() and file.suffix in ('.json', '.csv', '.txt', '.py'):
                tar.add(file, arcname=output.name + '/' + str(file.relative_to(output)))
    print('REPORT_ARCHIVE:', destination, flush=True)


def validation_f1(path):
    from revision.metrics import metrics
    rows = list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
    return metrics([int(r['label']) for r in rows], [float(r['probability']) for r in rows], .5)['f1']


def teacher_path(task, seed, taskwise_teacher_root, delegatecall_teacher_root):
    if task in ('reentrancy', 'integeroverflow'):
        return taskwise_teacher_root / task / str(seed) / 'teacher' / 'best.pt'
    if task == 'delegatecall':
        return delegatecall_teacher_root / task / str(seed) / 'best.pt'
    raise ValueError(task)


def summarize(output, records, protocol):
    rows, lines = [], ['VALIDATION ONLY; TEST was not loaded.',
                       'Task / k=2 / k=3 / k=4 / k=5 / best validation kernel']
    for task in protocol['tasks']:
        cells, means = [], {}
        for kernel in protocol['kernels']:
            chosen = [r for r in records if r['task'] == task and r['kernel_size'] == kernel]
            if len(chosen) != 5:
                cells.append('INCOMPLETE')
                continue
            values = [100*r['validation_f1'] for r in chosen]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            means[kernel] = mean
            rows.append(dict(task=task, kernel_size=kernel, n_seeds=5,
                             validation_f1_mean=mean, validation_f1_sample_sd=sd))
            cells.append(f'{mean:.2f} +/- {sd:.2f}')
        if len(means) == 4:
            best = max(means, key=lambda k: (means[k], -k))
            lines.append(' / '.join([task] + cells + [str(best)]))
    lines.append('timestamp / N/A / N/A / N/A / N/A / BiGRU (not a Conv1D kernel task)')
    if rows: csvsave(output/'kernel_validation_summary.csv', rows)
    (output/'readable_kernel_summary.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='.')
    p.add_argument('--node-root', default='artifacts/node_selection')
    p.add_argument('--branch-root', default='artifacts/branch_selection')
    p.add_argument('--gate-root', default='artifacts/gate_selection')
    p.add_argument('--overflow-root', default='artifacts/final_checkpoints')
    p.add_argument('--taskwise-teacher-root', default='artifacts/final_checkpoints/checkpoints')
    p.add_argument('--delegatecall-teacher-root', default='artifacts/kernel_reference/runs/kernel3/main/teacher')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()

    protocol, pairs = setup(args.project)
    forbid_test()
    import torch
    from revision.model import MITS
    import student_training as lowmem_trainer
    import normal_student_training as normal_trainer

    output = Path(args.output).resolve()
    roots = tuple(map(lambda x: Path(x).resolve(),
                      (args.node_root, args.branch_root, args.gate_root, args.overflow_root)))
    node_root, branch_root, gate_root, overflow_root = roots
    taskwise_teacher_root = Path(args.taskwise_teacher_root).resolve()
    delegatecall_teacher_root = Path(args.delegatecall_teacher_root).resolve()
    if output.exists():
        if not args.resume: raise ValueError('Output exists; use --resume')
        if json.loads((output/'protocol.json').read_text()) != protocol:
            raise ValueError('Resume protocol mismatch')
    else:
        output.mkdir(parents=True); save(output/'protocol.json', protocol)
    for name in ('run_final_kernel_validation.py','common.py','model_factory.py','gate_factory.py',
                 'student_training.py','normal_student_training.py','lowmem.py','adapter.py','protocol.json'):
        shutil.copy2(BUNDLE/name, output/name)

    records_path = output/'kernel_validation_records.csv'
    records = list(csv.DictReader(records_path.open(encoding='utf-8-sig'))) if records_path.exists() else []
    for r in records:
        r['seed'], r['kernel_size'] = int(r['seed']), int(r['kernel_size'])
        r['validation_f1'] = float(r['validation_f1'])
    start = time.time()
    def status(state, **extra):
        save(output/'status.json', dict(status=state,
             completed_new_training_runs=sum(r['origin']=='NEW_TRAINING' for r in records),
             total_new_training_runs=protocol['new_training_runs'], recorded_results=len(records),
             expected_records=protocol['expected_records'], no_test_evaluation=True,
             elapsed_seconds=time.time()-start, **extra))

    try:
        status('PREFLIGHT')
        work_map = {(w['task'],w['seed']):w for w in pairs}
        for task in protocol['tasks']:
            for seed in protocol['seeds']:
                work = work_map[(task,seed)]
                folder = selected_folder(task,seed,work,node_root,branch_root,gate_root,overflow_root)
                ckpt = torch.load(folder/'best.pt', map_location='cpu')
                if task == 'integeroverflow' and sha(folder/'best.pt') != OVERFLOW_RETAINED_SHA256[seed]:
                    raise ValueError('Retained normal-batch Overflow checkpoint hash mismatch: '+str(folder/'best.pt'))
                tp = teacher_path(task, seed, taskwise_teacher_root, delegatecall_teacher_root)
                if not tp.is_file() or sha(tp) != work['teacher_sha256']:
                    raise ValueError('Frozen teacher missing or hash mismatch: '+str(tp))
                if ckpt['config']['model']['node_encoder'] != 'conv':
                    raise ValueError('Expected retained Conv1D student: '+task)
                if task == 'reentrancy' and ckpt['config']['model']['branches'] != ['graph','sequence','derived']:
                    raise ValueError('Reentrancy final branches changed')
                if task == 'delegatecall' and ckpt['config']['model']['branches'] != ['graph','sequence']:
                    raise ValueError('Delegatecall final branches changed')
                if seed == 17:
                    cfg = copy.deepcopy(ckpt['config']['model']); cfg['kernel_size'] = 2
                    Parent = natural_parent(task,cfg,seed,MITS)
                    torch.manual_seed(seed); a=Parent(cfg,'student',ckpt['config']['teacher_type'])
                    torch.manual_seed(seed); b=Parent(cfg,'student',ckpt['config']['teacher_type'])
                    if state_hash(a) != state_hash(b): raise AssertionError('Initialization not reproducible')
        print('PREFLIGHT_PASS: 3 retained Conv1D tasks; 4 kernels; 5 seeds; frozen teachers verified; TEST forbidden', flush=True)
        status('PREFLIGHT_PASS')
        if args.preflight_only: archive(output); return
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')

        for task in protocol['tasks']:
            for seed in protocol['seeds']:
                work = work_map[(task,seed)]
                full_folder = selected_folder(task,seed,work,node_root,branch_root,gate_root,overflow_root)
                full_ckpt = torch.load(full_folder/'best.pt', map_location='cpu')
                full_config = full_ckpt['config']; selected_kernel = int(full_config['model']['kernel_size'])
                for kernel in protocol['kernels']:
                    key = (task,seed,kernel)
                    if any((r['task'],r['seed'],r['kernel_size']) == key for r in records): continue
                    if kernel == selected_kernel:
                        records.append(dict(task=task,seed=seed,kernel_size=kernel,
                          validation_f1=validation_f1(full_folder/'validation_predictions.csv'),
                          checkpoint_path=str(full_folder/'best.pt'),checkpoint_sha256=sha(full_folder/'best.pt'),
                          origin='REUSED_RETAINED_CONFIGURATION'))
                    else:
                        config=copy.deepcopy(full_config); config['variant']=f'final_kernel{kernel}'
                        config['output_root']=str(output/'runs'); config['model']=copy.deepcopy(full_config['model'])
                        tp = teacher_path(task, seed, taskwise_teacher_root, delegatecall_teacher_root)
                        config['model']['kernel_size']=kernel; config['teacher_checkpoint']=str(tp)
                        teacher_ckpt=torch.load(tp,map_location='cpu')
                        config['kernel_validation']=dict(protocol_id=protocol['id'],kernel_size=kernel,
                            frozen_teacher_model=teacher_ckpt['config']['model'],validation_only=True,test_forbidden=True)
                        # Low-memory trainer uses the same frozen-teacher description field.
                        config['node_student_experiment']=dict(frozen_teacher_model=teacher_ckpt['config']['model'])
                        target=output/'runs'/config['variant']/'student'/task/str(seed)
                        Parent=natural_parent(task,config['model'],seed,MITS)
                        active=normal_trainer if task=='integeroverflow' else lowmem_trainer
                        active.MITS=Parent
                        if not target.exists():
                            status('RUNNING',current_task=task,current_seed=seed,current_kernel=kernel)
                            print(f'KERNEL_TRAINING_START {task} k={kernel} seed={seed}',flush=True)
                            active.run(config,task,seed,'student',False)
                        elif json.loads((target/'status.json').read_text()).get('status')!='COMPLETED':
                            raise RuntimeError('Incomplete run preserved: '+str(target))
                        records.append(dict(task=task,seed=seed,kernel_size=kernel,
                          validation_f1=validation_f1(target/'validation_predictions.csv'),
                          checkpoint_path=str(target/'best.pt'),checkpoint_sha256=sha(target/'best.pt'),
                          origin='NEW_TRAINING'))
                    csvsave(records_path,records); summarize(output,records,protocol); status('RUNNING')
                    print(f"KERNEL_PROGRESS {len(records)}/60 {task} k={kernel} seed={seed} valF1={100*records[-1]['validation_f1']:.2f}",flush=True)
                    gc.collect(); torch.cuda.empty_cache()
        summarize(output,records,protocol); status('COMPLETE')
        print((output/'readable_kernel_summary.txt').read_text(),flush=True)
    except BaseException:
        status('FAILED_OR_INTERRUPTED',error=traceback.format_exc()); raise
    finally:
        archive(output)

if __name__=='__main__': main()
