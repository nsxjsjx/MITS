"""Validation-only pooling comparison under retained task-specific configurations."""
import argparse, copy, csv, gc, json, os, shutil, statistics, tarfile, time, traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

from common import *
from model_factory import natural_parent, state_hash
from pooling_factory import factory as pooling_factory, self_check as pooling_self_check


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
    return metrics([int(r['label']) for r in rows],
                   [float(r['probability']) for r in rows], .5)['f1']


def teacher_path(task, seed, taskwise_teacher_root, delegatecall_teacher_root):
    if task in ('reentrancy', 'timestamp', 'integeroverflow'):
        return taskwise_teacher_root / task / str(seed) / 'teacher' / 'best.pt'
    if task == 'delegatecall':
        return delegatecall_teacher_root / task / str(seed) / 'best.pt'
    raise ValueError(task)


def expected_final(task):
    return {
        'reentrancy': ('conv', 5, ['graph', 'sequence', 'derived']),
        'timestamp': ('bigru', None, ['graph', 'sequence', 'derived']),
        'integeroverflow': ('conv', 5, ['graph', 'sequence', 'derived']),
        'delegatecall': ('conv', 4, ['graph', 'sequence']),
    }[task]


def summarize(output, records, protocol):
    rows = []
    lines = ['VALIDATION ONLY; TEST was not loaded.',
             'Task / selected encoder / Mean / Max / Median / Sum / best validation variant']
    labels = {'selected_encoder': 'selected_encoder', 'mean': 'Mean', 'max': 'Max',
              'median': 'Median', 'sum': 'Sum'}
    for task in protocol['tasks']:
        cells, means = [], {}
        for variant in protocol['variants']:
            chosen = [r for r in records if r['task'] == task and r['variant'] == variant]
            if len(chosen) != 5:
                cells.append('INCOMPLETE')
                continue
            values = [100 * r['validation_f1'] for r in chosen]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            means[variant] = mean
            rows.append(dict(task=task, variant=variant, n_seeds=5,
                             validation_f1_mean=mean, validation_f1_sample_sd=sd))
            cells.append(f'{mean:.2f} +/- {sd:.2f}')
        best = max(means, key=lambda v: means[v]) if len(means) == 5 else 'INCOMPLETE'
        lines.append(' / '.join([task] + cells + [labels.get(best, best)]))
    if rows:
        csvsave(output / 'pooling_validation_summary.csv', rows)
    (output / 'readable_pooling_summary.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='.')
    parser.add_argument('--node-root', default='artifacts/node_selection')
    parser.add_argument('--branch-root', default='artifacts/branch_selection')
    parser.add_argument('--gate-root', default='artifacts/gate_selection')
    parser.add_argument('--overflow-root', default='artifacts/final_checkpoints')
    parser.add_argument('--taskwise-teacher-root', default='artifacts/final_checkpoints/checkpoints')
    parser.add_argument('--delegatecall-teacher-root', default='artifacts/kernel_reference/runs/kernel4/main/teacher')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()

    protocol, pairs = setup(args.project)
    forbid_test()
    import torch
    from revision.model import MITS
    import student_training as lowmem_trainer
    import normal_student_training as normal_trainer

    output = Path(args.output).resolve()
    node_root, branch_root, gate_root, overflow_root = tuple(Path(x).resolve() for x in
        (args.node_root, args.branch_root, args.gate_root, args.overflow_root))
    taskwise_teacher_root = Path(args.taskwise_teacher_root).resolve()
    delegatecall_teacher_root = Path(args.delegatecall_teacher_root).resolve()
    if output.exists():
        if not args.resume:
            raise ValueError('Output exists; use --resume')
        if json.loads((output / 'protocol.json').read_text()) != protocol:
            raise ValueError('Resume protocol mismatch')
    else:
        output.mkdir(parents=True)
        save(output / 'protocol.json', protocol)
    for name in ('run_final_pooling_validation.py', 'pooling_factory.py', 'common.py',
                 'model_factory.py', 'gate_factory.py', 'student_training.py',
                 'normal_student_training.py', 'lowmem.py', 'adapter.py', 'protocol.json'):
        shutil.copy2(BUNDLE / name, output / name)

    records_path = output / 'pooling_validation_records.csv'
    records = list(csv.DictReader(records_path.open(encoding='utf-8-sig'))) if records_path.exists() else []
    for row in records:
        row['seed'] = int(row['seed'])
        row['validation_f1'] = float(row['validation_f1'])
    started = time.time()

    def status(state, **extra):
        save(output / 'status.json', dict(
            status=state,
            completed_new_training_runs=sum(r['origin'] == 'NEW_TRAINING' for r in records),
            total_new_training_runs=protocol['new_training_runs'],
            recorded_results=len(records), expected_records=protocol['expected_records'],
            no_test_evaluation=True, elapsed_seconds=time.time() - started, **extra))

    try:
        status('PREFLIGHT')
        pooling_self_check()
        work_map = {(w['task'], w['seed']): w for w in pairs}
        for task in protocol['tasks']:
            for seed in protocol['seeds']:
                work = work_map[(task, seed)]
                folder = selected_folder(task, seed, work, node_root, branch_root, gate_root, overflow_root)
                if not (folder / 'validation_predictions.csv').is_file():
                    raise FileNotFoundError('Retained validation predictions missing: ' + str(folder))
                checkpoint = torch.load(folder / 'best.pt', map_location='cpu')
                if checkpoint.get('task') != task or checkpoint.get('seed') != seed:
                    raise ValueError('Retained checkpoint attribution mismatch: ' + str(folder))
                if task == 'integeroverflow' and sha(folder / 'best.pt') != OVERFLOW_RETAINED_SHA256[seed]:
                    raise ValueError('Retained normal-batch Overflow checkpoint hash mismatch: ' + str(folder))
                kind, kernel, branches = expected_final(task)
                model = checkpoint['config']['model']
                if model['node_encoder'] != kind or model['branches'] != branches:
                    raise ValueError('Final task configuration changed: ' + task)
                if kernel is not None and int(model['kernel_size']) != kernel:
                    raise ValueError('Final kernel changed: ' + task)
                teacher = teacher_path(task, seed, taskwise_teacher_root, delegatecall_teacher_root)
                if not teacher.is_file() or sha(teacher) != work['teacher_sha256']:
                    raise ValueError('Frozen teacher missing or hash mismatch: ' + str(teacher))
                if seed == 17:
                    Parent = natural_parent(task, model, seed, MITS)
                    Candidate = pooling_factory(Parent, 'mean')
                    torch.manual_seed(seed); first = Candidate(model, 'student', checkpoint['config']['teacher_type'])
                    torch.manual_seed(seed); second = Candidate(model, 'student', checkpoint['config']['teacher_type'])
                    if state_hash(first) != state_hash(second):
                        raise AssertionError('Pooling initialization is not reproducible')
                    torch.manual_seed(seed); teacher_a = Parent(model, 'teacher', checkpoint['config']['teacher_type'])
                    torch.manual_seed(seed); teacher_b = Candidate(model, 'teacher', checkpoint['config']['teacher_type'])
                    if state_hash(teacher_a) != state_hash(teacher_b):
                        raise AssertionError('Pooling wrapper changed teacher construction')
        print('PREFLIGHT_PASS: 4 final task configurations; 5 variants; 5 seeds; '
              'pooling math and frozen teachers verified; TEST forbidden', flush=True)
        status('PREFLIGHT_PASS')
        if args.preflight_only:
            archive(output)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')

        for task in protocol['tasks']:
            for seed in protocol['seeds']:
                work = work_map[(task, seed)]
                full_folder = selected_folder(task, seed, work, node_root, branch_root, gate_root, overflow_root)
                full_checkpoint = torch.load(full_folder / 'best.pt', map_location='cpu')
                full_config = full_checkpoint['config']
                for variant in protocol['variants']:
                    key = (task, seed, variant)
                    if any((r['task'], r['seed'], r['variant']) == key for r in records):
                        continue
                    if variant == 'selected_encoder':
                        record = dict(task=task, seed=seed, variant=variant,
                            validation_f1=validation_f1(full_folder / 'validation_predictions.csv'),
                            checkpoint_path=str(full_folder / 'best.pt'),
                            checkpoint_sha256=sha(full_folder / 'best.pt'),
                            origin='REUSED_RETAINED_CONFIGURATION')
                    else:
                        config = copy.deepcopy(full_config)
                        config['variant'] = 'final_pooling_' + variant
                        config['output_root'] = str(output / 'runs')
                        teacher = teacher_path(task, seed, taskwise_teacher_root, delegatecall_teacher_root)
                        config['teacher_checkpoint'] = str(teacher)
                        config['pooling_validation'] = dict(
                            protocol_id=protocol['id'], pooling=variant,
                            retained_encoder=full_config['model']['node_encoder'],
                            retained_kernel=full_config['model'].get('kernel_size'),
                            validation_only=True, test_forbidden=True,
                            semantics='valid opcode embeddings after ReLU and node dropout; PAD excluded')
                        teacher_checkpoint = torch.load(teacher, map_location='cpu')
                        config['node_student_experiment'] = dict(
                            frozen_teacher_model=teacher_checkpoint['config']['model'])
                        # Compatibility metadata required by the retained
                        # normal-batch Overflow trainer. This does not turn the
                        # pooling comparison into a kernel search; it supplies
                        # the same audited frozen-teacher description under the
                        # field name validated by that trainer.
                        config['kernel_validation'] = dict(
                            frozen_teacher_model=teacher_checkpoint['config']['model'],
                            pooling_control=variant,
                            validation_only=True,
                            test_forbidden=True)
                        target = output / 'runs' / config['variant'] / 'student' / task / str(seed)
                        Parent = natural_parent(task, full_config['model'], seed, MITS)
                        Candidate = pooling_factory(Parent, variant)
                        active = normal_trainer if task == 'integeroverflow' else lowmem_trainer
                        active.MITS = Candidate
                        if not target.exists():
                            status('RUNNING', current_task=task, current_seed=seed,
                                   current_variant=variant)
                            print(f'POOLING_TRAINING_START {task} {variant} seed={seed}', flush=True)
                            active.run(config, task, seed, 'student', False)
                        elif json.loads((target / 'status.json').read_text()).get('status') != 'COMPLETED':
                            raise RuntimeError('Incomplete run preserved: ' + str(target))
                        record = dict(task=task, seed=seed, variant=variant,
                            validation_f1=validation_f1(target / 'validation_predictions.csv'),
                            checkpoint_path=str(target / 'best.pt'),
                            checkpoint_sha256=sha(target / 'best.pt'), origin='NEW_TRAINING')
                    records.append(record)
                    csvsave(records_path, records)
                    summarize(output, records, protocol)
                    status('RUNNING')
                    print(f"POOLING_PROGRESS {len(records)}/100 {task} {variant} seed={seed} "
                          f"valF1={100 * record['validation_f1']:.2f}", flush=True)
                    gc.collect(); torch.cuda.empty_cache()
        summarize(output, records, protocol)
        status('COMPLETE')
        print((output / 'readable_pooling_summary.txt').read_text(), flush=True)
    except BaseException:
        status('FAILED_OR_INTERRUPTED', error=traceback.format_exc())
        raise
    finally:
        archive(output)


if __name__ == '__main__':
    main()
