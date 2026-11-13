"""Train predeclared final-configuration ablations using validation only."""
import argparse
import copy
import gc
import json
import os
import shutil
import statistics
import tarfile
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

from common import *
from model_factory import matched_ablation_parent, natural_parent, state_hash


def archive(output):
    destination = Path(str(output) + '_reports.tar.gz')
    with tarfile.open(destination, 'w:gz') as tar:
        for file in output.rglob('*'):
            if file.is_file() and file.suffix in ('.json', '.csv', '.txt', '.py'):
                tar.add(file, arcname=output.name + '/' + str(file.relative_to(output)))
    print('REPORT_ARCHIVE:', destination, flush=True)


def validation_f1(path, config):
    import csv
    from revision.metrics import metrics
    rows = list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
    return metrics(
        [int(row['label']) for row in rows],
        [float(row['probability']) for row in rows],
        0.5,
    )['f1']


def summarize(output, records, protocol):
    summary = []
    lines = ['VALIDATION ONLY. TEST was not loaded.',
             'Variant / task / validation F1 mean +/- sample SD / n']
    for variant in protocol['variants']:
        for task in protocol['tasks']:
            chosen = [row for row in records if row['variant'] == variant and row['task'] == task]
            if not chosen:
                if variant == 'no_derived' and task == 'delegatecall':
                    lines.append(f'{variant} / {task} / N/A (derived branch absent from full model)')
                continue
            if len(chosen) != 5:
                continue
            values = [100 * row['validation_f1'] for row in chosen]
            rec = dict(variant=variant, task=task, n_seeds=5,
                       validation_f1_mean=statistics.mean(values),
                       validation_f1_sample_sd=statistics.stdev(values))
            summary.append(rec)
            lines.append(f"{variant} / {task} / {rec['validation_f1_mean']:.2f} +/- "
                         f"{rec['validation_f1_sample_sd']:.2f} / 5")
    if summary:
        csvsave(output / 'validation_summary.csv', summary)
    (output / 'readable_validation_summary.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='.')
    parser.add_argument('--node-root', default='artifacts/node_selection')
    parser.add_argument('--branch-root', default='artifacts/branch_selection')
    parser.add_argument('--gate-root', default='artifacts/gate_selection')
    parser.add_argument('--no-kd-root', default='artifacts/no_kd_checkpoints')
    parser.add_argument('--overflow-root', default='artifacts/final_checkpoints')
    parser.add_argument('--overflow-no-kd-root', default='artifacts/overflow_no_kd')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()

    protocol, pairs = setup(args.project)
    forbid_test()
    import torch
    from revision.model import MITS
    import student_training as trainer

    output = Path(args.output).resolve()
    node_root = Path(args.node_root).resolve()
    branch_root = Path(args.branch_root).resolve()
    gate_root = Path(args.gate_root).resolve()
    no_kd_root = Path(args.no_kd_root).resolve()
    overflow_root = Path(args.overflow_root).resolve()
    overflow_no_kd_root = Path(args.overflow_no_kd_root).resolve()
    if output.exists():
        if not args.resume:
            raise ValueError('Output exists; use --resume')
        if json.loads((output / 'protocol.json').read_text()) != protocol:
            raise ValueError('Resume protocol mismatch')
    else:
        output.mkdir(parents=True)
        save(output / 'protocol.json', protocol)
    for filename in ('run_final_ablation_train.py', 'model_factory.py', 'common.py',
                     'student_training.py', 'lowmem.py', 'gate_factory.py', 'protocol.json'):
        shutil.copy2(BUNDLE / filename, output / filename)

    start = time.time()
    records = []

    def status(state, **extra):
        save(output / 'status.json', dict(
            status=state,
            completed_new_training_runs=sum(row['origin'] == 'NEW_TRAINING' for row in records),
            total_new_training_runs=protocol['new_training_runs'],
            recorded_validation_results=len(records),
            expected_validation_results=95,
            no_test_evaluation=True,
            elapsed_seconds=time.time() - start,
            **extra,
        ))

    try:
        status('PREFLIGHT')
        roots = ((node_root, 60), (branch_root, 20), (gate_root, 20), (no_kd_root, 20))
        for root, expected in roots:
            state = json.loads((root / 'status.json').read_text())
            complete = state.get('new_student_runs_complete', state.get('no_distillation_runs_complete'))
            if state.get('status') != 'COMPLETE' or complete != expected:
                raise ValueError('Required experiment incomplete: ' + str(root))

        # Audit all reused checkpoints and run construction checks before training.
        for work in pairs:
            task, seed = work['task'], work['seed']
            full_folder = selected_folder(task, seed, work, node_root, branch_root, gate_root, overflow_root)
            full_checkpoint = torch.load(full_folder / 'best.pt', map_location='cpu')
            full_config = full_checkpoint['config']
            if task == 'integeroverflow':
                teacher_path = overflow_root / 'checkpoints' / task / str(seed) / 'teacher' / 'best.pt'
                if not teacher_path.is_file():
                    raise FileNotFoundError('Frozen Overflow teacher missing: ' + str(teacher_path))
                if sha(teacher_path) != work['teacher_sha256']:
                    raise ValueError('Frozen Overflow teacher hash mismatch: ' + str(teacher_path))
            if full_checkpoint['task'] != task or full_checkpoint['seed'] != seed:
                raise ValueError('Full checkpoint attribution mismatch')
            if full_config['model']['branches'] != protocol_branch(task):
                raise ValueError('Final full branches changed for ' + task)
            if seed == 17:
                FullParent = natural_parent(task, full_config['model'], seed, MITS)
                torch.manual_seed(seed)
                first = FullParent(full_config['model'], 'student', full_config['teacher_type'])
                torch.manual_seed(seed)
                second = FullParent(full_config['model'], 'student', full_config['teacher_type'])
                if state_hash(first) != state_hash(second):
                    raise AssertionError('Full model initialization is not reproducible')
                for variant in ('no_graph', 'no_direct_sequence', 'no_derived'):
                    retained = removed_branch(task, variant)
                    if retained is None:
                        continue
                    Parent = matched_ablation_parent(task, full_config['model'], retained, seed, MITS)
                    reduced_config = copy.deepcopy(full_config['model'])
                    reduced_config['branches'] = retained
                    torch.manual_seed(seed)
                    candidate = Parent(reduced_config, 'student', full_config['teacher_type'])
                    if candidate.branches != retained:
                        raise AssertionError('Ablation branch construction failed')
        print('PREFLIGHT_PASS: final task configurations; matched initialization; 95 predefined validation records; TEST forbidden', flush=True)
        status('PREFLIGHT_PASS')
        if args.preflight_only:
            archive(output)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')

        for work in pairs:
            task, seed = work['task'], work['seed']
            full_folder = selected_folder(task, seed, work, node_root, branch_root, gate_root, overflow_root)
            full_checkpoint = torch.load(full_folder / 'best.pt', map_location='cpu')
            full_config = full_checkpoint['config']

            # Reuse the frozen complete and no-KD validation records.
            for variant, folder in (
                ('full', full_folder),
                ('no_kd', no_kd_folder(task, seed, no_kd_root, overflow_no_kd_root)),
            ):
                checkpoint = torch.load(folder / 'best.pt', map_location='cpu')
                records.append(dict(
                    variant=variant, task=task, seed=seed,
                    branches='|'.join(checkpoint['config']['model']['branches']),
                    validation_f1=validation_f1(folder / 'validation_predictions.csv', checkpoint['config']),
                    checkpoint_path=str(folder / 'best.pt'),
                    checkpoint_sha256=sha(folder / 'best.pt'),
                    origin='REUSED_FROZEN_REFERENCE',
                ))

            for variant in ('no_graph', 'no_direct_sequence', 'no_derived'):
                retained = removed_branch(task, variant)
                if retained is None:
                    continue
                config = copy.deepcopy(full_config)
                config['variant'] = 'final_' + variant
                config['output_root'] = str(output / 'runs')
                config['model'] = copy.deepcopy(full_config['model'])
                config['model']['branches'] = retained
                # revision.train otherwise rebuilds the teacher path below the
                # new ablation output_root.  Pin the already audited frozen
                # Overflow teacher explicitly instead.
                if task == 'integeroverflow':
                    config['teacher_checkpoint'] = str(
                        overflow_root / 'checkpoints' / task / str(seed) / 'teacher' / 'best.pt'
                    )
                config['final_ablation'] = dict(
                    protocol_id=protocol['id'],
                    complete_checkpoint_sha256=sha(full_folder / 'best.pt'),
                    removed_branch=({'no_graph': 'graph',
                                     'no_direct_sequence': 'sequence',
                                     'no_derived': 'derived'}[variant]),
                    validation_selection_only=True,
                    test_forbidden=True,
                )
                target = output / 'runs' / config['variant'] / 'student' / task / str(seed)
                Parent = matched_ablation_parent(task, full_config['model'], retained, seed, MITS)
                active_trainer = trainer
                if task == 'integeroverflow':
                    # The retained Overflow result is the original normal-batch
                    # checkpoint family. Its ablations must use that same trainer.
                    import revision.train as normal_trainer
                    normal_trainer.MITS = Parent
                    active_trainer = normal_trainer
                else:
                    trainer.MITS = Parent
                if not target.exists():
                    status('RUNNING', current_variant=variant, current_task=task, current_seed=seed)
                    print(f'ABLATION_TRAINING_START {variant} {task} seed={seed}', flush=True)
                    active_trainer.run(config, task, seed, 'student', False)
                elif not args.resume or json.loads((target / 'status.json').read_text()).get('status') != 'COMPLETED':
                    raise RuntimeError('Incomplete run preserved: ' + str(target))
                checkpoint = torch.load(target / 'best.pt', map_location='cpu')
                records.append(dict(
                    variant=variant, task=task, seed=seed,
                    branches='|'.join(retained),
                    validation_f1=validation_f1(target / 'validation_predictions.csv', config),
                    checkpoint_path=str(target / 'best.pt'),
                    checkpoint_sha256=sha(target / 'best.pt'),
                    origin='NEW_TRAINING',
                ))
                csvsave(output / 'validation_records.csv', records)
                summarize(output, records, protocol)
                status('RUNNING')
                print(f"ABLATION_PROGRESS {sum(r['origin']=='NEW_TRAINING' for r in records)}/55 "
                      f"{variant} {task} seed={seed} valF1={100*records[-1]['validation_f1']:.2f}", flush=True)
                del checkpoint
                gc.collect()
                torch.cuda.empty_cache()

        csvsave(output / 'validation_records.csv', records)
        summarize(output, records, protocol)
        status('COMPLETE')
        print((output / 'readable_validation_summary.txt').read_text(), flush=True)
    except BaseException:
        status('FAILED_OR_INTERRUPTED', error=traceback.format_exc())
        raise
    finally:
        archive(output)


def protocol_branch(task):
    return {
        'reentrancy': ['graph', 'sequence', 'derived'],
        'timestamp': ['graph', 'sequence', 'derived'],
        'integeroverflow': ['graph', 'sequence', 'derived'],
        'delegatecall': ['graph', 'sequence'],
    }[task]


if __name__ == '__main__':
    main()

