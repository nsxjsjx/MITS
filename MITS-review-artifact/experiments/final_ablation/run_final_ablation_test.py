"""Freeze all predeclared validation-selected ablations, then test each once."""
import argparse
import copy
import csv
import gc
import json
import os
import statistics
import tarfile
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

from common import *
from model_factory import matched_ablation_parent, natural_parent


def archive(output):
    destination = Path(str(output) + '_reports.tar.gz')
    with tarfile.open(destination, 'w:gz') as tar:
        for file in output.rglob('*'):
            if file.is_file() and file.suffix in ('.json', '.csv', '.txt', '.py'):
                tar.add(file, arcname=output.name + '/' + str(file.relative_to(output)))
    print('REPORT_ARCHIVE:', destination, flush=True)


def save_predictions(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def infer(model, loader, device, threshold):
    import torch
    from revision.metrics import metrics
    labels, probabilities, rows = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch['graph'].to(device), batch['tokens'].to(device), None)
            probs = output['logits'].sigmoid().cpu().tolist()
            ys = batch['label'].tolist()
            for sid, group, label, probability in zip(
                    batch['sample_id'], batch['split_group'], ys, probs):
                rows.append(dict(sample_id=sid, split_group=group, label=int(label),
                                 probability=float(probability),
                                 prediction=int(probability >= threshold)))
            labels.extend(int(label) for label in ys)
            probabilities.extend(float(probability) for probability in probs)
    return metrics(labels, probabilities, threshold), rows


def summarize(output, records, protocol):
    summary = []
    lines = ['FROZEN FINAL ABLATION TEST. No training and no test-based selection.',
             'Variant / task / test F1 mean +/- sample SD / precision / recall / accuracy / FPR']
    for variant in protocol['variants']:
        for task in protocol['tasks']:
            chosen = [row for row in records if row['variant'] == variant and row['task'] == task]
            if not chosen:
                if variant == 'no_derived' and task == 'delegatecall':
                    lines.append(f'{variant} / {task} / N/A (derived branch absent from full model)')
                continue
            if len(chosen) != 5:
                continue
            rec = dict(
                variant=variant, task=task, n_seeds=5,
                f1_mean=statistics.mean(row['f1'] for row in chosen),
                f1_sd=statistics.stdev(row['f1'] for row in chosen),
                precision_mean=statistics.mean(row['precision'] for row in chosen),
                recall_mean=statistics.mean(row['recall'] for row in chosen),
                accuracy_mean=statistics.mean(row['accuracy'] for row in chosen),
                fpr_mean=statistics.mean(row['fpr'] for row in chosen),
                n=chosen[0]['n'],
            )
            summary.append(rec)
            lines.append(
                f"{variant} / {task} / {100*rec['f1_mean']:.2f} +/- {100*rec['f1_sd']:.2f} / "
                f"{100*rec['precision_mean']:.2f} / {100*rec['recall_mean']:.2f} / "
                f"{100*rec['accuracy_mean']:.2f} / {100*rec['fpr_mean']:.2f}"
            )
    if summary:
        csvsave(output / 'test_summary.csv', summary)
    (output / 'readable_test_summary.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='.')
    parser.add_argument('--training-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--protocol-frozen', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.freeze_only == args.protocol_frozen:
        parser.error('Run --freeze-only first; then run --protocol-frozen with --resume')

    protocol, pairs = setup(args.project)
    import torch
    from torch.utils.data import DataLoader
    from revision.model import MITS
    from revision.data import RevisionDataset, collate
    from revision.config import data_identity

    training_root = Path(args.training_root).resolve()
    output = Path(args.output).resolve()
    training_status = json.loads((training_root / 'status.json').read_text())
    if training_status.get('status') != 'COMPLETE' or training_status.get('completed_new_training_runs') != 55:
        raise ValueError('Validation-only ablation training is incomplete')
    if output.exists():
        if not args.resume:
            raise ValueError('Output exists; use --resume')
        if json.loads((output / 'protocol.json').read_text()) != protocol:
            raise ValueError('Frozen protocol changed')
    else:
        output.mkdir(parents=True)
        save(output / 'protocol.json', protocol)

    start = time.time()
    try:
        validation_records = list(csv.DictReader(
            (training_root / 'validation_records.csv').open(encoding='utf-8-sig')))
        if len(validation_records) != 95:
            raise ValueError('Expected 95 predeclared variant/task/seed checkpoints')
        frozen = []
        for row in validation_records:
            checkpoint = Path(row['checkpoint_path'])
            if sha(checkpoint) != row['checkpoint_sha256']:
                raise ValueError('Checkpoint changed: ' + str(checkpoint))
            state = torch.load(checkpoint, map_location='cpu')
            if state['task'] != row['task'] or state['seed'] != int(row['seed']) or state['role'] != 'student':
                raise ValueError('Checkpoint attribution mismatch')
            if state.get('smoke_not_paper') or state.get('stage13_only'):
                raise ValueError('Non-paper checkpoint selected')
            if state.get('data_hashes') != data_identity(state['config']):
                raise ValueError('Dataset identity changed')
            frozen.append(dict(
                variant=row['variant'], task=row['task'], seed=int(row['seed']),
                branches=row['branches'], validation_f1=float(row['validation_f1']),
                checkpoint_path=str(checkpoint), checkpoint_sha256=row['checkpoint_sha256'],
                frozen_at_unix=start,
            ))
        frozen_path = output / 'frozen_checkpoints.json'
        if frozen_path.exists():
            previous = json.loads(frozen_path.read_text())
            stable = [{k: v for k, v in row.items() if k != 'frozen_at_unix'} for row in frozen]
            old = [{k: v for k, v in row.items() if k != 'frozen_at_unix'} for row in previous]
            if stable != old:
                raise ValueError('Frozen checkpoint set changed')
            frozen = previous
        else:
            save(frozen_path, frozen)
        csvsave(output / 'frozen_checkpoints.csv', frozen)
        save(output / 'status.json', dict(
            status='FROZEN_NO_TEST_LOADED' if args.freeze_only else 'RUNNING_TEST',
            frozen_checkpoints=len(frozen), completed_test_evaluations=0,
            training_performed=False, test_based_selection=False,
            elapsed_seconds=time.time() - start,
        ))
        print('FREEZE_PASS: 95 validation-selected checkpoint hashes fixed; TEST not loaded', flush=True)
        if args.freeze_only:
            archive(output)
            return
        if not args.resume:
            raise ValueError('Use --resume for the explicit test stage')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        records = []
        for index, item in enumerate(frozen, 1):
            checkpoint = Path(item['checkpoint_path'])
            if sha(checkpoint) != item['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint bytes changed')
            state = torch.load(checkpoint, map_location='cpu')
            config = state['config']
            task, seed, variant = item['task'], item['seed'], item['variant']
            full_item = next(row for row in frozen if row['variant'] == 'full' and
                             row['task'] == task and row['seed'] == seed)
            full_state = torch.load(full_item['checkpoint_path'], map_location='cpu')
            full_model_config = full_state['config']['model']
            if variant in ('full', 'no_kd'):
                Parent = natural_parent(task, full_model_config, seed, MITS)
            else:
                Parent = matched_ablation_parent(task, full_model_config,
                                                  config['model']['branches'], seed, MITS)
            model = Parent(config['model'], 'student', config['teacher_type'])
            model.load_state_dict(state['model'], strict=True)
            if 'source' in model.branches:
                raise ValueError('Student unexpectedly requires source')
            model.to(device)
            dataset = RevisionDataset(config, task, 'test', need_source=False)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
            metric, predictions = infer(model, loader, device, 0.5)
            run_output = output / 'test' / variant / task / str(seed)
            run_output.mkdir(parents=True, exist_ok=False)
            save_predictions(run_output / 'predictions.csv', predictions)
            save(run_output / 'metrics.json', metric)
            save(run_output / 'provenance.json', dict(
                protocol_id=protocol['id'], variant=variant,
                checkpoint_path=str(checkpoint), checkpoint_sha256=item['checkpoint_sha256'],
                validation_f1=item['validation_f1'], threshold=0.5,
                training_performed=False, test_based_selection=False,
                predictions_sha256=sha(run_output / 'predictions.csv'),
            ))
            records.append(dict(variant=variant, task=task, seed=seed,
                                checkpoint_sha256=item['checkpoint_sha256'], **metric))
            csvsave(output / 'test_metrics_by_seed.csv', records)
            summarize(output, records, protocol)
            save(output / 'status.json', dict(
                status='RUNNING_TEST', frozen_checkpoints=len(frozen),
                completed_test_evaluations=len(records), training_performed=False,
                test_based_selection=False, elapsed_seconds=time.time() - start,
            ))
            print(f"FINAL_ABLATION_TEST {index}/95 {variant} {task} seed={seed} F1={100*metric['f1']:.2f}", flush=True)
            del model, dataset, loader, state, full_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summarize(output, records, protocol)
        save(output / 'status.json', dict(
            status='COMPLETE', frozen_checkpoints=len(frozen),
            completed_test_evaluations=len(records), training_performed=False,
            test_based_selection=False, elapsed_seconds=time.time() - start,
        ))
        print((output / 'readable_test_summary.txt').read_text(), flush=True)
        archive(output)
    except BaseException:
        save(output / 'status.json', dict(status='FAILED_OR_INTERRUPTED',
                                           training_performed=False,
                                           elapsed_seconds=time.time() - start,
                                           error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

