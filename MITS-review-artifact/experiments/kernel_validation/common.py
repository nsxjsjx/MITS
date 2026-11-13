import csv
import hashlib
import json
import os
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def csvsave(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup(project):
    protocol = json.loads((BUNDLE / 'protocol.json').read_text(encoding='utf-8'))
    project = Path(project).resolve()
    for relative, expected in protocol['project_code_sha256'].items():
        if sha(project / relative) != expected:
            raise ValueError('Project code changed: ' + relative)
    os.chdir(project)
    sys.path.insert(0, str(project))
    import torch
    torch.set_num_threads(4)
    pairs = json.loads((BUNDLE / 'pairs.json').read_text(encoding='utf-8'))
    return protocol, pairs


def forbid_test():
    from revision.data import RevisionDataset
    original = RevisionDataset.__init__

    def guarded(self, config, task, split, *args, **kwargs):
        if split == 'test':
            raise RuntimeError('Training stage forbids TEST loading')
        original(self, config, task, split, *args, **kwargs)

    RevisionDataset.__init__ = guarded


def selected_folder(task, seed, work, node_root, branch_root, gate_root, overflow_root=None):
    if task == 'reentrancy':
        return gate_root / 'runs' / 'gated_conv_3branches' / 'student' / task / str(seed)
    if task == 'delegatecall':
        return branch_root / 'runs' / 'no_derived_conv' / 'student' / task / str(seed)
    if task == 'integeroverflow' and overflow_root is not None:
        return overflow_root / 'checkpoints' / task / str(seed) / 'student'
    return node_root / work['relative_folder']


def no_kd_folder(task, seed, no_kd_root, overflow_no_kd_root=None):
    if task == 'integeroverflow' and overflow_no_kd_root is not None:
        return overflow_no_kd_root / 'runs' / 'beta0' / 'main' / 'student' / task / str(seed)
    return no_kd_root / 'runs' / 'no_distillation_selected' / 'student' / task / str(seed)


def removed_branch(task, variant):
    final = {
        'reentrancy': ['graph', 'sequence', 'derived'],
        'timestamp': ['graph', 'sequence', 'derived'],
        'integeroverflow': ['graph', 'sequence', 'derived'],
        'delegatecall': ['graph', 'sequence'],
    }[task]
    mapping = {
        'no_graph': 'graph',
        'no_direct_sequence': 'sequence',
        'no_derived': 'derived',
    }
    removed = mapping[variant]
    if removed not in final:
        return None
    return [branch for branch in final if branch != removed]

