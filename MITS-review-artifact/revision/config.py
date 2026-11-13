import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_config(path=None):
    c = yaml.safe_load(Path(path or ROOT/'revision.yaml').read_text(encoding='utf-8-sig'))
    # Explicitly document actual search behavior; never silently restore old-paper rules.
    c['training'].setdefault('optimizer','adam')
    c['training'].setdefault('weight_decay',0.0)
    c['training'].setdefault('scheduler','none')
    if c['training']['optimizer']!='adam' or c['training']['scheduler']!='none':
        raise ValueError('This revision implements Adam without a scheduler; use a separately validated version to change it')
    if c['training']['weight_decay']<0:raise ValueError('Negative weight_decay')
    if c['bytecode_type'] != 'creation':
        raise ValueError('This frozen dataset contains creation bytecode, not runtime.')
    if c['training']['selection']!='val_f1' or c['training']['tie_policy']!='earliest':
        raise ValueError('Implemented selection is validation F1, earliest epoch on ties')
    if c['distillation']['position']!='post_dropout' or c['distillation']['reduction']!='paired_sample_mean':
        raise ValueError('Unsupported distillation placement/reduction')
    if c['distillation']['loss_type'] not in ['feature_kl','feature_mse','none'] or c['distillation']['beta']<0:
        raise ValueError('Invalid distillation settings')
    if not 0<c['source_coverage']<=1 or not 0<=c['training']['threshold']<=1:
        raise ValueError('Invalid coverage or threshold')
    for value in [c['training']['epochs'],c['training']['batch_size'],c['training']['patience']]:
        if value<1:raise ValueError('Training counts must be positive')
    return c

def dump(path, value):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def code_hashes():
    return {str(p.relative_to(ROOT)):digest(p) for base in ['revision','model','experiments','utils']
            for p in sorted((ROOT/base).glob('*.py'))}

def data_identity(c):
    report=Path(c['report_root']);data=Path(c['data_root'])
    paths=[report/'final_manifest.csv',report/'final_dataset_file_hashes.json',report/'split_protocol.json']
    splits=['train','validation'] if c.get('stage13') else c['splits']
    paths += [data/'splits'/task/(split+'_'+kind+'.txt') for task in c['tasks'] for split in splits for kind in ['name','label']]
    return {str(p):digest(p) for p in paths}
