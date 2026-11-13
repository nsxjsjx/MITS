import csv
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

def read_manifest(c):
    with (Path(c['report_root'])/'final_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    seen=set(); groups={}
    for r in rows:
        key=(r['task'],r['sample_id'])
        if key in seen: raise ValueError('Duplicate manifest ID: '+str(key))
        seen.add(key)
        if r['split'] not in c['splits'] or int(r['label']) not in [0,1]: raise ValueError('Invalid manifest')
        if r['training_only'].lower()=='true' and r['split']!='train': raise ValueError('Training-only leakage')
        group=r['split_group']
        if group in groups and groups[group]!=r['split']: raise ValueError('Family crosses splits: '+group)
        groups[group]=r['split']
    return rows

def coverage_ids(rows, fraction, seed):
    """Nested group selection, stratified approximately by task/label signature."""
    groups={}
    for r in rows:
        if r['split']=='train': groups.setdefault(r['split_group'],[]).append(r)
    buckets={}
    for g,rr in groups.items():
        signature=tuple(sorted({(r['task'],r['label']) for r in rr}))
        buckets.setdefault(signature,[]).append(g)
    selected=set()
    for gs in buckets.values():
        gs.sort(key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
        n=len(gs) if fraction==1 else max(1,round(len(gs)*fraction))
        selected.update(gs[:n])
    return {r['task']+'/'+r['sample_id'] for r in rows if r['split']=='train' and r['split_group'] in selected}

class RevisionDataset(Dataset):
    def __init__(self,c,task,split,need_source=True,teacher_subset=False):
        if c.get('stage13') and split=='test':raise ValueError('Stage 13 forbids loading test samples')
        allrows=read_manifest(c); self.c=c; self.task=task; self.split=split
        self.root=Path(c['data_root'])/task; self.need_source=need_source
        self.window_encoder_identity=None
        if need_source and c.get('source_strategy')=='window_mean':
            import json
            from revision.features import encoder_identity
            identity_file=Path(c['control_root'])/'window_mean/encoder_identity.json'
            self.window_encoder_identity=json.loads(identity_file.read_text(encoding='utf-8'))
            if self.window_encoder_identity!=encoder_identity(c['source_encoder_local']):raise ValueError('Window encoder files changed')
        by={r['sample_id']:r for r in allrows if r['task']==task and r['split']==split}
        names=(Path(c['data_root'])/'splits'/task/(split+'_name.txt')).read_text().splitlines()
        labels=(Path(c['data_root'])/'splits'/task/(split+'_label.txt')).read_text().splitlines()
        if len(names)!=len(labels) or len(names)!=len(by): raise ValueError('Manifest/list count mismatch')
        if len(set(names))!=len(names) or {Path(n).stem for n in names}!=set(by):raise ValueError('Split IDs do not match manifest')
        self.rows=[]; self.available=coverage_ids(allrows,c['source_coverage'],c['coverage_seed'])
        for name,label in zip(names,labels):
            r=by[Path(name).stem]
            if int(label)!=int(r['label']): raise ValueError('Label mismatch')
            if teacher_subset and split=='train' and task+'/'+r['sample_id'] not in self.available: continue
            self.rows.append(r)
        for r in self.rows:
            for folder in ['cfg_pyg_data','bytecode_tokens']+(['source_vectors'] if need_source else []):
                if not (self.root/folder/(r['sample_id']+'.pt')).is_file(): raise FileNotFoundError(str((self.root/folder/r['sample_id'])))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; sid=r['sample_id']
        def load(folder): return torch.load(self.root/folder/(sid+'.pt'),map_location='cpu')
        graph=load('cfg_pyg_data'); tokens=load('bytecode_tokens').reshape(-1)
        if graph.x.dtype!=torch.long or graph.x.ndim!=2 or not len(graph.x): raise ValueError('Invalid graph token input')
        if tokens.dtype!=torch.long or len(tokens)!=self.c['sequence_max_length'] or graph.x.size(1)!=self.c['node_max_length']:
            raise ValueError('Fixed feature dimensions/dtypes disagree with configuration')
        if tuple(graph.edge_index.shape)==(0,): graph.edge_index=graph.edge_index.reshape(2,0)
        if graph.edge_index.ndim!=2 or graph.edge_index.shape[0]!=2 or graph.edge_index.dtype!=torch.long: raise ValueError('Invalid edge shape/type')
        if graph.edge_index.numel() and (graph.edge_index.min()<0 or graph.edge_index.max()>=len(graph.x)): raise ValueError('Invalid edge reference')
        if tokens.min()<0 or graph.x.min()<0 or tokens.max()>=self.c['model']['vocab_size'] or graph.x.max()>=self.c['model']['vocab_size']: raise ValueError('Token outside vocabulary')
        for value in [tokens.unsqueeze(0),graph.x]:
            mask=value.ne(0);length=mask.sum(-1)
            expected=torch.arange(value.size(-1))[None,:]<length[:,None]
            if (length==0).any() or not torch.equal(mask,expected):raise ValueError('Empty input or nontrailing padding')
        if self.c.get('shuffle_instructions'):
            cache=Path(self.c['control_root'])/'shuffle'/str(self.c['run_seed'])/self.task/(sid+'.pt')
            if not cache.is_file():raise FileNotFoundError('Prepare instruction-shuffle cache before training: '+str(cache))
            control=torch.load(cache,map_location='cpu')
            from revision.config import digest
            if control.get('cache_version')!='visible-instructions-v2':raise ValueError('Rebuild outdated instruction-shuffle cache')
            if control['source_bytecode_hash']!=digest(self.root/'bytecode_pure'/(sid+'.sol')):raise ValueError('Stale shuffle cache')
            if not torch.equal(control['graph'].edge_index,graph.edge_index):
                old=sorted(map(tuple,graph.edge_index.t().tolist()));new=sorted(map(tuple,control['graph'].edge_index.t().tolist()))
                if old!=new:raise ValueError('Shuffle control changes graph edges')
            graph=control['graph']
        available=self.need_source and (self.split!='train' or self.task+'/'+sid in self.available)
        if available and self.c.get('source_strategy','truncate')=='window_mean':
            from revision.config import digest
            path=Path(self.c['control_root'])/'window_mean'/self.task/(sid+'.pt')
            control=torch.load(path,map_location='cpu')
            if control.get('cache_version')!='source-window-v2' or control.get('encoder_identity')!=self.window_encoder_identity:
                raise ValueError('Window cache version/encoder mismatch')
            if control['source_hash']!=digest(self.root/'sourcecode'/(sid+'.sol')):raise ValueError('Stale window cache')
            vector=control['vector'].reshape(-1)
        else:vector=load('source_vectors').reshape(-1) if available else torch.zeros(self.c['model']['source_dim'])
        if vector.numel()!=self.c['model']['source_dim'] or not torch.isfinite(vector).all(): raise ValueError('Invalid source vector')
        return dict(sample_id=sid,task=self.task,label=int(r['label']),graph=graph,tokens=tokens,source_vector=vector,source_available=available,split_group=r['split_group'])

def collate(rows):
    return dict(graph=Batch.from_data_list([r['graph'] for r in rows]),
                tokens=torch.stack([r['tokens'] for r in rows]),source_vector=torch.stack([r['source_vector'] for r in rows]),
                label=torch.tensor([r['label'] for r in rows],dtype=torch.float),
                source_available=torch.tensor([r['source_available'] for r in rows],dtype=torch.bool),
                sample_id=[r['sample_id'] for r in rows],split_group=[r['split_group'] for r in rows])
