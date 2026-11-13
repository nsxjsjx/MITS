"""Targeted regression checks for the methodological bugs; no paper training."""
import copy,json,tempfile
from pathlib import Path
import torch
from torch_geometric.data import Data,Batch
from revision.config import load_config,dump,ROOT
from revision.model import MITS
from revision.distillation import loss
from revision.data import RevisionDataset,collate,read_manifest,coverage_ids

def main():
    c=load_config();torch.set_num_threads(2);m=copy.deepcopy(c['model']);m.update(dim=16,seq_embedding=8,seq_hidden=8,seq_layers=1,graph_steps=1)
    results={};graph=Batch.from_data_list([Data(x=torch.tensor([[1,5,6,2,0],[1,9,2,0,0]]),edge_index=torch.tensor([[0],[1]]))])
    seq=torch.tensor([[1,5,6,2,0]]);source=torch.randn(1,768);x=graph.x.clone()
    for kind in ['conv','mlp','bigru','attention']:
        m['node_encoder']=kind;student=MITS(m);teacher=MITS(m,'teacher');student.eval();teacher.eval();teacher.requires_grad_(False)
        out=student(graph,seq);t=teacher(graph,seq,source)
        objective,_=loss(out,torch.ones(1),t);objective.backward()
        assert torch.equal(graph.x,x),'Model mutates graph input'
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is not None for p in student.parameters()),'Registered unused student parameter'
        padded=graph.clone();padded.x=torch.nn.functional.pad(padded.x,(0,7))
        assert torch.allclose(out['logits'],student(padded,torch.nn.functional.pad(seq,(0,7)))['logits'],atol=1e-6),'PAD changes output'
        bce=torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],torch.ones(1))
        assert torch.equal(loss(out,torch.ones(1),None,beta=0)[0],bce)
        results[kind]='graph unchanged; gradients; teacher frozen; padding invariance; beta0 pass'
    s=torch.randn(2,16,requires_grad=True);t=torch.randn(2,16)
    one={'logits':torch.zeros(1),'distill_features':s[:1]};one_t={'distill_features':t[:1]}
    _,first=loss(one,torch.zeros(1),one_t)
    _,masked=loss({'logits':torch.zeros(2),'distill_features':s},torch.zeros(2),{'distill_features':t},torch.tensor([True,False]))
    assert abs(first['kd']-masked['kd'])<1e-6
    results['batch_independent_kd']='PASS'
    _,zero=loss({'logits':torch.zeros(2),'distill_features':s},torch.zeros(2),{'distill_features':t},torch.tensor([False,False]))
    assert zero['kd']==0
    manifest=read_manifest(c);previous=set()
    for fraction in [.1,.25,.5,1.]:
        selected=coverage_ids(manifest,fraction,c['coverage_seed']);assert previous<=selected;previous=selected
        for task in c['tasks']:
            cc=copy.deepcopy(c);cc['source_coverage']=fraction
            teacher_data=RevisionDataset(cc,task,'train',teacher_subset=True)
            assert all(task+'/'+r['sample_id'] in selected for r in teacher_data.rows)
    results['coverage_nested_and_teacher_subset']='PASS'
    m['node_encoder']='mlp';node=MITS(m).node;node.eval()
    a=torch.tensor([[1,5,6,2,0]]);b=torch.tensor([[6,1,2,5,0]])
    assert torch.allclose(node(a),node(b),atol=1e-6)
    results['MLP_order_invariance']='PASS'
    from revision.experiments import configure,VARIANTS
    small=copy.deepcopy(c);small['model']=m
    for variant in VARIANTS:
        # Strict capacity matching is defined at the actual paper dimension,
        # not a tiny test model where discrete GRU sizes cannot meet tolerance.
        vc=configure(c if variant.startswith('node_') else small,variant);vm=MITS(vc['model']);vm.eval()
        output=vm(graph,seq);output['logits'].sum().backward()
        assert all(p.grad is not None for p in vm.parameters()),'Inactive registered parameters: '+variant
    results['all_variant_forward_and_active_parameters']=len(VARIANTS)
    counts={}
    for task in c['tasks']:
        for split in c['splits']:
            ds=RevisionDataset(c,task,split);counts[task+'/'+split]=len(ds)
            for i in range(len(ds)):ds[i]
            b=collate([ds[0],ds[-1]]);assert b['tokens'].shape==(2,512)
    results['all_input_files_and_12_splits']=counts
    dump(ROOT/'revision_reports'/'checks.json',results);print(json.dumps(results,ensure_ascii=False))

if __name__=='__main__':main()
