"""Bounded synthetic checks: masks, gradients, capacity guards and portability."""
import argparse,copy,tempfile,json
from pathlib import Path
import torch
from revision.config import load_config,dump
from revision.model import NodeEncoder,matched_node_config,validate_node_matching
from revision.experiments import configure
from revision.probe import linear_cka
from revision.efficiency import latency_summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output',required=True);a=p.parse_args()
    c=load_config(a.config);torch.set_num_threads(2);report={}
    tokens=torch.tensor([[1,5,6,2,0],[1,8,2,0,0]])
    permutation=tokens.clone();permutation[0,:4]=tokens[0,torch.tensor([2,0,3,1])];permutation[1,:3]=tokens[1,torch.tensor([1,2,0])]
    for kind in ['mean','max','median','sum']:
        mc=copy.deepcopy(c['model']);mc.update(dim=16,node_encoder=kind)
        node=NodeEncoder(mc);node.eval();x=node(tokens)
        assert torch.allclose(x,node(torch.nn.functional.pad(tokens,(0,7))),atol=1e-6)
        assert torch.allclose(x,node(permutation),atol=1e-6)
        x.sum().backward();assert node.embedding.weight.grad is not None
        assert torch.equal(node.embedding.weight.grad[0],torch.zeros(16))
        assert sum(p.numel() for p in node.net.parameters())==0
        empty=False
        try:node(torch.zeros(1,5,dtype=torch.long))
        except ValueError:empty=True
        assert empty;report['pool_'+kind]='PAD/permutation/gradient/empty guards PASS'
    # Actual parameter values match manual pooling (including even median), not just an implementation mirror.
    mc=copy.deepcopy(c['model']);mc.update(dim=2,node_encoder='median',vocab_size=10)
    n=NodeEncoder(mc);n.eval()
    with torch.no_grad():n.embedding.weight[1]=torch.tensor([2.,4.]);n.embedding.weight[2]=torch.tensor([6.,10.])
    assert torch.equal(n(torch.tensor([[1,2,0]])),torch.tensor([[4.,7.]]))
    report['even_median']='PASS'
    for kind in ['mlp','bigru','attention']:
        mc,r=matched_node_config(c['model'],kind);actual=validate_node_matching(mc,r)
        bad=dict(r);bad['actual']+=1;rejected=False
        try:validate_node_matching(mc,bad)
        except ValueError:rejected=True
        assert rejected
        report['matched_'+kind]=actual
    for kind in ['mean','max','median','sum']:
        cfg=configure(c,'pool_'+kind);assert cfg['teacher_variant']=='pool_'+kind
    import numpy as np
    rng=np.random.RandomState(3);x=rng.randn(12,8);assert abs(linear_cka(x,x)-1)<1e-12
    assert linear_cka(np.zeros_like(x),x) is None
    report['cka_invariants']='PASS'
    latency=latency_summary([dict(status='SUCCESS',total_wall_seconds=1),dict(status='SUCCESS',total_wall_seconds=3),dict(status='TIMEOUT',total_wall_seconds=60)])
    assert latency['success']==2 and latency['successful_mean_seconds']==2 and latency['all_attempts_mean_seconds']>20
    report['failure_coverage_not_fake_negative']='PASS'
    assert c['training']['optimizer']=='adam' and c['training']['scheduler']=='none'
    report['training_rule']={k:c['training'][k] for k in ['optimizer','learning_rate','weight_decay','scheduler']}
    dump(a.output,dict(status='PASS',checks=report,training_performed=False,scope='Synthetic checks, not real baseline or paper accuracy'))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
