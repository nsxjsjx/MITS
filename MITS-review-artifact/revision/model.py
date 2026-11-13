import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch_geometric.nn import GatedGraphConv,global_mean_pool

def masked_max(x,mask):
    return x.masked_fill(~mask.unsqueeze(-1),torch.finfo(x.dtype).min).max(1).values

def masked_pool(x,mask,kind):
    """Pool valid tokens only. Median uses midpoint for an even valid length."""
    lengths=mask.sum(1)
    if (lengths==0).any():raise ValueError('All-PAD node')
    if kind=='max':return masked_max(x,mask)
    if kind in ['mean','sum']:
        total=x.masked_fill(~mask.unsqueeze(-1),0).sum(1)
        return total/lengths.unsqueeze(-1) if kind=='mean' else total
    if kind=='median':
        ordered=x.masked_fill(~mask.unsqueeze(-1),float('inf')).sort(dim=1).values
        batch=torch.arange(x.size(0),device=x.device)
        return (ordered[batch,(lengths-1)//2]+ordered[batch,lengths//2])/2
    raise ValueError(kind)

class NodeEncoder(nn.Module):
    def __init__(self,c):
        super().__init__(); d=c['dim']; self.kind=c['node_encoder']; self.k=c['kernel_size']
        self.embedding=nn.Embedding(c['vocab_size'],d,padding_idx=0)
        inner=c.get('node_inner') or d
        if self.kind in ['mean','max','median','sum']:self.net=nn.Identity()
        elif self.kind=='conv': self.net=nn.Conv1d(d,d,self.k,padding=self.k//2)
        elif self.kind=='mlp': self.net=nn.Sequential(nn.Linear(d,inner),nn.ReLU(),nn.Linear(inner,d))
        elif self.kind=='bigru':
            self.net=nn.GRU(d,inner,batch_first=True,bidirectional=True); self.proj=nn.Linear(2*inner,d)
        elif self.kind=='attention':
            self.attn_in=nn.Linear(d,inner); self.net=nn.MultiheadAttention(inner,8,batch_first=True); self.proj=nn.Linear(inner,d)
        else: raise ValueError(self.kind)
        self.drop=nn.Dropout(c['node_dropout'])
    def forward(self,tokens):
        mask=tokens.ne(0); lengths=mask.sum(1)
        if (lengths==0).any(): raise ValueError('All-PAD node')
        x=self.embedding(tokens)
        if self.kind in ['mean','max','median','sum']:
            # Nonparametric historical controls: no extra hidden trainable projection.
            return masked_pool(self.drop(F.relu(x)),mask,self.kind)
        if self.kind=='conv':
            x=self.net(x.transpose(1,2)).transpose(1,2)
            # Retain exactly the output extent obtained by convolving the unpadded node.
            out_lengths=lengths+2*(self.k//2)-self.k+1
            mask=torch.arange(x.size(1),device=x.device)[None,:]<out_lengths[:,None]
        elif self.kind=='bigru':
            x,_=self.net(pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False))
            x,_=pad_packed_sequence(x,batch_first=True,total_length=tokens.size(1)); x=self.proj(x)
        elif self.kind=='attention':
            x=self.attn_in(x);x,_=self.net(x,x,x,key_padding_mask=~mask,need_weights=False); x=self.proj(x)
        else: x=self.net(x)
        return masked_max(self.drop(F.relu(x)),mask)

class MITS(nn.Module):
    def __init__(self,config,role='student',teacher_type='multimodal'):
        super().__init__(); c=copy.deepcopy(config); self.config=c; self.role=role; self.teacher_type=teacher_type
        d=c['dim']; self.branches=list(c['branches'])
        if role=='teacher': self.branches=['graph','sequence','source' if teacher_type=='multimodal' else 'derived']
        if 'graph' in self.branches:
            self.node=NodeEncoder(c); self.gnn=GatedGraphConv(d,c['graph_steps'])
            self.graph_drop=nn.Dropout(c['graph_dropout']); self.graph_proj=nn.Linear(d,d)
        if 'sequence' in self.branches or 'derived' in self.branches:
            self.seq_embedding=nn.Embedding(c['vocab_size'],c['seq_embedding'],padding_idx=0)
            self.seq=nn.LSTM(c['seq_embedding'],c['seq_hidden'],num_layers=c['seq_layers'],bidirectional=True,
                            batch_first=True,dropout=c['seq_dropout'] if c['seq_layers']>1 else 0)
            self.seq_proj=nn.Linear(2*c['seq_hidden'],d)
        if 'derived' in self.branches: self.derived=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(c['seq_dropout']))
        if 'source' in self.branches:
            layers=[]; inp=c['source_dim']
            for _ in range(c['source_layers']):
                layers.extend([nn.Linear(inp,d),nn.LayerNorm(d),nn.ReLU(),nn.Dropout(c['source_dropout'])]); inp=d
            self.source=nn.Sequential(*layers)
        self.fusion=nn.Sequential(nn.Linear(len(self.branches)*d,d),nn.ReLU(),nn.Dropout(c['fusion_dropout']))
        self.classifier=nn.Linear(d,1)
    def forward(self,graph,tokens,source=None,return_branches=False):
        b={}
        if 'graph' in self.branches:
            nodes=self.node(graph.x)
            b['graph']=F.relu(self.graph_proj(global_mean_pool(self.graph_drop(self.gnn(nodes,graph.edge_index)),graph.batch)))
        if 'sequence' in self.branches or 'derived' in self.branches:
            mask=tokens.ne(0); lengths=mask.sum(1)
            if (lengths==0).any(): raise ValueError('Empty sequence')
            out,_=self.seq(pack_padded_sequence(self.seq_embedding(tokens),lengths.cpu(),batch_first=True,enforce_sorted=False))
            out,_=pad_packed_sequence(out,batch_first=True,total_length=tokens.size(1))
            seq=F.relu(self.seq_proj(masked_max(out,mask)))
            if 'sequence' in self.branches:b['sequence']=seq
            if 'derived' in self.branches:b['derived']=self.derived(seq)
        if 'source' in self.branches:
            if source is None:raise ValueError('Multimodal teacher requires paired source')
            b['source']=self.source(source)
        g=self.fusion(torch.cat([b[k] for k in self.branches],dim=-1))
        result={'logits':self.classifier(g).squeeze(-1),'distill_features':g}
        if return_branches:result['branches']=b
        return result

def parameter_report(model):
    return {'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'branches':model.branches,'role':model.role,'teacher_type':model.teacher_type,
            'modules':{n:sum(p.numel() for p in m.parameters()) for n,m in model.named_children()}}

def matched_node_config(config,kind,tolerance=.05):
    """Match active node-encoder parameters; never add disconnected dummy weights."""
    c=copy.deepcopy(config); baseline=copy.deepcopy(c);baseline.update(node_encoder='conv',node_inner=None)
    target=sum(p.numel() for p in NodeEncoder(baseline).parameters()); c['node_encoder']=kind
    best=None
    # Parameter counts are analytic, avoiding hundreds of large allocations.
    d=c['dim']; embed=c['vocab_size']*d
    for h in range(8,4*d+1,8):
        counts={'mlp':embed+d*h+h+h*d+d,'bigru':embed+6*h*(d+h+2)+2*h*d+d,
                'attention':embed+d*h+h+4*h*h+4*h+h*d+d}
        count=counts[kind]; gap=abs(count-target)/target
        if best is None or gap<best[0]:best=(gap,h,count)
    c['node_inner']=best[1]
    target_core=target-embed;actual_core=best[2]-embed
    core_gap=abs(actual_core-target_core)/max(target_core,1)
    report=dict(target=target,actual=best[2],relative_gap=best[0],target_core=target_core,actual_core=actual_core,
                core_relative_gap=core_gap,matched=max(best[0],core_gap)<=tolerance,tolerance=tolerance)
    if not report['matched']:raise ValueError('Node parameter matching exceeds tolerance: '+str(report))
    return c,report

def validate_node_matching(model_config,report):
    """Recount registered active parameters; reject invalid or stale matching claims."""
    actual=sum(p.numel() for p in NodeEncoder(model_config).parameters())
    baseline=copy.deepcopy(model_config);baseline.update(node_encoder='conv',node_inner=None)
    target=sum(p.numel() for p in NodeEncoder(baseline).parameters())
    embed=model_config['vocab_size']*model_config['dim'];tol=report['tolerance']
    full_gap=abs(actual-target)/max(target,1);core_gap=abs(actual-target)/max(target-embed,1)
    if report['actual']!=actual or report['target']!=target or max(full_gap,core_gap)>tol:
        raise ValueError('Actual node parameters do not match required capacity: '+str((target,actual,full_gap,core_gap,tol)))
    return dict(report,actual_recounted=actual,target_recounted=target,verified=True)
