import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def chunks(batch,size=1):
 from torch_geometric.data import Batch
 graphs=batch['graph'].to_data_list();n=len(batch['label'])
 for lo in range(0,n,size):
  hi=min(n,lo+size);out={}
  for k,v in batch.items():
   if k=='graph':out[k]=Batch.from_data_list(graphs[lo:hi])
   elif torch.is_tensor(v):out[k]=v[lo:hi]
   elif isinstance(v,list):out[k]=v[lo:hi]
   else:raise ValueError('Unexpected batch field '+k)
  yield out

def objective(student,labels,teacher,paired,logical_n,logical_paired,args,loss_fn):
 # BCE is averaged over logical examples; KD over logical source-paired examples.
 # Never average microbatch means equally when lengths or pairing differ.
 n=len(labels);p=int(paired.sum())
 bce=F.binary_cross_entropy_with_logits(student['logits'],labels.float())
 kd_args=dict(args);kd_args['alpha']=0.
 kd_only,parts=loss_fn(student,labels,teacher,paired,**kd_args)
 weighted=float(args.get('alpha',1.))*bce*(n/logical_n)
 if p and logical_paired:weighted=weighted+kd_only*(p/logical_paired)
 return weighted,dict(bce=float(bce.detach()),kd=parts['kd'])

def attach_node_chunking(model,size=64):
 if not hasattr(model,'node'):return model
 original=model.node.forward
 def bounded(tokens):
  out=[]
  for lo in range(0,len(tokens),size):
   piece=tokens[lo:lo+size]
   if model.role=='student' and model.training and torch.is_grad_enabled():
    value=checkpoint(original,piece,use_reentrant=False,preserve_rng_state=True)
   else:value=original(piece)
   out.append(value)
  return torch.cat(out,dim=0)
 model.node.forward=bounded
 return model

def factory(Base,original_model,seed):
 from adapter import model_factory
 Parent=model_factory(Base,original_model,seed)
 class LowMemoryModel(Parent):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);attach_node_chunking(self,64)
 return LowMemoryModel
