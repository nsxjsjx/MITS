import copy,hashlib
import torch

def tensor_hash(model):
 return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()

def model_factory(Base,original_model,seed):
 class StudentOnlyNodeChange(Base):
  def __init__(self,config,role='student',teacher_type='multimodal'):
   super().__init__(config,role,teacher_type)
   if role!='student':return
   # Match initial values of every shared module, including opcode embeddings.
   # Node-core parameters differ in shape; no claim of identical full initialization.
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed);reference=Base(original_model,'student',teacher_type)
   source=reference.state_dict();state=self.state_dict();copied=[]
   for name,value in state.items():
    if name.startswith(('node.net.','node.proj.','node.attn_in.')):continue
    if name not in source or source[name].shape!=value.shape:raise ValueError('Unexpected common-module shape: '+name)
    value.copy_(source[name]);copied.append(name)
   self.load_state_dict(state)
   self.shared_init_audit={'shared_parameter_tensors_identical':len(copied),'node_core_initialization':'seeded, architecture-specific','full_initialization_identical':False}
 return StudentOnlyNodeChange
