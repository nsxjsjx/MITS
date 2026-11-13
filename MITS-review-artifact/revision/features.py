"""Bytecode-only extraction, no dataset writes or source encoder imports."""
import random,re,sys
from pathlib import Path
import torch
from torch_geometric.data import Data
from revision.config import ROOT
from utils.bytecode_to_token import BytecodeOrOpcodeTokenizer

def tokenizer(length):return BytecodeOrOpcodeTokenizer(length,str(ROOT/'preprocess/opcode_groups.json'))

def normalize_bytecode(value):
    value=re.sub(r'\s+','',value)
    if value.startswith('0x'):value=value[2:]
    if not value or len(value)%2 or not re.fullmatch('[0-9a-fA-F]+',value):raise ValueError('Malformed bytecode hex')
    return value

def get_cfg(value):
    sys.path.insert(0,str(ROOT/'preprocess'))
    from binary_extractor.arch.evm.emulator import EvmEmulatorEngine
    from binary_extractor.arch.evm.vmstate import EvmVMstate
    from binary_extractor.platforms.ETH.cfg import EthereumCFG
    EvmEmulatorEngine.emulate.__defaults__=(EvmVMstate(),0)
    return EthereumCFG(normalize_bytecode(value))

def instruction_units(text):
    units=[]
    for line in text.splitlines():
        tokens=[x for x in line.split() if not re.fullmatch('[0-9a-fA-F]+:',x)]
        if tokens:units.append(tokens)
    return units

def graph_from_bytecode(value,c,shuffle_seed=None,with_lengths=False):
    cfg=get_cfg(value);tok=tokenizer(c['node_max_length']);nodes=[];lengths=[];mapping={};rng=random.Random(shuffle_seed)
    for block in cfg.basicblocks:
        mapping[block.name]=len(nodes);units=instruction_units(block.instructions_details())
        if shuffle_seed is not None:
            # Preserve the SAME truncated information as the baseline, then permute
            # complete visible instructions. A pre-existing clipped boundary stays pinned.
            kept=[];boundary=[];budget=c['node_max_length']-2
            for unit in units:
                if len(unit)>budget:
                    boundary=unit[:budget];break
                kept.append(unit);budget-=len(unit)
            rng.shuffle(kept)
            units=kept+([boundary] if boundary else [])
        flat=[x for unit in units for x in unit];lengths.append(len(flat)+2)
        ids,_=tok.tokenize_opcode(flat);nodes.append(ids)
    if not nodes:raise ValueError('CFG has no basic blocks')
    edges=[]
    for e in set(cfg.edges):
        if e.node_from not in mapping or e.node_to not in mapping:raise ValueError('CFG edge has no corresponding block')
        edges.append([mapping[e.node_from],mapping[e.node_to]])
    graph=Data(x=torch.stack(nodes),edge_index=torch.tensor(edges,dtype=torch.long).reshape(-1,2).t().contiguous())
    return (graph,lengths) if with_lengths else graph

def extract(value,c):
    value=normalize_bytecode(value)
    # Fully consume disassembly before the legacy tokenizer can suppress an exception.
    from pyevmasm import disassemble_all
    list(disassemble_all(bytes.fromhex(value)))
    tokens,_=tokenizer(c['sequence_max_length']).tokenize_bytecode(value)
    return graph_from_bytecode(value,c),tokens

def source_vector(code,tokenizer_,encoder,strategy='truncate'):
    """Optional control: frozen CLS vectors, 510-content-token windows averaged."""
    ids=tokenizer_.encode(code,add_special_tokens=False,truncation=False)
    if strategy not in ['truncate','window_mean']:raise ValueError(strategy)
    chunks=[ids[:510]] if strategy=='truncate' else [ids[i:i+510] for i in range(0,len(ids),510)] or [[]]
    encoder.eval();vectors=[];device=next(encoder.parameters()).device
    with torch.no_grad():
        for chunk in chunks:
            batch=tokenizer_.prepare_for_model(chunk,add_special_tokens=True,padding='max_length',max_length=512,return_tensors='pt')
            batch={k:v.reshape(1,-1).to(device) for k,v in batch.items() if k in ['input_ids','attention_mask']}
            vectors.append(encoder(**batch).last_hidden_state[0,0].cpu())
    return torch.stack(vectors).mean(0)

def encoder_identity(path):
    from revision.config import digest
    root=Path(path)
    names=['config.json','vocab.json','merges.txt','tokenizer_config.json']
    weight=root/'pytorch_model.bin'
    if not weight.exists():weight=root/'model.safetensors'
    if not weight.is_file():raise FileNotFoundError('Local encoder weight absent')
    return {name:digest(root/name) for name in names if (root/name).is_file()}|{weight.name:digest(weight)}
