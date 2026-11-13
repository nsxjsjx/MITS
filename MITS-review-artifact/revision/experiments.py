import copy
from revision.model import matched_node_config

VARIANTS=['main','beta0','bytecode_teacher','feature_mse','no_graph','no_derived','no_direct_sequence',
          'node_mlp','node_bigru','node_attention','pool_mean','pool_max','pool_median','pool_sum','kernel1','kernel3','kernel5','shuffle',
          'coverage10','coverage25','coverage50','coverage100','source_window']

def configure(c,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    c=copy.deepcopy(c);c['variant']=variant;c['teacher_type']='multimodal'
    if variant=='beta0':c['distillation']['beta']=0.;c['distillation']['loss_type']='none'
    if variant=='bytecode_teacher':c['teacher_type']='bytecode'
    if variant=='feature_mse':c['distillation']['loss_type']='feature_mse'
    remove={'no_graph':'graph','no_derived':'derived','no_direct_sequence':'sequence'}
    if variant in remove:c['model']['branches'].remove(remove[variant])
    if variant.startswith('node_'):
        c['model'],c['node_matching']=matched_node_config(c['model'],variant[5:])
    if variant.startswith('pool_'):c['model']['node_encoder']=variant[5:]
    if variant.startswith('kernel'):c['model']['kernel_size']=int(variant[6:])
    if variant=='shuffle':c['shuffle_instructions']=True
    if variant.startswith('coverage'):c['source_coverage']=int(variant[8:])/100
    if variant=='source_window':c['source_strategy']='window_mean'
    # Unchanged teachers are reused by explicit identity, not by finding any checkpoint.
    c['teacher_variant']=variant if variant.startswith(('node_','pool_','kernel','coverage')) or variant in ['bytecode_teacher','shuffle','source_window'] else 'main'
    return c
