import torch
from torch.nn import functional as F

def loss(student,labels,teacher=None,paired=None,loss_type='feature_kl',alpha=1.,beta=.5,temperature=2.,**unused):
    bce=F.binary_cross_entropy_with_logits(student['logits'],labels.float())
    kd=bce.new_zeros(())
    if beta and loss_type!='none':
        if teacher is None: raise ValueError('KD requires a frozen teacher')
        s=student['distill_features']; t=teacher['distill_features'].detach()
        if s.ndim!=2 or s.shape!=t.shape or s.size(1)<=1: raise ValueError('KD expects matched [B,D], D>1')
        paired=torch.ones(s.size(0),dtype=torch.bool,device=s.device) if paired is None else paired
        if paired.any():
            if loss_type=='feature_kl':
                if temperature<=0:raise ValueError('Invalid temperature')
                kd=F.kl_div(F.log_softmax(s[paired]/temperature,dim=1),F.softmax(t[paired]/temperature,dim=1),reduction='batchmean')*temperature**2
            elif loss_type=='feature_mse':kd=F.mse_loss(s[paired],t[paired])
            else:raise ValueError(loss_type)
    return alpha*bce+beta*kd,{'bce':float(bce.detach()),'kd':float(kd.detach())}
