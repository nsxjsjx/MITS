import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,balanced_accuracy_score,average_precision_score,precision_recall_curve,auc,confusion_matrix
from scipy.stats import wilcoxon,binomtest

def metrics(y,prob,threshold=.5):
    y=np.asarray(y);prob=np.asarray(prob); pred=(prob>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    p,r,_=precision_recall_curve(y,prob)
    return dict(accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
                mcc=float(matthews_corrcoef(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                fpr=float(fp/(fp+tn)) if fp+tn else None,pr_auc=float(auc(r,p)) if y.sum() else None,
                average_precision=float(average_precision_score(y,prob)) if y.sum() else None,
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),n=len(y))

def summarize(values):
    a=np.array(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else None)

def paired_runs(a,b):
    if set(a)!=set(b):raise ValueError('Seeds must match; do not drop low-performing runs')
    delta=np.array([a[s]-b[s] for s in sorted(a)])
    return dict(difference=summarize(delta),wilcoxon_p=float(wilcoxon(delta).pvalue) if np.any(delta) else 1.,
                caution='Five pairs have low power; report effect sizes and do not promise p<0.05.')

def paired_sample_test(y,a,b):
    ca=np.array(a)==np.array(y);cb=np.array(b)==np.array(y)
    ab=int((ca & ~cb).sum());ba=int((~ca & cb).sum())
    return dict(a_only_correct=ab,b_only_correct=ba,mcnemar_exact_p=float(binomtest(ab,ab+ba,.5).pvalue) if ab+ba else 1.)

def group_bootstrap(y,pa,pb,groups,seed=42,repeats=2000):
    y=np.asarray(y);pa=np.asarray(pa);pb=np.asarray(pb);groups=np.asarray(groups);unique=np.unique(groups)
    rng=np.random.RandomState(seed);values=[]
    for _ in range(repeats):
        idx=np.concatenate([np.where(groups==g)[0] for g in rng.choice(unique,len(unique),replace=True)])
        values.append(f1_score(y[idx],pa[idx],zero_division=0)-f1_score(y[idx],pb[idx],zero_division=0))
    return dict(delta_f1=float(f1_score(y,pa,zero_division=0)-f1_score(y,pb,zero_division=0)),
                ci95=[float(x) for x in np.percentile(values,[2.5,97.5])],unit='split_group',repeats=repeats)

def four_task_mean(f1_by_task):
    if set(f1_by_task)!={'reentrancy','timestamp','integeroverflow','delegatecall'}:raise ValueError('Four complete tasks required')
    return float(np.mean(list(f1_by_task.values())))
