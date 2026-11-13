"""Representation comparison; probes fit train features only."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def linear_cka(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)!=len(y) or len(x)<2:raise ValueError('Aligned samples required')
    x=x-x.mean(0);y=y-y.mean(0)
    denom=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2/denom) if denom else None

def fit_probe(train_x,train_y,val_x,val_y,seed=17):
    best=None
    for strength in [.01,.1,1.,10.]:
        model=LogisticRegression(C=strength,max_iter=2000,random_state=seed).fit(train_x,train_y)
        score=f1_score(val_y,model.predict(val_x),zero_division=0)
        if best is None or score>best[0]:best=(score,model,strength)
    return best[1],dict(validation_f1=float(best[0]),C=best[2],claim='Auxiliary representation evidence, not proof of semantic transfer')
