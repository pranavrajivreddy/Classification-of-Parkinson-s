"""Shared classification metrics; extracted without changing numerical operations."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from research.core import metrics

def allmetrics(y,p,pred):
 m=metrics(y,p,pred);tp,fp,tn,fn=[m[k] for k in ['tp','fp','tn','fn']]
 m.update(accuracy=(tp+tn)/m['n'],f1=2*tp/max(2*tp+fp+fn,1),mcc=(tp*tn-fp*fn)/max(np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),1))
 np.testing.assert_allclose([m['accuracy'],m['f1'],m['mcc']],[accuracy_score(y,pred),f1_score(y,pred,zero_division=0),matthews_corrcoef(y,pred)])
 return m
