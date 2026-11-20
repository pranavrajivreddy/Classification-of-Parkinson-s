"""Training-only selection API: fit_fold intentionally accepts no test outcomes."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix

POLICIES = ['default','utility','sensitivity90','specificity60']


def feature_sets(df):
    tap=[c for c in df if c.startswith('tap_')]
    walk=[c for c in df if c.startswith('walk_')]
    voice=[c for c in df if c.startswith('voice_') and not c in
           ['voice_handle','voice_createdOn','voice_device','voice_medication','voice_sessions_total',
            'voice_voiced_fraction','voice_clipping_fraction']]
    demo=['demo_age','demo_gender_male','demo_smoked']
    return {'demographics':demo,'tapping':tap,'walking':walk,'voice':voice,
            'tap_walk':tap+walk,'tap_voice':tap+voice,'walk_voice':walk+voice,
            'sensors':tap+walk+voice,'full':demo+tap+walk+voice}


def engineered(df):
    """Frozen hypotheses, per-row only; not evidence of autonomous discovery."""
    x=df.copy()
    x['eng_tap_iqr_ratio']=x.tap_flight_iqr/(x.tap_flight_mean+1e-4)
    x['eng_tap_variability_miss']=x.tap_flight_std*x.tap_miss_ratio
    x['eng_voice_perturbation_mean']=(x.voice_jitter_local+x.voice_shimmer_local)/2
    x['eng_voice_f0_cv']=x.voice_f0_std/(x.voice_f0_mean+1e-4)
    x['eng_voice_spectral_ratio']=x.voice_spectral_rolloff/(x.voice_spectral_centroid+1e-4)
    x['eng_tap_voice_interaction']=x.tap_flight_iqr*x.voice_jitter_local
    # Do not include algebraic duplicate speed/cadence or unvalidated step regularity.
    return x.replace([np.inf,-np.inf],np.nan)


def splits(y,groups,n=3,seed=42):
    result=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))
    for tr,va in result:
        assert not set(groups[tr]) & set(groups[va])
        assert len(np.unique(y[tr]))==len(np.unique(y[va]))==2
    return result


def candidate_models(seed=42):
    return {
        'lr_c01':LogisticRegression(C=.1,max_iter=3000,random_state=seed),
        'lr_c1':LogisticRegression(C=1,max_iter=3000,random_state=seed),
        'lr_c10':LogisticRegression(C=10,max_iter=3000,random_state=seed),
        'elastic':LogisticRegression(solver='saga',l1_ratio=.5,C=1,max_iter=5000,random_state=seed),
        'rf':RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_leaf=3,random_state=seed,n_jobs=1),
        'gbdt':GradientBoostingClassifier(n_estimators=100,max_depth=2,learning_rate=.05,random_state=seed),
        'svm':SVC(C=1,kernel='rbf',probability=True,random_state=seed)}


def pipeline(model):
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RobustScaler(),clone(model))


def oof(X,y,folds,model):
    p=np.full(len(y),np.nan)
    for tr,va in folds:
        est=pipeline(model);est.fit(X[tr],y[tr]);p[va]=est.predict_proba(X[va])[:,1]
    assert np.isfinite(p).all()
    return p


def operating_table(y,p):
    """Exact unique-score enumeration, both constant decisions included."""
    y=np.asarray(y,int);p=np.asarray(p,float)
    if not np.isfinite(p).all():raise ValueError('nonfinite probabilities')
    order=np.argsort(-p,kind='stable');ss=p[order]; yy=y[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(y)-1]
    tp=np.r_[0,np.cumsum(yy)[ends]];fp=np.r_[0,np.cumsum(1-yy)[ends]]
    fn=y.sum()-tp;tn=(1-y).sum()-fp
    tau=np.r_[np.nextafter(p.max(),np.inf),ss[ends]]
    return pd.DataFrame(dict(threshold=tau,tp=tp,fp=fp,fn=fn,tn=tn,
                             sensitivity=tp/y.sum(),specificity=tn/(1-y).sum(),
                             utility=1-(4*fn+2*fp)/len(y)))


def select_thresholds(y,p):
    t=operating_table(y,p)
    u=t.sort_values(['utility','specificity','threshold'],ascending=False).iloc[0]
    a=t[t.sensitivity>=.90].sort_values(['specificity','sensitivity','threshold'],ascending=False).iloc[0]
    b=t[t.specificity>=.60].sort_values(['sensitivity','specificity','threshold'],ascending=False).iloc[0]
    return dict(default=.5,utility=float(u.threshold),sensitivity90=float(a.threshold),specificity60=float(b.threshold))


def metrics(y,p,pred):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=np.asarray(pred,int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();n=len(y)
    div=lambda a,b:float(a/b) if b else np.nan
    return dict(n=n,tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),
        auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
        sensitivity=div(tp,tp+fn),specificity=div(tn,tn+fp),ppv=div(tp,tp+fp),npv=div(tn,tn+fn),
        f2=div(5*tp,5*tp+4*fn+fp),balanced_accuracy=(div(tp,tp+fn)+div(tn,tn+fp))/2,
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,np.clip(p,1e-9,1-1e-9))),
        utility=float((tp-3*fn-fp+tn)/n),error_cost_3to1=float((3*fn+fp)/n))


def fit_fold(Xtrain,ytrain,groups,Xtest,seed=42):
    folds=splits(ytrain,groups,seed=seed+101)
    models=candidate_models(seed);inner=[];outer=[]
    for name,model in models.items():
        inner.append(oof(Xtrain,ytrain,folds,model))
        est=pipeline(model);est.fit(Xtrain,ytrain);outer.append(est.predict_proba(Xtest)[:,1])
    P=np.column_stack(inner);Q=np.column_stack(outer)
    aucs=[roc_auc_score(ytrain,p) for p in P.T];winner=int(np.argmax(aucs))
    predictions={'selected':(P[:,winner],Q[:,winner]),'equal_vote':(P.mean(1),Q.mean(1))}
    # A train-only calibrator with fresh internal preprocessing for each calibration fit.
    calibrated=CalibratedClassifierCV(pipeline(models[list(models)[winner]]),cv=folds,method='sigmoid',ensemble=True)
    calibrated.fit(Xtrain,ytrain)
    # Cross-fit the calibrator for honest training-only threshold selection.
    cal_inner=np.zeros(len(ytrain))
    for tr,va in folds:
        sub=splits(ytrain[tr],groups[tr],n=3,seed=seed+211)
        c=CalibratedClassifierCV(pipeline(models[list(models)[winner]]),cv=sub,method='sigmoid',ensemble=True)
        c.fit(Xtrain[tr],ytrain[tr]);cal_inner[va]=c.predict_proba(Xtrain[va])[:,1]
    predictions['selected_sigmoid']=(cal_inner,calibrated.predict_proba(Xtest)[:,1])
    return predictions,dict(selected_model=list(models)[winner],candidate_auc=dict(zip(models,map(float,aucs)))),P,Q,folds
