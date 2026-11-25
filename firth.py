"""Firth penalized-likelihood logistic regression (Jeffreys prior), robust with step-halving."""
import numpy as np
from scipy import stats

def _ploglik(X, y, beta):
    eta = np.clip(X @ beta, -35, 35)
    pr = 1.0 / (1.0 + np.exp(-eta))
    ll = np.sum(y*np.log(pr+1e-12) + (1-y)*np.log(1-pr+1e-12))
    W = pr*(1-pr); I = (X.T*W)@X
    sign, logdet = np.linalg.slogdet(I)
    return ll + 0.5*(logdet if sign>0 else -1e6)

def _fit(X, y, max_iter=200, tol=1e-8):
    n,p = X.shape; beta = np.zeros(p)
    for _ in range(max_iter):
        eta = np.clip(X@beta,-35,35); pr = 1/(1+np.exp(-eta)); W = pr*(1-pr)
        XtW = X.T*W; I = XtW@X
        try: Iinv = np.linalg.inv(I)
        except np.linalg.LinAlgError: Iinv = np.linalg.pinv(I)
        H = (X@Iinv)*XtW.T; h = H.sum(1)
        U = X.T@(y-pr+h*(0.5-pr)); step = Iinv@U
        base=_ploglik(X,y,beta); f=1.0; bn=beta+step
        while f>1e-6 and _ploglik(X,y,bn)<base: f*=0.5; bn=beta+f*step
        beta=bn
        if np.max(np.abs(f*step))<tol: break
    eta=np.clip(X@beta,-35,35); pr=1/(1+np.exp(-eta)); W=pr*(1-pr); I=(X.T*W)@X
    try: cov=np.linalg.inv(I)
    except np.linalg.LinAlgError: cov=np.linalg.pinv(I)
    return beta, np.sqrt(np.diag(cov))

def firth_logit(X, y):
    X=np.asarray(X,float); y=np.asarray(y,float)
    Xi=np.column_stack([np.ones(len(y)),X])
    beta,se=_fit(Xi,y); z=beta/se
    return beta, se, 2*stats.norm.sf(np.abs(z))

def or_ci(beta, se, idx, sign=1.0, scale=1.0):
    b=sign*scale*beta[idx]; s=scale*se[idx]
    return (np.exp(b), np.exp(b-1.96*s), np.exp(b+1.96*s), 2*stats.norm.sf(abs(beta[idx]/se[idx])))
