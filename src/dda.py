import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

def gaussian_kernel(x, y, sigmas=(0.5,1,2,4,8)):
    d2 = torch.cdist(x, y, p=2).pow(2)
    out = 0.0
    for s in sigmas:
        out = out + torch.exp(-d2/(2.0*s*s))
    return out / len(sigmas)

def mmd2(x, y):
    return gaussian_kernel(x,x).mean() + gaussian_kernel(y,y).mean() - 2*gaussian_kernel(x,y).mean()

def conditional_mmd2(xs, xt, ys, pt, n_classes=2):
    """Soft target conditional MMD. pt is target class probability."""
    loss = xs.new_tensor(0.0)
    used = 0
    for c in range(n_classes):
        ws = (ys == c).float()
        wt = pt[:, c]
        if ws.sum() < 2 or wt.sum() < 1e-6:
            continue
        ws = ws / (ws.sum() + 1e-8)
        wt = wt / (wt.sum() + 1e-8)
        Kss = gaussian_kernel(xs,xs)
        Ktt = gaussian_kernel(xt,xt)
        Kst = gaussian_kernel(xs,xt)
        loss = loss + ws @ Kss @ ws + wt @ Ktt @ wt - 2*ws @ Kst @ wt
        used += 1
    return loss / max(used, 1)

def proxy_a_distance(Xs, Xt, seed=0):
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs), dtype=int), np.ones(len(Xt), dtype=int)]
    counts = np.bincount(y)
    nfold = int(min(5, counts.min()))
    if nfold < 2:
        return 0.0
    cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    clf = LinearSVC(C=1.0, dual="auto", max_iter=5000)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    err = min(acc, 1-acc)  # symmetric domain-label orientation
    return float(2*(1-2*err))

def estimate_mu(Xs, ys, Xt, pseudo, seed=0):
    dm = proxy_a_distance(Xs, Xt, seed)
    dc = []
    for c in np.unique(ys):
        a = Xs[ys == c]
        b = Xt[pseudo == c]
        if len(a) >= 4 and len(b) >= 4:
            dc.append(proxy_a_distance(a, b, seed))
    dcond = float(np.mean(dc)) if dc else 0.0
    if dm + dcond < 1e-12:
        return 0.5, dm, dcond
    return dcond/(dm+dcond), dm, dcond
