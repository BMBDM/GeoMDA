import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix
from .model import Predictor
from .dda import mmd2, conditional_mmd2, estimate_mu

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def weighted_indices(y, batch, rng):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    class_weight = {c: 1.0 / n for c, n in zip(classes, counts)}
    sample_weight = np.asarray([class_weight[v] for v in y], dtype=float)
    prob = sample_weight / sample_weight.sum()
    return rng.choice(len(y), size=batch, replace=True, p=prob)

def smote_resample(X, y, rng, k_neighbors=5):
    """Binary/multiclass SMOTE to the size of the largest class.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target_n = int(counts.max())
    xs, ys = [X], [y]

    for c, n_c in zip(classes, counts):
        n_new = target_n - int(n_c)
        if n_new <= 0:
            continue
        idx = np.where(y == c)[0]
        Xc = X[idx]
        if len(Xc) < 2:
            # Degenerate fallback: duplicate the only available minority sample.
            synth = np.repeat(Xc, n_new, axis=0)
        else:
            k = min(int(k_neighbors), len(Xc) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
            neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
            synth = np.empty((n_new, X.shape[1]), dtype=np.float32)
            for j in range(n_new):
                i = int(rng.integers(len(Xc)))
                nb = int(rng.choice(neigh[i]))
                alpha = float(rng.random())
                synth[j] = Xc[i] + alpha * (Xc[nb] - Xc[i])
        xs.append(synth)
        ys.append(np.full(n_new, c, dtype=y.dtype))

    Xr = np.concatenate(xs, axis=0)
    yr = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(yr))
    return Xr[perm], yr[perm]

def metrics_from_prob(y, p):
    pred=(p>=0.5).astype(int)
    return {
        "AUROC":float(roc_auc_score(y,p)),
        "AUPR":float(average_precision_score(y,p)),
        "ACC":float(accuracy_score(y,pred)),
        "F1":float(f1_score(y,pred,zero_division=0)),
        "TN":int(confusion_matrix(y,pred,labels=[0,1])[0,0]),
        "FP":int(confusion_matrix(y,pred,labels=[0,1])[0,1]),
        "FN":int(confusion_matrix(y,pred,labels=[0,1])[1,0]),
        "TP":int(confusion_matrix(y,pred,labels=[0,1])[1,1]),
    }

def train_variant(Xs, ys, Xt, yt_eval, use_dda=False, epochs=100, batch_size=128,
                  hidden=64, lr=1e-3, lambda_dda=1.0, mu_every=10, seed=0, device=None,
                  sampling="weight", return_posthoc=False):
    """Train source-only predictor or predictor+DDA."""
    set_seed(seed)
    rng=np.random.default_rng(seed)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    sampling = str(sampling).lower()
    if sampling not in {"weight", "smote"}:
        raise ValueError("sampling must be 'weight' or 'smote'")

    if sampling == "smote":
        Xs_train, ys_train = smote_resample(Xs, ys, rng)
    else:
        Xs_train, ys_train = np.asarray(Xs), np.asarray(ys)

    model=Predictor(Xs.shape[1],hidden).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    XS=torch.tensor(Xs_train,dtype=torch.float32,device=device)
    YS=torch.tensor(ys_train,dtype=torch.long,device=device)
    XT=torch.tensor(Xt,dtype=torch.float32,device=device)
    mu=0.0
    hist=[]
    steps=max(1,int(np.ceil(max(len(Xs_train),len(Xt))/batch_size)))

    for epoch in range(1,epochs+1):
        model.train(); vals=[]
        for _ in range(steps):
            source_batch = min(batch_size, len(Xs_train))
            if sampling == "weight":
                si = weighted_indices(ys_train, source_batch, rng)
            else:
                si = rng.choice(len(Xs_train), source_batch, replace=len(Xs_train) < source_batch)
            xs,yb=XS[si],YS[si]
            hs,ls=model(xs)
            lcls=F.cross_entropy(ls,yb)
            if use_dda:
                ti=rng.choice(len(Xt),min(batch_size,len(Xt)),replace=len(Xt)<batch_size)
                xt=XT[ti]
                ht,lt=model(xt)
                pt=torch.softmax(lt.detach(),dim=1)
                lm=mmd2(hs,ht)
                lc=conditional_mmd2(hs,ht,yb,pt)
                ldda=(1-mu)*lm+mu*lc
                loss=lcls+lambda_dda*ldda
            else:
                lm=lc=ldda=xs.new_tensor(0.)
                loss=lcls
            opt.zero_grad(); loss.backward(); opt.step()
            vals.append([loss.item(),lcls.item(),lm.item(),lc.item()])

        dm=dc=np.nan
        if use_dda and (epoch==1 or epoch%mu_every==0):
            model.eval()
            with torch.no_grad():
                hs,_=model(XS); ht,lt=model(XT)
                pseudo=lt.argmax(1).cpu().numpy()
            mu,dm,dc=estimate_mu(hs.cpu().numpy(),ys_train,ht.cpu().numpy(),pseudo,seed)
        a=np.mean(vals,axis=0)
        hist.append({"epoch":epoch, "loss":a[0], "cls":a[1], "marginal":a[2],
                     "conditional":a[3], "mu":mu, "proxy_marginal":dm, "proxy_conditional":dc})

    model.eval()
    XS_DIAG=torch.tensor(np.asarray(Xs),dtype=torch.float32,device=device)
    with torch.no_grad():
        hs_diag,_=model(XS_DIAG)
        ht_diag,logits=model(XT)
        prob=torch.softmax(logits,1)[:,1].cpu().numpy()
        pseudo=logits.argmax(1).cpu().numpy()
    mu_final,dm_final,dc_final=estimate_mu(
        hs_diag.cpu().numpy(),np.asarray(ys),ht_diag.cpu().numpy(),pseudo,seed)
    posthoc={"mu":float(mu_final),"proxy_marginal":float(dm_final),
             "proxy_conditional":float(dc_final)}
    result=(prob,metrics_from_prob(yt_eval,prob),hist)
    if return_posthoc:
        return result + (posthoc,)
    return result
