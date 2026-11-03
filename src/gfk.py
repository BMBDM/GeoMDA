import numpy as np
from sklearn.decomposition import PCA

def _orth(X):
    q, _ = np.linalg.qr(X)
    return q

def fit_gfk(Xs, Xt, dim=20, center=True, eps=1e-10):
    Xs = np.asarray(Xs, float)
    Xt = np.asarray(Xt, float)
    ns, p = Xs.shape
    nt, p2 = Xt.shape

    # Domain-specific centering, matching the PCA idea in original GFK/MEDA.
    ms = Xs.mean(0, keepdims=True) if center else np.zeros((1,p))
    mt = Xt.mean(0, keepdims=True) if center else np.zeros((1,p))
    Xsc, Xtc = Xs-ms, Xt-mt

    Ps = PCA(n_components=dim, svd_solver="randomized", random_state=0).fit(Xsc).components_.T
    Pt = PCA(n_components=dim, svd_solver="randomized", random_state=0).fit(Xtc).components_.T
    Ps, Pt = _orth(Ps), _orth(Pt)

    # Principal vectors: A^T B = diag(cos(theta))
    U, s, Vt = np.linalg.svd(Ps.T @ Pt, full_matrices=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)
    A = Ps @ U
    B = Pt @ Vt.T

    # Orthogonal directions C satisfying B = A cos(theta) + C sin(theta).
    C = np.zeros_like(A)
    for j, th in enumerate(theta):
        st = np.sin(th)
        if st > 1e-7:
            C[:, j] = (B[:, j] - A[:, j] * np.cos(th)) / st
        else:
            C[:, j] = 0.0

    Q = np.concatenate([A, C], axis=1)  # p x 2d
    H = np.zeros((2*dim, 2*dim), dtype=float)
    for j, th in enumerate(theta):
        if abs(th) < 1e-8:
            cc, ss, cs = 1.0, 0.0, 0.0
        else:
            cc = 0.5 + np.sin(2*th)/(4*th)
            ss = 0.5 - np.sin(2*th)/(4*th)
            cs = (1-np.cos(2*th))/(4*th)
        H[j,j] = cc
        H[dim+j,dim+j] = ss
        H[j,dim+j] = H[dim+j,j] = cs

    # G = Q H Q^T.  Eigendecompose only the small 2d x 2d representation after QR because C can be rank deficient for theta~0.
    Qr, R = np.linalg.qr(Q, mode="reduced")
    small = R @ H @ R.T
    small = (small + small.T)/2
    w, V = np.linalg.eigh(small)
    keep = w > eps
    fmap = Qr @ V[:, keep] @ np.diag(np.sqrt(w[keep]))  # p x r_eff

    # Apply to ORIGINAL z-scored expression values. Translation is not part of the GFK kernel; centering above is only for estimating subspaces.
    Zs = Xs @ fmap
    Zt = Xt @ fmap
    dG = float(np.linalg.norm(theta))
    return {"Zs": Zs.astype(np.float32), "Zt": Zt.astype(np.float32), 
            "feature_map": fmap, "theta": theta, "grassmann_distance": dG, "dim": dim,}
