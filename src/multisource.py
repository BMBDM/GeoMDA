import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .gfk import fit_gfk


def fixed_scale(x):
    x = np.asarray(x, dtype=float)
    return x / (1.0 + x)


def alpha_entropy(alpha, eps=1e-12):
    a = np.asarray(alpha, dtype=float)
    return float(-np.sum(a * np.log(a + eps)))


def effective_n_sources(alpha, eps=1e-12):
    return float(np.exp(alpha_entropy(alpha, eps=eps)))


def solve_alpha(dg, dm, dc, mu, C, beta=0.5, lambda_c=0.1, lambda_alpha=0.0, tau=0.1, eps=1e-12):
    """Solve adaptive source weights on the probability simplex.

    D_DA,k = (1-mu_k) delta_M,k + mu_k delta_C,k
    s(x)   = x/(1+x)
    R_k    = beta*s(D_G,k) + (1-beta)*s(D_DA,k)

    min_a  a^T R + lambda_c*a^T C a + lambda_alpha*||a||_2^2 + tau * sum_k a_k log(a_k + eps)
    s.t.   a_k >= 0, sum_k a_k = 1
    """
    dg = np.asarray(dg, float)
    dm = np.asarray(dm, float)
    dc = np.asarray(dc, float)
    mu = np.asarray(mu, float)
    C = np.asarray(C, float).copy()
    np.fill_diagonal(C, 0.0)

    dda = (1.0 - mu) * dm + mu * dc
    dg_scaled = fixed_scale(dg)
    dda_scaled = fixed_scale(dda)
    R = beta * dg_scaled + (1.0 - beta) * dda_scaled
    K = len(R)

    def objective(a):
        entropy_regularizer = np.sum(a * np.log(a + eps))
        return float(a @ R + lambda_c * (a @ C @ a)  + lambda_alpha * (a @ a) + tau * entropy_regularizer)

    res = minimize(
        objective,
        np.ones(K, dtype=float) / K,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * K,
        constraints={"type": "eq", "fun": lambda a: np.sum(a) - 1.0},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"alpha optimization failed: {res.message}")
    alpha = np.clip(res.x, 0.0, 1.0)
    alpha /= alpha.sum()
    H = alpha_entropy(alpha, eps=eps)
    Neff = effective_n_sources(alpha, eps=eps)
    return alpha, R, dda, dg_scaled, dda_scaled, float(res.fun), H, Neff


def _read_source_expression(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    meta = {"response", "logIC50"}
    genes = [c for c in df.columns if c not in meta]
    return df, genes


def source_source_redundancy(source_paths, dim, eta=1.0):
    """Pairwise source-source Grassmann distances and redundancy matrix.
    Distance is normalized by sqrt(dim), C_ij = exp(-eta * dG_norm_ij).
    """
    K = len(source_paths)
    D = np.zeros((K, K), dtype=float)
    cache = [_read_source_expression(p) for p in source_paths]
    for i in range(K):
        dfi, gi = cache[i]
        for j in range(i + 1, K):
            dfj, gj = cache[j]
            common = [g for g in gi if g in set(gj)]
            Xi = dfi[common].to_numpy(dtype=np.float64)
            Xj = dfj[common].to_numpy(dtype=np.float64)
            g = fit_gfk(Xi, Xj, dim)
            dnorm = float(np.linalg.norm(g["theta"]) / np.sqrt(len(g["theta"])))
            D[i, j] = D[j, i] = dnorm
    C = np.exp(-float(eta) * D)
    np.fill_diagonal(C, 0.0)
    return D, C
