from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class DrugData:
    Xs: np.ndarray
    ys: np.ndarray
    Xt: np.ndarray
    yt: np.ndarray
    genes: list
    source_ids: np.ndarray
    target_ids: np.ndarray

def load_scad_pair(source_path, target_path):
    s = pd.read_csv(source_path, sep="\t", index_col=0)
    t = pd.read_csv(target_path, sep="\t", index_col=0)
    s_meta = {"response", "logIC50"}
    t_meta = {"response"}
    s_genes = [c for c in s.columns if c not in s_meta]
    t_genes = [c for c in t.columns if c not in t_meta]
    if s_genes != t_genes:
        common = [g for g in s_genes if g in set(t_genes)]
        s_genes = common
        t_genes = common

    Xs = s[s_genes].to_numpy(dtype=np.float64)
    Xt = t[t_genes].to_numpy(dtype=np.float64)
    ys = s["response"].to_numpy(dtype=np.int64)
    yt = t["response"].to_numpy(dtype=np.int64)

    return DrugData(Xs, ys, Xt, yt, s_genes, s.index.to_numpy(), t.index.to_numpy())
