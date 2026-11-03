import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_scad_pair
from src.gfk import fit_gfk
from src.ablation import train_variant, metrics_from_prob
from src.multisource import solve_alpha, source_source_redundancy


def geometry_stats(theta):
    a = np.asarray(theta, dtype=float)
    d = len(a); dg = float(np.linalg.norm(a))
    return {"grassmann_distance": dg, "dG_norm": float(dg/np.sqrt(d)),
            "theta_mean": float(a.mean()), "theta_std": float(a.std(ddof=0)),
            "theta_min": float(a.min()), "theta_max": float(a.max()),
            "theta_q25": float(np.quantile(a,.25)),
            "theta_median": float(np.median(a)),
            "theta_q75": float(np.quantile(a,.75))}


def last_proxy(hist):
    for h in reversed(hist):
        dm, dc = h.get("proxy_marginal", np.nan), h.get("proxy_conditional", np.nan)
        if np.isfinite(dm) and np.isfinite(dc):
            return float(dm), float(dc), float(h["mu"])
    return 0.0, 0.0, float(hist[-1]["mu"] if hist else 0.5)


def check_target(reference, current, task, source):
    if not np.array_equal(reference.target_ids, current.target_ids):
        raise ValueError(f"{task}/{source}: target cell IDs/order differ across sources")
    if not np.array_equal(reference.yt, current.yt):
        raise ValueError(f"{task}/{source}: target evaluation labels differ across sources")


def make_fixed_source_splits(tasks_single, outdir, n_sources=3, split_seed=0):
    required={"task_name","source_path","target_path"}
    if not required.issubset(tasks_single.columns):
        raise ValueError(f"single-source tasks CSV must contain {sorted(required)}")
    split_root=Path(outdir)/"source_splits"
    split_root.mkdir(parents=True,exist_ok=True)
    rows=[]; manifest=[]
    for _,r in tasks_single.iterrows():
        task=str(r.task_name); sp=Path(str(r.source_path)); tp=str(r.target_path)
        df=pd.read_csv(sp,sep="\t",index_col=0)
        if "response" not in df.columns:
            raise ValueError(f"{task}: source file has no response column: {sp}")
        # Fixed stratified split.  Separate RNG stream per task makes the split reproducible and independent of training seeds 0..9.
        rng=np.random.RandomState(int(split_seed))
        bins=[[] for _ in range(int(n_sources))]
        for cls in sorted(df["response"].unique()):
            idx=np.flatnonzero(df["response"].to_numpy()==cls)
            idx=rng.permutation(idx)
            for j,ii in enumerate(idx):
                bins[j % int(n_sources)].append(int(ii))
        tdir=split_root/task; tdir.mkdir(parents=True,exist_ok=True)
        for k,ids in enumerate(bins,1):
            ids=np.asarray(sorted(ids),dtype=int)
            part=df.iloc[ids].copy()
            outp=tdir/f"S{k}.tsv"
            part.to_csv(outp,sep="\t")
            counts=part["response"].value_counts().to_dict()
            rows.append({"task_name":task,"source_name":f"S{k}",
                         "source_path":str(outp),"target_path":tp})
            manifest.append({"task":task,"source":f"S{k}",
                             "split_seed":int(split_seed),"n_sources":int(n_sources),
                             "n_samples":len(part),
                             "n_response_0":int(counts.get(0,0)),
                             "n_response_1":int(counts.get(1,0)),
                             "original_source_path":str(sp),
                             "saved_source_path":str(outp)})
    split_tasks=pd.DataFrame(rows)
    split_tasks.to_csv(Path(outdir)/"multisource_tasks_generated.csv",index=False)
    pd.DataFrame(manifest).to_csv(Path(outdir)/"source_split_manifest.csv",index=False)
    return split_tasks


def main():
    ap = argparse.ArgumentParser(description="Optimized multi-source GFK-DDA")
    ap.add_argument("--tasks", default=str(Path(__file__).with_name("tasks.csv")),
                    help="Single-source task CSV: task_name,source_path,target_path")
    ap.add_argument("--n-sources", type=int, choices=[2,3], default=3,
                    help="Split each pan-cancer GDSC source into fixed S1..SK")
    ap.add_argument("--source-split-seed", type=int, default=0,
                    help="Fixed seed used only to construct S1..SK; independent of training seed")
    ap.add_argument("--outdir", default=str(ROOT/"results_multisource_optimized"))
    ap.add_argument("--dims", default="5,10,15,20,30,50")
    ap.add_argument("--lambdas", default="0,0.01,0.1,0.5,1,2")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mu-every", type=int, default=10)
    ap.add_argument("--sampling", choices=["weight","smote"], default="weight")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Trade-off in R_k between normalized geometry and normalized DDA discrepancy")
    ap.add_argument("--lambda-c", type=float, default=0.1,
                    help="Source-source redundancy penalty")
    ap.add_argument("--lambda-alpha", type=float, default=0.0,
                    help="Optional L2 regularization on alpha; revised default is 0")
    ap.add_argument("--tau", type=float, default=0.1,
                    help="Single entropy regularization strength (used when --taus is omitted)")
    ap.add_argument("--taus", default=None,
                    help="Optional comma-separated tau sensitivity values, e.g. 0,0.01,0.03,0.05,0.1. "
                         "Source models are trained only once per task/d/lambda/seed; alpha is then re-solved for each tau.")
    ap.add_argument("--task-name", default=None,
                    help="Optional exact task_name filter, e.g. Etoposide_PC9")
    ap.add_argument("--eta", type=float, default=1.0,
                    help="C_ij=exp(-eta*dG_source-source_norm)")
    args = ap.parse_args()

    dims=[int(x) for x in args.dims.split(",")]
    lambdas=[float(x) for x in args.lambdas.split(",")]
    seeds=[int(x) for x in args.seeds.split(",")]
    taus=([float(x) for x in args.taus.split(",")] if args.taus else [float(args.tau)])
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    single_tab=pd.read_csv(args.tasks)
    if args.task_name is not None:
        single_tab = single_tab[single_tab["task_name"].astype(str) == str(args.task_name)].copy()
        if single_tab.empty:
            raise ValueError(f"task_name not found in tasks CSV: {args.task_name}")
    tab=make_fixed_source_splits(single_tab,out,args.n_sources,args.source_split_seed)
    (out/"histories").mkdir(exist_ok=True)
    (out/"principal_angles").mkdir(exist_ok=True)

    all_rows=[]; src_rows=[]; ss_rows=[]

    for task, q in tab.groupby("task_name", sort=False):
        q=q.reset_index(drop=True)
        names=q.source_name.astype(str).tolist()
        paths=q.source_path.astype(str).tolist()
        target_paths=q.target_path.astype(str).unique().tolist()
        if len(target_paths)!=1:
            raise ValueError(f"{task}: all sources must point to the same target_path")
        pairs=[]
        for _,r in q.iterrows():
            pairs.append(load_scad_pair(r.source_path,r.target_path))
        for k in range(1,len(pairs)):
            check_target(pairs[0],pairs[k],task,names[k])
        yt=pairs[0].yt
        print(f"\n=== {task}: {len(pairs)} sources -> target n={len(yt)} ===")

        for dim in dims:
            Dss,C=source_source_redundancy(paths,dim,eta=args.eta)
            for i in range(len(names)):
                for j in range(i+1,len(names)):
                    ss_rows.append({"task":task,"dim":dim,"source_i":names[i],"source_j":names[j],
                                    "dG_ss_norm":Dss[i,j],"C_ij":C[i,j]})

            gfks=[]; geoms=[]
            for name,data in zip(names,pairs):
                g=fit_gfk(data.Xs,data.Xt,dim); gfks.append(g)
                geom=geometry_stats(g["theta"]); geoms.append(geom)
                pd.DataFrame({"angle_index":np.arange(1,len(g["theta"])+1),
                              "angle_rad":g["theta"],"angle_deg":np.degrees(g["theta"])}).to_csv(
                    out/"principal_angles"/f"{task}__{name}_d{dim}.csv",index=False)

            for lam in lambdas:
                for seed in seeds:
                    t0=time.time(); probs=[]; diag=[]
                    for name,data,g,geom in zip(names,pairs,gfks,geoms):
                        prob,met,hist,posthoc=train_variant(
                            g["Zs"],data.ys,g["Zt"],data.yt,
                            use_dda=(lam>0),epochs=args.epochs,batch_size=args.batch_size,
                            hidden=args.hidden,lr=args.lr,lambda_dda=lam,
                            mu_every=args.mu_every,seed=seed,sampling=args.sampling,
                            return_posthoc=True)
                        probs.append(prob)
                        # Unified final distribution state for alpha. It is computed even
                        # when lambda_dda=0 and never uses target response labels.
                        dm=float(posthoc["proxy_marginal"])
                        dc=float(posthoc["proxy_conditional"])
                        mu=float(posthoc["mu"])
                        diag.append((dm,dc,mu,met,hist))
                        hdir=out/"histories"/task/name; hdir.mkdir(parents=True,exist_ok=True)
                        pd.DataFrame(hist).to_csv(hdir/f"history_d{dim}_lambda{lam:g}_seed{seed}.csv",index=False)

                    dg=np.array([g["dG_norm"] for g in geoms])
                    dm=np.array([x[0] for x in diag]); dc=np.array([x[1] for x in diag]); mu=np.array([x[2] for x in diag])

                    elapsed_train=time.time()-t0
                    for tau in taus:
                        alpha,R,dda,dg_scaled,dda_scaled,obj,alpha_H,n_eff=solve_alpha(
                            dg,dm,dc,mu,C,beta=args.beta,lambda_c=args.lambda_c,
                            lambda_alpha=args.lambda_alpha,tau=tau)
                        fused=np.sum(np.vstack(probs)*alpha[:,None],axis=0)
                        met=metrics_from_prob(yt,fused)
                        all_rows.append({"task":task,"dim":dim,"lambda_dda":lam,"seed":seed,
                                         "sampling":args.sampling,"n_sources":len(names),
                                         "source_split_seed":args.source_split_seed,
                                         "beta":args.beta,"lambda_c":args.lambda_c,
                                         "lambda_alpha":args.lambda_alpha,"tau":tau,"eta":args.eta,
                                         "alpha_objective":obj,"alpha_entropy":alpha_H,
                                         "effective_n_sources":n_eff,"seconds":elapsed_train,**met})
                        for k,name in enumerate(names):
                            src_rows.append({"task":task,"dim":dim,"lambda_dda":lam,"seed":seed,
                                             "sampling":args.sampling,"source":name,
                                             "n_sources":len(names),"source_split_seed":args.source_split_seed,
                                             "beta":args.beta,"lambda_c":args.lambda_c,
                                             "lambda_alpha":args.lambda_alpha,"tau":tau,"eta":args.eta,
                                             **geoms[k],"D_M":dm[k],"D_C":dc[k],"mu":mu[k],
                                             "D_DA":dda[k],"dG_scaled":dg_scaled[k],
                                             "D_DA_scaled":dda_scaled[k],"R_k":R[k],"alpha":alpha[k],
                                             "alpha_entropy":alpha_H,"effective_n_sources":n_eff,
                                             "source_AUROC":diag[k][3]["AUROC"],"source_AUPR":diag[k][3]["AUPR"]})
                        print(task,"d",dim,"lambda",lam,"seed",seed,"tau",tau,
                              "alpha",np.round(alpha,4),"Neff",round(n_eff,4),
                              "AUROC",round(met["AUROC"],4))
                    pd.DataFrame(all_rows).to_csv(out/"multisource_all_runs.csv",index=False)
                    pd.DataFrame(src_rows).to_csv(out/"source_diagnostics.csv",index=False)

    df=pd.DataFrame(all_rows); sdf=pd.DataFrame(src_rows); ssdf=pd.DataFrame(ss_rows).drop_duplicates()
    df.to_csv(out/"multisource_all_runs.csv",index=False)
    sdf.to_csv(out/"source_diagnostics.csv",index=False)
    ssdf.to_csv(out/"source_source_geometry.csv",index=False)
    metrics=["AUROC","AUPR","ACC","F1"]
    df.groupby(["task","dim","lambda_dda","sampling","tau"])[metrics].agg(["mean","std"]).reset_index().to_csv(
        out/"performance_summary.csv",index=False)
    sdf.groupby(["task","dim","lambda_dda","sampling","tau","source"])[["alpha","R_k","D_DA","D_M","D_C","mu","effective_n_sources"]].agg(["mean","std"]).reset_index().to_csv(
        out/"alpha_summary.csv",index=False)
    print("\nFinished:",out)

if __name__=="__main__": main()
