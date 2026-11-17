"""Independent audit and publication tables for the supplied completed study.
Run from the extracted review bundle: python audit_results.py --input evidence --out revised
No calibration is rerun and no uploaded code is executed.
"""
from pathlib import Path
from itertools import product
import argparse, hashlib, json, shutil, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL={"expm":"Exact expm","fd":"FD + adjoint","lchs-beta":"LCHS beta","lchs-simple":"LCHS simple",
       "AH":"Andreasen--Huge","PINN-fixed":"PINN fixed","PINN-Wam":"PINN adaptive"}
ORDER=list(LABEL)

def read_json(path): return json.loads(Path(path).read_text())
def write_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
def interp(K,query,C): return np.interp(np.asarray(query),K,np.asarray(C))

def audit_run(row,result,data):
    N=int(row["N"]); K=np.exp(np.linspace(-1.2,1.2,N+2)); C=np.asarray(result["C"])
    assert C.shape==(2,N+2) and np.isfinite(C).all()
    quote_res=[]; test_res=[]; model_tests=[]; reference_tests=[]
    for s,(q,test) in enumerate(zip(data["quotes"],data["tests"])):
        train_k=np.asarray(q["k"]); test_k=np.asarray(test["k"])
        assert not np.any(np.isclose(test_k[:,None],train_k[None,:],rtol=0,atol=1e-10))
        qp=np.asarray(result["quote_predictions"][s]) if "quote_predictions" in result else interp(K,train_k,C[s])
        hp=np.asarray(result["test_predictions"][s]) if "test_predictions" in result else interp(K,test_k,C[s])
        quote_res.extend(qp-np.asarray(q["mid"])); test_res.extend(hp-np.asarray(test["price"]))
        model_tests.extend(hp); reference_tests.extend(test["price"])
    dm=np.diff(K)[:-1]; dp=np.diff(K)[1:]
    b=(dm*C[:,2:]+dp*C[:,:-2])/(dm+dp)-C[:,1:-1]
    cal=np.diff(np.vstack([np.maximum(1-K,0),C]),axis=0)
    band=(K[:-2]>=.7)&(K[2:]<=1.4); calband=(K>=.7)&(K<=1.4)
    audit=dict(rmse_quotes=float(np.sqrt(np.mean(np.square(quote_res)))),
        rmse_heldout=float(np.sqrt(np.mean(np.square(test_res)))),
        butterfly_violation=float(max(0,-b.min())),calendar_violation=float(max(0,-cal.min())),
        density_violation=float(max(0,-(2*b/(dm*dp)).min())),
        bfly_count=int((b < -1e-8).sum()),cal_count=int((cal < -1e-8).sum()),
        band_bfly_count=int((b[:,band] < -1e-8).sum()),band_cal_count=int((cal[:,calband] < -1e-8).sum()),
        band_bfly_violation=float(max(0,-b[:,band].min())),band_cal_violation=float(max(0,-cal[:,calband].min())),
        invalid_model_price_count=int((~np.isfinite(model_tests)).sum()),
        maximum_metric_discrepancy=0.)
    for name in ["rmse_quotes","rmse_heldout","butterfly_violation","calendar_violation","density_violation","bfly_count","cal_count"]:
        discrepancy=abs(audit[name]-float(row[name]))
        audit["maximum_metric_discrepancy"]=max(audit["maximum_metric_discrepancy"],discrepancy)
        assert np.isclose(audit[name],row[name],rtol=2e-8,atol=2e-11),(name,audit[name],row[name])
    if "C_exact" in result:
        exact=np.asarray(result["C_exact"]); delta=np.max(np.abs(C-exact),axis=1)
        be=(dm*exact[:,2:]+dp*exact[:,:-2])/(dm+dp)-exact[:,1:-1]
        ce=np.diff(np.vstack([np.maximum(1-K,0),exact]),axis=0)
        audit["exact_bf_violation"]=float(max(0,-be.min()))
        audit["exact_cal_violation"]=float(max(0,-ce.min()))
        assert audit["exact_bf_violation"]<1e-8 and audit["exact_cal_violation"]<1e-8
        cb=delta+np.r_[0,delta[:-1]]
        assert np.all(b>=-2*delta[:,None]-2e-10)
        assert np.all(cal>=-cb[:,None]-2e-10)
        audit["conditional_certificate_pass"]=True
        audit["measured_price_error_inf"]=float(delta.max())
    return audit

def figure(fig,out,name,source,caption):
    for ax in fig.axes:
        if ax.get_label()!="<colorbar>": ax.grid(alpha=.15,linewidth=.5)
    for suffix in ["pdf","png"]:
        stream=io.BytesIO()
        fig.savefig(stream,format=suffix,bbox_inches="tight",pad_inches=.05,dpi=240)
        content=stream.getvalue()
        if suffix=="pdf": assert content.rstrip().endswith(b"%%EOF")
        destination=out/"figures"/(name+"."+suffix)
        temporary=destination.with_suffix(destination.suffix+".tmp")
        temporary.write_bytes(content);temporary.replace(destination)
    write_json(out/"figures"/(name+".json"),dict(source=source,caption=caption))
    plt.close(fig)

def run(root,out):
    root=Path(root); out=Path(out)
    for name in ["tables","figures","audit"]: (out/name).mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(root/"results.csv"); m=read_json(root/"manifest.json"); cfg=m["config"]
    assert cfg["profile"]=="paper" and cfg["fingerprint"]=="9684a7f0bec7", "This review is specific to the supplied paper run."
    hashes={n:hashlib.sha256((root/"source"/n).read_bytes()).hexdigest()==h for n,h in m["source_sha256"].items()}
    assert all(hashes.values())
    expected=set()
    for truth,noise in product(["in","out"],[0.,5.]):
        for ds in ([101] if noise==0 else cfg["noise_seeds"]):
            dataset=f"{truth}_bp{noise:g}_d{ds}"
            for n in cfg["ns"]:
                methods=ORDER if n==6 else ORDER[:4]
                for method in methods:
                    for seed in ([0] if method=="AH" else cfg["seeds"]): expected.add((dataset,n,method,seed))
    observed=set(zip(d.dataset,d.n,d.method,d.seed)); assert len(observed)==len(d)
    failed=read_json(root/"failures.json")
    failures={(f["dataset"],int(f["key"].split("_n")[-1].split("_")[0]),f["method"],int(f["key"].rsplit("_s",1)[1])) for f in failed}
    assert not observed.intersection(failures) and expected==observed|failures
    raw={f.stem:read_json(f) for f in (root/"raw").glob("*.json")}
    records=[]
    for row in d.to_dict("records"):
        key=f'{row["dataset"]}_n{row["n"]}_{row["method"]}_s{row["seed"]}'
        payload=read_json(root/"checkpoints"/(key+".json"))
        for name in ["rmse_quotes","rmse_heldout","opt_s","compile_warm_s"]:
            assert np.isclose(row[name],payload["row"][name],rtol=1e-12,atol=1e-12)
        check=audit_run(row,payload["result"],raw[row["dataset"]])
        records.append(dict(key=key,dataset=row["dataset"],method=row["method"],n=row["n"],seed=row["seed"],**check))
    checks=pd.DataFrame(records); checks.to_csv(out/"audit"/"independent_checks.csv",index=False)
    # Audit outcomes are statements about nodal tests on their stated domains.
    arb=checks.groupby(["method","n"]).agg(runs=("key","size"),bf_fail=("bfly_count",lambda x:int((x>0).sum())),
         cal_fail=("cal_count",lambda x:int((x>0).sum())),bf_max=("butterfly_violation","max"),cal_max=("calendar_violation","max"),
         band_bf_fail=("band_bfly_count",lambda x:int((x>0).sum())),band_cal_fail=("band_cal_count",lambda x:int((x>0).sum())),
         band_bf_max=("band_bfly_violation","max"),band_cal_max=("band_cal_violation","max")).reset_index()
    arb.to_csv(out/"audit"/"arbitrage_by_method_grid.csv",index=False)
    ds=d.groupby(["truth","noise_bp","n","method","dataset"],as_index=False).agg(
         heldout=("rmse_heldout","mean"),quote=("rmse_quotes","mean"),opt_s=("opt_s","mean"),warm_s=("compile_warm_s","mean"),runs=("seed","size"))
    summary=ds.groupby(["truth","noise_bp","n","method"]).agg(heldout=("heldout","mean"),heldout_sd=("heldout","std"),
         quote=("quote","mean"),opt_s=("opt_s","mean"),warm_s=("warm_s","mean"),datasets=("dataset","size"),runs=("runs","sum")).reset_index()
    ds.to_csv(out/"audit"/"dataset_means_verified.csv",index=False); summary.to_csv(out/"audit"/"summary_verified.csv",index=False)
    iv=d.groupby(["method","n"]).agg(valid=("iv_valid","sum"),total=("iv_total","sum")).reset_index()
    iv["invalid"]=iv.total-iv.valid; iv.to_csv(out/"audit"/"iv_coverage.csv",index=False)
    b=pd.read_csv(root/"error_budget.csv"); assert len(b)==90 and b.certificate_pass.all()
    # A quadrature certificate is conditional and is not an arbitrage pass.
    b.to_csv(out/"audit"/"kernel_budget_source.csv",index=False)
    q=read_json(root/"quantum_resources.json"); qo=pd.read_csv(root/"quantum_objective.csv")
    oracle=float(np.max(np.abs(qo.circuit_block_real+1j*qo.circuit_block_imag-qo.normalised_loss)))
    assert abs(oracle-q["objective_block_error"])<1e-14
    outcase=summary[(summary.n==6)&(summary.truth=="out")&(summary.noise_bp==5)].set_index("method")
    stats=dict(planned_runs=len(expected),completed_runs=len(d),failed_runs=len(failed),source_hashes_match=hashes,
          unique_run_keys=True,all_expected_runs_accounted=True,
          maximum_recomputed_metric_discrepancy=float(checks.maximum_metric_discrepancy.max()),
          conditional_calibration_certificates=int((checks.conditional_certificate_pass==True).sum()),
          lchs_conditional_calibration_certificates=int(((checks.conditional_certificate_pass==True)&checks.method.str.startswith("lchs")).sum()),
          exact_re_evaluation_bf_max=float(checks.loc[checks.method.str.startswith("lchs"),"exact_bf_violation"].max()),
          exact_re_evaluation_cal_max=float(checks.loc[checks.method.str.startswith("lchs"),"exact_cal_violation"].max()),
          forward_budget_certificates=int(b.certificate_pass.sum()),
          lchs_beta_raw_bf_fail=int(((checks.method=="lchs-beta")&(checks.bfly_count>0)).sum()),
          lchs_beta_raw_cal_fail=int(((checks.method=="lchs-beta")&(checks.cal_count>0)).sum()),
          classical_beta_to_fd_opt_time_ratio=float(outcase.loc["lchs-beta","opt_s"]/outcase.loc["fd","opt_s"]),
          classical_beta_to_expm_opt_time_ratio=float(outcase.loc["lchs-beta","opt_s"]/outcase.loc["expm","opt_s"]),
          oracle_block_error=oracle,market_status=read_json(root/"market_status.json")["status"],
          scope="Independent audit of supplied results; original optimisation was not rerun")
    write_json(out/"audit"/"audit_summary.json",stats)
    # Main comparison: all seven methods, three noise datasets, dataset-level SD.
    rows=[r"\begin{table}[t]\centering\caption{Out-of-family, 5 bp noise, $N=64$. Price errors are normalised bp. Test: mean $\pm$ sample SD across three datasets after averaging initialisations. Time is mean classical optimisation seconds.}",
          r"\label{tab:bench}\scriptsize\begin{tabular}{lrrrr}\toprule Method & Fit & Test & Time & Runs\\\midrule"]
    for method in ORDER:
        r=outcase.loc[method]
        rows.append(f"{LABEL[method]} & {r.quote*1e4:.2f} & ${r.heldout*1e4:.2f}\\pm{r.heldout_sd*1e4:.2f}$ & {r.opt_s:.3g} & {int(r.runs)}"+r"\\")
    rows.append(r"\bottomrule\end{tabular}\end{table}"); (out/"tables"/"benchmark.tex").write_text("\n".join(rows))
    rows=[r"\begin{table}[t]\centering\caption{Raw beta-LCHS audits at price tolerance $10^{-8}$. Full-domain violation maxima are normalised bp. The last column counts butterfly-failing runs within $[0.7,1.4]$.}",
          r"\label{tab:arb}\scriptsize\begin{tabular}{rrrrrr}\toprule $N$ & BF fail & Cal. fail & BF max & Cal. max & Band fail\\\midrule"]
    for r in arb[arb.method=="lchs-beta"].itertuples():
        rows.append(f"{2**r.n} & {r.bf_fail}/{r.runs} & {r.cal_fail}/{r.runs} & {r.bf_max*1e4:.4f} & {r.cal_max*1e4:.4f} & {r.band_bf_fail}/{r.runs}"+r"\\")
    rows.append(r"\bottomrule\end{tabular}\end{table}");(out/"tables"/"arbitrage_audit.tex").write_text("\n".join(rows))
    # Full-case appendix table prevents one-case selection from hiding results.
    rows=[r"\begin{table}[htbp]\centering\caption{Held-out price RMSE at $N=64$ (normalised bp). Noisy columns average initialisations within each dataset, then datasets. The in-family AH noisy entry uses only two valid datasets.}\small",
          r"\begin{tabular}{lrrrr}\toprule Method & In, clean & In, noisy & Out, clean & Out, noisy\\\midrule"]
    for method in ORDER:
        values=[]
        for truth,noise in [("in",0),("in",5),("out",0),("out",5)]:
            r=summary[(summary.n==6)&(summary.method==method)&(summary.truth==truth)&(summary.noise_bp==noise)].iloc[0]
            values.append(f"{1e4*r.heldout:.3f}"+(f" $\\pm$ {1e4*r.heldout_sd:.3f}" if noise else ""))
        rows.append(LABEL[method]+" & "+" & ".join(values)+r"\\")
    rows.append(r"\bottomrule\end{tabular}\end{table}");(out/"tables"/"all_cases.tex").write_text("\n".join(rows))
    # Pair the two feasible noisy in-family datasets for all seven methods.
    paired=ds[(ds.n==6)&(ds.truth=="in")&(ds.noise_bp==5)&ds.dataset.isin(["in_bp5_d101","in_bp5_d202"])]
    paired=paired.groupby("method").agg(mean=("heldout","mean"),sd=("heldout","std"),datasets=("dataset","size"))
    paired.to_csv(out/"audit"/"in_noisy_complete_case.csv")
    rows=[r"\begin{table}[htbp]\centering\caption{Paired feasible in-family noisy datasets (101 and 202), $N=64$. Held-out RMSE in normalised bp; sample SD over two datasets. Dataset 303 remains a separately reported failed-input stress case.}\small",
          r"\begin{tabular}{lrr}\toprule Method & Mean & Dataset SD\\\midrule"]
    for method in ORDER:
        r=paired.loc[method];rows.append(f"{LABEL[method]} & {r['mean']*1e4:.3f} & {r.sd*1e4:.3f}"+r"\\")
    rows.append(r"\bottomrule\end{tabular}\end{table}");(out/"tables"/"paired_in_family.tex").write_text("\n".join(rows))
    # Scientific figure contracts: explicit method identities, two-root palette,
    # neutral baselines, published units and all specified dataset replicates.
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8,"legend.fontsize":7,
        "xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(3.5,2.25),layout="constrained")
    for i,method in enumerate(ORDER):
        r=outcase.loc[method]; color="#CB702C" if method=="lchs-beta" else "#236C9C" if method=="lchs-simple" else "#59636B"
        ax.errorbar(r.heldout*1e4,i,xerr=r.heldout_sd*1e4,fmt="o",color=color,capsize=2,ms=4)
    ax.set_yticks(range(len(ORDER)),[LABEL[x].replace("--","-") for x in ORDER]);ax.invert_yaxis()
    ax.set(xlabel="Held-out RMSE (normalised bp)",title="N=64 / out-of-family / 5 bp noise",xlim=(0,None))
    figure(fig,out,"heldout_comparison",["results.csv","audit/dataset_means_verified.csv"],"Mean and sample SD over three data seeds after averaging five initialisations (AH deterministic).")
    fig,axs=plt.subplots(1,2,figsize=(3.5,2.1),layout="constrained")
    for n,marker in [(5,"o"),(6,"s"),(7,"^")]:
        p=b[(b.n==n)&(b.kernel=="beta")]
        for cutoff,linestyle in [(16.,":"),(32.,"--"),(64.,"-")]:
            v=p[p.cutoff==cutoff].sort_values("terms")
            if n==7: axs[0].loglog(v.terms,v.price_error_inf,marker=marker,ms=3,ls=linestyle,color="#236C9C",label=f"K={cutoff:g}")
        v=p[p.cutoff==64.].sort_values("terms")
        axs[1].loglog(v.terms,v.price_error_inf,marker=marker,ms=3,color={5:"#60676C",6:"#236C9C",7:"#CB702C"}[n],label=f"N={2**n}")
    from matplotlib.ticker import NullFormatter
    for ax in axs:
        ax.set_xticks([256,1024,4096],["256","1024","4096"]);ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xlabel("LCU terms J");ax.legend()
    axs[0].set(title="N=128, beta kernel",ylabel="Max price error");axs[1].set(title="Cutoff K=64",ylabel="Max price error")
    figure(fig,out,"kernel_resolution",["error_budget.csv"],"All beta-kernel cutoffs at N=128 and fixed-cutoff spatial-grid comparison. Designs differ across spatial grids, as in the input protocol.")
    # Example selected by a fixed source key, not the final loop iteration.
    key="in_bp0_d101_n7_lchs-beta_s0"; r=read_json(root/"checkpoints"/(key+".json"))["result"]
    K=np.exp(np.linspace(-1.2,1.2,130)); C=np.asarray(r["C"]); E=np.asarray(r["C_exact"])
    dm=np.diff(K)[:-1];dp=np.diff(K)[1:]
    bf=lambda A:((dm*A[:,2:]+dp*A[:,:-2])/(dm+dp)-A[:,1:-1])
    fig,axs=plt.subplots(2,1,figsize=(3.5,2.8),layout="constrained",sharex=True)
    axs[0].plot(K,(C[1]-E[1])*1e4,color="#CB702C",lw=.9);axs[0].axhline(0,color=".5",lw=.7)
    axs[0].set(ylabel="Raw minus exact (bp)",title="N=128 / in-family / no noise / seed 0 / T=1")
    axs[1].plot(K[1:-1],bf(C)[1]*1e4,color="#CB702C",lw=.9,label="Raw beta LCHS")
    axs[1].plot(K[1:-1],bf(E)[1]*1e4,color="#59636B",ls="--",lw=.9,label="Exact re-evaluation")
    axs[1].axhline(0,color=".5",lw=.7);axs[1].set(xlabel="Normalised strike k",ylabel="Price butterfly (bp)");axs[1].legend()
    for ax in axs: ax.axvspan(.7,1.4,color=".9",alpha=.5);ax.set_xlim(K[0],K[-1])
    figure(fig,out,"raw_output_audit",[f"checkpoints/{key}.json"],"Fixed design, second maturity. Shading marks the quote/test strike band. No clipping or repair is applied.")
    print(json.dumps(stats,indent=2))
    return stats

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--input",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args();run(args.input,args.out)
