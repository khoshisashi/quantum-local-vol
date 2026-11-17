"""Publication figures and evidence-only LaTeX tables; no simulated results supplied."""
import zipfile, shutil

COLORS={"expm":"#1764A0","fd":"#55813B","lchs-beta":"#D16A22","lchs-simple":"#8A619A",
        "AH":"#4B5961","PINN-fixed":"#918267","PINN-Wam":"#008D91"}
LABELS={"expm":"Exact expm","fd":"FD + adjoint","lchs-beta":"LCHS beta",
        "lchs-simple":"LCHS simple","AH":"Andreasen-Huge","PINN-fixed":"PINN fixed","PINN-Wam":"PINN adaptive"}

def figure_style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8,
        "axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5,
        "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,
        "ps.fonttype":42,"savefig.dpi":220,"axes.axisbelow":True})

def export_figure(fig,cfg,name,sources,caption):
    for ax in fig.axes:
        if ax.get_label()!="<colorbar>": ax.grid(alpha=.16,linewidth=.5)
    if cfg["profile"]!="paper":
        fig.text(.99,-.075,cfg["profile"].upper()+" / implementation check",ha="right",fontsize=6,color="#80562D")
    fig.savefig(cfg["out"]/"figures"/(name+".pdf"),bbox_inches="tight",pad_inches=.06)
    fig.savefig(cfg["out"]/"figures"/(name+".png"),bbox_inches="tight",pad_inches=.06)
    save_json(cfg["out"]/"figures"/(name+".json"),dict(sources=sources,caption=caption,profile=cfg["profile"]))
    plt.close(fig)

def make_figures(cfg,results,selected):
    figure_style(); out=cfg["out"]; n=cfg["ns"][min(1,len(cfg["ns"])-1)]; g=make_grid(n)
    fig,ax=plt.subplots(2,2,figsize=(3.5,3.45),layout="constrained")
    for r,truth in enumerate(["in","out"]):
        data=make_dataset(truth,0.,101,cfg["ref_n"]); s=1
        plotk=np.linspace(.7,1.4,401)
        ax[r,0].plot(plotk,true_sigma2(np.log(plotk),s,truth),color="black",lw=1.3,label="Truth")
        for method in ["expm","lchs-beta"]:
            key=f'{data["dataset_id"]}_n{n}_{method}_s{cfg["seeds"][0]}'
            if key not in selected: continue
            result=selected[key]["result"]
            ax[r,0].plot(plotk,.01+np.asarray(result["hats"][s])@basis(np.log(plotk),g["M"]),color=COLORS[method],lw=1,label=LABELS[method])
            test=data["tests"][s]; p=interpolation(g["K"],test["k"])@np.asarray(result["C"])[s]
            ax[r,1].plot(test["k"],(p-test["price"])*1e4,color=COLORS[method],lw=1,label=LABELS[method]+" raw")
            if method=="lchs-beta":
                px=interpolation(g["K"],test["k"])@np.asarray(result["C_exact"])[s]
                ax[r,1].plot(test["k"],(px-test["price"])*1e4,color=COLORS[method],ls=":",lw=1,label="Beta exact re-eval.")
        ax[r,0].set(xlim=(.7,1.4),ylabel="Local variance",title=("In-family" if truth=="in" else "Out-of-family")+", T=1")
        ax[r,1].axhline(0,color=".6",lw=.6); ax[r,1].set(ylabel="Price error (bp)",title="Held-out strikes, T=1")
        for a in ax[r]: a.set_xlabel("Normalised strike k")
    ax[0,0].legend(loc="best"); ax[0,1].legend(loc="best")
    export_figure(fig,cfg,"calibration",["selected_examples.json","raw/"],"Seed 0, zero noise, second maturity; explicit dataset identifiers in selected_examples.json.")

    df=pd.read_csv(out/"error_budget.csv")
    fig,axs=plt.subplots(1,2,figsize=(3.5,2.15),layout="constrained")
    for kernel,color in [("simple",COLORS["lchs-simple"]),("beta",COLORS["lchs-beta"])]:
        part=df[(df.n==n)&(df.kernel==kernel)]
        for _,p in part.groupby("cutoff"):
            axs[0].loglog(p.terms,p.price_error_inf,":",color=color,alpha=.25,lw=.7)
        best=part.loc[part.groupby("terms").price_error_inf.idxmin()].sort_values("terms")
        axs[0].loglog(best.terms,best.price_error_inf,"o-",ms=3,color=color,label=kernel)
        axs[1].loglog(part.solver_s*1000,part.price_error_inf,"o",ms=3,color=color,label=kernel)
    axs[0].set(xlabel="LCU terms J",ylabel="Max price error",title=f"Cutoff envelope, N={2**n}")
    from matplotlib.ticker import NullFormatter
    axs[0].set_xticks(cfg["kernels"],[str(x) for x in cfg["kernels"]])
    axs[0].xaxis.set_minor_formatter(NullFormatter())
    timing=df[df.n==n].solver_s.to_numpy()*1000
    ticks=np.geomspace(timing.min(),timing.max(),3)
    axs[1].set_xticks(ticks,[f"{x:.2g}" for x in ticks]); axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].set(xlabel="Warm evaluation (ms)",ylabel="Max price error",title="Classical simulation")
    axs[0].legend(); axs[1].legend()
    export_figure(fig,cfg,"kernel_budget",["error_budget.csv"],"Faint lines show every cutoff; envelope minimises reference error and is not a deployable tuning rule.")

    fig,axs=plt.subplots(1,3,figsize=(3.5,1.85),layout="constrained")
    for ax,(kind,label) in zip(axs,[("butterfly","Butterfly"),("calendar","Calendar"),("density","Density")]):
        x=np.maximum(df[kind+"_bound"].to_numpy(),1e-16); y=np.maximum(df[kind+"_violation"].to_numpy(),1e-16)
        ax.loglog(x,y,"o",ms=2.3,color=COLORS["lchs-beta"],alpha=.6)
        low=min(x.min(),y.min()); high=max(x.max(),y.max()); ax.plot([low,high],[low,high],"--",color=".5",lw=.7)
        ax.set(title=label,xlabel="Bound",ylabel="Violation" if kind=="butterfly" else "")
    export_figure(fig,cfg,"error_certificate",["error_budget.csv"],"Zero violations plotted at 1e-16. Units are price for butterfly/calendar, price per strike squared for density. All bounds are conditional on measured exact-model price error.")

    fig,ax=plt.subplots(figsize=(5.5,3.2),layout="constrained")
    part=results[(results.n==n)&(results.truth=="out")&(results.noise_bp==5)]
    methods=[m for m in LABELS if m in set(part.method)]
    for i,m in enumerate(methods):
        values=part[part.method==m].rmse_heldout.to_numpy()*1e4
        ax.scatter(values,i+np.linspace(-.1,.1,len(values)),color=COLORS[m],s=13,alpha=.7)
    ax.set_yticks(range(len(methods)),[LABELS[m] for m in methods]); ax.invert_yaxis()
    ax.set(xlabel="Held-out price RMSE (normalised bp)",title=f"Out-of-family, 5 bp noise, N={2**n}")
    export_figure(fig,cfg,"benchmark_variability",["results.csv"],"Each dot is an executed run; initialisation seeds are not independent market observations.")

    q=pd.read_csv(out/"quantisation.csv"); f=pd.read_csv(out/"fourier_basis.csv")
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.4),layout="constrained")
    axs[0].semilogy(q.bits,q.rmse_quotes,"o-",color=COLORS["expm"],label="Quote RMSE")
    axs[0].semilogy(q.bits,q.rmse_heldout,"s-",color=COLORS["lchs-beta"],label="Held-out RMSE")
    axs[0].axhline(q.continuous_reference.iloc[0],ls=":",color=".4",label="Continuous quote RMSE")
    axs[0].set(xlabel="Bits per coefficient",ylabel="Price RMSE",title="Rounded fitted coefficients"); axs[0].legend()
    axs[1].loglog(f.degree,f.error_inf,"o-",color=COLORS["expm"],label="Basis sup error")
    axs[1].loglog(f.degree,np.maximum(-f.minimum_basis,1e-18),"s--",color=COLORS["lchs-beta"],label="Negative basis overshoot")
    axs[1].set(xlabel="Fourier degree",ylabel="Absolute basis error",title="Compact C1 basis"); axs[1].legend()
    export_figure(fig,cfg,"precision_basis",["quantisation.csv","fourier_basis.csv"],"Quantisation does not re-optimise the discrete objective. Fourier approximation is diagnosed separately from exact pointwise calibration.")

    q=pd.read_csv(out/"quantum_objective.csv")
    fig,axs=plt.subplots(1,2,figsize=(3.5,1.85),layout="constrained")
    axs[0].plot(q.normalised_loss,q.circuit_block_real,"o",ms=4,color=COLORS["expm"])
    lo=q.normalised_loss.min(); hi=q.normalised_loss.max(); axs[0].plot([lo,hi],[lo,hi],"--",color=".5",lw=.7)
    axs[0].set(xlabel="Matrix objective / scale²",ylabel="Circuit clean block",title="All coherent designs")
    err=np.abs(q.circuit_block_real+1j*q.circuit_block_imag-q.normalised_loss)
    axs[1].semilogy(q.index,np.maximum(err,1e-16),"o-",color=COLORS["lchs-beta"],ms=3)
    axs[1].set(xlabel="Design index",ylabel="Block error",title="Statevector validation")
    export_figure(fig,cfg,"quantum_validation",["quantum_objective.csv","quantum_resources.json"],"Actual residual/objective circuit, one maturity and two design parameters; dense enumerated SELECT.")

    q=pd.read_csv(out/"qhd_trajectory.csv"); d=pd.read_csv(out/"qhd_distribution.csv")
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.4),layout="constrained")
    axs[0].plot(q.time,q.near_optimal_probability,color=COLORS["expm"])
    axs[0].axhline(q.near_optimal_probability.iloc[0],ls=":",color=".5",label="Uniform initial state")
    axs[0].set(xlabel="Schedule time",ylabel="Near-grid-optimum probability",ylim=(0,1),title="Classical QHD simulation"); axs[0].legend()
    sz=int(np.sqrt(len(d))); im=axs[1].imshow(d.probability.to_numpy().reshape(sz,sz),origin="lower",extent=[0,1,0,1],aspect="auto",cmap="Blues")
    axs[1].set(xlabel="Coefficient 1",ylabel="Coefficient 2",title="Final design probability"); fig.colorbar(im,ax=axs[1],label="Probability")
    export_figure(fig,cfg,"qhd_diagnostics",["qhd_trajectory.csv","qhd_distribution.csv","qhd_refinement.csv"],"Classically tabulated denser landscape; only the separate coarse design circuit is coherently verified. Near optimal = within 1% of the finite objective range.")

    f=pd.read_csv(out/"fd_convergence.csv"); j=pd.read_csv(out/"identifiability.csv")
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.3),layout="constrained")
    axs[0].loglog(f.steps,f.price_error_inf,"o-",color=COLORS["fd"]); axs[0].set(xlabel="Backward Euler steps",ylabel="Max price error",title="Time discretisation at fixed design")
    for s,p in j.groupby("slice"): axs[1].semilogy(p["index"],p.singular_value,"o-",label=f"Slice {s}")
    axs[1].set(xlabel="Singular-value index",ylabel="Quote-Jacobian singular value",title="Local identifiability"); axs[1].legend()
    export_figure(fig,cfg,"fd_identifiability",["fd_convergence.csv","identifiability.csv"],"FD convergence separates time error from optimiser error. Jacobian spectra are local diagnostics.")

    fig,axs=plt.subplots(1,2,figsize=(5.5,2.3),layout="constrained")
    for method in ["PINN-fixed","PINN-Wam"]:
        key=f'out_bp5_d101_n{n}_{method}_s{cfg["seeds"][0]}'
        if key not in selected: continue
        hist=pd.DataFrame(selected[key]["result"]["pinn_history"])
        for ax,metric in zip(axs,["fit","pde"]): ax.semilogy(hist.step,np.maximum(hist[metric],1e-18),label=LABELS[method],color=COLORS[method])
    for ax,metric in zip(axs,["Unweighted fit MSE","Unweighted PDE MSE"]):
        ax.set(xlabel="Training step",ylabel=metric,title="Out-of-family / seed 0"); ax.legend()
    export_figure(fig,cfg,"pinn_losses",["selected_examples.json"],"Fixed/adaptive network losses; adaptation is not an exact reproduction of the cited study.")

    if (out/"ablation.csv").exists():
        ab=pd.read_csv(out/"ablation.csv"); fig,axs=plt.subplots(1,2,figsize=(5.5,2.3),layout="constrained")
        for ax,method in zip(axs,["expm","lchs-beta"]):
            p=ab[ab.method==method].pivot(index="M",columns="lam",values="rmse_heldout")
            im=ax.imshow(p.to_numpy()*1e4,aspect="auto",cmap="Blues",vmin=ab.rmse_heldout.min()*1e4,vmax=ab.rmse_heldout.max()*1e4)
            ax.set_xticks(range(len(p.columns)),[f"{x:g}" for x in p.columns],rotation=30)
            ax.set_yticks(range(len(p.index)),p.index); ax.set(xlabel="Regularisation lambda",ylabel="Basis size M",title=LABELS[method]); fig.colorbar(im,ax=ax,label="Held-out RMSE (bp)")
        export_figure(fig,cfg,"ablation",["ablation.csv"],"Out-of-family, fixed 5 bp noise dataset and initialisation. Not a hyperparameter selection study.")
    if (out/"domain_slice_sensitivity.csv").exists():
        p=pd.read_csv(out/"domain_slice_sensitivity.csv"); fig,axs=plt.subplots(1,2,figsize=(5.5,2.3),layout="constrained")
        for method,d in p[p.test=="maturity_slices"].groupby("method"):
            axs[0].plot(d.actual_slices,d.rmse_heldout*1e4,"o-",label=LABELS[method],color=COLORS[method])
        d=p[p.test=="fitted_domain"]; axs[1].plot(d.domain_halfwidth,d.rmse_heldout*1e4,"o-",color=COLORS["expm"])
        axs[0].set(xlabel="Actual maturity slices",ylabel="Held-out RMSE (bp)",title="Extra expiries add observations"); axs[0].legend()
        axs[1].set(xlabel="Log-strike domain half-width",ylabel="Held-out RMSE (bp)",title="Fixed N: boundary and grid effects")
        export_figure(fig,cfg,"domain_slices",["domain_slice_sensitivity.csv"],"The true 182/365 breakpoint is retained. More expiries add data; fixed N domain changes also alter h.")

def tex_escape(s):
    return str(s).replace("\\",r"\textbackslash{}").replace("_",r"\_").replace("%",r"\%").replace("&",r"\&").replace("#",r"\#")

def tex_number(x):
    if not np.isfinite(x): return "--"
    return f"{x:.3g}"

def write_tables(cfg,results):
    out=cfg["out"]; t=out/"tables"; n=cfg["ns"][min(1,len(cfg["ns"])-1)]
    failures=json.loads((out/"failures.json").read_text())
    status=json.loads((out/"market_status.json").read_text())
    market_status=status["status"]
    macro=("\\renewcommand{\\StudyProfile}{"+cfg["profile"]+"}\n"+
           "\\renewcommand{\\RunCount}{"+str(len(results))+"}\n"+
           "\\renewcommand{\\MarketStatus}{"+tex_escape(market_status)+"}\n"+
           "\\renewcommand{\\RunStatus}{"+str(len(failures))+" failed runs are retained in the failure log.}\n")
    (t/"run_macros.tex").write_text(macro)
    # Dataset means prevent initialisation replication from becoming fake data n.
    part=results[(results.n==n)&(results.truth=="out")&(results.noise_bp==5)]
    cols=["rmse_quotes","rmse_heldout","iv_rmse","opt_s","butterfly_violation","calendar_violation"]
    means=part.groupby(["method","dataset"],as_index=False)[cols].mean()
    abstract=""
    if cfg["profile"]=="paper":
        vals={m:float(d.rmse_heldout.mean()*1e4) for m,d in means.groupby("method")}
        if all(m in vals for m in ["expm","fd","lchs-beta","AH"]):
            abstract=(f"For the out-of-family, 5-bp-noise case at $N={2**n}$, mean held-out price errors are "
                f"{vals['lchs-beta']:.3g}, {vals['expm']:.3g}, {vals['fd']:.3g}, and {vals['AH']:.3g} normalised bp "
                "for beta-LCHS, exact propagation, finite differences, and Andreasen--Huge, respectively.")
    with (t/"run_macros.tex").open("a") as handle:
        handle.write("\\renewcommand{\\AbstractEvidence}{"+abstract+"}\n")
    means.to_csv(out/"dataset_means.csv",index=False)
    lines=[r"\begin{table}[t]\centering\caption{Out-of-family, 5 bp noise, $N="+str(2**n)+r"$. Means over datasets after averaging initialisations. Price errors are normalised bp; time is classical optimisation seconds.}",
           r"\label{tab:bench}\scriptsize\begin{tabular}{lrrrr}\toprule Method & Fit & Test & Seconds & Runs\\\midrule"]
    for method in LABELS:
        d=means[means.method==method]
        if d.empty: continue
        lines.append(tex_escape(LABELS[method])+" & "+" & ".join([tex_number(d.rmse_quotes.mean()*1e4),tex_number(d.rmse_heldout.mean()*1e4),tex_number(d.opt_s.mean()),str(int((part.method==method).sum()))])+r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (t/"benchmark.tex").write_text("\n".join(lines))
    # Full uncertainty summaries retain both hierarchy levels in CSV.
    metrics=[c for c in cols+["sigma2_linf","compile_warm_s","iv_valid","iv_total"] if c in results]
    groups=["truth","noise_bp","n","method"]
    results.groupby(groups)[metrics].agg(["mean","std","min","max","count"]).to_csv(out/"all_run_summary.csv")
    dataset=results.groupby(groups+["dataset"])[metrics].mean()
    dataset.groupby(groups)[metrics].agg(["mean","std","count"]).to_csv(out/"between_dataset_summary.csv")
    results.groupby(groups+["dataset"])[metrics].std().to_csv(out/"within_dataset_initialisation_sd.csv")
    lc=results[results.method.str.startswith("lchs")]
    exact=results[results.method=="expm"]
    summary=(f"The executed {tex_escape(cfg['profile'])} profile contains {len(results)} completed synthetic calibration runs on "
       f"{results.dataset.nunique()} distinct datasets; {len(failures)} failures are logged. "
       f"Raw LCHS price-butterfly and calendar violation maxima are {lc.butterfly_violation.max():.3g} and {lc.calendar_violation.max():.3g}, respectively. "
       f"The exact-exponential price-butterfly violation maximum is {exact.butterfly_violation.max():.3g}. "
       "These are normalised price quantities, not derivative tolerances. The complete per-run CSV retains non-convergence, invalid-IV counts, compilation costs, and exact re-evaluation audits. ")
    if cfg["profile"]!="paper": summary+="These reduced-budget results are implementation evidence only and must not be presented as the full study. "
    summary+="Between-dataset standard deviations and within-dataset initialisation standard deviations are exported separately; a single dataset does not support a between-dataset uncertainty estimate.\n"
    (t/"results_summary.tex").write_text(summary)
    resource=json.loads((out/"quantum_resources.json").read_text())
    text=(f"The executed one-slice circuit uses {resource['qubits']} qubits and {resource['design_points']} coherent design points. "
          f"The maximum clean-block error is {resource['objective_block_error']:.3g}. "
          f"Dense synthesis produces {resource['cx']:,} CNOT gates, {resource['u']:,} single-qubit gates, and depth {resource['depth']:,}. "
          "These counts use the declared all-to-all u/CNOT basis and generic design enumeration; they are not a fault-tolerant resource estimate.\n")
    (t/"quantum_summary.tex").write_text(text)
    if market_status=="not_run":
        mt="Real-market study: not run. No market observations were supplied. The input template and validation/held-out evaluation code are provided; no synthetic result is labelled as market evidence.\n"
    else:
        market=pd.read_csv(out/"market_results.csv")
        mt=(f"The market input contains {status.get('dates',0)} date/underlying datasets with {len(market)} completed method runs and {len(status.get('failures',[]))} failures. "
            "The source checksum, raw normalised snapshots, held-out bid--ask coverage and full method results accompany the manuscript. ")
        if len(market):
            mt+=r"\begin{table}[t]\centering\caption{Executed real-market runs: held-out price error and bid--ask coverage.}\scriptsize\begin{tabular}{lrr}\toprule Method & RMSE (bp) & Coverage\\\midrule"+"\n"
            for m,d in market.groupby("method"):
                mt+=tex_escape(LABELS[m])+f" & {1e4*d.rmse_heldout.mean():.3g} & {d.heldout_bidask_fraction.mean():.3f}"+r"\\"+"\n"
            mt+=r"\bottomrule\end{tabular}\end{table}"
    (t/"market_summary.tex").write_text(mt)
    save_json(out/"evidence_status.json",dict(profile=cfg["profile"],completed_runs=len(results),failures=len(failures),
             market_status=market_status,quantum_hardware_executed=False,full_coherent_qhd_executed=False,
             evidence_ready_for_author_review=cfg["profile"]=="paper" and market_status=="completed"))

def make_supplement(cfg):
    out=cfg["out"]
    intro=r"""\documentclass[10pt]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{graphicx,booktabs,amsmath}
\usepackage[hidelinks]{hyperref}
\title{Supplementary diagnostics: local-volatility calibration}
\author{Anonymous submission}
\date{}
\begin{document}\maketitle
This supplement is generated from executed experiments. Profile: PROFILE.
Smoke and pilot outputs are validation/exploration, not final study evidence.
Every figure has a same-name JSON source record. Full metrics, unsuccessful
optimisation runs, failed baseline calls, and invalid implied-volatility counts
are retained in the accompanying CSV/JSON files.
""".replace("PROFILE",cfg["profile"])
    names=[("benchmark_variability","All executed initialisations, out-of-family noisy case."),
           ("precision_basis","Coefficient rounding and global Fourier approximation."),
           ("fd_identifiability","Time-discretisation and local identifiability diagnostics."),
           ("pinn_losses","Documented fixed/adaptive PINN comparison."),
           ("qhd_diagnostics",r"Classically tabulated QHD simulation; see qhd\_refinement.csv."),
           ("ablation","Independent basis-size and regularisation ablation."),
           ("domain_slices","Maturity and finite-domain sensitivity, with explicit confounders.")]
    for name,caption in names:
        if (out/"figures"/(name+".pdf")).exists():
            intro+=r"\begin{figure}[htbp]\centering\includegraphics[width=.92\linewidth]{figures/"+name+r".pdf}\caption{"+caption+r"}\end{figure}"+"\n"
    intro+=r"\end{document}"
    (out/"supplement.tex").write_text(intro)

def finalise_outputs(cfg,results,selected,main_tex,readme,source_files):
    make_figures(cfg,results,selected); write_tables(cfg,results); make_supplement(cfg)
    out=cfg["out"]; (out/"main.tex").write_text(main_tex,encoding="utf-8")
    (out/"README.md").write_text(readme,encoding="utf-8")
    (out/"source").mkdir(exist_ok=True)
    for name,content in source_files.items(): (out/"source"/name).write_text(content,encoding="utf-8")
    manifest=json.loads((out/"manifest.json").read_text()); manifest["study_status"]="completed"
    manifest["market_status"]=json.loads((out/"market_status.json").read_text())["status"]
    manifest["source_sha256"]={name:hashlib.sha256(content.encode()).hexdigest() for name,content in source_files.items()}
    manifest["manuscript_sha256"]=hashlib.sha256(main_tex.encode()).hexdigest()
    save_json(out/"manifest.json",manifest)
    archive=out.parent/(out.name+"_results.zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix not in [".zip",".tmp"]: z.write(p,p.relative_to(out))
    print("Generated",len(list((out/"figures").glob("*.pdf"))),"figure PDFs, LaTeX tables, manuscript and supplement.")
    print("Result archive:",archive)
    return archive
