"""Sensitivity analyses and QHD resolution checks; require preceding cells."""

def run_domain_slice_sensitivity(cfg):
    if not cfg["ablation"]:
        save_json(cfg["out"]/"sensitivity_status.json",dict(status="skipped_smoke_profile"))
        return pd.DataFrame()
    rows=[]; n=cfg["ns"][1]; gm=make_grid(n)
    for S in [2,4,8]:
        times=np.unique(np.r_[np.arange(1,S+1)/S,float(TIMES[0])])
        K,C=reference_surface("in",cfg["ref_n"],tuple(times))
        qs=[]; tests=[]; ks=TRAIN_K[0]; kt=np.linspace(.705,1.395,151)
        kt=kt[~np.any(np.isclose(kt[:,None],ks[None,:],atol=1e-10,rtol=0),axis=1)]
        for T,p in zip(times,C):
            mid=interpolation(K,ks)@p
            qs.append(dict(T=float(T),k=ks,mid=mid,bid=mid,ask=mid))
            tests.append(dict(T=float(T),k=kt,price=interpolation(K,kt)@p))
        data=dict(dataset_id=f"slice_{len(times)}",kind="synthetic",truth="in",noise_bp=0.,noise_seed=101,quotes=qs,tests=tests)
        for method in ["expm","lchs-beta"]:
            result=calibrate_model(gm,data,method,cfg,0,0.)
            row=evaluate_model(gm,data,result,method,0,0.)
            # The original two-slice coefficient-error indexing does not apply.
            row.pop("sigma2_linf",None)
            row.update(test="maturity_slices",nominal_slices=S,actual_slices=len(times),total_quotes=sum(len(q["k"]) for q in qs))
            rows.append(row)
            pd.DataFrame(rows).to_csv(cfg["out"]/"domain_slice_sensitivity.csv",index=False)
    data=make_dataset("in",0.,101,cfg["ref_n"])
    for domain in [1.,1.2,1.6]:
        # Fixed N means h changes: report it rather than calling this pure
        # boundary error. Independent fine-grid domain checks follow.
        g=make_grid(n,4,-domain,domain)
        r=calibrate_model(g,data,"expm",cfg,0,0.)
        row=evaluate_model(g,data,r,"expm",0,0.)
        row.update(test="fitted_domain",domain_halfwidth=domain,h=g["h"])
        rows.append(row)
    pts=np.linspace(.7,1.4,101)
    K0,C0=reference_surface("in",cfg["ref_n"]+1,tuple(TIMES),-1.6,1.6)
    for domain in [1.,1.2]:
        K,C=reference_surface("in",cfg["ref_n"],tuple(TIMES),-domain,domain)
        err=float(np.max(np.abs(C@interpolation(K,pts).T-C0@interpolation(K0,pts).T)))
        rows.append(dict(test="reference_domain",domain_halfwidth=domain,reference_domain_price_inf=err,
                         note="Includes finite reference discretisation error"))
    frame=pd.DataFrame(rows); frame.to_csv(cfg["out"]/"domain_slice_sensitivity.csv",index=False)
    save_json(cfg["out"]/"sensitivity_status.json",dict(status="completed"))
    return frame

def qhd_landscape_and_refinement(cfg,circuit_landscape):
    """Denser CLASSICAL tabulation of the same finite-LCHS objective.
    The actual circuit validates its coarse design points only. The denser
    landscape does not imply a coherent multi-bit experiment was performed.
    """
    size={"smoke":4,"pilot":8,"paper":16}[cfg["profile"]]
    g=make_grid(cfg["circuit_n"],2); dt=float(TIMES[0])
    qk=np.array([.85,1.1])
    W=interpolation(g["K"],qk)[:,1:-1]
    y=W@(sla.expm(-dt*assemble(g,[.7,.25]))@g["v0"])
    k,w,_=lchs_weights("beta",cfg["circuit_terms"],4.)
    rows=[]
    for j in range(size):
        for i in range(size):
            xi=np.array([i,j])/(size-1); A=assemble(g,xi)
            L=(A+A.T)/2+g["cbar"]*np.eye(g["N"]); H=-.5j*(A-A.T)
            p=sum(wl*(sla.expm(-1j*dt*(H+kl*L))@g["v0"]) for kl,wl in zip(k,w))*np.exp(g["cbar"]*dt)
            rows.append(dict(xi1=xi[0],xi2=xi[1],physical_loss=float(np.linalg.norm(W@p-y)**2)))
    landscape=pd.DataFrame(rows)
    for row in circuit_landscape.itertuples():
        match=landscape[(np.isclose(landscape.xi1,row.xi1))&(np.isclose(landscape.xi2,row.xi2))]
        assert len(match)==1 and abs(match.physical_loss.iloc[0]-row.physical_loss)<1e-12
    landscape.to_csv(cfg["out"]/"qhd_classical_landscape.csv",index=False)
    trajectory=run_qhd(cfg,landscape)
    fine=dict(cfg); fine["out"]=cfg["out"]/"diagnostics"/"qhd_refined"
    fine["out"].mkdir(parents=True,exist_ok=True); fine["qhd_steps"]=2*cfg["qhd_steps"]
    refined=run_qhd(fine,landscape)
    p=pd.read_csv(cfg["out"]/"qhd_distribution.csv").probability.to_numpy()
    q=pd.read_csv(fine["out"]/"qhd_distribution.csv").probability.to_numpy()
    row=dict(steps=cfg["qhd_steps"],refined_steps=fine["qhd_steps"],design_points=size*size,
             probability_total_variation=float(.5*np.abs(p-q).sum()),
             expected_loss_difference=float(abs(trajectory.expected_loss.iloc[-1]-refined.expected_loss.iloc[-1])),
             scope="Classical tabulation; coarse circuit points cross-checked; not full coherent QHD")
    pd.DataFrame([row]).to_csv(cfg["out"]/"qhd_refinement.csv",index=False)
    return trajectory
