"""Benchmark and diagnostic cells. Requires core.py definitions in this namespace."""
import QuantLib as ql

def ah_calibrate(g,data,cfg):
    today=ql.Date(1,1,2025); ql.Settings.instance().evaluationDate=today
    dc=ql.Actual365Fixed(); zero=ql.YieldTermStructureHandle(ql.FlatForward(today,0.,dc))
    quotes=ql.CalibrationSet()
    for q in data["quotes"]:
        days=int(round(q["T"]*365))
        if abs(days/365-q["T"])>1e-9:
            raise ValueError("AH requires ACT/365F maturities, no silent date rounding")
        for k,price in zip(q["k"],q["mid"]):
            vol=implied_vol(price,k,q["T"])
            if not np.isfinite(vol):
                raise ValueError("AH input has no finite Black IV; quotes retained, baseline marked failed")
            option=ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call,float(k)),ql.EuropeanExercise(today+days))
            quotes.push_back((option,ql.SimpleQuote(float(vol))))
    t=time.perf_counter()
    ah=ql.AndreasenHugeVolatilityInterpl(quotes,ql.QuoteHandle(ql.SimpleQuote(1.)),zero,zero,
                ql.AndreasenHugeVolatilityInterpl.Linear,ql.AndreasenHugeVolatilityInterpl.Call,
                max(200,2*g["N"]),float(g["K"][0]*.8),float(g["K"][-1]*1.2))
    error=ah.calibrationError()  # force the native calibration inside the timer
    elapsed=time.perf_counter()-t
    Cs=np.array([[ah.optionPrice(q["T"],float(k),ql.Option.Call) for k in g["K"]]
                 for q in data["quotes"]])
    qp=[[ah.optionPrice(q["T"],float(k),ql.Option.Call) for k in q["k"]] for q in data["quotes"]]
    hp=[[ah.optionPrice(q["T"],float(k),ql.Option.Call) for k in q["k"]] for q in data["tests"]]
    return dict(C=Cs,quote_predictions=qp,test_predictions=hp,
                converged=True,opt_s=elapsed,compile_warm_s=0.,nfev=0,
                native_error=str(error),method_note="QuantLib Andreasen-Huge linear interpolation")

def init_mlp(key,width):
    sizes=[2,width,width,1]; pars=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        key,sub=jax.random.split(key)
        pars.append((jax.random.normal(sub,(a,b))*np.sqrt(2/(a+b)),jnp.zeros(b)))
    return pars

def mlp(pars,x):
    for w,b in pars[:-1]: x=jnp.tanh(x@w+b)
    return (x@pars[-1][0]+pars[-1][1]).squeeze()

def pinn_calibrate(g,data,cfg,seed=0,adaptive=True):
    """Wam-style adaptation, NOT a numerical reproduction of CIFEr 2024.
    The original alternating SA / auto-balancing schedule is retained. Two
    independent networks make the PDE residual non-tautological. See README.
    """
    key=jax.random.PRNGKey(seed); k1,k2=jax.random.split(key)
    tree=(init_mlp(k1,cfg["pinn_width"]),init_mlp(k2,cfg["pinn_width"]))
    theta,unravel=ravel_pytree(tree)
    Tmax=data["quotes"][-1]["T"]
    kt=np.concatenate([q["k"] for q in data["quotes"]]); tt=np.concatenate([np.full(len(q["k"]),q["T"]) for q in data["quotes"]])
    yt=jnp.array(np.concatenate([q["mid"] for q in data["quotes"]])); train=jnp.array(np.c_[kt,tt])
    quote_weight=jnp.array(np.concatenate([1/np.maximum((q["ask"]-q["bid"])/2,1e-5)
                 if data["kind"]=="market" else np.ones(len(q["k"])) for q in data["quotes"]]))
    rng=np.random.default_rng(919+seed)
    xc=np.exp(rng.uniform(g["xmin"],g["xmax"],cfg["collocation"]))
    tc=rng.uniform(.02*Tmax,Tmax,cfg["collocation"])
    coll=jnp.array(np.c_[xc,tc]); boundary=jnp.array(np.c_[np.tile(g["K"][[0,-1]],24),np.repeat(np.linspace(.02*Tmax,Tmax,24),2)])
    def functions(th,z):
        p,lv=unravel(th); k,t=z
        inp=jnp.array([jnp.log(k)/1.2,2*t/Tmax-1])
        iv=.03+.35*jax.nn.softplus(mlp(p,inp))
        s=iv*jnp.sqrt(jnp.maximum(t,1e-10)); d=-jnp.log(k)/s+s/2
        c=jax.scipy.special.ndtr(d)-k*jax.scipy.special.ndtr(d-s)
        variance=.01+.04*jax.nn.softplus(mlp(lv,inp))
        return c,variance
    def cfun(th,z): return functions(th,z)[0]
    def errors(th):
        pred=jax.vmap(cfun,(None,0))(th,train)
        vals=jax.vmap(functions,(None,0))(th,coll)
        deriv=jax.vmap(jax.grad(cfun,argnums=1),(None,0))(th,coll)
        hess=jax.vmap(jax.hessian(cfun,argnums=1),(None,0))(th,coll)
        ck,ct=deriv[:,0],deriv[:,1]; ckk=hess[:,0,0]
        pde=ct-.5*vals[1]*coll[:,0]**2*ckk
        cb=jax.vmap(cfun,(None,0))(th,boundary)
        btarget=jnp.maximum(1-boundary[:,0],0)
        return (((pred-yt)*quote_weight)**2,pde**2,jax.nn.relu(-ckk)**2,
                jax.nn.relu(-ct)**2,(jax.nn.relu(ck)**2+jax.nn.relu(-1-ck)**2),
                (cb-btarget)**2)
    sa=tuple(jnp.ones(len(x)) for x in errors(theta)); weights=jnp.ones(6)
    def components(th,sa,weights): return jnp.stack([w*jnp.mean(a*e) for w,a,e in zip(weights,sa,errors(th))])
    @jax.jit
    def update(th,m,v,step,sa,weights):
        loss,gr=jax.value_and_grad(lambda p:jnp.sum(components(p,sa,weights)))(th)
        m=.9*m+.1*gr; v=.999*v+.001*gr**2
        lr=1e-3*.9**(step/2000)
        th=th-lr*(m/(1-.9**(step+1)))/(jnp.sqrt(v/(1-.999**(step+1)))+1e-8)
        return th,m,v,loss
    @jax.jit
    def rebalance(th,sa,w):
        jac=jax.jacrev(components)(th,sa,w)
        norms=jnp.mean(jnp.abs(jac),axis=1)
        target=jnp.where(norms>1e-15,jnp.sum(norms)/jnp.maximum(norms,1e-15),1.)
        return jax.lax.stop_gradient(.9*w+.1*jnp.clip(target,1e-3,1e3))
    @jax.jit
    def ascend(th,sa,w):
        # Gradient ascent of weighted mean loss w.r.t. per-point multipliers.
        return tuple(jnp.minimum(a+weight*err/len(err),1e3) for a,weight,err in zip(sa,w,errors(th)))
    m=jnp.zeros_like(theta); v=m; hist=[]
    t=time.perf_counter(); jax.block_until_ready(update(theta,m,v,0,sa,weights))
    if adaptive:
        jax.block_until_ready(rebalance(theta,sa,weights)); jax.block_until_ready(ascend(theta,sa,weights))
    warm=time.perf_counter()-t; t=time.perf_counter()
    for step in range(cfg["pinn_steps"]):
        if adaptive and step%100==0: weights=rebalance(theta,sa,weights)
        if adaptive and (step+50)%100==0: sa=ascend(theta,sa,weights)
        theta,m,v,loss=update(theta,m,v,step,sa,weights)
        if step%100==0 or step==cfg["pinn_steps"]-1:
            comp=np.asarray(components(theta,tuple(jnp.ones_like(a) for a in sa),jnp.ones(6)))
            hist.append(dict(step=step,loss=float(loss),fit=comp[0],pde=comp[1],butterfly=comp[2],calendar=comp[3]))
    jax.block_until_ready(theta); elapsed=time.perf_counter()-t
    if not np.all(np.isfinite(theta)): raise FloatingPointError("PINN non-finite parameters")
    predict=jax.jit(jax.vmap(cfun,(None,0)))
    Cs=np.array([predict(theta,jnp.array(np.c_[g["K"],np.full(len(g["K"]),q["T"])])) for q in data["quotes"]])
    qp=[np.asarray(predict(theta,jnp.array(np.c_[q["k"],np.full(len(q["k"]),q["T"])]))) for q in data["quotes"]]
    hp=[np.asarray(predict(theta,jnp.array(np.c_[q["k"],np.full(len(q["k"]),q["T"])]))) for q in data["tests"]]
    return dict(C=Cs,quote_predictions=qp,test_predictions=hp,
                converged=False,training_completed=True,opt_s=elapsed,compile_warm_s=warm,
                nfev=cfg["pinn_steps"],pinn_history=hist,weights=np.asarray(weights),
                parameter_count=len(theta),method_note="Fixed epoch budget; convergence not asserted")

def run_calibration_suite(cfg):
    allrows=[]; failures=[]; selected={}; out=cfg["out"]
    cases=[(truth,bp,ds) for truth in ["in","out"] for bp in [0.,5.]
           for ds in ([101] if bp==0 else cfg["noise_seeds"])]
    comparison_n=cfg["ns"][min(1,len(cfg["ns"])-1)]
    for truth,bp,ds in cases:
        data=make_dataset(truth,bp,ds,cfg["ref_n"])
        save_json(out/"raw"/(data["dataset_id"]+".json"),data)
        for n in cfg["ns"]:
            g=make_grid(n)
            methods=["expm","fd","lchs-beta","lchs-simple"]
            if n==comparison_n: methods += ["AH","PINN-fixed","PINN-Wam"]
            for method in methods:
                seeds=[cfg["seeds"][0]] if method=="AH" else cfg["seeds"]
                for seed in seeds:
                    lam=0.  # core solver comparison uses exactly the same objective
                    key=f'{data["dataset_id"]}_n{n}_{method}_s{seed}'
                    cp=out/"checkpoints"/(key+".json")
                    if cp.exists():
                        payload=json.loads(cp.read_text()); allrows.append(payload["row"])
                        if seed==cfg["seeds"][0] and n==comparison_n:
                            selected[key]=payload
                        continue
                    fp=out/"checkpoints"/(key+"_failure.json")
                    if fp.exists():
                        failures.append(json.loads(fp.read_text())); continue
                    try:
                        if method=="AH": result=ah_calibrate(g,data,cfg)
                        elif method.startswith("PINN"):
                            result=pinn_calibrate(g,data,cfg,seed,adaptive=(method=="PINN-Wam"))
                        else: result=calibrate_model(g,data,method,cfg,seed,lam)
                        row=evaluate_model(g,data,result,method,seed,lam)
                        row["profile"]=cfg["profile"]; row["kind"]="synthetic"
                        payload=dict(row=row,result=result)
                        save_json(cp,payload); allrows.append(row)
                        if seed==cfg["seeds"][0] and n==comparison_n: selected[key]=jsonable(payload)
                        print(f'{key}: quote {row["rmse_quotes"]:.3e}, test {row["rmse_heldout"]:.3e}, opt {row["opt_s"]:.1f}s')
                    except (ValueError,FloatingPointError,RuntimeError) as exc:
                        failure=dict(key=key,method=method,dataset=data["dataset_id"],reason=str(exc))
                        save_json(fp,failure)
                        failures.append(failure); print("FAILED:",failure)
                    pd.DataFrame(allrows).to_csv(out/"results.csv",index=False)
                    save_json(out/"failures.json",failures)
    pd.DataFrame(allrows).to_csv(out/"results.csv",index=False)
    save_json(out/"failures.json",failures)
    save_json(out/"selected_examples.json",selected)
    return pd.DataFrame(allrows),selected

def finite_difference_grad(fun,x,eps=1e-5):
    eye=np.eye(len(x))*eps
    return np.array([(fun(x+d)-fun(x-d))/(2*eps) for d in eye])

def run_validation(cfg):
    rows=[]
    for n in cfg["ns"]:
        g=make_grid(n); N=g["N"]; h=g["h"]
        rho=4*np.sinh(h/2)**2/h**2
        operator=rho*g["K"][1:-1,None]**2*g["B"]
        assert np.allclose(operator[:,1:-1],-g["Lt"],atol=1e-8,rtol=1e-11)
        lift=float(np.max(np.abs(operator@g["G"])))
        rng=np.random.default_rng(901); worst=0.; positive_shift=None
        for xi in np.r_[rng.uniform(0,1,(16,4)),np.eye(4),1-np.eye(4)]:
            A=assemble(g,xi); c=max(0.,-sla.eigvalsh((A+A.T)/2)[0]); worst=max(worst,c)
            if c>1e-5: positive_shift=xi
            off=-A+np.diag(np.diag(A)); assert np.min(off)>=-1e-12
            density_generator=g["B"][:,1:-1]@np.diag(rho*.5*sigma2(g,xi)*g["K"][1:-1]**2)
            assert np.min(density_generator-np.diag(np.diag(density_generator)))>=-1e-10
            assert c<=g["cbar"]+1e-10
        xi=np.array([.4,.6,.5,.3]); data=make_dataset("in",0.,101,cfg["ref_n"])
        q=data["quotes"][0]; Wf=interpolation(g["K"],q["k"]); W=Wf[:,1:-1]; y=q["mid"]-Wf@g["G"]
        val,grad,_=fd_value_grad(g,xi,g["v0"],q["T"],W,y,np.ones(len(y)),0.,cfg["fd_steps"])
        fd=finite_difference_grad(lambda z:fd_value_grad(g,z,g["v0"],q["T"],W,y,np.ones(len(y)),0.,cfg["fd_steps"])[0],xi)
        fd_error=float(np.linalg.norm(grad-fd)/max(1e-12,np.linalg.norm(fd)))
        assert fd_error<1e-4
        fw,vg=solver_functions(n,4,"lchs-beta",cfg["lchs_terms"],48.,float(q["T"]))
        args=tuple(jnp.array(x) for x in [g["v0"],W,y,np.ones(len(y))])+(0.,)
        _,grad=vg(jnp.array(xi),*args)
        fd=finite_difference_grad(lambda z:float(vg(jnp.array(z),*args)[0]),xi)
        lerr=float(np.linalg.norm(np.asarray(grad)-fd)/max(1e-12,np.linalg.norm(fd)))
        assert lerr<3e-4
        # Constant-vector and lift consistency in strike-space observation.
        assert np.allclose(Wf.sum(1),1) and np.allclose(Wf@g["K"],q["k"])
        rows.append(dict(n=n,lift_residual=lift,fitted_identity_residual=float(np.max(np.abs(operator[:,1:-1]+g["Lt"]))),
                         fd_gradient_relative_error=fd_error,lchs_gradient_relative_error=lerr,
                         maximum_sampled_shift=worst,uniform_shift=g["cbar"],
                         positive_shift_sample=positive_shift is not None))
    # Reference refinement at unchanged quotes/tests, not changing the data per N.
    for truth in ["in","out"]:
        K,C=reference_surface(truth,cfg["ref_n"]); K2,C2=reference_surface(truth,cfg["ref_n"]+1)
        pts=np.linspace(.7,1.4,101)
        err=float(np.max(np.abs(C@interpolation(K,pts).T-C2@interpolation(K2,pts).T)))
        rows.append(dict(test="reference_refinement",truth=truth,reference_price_inf=err))
    frame=pd.DataFrame(rows); frame.to_csv(cfg["out"]/"validation.csv",index=False)
    print("Structural identities, fixed quote operators, and implemented gradients: PASS")
    return frame

def run_error_budget(cfg):
    rows=[]; rng=np.random.default_rng(707)
    for n in cfg["ns"]:
        g=make_grid(n); xi=rng.uniform(.15,.85,(2,4))
        exact=[]; v=g["v0"].copy(); prior=0.
        for s,T in enumerate(TIMES):
            v=sla.expm(-(T-prior)*assemble(g,xi[s]))@v; exact.append(price_vector(g,v)); prior=T
        for kernel in ["simple","beta"]:
            for terms in cfg["kernels"]:
                for cutoff in ([16.,32.,64.] if cfg["profile"]!="smoke" else [8.,24.]):
                    k,w,meta=lchs_weights(kernel,terms,cutoff)
                    v=jnp.array(g["v0"]); approx=[]; previous=0.
                    call=jax.jit(lambda A,v,dt:lchs_apply_jax(A,v,dt,k,w,g["cbar"]))
                    t=time.perf_counter(); jax.block_until_ready(call(jnp.array(assemble(g,xi[0])),v,TIMES[0])); warm=time.perf_counter()-t
                    t=time.perf_counter()
                    for s,T in enumerate(TIMES):
                        v=call(jnp.array(assemble(g,xi[s])),v,T-previous); previous=T
                        approx.append(price_vector(g,v))
                    jax.block_until_ready(v); elapsed=time.perf_counter()-t
                    cert=certificate(g,approx,exact)
                    assert cert["certificate_pass"],"Error-bound check failed"
                    row=dict(n=n,kernel=kernel,**meta,**cert,**audit_prices(g["K"],approx),
                             solver_s=elapsed,compile_warm_s=warm,shift=g["cbar"],
                             simulation_time_proxy=cutoff*sum(TIMES-np.r_[0,TIMES[:-1]])*np.linalg.norm(assemble(g,xi[0]),2))
                    rows.append(row)
        pd.DataFrame(rows).to_csv(cfg["out"]/"error_budget.csv",index=False)
    return pd.DataFrame(rows)

def run_diagnostics(cfg,selected):
    rows=[]; n=cfg["ns"][min(1,len(cfg["ns"])-1)]; g=make_grid(n)
    data=make_dataset("in",0.,101,cfg["ref_n"])
    key=f'{data["dataset_id"]}_n{n}_expm_s{cfg["seeds"][0]}'
    hats=np.asarray(selected[key]["result"]["hats"])
    for bits in [2,4,6,8,10,12]:
        quant=np.round(hats*(2**bits-1))/(2**bits-1)
        exact=[]; v=g["v0"].copy(); previous=0.
        for h,q in zip(quant,data["quotes"]):
            v=sla.expm(-(q["T"]-previous)*assemble(g,h))@v; previous=q["T"]
            exact.append(price_vector(g,v))
        result=dict(hats=quant,C=np.array(exact),C_exact=np.array(exact),converged=True)
        row=evaluate_model(g,data,result,"quantised-expm",0,0.)
        row.update(bits=bits,design_qubits=bits*g["M"],hxi=1/(2**bits-1),
                   inverse_hxi_squared=(2**bits-1)**2,continuous_reference=selected[key]["row"]["rmse_quotes"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(cfg["out"]/"quantisation.csv",index=False)
    # Smoothness: global Fourier approximation of the compactly supported bump.
    x=np.linspace(-1.2,1.2,4096,endpoint=False); phi=basis(x)[1]; ft=np.fft.fft(phi)/len(x)
    freqs=np.fft.fftfreq(len(x))*len(x); fourier=[]
    for degree in [2,4,8,16,32,64,128]:
        approx=np.fft.ifft(ft*(np.abs(freqs)<=degree)*len(x)).real
        fourier.append(dict(degree=degree,error_inf=float(np.max(np.abs(approx-phi))),
                            minimum_basis=float(approx.min()),variance_floor=.01,
                            coefficient_error_budget=float(np.max(np.abs(approx-phi))*4),
                            operator_error_bound=float(2*np.max(np.abs(approx-phi))*np.linalg.norm(g["Lt"],2))))
    pd.DataFrame(fourier).to_csv(cfg["out"]/"fourier_basis.csv",index=False)
    # FD accuracy at one fixed coefficient, separate from optimisation.
    q=data["quotes"][0]; A=assemble(g,hats[0]); ex=sla.expm(-q["T"]*A)@g["v0"]
    fdrows=[]
    for steps in ([8,16,32,64] if cfg["profile"]=="smoke" else [32,64,128,256,512,1024,2048]):
        _,_,v=fd_value_grad(g,hats[0],g["v0"],q["T"],np.eye(g["N"]),np.zeros(g["N"]),np.ones(g["N"]),0.,steps)
        fdrows.append(dict(steps=steps,price_error_inf=float(np.max(np.abs(v-ex)))))
    pd.DataFrame(fdrows).to_csv(cfg["out"]/"fd_convergence.csv",index=False)
    # Quote Jacobian singular values identify unobserved parameter directions.
    jacrows=[]; v=jnp.asarray(g["v0"]); previous=0.
    for s,q in enumerate(data["quotes"]):
        fw,_=solver_functions(n,4,"expm",1,1.,float(q["T"]-previous)); previous=q["T"]
        W=jnp.asarray(interpolation(g["K"],q["k"])[:,1:-1])
        J=np.asarray(jax.jacrev(lambda z:W@fw(z,v))(jnp.array(hats[s])))
        sv=sla.svdvals(J)
        for j,val in enumerate(sv): jacrows.append(dict(slice=s+1,index=j+1,singular_value=val,condition=float(sv[0]/max(sv[-1],1e-15))))
        v=fw(jnp.array(hats[s]),v)
    pd.DataFrame(jacrows).to_csv(cfg["out"]/"identifiability.csv",index=False)
    # Same out-of-span data: M and regularisation are varied independently.
    ab=[]
    if cfg["ablation"]:
        d=make_dataset("out",5.,101,cfg["ref_n"])
        for M in [2,4,8]:
            gm=make_grid(n,M)
            for lam in [0.,1e-7,1e-5,1e-3]:
                for method in ["expm","lchs-beta"]:
                    r=calibrate_model(gm,d,method,cfg,0,lam)
                    ab.append(evaluate_model(gm,d,r,method,0,lam))
                    pd.DataFrame(ab).to_csv(cfg["out"]/"ablation.csv",index=False)
    return pd.DataFrame(rows)

MARKET_COLUMNS=["date","expiry","underlying","strike","forward","discount","bid","ask",
                "option_type","exercise_style","carry_model","source"]

def load_market_csv(path):
    frame=pd.read_csv(path)
    missing=set(MARKET_COLUMNS)-set(frame.columns)
    if missing: raise ValueError("Missing market columns: "+str(sorted(missing)))
    for c in ["date","expiry"]: frame[c]=pd.to_datetime(frame[c],errors="raise")
    for c in ["strike","forward","discount","bid","ask"]:
        frame[c]=pd.to_numeric(frame[c],errors="raise")
        if not np.isfinite(frame[c]).all(): raise ValueError("Non-finite "+c)
    if not (frame.exercise_style.str.lower()=="european").all(): raise ValueError("Only European quotes are supported")
    if not (frame.carry_model=="deterministic_proportional").all():
        raise ValueError("Declare deterministic_proportional carry; discrete cash dividends are unsupported")
    if ((frame[["strike","forward","discount"]]<=0).any().any() or (frame.bid<0).any() or (frame.ask<frame.bid).any()):
        raise ValueError("Invalid prices, discount, forward or crossed bid/ask")
    if frame.duplicated(["date","expiry","underlying","strike"]).any():
        raise ValueError("Duplicate contract/strike rows. Resolve explicitly; do not average silently")
    if frame.source.fillna("").str.strip().eq("").any(): raise ValueError("A data source is required on every row")
    datasets=[]
    for (date,underlying),day in frame.groupby(["date","underlying"]):
        qs=[]; tests=[]
        for expiry,part in day.groupby("expiry"):
            part=part.sort_values("strike").copy()
            T=(expiry-date).days/365.
            if T<=0: raise ValueError("Expired or same-day quotes")
            if len(part)<6: raise ValueError("Need at least six quotes per expiry for held-out evaluation")
            if part.forward.nunique()!=1 or part.discount.nunique()!=1: raise ValueError("Inconsistent forward/discount per expiry")
            norm=part.forward.to_numpy()*part.discount.to_numpy()
            parity=part.discount.to_numpy()*(part.forward.to_numpy()-part.strike.to_numpy())
            cp=part.option_type.str.lower()
            if not cp.isin(["call","put"]).all(): raise ValueError("option_type must be call or put")
            bid=(part.bid.to_numpy()+np.where(cp=="put",parity,0))/norm
            ask=(part.ask.to_numpy()+np.where(cp=="put",parity,0))/norm
            k=part.strike.to_numpy()/part.forward.to_numpy(); mid=(bid+ask)/2
            if k.min()<np.exp(-1.2) or k.max()>np.exp(1.2): raise ValueError("Market strikes exceed the supported domain")
            held=np.arange(len(k))%4==1  # deterministic split before any fitting
            qs.append(dict(T=T,k=k[~held],mid=mid[~held],bid=bid[~held],ask=ask[~held]))
            tests.append(dict(T=T,k=k[held],price=mid[held],bid=bid[held],ask=ask[held]))
        if len(qs)<2: raise ValueError("Need at least two expiries per market date")
        ident=hashlib.sha256(f"{date}-{underlying}".encode()).hexdigest()[:12]
        datasets.append(dict(dataset_id="market_"+ident,kind="market",truth="market",noise_bp=0.,noise_seed=0,
                             date=str(date.date()),underlying=str(underlying),quotes=qs,tests=tests))
    return datasets

def run_market(cfg):
    path=cfg.get("market_csv",""); out=cfg["out"]
    if not path:
        pd.DataFrame(columns=MARKET_COLUMNS).to_csv(out/"market_template.csv",index=False)
        save_json(out/"market_status.json",dict(status="not_run",reason="No real market CSV supplied; template has headers only"))
        print("Market study: NOT RUN. Set MARKET_CSV to a bid-ask CSV. No synthetic substitute is used.")
        return pd.DataFrame()
    datasets=load_market_csv(path); g=make_grid(cfg["ns"][min(1,len(cfg["ns"])-1)])
    rows=[]; failed=[]
    for d in datasets:
        save_json(out/"raw"/(d["dataset_id"]+".json"),d)
        for method in ["expm","fd","lchs-beta","AH","PINN-Wam"]:
            try:
                if method=="AH": r=ah_calibrate(g,d,cfg)
                elif method=="PINN-Wam": r=pinn_calibrate(g,d,cfg,0,True)
                else: r=calibrate_model(g,d,method,cfg,0,1e-7)
                row=evaluate_model(g,d,r,method,0,1e-7); row.update(date=d["date"],kind="market")
                hits=[]
                for s,test in enumerate(d["tests"]):
                    p=(np.asarray(r["test_predictions"][s]) if "test_predictions" in r
                       else interpolation(g["K"],test["k"])@r["C"][s])
                    hits.extend((p>=test["bid"])&(p<=test["ask"]))
                row["heldout_bidask_fraction"]=float(np.mean(hits)); rows.append(row)
                save_json(out/"checkpoints"/(d["dataset_id"]+"_"+method+".json"),dict(row=row,result=r))
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                failed.append(dict(dataset=d["dataset_id"],method=method,reason=str(exc)))
        pd.DataFrame(rows).to_csv(out/"market_results.csv",index=False)
    save_json(out/"market_status.json",dict(status="completed" if rows else "failed",dates=len(datasets),
              source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),failures=failed))
    return pd.DataFrame(rows)
