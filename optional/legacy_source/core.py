"""Reproducible local-volatility calibration; embedded verbatim in the notebook.

All prices are forward-normalised. No market observations are bundled.
"""
from pathlib import Path
from functools import lru_cache, partial
import os, sys, json, time, hashlib, platform, importlib.metadata, warnings
import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize, brentq
from scipy.special import ndtr
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib
import matplotlib.pyplot as plt

VERSION = "2026-09-05-r2"
TIMES = np.array([182.0 / 365.0, 1.0])
TRAIN_K = [np.array([.7,.85,.95,1.,1.05,1.2,1.4]),
           np.array([.75,.9,1.,1.1,1.3])]
XI_TRUTH = np.array([[.8,.3,.25,.7],[.6,.35,.3,.5]])
PROFILES = {
    "smoke": dict(ns=[3,4], ref_n=7, seeds=[0], noise_seeds=[101],
                  kernels=[16,32,64], lchs_terms=64, maxiter=12, fd_steps=24,
                  pinn_steps=30, pinn_width=12, collocation=48, ablation=False,
                  circuit_n=2, circuit_m=1, circuit_terms=4, qhd_steps=80),
    "pilot": dict(ns=[5,6], ref_n=9, seeds=[0,1], noise_seeds=[101],
                  kernels=[128,256,512,1024], lchs_terms=512, maxiter=120,
                  fd_steps=256, pinn_steps=1500, pinn_width=24, collocation=256,
                  ablation=True, circuit_n=2, circuit_m=1, circuit_terms=4,
                  qhd_steps=600),
    "paper": dict(ns=[5,6,7], ref_n=11, seeds=list(range(5)),
                  noise_seeds=[101,202,303], kernels=[256,512,1024,2048,4096],
                  lchs_terms=2048, maxiter=220, fd_steps=1024,
                  pinn_steps=6000, pinn_width=32, collocation=768,
                  ablation=True, circuit_n=2, circuit_m=1, circuit_terms=4,
                  qhd_steps=1500),
}

def jsonable(v):
    if isinstance(v, dict): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [jsonable(x) for x in v]
    if isinstance(v, (np.ndarray,jax.Array)): return jsonable(np.asarray(v).tolist())
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    return v

def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(jsonable(obj),indent=2,allow_nan=True),encoding="utf-8")
    tmp.replace(path)

def configure(profile="paper", output_root="cifer_outputs", market_csv=""):
    if profile not in PROFILES: raise ValueError("profile must be smoke, pilot or paper")
    cfg=dict(PROFILES[profile], profile=profile, version=VERSION, market_csv=market_csv)
    if market_csv:
        cfg["market_input_sha256"]=hashlib.sha256(Path(market_csv).read_bytes()).hexdigest()
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
    out=Path(output_root)/f"{profile}_{digest}"
    for name in ["figures","tables","checkpoints","raw","diagnostics"]:
        (out/name).mkdir(parents=True,exist_ok=True)
    cfg["out"]=out
    cfg["fingerprint"]=digest
    cfg["backend"]=str(jax.devices())
    versions={m:importlib.metadata.version(m) for m in
              ["numpy","scipy","pandas","matplotlib","jax","qiskit","QuantLib"]}
    save_json(out/"manifest.json",dict(config=cfg,packages=versions,
              python=sys.version,platform=platform.platform(),
              study_status="not_completed",market_status="not_run",
              time_units="ACT/365F years",price_units="C/(discount*forward)"))
    print("Profile:",profile,"| device:",jax.devices(),"| output:",out)
    return cfg

def basis(x, M=4):
    centres=np.array([-.6,-.2,.2,.6]) if M==4 else np.linspace(-.6,.6,M)
    width=.45 if M==4 else 1.8/max(M-1,1)
    z=(np.asarray(x)[None,:]-centres[:,None])/width
    phi=.04*(1+np.cos(np.pi*np.clip(z,-1,1)))*(np.abs(z)<=1)
    return phi

def curvature_matrix(K):
    K=np.asarray(K); dm=np.diff(K)[:-1]; dp=np.diff(K)[1:]; N=len(K)-2
    B=np.zeros((N,N+2))
    ii=np.arange(N)
    B[ii,ii]=2/(dm*(dm+dp)); B[ii,ii+1]=-2/(dm*dp)
    B[ii,ii+2]=2/(dp*(dm+dp))
    return B

@lru_cache(maxsize=32)
def make_grid(n, M=4, xmin=-1.2, xmax=1.2):
    N=2**n; h=(xmax-xmin)/(N+1)
    xe=np.linspace(xmin,xmax,N+2); K=np.exp(xe); x=xe[1:-1]
    kap=2*np.tanh(h/2)/h
    Lt=np.diag(np.full(N,2/h**2))
    Lt+=np.diag(np.full(N-1,-1/h**2+kap/(2*h)),1)
    Lt+=np.diag(np.full(N-1,-1/h**2-kap/(2*h)),-1)
    beta=(1-K[0])/(K[0]-K[-1]); G=-beta*K[-1]+beta*K
    phi=basis(x,M)
    A0=.005*Lt; Am=.5*phi[:,:,None]*Lt[None,:,:]
    cbar=sum(max(0.,-sla.eigvalsh((a+a.T)/2,subset_by_index=[0,0])[0])
             for a in [A0,*Am])
    return dict(n=n,N=N,M=M,h=h,x=x,xe=xe,K=K,G=G,phi=phi,
                Lt=Lt,A0=A0,Am=Am,cbar=float(cbar),
                v0=np.maximum(1-K[1:-1],0)-G[1:-1],B=curvature_matrix(K),
                xmin=xmin,xmax=xmax)

def assemble(g, xi): return g["A0"]+np.einsum("m,mij->ij",np.asarray(xi),g["Am"])
def sigma2(g, xi): return .01+np.asarray(xi)@g["phi"]

def interpolation(K, query):
    """Piecewise linear in strike, never log-strike; no snapping or clipping."""
    query=np.asarray(query)
    if np.any(query<K[0]) or np.any(query>K[-1]):
        raise ValueError("Query outside the declared strike domain")
    left=np.clip(np.searchsorted(K,query,side="right")-1,0,len(K)-2)
    theta=(query-K[left])/(K[left+1]-K[left])
    W=np.zeros((len(query),len(K))); rr=np.arange(len(query))
    W[rr,left]=1-theta; W[rr,left+1]=theta
    return W

def price_vector(g,v): return np.r_[g["G"][0],np.asarray(v)+g["G"][1:-1],g["G"][-1]]

def audit_prices(K,Cs,tol=1e-8):
    """All metrics retain units; density tolerance is derived, not reused blindly."""
    Cs=np.asarray(Cs); dm=np.diff(K)[:-1]; dp=np.diff(K)[1:]
    butterfly=(dm*Cs[:,2:]+dp*Cs[:,:-2])/(dm+dp)-Cs[:,1:-1]
    density=2*butterfly/(dm*dp)
    calendar=np.diff(np.vstack([np.maximum(1-K,0),Cs]),axis=0)
    slope=np.diff(Cs,axis=1)/np.diff(K)
    bounds=np.minimum(Cs-np.maximum(1-K,0),1-Cs)
    return dict(butterfly_min=float(butterfly.min()),density_min=float(density.min()),
                calendar_min=float(calendar.min()),vertical_min=float(np.minimum(-slope,1+slope).min()),
                price_bound_min=float(bounds.min()),
                butterfly_violation=float(max(0,-butterfly.min())),
                density_violation=float(max(0,-density.min())),
                calendar_violation=float(max(0,-calendar.min())),
                bfly_count=int((butterfly < -tol).sum()),
                cal_count=int((calendar < -tol).sum()),
                audit_price_tolerance=tol)

def certificate(g, approx, exact, roundoff=2e-10):
    """Conditional certificate given a measured absolute price error, not a priori LCHS proof."""
    ap=np.asarray(approx); ex=np.asarray(exact)
    delta=np.max(np.abs(ap-ex),axis=1)
    dm=np.diff(g["K"])[:-1]; dp=np.diff(g["K"])[1:]
    b=(dm*ap[:,2:]+dp*ap[:,:-2])/(dm+dp)-ap[:,1:-1]
    den=2*b/(dm*dp)
    cal=np.diff(np.vstack([np.maximum(1-g["K"],0),ap]),axis=0)
    cb=delta+np.r_[0,delta[:-1]]
    ok=(np.all(b>=-2*delta[:,None]-roundoff) and
        np.all(den>=-4*delta[:,None]/(dm*dp)-roundoff*2/(dm*dp)) and
        np.all(cal>=-cb[:,None]-roundoff))
    return dict(price_error_inf=float(delta.max()),butterfly_bound=float(2*delta.max()),
                calendar_bound=float(cb.max()),density_bound=float((4*delta[:,None]/(dm*dp)).max()),
                certificate_pass=bool(ok))

def lchs_weights(kernel="beta", terms=512, cutoff=48., beta=.8):
    """Symmetric midpoint rule makes conjugate pairing exact up to rounding.
    Normalisation enforces the identity at t=0; it does not recover omitted tails.
    """
    k=-cutoff+(np.arange(terms)+.5)*2*cutoff/terms
    f=1/(np.pi*(1+1j*k)) if kernel=="simple" else np.exp(2**beta-(1+1j*k)**beta)/(2*np.pi)
    w=(2*cutoff/terms)*f/(1-1j*k); total=w.sum()
    if abs(total)<1e-8: raise ValueError("Unstable quadrature normalisation")
    return k,w/total,dict(terms=terms,cutoff=cutoff,dk=2*cutoff/terms,
                          normalisation_defect=float(abs(total-1)),alpha_lcu=float(np.abs(w/total).sum()))

@jax.custom_jvp
def hermitian_unitary(X):
    lam,V=jnp.linalg.eigh(X)
    return (V*jnp.exp(-1j*lam)[None,:])@V.conj().T

@hermitian_unitary.defjvp
def _hermitian_unitary_jvp(primals,tangents):
    (X,),(dX,)=primals,tangents
    lam,V=jnp.linalg.eigh(X)
    U=(V*jnp.exp(-1j*lam)[None,:])@V.conj().T
    diff=lam[:,None]-lam[None,:]; avg=(lam[:,None]+lam[None,:])/2
    divided=-1j*jnp.exp(-1j*avg)*jnp.sinc(diff/(2*jnp.pi))
    dU=V@(divided*(V.conj().T@dX@V))@V.conj().T
    return U,dU

def lchs_apply_jax(A,v,dt,k,w,shift,chunk=16):
    As=A+shift*jnp.eye(A.shape[0]); L=(As+As.T)/2; H=-.5j*(As-As.T)
    pad=(-len(k))%chunk
    kk=jnp.pad(jnp.asarray(k),(0,pad)).reshape(-1,chunk)
    ww=jnp.pad(jnp.asarray(w),(0,pad)).reshape(-1,chunk)
    def part(carry,kw):
        kp,wp=kw
        U=jax.vmap(hermitian_unitary)(dt*(H[None,:,:]+kp[:,None,None]*L))
        return carry+jnp.einsum("k,kij,j->i",wp,U,v),None
    val,_=jax.lax.scan(jax.checkpoint(part),jnp.zeros(v.shape,dtype=jnp.complex128),(kk,ww))
    return jnp.exp(shift*dt)*jnp.real(val)

@lru_cache(maxsize=48)
def solver_functions(n,M,method,terms,cutoff,dt,fd_steps=256,xmin=-1.2,xmax=1.2):
    g=make_grid(n,M,xmin,xmax)
    if method=="fd": return None
    A0=jnp.asarray(g["A0"]); Am=jnp.asarray(g["Am"]); shift=g["cbar"]
    if method.startswith("lchs"):
        k,w,_=lchs_weights(method.split("-")[1],terms,cutoff)
    def forward(xi,v):
        A=A0+jnp.einsum("m,mij->ij",xi,Am)
        if method=="expm": return jax.scipy.linalg.expm(-dt*A,max_squarings=24)@v
        return lchs_apply_jax(A,v,dt,k,w,shift)
    def objective(xi,v,W,y,weight,lam):
        residual=(W@forward(xi,v)-y)*weight
        smooth=jnp.diff(xi,n=2)
        return jnp.mean(residual**2)+lam*jnp.mean(smooth**2) if M>2 else jnp.mean(residual**2)
    return jax.jit(forward),jax.jit(jax.value_and_grad(objective))

def fd_value_grad(g,xi,v0,dt,W,y,weight,lam,steps):
    """Backward Euler, one sparse LU, exact discrete adjoint of the implemented loss."""
    A=assemble(g,xi); dh=dt/steps
    fac=spla.splu(sp.eye(g["N"],format="csc")+dh*sp.csc_matrix(A))
    trajectory=[np.asarray(v0)]
    for _ in range(steps): trajectory.append(fac.solve(trajectory[-1]))
    v=trajectory[-1]; r=W@v-y
    value=float(np.mean((weight*r)**2)); adj=2*W.T@(weight**2*r)/len(y)
    grad=np.zeros(g["M"])
    for state in reversed(trajectory[1:]):
        adj=fac.solve(adj,trans="T")
        grad-=dh*.5*(g["phi"]@(adj*(g["Lt"]@state)))
    if g["M"]>2:
        D=np.diff(np.eye(g["M"]),n=2,axis=0); d=D@xi
        value+=lam*np.mean(d*d); grad+=2*lam*D.T@d/len(d)
    return value,grad,v

def normalised_call(k,T,vol):
    if T<=0: return np.maximum(1-np.asarray(k),0)
    s=np.asarray(vol)*np.sqrt(T)
    d1=-np.log(k)/s+s/2
    return ndtr(d1)-np.asarray(k)*ndtr(d1-s)

def implied_vol(price,k,T):
    lo=max(1-k,0)
    if not np.isfinite(price) or price<=lo+1e-12 or price>=1-1e-12: return np.nan
    try: return brentq(lambda v:normalised_call(k,T,v)-price,1e-6,8.,xtol=1e-12)
    except ValueError: return np.nan

def true_sigma2(x,s,truth):
    if truth=="in": return .01+XI_TRUTH[min(s,1)]@basis(np.asarray(x),4)
    return .01+.055*np.exp(-((np.asarray(x)+.1)/.3)**2)+.035/(1+np.exp(4*np.asarray(x)))

@lru_cache(maxsize=12)
def reference_surface(truth,ref_n,times_tuple=tuple(TIMES),xmin=-1.2,xmax=1.2):
    """Fixed fine strike-space Dupire discretisation, rho=1, sparse exponential action."""
    N=2**ref_n; K=np.exp(np.linspace(xmin,xmax,N+2)); B=curvature_matrix(K)
    beta=(1-K[0])/(K[0]-K[-1]); G=beta*(K-K[-1]); v=np.maximum(1-K[1:-1],0)-G[1:-1]
    Cs=[]; previous=0.
    for T in times_tuple:
        stops=([float(TIMES[0])] if previous<TIMES[0]<T else [])+[T]
        for stop in stops:
            s=int(previous>=TIMES[0]-1e-12)
            a=.5*true_sigma2(np.log(K[1:-1]),s,truth)*K[1:-1]**2
            Q=sp.diags(a)@sp.csc_matrix(B[:,1:-1])
            v=spla.expm_multiply((stop-previous)*Q,v,traceA=float((stop-previous)*Q.diagonal().sum()))
            previous=stop
        Cs.append(np.r_[G[0],v+G[1:-1],G[-1]])
    return K,np.asarray(Cs)

def make_dataset(truth,noise_bp,noise_seed,ref_n):
    K,Cs=reference_surface(truth,ref_n); rng=np.random.default_rng(noise_seed)
    quotes=[]
    for s,T in enumerate(TIMES):
        ks=TRAIN_K[s]; exact=interpolation(K,ks)@Cs[s]
        perturb=rng.uniform(-noise_bp*1e-4,noise_bp*1e-4,len(ks))
        mid=exact+perturb; half=np.full(len(ks),noise_bp*1e-4)
        # Do not clip inconsistent noisy observations into the feasible set.
        quotes.append(dict(T=float(T),k=ks,mid=mid,bid=mid-half,ask=mid+half,clean=exact))
    kid=np.linspace(.7,1.4,181)
    tests=[]
    for s,q in enumerate(quotes):
        mask=~np.any(np.isclose(kid[:,None],q["k"][None,:],atol=1e-10,rtol=0),axis=1)
        kh=kid[mask]; tests.append(dict(T=q["T"],k=kh,price=interpolation(K,kh)@Cs[s]))
    return dict(truth=truth,noise_bp=noise_bp,noise_seed=noise_seed,quotes=quotes,tests=tests,
                dataset_id=f"{truth}_bp{noise_bp:g}_d{noise_seed}",kind="synthetic")

def calibrate_model(g,data,method,cfg,seed=0,lam=0.,start=None):
    rng=np.random.default_rng(seed); v=g["v0"].copy(); hats=[]; outputs=[]; traces=[]
    compilation=0.; opt_time=0.; forward_calls=0; previous=0.; status=[]
    for s,q in enumerate(data["quotes"]):
        dt=q["T"]-previous; previous=q["T"]
        Wfull=interpolation(g["K"],q["k"]); W=Wfull[:,1:-1]
        y=q["mid"]-Wfull@g["G"]
        weight=np.ones(len(y))
        if data.get("kind")=="market":
            weight=1/np.maximum((q["ask"]-q["bid"])/2,1e-5)
        z0=np.clip(.5+.15*rng.standard_normal(g["M"]),0,1) if start is None else np.array(start[s])
        args=(jnp.asarray(v),jnp.asarray(W),jnp.asarray(y),jnp.asarray(weight),float(lam))
        if method=="fd":
            valuegrad=lambda z:fd_value_grad(g,z,v,dt,W,y,weight,lam,cfg["fd_steps"])[:2]
            t=time.perf_counter(); valuegrad(z0); warm=time.perf_counter()-t
        else:
            cutoff=64. if method=="lchs-simple" else 48.
            fwd,vg=solver_functions(g["n"],g["M"],method,cfg["lchs_terms"],cutoff,float(dt),
                                     cfg["fd_steps"],g["xmin"],g["xmax"])
            t=time.perf_counter(); jax.block_until_ready(vg(jnp.asarray(z0),*args)); warm=time.perf_counter()-t
            valuegrad=lambda z:tuple(np.asarray(x) for x in vg(jnp.asarray(z),*args))
        compilation+=warm; records=[]
        def objective(z):
            value,gradient=valuegrad(z); records.append(float(value))
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("Non-finite calibration loss or gradient")
            return float(value),np.asarray(gradient,dtype=float)
        t=time.perf_counter()
        result=minimize(objective,z0,jac=True,bounds=[(0,1)]*g["M"],method="L-BFGS-B",
                        options=dict(maxiter=cfg["maxiter"],ftol=1e-13,gtol=1e-9,maxls=30))
        opt_time+=time.perf_counter()-t; forward_calls+=len(records)
        hats.append(result.x); traces.append(records); status.append(bool(result.success))
        if method=="fd": v=fd_value_grad(g,result.x,v,dt,W,y,weight,lam,cfg["fd_steps"])[2]
        else: v=np.asarray(fwd(jnp.asarray(result.x),jnp.asarray(v)))
        outputs.append(price_vector(g,v))
    exact=[]; v=g["v0"].copy(); previous=0.
    for xi,q in zip(hats,data["quotes"]):
        v=sla.expm(-(q["T"]-previous)*assemble(g,xi))@v; previous=q["T"]
        exact.append(price_vector(g,v))
    return dict(hats=np.array(hats),C=np.array(outputs),C_exact=np.array(exact),
                trace=traces,converged=all(status),slice_converged=status,
                compile_warm_s=compilation,opt_s=opt_time,nfev=forward_calls)

def evaluate_model(g,data,result,method,seed,lam):
    rs=[]; held=[]; ivs=[]; total=0; validiv=0; inside=[]; invalid_obs=0
    Cs=result["C"]
    for s,q in enumerate(data["quotes"]):
        p=(np.asarray(result["quote_predictions"][s]) if "quote_predictions" in result
           else interpolation(g["K"],q["k"])@Cs[s]); rs.extend(p-q["mid"])
        inside.extend((p>=q["bid"]-1e-10)&(p<=q["ask"]+1e-10))
        invalid_obs+=int(((q["mid"]<np.maximum(1-q["k"],0))|(q["mid"]>1)).sum())
        test=data["tests"][s]
        ph=(np.asarray(result["test_predictions"][s]) if "test_predictions" in result
            else interpolation(g["K"],test["k"])@Cs[s])
        held.extend(ph-test["price"])
        for k,mod,ref in zip(test["k"],ph,test["price"]):
            a=implied_vol(mod,k,q["T"]); b=implied_vol(ref,k,q["T"]); total+=1
            if np.isfinite(a) and np.isfinite(b): ivs.append((a-b)**2); validiv+=1
    aud=audit_prices(g["K"],Cs)
    row=dict(dataset=data["dataset_id"],truth=data["truth"],noise_bp=data["noise_bp"],
             noise_seed=data["noise_seed"],method=method,n=g["n"],N=g["N"],M=g["M"],
             seed=seed,lam=lam,rmse_quotes=float(np.sqrt(np.mean(np.square(rs)))),
             rmse_heldout=float(np.sqrt(np.mean(np.square(held)))),
             iv_rmse=float(np.sqrt(np.mean(ivs))) if ivs else np.nan,
             iv_valid=validiv,iv_total=total,bidask_fraction=float(np.mean(inside)),
             invalid_observations=invalid_obs,nfev=result.get("nfev",0),
             compile_warm_s=result.get("compile_warm_s",0),opt_s=result.get("opt_s",0),
             converged=result.get("converged",False),**aud)
    if "C_exact" in result:
        row.update(certificate(g,Cs,result["C_exact"]))
        row.update({"exact_"+k:v for k,v in audit_prices(g["K"],result["C_exact"]).items()})
    if "hats" in result and data["kind"]=="synthetic":
        covered=(g["K"][1:-1]>=.7)&(g["K"][1:-1]<=1.4)
        row["sigma2_linf"]=float(max(np.max(np.abs(sigma2(g,h)-true_sigma2(g["x"],s,data["truth"]))[covered])
                                    for s,h in enumerate(result["hats"])))
    return row
