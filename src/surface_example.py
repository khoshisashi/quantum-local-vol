"""Synthetic smile/skew examples in a positive quadratic LV family.

Calibrate with classical exact propagation, then compare exact, backward Euler
and finite LCHS at the SAME fitted coefficients. This experiment isolates the
surface representation and forward approximation; it is not a measured-circuit
run or an independently optimised LCHS calibration.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal,eigvalsh
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import least_squares
from surface_model import observation,weights,save_json

TIMES=np.array([.25,.5,.75,1.])
QUOTES=np.array([.70,.78,.87,.95,1.,1.06,1.15,1.28,1.4])
COEFFICIENTS={
 'smile':np.array([[.10,.50,.50],[.16,.50,.50],[.22,.50,.50],[.28,.50,.50]]),
 'skew':np.array([[.20,.10,.85],[.22,.12,.80],[.24,.14,.75],[.26,.16,.70]])}

def make_grid(N):
    x=np.linspace(-1.2,1.2,N+2);k=np.exp(x);h=x[1]-x[0]
    drift=2*np.tanh(h/2)/h
    Q=(np.full(N-1,-1/h**2-drift/(2*h)),np.full(N,2/h**2),np.full(N-1,-1/h**2+drift/(2*h)))
    xi=x[1:-1]
    phi=np.array([np.full(N,.08),.1*(xi+.45)**2,.1*(xi-.45)**2])
    G=(1-k[0])*(k[-1]-k)/(k[-1]-k[0]);v0=np.maximum(1-k[1:-1],0)-G[1:-1]
    return dict(N=N,K=k,x=x,h=h,Q=Q,phi=phi,G=G,v0=v0)

def entries(g,xi):
    a=.01+np.asarray(xi)@g['phi'];lo,d,up=g['Q']
    return .5*a[1:]*lo,.5*a*d,.5*a[:-1]*up

def advance(g,xi,dt,v,method='exact',J=2048,cutoff=48.):
    lo,d,up=entries(g,xi)
    if method in ['exact','fd']:
        log_r=np.r_[0.,np.cumsum(.5*(np.log(-lo)-np.log(-up)))];log_r-=np.mean(log_r);r=np.exp(log_r)
        ev,Q=eigh_tridiagonal(d,-np.sqrt(lo*up))
        factor=np.exp(-dt*ev) if method=='exact' else (1+dt*ev/1024)**-1024
        return r*(Q@(factor*(Q.T@(v/r))))
    # Complex Hermitian tridiagonals are reduced by a diagonal unitary gauge.
    shift=g['shift'];ks,ws,_=weights(J,cutoff)
    result=np.zeros(g['N'],complex)
    for k,w in zip(ks,ws):
        b=k*(up+lo)/2+.5j*(lo-up)
        phase=np.r_[1.+0j,np.exp(-1j*np.cumsum(np.angle(b)))]
        ev,Q=eigh_tridiagonal(k*(d+shift),abs(b))
        result+=w*phase*(Q@(np.exp(-1j*dt*ev)*(Q.T@(phase.conj()*v))))
    return np.exp(shift*dt)*result

def add_shift(g):
    vals=[np.full(g['N'],.01),*g['phi']]
    lo,d,up=g['Q'];shift=0.
    for a in vals:
        diag=.5*a*d;off=.25*(a[:-1]*up+a[1:]*lo)
        smallest=eigh_tridiagonal(diag,off,select='i',select_range=(0,0),eigvals_only=True)[0]
        shift+=max(0.,-smallest)
    g['shift']=shift;return g

def surface(g,coefficients,method='exact',times=TIMES,J=2048,cutoff=48.):
    v=g['v0'].astype(complex);last=0.;out=[]
    for s,T in enumerate(times):
        v=advance(g,coefficients[s],T-last,v,method,J,cutoff);last=T
        c=g['G'].astype(complex).copy();c[1:-1]+=v;out.append(c)
    return np.array(out)

def run(out,profile='paper'):
    import time
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    N=128 if profile!='smoke' else 64;refN=1024 if profile!='smoke' else 256
    J=8192 if profile!='smoke' else 4096;cutoff=96.
    target_grid=make_grid(refN);g=add_shift(make_grid(N));O=observation(g,QUOTES)
    ks=np.linspace(.7,1.4,141);W=observation(g,ks);Wref=observation(target_grid,ks)
    rows=[];fits=[];timings=[]
    for case,truth in COEFFICIENTS.items():
        target=surface(target_grid,truth).real
        quotes=target@observation(target_grid,QUOTES).T
        v=g['v0'].astype(complex);last=0.;coeff=[];t0=time.perf_counter()
        for s,T in enumerate(TIMES):
            def residual(x):
                c=g['G'].copy();c[1:-1]+=advance(g,x,T-last,v).real
                return (O@c-quotes[s])*1e4
            fit=least_squares(residual,np.full(g['phi'].shape[0],.5),bounds=(0,1),ftol=1e-10,xtol=1e-10,gtol=1e-9,max_nfev=100)
            coeff.append(fit.x);v=advance(g,fit.x,T-last,v);last=T
            fits.append(dict(case=case,T=T,parameters=fit.x.tolist(),success=bool(fit.success),nfev=fit.nfev,cost=fit.cost))
        coeff=np.array(coeff);calib=time.perf_counter()-t0
        for method in ['exact','fd','lchs']:
            t0=time.perf_counter();p=surface(g,coeff,method,J=J,cutoff=cutoff);seconds=time.perf_counter()-t0
            for s,T in enumerate(TIMES):
                reference=Wref@target[s];pred=W@p[s]
                rows.extend(dict(case=case,method=method,N=N,J=J,cutoff=cutoff,T=T,k=k,
                    target_price=float(t),price=float(z.real),imaginary_price=float(z.imag),
                    fit_scope='exact-calibrated common coefficients') for k,t,z in zip(ks,reference,pred))
            error=(p@W.T-target@Wref.T);quoted=(p@O.T-quotes)
            held=~np.any(np.isclose(ks[:,None],QUOTES[None,:],atol=1e-10),axis=1)
            timings.append(dict(case=case,method=method,N=N,J=J,cutoff=cutoff,
                calibration_seconds=calib,forward_seconds=seconds,
                heldout_rmse_bp=float(np.sqrt(np.mean(abs(error[:,held])**2))*1e4),
                quote_rmse_bp=float(np.sqrt(np.mean(abs(quoted)**2))*1e4)))
        print('Surface example complete:',case,flush=True)
        save_json(out/f'{case}_quotes.json',dict(k=QUOTES,times=TIMES,prices=quotes,truth=truth,coefficients=coeff))
    pd.DataFrame(rows).to_csv(out/'surface_predictions.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'surface_accuracy.csv',index=False)
    save_json(out/'calibration_records.json',fits)
    save_json(out/'configuration.json',dict(profile=profile,N=N,reference_N=refN,J=J,cutoff=cutoff,times=TIMES,
        examples='synthetic smile and skew from positive quadratic local variance; no market data',
        basis=['0.08','0.1*(x+0.45)^2','0.1*(x-0.45)^2'],
        calibration='classical exact propagation, sequential bounded least squares, common coefficients',
        methods=['exact','fd','lchs'],shift=g['shift']))
    return pd.DataFrame(timings)


def validate_reference(out,profile='paper'):
    """Independent dense check and refinement of synthetic reference prices."""
    from scipy.linalg import expm
    out=Path(out)
    g=add_shift(make_grid(16));xi=np.array([.2,.4,.7]);lo,d,up=entries(g,xi)
    A=np.diag(d)+np.diag(lo,-1)+np.diag(up,1);v=g['v0'].astype(complex);dt=.25
    exact_error=float(np.max(abs(advance(g,xi,dt,v)-expm(-dt*A)@v)))
    shift=g['shift'];L=(A+A.T)/2+shift*np.eye(16);H=(A-A.T)/(2j)
    nodes,ws,_=weights(8,12.)
    dense=sum(w*expm(-1j*dt*(H+k*L))@v for k,w in zip(nodes,ws))*np.exp(shift*dt)
    lchs_error=float(np.max(abs(advance(g,xi,dt,v,'lchs',8,12.)-dense)))
    assert exact_error<1e-10 and lchs_error<1e-10
    n=1024 if profile=='paper' else 256;g1=make_grid(n);g2=make_grid(2*n)
    W1=observation(g1,QUOTES);W2=observation(g2,QUOTES)
    refinement={}
    for case,truth in COEFFICIENTS.items():
        p1=surface(g1,truth)@W1.T;p2=surface(g2,truth)@W2.T
        refinement[case]=dict(rmse_bp=float(np.sqrt(np.mean(abs(p1-p2)**2))*1e4),
                             max_error_bp=float(np.max(abs(p1-p2))*1e4))
    preds=pd.read_csv(out/'surface_predictions.csv')
    result=dict(profile=profile,exact_dense_max_error=exact_error,lchs_dense_max_error=lchs_error,
                reference_nodes=[n,2*n],reference_refinement=refinement,
                max_imaginary_price=float(preds.imaginary_price.abs().max()),
                all_fits_converged=all(z['success'] for z in __import__('json').loads((out/'calibration_records.json').read_text())))
    assert result['all_fits_converged'] and result['max_imaginary_price']<1e-10
    save_json(out/'validation.json',result)
    return result
