"""Financial model shared by classical references and measured quantum experiments.

The small circuit experiment deliberately uses same-grid synthetic quotes to
isolate implementation error. It is not a fine-grid accuracy benchmark.
"""
from pathlib import Path
import json, hashlib, time, platform, importlib.metadata
import numpy as np
import pandas as pd
from scipy.linalg import expm, eigvalsh, eigh
from scipy.optimize import brentq
from scipy.special import ndtr

VERSION = '2026-09-09-quantum-surface-r1'
TIMES = np.array([0.5, 1.0])
QUOTES = [np.array([0.80, 0.95, 1.10, 1.30]),
          np.array([0.85, 1.00, 1.15, 1.35])]
TRUTH = np.array([[0.75, 0.25], [0.25, 0.75]])

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=convert, indent=2, allow_nan=False))
    tmp.replace(path)

def grid(n=2, M=2, extent=1.2):
    N = 2**n
    x = np.linspace(-extent, extent, N+2); K = np.exp(x)
    h = x[1]-x[0]; fitted = 2*np.tanh(h/2)/h
    L = np.diag(np.full(N, 2/h**2))
    L += np.diag(np.full(N-1, -1/h**2+fitted/(2*h)), 1)
    L += np.diag(np.full(N-1, -1/h**2-fitted/(2*h)), -1)
    centres = np.linspace(-0.55, 0.55, M)
    width = 0.85 if M == 2 else 1.8/max(1,M-1)
    z = (x[None,1:-1]-centres[:,None])/width
    phi = 0.04*(1+np.cos(np.pi*np.clip(z,-1,1)))*(abs(z)<=1)
    slope = (1-K[0])/(K[0]-K[-1]); G = slope*(K-K[-1])
    A0 = .005*L; Am = .5*phi[:,:,None]*L[None,:,:]
    shift = sum(max(0., -eigvalsh((a+a.T)/2)[0]) for a in [A0,*Am])
    return dict(n=n,N=N,M=M,K=K,x=x,h=h,L=L,phi=phi,G=G,
                A0=A0,Am=Am,shift=shift,
                v0=np.maximum(1-K[1:-1],0)-G[1:-1])

def generator(g, xi): return g['A0']+np.einsum('m,mij->ij',xi,g['Am'])

def observation(g, strikes):
    K=g['K']; strikes=np.asarray(strikes)
    if np.any((strikes<K[0])|(strikes>K[-1])): raise ValueError('Quote outside domain')
    j=np.clip(np.searchsorted(K,strikes)-1,0,len(K)-2)
    t=(strikes-K[j])/(K[j+1]-K[j])
    W=np.zeros((len(strikes),len(K)))
    W[np.arange(len(j)),j]=1-t; W[np.arange(len(j)),j+1]=t
    return W

def weights(terms=8, cutoff=4., beta=.8):
    k=-cutoff+(np.arange(terms)+.5)*(2*cutoff/terms)
    w=np.exp(2**beta-(1+1j*k)**beta)/(2*np.pi*(1-1j*k))*(2*cutoff/terms)
    defect=abs(w.sum()-1); w=w/w.sum()
    return k,w,dict(alpha=float(abs(w).sum()),normalisation_defect=float(defect))

def propagator(g, xi, dt, method='exact', terms=8, cutoff=4., fd_steps=128):
    A=generator(g,xi)
    if method=='exact': return expm(-dt*A)
    if method=='fd': return np.linalg.matrix_power(np.linalg.inv(np.eye(g['N'])+dt*A/fd_steps),fd_steps)
    k,w,_=weights(terms,cutoff)
    H=-.5j*(A-A.T); L=(A+A.T)/2+g['shift']*np.eye(g['N'])
    vals,vecs=np.linalg.eigh(H[None,:,:]+k[:,None,None]*L[None,:,:])
    U=np.einsum('kij,kj,klj->kil',vecs,np.exp(-1j*dt*vals),vecs.conj())
    return np.exp(dt*g['shift'])*np.einsum('k,kij->ij',w,U)

def prices(g, coefficients, times=TIMES, method='exact', terms=8, cutoff=4.):
    coefficients=np.asarray(coefficients).reshape(len(times),g['M'])
    v=g['v0'].astype(complex); last=0.; result=[]
    for xi,T in zip(coefficients,times):
        v=propagator(g,xi,T-last,method,terms,cutoff)@v; last=T
        c=g['G'].astype(complex).copy(); c[1:-1]+=v
        result.append(c)
    return np.asarray(result)

def data_for(g, truth=TRUTH, noise_bp=0., seed=101):
    p=prices(g,truth); rng=np.random.default_rng(seed)
    obs=[observation(g,k) for k in QUOTES]
    quotes=[(W@p[i]).real+rng.normal(0,noise_bp/1e4,len(W)) for i,W in enumerate(obs)]
    return dict(g=g,quotes=quotes,observations=obs,truth=np.asarray(truth),
                noise_bp=noise_bp,data_seed=seed,times=TIMES.copy())

def loss(problem, xi, method='lchs', terms=8, cutoff=4.):
    p=prices(problem['g'],xi,problem['times'],method,terms,cutoff)
    return float(np.mean([np.mean(abs(W@p[s]-problem['quotes'][s])**2)
                          for s,W in enumerate(problem['observations'])]))

def financial_metrics(problem, xi, method='exact', terms=8, cutoff=4.):
    g=problem['g']; p=prices(g,xi,problem['times'],method,terms,cutoff)
    target=prices(g,problem['truth'],problem['times'])
    strikes=np.linspace(.7,1.4,81); W=observation(g,strikes)
    error=(p-target)@W.T
    quoted=np.concatenate([O@p[s]-problem['quotes'][s] for s,O in enumerate(problem['observations'])])
    return dict(quote_rmse_bp=float(np.sqrt(np.mean(abs(quoted)**2))*1e4),
                heldout_rmse_bp=float(np.sqrt(np.mean(abs(error)**2))*1e4),
                coefficient_rmse=float(np.sqrt(np.mean((np.asarray(xi).reshape(problem['truth'].shape)-problem['truth'])**2))))

def black_call(k,T,vol):
    root=vol*np.sqrt(T); d1=-np.log(k)/root+root/2
    return ndtr(d1)-k*ndtr(d1-root)

def implied_vol(k,T,c):
    if not np.isfinite(c) or c<=max(1-k,0)+1e-13 or c>=1-1e-13:return np.nan
    try:return brentq(lambda v:black_call(k,T,v)-c,1e-5,5.)
    except ValueError:return np.nan
