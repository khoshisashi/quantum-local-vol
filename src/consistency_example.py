"""Fixed-coefficient financial diagnostic of a retained calibration stress case.

No optimisation, new quotes, circuit measurements, or smoothing are performed.
The same fitted first-expiry coefficients are propagated with the old LCHS
budget and a doubled budget; exact and backward Euler are classical references.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from surface_example import make_grid,add_shift,advance
from surface_model import save_json

KEY='out_bp0_d101_n7_lchs-beta_s0'
T=182/365

def run(evidence,out):
    evidence=Path(evidence);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=evidence/'checkpoints'/f'{KEY}.json'
    data=json.loads(checkpoint.read_text());r=data['result'];xi=np.array(r['hats'])[0]
    # Reconstruct the archived four-cosine basis on its fitted log-strike grid.
    g=make_grid(128);x=g['x'][1:-1]
    z=(x[None,:]-np.array([-.6,-.2,.2,.6])[:,None])/.45
    g['phi']=.04*(1+np.cos(np.pi*np.clip(z,-1,1)))*(abs(z)<=1)
    add_shift(g)
    records=[];prices={};timings={};imaginary={}
    for method,J in [('exact',0),('fd',0),('lchs',2048),('lchs',4096)]:
        label=method if not J else f'lchs_{J}'
        start=time.perf_counter();v=advance(g,xi,T,g['v0'],method,J or 2048,48.)
        timings[label]=time.perf_counter()-start;imaginary[label]=float(np.max(abs(np.imag(v))))
        assert imaginary[label]<1e-9
        c=g['G'].copy();c[1:-1]+=np.real(v);prices[label]=c
    exact_diff=float(np.max(abs(prices['exact']-np.array(r['C_exact'])[0])))
    lchs_diff=float(np.max(abs(prices['lchs_2048']-np.array(r['C'])[0])))
    assert exact_diff<1e-9 and lchs_diff<1e-9,(exact_diff,lchs_diff)
    k=g['K'];dm=np.diff(k)[:-1];dp=np.diff(k)[1:]
    interior=(k[:-2]>=.7)&(k[2:]<=1.4);band=(k>=.7)&(k<=1.4)
    checks={}
    for method,c in prices.items():
        bf=(dm*c[2:]+dp*c[:-2])/(dm+dp)-c[1:-1]
        delta=float(np.max(abs(c-prices['exact'])))
        assert bf.min()>=-2*delta-1e-9
        for i in range(len(k)):
            records.append(dict(method=method,T=T,k=k[i],price=c[i],exact_price=prices['exact'][i],
                error_bp=(c[i]-prices['exact'][i])*1e4,
                butterfly_bp=bf[i-1]*1e4 if 0<i<len(k)-1 else np.nan,
                in_quote_band=bool(band[i]),complete_butterfly_in_band=bool(interior[i-1]) if 0<i<len(k)-1 else False))
        checks[method]=dict(max_inband_price_error_bp=float(np.max(abs(c[band]-prices['exact'][band]))*1e4),
            min_inband_butterfly_bp=float(bf[interior].min()*1e4),
            inband_butterfly_failures=int(np.sum(bf[interior]<-1e-8)),
            global_butterfly_failures=int(np.sum(bf<-1e-8)),
            global_max_price_error_bp=delta*1e4)
    pd.DataFrame(records).to_csv(out/'consistency_predictions.csv',index=False)
    result=dict(checkpoint=KEY,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        scope='First-expiry forward re-evaluation at the same retained LCHS-fitted coefficients; no recalibration or quantum hardware execution',
        selection='Clean outside-model target, N=128, optimiser start 0. This stress case attains the archived maximum quoted-band butterfly deficit; it is not an average case.',
        N=128,T=T,coefficients=xi.tolist(),J=[2048,4096],cutoff=48.,beta=.8,fd_steps=1024,shift=g['shift'],
        exact_archive_max_difference=exact_diff,lchs_archive_max_difference=lchs_diff,
        max_imaginary_part=imaginary,forward_seconds=timings,metrics=checks)
    save_json(out/'consistency_validation.json',result)
    return result

if __name__=='__main__':
    import sys
    print(json.dumps(run(sys.argv[1],sys.argv[2]),indent=2))
