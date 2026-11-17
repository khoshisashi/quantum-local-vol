"""Source-backed paper figures and TeX values for the new circuit experiments."""
from pathlib import Path
import ast, json, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from surface_model import (grid,data_for,prices,observation,implied_vol,TRUTH,
                           financial_metrics,save_json)

BLUE='#286F9F'; ORANGE='#CB702C'; INK='#252A2D'; GREY='#737C82'

def finish(fig,out,name,sources):
    out=Path(out);(out/'figures').mkdir(parents=True,exist_ok=True)
    for fmt in ['pdf','png']:
        buf=io.BytesIO();fig.savefig(buf,format=fmt,dpi=210,bbox_inches='tight',facecolor='white')
        payload=buf.getvalue()
        if fmt=='pdf':assert payload.rstrip().endswith(b'%%EOF')
        path=out/'figures'/f'{name}.{fmt}';tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_bytes(payload);tmp.replace(path)
    save_json(out/'figures'/f'{name}.json',dict(sources=sources))
    plt.close(fig)

def generate(run,out):
    run=Path(run);out=Path(out);(out/'tables').mkdir(parents=True,exist_ok=True)
    cfg=json.loads((run/'configuration.json').read_text())
    r=pd.read_csv(run/'calibration_results.csv');quant=r[r.method=='quantum_counts']
    low,high=cfg['shots'][0],cfg['shots'][-1]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'legend.fontsize':7,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42})
    problem=data_for(grid());g=problem['g'];k=np.linspace(.7,1.4,121);W=observation(g,k)
    target=prices(g,TRUTH);mat=r[r.method=='lchs'].iloc[0]
    chosen=[('Same-grid target',TRUTH,'exact',INK,'-'),
            ('Matrix LCHS',np.array(ast.literal_eval(mat.parameters)).reshape(2,2),'lchs',ORANGE,'--')]
    for B,style,name in [(low,':','Measured: lower shots'),(high,'-.','Measured: higher shots')]:
        subset=quant[(quant.shots==B)&(quant.seed==cfg['seeds'][0])]
        if len(subset):chosen.append((name,np.array(ast.literal_eval(subset.iloc[0].parameters)).reshape(2,2),'lchs',BLUE,style))
    fig,axs=plt.subplots(1,3,figsize=(7.15,2.45),gridspec_kw={'width_ratios':[1,1,1.1]})
    records=[]
    for label,xi,method,col,ls in chosen:
        p=prices(g,xi,method=method,terms=cfg['terms'],cutoff=cfg['cutoff'])
        for s,T in enumerate(problem['times']):
            pred=(W@p[s]).real;iv=np.array([implied_vol(a,T,b) for a,b in zip(k,pred)])*100
            axs[s].plot(k,iv,color=col,ls=ls,lw=1.35,label=label)
            records.extend(dict(method=label,T=T,k=a,price=b,iv_pct=c,selected_seed=cfg['seeds'][0])
                           for a,b,c in zip(k,pred,iv))
    for s,T in enumerate(problem['times']):
        q=problem['quotes'][s];ks=__import__('surface_model').QUOTES[s]
        iv=[100*implied_vol(a,T,b) for a,b in zip(ks,q)]
        axs[s].scatter(ks,iv,s=13,facecolor='white',edgecolor=INK,zorder=5)
        axs[s].set(title=f'{T:g}-year implied volatility',xlabel='Strike / forward',ylabel='Volatility (%)')
        axs[s].grid(alpha=.16)
    ymin=min(a.get_ylim()[0] for a in axs[:2]);ymax=max(a.get_ylim()[1] for a in axs[:2])
    for a in axs[:2]:a.set_ylim(ymin,ymax)
    categories=[('Exact',r[r.method=='exact']),('FD',r[r.method=='fd']),
                ('Matrix\nLCHS',r[r.method=='lchs'])]
    for B in dict.fromkeys([low,high]):categories.append((f'{B/1e6:.2f}M\nshots',quant[quant.shots==B]))
    for i,(name,data) in enumerate(categories):
        if len(data)==0:continue
        vals=data.heldout_rmse_bp.to_numpy();col=BLUE if i>=3 else (ORANGE if i==2 else GREY)
        axs[2].bar(i,vals.mean(),color='white',edgecolor=col,width=.62,lw=1)
        axs[2].scatter(i+np.linspace(-.13,.13,len(vals)),vals,s=18,color=col,zorder=4)
    axs[2].set_xticks(range(len(categories)),[n for n,_ in categories],fontsize=6.5)
    axs[2].set(ylabel='Held-out price RMSE (bp)',title='All sampler seeds',ylim=(0,None))
    axs[2].grid(axis='y',alpha=.16)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.03))
    fig.tight_layout(rect=(0,.18,1,1),w_pad=1.)
    finish(fig,out,'measured_calibration',['new_experiment/calibration_results.csv','new_experiment/configuration.json'])
    pd.DataFrame(records).to_csv(out/'tables'/'measured_smile_data.csv',index=False)
    checks=pd.read_csv(run/'circuit_validation.csv')
    def mean(B):
        vals=quant[quant.shots==B].heldout_rmse_bp
        return f'{vals.mean():.2f}' if len(vals) else 'not yet run'
    macros={'LowShots':f'{low:,}','HighShots':f'{high:,}',
        'LowHeld':mean(low),'HighHeld':mean(high),'MatrixHeld':f'{mat.heldout_rmse_bp:.2f}',
        'CircuitError':r'$'+f'{checks.absolute_error.max():.2e}'.replace('e-',r'\times10^{-')+'}$',
        'RunProfile':cfg['profile'],'MeasuredRuns':str(len(quant))}
    (out/'tables'/'new_run_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    lines=[r'\begin{table}[h]\centering',r'\caption{New two-expiry circuit experiment. Held-out errors are in bp; ranges cover '+str(len(cfg['seeds']))+r' sampler seeds. Shots/evaluation sum both maturities.}\label{tab:measured}',
        r'\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrr}\toprule',
        r'Objective & Mean RMSE & Range & Shots/evaluation\\\midrule']
    for name,m in [('Exact','exact'),('Finite differences','fd'),('Matrix LCHS','lchs')]:
        a=r[r.method==m].iloc[0];lines.append(f'{name} & {a.heldout_rmse_bp:.2f} & --- & 0'+r'\\')
    for B in dict.fromkeys([low,high]):
        d=quant[quant.shots==B]
        if len(d):lines.append(f'Measured LCHS & {d.heldout_rmse_bp.mean():.2f} & {d.heldout_rmse_bp.min():.2f}--{d.heldout_rmse_bp.max():.2f} & {2*B/1e6:.2f}M'+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (out/'tables'/'measured_results.tex').write_text('\n'.join(lines)+'\n')
    if (run/'resource_sweep.csv').exists():
        d=pd.read_csv(run/'resource_sweep.csv');keys=sorted(set(zip(d.N,d.J)))
        fig,axs=plt.subplots(1,2,figsize=(3.48,2.2),sharey=True)
        for S,col,mark in [(1,ORANGE,'o'),(2,BLUE,'s')]:
            a=d[d.expiries==S];y=np.array([keys.index((n,j)) for n,j in zip(a.N,a.J)])+(S-1.5)*.17
            axs[0].scatter(a.qubits,y,c=col,marker=mark,s=17,label=f'{S} interval'+('s' if S>1 else ''))
            axs[1].scatter(a.cx,y,c=col,marker=mark,s=17)
        axs[0].set_yticks(range(len(keys)),[f'N={n}, J={j}' for n,j in keys],fontsize=6.5)
        axs[0].invert_yaxis();axs[0].set(xlabel='Qubits',xlim=(0,None));axs[1].set(xlabel='CNOT gates',xscale='log')
        for a in axs:a.grid(axis='x',alpha=.18)
        fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.54,-.02))
        fig.tight_layout(rect=(0,.10,1,1),w_pad=.8)
        finish(fig,out,'circuit_resources',['new_experiment/resource_sweep.csv'])
    save_json(out/'tables'/'new_result_summary.json',dict(profile=cfg['profile'],completed_runs=len(quant),
        planned_runs=len(cfg['shots'])*len(cfg['seeds']),max_circuit_probability_error=float(checks.absolute_error.max()),
        measured_low_mean_bp=mean(low),measured_high_mean_bp=mean(high),
        target='same-grid synthetic exact PDE',optimizer_uses='measurement counts only'))
    return r

def quadrature_diagnostic(out):
    out=Path(out);rows=[]
    for n in [2,3,4]:
        p=data_for(grid(n))
        for J in [8,16,32,64,128]:
            for cutoff in [8.,12.,24.,32.,48.]:
                m=financial_metrics(p,TRUTH,'lchs',J,cutoff)
                rows.append(dict(N=2**n,J=J,cutoff=cutoff,**m))
    d=pd.DataFrame(rows);d.to_csv(out/'quadrature_diagnostic.csv',index=False)
    return d
