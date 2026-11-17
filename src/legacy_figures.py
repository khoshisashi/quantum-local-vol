"""Finance-facing figures from saved predictions, with explicit display interpolation.
No supplied Python module is executed and no calibration is repeated.
"""
from pathlib import Path
import argparse, json, io
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

BLUE='#286F9F'; ORANGE='#CB702C'; INK='#252A2D'; GREY='#737C82'
STYLE={'target':(INK,'-',1.7),'expm':(GREY,'--',1.2),'fd':(GREY,':',1.2),
       'lchs-beta':(BLUE,'-',1.35),'lchs-simple':('#78A9C8','--',1.2),
       'AH':(ORANGE,'--',1.2),'PINN-fixed':('#D7A176',':',1.3),'PINN-Wam':('#8F491C','-.',1.3)}
LABEL={'target':'Fine-grid target','expm':'PDE: matrix exp.','fd':'PDE: finite diff.',
       'lchs-beta':'PDE: LCHS beta','lchs-simple':'PDE: LCHS simple','AH':'Andreasen-Huge',
       'PINN-fixed':'PINN: fixed','PINN-Wam':'PINN: adaptive'}
TIMES=np.array([182/365,1.]);XI_TRUTH=np.array([[.8,.3,.25,.7],[.6,.35,.3,.5]])
METHODS=['expm','fd','lchs-beta','lchs-simple','AH','PINN-fixed','PINN-Wam']
CASES=['in_bp0_d101','out_bp0_d101','out_bp5_d101']

def read_json(p):return json.loads(Path(p).read_text())
def call(k,T,vol):
    sd=vol*np.sqrt(T);d=-np.log(k)/sd+sd/2
    return ndtr(d)-k*ndtr(d-sd)

def black_iv(price,k,T):
    """Vectorised bracketed inversion; infeasible prices remain NaN, never clipped."""
    p,k,t=np.broadcast_arrays(np.asarray(price,float),np.asarray(k,float),np.asarray(T,float))
    valid=np.isfinite(p)&(t>0)&(p>np.maximum(1-k,0)+1e-12)&(p<1-1e-12)
    valid&=(call(k,t,np.full(p.shape,1e-6))<=p)&(call(k,t,np.full(p.shape,8.))>=p)
    low=np.full(p.shape,1e-6);high=np.full(p.shape,8.)
    for _ in range(64):
        mid=(low+high)/2;below=call(k,t,mid)<p
        low=np.where(below,mid,low);high=np.where(below,high,mid)
    result=np.where(valid,(low+high)/2,np.nan)
    if valid.any():assert np.max(np.abs(call(k[valid],t[valid],result[valid])-p[valid]))<2e-12
    return result

def payload(root,case,method,n=6,seed=0):
    return read_json(root/'checkpoints'/f'{case}_n{n}_{method}_s{seed}.json')['result']

def profile(root,case,method,s,n=6,seed=0):
    """Use native predictions for AH/PINN, preserving the original benchmark."""
    data=read_json(root/'raw'/f'{case}.json');q=data['quotes'][s];test=data['tests'][s]
    k=np.r_[q['k'],test['k']];target=np.r_[q['clean'],test['price']]
    train=np.r_[np.ones(len(q['k']),bool),np.zeros(len(test['k']),bool)]
    observed=np.r_[q['mid'],np.full(len(test['k']),np.nan)]
    if method=='target':prices=target.copy()
    else:
        res=payload(root,case,method,n,seed)
        if 'quote_predictions' in res:
            prices=np.r_[res['quote_predictions'][s],res['test_predictions'][s]]
        else:
            K=np.exp(np.linspace(-1.2,1.2,2**n+2));prices=np.interp(k,K,res['C'][s])
    ix=np.argsort(k);assert np.all(np.diff(k[ix])>0)
    return pd.DataFrame({'dataset':case,'method':method,'initialisation':seed,'N':2**n,
        'T':q['T'],'k':k[ix],'is_training':train[ix],'observed_price':observed[ix],
        'target_price':target[ix],'price':prices[ix],
        'iv_pct':100*black_iv(prices[ix],k[ix],q['T']),
        'target_iv_pct':100*black_iv(target[ix],k[ix],q['T']),
        'price_error_bp':1e4*(prices[ix]-target[ix])})

def basis(x):
    z=(np.asarray(x)[None,:]-np.array([-.6,-.2,.2,.6])[:,None])/.45
    return .04*(1+np.cos(np.pi*np.clip(z,-1,1)))*(np.abs(z)<=1)

def true_variance(k,s,truth):
    x=np.log(k)
    return (.01+XI_TRUTH[s]@basis(x)) if truth=='in' else (.01+.055*np.exp(-((x+.1)/.3)**2)+.035/(1+np.exp(4*x)))

def line(ax,x,y,method,label=None):
    color,ls,lw=STYLE[method];ax.plot(x,y,color=color,ls=ls,lw=lw,label=label or LABEL[method])

def finish(fig,out,name,source,note):
    for ax in fig.axes:
        if ax.name!='3d':ax.grid(alpha=.15,lw=.5)
    for ext in ['pdf','png']:
        stream=io.BytesIO()
        fig.savefig(stream,format=ext,dpi=240,bbox_inches='tight',pad_inches=.04)
        content=stream.getvalue()
        if ext=='pdf':assert content.rstrip().endswith(b'%%EOF'), 'Incomplete PDF stream'
        destination=out/'figures'/f'{name}.{ext}'
        temporary=destination.with_suffix(destination.suffix+'.tmp')
        temporary.write_bytes(content);temporary.replace(destination)
    (out/'figures'/f'{name}.json').write_text(json.dumps({'source':source,'note':note},indent=2))
    plt.close(fig)

def generate(root,out):
    root=Path(root);out=Path(out)
    for d in ['figures','tables','finance_data']: (out/d).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,
        'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    profiles=pd.concat([profile(root,c,m,s) for c in CASES for m in ['target',*METHODS] for s in range(2)],ignore_index=True)
    recorded=pd.read_csv(root/'results.csv');reconciled=0
    for c in CASES:
        for m in METHODS:
            d=profiles[(profiles.dataset==c)&(profiles.method==m)]
            row=recorded[(recorded.dataset==c)&(recorded.method==m)&(recorded.n==6)&(recorded.seed==0)].iloc[0]
            hold=d[~d.is_training];train=d[d.is_training]
            assert len(hold)==int(row.iv_total)
            assert abs(np.sqrt(np.mean(hold.price_error_bp**2))-row.rmse_heldout*1e4)<2e-9
            assert abs(np.sqrt(np.mean((train.price-train.observed_price)**2))-row.rmse_quotes)<2e-12
            reconciled+=1
    profiles.to_csv(out/'finance_data'/'saved_smiles.csv',index=False)
    def get(c,m,s):return profiles[(profiles.dataset==c)&(profiles.method==m)&np.isclose(profiles['T'],TIMES[s])]
    # Four surfaces: identical camera, scales and maturity interpolation rule.
    case=CASES[-1];k=np.linspace(.7,1.4,73);tt=np.linspace(TIMES[0],TIMES[1],25)
    K,TT=np.meshgrid(k,tt);surface_methods=['target','lchs-beta','AH','PINN-Wam']
    interpolated={};records=[];surfaces={}
    for m in surface_methods:
        p=np.array([np.interp(k,get(case,m,s).k,get(case,m,s).price) for s in range(2)])
        w=(tt-TIMES[0])/(TIMES[1]-TIMES[0]);C=(1-w[:,None])*p[0]+w[:,None]*p[1]
        V=100*black_iv(C,K,TT);assert np.isfinite(V).all();surfaces[m]=V;interpolated[m]=p
        records.append(pd.DataFrame({'dataset':case,'method':m,'N':64,'initialisation':0,
            'T_display':TT.ravel(),'k':K.ravel(),'price_display':C.ravel(),'iv_display_pct':V.ravel(),
            'display_rule':'Linear interpolation of saved prices between two expiries'}))
    pd.concat(records,ignore_index=True).to_csv(out/'finance_data'/'surface_display.csv',index=False)
    lo=np.floor(min(v.min() for v in surfaces.values()));hi=np.ceil(max(v.max() for v in surfaces.values()))
    fig=plt.figure(figsize=(7.16,4.0));gs=fig.add_gridspec(2,4,height_ratios=[1.3,1.],hspace=.30,wspace=.08,
        top=.90,bottom=.10,left=.11,right=.92)
    for i,m in enumerate(surface_methods):
        ax=fig.add_subplot(gs[0,i],projection='3d')
        ax.plot_surface(K,TT,surfaces[m],color=STYLE[m][0],alpha=.78,edgecolor='white',linewidth=.12,
            rstride=2,cstride=3,shade=False,antialiased=True)
        ax.view_init(elev=25,azim=-125);ax.set_box_aspect((1.1,.8,.85))
        ax.set(xlim=(.7,1.4),ylim=(TIMES[0],1),zlim=(lo,hi))
        ax.set_xticks([.7,1.,1.4]);ax.set_yticks([.5,1.]);ax.set_zticks([lo,(lo+hi)/2,hi])
        ax.set_xlabel('K / F',labelpad=-5);ax.set_ylabel('Years',labelpad=-6)
        ax.set_zlabel('IV (%)' if i==0 else '',labelpad=-1);ax.tick_params(pad=-2,labelsize=6)
        ax.set_title(f'({chr(97+i)}) '+LABEL[m],pad=1,fontweight='bold',fontsize=7)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False;axis._axinfo['grid']['color']=(.8,.8,.8,.35)
    ax=fig.add_subplot(gs[1,:]);errors=np.vstack([1e4*(interpolated[m]-interpolated['target']) for m in surface_methods[1:]])
    cmap=LinearSegmentedColormap.from_list('signed',[BLUE,'#FAFAFA',ORANGE]);lim=float(np.ceil(np.abs(errors).max()))
    mesh=ax.imshow(errors,origin='upper',extent=[.7,1.4,6,0],aspect='auto',cmap=cmap,vmin=-lim,vmax=lim,
        interpolation='nearest')
    ax.set_yticks(np.arange(6)+.5,[LABEL[m].replace('PDE: ','')+f' / {"6m" if s==0 else "1y"}' for m in surface_methods[1:] for s in range(2)])
    ax.set_xticks([.7,.85,1.,1.2,1.4]);ax.set_xlabel('Normalised strike K / F')
    ax.set_title('(e) Price errors at the two fitted expiries',loc='left',fontweight='bold',pad=4)
    for yy in [2,4]:ax.axhline(yy,color='white',lw=1.5)
    cb=fig.colorbar(mesh,ax=ax,pad=.015,fraction=.018);cb.set_label('Model minus target (bp)',fontsize=7);cb.ax.tick_params(labelsize=7)
    fig.suptitle('Out-of-family surface / noise +/-5 bp / N=64 / dataset 101, initialisation 0',fontsize=9,y=.995)
    finish(fig,out,'surface_comparison',['finance_data/saved_smiles.csv','finance_data/surface_display.csv'],
        '3D maturity values are linear price interpolation for display, not additional calibration or PDE solutions. The error heatmap uses the two fitted expiries only.')
    # Shared-scale smiles and signed price errors, across three named cases.
    fig,axs=plt.subplots(2,3,figsize=(7.16,3.35),sharex=True,sharey='row',layout='constrained',
        gridspec_kw={'height_ratios':[1.25,1]})
    titles=['In-family / clean','Out-of-family / clean','Out-of-family / noise +/-5 bp']
    allvol=[];allerr=[]
    for j,c in enumerate(CASES):
        for m in ['target','lchs-beta','AH','PINN-Wam']:
            d=get(c,m,1);line(axs[0,j],d.k,d.iv_pct,m);allvol.extend(d.iv_pct)
            if m!='target':line(axs[1,j],d.k,d.price_error_bp,m);allerr.extend(d.price_error_bp)
        q=read_json(root/'raw'/f'{c}.json')['quotes'][1]
        obs=100*black_iv(q['mid'],q['k'],q['T'])
        axs[0,j].scatter(q['k'],obs,s=21,facecolors='none',edgecolors=INK,lw=.85,zorder=5,label='Input quotes')
        axs[0,j].set_title(f'({chr(97+j)}) {titles[j]}',fontweight='bold',fontsize=8)
        axs[1,j].axhline(0,color=INK,lw=.75);axs[1,j].set_xlabel('Normalised strike K / F')
        axs[1,j].set_xlim(.7,1.4);axs[1,j].set_xticks([.7,1.,1.4])
    ylim=(np.floor(np.nanmin(allvol))-1,np.ceil(np.nanmax(allvol))+1)
    for ax in axs[0]:ax.set_ylim(*ylim)
    lim=np.ceil(np.max(np.abs(allerr)));axs[1,0].set_ylim(-lim,lim)
    axs[0,0].set_ylabel('1-year implied volatility (%)');axs[1,0].set_ylabel('Price error (bp)')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='outside lower center',frameon=False,fontsize=7)
    fig.suptitle('Smile shape and pricing error / N=64 / dataset 101, initialisation 0',fontsize=9)
    finish(fig,out,'smile_comparison',['finance_data/saved_smiles.csv'],
        'Fixed examples at maturity one year. Native AH/PINN predictions are retained. Input quotes are noisy observations in the third column. Lower panels share one signed bp scale.')
    # Local volatility is distinct from implied volatility. Only stored PDE coefficients are used.
    fig,axs=plt.subplots(2,2,figsize=(3.5,2.65),sharex=True,sharey=True,layout='constrained');records=[]
    k=np.linspace(.7,1.4,201)
    for j,truth in enumerate(['in','out']):
        c=f'{truth}_bp0_d101'
        for s in range(2):
            target=100*np.sqrt(true_variance(k,s,truth));line(axs[s,j],k,target,'target',label='True local vol.')
            for m in ['expm','lchs-beta']:
                hats=np.asarray(payload(root,c,m)['hats']);vol=100*np.sqrt(.01+hats[s]@basis(np.log(k)))
                line(axs[s,j],k,vol,m,label='Exact PDE' if m=='expm' else 'LCHS beta')
                if m=='expm':axs[s,j].plot(k[::30],vol[::30],ls='none',marker='o',ms=2.4,mfc='white',mec=GREY,mew=.6,zorder=6)
                records.append(pd.DataFrame({'dataset':c,'method':m,'slice':s+1,'k':k,'local_vol_pct':vol,'target_local_vol_pct':target}))
            axs[s,j].set_title(('In-family' if j==0 else 'Out-of-family')+(' / 0-6m' if s==0 else ' / 6-12m'),fontsize=7)
            axs[s,j].set_xlim(.7,1.4);axs[s,j].set_xticks([.7,1.,1.4]);axs[s,j].set_ylim(10,32)
            if j==0:axs[s,j].set_ylabel('Local vol. (%)')
            if s==1:axs[s,j].set_xlabel('K / F')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='outside lower center',frameon=False,fontsize=6.5)
    pd.concat(records).to_csv(out/'finance_data'/'local_volatility.csv',index=False)
    finish(fig,out,'local_volatility_recovery',['finance_data/local_volatility.csv'],
        'Clean data, N=64, initialisation0. Local volatility is sqrt(local variance); piecewise constant in time. AH/PINN coefficient predictions were not saved and are not imputed.')
    # Financial cost of a nonnegative-payoff butterfly, at identical fitted coefficients.
    key='out_bp0_d101_n7_lchs-beta_s0';r=read_json(root/'checkpoints'/f'{key}.json')['result']
    K=np.exp(np.linspace(-1.2,1.2,130));C=np.asarray(r['C']);E=np.asarray(r['C_exact'])
    dm=np.diff(K)[:-1];dp=np.diff(K)[1:];bf=lambda z:(dm*z[:,2:]+dp*z[:,:-2])/(dm+dp)-z[:,1:-1]
    s=0;raw=bf(C)[s]*1e4;exact=bf(E)[s]*1e4;band=(K[:-2]>=.7)&(K[2:]<=1.4)
    minimum=np.where(band,raw,np.inf).argmin();assert raw[minimum]<0
    fig,axs=plt.subplots(2,1,figsize=(3.5,2.75),sharex=True,layout='constrained')
    line(axs[0],K,E[s]*100,'expm',label='Exact, same coefficients');line(axs[0],K,C[s]*100,'lchs-beta',label='Raw LCHS beta')
    axs[0].set_ylabel('Call price (% of D F)');axs[0].set_title('Clean out-of-family / N=128 / 6 months / start 0',fontsize=7.5)
    axs[0].set_ylim(0,105*max(np.interp(.7,K,C[s]),np.interp(.7,K,E[s])))
    axs[0].legend(frameon=False,fontsize=6.5,loc='upper right')
    line(axs[1],K[1:-1],exact,'expm');line(axs[1],K[1:-1],raw,'lchs-beta')
    axs[1].axhline(0,color=INK,lw=.8);axs[1].fill_between(K[1:-1],raw,0,where=raw<0,color=BLUE,alpha=.2)
    axs[1].scatter(K[minimum+1],raw[minimum],marker='v',s=24,color=BLUE,zorder=5)
    axs[1].annotate(f'{raw[minimum]:.2f} bp',xy=(K[minimum+1],raw[minimum]),xytext=(1.13,-2.3),fontsize=7,
        arrowprops={'arrowstyle':'->','lw':.7,'color':INK})
    axs[1].set(xlim=(.7,1.4),xlabel='Normalised strike K / F',ylabel='Butterfly cost (bp)')
    axs[1].set_xticks([.7,.85,1.,1.2,1.4]);axs[1].set_ylim(min(-3.5,raw[band].min()-.3),max(5,raw[band].max()+.3))
    pd.DataFrame({'source_key':key,'T':TIMES[s],'k_left':K[:-2],'k_centre':K[1:-1],'k_right':K[2:],
        'raw_butterfly_bp':raw,'exact_butterfly_bp':exact,'inside_quote_band':band}).to_csv(out/'finance_data'/'butterfly_portfolios.csv',index=False)
    finish(fig,out,'arbitrage_finance',[f'checkpoints/{key}.json','finance_data/butterfly_portfolios.csv'],
        'Upper: prices at identical fitted coefficients. Lower: cost of long weighted wing calls and short centre call; negative cost violates the nonnegative butterfly payoff. Displayed points use the original strike grid.')
    # Numerical comparison with financially interpretable units and dataset replication.
    summ=pd.read_csv(out/'audit'/'summary_verified.csv');res=pd.read_csv(root/'results.csv')
    rows=[r'\begin{table*}[t]\centering\caption{Held-out call-price RMSE (normalised price bp). Noisy entries are means $\pm$ sample SD over three datasets after averaging five starts; AH is deterministic. Times are CPU calibration seconds per run in the out-of-family noisy case, excluding compilation and evaluation.}\label{tab:performance}',
          r'\small\setlength{\tabcolsep}{5pt}\begin{tabular}{lrrrrr}\toprule & \multicolumn{2}{c}{In-family target} & \multicolumn{2}{c}{Out-of-family target} & CPU time\\',
          r'Method & Clean & Noise $\pm5$ bp & Clean & Noise $\pm5$ bp & Seconds\\\midrule']
    ivrecords=[]
    for m in METHODS:
        values=[]
        for truth,noise in [('in',0),('in',5),('out',0),('out',5)]:
            r=summ[(summ.n==6)&(summ.method==m)&(summ.truth==truth)&(summ.noise_bp==noise)].iloc[0]
            v=f'{r.heldout*1e4:.2f}'
            if noise:v=f'${v}\\pm{r.heldout_sd*1e4:.2f}$'
            if m=='AH' and truth=='in' and noise:v+=r'$^{\dagger}$'
            values.append(v)
        r=summ[(summ.n==6)&(summ.method==m)&(summ.truth=='out')&(summ.noise_bp==5)].iloc[0]
        rows.append(LABEL[m].replace('Andreasen-Huge','Andreasen--Huge')+' & '+' & '.join(values)+f' & {r.opt_s:.3g}'+r'\\')
        rr=res[(res.n==6)&(res.method==m)&(res.truth=='out')&(res.noise_bp==5)]
        ivrecords.append({'method':m,'heldout_price_bp':r.heldout*1e4,'opt_seconds':r.opt_s,
                         'iv_valid':int(rr.iv_valid.sum()),'iv_total':int(rr.iv_total.sum())})
    rows.extend([r'\bottomrule\end{tabular}',r'\par\vspace{2pt}\footnotesize $^{\dagger}$AH uses two feasible noisy in-family datasets; one input quote is below intrinsic value. A paired two-dataset comparison is provided in the supplement.\end{table*}'])
    (out/'tables'/'financial_performance.tex').write_text('\n'.join(rows))
    pd.DataFrame(ivrecords).to_csv(out/'finance_data'/'price_time_iv_coverage.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.25),layout='constrained',sharey=True)
    for i,m in enumerate(METHODS[:4]):
        r=summ[(summ.n==6)&(summ.method==m)&(summ.truth=='out')&(summ.noise_bp==5)].iloc[0]
        axs[0].errorbar(r.heldout*1e4,i,xerr=r.heldout_sd*1e4,fmt='o',color=STYLE[m][0],capsize=2)
        axs[1].scatter(r.opt_s,i,color=STYLE[m][0],marker=['o','s','D','^'][i],s=25)
    axs[0].set_yticks(range(4),[LABEL[m] for m in METHODS[:4]]);axs[0].invert_yaxis();axs[0].set_xlim(left=0)
    axs[0].set_xlabel('Held-out price RMSE (bp)');axs[1].set_xscale('log');axs[1].set_xlabel('CPU calibration time (seconds)')
    axs[0].set_title('Same model: price accuracy');axs[1].set_title('Same model: classical computation')
    finish(fig,out,'solver_comparison',['audit/summary_verified.csv'],
        'N64, out-of-family, noise5bp, three independent datasets and five starts. Time excludes compilation and evaluation; no quantum runtime is measured.')
    checks={'native_prediction_source_preserved':True,'iv_inversion_repricing_tolerance':2e-12,
        'illustrated_run_metrics_reconciled':reconciled,
        'surface_interpolation':'linear prices between saved expiries; display only','new_maturities_calibrated':0,
        'surface_case':case,'smile_cases':CASES,'example_start':0,'band_butterfly_example_min_bp':float(raw[band].min()),
        'all_surface_display_iv_finite':True,'plots':['surface_comparison','smile_comparison','local_volatility_recovery','arbitrage_finance','solver_comparison']}
    (out/'finance_data'/'figure_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2));return checks

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();generate(a.input,a.out)
