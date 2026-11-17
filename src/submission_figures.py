"""Figures for the revised manuscript: architecture, surfaces and measured precision."""
from pathlib import Path
import json,io
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from legacy_figures import black_iv
from surface_model import save_json

BLUE='#25618C';GOLD='#A36B25';INK='#202124';GREY='#777777'
def style():
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
        'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,
        'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,out,name,sources):
    out=Path(out);(out/'figures').mkdir(exist_ok=True,parents=True)
    for ext in ['pdf','png']:
        stream=io.BytesIO();fig.savefig(stream,format=ext,dpi=240,bbox_inches='tight',pad_inches=.03)
        (out/'figures'/f'{name}.{ext}').write_bytes(stream.getvalue())
    save_json(out/'figures'/f'{name}.json',dict(sources=sources))
    plt.close(fig)

def architecture(out):
    """Original maturity-indexed diagram of this paper's actual hybrid objective."""
    from matplotlib.patches import Rectangle, Circle
    style();fig,ax=plt.subplots(figsize=(7.15,2.8))
    ax.set(xlim=(0,14.3),ylim=(0,5.6));ax.axis('off')
    edge='#30434b';gate='#e6f0f1';classical='#f1ece3'
    def box(x,y,w,h,label,fill='white',size=9):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,lw=.7,zorder=3))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=INK,zorder=4)
    def arr(points,ls='-',colour=edge):
        if len(points)>2:
            xx,yy=zip(*points[:-1]);ax.plot(xx,yy,color=colour,lw=.8,ls=ls)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='->',mutation_scale=8,
                                     linewidth=.8,color=colour,linestyle=ls))
    ax.text(3.25,5.32,r'Candidate local-variance coefficients $\xi=(\xi_1,\ldots,\xi_S)$',ha='center',fontsize=9)
    ax.plot([.75,5.85],[5.05,5.05],color=edge,lw=.6)
    ax.text(7.65,5.32,'Quote discrepancy',ha='center',fontsize=9)
    ax.text(10.25,5.32,'Circuit counts',ha='center',fontsize=9)
    xs=[1.65,2.85,4.0,5.1]
    for row,(lab,y,indices) in enumerate([('1',4.32,[0]),('2',3.02,[0,1]),('S',1.72,[0,1,2,3])]):
        ax.text(.62,y+.38,r'$T_'+lab+'$',ha='center',fontsize=9)
        ax.text(.62,y-.02,r'$v_0$',ha='center',fontsize=10)
        ax.plot([.87,6.2],[y,y],color=edge,lw=.8)
        for ind in indices:
            x=xs[ind]
            if ind==2:
                ax.text(x,y,r'$\cdots$',ha='center',va='center',fontsize=13,
                        bbox=dict(facecolor='white',edgecolor='none',pad=.5))
            else:
                j=['1','2','','S'][ind]
                box(x-.4,y-.29,.8,.58,r'$\widetilde P_'+j+'$',gate,11)
        box(6.2,y-.34,2.85,.68,r'$W_'+lab+r'\widetilde v_'+lab+r'-y_'+lab+'$',size=10)
        arr([(9.08,y),(9.7,y)])
        ax.add_patch(Circle((10.05,y),.19,edgecolor=edge,facecolor='white',lw=.7))
        # A measurement dial, kept distinct from the unitary pricing blocks.
        theta=np.linspace(.2,np.pi-.2,35)
        ax.plot(10.05+.125*np.cos(theta),y-.04+.125*np.sin(theta),color=edge,lw=.6)
        ax.plot([10.05,10.135],[y-.075,y+.07],color=edge,lw=.7)
        ax.text(10.67,y,r'$h_'+lab+r'/B_'+lab+'$',ha='center',va='center',fontsize=10)
        ax.plot([11.18,11.55],[y,y],color=edge,lw=.75)
    ax.plot([11.55,11.55],[1.72,4.32],color=edge,lw=.75)
    arr([(11.55,3.02),(12.02,3.02)])
    box(12.05,2.40,1.95,1.24,'Scale and\naverage\n'+r'$\widehat F(\xi)$',classical,9)
    ax.text(3.35,.98,'Each maturity restarts from the payoff',ha='center',fontsize=8)
    box(7.65,.20,3.4,.65,r'Classical optimiser: update $\xi$',classical,9)
    arr([(13.03,2.36),(13.03,.525),(11.10,.525)])
    arr([(7.60,.525),(.12,.525),(.12,5.32),(.65,5.32)])
    ax.text(3.5,.32,'next candidate',ha='center',va='top',fontsize=8,color=edge)
    fig.subplots_adjust(left=.006,right=.994,top=.99,bottom=.005)
    save(fig,out,'calibration_architecture',['surface_model.py: residual circuit construction',
         'quantum_experiments.py: count-driven optimisation','main.tex: propagation and residual equations'])


def surfaces(examples,out):
    """Column-width financial figure: each row combines a surface and a slice."""
    style();examples=Path(examples);out=Path(out)
    d=pd.read_csv(examples/'surface_predictions.csv')
    quotes={c:json.loads((examples/f'{c}_quotes.json').read_text()) for c in ['smile','skew']}
    d['iv_pct']=100*black_iv(d.price,d.k,d['T'])
    d['target_iv_pct']=100*black_iv(d.target_price,d.k,d['T'])
    assert np.isfinite(d.iv_pct).all(), 'Invalid IV in the displayed strike/maturity window'
    d.to_csv(examples/'surface_plot_data.csv',index=False)
    fig=plt.figure(figsize=(3.5,3.05))
    for j,case in enumerate(['smile','skew']):
        y=.58 if j==0 else .11
        ax=fig.add_axes([-.035,y, .67,.385],projection='3d')
        z=d[(d.case==case)&(d.method=='lchs')].pivot(index='T',columns='k',values='iv_pct')
        K,T=np.meshgrid(z.columns,z.index)
        ax.plot_surface(K,T,z.to_numpy(),color=BLUE,alpha=.66,edgecolor='none',
                        antialiased=True,rstride=1,cstride=4)
        q=quotes[case];qk,qt=np.meshgrid(q['k'],q['times']);qiv=100*black_iv(q['prices'],qk,qt)
        ax.scatter(qk.ravel(),qt.ravel(),qiv.ravel(),c=INK,s=3,depthshade=False)
        ax.set(xlabel=r'$k$',ylabel=r'$T$',zlabel='IV (%)')
        ax.set_xticks([.7,1,1.4]);ax.set_yticks([.25,1]);ax.set_zticks([20,22] if j==0 else [20,24])
        ax.view_init(elev=23,azim=-55);ax.set_proj_type('ortho')
        ax.set_box_aspect((1.55,1,.9),zoom=1.12)
        ax.tick_params(pad=-1,labelsize=6.5)
        ax.xaxis.labelpad=-8;ax.yaxis.labelpad=-9;ax.zaxis.labelpad=-5
        ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.zaxis.label.set_size(7)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.fill=False;axis._axinfo['grid']['linewidth']=.35
        fig.text(.045,y+.39,f'({chr(97+j)}) '+case.title(),fontsize=9)
        ax=fig.add_axes([.62025,y+.075,.36975,.27])
        a=d[(d.case==case)&(d['T']==1.)];t=a[a.method=='exact'];b=a[a.method=='lchs']
        ax.plot(t.k,t.target_iv_pct,color=INK,lw=1.1,label='Target')
        ax.plot(t.k,t.iv_pct,color=GREY,lw=1.1,ls='--',label='Exact')
        ax.plot(b.k,b.iv_pct,color=BLUE,lw=1.05,ls=':',label='LCHS')
        qiv=100*black_iv(np.array(q['prices'])[-1],np.array(q['k']),1.)
        ax.scatter(q['k'],qiv,facecolor='white',edgecolor=INK,s=7,lw=.5,label='Quotes',zorder=5)
        ax.set(title=r'$T=1$',xlabel=r'$k$',xlim=(.7,1.4))
        ax.set_xticks([.7,1,1.4]);ax.set_yticks([21.6,22.2] if j==0 else [21,24])
        ax.tick_params(labelsize=6.5,pad=1,length=2)
        ax.xaxis.labelpad=0;ax.xaxis.label.set_size(8);ax.title.set_size(8)
        ax.grid(alpha=.16,lw=.5)
    h,l=ax.get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.005),
               fontsize=7,handlelength=1.5,columnspacing=.75,handletextpad=.35)
    save(fig,out,'financial_surfaces',['surface_examples/surface_predictions.csv','surface_examples/*_quotes.json'])

def precision(run,out):
    style();run=Path(run);d=pd.read_csv(run/'calibration_results.csv');d=d[d.method=='quantum_counts'].copy()
    fig,ax=plt.subplots(figsize=(3.45,1.9))
    B=sorted(d.shots.unique());labels=[]
    for i,(_,r) in enumerate(d.sort_values(['shots','seed']).iterrows()):
        est=1e4*np.sqrt(r.independent_final_measured_loss)
        low=1e4*np.sqrt(r.independent_final_mse_lower);high=1e4*np.sqrt(r.independent_final_mse_upper)
        colour=BLUE if r.shots==B[0] else GOLD
        ax.errorbar(est,i,xerr=np.array([[est-low],[high-est]]),fmt='o',color=colour,ms=3,capsize=2,lw=.8)
        ax.scatter(r.quote_rmse_bp,i,marker='x',color=INK,s=19,lw=1,zorder=4)
        labels.append(f'{r.shots/1e6:.2f}M / {int(r.seed)}')
    ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.invert_yaxis()
    ax.set(xlabel='Final quote RMSE (price bp)',ylabel='Shots / seed',xlim=(0,None))
    ax.grid(axis='x',alpha=.18,lw=.5)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=INK,marker='x',ls='',label='Matrix value'),Line2D([],[],color=GREY,marker='o',ls='-',label='Measurement + 95% CI')]
    ax.legend(handles=handles,loc='lower right',fontsize=7,frameon=False)
    fig.tight_layout(pad=.3)
    save(fig,out,'measurement_precision',['new_experiment/calibration_results.csv'])

def generate(examples,run,out):
    architecture(out);surfaces(examples,out);precision(run,out);numerical_tables(run,out)
    a=pd.read_csv(Path(examples)/'surface_accuracy.csv')
    vals={}
    for case in ['smile','skew']:
        for m,tag in [('exact','Exact'),('lchs','Lchs')]:
            vals[case.title()+tag]=f'{a[(a.case==case)&(a.method==m)].iloc[0].heldout_rmse_bp:.2f}'
    out=Path(out);(out/'tables').mkdir(exist_ok=True)
    (out/'tables/surface_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in vals.items())+'\n')
    return vals


def numerical_tables(run,out):
    run=Path(run);out=Path(out);(out/'tables').mkdir(exist_ok=True,parents=True)
    cfg=json.loads((run/'configuration.json').read_text())
    r=pd.read_csv(run/'calibration_results.csv');q=r[r.method=='quantum_counts']
    lo,hi=cfg['shots'][0],cfg['shots'][-1]
    a=q[q.shots==lo];b=q[q.shots==hi];m=r[r.method=='lchs'].iloc[0]
    checks=pd.read_csv(run/'circuit_validation.csv')
    mant,exp=f'{checks.absolute_error.max():.2e}'.split('e')
    vals={'LowShots':f'{lo:,}','HighShots':f'{hi:,}',
        'LowHeld':f'{a.heldout_rmse_bp.mean():.2f}','HighHeld':f'{b.heldout_rmse_bp.mean():.2f}',
        'MatrixHeld':f'{m.heldout_rmse_bp:.2f}',
        'LowCoeff':f'{a.coefficient_rmse.mean():.3f}','HighCoeff':f'{b.coefficient_rmse.mean():.3f}',
        'CircuitError':'$'+mant+r'\times10^{'+str(int(exp))+'}$',
        'RunProfile':cfg['profile'],'MeasuredRuns':str(len(q))}
    (out/'tables/new_run_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in vals.items())+'\n')
    rows=[]
    for method,label in [('exact','Exact'),('fd','Finite differences'),('lchs','Matrix LCHS')]:
        z=r[r.method==method].iloc[0]
        rows.append(f'{label} & n/a & {z.heldout_rmse_bp:.2f} & {z.coefficient_rmse:.3f}'+r'\\')
    for shots in dict.fromkeys([lo,hi]):
        z=q[q.shots==shots]
        rows.append(f'Measured LCHS & $2^{{{int(np.log2(shots))}}}$ & {z.heldout_rmse_bp.mean():.2f} & {z.coefficient_rmse.mean():.3f}'+r'\\')
    (out/'tables/measured_rows.tex').write_text('\n'.join(rows)+r'\bottomrule'+'\n')
    d=pd.read_csv(run/'resource_sweep.csv');rows=[]
    for (N,J),z in d.groupby(['N','J']):
        one=z[z.expiries==1].iloc[0];two=z[z.expiries==2].iloc[0]
        rows.append(f'{N} & {J} & {int(one.qubits)} & {int(one.cx):,} & {int(two.qubits)} & {int(two.cx):,}'+r'\\')
    (out/'tables/resource_rows.tex').write_text('\n'.join(rows)+r'\bottomrule'+'\n')
    return vals


def consistency(example,out):
    """Accuracy and butterfly comparison at one fixed coefficient vector."""
    style();example=Path(example);out=Path(out)
    d=pd.read_csv(example/'consistency_predictions.csv')
    meta=json.loads((example/'consistency_validation.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(3.45,2.50),sharex=True)
    specs=[('exact','Exact',INK,'--',1.25),
           ('fd','FD (1,024 steps)',GREY,':',1.05),
           ('lchs_2048',r'LCHS $J=2{,}048$',GOLD,'-',1.05),
           ('lchs_4096',r'LCHS $J=4{,}096$',BLUE,'-',1.15)]
    for method,label,col,ls,lw in specs:
        a=d[(d.method==method)&d.in_quote_band]
        axes[0].plot(a.k,a.error_bp,label=label,color=col,ls=ls,lw=lw,
                     marker='o' if method=='fd' else None,ms=2,markevery=7,mfc='white')
        b=d[(d.method==method)&d.complete_butterfly_in_band]
        axes[1].plot(b.k,b.butterfly_bp,label=label,color=col,ls=ls,lw=lw,
                     marker='o' if method=='fd' else None,ms=2,markevery=7,mfc='white')
    for ax in axes:
        ax.set(xlim=(.7,1.4));ax.set_xticks([.7,.9,1.1,1.3,1.4])
        ax.tick_params(labelsize=7,pad=1,length=2)
        ax.grid(alpha=.16,lw=.5);ax.axhline(0,color=GREY,lw=.5,zorder=0)
        ax.xaxis.labelpad=0;ax.yaxis.labelpad=2
    axes[0].set(ylabel='Error (bp)',ylim=(-4,4.7))
    axes[0].text(.015,.95,'(a) Price error',ha='left',va='top',transform=axes[0].transAxes,fontsize=8)
    axes[0].set_yticks([-4,0,4])
    axes[0].text(.98,.98,r'$N=128$, $T=182/365$',ha='right',va='top',transform=axes[0].transAxes,fontsize=8)
    axes[1].set(ylabel='Butterfly (bp)',xlabel=r'$k$',ylim=(-4,10))
    axes[1].text(.015,.95,'(b) Butterfly price',ha='left',va='top',transform=axes[1].transAxes,fontsize=8)
    axes[1].set_yticks([-4,0,4,8]);axes[1].axhspan(-4,0,facecolor='#f4e8e3',edgecolor='none',zorder=-1)
    axes[1].text(1.38,-3.35,'Negative',ha='right',color='#8f422c',fontsize=8)
    h,l=axes[1].get_legend_handles_labels()
    fig.legend(h,l,ncol=2,loc='lower center',bbox_to_anchor=(.55,-.01),frameon=False,
               fontsize=7,handlelength=2.0,columnspacing=1.2,labelspacing=.3)
    fig.subplots_adjust(left=.135,right=.988,top=.985,bottom=.275,hspace=.15)
    save(fig,out,'financial_consistency',['consistency_example/consistency_predictions.csv',
         'consistency_example/consistency_validation.json','paper/evidence/checkpoints/'+meta['checkpoint']+'.json'])
    # Values inserted into the manuscript derive from the plotted arrays.
    a=meta['metrics'];vals={'CoarseMax':f"{a['lchs_2048']['max_inband_price_error_bp']:.2f}",
        'RefinedMax':f"{a['lchs_4096']['max_inband_price_error_bp']:.3f}",
        'FDMax':f"{a['fd']['max_inband_price_error_bp']:.3f}",
        'CoarseButterfly':f"{a['lchs_2048']['min_inband_butterfly_bp']:.2f}"}
    (out/'tables').mkdir(exist_ok=True,parents=True)
    (out/'tables/consistency_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in vals.items())+'\n')
    return meta
