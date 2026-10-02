# fig6_optionB_2oct.py : Figure 6 option B, per bin limits without error bars, raw 50 sample cut
import matplotlib; matplotlib.use('Agg')
import numpy as np, h5py
import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib import colors
H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'; OUTD='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_2oct/'
TARGET=75.4; TOL=0.01; Q=0.95; NMIN=50
DF=1/(14.962*365.25*86400.0)
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*Tsun**(5/3)*np.pi**(2/3)*c/mpc)
F_EM=np.log10(3.7e-9); STAR={'qcw':9.793,'loki':9.771}
def cols(fn,names):
    with h5py.File(H5+fn,'r') as f:
        pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
        ds=f['samples_cold']; N=ds.shape[1]; idx=[pn.index(n) for n in names]; lo,hi=min(idx),max(idx)+1
        out={n:np.empty(N) for n in names}
        for a in range(0,N,5_000_000):
            b=min(a+5_000_000,N); blk=ds[0,a:b,lo:hi]
            for n,i in zip(names,idx): out[n][a:b]=blk[:,i-lo]
    return out
C=cols('G2D1_broad_UL_4core.h5',['0_log10_fgw','0_log10_h','0_log10_mc'])
ldl=CONST+(5/3)*C['0_log10_mc']+(2/3)*C['0_log10_fgw']-C['0_log10_h']; m=np.abs(10**ldl-TARGET)<=TOL*TARGET
qf,qm=C['0_log10_fgw'][m],C['0_log10_mc'][m]; del C,ldl,m
L=cols('G2D2_broad_UL_loki_100M_lastTOA_4core.h5',['0_log10_dist','0_log10_fgw','0_log10_mc'])
m=np.abs(10**L['0_log10_dist']-TARGET)<=TOL*TARGET
lf,lm=L['0_log10_fgw'][m],L['0_log10_mc'][m]; del L,m
def seg(lfv,lmv):
    f=10**lfv; edges=np.arange(f.min(),f.max()+DF,DF); k=np.digitize(f,edges)-1
    x,y=[],[]
    for b in range(len(edges)-1):
        s=lmv[k==b]
        if len(s)<NMIN: continue
        x.append(np.log10(0.5*(edges[b]+edges[b+1]))); y.append(np.log10(np.quantile(10**s,Q)))
    return np.array(x),np.array(y)
q=seg(qf,qm); l=seg(lf,lm)
plt.rcParams.update({'font.size':16})
fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
stroke=[pe.withStroke(linewidth=4,foreground='white')]
for ax,sf,sm,(x,y),cmap,ulc,kind in [(axes[0],qf,qm,q,'Blues','#08306b','qcw'),(axes[1],lf,lm,l,'RdPu','#7a1750','loki')]:
    im=ax.hist2d(sf,sm,bins=50,norm=colors.LogNorm(),cmap=cmap)[3]
    ax.step(x,y,where='mid',color=ulc,lw=2.6,zorder=3,path_effects=stroke)
    ax.plot(x,y,'o',color=ulc,mec='white',mew=1.2,ms=7,zorder=4)
    ax.plot(F_EM,STAR[kind],marker='*',ms=22,color=ulc,mec='white',mew=1.2,zorder=5)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.15); cb.set_label('Number of Samples',fontsize=13)
    ax.tick_params(direction='in',top=True,right=True,which='both')
    ax.set_xlabel(r'$\log_{10} f_{\rm GW}$',fontsize=20)
axes[0].set_ylabel(r'$\log_{10} \mathcal{M}_c$',fontsize=20)
plt.tight_layout()
fig.savefig(OUTD+'G2D1_broad_freq_segmented_UL_sidebyside_4core_optionB.pdf',bbox_inches='tight')
fig.savefig(OUTD+'G2D1_broad_freq_segmented_UL_sidebyside_4core_optionB.png',dpi=110,bbox_inches='tight')
print('done', len(q[0]), len(l[0]), np.round(q[1][:2],3), np.round(l[1][:2],3))
