# fig6_boot_2oct.py : Figure 6 with per bin block bootstrap errors and a 20 block inclusion rule
import matplotlib; matplotlib.use('Agg')
import numpy as np, h5py, json
import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib import colors
H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'; OUTD='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_2oct/'
J=json.load(open(OUTD+'ul_audit_C_M_L.json'))
TARGET=75.4; TOL=0.01; Q=0.95; NBLK=20
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*Tsun**(5/3)*np.pi**(2/3)*c/mpc)
F_EM=np.log10(3.7e-9)
STAR={'qcw':(9.793,0.081),'loki':(9.771,0.006)}   # Table 5, Runs D and M
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
ldl=CONST+(5/3)*C['0_log10_mc']+(2/3)*C['0_log10_fgw']-C['0_log10_h']; m=np.abs(10**ldl-TARGET)<TOL*TARGET
qf,qm=C['0_log10_fgw'][m],C['0_log10_mc'][m]; del C,ldl,m
L=cols('G2D2_broad_UL_loki_100M_lastTOA_4core.h5',['0_log10_dist','0_log10_fgw','0_log10_mc'])
lf,lm=L['0_log10_fgw'],L['0_log10_mc']; del L
def pts(f_log,m_log,bins):
    f=10**f_log; x,y,e,drop=[],[],[],[]
    for b in bins:
        sel=(f>=b['f_lo']*1e-9)&(f<b['f_hi']*1e-9)
        if b['blocks']<NBLK or sel.sum()<2: drop.append(b['f_lo']); continue
        x.append(np.log10(0.5*(b['f_lo']+b['f_hi'])*1e-9)); y.append(np.log10(np.quantile(10**m_log[sel],Q))); e.append(b['err_boot'])
    return np.array(x),np.array(y),np.array(e),drop
q=pts(qf,qm,J['C']['bins']); l=pts(lf,lm,J['L']['bins'])
json.dump({'qcw':[list(map(float,a)) for a in q[:3]],'qcw_dropped':q[3],'loki':[list(map(float,a)) for a in l[:3]],'loki_dropped':l[3]},open(OUTD+'fig6_boot_points.json','w'),indent=1)
plt.rcParams.update({'font.size':16})
fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
stroke=[pe.withStroke(linewidth=4,foreground='white')]
for ax,sf,sm,(x,y,ye,_),cmap,ulc,kind in [(axes[0],qf,qm,q,'Blues','#08306b','qcw'),(axes[1],lf,lm,l,'RdPu','#7a1750','loki')]:
    im=ax.hist2d(sf,sm,bins=50,norm=colors.LogNorm(),cmap=cmap)[3]
    ax.step(x,y,where='mid',color=ulc,lw=2.6,zorder=3,path_effects=stroke)
    ax.fill_between(x,y-ye,y+ye,step='mid',color=ulc,alpha=0.30,lw=0,zorder=2)
    ax.errorbar(x,y,yerr=ye,fmt='o',color=ulc,ecolor=ulc,elinewidth=1.6,capsize=3,lw=0,zorder=4,path_effects=stroke)
    ax.errorbar([F_EM],[STAR[kind][0]],yerr=[STAR[kind][1]],fmt='*',ms=22,color=ulc,mec='white',mew=1.2,ecolor=ulc,elinewidth=1.6,capsize=4,zorder=5)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.15); cb.set_label('Number of Samples',fontsize=13)
    ax.tick_params(direction='in',top=True,right=True,which='both')
    ax.set_xlabel(r'$\log_{10}(f_{\rm GW}\,[{\rm Hz}])$',fontsize=20)
axes[0].set_ylabel(r'$\log_{10}(\mathcal{M}_c\,/\,M_\odot)$',fontsize=20)
plt.tight_layout()
fig.savefig(OUTD+'G2D1_broad_freq_segmented_UL_sidebyside_4core_boot.pdf',bbox_inches='tight'); fig.savefig(OUTD+'G2D1_broad_freq_segmented_UL_sidebyside_4core_boot.png',dpi=150,bbox_inches='tight')
print('done', len(q[0]), len(l[0]), q[3], l[3])
