# fig6_final_19oct.py : fig:mc with all-block bootstrap errors on both panels, bins spanning < NMIN blocks omitted
import matplotlib; matplotlib.use('Agg')
import numpy as np, json, sys
import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib import colors
D='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_19oct/'
J=json.load(open(D+'allblock_boot_19oct.json')); NMIN=int(sys.argv[1]) if len(sys.argv)>1 else 10
F_EM=np.log10(3.7e-9); STAR={'C':9.793,'L':9.771}
plt.rcParams.update({'font.size':16,'pdf.compression':9})
fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
stroke=[pe.withStroke(linewidth=4,foreground='white')]
for ax,tag,cmap,ulc in [(axes[0],'C','Blues','#08306b'),(axes[1],'L','RdPu','#7a1750')]:
    S=np.load(D+'samples_%s.npz'%tag)
    im=ax.hist2d(S['lf'],S['lm'],bins=50,norm=colors.LogNorm(),cmap=cmap,rasterized=True)[3]
    R=[r for r in J[tag]['rows'] if r['blocks']>=NMIN]
    x=np.array([np.log10(r['f_c']*1e-9) for r in R]); y=np.array([r['UL'] for r in R]); s=np.array([r['sig'] for r in R])
    ax.step(x,y,where='mid',color=ulc,lw=2.0,zorder=3,path_effects=stroke)
    ax.errorbar(x,y,yerr=s,fmt='o',color=ulc,mfc=ulc,mec='white',mew=0.8,ms=7,ecolor=ulc,elinewidth=1.4,capsize=3,zorder=4)
    ax.plot(F_EM,STAR[tag],marker='*',ms=22,color=ulc,mec='white',mew=1.2,zorder=6)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.15); cb.set_label('Number of Samples',fontsize=13)
    ax.tick_params(direction='in',top=True,right=True,which='both')
    ax.set_xlabel(r'$\log_{10} f_{\rm GW}$',fontsize=20)
    print(tag,'plotted',len(R),'first bin UL',R[0]['UL'],'star offset',round(R[0]['UL']-STAR[tag],4))
axes[0].set_ylim(9.0,10.0); axes[0].set_ylabel(r'$\log_{10} \mathcal{M}_c$',fontsize=20)
plt.tight_layout()
fn='G2D1_broad_freq_segmented_UL_sidebyside_4core_bootstrap.pdf'
fig.savefig(D+fn,bbox_inches='tight',dpi=110); fig.savefig(D+fn.replace('.pdf','.png'),dpi=110,bbox_inches='tight'); print('saved',fn)
