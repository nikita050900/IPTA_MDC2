# fig6_ess_19oct.py : Figure 6 with Eq. 26 order statistic error bars on every bin, ESS < 50 bins as open markers
import matplotlib; matplotlib.use('Agg')
import numpy as np, json
import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib import colors
from matplotlib.lines import Line2D
D='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_19oct/'
P=json.load(open(D+'post_19oct.json'))
F_EM=np.log10(3.7e-9); STAR={'qcw':9.793,'loki':9.771}
plt.rcParams.update({'font.size':16,'pdf.compression':9})
fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
stroke=[pe.withStroke(linewidth=4,foreground='white')]
for ax,tag,cmap,ulc,kind in [(axes[0],'C','Blues','#08306b','qcw'),(axes[1],'L','RdPu','#7a1750','loki')]:
    S=np.load(D+'samples_%s.npz'%tag)
    im=ax.hist2d(S['lf'],S['lm'],bins=50,norm=colors.LogNorm(),cmap=cmap,rasterized=True)[3]
    R=P[tag]['rows']; x=np.array([np.log10(r['f_c']*1e-9) for r in R]); y=np.array([r['UL'] for r in R])
    lo=np.array([r['lo'] for r in R]); hi=np.array([r['hi'] for r in R]); ok=np.array([r['pas'] for r in R])
    ax.step(x,y,where='mid',color=ulc,lw=1.6,alpha=0.8,zorder=3,path_effects=stroke)
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',ecolor=ulc,elinewidth=1.4,capsize=3,zorder=4)
    ax.plot(x[ok],y[ok],'o',color=ulc,mec='white',mew=1.2,ms=8,zorder=5)
    ax.plot(x[~ok],y[~ok],'o',mfc='white',mec=ulc,mew=1.5,ms=7,zorder=5)
    ax.plot(F_EM,STAR[kind],marker='*',ms=22,color=ulc,mec='white',mew=1.2,zorder=6)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.15); cb.set_label('Number of Samples',fontsize=13)
    ax.tick_params(direction='in',top=True,right=True,which='both')
    ax.set_xlabel(r'$\log_{10} f_{\rm GW}$',fontsize=20)
    ax.legend(handles=[Line2D([],[],marker='o',ls='none',color=ulc,mec='white',ms=8,label=r'ESS $\geq 50$'),Line2D([],[],marker='o',ls='none',mfc='white',mec=ulc,ms=7,label=r'ESS $< 50$')],loc='lower left',fontsize=12,framealpha=0.9)
axes[0].set_ylabel(r'$\log_{10} \mathcal{M}_c$',fontsize=20)
plt.tight_layout()
fn='G2D1_broad_freq_segmented_UL_sidebyside_4core_ess.pdf'
fig.savefig(D+fn,bbox_inches='tight',dpi=110); fig.savefig(D+fn.replace('.pdf','.png'),dpi=90,bbox_inches='tight')
import os; print('done',os.path.getsize(D+fn))
