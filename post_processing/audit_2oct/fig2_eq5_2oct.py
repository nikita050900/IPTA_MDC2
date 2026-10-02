# fig2_eq5_2oct.py : Figure 2 with Enterprise band and QuickGWecc diamonds from Equation 5 (N/tau_Mc), as in Table 5
import matplotlib; matplotlib.use('Agg')
import numpy as np, h5py, json, emcee
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
PP='/scratch/na00078/projects/IPTA_MDC2/post_processing/'; H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'; OUTD=PP+'audit_2oct/'
OUT=json.load(open(PP+'final_paper_analysis.json'))
A=json.load(open(OUTD+'ul_audit_ENT.json')); M=json.load(open(OUTD+'ul_audit_C_M_L.json'))
ent=A['ENT_I']['mc_eq5_taumc']; lokiM=M['M']['mc_eq5_taumc']
def eq5(logv,n_eff):
    v=10**logv; ul=np.quantile(v,0.95); sub=v[::max(1,len(v)//500_000)]
    f=gaussian_kde(sub).evaluate([ul])[0]; err=np.sqrt(0.95*0.05)/(f*np.sqrt(n_eff))
    return float(np.log10(ul)), float(err/(ul*np.log(10)))
with h5py.File(H5+'G2D1_fixed_UL_loki_100M_lastTOA_ntol_10_15_Jul_2026.h5','r') as f:
    pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
    mc10=np.asarray(f['samples_cold'][0,:,pn.index('0_log10_mc')],np.float64)
tau10=10*float(emcee.autocorr.integrated_time(mc10[::10],quiet=True)[0])
loki10=eq5(mc10,len(mc10)/tau10); del mc10
with h5py.File(H5+'G2D1_narrow_UL_4core.h5','r') as f:
    pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
    mcD=np.empty(f['samples_cold'].shape[1]); i=pn.index('0_log10_mc')
    for a in range(0,len(mcD),5_000_000): mcD[a:a+5_000_000]=f['samples_cold'][0,a:a+5_000_000,i]
unmasked=float(np.quantile(mcD[mcD<=10.0],0.95))  # bound 10 of Table 1, as in fig2_nomask_final.py; del mcD
json.dump(dict(ent=ent,lokiM=lokiM,loki10=loki10,tau10=tau10,unmasked=unmasked),open(OUTD+'fig2_eq5_values.json','w'),indent=1)
plt.rcParams.update({'font.size':9})
fig,b=plt.subplots(1,1,figsize=(3.5,2.6))
nt=[r[0] for r in OUT['sweep']]; ul=[r[2] for r in OUT['sweep']]; er=[r[3] for r in OUT['sweep']]
b.errorbar(nt,ul,yerr=er,fmt='o-',color='#4477AA',ms=4,lw=1.2,capsize=2)
b.axhline(ent[0],color='#228833',ls='--',lw=1.0)
b.fill_between([0.4,60],[ent[0]-ent[1]]*2,[ent[0]+ent[1]]*2,color='#228833',alpha=0.18,lw=0)
b.axhline(unmasked,color='k',ls='--',lw=1.0)
b.errorbar([1.0],[lokiM[0]],yerr=[lokiM[1]],fmt='D',color='#CC6677',ms=5,capsize=2,zorder=5)
b.errorbar([10.0],[loki10[0]],yerr=[loki10[1]],fmt='D',color='#66CCEE',ms=5,capsize=2,zorder=5)
b.set_xscale('log'); b.set_xlim(0.4,60); b.set_ylim(9.68,10.0)
b.set_xlabel(r'distance tolerance $\eta_{\rm tol}$ [%]')
b.set_ylabel(r'95% UL on $\log_{10}\mathcal{M}_c$')
fig.tight_layout()
fig.savefig(OUTD+'ntol_sweep_4core_eq5.pdf'); fig.savefig(OUTD+'ntol_sweep_4core_eq5.png',dpi=200)
print('done',ent,lokiM,loki10,tau10,unmasked)
