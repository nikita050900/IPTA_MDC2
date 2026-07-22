# ntol figure, final version: log-x, ESS-corrected errors, Run M diamond,
# ntol-10 QuickCW-dL cyan point. Estimators identical to the sweep notebook.
import numpy as np, h5py, emcee
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
Q=0.95; TARGET=75.4
NTOL=np.array([0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.30,0.50])

megaparsec=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
F=np.log10(3.7e-9)
def dl_of(mc,h):  # Mpc, from Eq 2
    return 10**(np.log10(2.0)+(5.0/3.0)*(mc+np.log10(Tsun))+(2.0/3.0)*(np.log10(np.pi)+F)-h-np.log10(megaparsec)+np.log10(c))

def tau_of(x):
    th=10 if len(x)>5_000_000 else 1
    return th*float(emcee.autocorr.integrated_time(x[::th],quiet=True)[0])

def ul_err(mc_log, n_eff):
    mc=10**mc_log
    ul=np.quantile(mc,Q)
    f=gaussian_kde(mc if len(mc)<500_000 else mc[::max(1,len(mc)//500_000)]).evaluate([ul])[0]
    err=np.sqrt(Q*(1-Q))/(f*np.sqrt(n_eff))
    return np.log10(ul), err/(ul*np.log(10))

print('loading Run D outfile...', flush=True)
with h5py.File(H5+'G2D1_narrow_UL_fixed_gamma_17_dec_2025_outfile.h5','r') as f:
    pn=[p.decode() if isinstance(p,bytes) else p for p in f['par_names'][:]]
    d=f['samples_cold'][0,:,:]
mc=d[:,pn.index('0_log10_mc')]; h=d[:,pn.index('0_log10_h')]
dl=dl_of(mc,h); del d
tauD=tau_of(mc); essD=len(mc)/tauD
print('Run D: N=%d tau_mc=%.0f ESS=%.0f'%(len(mc),tauD,essD), flush=True)

rows=[]
for nt in NTOL:
    m=np.abs(dl-TARGET)<=nt*TARGET
    ns=int(m.sum())
    neff=min(ns, essD)
    ul,er=ul_err(mc[m], neff)
    rows.append((nt*100, ns, ul, er))
    print(' ntol %.1f%%: n=%d neff=%d UL=%.4f +/- %.4f'%(nt*100,ns,neff,ul,er), flush=True)

def loki_point(fn):
    with h5py.File(H5+fn,'r') as f:
        pn=[p.decode() if isinstance(p,bytes) else p for p in f['par_names'][:]]
        x=f['samples_cold'][0,:,pn.index('0_log10_mc')]
    tau=tau_of(x); ess=len(x)/tau
    ul,er=ul_err(x, ess)
    print('%s: N=%d tau=%.0f ESS=%.0f UL=%.4f +/- %.4f'%(fn,len(x),tau,ess,ul,er), flush=True)
    return len(x), ul, er

nM,ulM,erM=loki_point('G2D1_fixed_UL_loki_100M_lastTOA_10_Jul_2026.h5')
nC,ulC,erC=loki_point('G2D1_fixed_UL_loki_100M_lastTOA_ntol_10_15_Jul_2026.h5')

print('loading Enterprise...', flush=True)
with h5py.File(H5+'core_single_MDC2_DS1.h5','r') as f:
    pn=[p.decode() if isinstance(p,bytes) else p for p in f['params'][:]]
    ch=f['chain'][...]
    try: burn=f['metadata/burn'][()]
    except Exception: burn=3000
eMC=ch[burn:, pn.index('log10_mc')]
tauE=tau_of(eMC); essE=len(eMC)/tauE
ulE,erE=ul_err(eMC, essE)
print('ENT: N=%d ESS=%.0f UL=%.4f +/- %.4f'%(len(eMC),essE,ulE,erE), flush=True)

# figure
plt.rcParams.update({'font.size':9})
fig,(a,b)=plt.subplots(2,1,figsize=(3.5,4.6),sharex=True)
nt=[r[0] for r in rows]; ns=[r[1] for r in rows]; ul=[r[2] for r in rows]; er=[r[3] for r in rows]
a.plot(nt,ns,'o-',color='#4477AA',ms=4,lw=1.2)
a.axhline(essD,color='#4477AA',ls=':',lw=1.0)
a.scatter([1.0],[nM],marker='D',color='#CC6677',zorder=5,s=28)
a.scatter([10.0],[nC],marker='D',color='#66CCEE',zorder=5,s=28)
a.axhline(len(eMC),color='#228833',ls='--',lw=1.0)
a.set_yscale('log'); a.set_xscale('log'); a.set_ylabel(r'$N_{\rm surviving}$')
a.text(0.03,0.86,'(a)',transform=a.transAxes)
b.errorbar(nt,ul,yerr=er,fmt='o-',color='#4477AA',ms=4,lw=1.2,capsize=2)
b.axhline(ulE,color='#228833',ls='--',lw=1.0)
b.fill_between([0.4,60],[ulE-erE]*2,[ulE+erE]*2,color='#228833',alpha=0.18,lw=0)
b.errorbar([1.0],[ulM],yerr=[erM],fmt='D',color='#CC6677',ms=5,capsize=2,zorder=5)
b.errorbar([10.0],[ulC],yerr=[erC],fmt='D',color='#66CCEE',ms=5,capsize=2,zorder=5)
b.set_xscale('log'); b.set_xlim(0.4,60)
b.set_xlabel(r'$\eta_{\rm tol}$ [%]'); b.set_ylabel(r'$\log_{10}(\mathcal{M}_c/M_\odot)^{95\%}$')
b.text(0.03,0.86,'(b)',transform=b.transAxes)
plt.tight_layout()
fig.savefig('ntol_sweep.png',dpi=300,bbox_inches='tight')
fig.savefig('ntol_sweep.pdf',bbox_inches='tight')
print('saved ntol_sweep.png/pdf', flush=True)
