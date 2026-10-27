# ess_indicator_L_19oct.py : per bin N_s for QuickGWecc from the autocorrelation of the quantile indicator (no bootstrap)
import numpy as np, h5py, json, emcee, time
D='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_19oct/'; H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
Q=0.95; TARGET=75.4; TOL=0.01; TH=10
with h5py.File(H5+'G2D2_broad_UL_loki_100M_lastTOA_4core.h5','r') as f:
    pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]; ds=f['samples_cold']; N=ds.shape[1]
    names=['0_log10_dist','0_log10_fgw','0_log10_mc']; idx=[pn.index(n) for n in names]; lo,hi=min(idx),max(idx)+1
    X={n:np.empty(N) for n in names}
    for a in range(0,N,5_000_000):
        b=min(a+5_000_000,N); blk=ds[0,a:b,lo:hi]
        for n,i in zip(names,idx): X[n][a:b]=blk[:,i-lo]
win=np.abs(10**X['0_log10_dist']-TARGET)<=TOL*TARGET
lf=X['0_log10_fgw']; lm=X['0_log10_mc']; del X
P=json.load(open(D+'post_19oct.json'))['L']['rows']
fk=10**lf[win]; DF=1/(14.962*365.25*86400.0); edges=np.arange(fk.min(),fk.max()+DF,DF)
kall=np.full(N,-1); kall[win]=np.digitize(10**lf[win],edges)-1
Nh=N//2; out=[]
def lq(x,q): return float(np.log10(np.quantile(10**x,q)))
for r in P:
    b=int(round((r['f_lo']*1e-9-edges[0])/DF)); inb=(kall==b); nb=int(inb.sum())
    x=lm[inb]; ul=np.quantile(10**x,Q); ull=np.log10(ul)
    Z=inb*((10**lm<=ul).astype(float)-Q)          # ratio estimator influence function
    Zt=Z[::TH]; tz=TH*float(emcee.autocorr.integrated_time(Zt,c=5,quiet=True)[0])
    varF=Zt.var()*tz/N/(nb/N)**2                  # delta method variance of the bin CDF at the limit
    ess=Q*(1-Q)/varF
    d=np.sqrt(Q*(1-Q)/ess); lo_,hi_=lq(x,max(Q-d,0)),lq(x,min(Q+d,1))
    h1=lm[:Nh][inb[:Nh]]; h2=lm[Nh:][inb[Nh:]]
    u1=lq(h1,Q) if h1.size else np.nan; u2=lq(h2,Q) if h2.size else np.nan
    out.append(dict(f_lo=r['f_lo'],N=nb,tau_Z=tz,ESS_ind=ess,ESS_Ntau=r['ESS'],UL=ull,lo=lo_,hi=hi_,sig_boot=r['sig_boot'],u1=u1,u2=u2))
    print(json.dumps(out[-1]),flush=True)
json.dump(out,open(D+'ess_indicator_L_19oct.json','w'),indent=1)
z=np.array([(o['u1']-o['u2'])/(2*max((o['hi']-o['lo'])/2,1e-6)) for o in out if np.isfinite(o['u1']) and np.isfinite(o['u2'])])
zb=np.array([(o['u1']-o['u2'])/(2*o['sig_boot']) for o in out if np.isfinite(o['u1']) and np.isfinite(o['u2'])])
print('split half rms indicator-ESS interval',round(float(np.sqrt(np.mean(z**2))),2),'frac>2',round(float(np.mean(abs(z)>2)),2),'| boot',round(float(np.sqrt(np.mean(zb**2))),2))
print('ESS_ind range',min(o['ESS_ind'] for o in out),max(o['ESS_ind'] for o in out))
