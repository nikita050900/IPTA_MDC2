# ess_bins_12oct.py : per bin ESS for Figure 6, Runs C and L
import numpy as np, h5py, json, time
OUTD='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_2oct/'; H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
LOG=open(OUTD+'ess_bins_12oct.log','w')
def log(*a):
    s=' '.join(str(v) for v in a); LOG.write(s+'\n'); LOG.flush()
# acor: exact numpy port of acor.c v1.1.1 (J. Goodman, wrapped by D. Foreman-Mackey), maxlag=10 as in acor.acor default,
# the call used by enterprise_extensions.model_utils.ul. Verified against the compiled C to 1e-12 on AR(1) series.
ACOR='acor (numpy port of acor 1.1.1 C code, maxlag 10)'
def _acor(X, maxlag, st):
    L=len(X); iMax=L-maxlag
    if L < 5*maxlag: return 1
    X=X-X.mean()
    C=np.array([np.dot(X[:iMax],X[s:s+iMax]) for s in range(maxlag+1)])/iMax
    D=C[0]+2*C[1:].sum()
    if D<0: return 2
    st['sigma']=np.sqrt(D/L); st['tau']=D/C[0]
    if st['tau']*5 < maxlag: return 0
    Lh=L//2
    Y=X[0:2*Lh:2]+X[1:2*Lh:2]
    _acor(Y,maxlag,st)
    D=.25*st['sigma']**2*L; st['tau']=D/C[0]; st['sigma']=np.sqrt(D/L)
    return 0
def tau_of(x, maxlag=10):
    st={}; r=_acor(np.asarray(x,dtype=np.float64),maxlag,st)
    if r!=0: raise RuntimeError('acor failed code %d'%r)
    return float(st['tau'])
import emcee
def tau_emcee(x): return float(emcee.autocorr.integrated_time(x,quiet=True)[0])
log('tau estimator:',ACOR)
TARGET=75.4; TOL=0.01; Q=0.95
DF=1/(14.962*365.25*86400.0)
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*Tsun**(5/3)*np.pi**(2/3)*c/mpc)
TAU_C=11136.91
def cols(fn,names):
    with h5py.File(H5+fn,'r') as f:
        pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
        ds=f['samples_cold']; N=ds.shape[1]; idx=[pn.index(n) for n in names]; lo,hi=min(idx),max(idx)+1
        out={n:np.empty(N) for n in names}
        for a in range(0,N,5_000_000):
            b=min(a+5_000_000,N); blk=ds[0,a:b,lo:hi]
            for n,i in zip(names,idx): out[n][a:b]=blk[:,i-lo]
    return out,N
def ul(x): return float(np.log10(np.quantile(10**x,Q)))
def bins(tag,lf,lm,pos,N,essfun):
    f=10**lf; edges=np.arange(f.min(),f.max()+DF,DF); k=np.digitize(f,edges)-1; rows=[]
    for b in range(len(edges)-1):
        m=k==b; n=int(m.sum())
        if n==0: continue
        x=lm[m]; p=pos[m]
        r=dict(f_lo=float(edges[b]*1e9),f_hi=float(edges[b+1]*1e9),N=n,states=int(len(np.unique(np.c_[x,lf[m]],axis=0))),UL=ul(x))
        r.update(essfun(x,p))
        h1=x[p<N//2]; h2=x[p>=N//2]
        r['UL_h1']=ul(h1) if h1.size else None; r['UL_h2']=ul(h2) if h2.size else None
        r['dhalf']=abs(r['UL_h1']-r['UL_h2']) if (h1.size and h2.size) else None
        rows.append(r); log(tag,json.dumps(r))
    return rows
OUT={'estimator':ACOR,'DF_nHz':DF*1e9}
t0=time.time()
C,N=cols('G2D1_broad_UL_4core.h5',['0_log10_fgw','0_log10_h','0_log10_mc'])
ldl=CONST+(5/3)*C['0_log10_mc']+(2/3)*C['0_log10_fgw']-C['0_log10_h']
idx=np.flatnonzero(np.abs(10**ldl-TARGET)<=TOL*TARGET); del ldl
lf=C['0_log10_fgw'][idx]; lm=C['0_log10_mc'][idx]; del C
def essC(x,p): return dict(ESS=int(len(np.unique(p//int(20*TAU_C)))))
OUT['C']=dict(N=N,Nsurv=int(idx.size),tau_mc_full=TAU_C,block_len=int(20*TAU_C),bins=bins('C',lf,lm,idx,N,essC))
log('C done',time.time()-t0); del lf,lm,idx
json.dump(OUT,open(OUTD+'ess_bins_12oct.json','w'),indent=1)
t0=time.time()
L,N=cols('G2D2_broad_UL_loki_100M_lastTOA_4core.h5',['0_log10_dist','0_log10_fgw','0_log10_mc'])
m=np.abs(10**L['0_log10_dist']-TARGET)<=TOL*TARGET; pos=np.flatnonzero(m)
lf=L['0_log10_fgw'][m]; lm=L['0_log10_mc'][m]; full_mc=L['0_log10_mc']; del L,m
tau_whole=tau_of(full_mc); tau_whole_emcee=10*tau_emcee(full_mc[::10]); del full_mc
log('L whole tau emcee',tau_whole_emcee)
log('L whole chain tau',tau_whole,time.time()-t0)
def essL(x,p):
    try:
        t=tau_of(x)
        try: te=tau_emcee(x)
        except Exception: te=None
        return dict(tau_bin=t,ESS=float(len(x)/t),ESS_wholetau=float(len(x)/tau_whole),tau_bin_emcee=te,ESS_emcee=(float(len(x)/te) if te else None))
    except Exception as e: return dict(tau_bin=None,ESS=None,ESS_wholetau=float(len(x)/tau_whole),err=repr(e)[:120])
OUT['L']=dict(N=N,in_window=int(pos.size),tau_mc_whole=tau_whole,tau_mc_whole_emcee=tau_whole_emcee,bins=bins('L',lf,lm,pos,N,essL))
json.dump(OUT,open(OUTD+'ess_bins_12oct.json','w'),indent=1)
log('DONE')
