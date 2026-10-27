# perbin_ess_19oct.py : per bin ESS, Eq. 26 order statistic intervals, h0 limits and slope, 2.6 sigma recheck
import numpy as np, h5py, json, time, emcee
from scipy.stats import gaussian_kde
OUTD='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_19oct/'; H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
LOG=open(OUTD+'perbin_ess_19oct.log','w')
def log(*a):
    s=time.strftime('%H:%M:%S ')+' '.join(str(v) for v in a); print(s,flush=True); LOG.write(s+'\n'); LOG.flush()
TARGET=75.4; TOL=0.01; Q=0.95; ESSMIN=50; NBOOT=300
DF=1/(14.962*365.25*86400.0)
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*Tsun**(5/3)*np.pi**(2/3)*c/mpc)
def tau_emcee(x, thin=1):
    return thin*float(emcee.autocorr.integrated_time(np.asarray(x[::thin],dtype=np.float64),c=5,quiet=True)[0])
def _acor(X, maxlag, st):
    L=len(X); iMax=L-maxlag
    if L < 5*maxlag: return 1
    X=X-X.mean()
    C=np.array([np.dot(X[:iMax],X[s:s+iMax]) for s in range(maxlag+1)])/iMax
    D=C[0]+2*C[1:].sum()
    if D<0: return 2
    st['sigma']=np.sqrt(D/L); st['tau']=D/C[0]
    if st['tau']*5 < maxlag: return 0
    Lh=L//2; Y=X[0:2*Lh:2]+X[1:2*Lh:2]
    _acor(Y,maxlag,st)
    D=.25*st['sigma']**2*L; st['tau']=D/C[0]; st['sigma']=np.sqrt(D/L)
    return 0
def tau_acor(x):
    st={}; r=_acor(np.asarray(x,dtype=np.float64),10,st)
    return float(st['tau']) if r==0 else float('nan')
def cols(fn,names):
    with h5py.File(H5+fn,'r') as f:
        pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
        ds=f['samples_cold']; nch,N=ds.shape[0],ds.shape[1]; idx=[pn.index(n) for n in names]; lo,hi=min(idx),max(idx)+1
        out={n:np.empty((nch,N)) for n in names}
        for ch in range(nch):
            for a in range(0,N,5_000_000):
                b=min(a+5_000_000,N); blk=ds[ch,a:b,lo:hi]
                for n,i in zip(names,idx): out[n][ch,a:b]=blk[:,i-lo]
    log('read',fn,'chains',nch,'N',N); return out,nch,N
def lq(x,q): return float(np.log10(np.quantile(10**x,q)))
def os_interval(x,ess):
    d=np.sqrt(Q*(1-Q)/ess) if ess>0 else np.inf
    lo=lq(x,max(Q-d,0.0)); hi=lq(x,min(Q+d,1.0))
    return lo,hi,float(d),bool(Q+d>=1.0)
def eq26_kde(x,ness):
    v=10**x; ul=np.quantile(v,Q); sub=v if len(v)<500_000 else v[::len(v)//500_000]
    f=gaussian_kde(sub).evaluate([ul])[0]
    return float(np.log10(ul)), float(np.sqrt(Q*(1-Q)/ness)/f/(ul*np.log(10)))
def groups(vals,pos,L):
    bid=pos//int(L); o=np.argsort(bid,kind='stable'); bs=bid[o]; vs=vals[o]
    e=np.searchsorted(bs,np.unique(bs)); return np.split(vs,e[1:])
def boot(x,pos,L,rng,nboot=NBOOT):
    g=groups(x,pos,L); ul=np.quantile(10**x,Q); Fk=np.empty(nboot); qk=np.empty(nboot)
    for k in range(nboot):
        s=np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))])
        Fk[k]=np.mean(10**s<=ul); qk[k]=np.log10(np.quantile(10**s,Q))
    v=Fk.var(ddof=1)
    return dict(blocks=len(g), ESS_boot=(float(Q*(1-Q)/v) if v>0 else None), sig_boot_q=float(qk.std(ddof=1)))
def fit(x,y):
    x=np.asarray(x); y=np.asarray(y); A=np.c_[np.ones_like(x),x]
    co,res,_,_=np.linalg.lstsq(A,y,rcond=None); r=y-A@co; n=len(x)
    if n<3: return dict(n=n,slope=float(co[1]) if n==2 else None)
    s2=r@r/(n-2); cov=s2*np.linalg.inv(A.T@A)
    return dict(n=n,slope=float(co[1]),slope_se=float(np.sqrt(cov[1,1])),intercept=float(co[0]))
OUT={'estimator':'emcee.autocorr.integrated_time c=5 on the full stored cold chain of log10 Mc, no thinning','Q':Q,'ESSMIN':ESSMIN,'DF_nHz':DF*1e9}
def run(tag,fn,kind):
    t0=time.time(); rng=np.random.default_rng(11)
    if kind=='qcw':
        D,nch,N=cols(fn,['0_log10_fgw','0_log10_h','0_log10_mc'])
    else:
        D,nch,N=cols(fn,['0_log10_dist','0_log10_fgw','0_log10_mc'])
    taus=[]; taus_ac=[]
    for ch in range(nch):
        taus.append(tau_emcee(D['0_log10_mc'][ch])); taus_ac.append(tau_acor(D['0_log10_mc'][ch]))
    log(tag,'tau emcee per chain',taus,'acor',taus_ac,'ratio',[a/b for a,b in zip(taus,taus_ac)])
    LF=[];LM=[];LH=[];POS=[];CH=[]
    for ch in range(nch):
        lf=D['0_log10_fgw'][ch]; lm=D['0_log10_mc'][ch]
        if kind=='qcw':
            lh=D['0_log10_h'][ch]; ldl=CONST+(5/3)*lm+(2/3)*lf-lh; m=np.abs(10**ldl-TARGET)<=TOL*TARGET
        else:
            ld=D['0_log10_dist'][ch]; lh=CONST+(5/3)*lm+(2/3)*lf-ld; m=np.abs(10**ld-TARGET)<=TOL*TARGET
        p=np.flatnonzero(m); LF.append(lf[p]); LM.append(lm[p]); LH.append(lh[p]); POS.append(p); CH.append(np.full(p.size,ch))
    del D
    lf=np.concatenate(LF); lm=np.concatenate(LM); lh=np.concatenate(LH); pos=np.concatenate(POS); chn=np.concatenate(CH)
    tau_arr=np.array(taus)
    r=dict(file=fn,nchains=nch,N=N,tau_emcee=taus,tau_acor=taus_ac,Nkept=int(lf.size))
    # marginalized broad limit
    ness=float(sum(np.sum(chn==ch)/taus[ch] for ch in range(nch)))
    r['marg_mc_eq26_kde']=eq26_kde(lm,ness); r['marg_ness']=ness
    if kind=='qcw':
        r['marg_boot']=boot(lm,pos+chn*N,20*taus[0],rng)
    f=10**lf; edges=np.arange(f.min(),f.max()+DF,DF); k=np.digitize(f,edges)-1; rows=[]
    for b in range(len(edges)-1):
        m=k==b; n=int(m.sum())
        if n==0: continue
        x=lm[m]; xh=lh[m]; p=pos[m]; cc=chn[m]
        row=dict(f_lo=float(edges[b]*1e9),f_hi=float(edges[b+1]*1e9),f_c=float(0.5*(edges[b]+edges[b+1])*1e9),N=n,UL=lq(x,Q),ULh=lq(xh,Q))
        row['ESS_Ntau']=float(sum(np.sum(cc==ch)/taus[ch] for ch in range(nch)))
        row['ESS_Ntau_acor']=float(sum(np.sum(cc==ch)/taus_ac[ch] for ch in range(nch)))
        if kind=='qcw' or True:
            if kind=='qcw':
                bt=boot(x,p+cc*N,20*taus[0],rng)
            else:
                th=100; bt=boot(x[::th],(p+cc*N)[::th],20*taus[0],rng,200)
                if bt['ESS_boot'] is not None: bt['ESS_boot']=bt['ESS_boot']  # thinning by 100 << 20 tau keeps block structure
            row.update(bt)
        ess=row['ESS_boot'] if kind=='qcw' else row['ESS_Ntau']
        row['ESS']=ess if ess is not None else 0.0
        lo,hi,d,unb=os_interval(x,row['ESS']); row.update(UL_lo=lo,UL_hi=hi,d=d,hi_unbounded=unb)
        lo,hi,_,unbh=os_interval(xh,row['ESS']); row.update(ULh_lo=lo,ULh_hi=hi,hih_unbounded=unbh)
        row['pass']=bool(row['ESS']>=ESSMIN); row['old_plotted']=bool(n>=50)
        rows.append(row)
    r['bins']=rows
    P=[w for w in rows if w['pass']]
    lfc=[np.log10(w['f_c']*1e-9) for w in P]
    r['fit_h0_pass']=fit(lfc,[w['ULh'] for w in P]); r['fit_mc_pass']=fit(lfc,[w['UL'] for w in P])
    A=[w for w in rows if w['old_plotted']]; lfa=[np.log10(w['f_c']*1e-9) for w in A]
    r['fit_h0_oldplotted']=fit(lfa,[w['ULh'] for w in A]); r['fit_mc_oldplotted']=fit(lfa,[w['UL'] for w in A])
    r['npass']=len(P); r['nbins']=len(rows); r['nold']=len(A)
    np.savez(OUTD+'samples_%s.npz'%tag,lf=lf,lm=lm,lh=lh)
    log(tag,'done',time.time()-t0,'npass',len(P),'of',len(rows),json.dumps({k:v for k,v in r.items() if k!='bins'}))
    for w in rows: log(tag,json.dumps(w))
    return r
OUT['C']=run('C','G2D1_broad_UL_4core.h5','qcw'); json.dump(OUT,open(OUTD+'perbin_ess_19oct.json','w'),indent=1)
OUT['L']=run('L','G2D2_broad_UL_loki_100M_lastTOA_4core.h5','loki'); json.dump(OUT,open(OUTD+'perbin_ess_19oct.json','w'),indent=1)
# Enterprise Run H for the sigma comparison
with h5py.File(H5+'core_single_MDC2_DS1_varyfgw.h5','r') as fh:
    pn=[p.decode() if isinstance(p,bytes) else str(p) for p in fh['params'][:]]; chn=fh['chain'][3000:,:]
mc=chn[:,pn.index('log10_mc')]; th=tau_emcee(mc); ta=tau_acor(mc)
OUT['H']=dict(N=len(mc),tau_emcee=th,tau_acor=ta,eq26_emcee=eq26_kde(mc,len(mc)/th),eq26_acor=eq26_kde(mc,len(mc)/ta))
log('H',json.dumps(OUT['H']))
json.dump(OUT,open(OUTD+'perbin_ess_19oct.json','w'),indent=1)
log('DONE')
