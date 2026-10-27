# fig6_allblock_boot_19oct.py : per bin block bootstrap resampling ALL chain blocks (incl. empty ones), tau(fGW) check, bins kept, ESS, split test
import numpy as np, h5py, json, time, emcee
D='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_19oct/'; H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
LOG=open(D+'allblock_boot_19oct.log','w')
def log(*a):
    s=' '.join(str(v) for v in a); print(s,flush=True); LOG.write(s+'\n'); LOG.flush()
Q=0.95; TARGET=75.4; TOL=0.01; NBOOT=300; DF=1/(14.962*365.25*86400.0)
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3; CONST=np.log10(2*Tsun**(5/3)*np.pi**(2/3)*c/mpc)
def cols(fn,names):
    with h5py.File(H5+fn,'r') as f:
        pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]; ds=f['samples_cold']; N=ds.shape[1]
        idx=[pn.index(n) for n in names]; lo,hi=min(idx),max(idx)+1; out={n:np.empty(N) for n in names}
        for a in range(0,N,5_000_000):
            b=min(a+5_000_000,N); blk=ds[0,a:b,lo:hi]
            for n,i in zip(names,idx): out[n][a:b]=blk[:,i-lo]
    return out,N
def tau(x): return float(emcee.autocorr.integrated_time(np.asarray(x,dtype=np.float64),c=5,quiet=True)[0])
def lq(v,q=Q): return float(np.log10(np.quantile(v,q)))
OUT={}
for tag,fn,kind in [('C','G2D1_broad_UL_4core.h5','qcw'),('L','G2D2_broad_UL_loki_100M_lastTOA_4core.h5','loki')]:
    t0=time.time()
    X,N=cols(fn,['0_log10_fgw','0_log10_h','0_log10_mc'] if kind=='qcw' else ['0_log10_dist','0_log10_fgw','0_log10_mc'])
    tmc=tau(X['0_log10_mc']); tf=tau(X['0_log10_fgw'])
    BL=int(20*tmc) if tf<20*tmc else int(20*tf); log(tag,'tau_mc',round(tmc,1),'tau_fgw',round(tf,1),'20tau_mc',int(20*tmc),'rule tau_fgw<20tau_mc',tf<20*tmc,'block length used',BL)
    if kind=='qcw':
        ldl=CONST+(5/3)*X['0_log10_mc']+(2/3)*X['0_log10_fgw']-X['0_log10_h']; m=np.abs(10**ldl-TARGET)<=TOL*TARGET; del ldl
    else:
        m=np.abs(10**X['0_log10_dist']-TARGET)<=TOL*TARGET
    pos=np.flatnonzero(m); lf=X['0_log10_fgw'][pos]; lm=X['0_log10_mc'][pos]; del X,m
    TH=1 if kind=='qcw' else 10   # thin L by 10 for the bootstrap only (BL >> 10)
    NB=int(np.ceil(N/BL)); rng=np.random.default_rng(11)
    draws=[np.bincount(rng.integers(0,NB,NB),minlength=NB) for _ in range(NBOOT)]   # same block draws for every bin
    f=10**lf; edges=np.arange(f.min(),f.max()+DF,DF); k=np.digitize(f,edges)-1
    rows=[]
    for b in range(len(edges)-1):
        sel=np.flatnonzero(k==b); n=int(sel.size)
        if n==0: continue
        x=10**lm[sel]; p=pos[sel]; ul=lq(x)
        bid=p//BL; nblk=int(np.unique(bid).size)
        xs=x[::TH]; bs=bid[::TH]; o=np.argsort(bs,kind='stable'); xs=xs[o]; bs=bs[o]
        ub,st=np.unique(bs,return_index=True); en=np.r_[st[1:],len(bs)]
        qs=[]
        for d in draws:
            w=d[ub]
            if w.sum()==0: continue
            vals=np.concatenate([np.repeat(xs[s:e],ww) if ww==1 else np.tile(xs[s:e],ww) for s,e,ww in zip(st,en,w) if ww>0])
            qs.append(np.log10(np.quantile(vals,Q)))
        sig=float(np.std(qs,ddof=1)) if len(qs)>1 else float('nan')
        h1=x[p<N//2]; h2=x[p>=N//2]
        u1=lq(h1) if h1.size else None; u2=lq(h2) if h2.size else None
        ess=float(nblk) if kind=='qcw' else float(n/tmc)
        rows.append(dict(f_lo=float(edges[b]*1e9),f_hi=float(edges[b+1]*1e9),f_c=float(0.5*(edges[b]+edges[b+1])*1e9),N=n,blocks=nblk,UL=ul,sig=sig,nres=len(qs),u1=u1,u2=u2,ESS=ess))
        log(tag,json.dumps(rows[-1]))
    def summ(nmin):
        K=[r for r in rows if r['blocks']>=nmin]
        z=np.array([(r['u1']-r['u2'])/(2*r['sig']) for r in K if r['u1'] is not None and r['u2'] is not None and r['sig']>0])
        return dict(nmin=nmin,kept=len(K),of=len(rows),ess50=sum(r['ESS']>=50 for r in K),ess50_fmax=max([r['f_hi'] for r in K if r['ESS']>=50],default=None),
                    split_rms=float(np.sqrt(np.mean(z**2))),split_frac2=float(np.mean(np.abs(z)>2)),nz=len(z),dropped=[round(r['f_lo'],2) for r in rows if r['blocks']<nmin])
    S10=summ(10); S20=summ(20)
    K=[r for r in rows if r['blocks']>=10 and r['f_hi']<=32.0]
    x=np.log10([r['f_c']*1e-9 for r in K]); y=np.array([r['UL'] for r in K]); s=np.array([r['sig'] for r in K])
    W=np.diag(1/s**2); A=np.c_[np.ones_like(x),x]; cov=np.linalg.inv(A.T@W@A); co=cov@A.T@W@y; chi=float(((y-A@co)**2/s**2).sum())
    flat=dict(nbins=len(K),wslope=float(co[1]),wslope_se=float(np.sqrt(cov[1,1])),chi2=chi,dof=len(K)-2,range=[float(y.min()),float(y.max())],wmean=float(np.sum(y/s**2)/np.sum(1/s**2)),chi2_const=float((((y-np.sum(y/s**2)/np.sum(1/s**2))/s)**2).sum()))
    OUT[tag]=dict(N=N,tau_mc=tmc,tau_fgw=tf,BL=BL,NB=NB,Nsurv=int(pos.size),S10=S10,S20=S20,flat_below32=flat,rows=rows)
    log(tag,'S10',json.dumps(S10)); log(tag,'S20',json.dumps(S20)); log(tag,'flat',json.dumps(flat)); log(tag,'time',time.time()-t0)
    json.dump(OUT,open(D+'allblock_boot_19oct.json','w'),indent=1)
log('star offsets: C',OUT['C']['rows'][0]['UL']-9.793,'L',OUT['L']['rows'][0]['UL']-9.771)
log('DONE')
