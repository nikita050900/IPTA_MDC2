# freqseg_check_12oct.py : checks for Figure 6 (frequency segmented limits), Runs C and L
import numpy as np, h5py, json, pickle, time, sys
from scipy.stats import gaussian_kde, spearmanr
import emcee
OUTD='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_2oct/'
H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
LOG=open(OUTD+'freqseg_check_12oct.log','w')
def log(*a):
    s=' '.join(str(x) for x in a); print(s); LOG.write(s+'\n'); LOG.flush()
TARGET=75.4; TOL=0.01; Q=0.95; NMIN=50; NBOOT=300
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*Tsun**(5/3)*np.pi**(2/3)*c/mpc)
TAU={'C':11136.91,'L':46849.0}
OUT={}
# T_span from the pulsar objects, as in run_freq_seg_UL_4core.py
try:
    ps=pickle.load(open('/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/psr_objects/G2D1_IPTA_MDC2_all_pulsars.pkl','rb'))
    Tspan=max(p.toas.max() for p in ps)-min(p.toas.min() for p in ps)
    OUT['Tspan_yr']=Tspan/(365.25*86400); OUT['df_nHz']=1e9/Tspan
except Exception as e:
    log('pkl failed', repr(e)); Tspan=14.962*365.25*86400; OUT['Tspan_yr']='pkl failed, used 14.962'
log('Tspan_yr', OUT['Tspan_yr'], 'df_nHz', OUT.get('df_nHz'))
DF=1/(14.962*365.25*86400.0)   # value hardcoded in fig6_optionB_2oct.py
def cols(fn,names):
    with h5py.File(H5+fn,'r') as f:
        pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
        ds=f['samples_cold']; N=ds.shape[1]; idx=[pn.index(n) for n in names]; lo,hi=min(idx),max(idx)+1
        out={n:np.empty(N) for n in names}
        for a in range(0,N,5_000_000):
            b=min(a+5_000_000,N); blk=ds[0,a:b,lo:hi]
            for n,i in zip(names,idx): out[n][a:b]=blk[:,i-lo]
    return out,N
def eq7(logv,n_eff,dens='kde'):
    v=10**logv; ul=np.quantile(v,Q)
    if dens=='kde':
        sub=v[::max(1,len(v)//200000)]; f=gaussian_kde(sub).evaluate([ul])[0]
    else:
        h,e=np.histogram(v,bins=100,density=True); j=min(max(np.searchsorted(e,ul)-1,0),99); f=h[j]
    if f<=0 or n_eff<=0: return np.nan
    return float(np.sqrt(Q*(1-Q))/(f*np.sqrt(n_eff))/(ul*np.log(10)))
rng=np.random.default_rng(11)
def boot(vals,pos,L):
    bid=pos//max(int(L),1); o=np.argsort(bid,kind='stable'); bs,vs=bid[o],vals[o]
    e=np.searchsorted(bs,np.unique(bs)); g=np.split(vs,e[1:]); res=np.empty(NBOOT)
    for k in range(NBOOT):
        pick=rng.integers(0,len(g),len(g)); res[k]=np.quantile(np.concatenate([g[i] for i in pick]),Q)
    sizes=np.sort(np.array([len(x) for x in g]))[::-1]
    return float(res.std(ddof=1)), len(g), float(sizes[0]/sizes.sum()), int(np.searchsorted(np.cumsum(sizes)/sizes.sum(),0.9)+1)
def analyse(tag,lf,lm,pos,N,tau,rho_fig,thin):
    f=10**lf; edges=np.arange(f.min(),f.max()+DF,DF); k=np.digitize(f,edges)-1
    rows=[]
    for b in range(len(edges)-1):
        m=k==b; n=int(m.sum())
        if n==0: continue
        x=lm[m]; p=pos[m]
        st=int(len(np.unique(np.c_[x,lf[m]],axis=0)))
        ul=float(np.log10(np.quantile(10**x,Q)))
        r=dict(f_lo=edges[b]*1e9,f_hi=edges[b+1]*1e9,n_stored=n,states=st,UL=ul,plotted=n>=NMIN)
        if n>=2:
            sel=slice(None,None,thin)
            se,nb,top,n90=boot(x[sel],p[sel],20*tau)
            r.update(err_boot=se,blocks=nb,top_block_share=top,blocks_for_90pct=n90)
            r['N_over_tau']=n/tau
            r['err_eq7_Ntau_kde']=eq7(x,n/tau,'kde'); r['err_eq7_Ntau_hist']=eq7(x,n/tau,'hist')
            if rho_fig is not None:
                r['n_rho_fig']=n*rho_fig; r['err_fig_kde']=eq7(x,n*rho_fig,'kde')
            h1=x[p<N//2]; h2=x[p>=N//2]
            r['UL_h1']=float(np.log10(np.quantile(10**h1,Q))) if h1.size else np.nan
            r['UL_h2']=float(np.log10(np.quantile(10**h2,Q))) if h2.size else np.nan
        rows.append(r)
        log(tag, json.dumps({kk:(round(v,4) if isinstance(v,float) else v) for kk,v in r.items()}))
    return rows,edges
# Run C, masked QuickCW
t0=time.time()
C,N=cols('G2D1_broad_UL_4core.h5',['0_log10_fgw','0_log10_h','0_log10_mc'])
ldl=CONST+(5/3)*C['0_log10_mc']+(2/3)*C['0_log10_fgw']-C['0_log10_h']
idx=np.flatnonzero(np.abs(10**ldl-TARGET)<=TOL*TARGET); del ldl
lf=C['0_log10_fgw'][idx]; lm=C['0_log10_mc'][idx]; del C
tau_re=float(emcee.autocorr.integrated_time(lm,quiet=True)[0]); rho=min(1.0,(len(lm)/tau_re)/len(lm))
OUT['C']=dict(N=N,Nsurv=int(idx.size),distinct_states=int(len(np.unique(np.c_[lm,lf],axis=0))),tau_full_mc=TAU['C'],tau_reindexed=tau_re,rho_fig=rho,
              fullband_UL_all=float(np.log10(np.quantile(10**lm,Q))))
log('C', json.dumps(OUT['C']), time.time()-t0)
OUT['C']['bins'],eC=analyse('C',lf,lm,idx,N,TAU['C'],rho,1)
pl=np.zeros(len(lm),bool); fC=10**lf; kC=np.digitize(fC,eC)-1
for b in set(kC):
    if (kC==b).sum()>=NMIN: pl|=(kC==b)
OUT['C']['fullband_UL_plotted_bins_only']=float(np.log10(np.quantile(10**lm[pl],Q))); OUT['C']['n_in_dropped_bins']=int((~pl).sum())
del lf,lm,idx
json.dump(OUT,open(OUTD+'freqseg_check_12oct.json','w'),indent=1,default=float)
# Run L, QuickGWecc
t0=time.time()
L,N=cols('G2D2_broad_UL_loki_100M_lastTOA_4core.h5',['0_log10_dist','0_log10_fgw','0_log10_mc'])
m=np.abs(10**L['0_log10_dist']-TARGET)<=TOL*TARGET
pos=np.flatnonzero(m); lf=L['0_log10_fgw'][m]; lm=L['0_log10_mc'][m]; del L,m
OUT['L']=dict(N=N,in_window=int(pos.size),distinct_states=int(len(np.unique(np.c_[lm,lf],axis=0))),tau_full_mc=TAU['L'],fullband_UL_all=float(np.log10(np.quantile(10**lm,Q))))
log('L', json.dumps(OUT['L']), time.time()-t0)
OUT['L']['bins'],_=analyse('L',lf,lm,pos,N,TAU['L'],None,10)
json.dump(OUT,open(OUTD+'freqseg_check_12oct.json','w'),indent=1,default=float)
log('DONE')
