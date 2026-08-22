#!/usr/bin/env python
# final_paper_analysis.py: masked ESS bootstraps (A,B,C,D,E,F), ntol sweep with
# block-bootstrap errors, QuickCW-dL + Enterprise points, publication figure.
# Estimators follow the sweep notebook; errors via block bootstrap in chain time.
import numpy as np, h5py, json, time, os, traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
PP='/scratch/na00078/projects/IPTA_MDC2/post_processing/'
LOG=PP+'final_paper_analysis.log'
def log(s):
    with open(LOG,'a') as f: f.write(time.strftime('%H:%M:%S ')+str(s)+'\n')

Q=0.95; TARGET=75.4
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*(Tsun)**(5/3)*np.pi**(2/3)*c/mpc)  # log10 dl = CONST + 5/3 mc + 2/3 log10 f - h
NBOOT=300
rng=np.random.default_rng(11)

with open(PP+'sec6_ess_4core.json') as f: ESSJ=json.load(f)

def read_cols(fn, want):
    t0=time.time()
    with h5py.File(H5+fn,'r') as f:
        pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
        ds=f['samples_cold']; out={}
        idxs=[pn.index(w) for w in want if w in pn]
        lo,hi=min(idxs),max(idxs)+1
        if hi-lo<=12:
            blk=ds[0,:,lo:hi]
            for w in want:
                if w in pn: out[w]=np.asarray(blk[:,pn.index(w)-lo],np.float64)
            del blk
        else:
            for w in want:
                if w in pn: out[w]=np.asarray(ds[0,:,pn.index(w)],np.float64)
    log('read %s %s (%.0fs)'%(fn,list(out.keys()),time.time()-t0))
    return out

def dl_of(mc,h,logf):
    return 10**(CONST + (5/3)*mc + (2/3)*logf - h)

def boot(vals, pos, N, Lblk, stat, nboot=NBOOT):
    bid=pos//max(int(Lblk),1)
    o=np.argsort(bid,kind='stable'); bs=bid[o]; vs=vals[o]
    edges=np.searchsorted(bs,np.unique(bs)); groups=np.split(vs,edges[1:])
    base=stat(vals); res=np.empty(nboot)
    for k in range(nboot):
        pick=rng.integers(0,len(groups),len(groups))
        res[k]=stat(np.concatenate([groups[i] for i in pick]))
    return base, res.std(ddof=1), len(groups)

def q95(v): return np.quantile(v,Q)

OUT={}
def masked_stats(run, fn, broad, twall):
    tau=ESSJ[run]['tau_by_param']; taumax=ESSJ[run]['tau_max']
    want=['0_log10_mc','0_log10_h','gwb_log10_A']+(['0_log10_fgw'] if broad else [])
    C=read_cols(fn,want)
    N=len(C['0_log10_mc'])
    logf=C['0_log10_fgw'] if broad else np.log10(3.7e-9)
    dl=dl_of(C['0_log10_mc'],C['0_log10_h'],logf)
    idx=np.flatnonzero(np.abs(dl-TARGET)<0.01*TARGET); del dl
    g=np.diff(idx) if idx.size>1 else np.array([0])
    r={'Nsurv':int(idx.size),'gap_median':float(np.median(g)),'gap_mean':float(g.mean())}
    tmc=tau.get('0_log10_mc',taumax)
    for p,tt in (('0_log10_mc',tmc),('gwb_log10_A',tau.get('gwb_log10_A',taumax))):
        b,se,nb=boot(C[p][idx],idx,N,20*tt,np.mean)
        sd=C[p][idx].std(ddof=1)
        r['ESSmask_'+p]=float((sd/se)**2)
    ul,ulse,nb=boot(C['0_log10_mc'][idx],idx,N,20*tmc,q95)
    r['UL']=float(ul); r['ULerr']=float(ulse)
    r['ESS_full']=float(ESSJ[run]['ESS']); r['Twall']=twall
    r['ESSmask_worst']=float(min(r['ESSmask_0_log10_mc'],r['ESSmask_gwb_log10_A']))
    r['Rpost_mask']=r['ESSmask_worst']/twall
    OUT[run]=r; log('%s: %s'%(run,json.dumps(r)))
    return C,idx,N,tmc

try:
    log('=== START ===')
    # Run D + ntol sweep
    C,idx,N,tmcD=masked_stats('D','G2D1_narrow_UL_4core.h5',False,34947)
    mc=C['0_log10_mc']; h=C['0_log10_h']
    dl=dl_of(mc,h,np.log10(3.7e-9))
    NTOL=[0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.30,0.50]
    rows=[]
    for nt in NTOL:
        m=np.flatnonzero(np.abs(dl-TARGET)<nt*TARGET)
        ul,se,nb=boot(mc[m],m,N,20*tmcD,q95)
        rows.append((nt*100,int(m.size),float(ul),float(se)))
        log('ntol %.1f%%: n=%d UL=%.4f +/- %.4f (%d blocks)'%(nt*100,m.size,ul,se,nb))
    OUT['sweep']=rows
    del C,mc,h,dl
    # QuickCW-dL points (full chains, block bootstrap on contiguous blocks)
    def full_point(tag,fn,taumc):
        C=read_cols(fn,['0_log10_mc'])
        x=C['0_log10_mc']; n=len(x); pos=np.arange(n)
        ul,se,nb=boot(x,pos,n,20*taumc,q95)
        OUT[tag]={'N':n,'UL':float(ul),'ULerr':float(se)}
        log('%s: UL=%.4f +/- %.4f'%(tag,ul,se)); del C,x
    full_point('lokiM_1pc', ESSJ['M']['file'], ESSJ['M']['tau_by_param'].get('0_log10_mc',ESSJ['M']['tau_max']))
    try:
        full_point('loki_10pc','G2D1_fixed_UL_loki_100M_lastTOA_ntol_10_15_Jul_2026.h5', ESSJ['M']['tau_by_param'].get('0_log10_mc',ESSJ['M']['tau_max']))
    except Exception as e: log('10pc failed: %r'%e)
    # Enterprise
    with h5py.File(H5+'core_single_MDC2_DS1.h5','r') as f:
        pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['params'][:]]
        ch=f['chain'][...]
    eMC=np.asarray(ch[3000:,pn.index('log10_mc')],np.float64); del ch
    import emcee
    tauE=float(emcee.autocorr.integrated_time(eMC,quiet=True)[0])
    ul,se,nb=boot(eMC,np.arange(len(eMC)),len(eMC),20*tauE,q95)
    OUT['ent']={'N':len(eMC),'tau':tauE,'UL':float(ul),'ULerr':float(se)}
    log('ENT: tau=%.0f UL=%.4f +/- %.4f'%(tauE,ul,se))
    # Runs C, A, B masked stats
    masked_stats('C','G2D1_broad_UL_4core.h5',True,34860)
    masked_stats('A','G2D1_broad_detect_4core.h5',True,34612)
    masked_stats('B','G2D1_narrow_detect_4core.h5',False,34950)
    # E, F from masked files (fast)
    for run,fn,tw in (('E','G2D2_broad_detect_tref_4core_dLmasked_75.400Mpc.h5',34091),('F','G2D2_narrow_detect_tref_4core_dLmasked_75.400Mpc.h5',33744)):
        with h5py.File(H5+fn,'r') as f:
            pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
            idx=f['mask_indices'][:].astype(np.int64); S=f['samples_masked']
            vmc=np.asarray(S[:,pn.index('0_log10_mc')],np.float64)
            vgw=np.asarray(S[:,pn.index('gwb_log10_A')],np.float64)
        tau=ESSJ[run]['tau_by_param']; r={'Nsurv':int(idx.size)}
        for p,v in (('0_log10_mc',vmc),('gwb_log10_A',vgw)):
            b,se,nb=boot(v,idx,int(1e8),20*tau.get(p,ESSJ[run]['tau_max']),np.mean)
            r['ESSmask_'+p]=float((v.std(ddof=1)/se)**2)
        r['ESS_full']=float(ESSJ[run]['ESS']); r['Twall']=tw
        r['ESSmask_worst']=float(min(r['ESSmask_0_log10_mc'],r['ESSmask_gwb_log10_A']))
        r['Rpost_mask']=r['ESSmask_worst']/tw
        OUT[run]=r; log('%s: %s'%(run,json.dumps(r)))
    with open(PP+'final_paper_analysis.json','w') as f: json.dump(OUT,f,indent=1)
    # ---- figure: single panel, her style ----
    plt.rcParams.update({'font.size':9})
    fig,b=plt.subplots(1,1,figsize=(3.5,2.6))
    nt=[r[0] for r in OUT['sweep']]; ul=[r[2] for r in OUT['sweep']]; er=[r[3] for r in OUT['sweep']]
    b.errorbar(nt,ul,yerr=er,fmt='o-',color='#4477AA',ms=4,lw=1.2,capsize=2,label=r'{\sc QuickCW} + mask')
    E=OUT['ent']
    b.axhline(E['UL'],color='#228833',ls='--',lw=1.0)
    b.fill_between([0.4,60],[E['UL']-E['ULerr']]*2,[E['UL']+E['ULerr']]*2,color='#228833',alpha=0.18,lw=0)
    b.errorbar([1.0],[OUT['lokiM_1pc']['UL']],yerr=[OUT['lokiM_1pc']['ULerr']],fmt='D',color='#CC6677',ms=5,capsize=2,zorder=5)
    if 'loki_10pc' in OUT:
        b.errorbar([10.0],[OUT['loki_10pc']['UL']],yerr=[OUT['loki_10pc']['ULerr']],fmt='D',color='#66CCEE',ms=5,capsize=2,zorder=5)
    b.set_xscale('log'); b.set_xlim(0.4,60)
    b.set_xlabel(r'distance tolerance $\eta_{\rm tol}$ [%]')
    b.set_ylabel(r'95% UL on $\log_{10}\mathcal{M}_c$')
    fig.tight_layout()
    fig.savefig(PP+'ntol_sweep_4core.pdf'); fig.savefig(PP+'ntol_sweep_4core_new.png',dpi=200)
    log('figure saved')
    log('=== DONE ===')
except Exception:
    log('FATAL: '+traceback.format_exc())
