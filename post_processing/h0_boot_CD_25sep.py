#!/usr/bin/env python
# h0_boot_CD_25sep.py
# Table 5: the masked QuickCW log10 h0 errors currently come from Equation 5
# with a capped ESS while the Mc errors in the same row come from the block
# bootstrap. This recomputes the h0 upper limit errors with the same block
# bootstrap, on Runs C (broad) and D (fixed), and cross checks Mc against
# the published values.
import numpy as np, h5py, json, time, traceback

H5='/anvil/projects/x-phy260442/projects/IPTA_MDC2/h5_files/'
PP='/anvil/projects/x-phy260442/projects/IPTA_MDC2/post_processing/'
LOG=PP+'h0_boot_CD_25sep.log'
def log(s):
    with open(LOG,'a') as f: f.write(time.strftime('%H:%M:%S ')+str(s)+'\n')
    print(s, flush=True)

Q=0.95; TARGET=75.4; NBOOT=300
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*(Tsun)**(5/3)*np.pi**(2/3)*c/mpc)
rng=np.random.default_rng(11)

with open(PP+'sec6_ess_4core.json') as f: ESSJ=json.load(f)

def dl_of(mc,h,logf): return 10**(CONST+(5/3)*mc+(2/3)*logf-h)
def q95(v): return np.quantile(v,Q)

def boot(vals,pos,Lblk,stat,nboot=NBOOT):
    bid=pos//max(int(Lblk),1)
    o=np.argsort(bid,kind='stable'); bs=bid[o]; vs=vals[o]
    edges=np.searchsorted(bs,np.unique(bs)); groups=np.split(vs,edges[1:])
    base=stat(vals); res=np.empty(nboot)
    for k in range(nboot):
        pick=rng.integers(0,len(groups),len(groups))
        res[k]=stat(np.concatenate([groups[i] for i in pick]))
    return float(base), float(res.std(ddof=1)), len(groups)

OUT={}
try:
    log('=== START ===')
    for run,broad in (('D',False),('C',True)):
        fn=ESSJ[run]['file']; tau=ESSJ[run]['tau_by_param']; taumax=ESSJ[run]['tau_max']
        want=['0_log10_mc','0_log10_h']+(['0_log10_fgw'] if broad else [])
        log('=== run %s broad=%s file %s ==='%(run,broad,fn))
        C={}
        with h5py.File(H5+fn,'r') as f:
            pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
            ds=f['samples_cold']
            for w in want: C[w]=np.asarray(ds[0,:,pn.index(w)],np.float64)
        N=len(C['0_log10_mc'])
        logf=C['0_log10_fgw'] if broad else np.log10(3.7e-9)
        dl=dl_of(C['0_log10_mc'],C['0_log10_h'],logf)
        idx=np.flatnonzero(np.abs(dl-TARGET)<0.01*TARGET); del dl
        th=float(tau.get('0_log10_h',taumax)); tmc=float(tau.get('0_log10_mc',taumax))
        r={'file':fn,'N':int(N),'Nsurv':int(idx.size),'tau_h':th,'tau_mc':tmc}
        log('%s Nsurv=%d tau_h=%.1f tau_mc=%.1f'%(run,idx.size,th,tmc))
        for label,L in (('blk_20tau_h',20*th),('blk_20tau_mc',20*tmc)):
            ul,se,nb=boot(C['0_log10_h'][idx],idx,L,q95)
            r[label]={'UL':ul,'ULerr':se,'nblocks':nb,'Lblk':float(L)}
            log('%s h0 %s: UL=%.4f +/- %.4f (%d blocks, Lblk=%.0f)'%(run,label,ul,se,nb,L))
        ulm,sem,nbm=boot(C['0_log10_mc'][idx],idx,20*tmc,q95)
        r['mc_cross_check']={'UL':ulm,'ULerr':sem,'nblocks':nbm}
        log('%s Mc cross check: UL=%.4f +/- %.4f (paper: D 9.793+/-0.081, C 9.752+/-0.011)'%(run,ulm,sem))
        OUT[run]=r
        del C
    with open(PP+'h0_boot_CD_25sep.json','w') as f: json.dump(OUT,f,indent=1)
    log('=== DONE ===')
except Exception:
    log('FATAL: '+traceback.format_exc())
