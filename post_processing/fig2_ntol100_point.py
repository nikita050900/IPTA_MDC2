# literal ntol = 100% mask point (dL within [0, 2 x 75.4] Mpc) with the same block bootstrap as the sweep
import numpy as np, h5py, os, json
H5=os.environ['PROJECT']+'/projects/IPTA_MDC2/h5_files/'; PP=os.environ['PROJECT']+'/projects/IPTA_MDC2/post_processing/'
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*(Tsun)**(5/3)*np.pi**(2/3)*c/mpc); Q=0.95; NBOOT=300; TARGET=75.4
rng=np.random.default_rng(11)
ESSJ=json.load(open(PP+'sec6_ess_4core.json')); tau=ESSJ['D']['tau_by_param'].get('0_log10_mc',ESSJ['D']['tau_max'])
def boot(vals,pos,Lblk,stat,nboot=NBOOT):
    bid=pos//max(int(Lblk),1); o=np.argsort(bid,kind='stable'); bs=bid[o]; vs=vals[o]
    edges=np.searchsorted(bs,np.unique(bs)); groups=np.split(vs,edges[1:]); base=stat(vals); res=np.empty(nboot)
    for k in range(nboot):
        pick=rng.integers(0,len(groups),len(groups)); res[k]=stat(np.concatenate([groups[i] for i in pick]))
    return base,res.std(ddof=1),len(groups)
with h5py.File(H5+'G2D1_narrow_UL_4core_UNMASKED_outfile.h5','r') as f:
    pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
    ds=f['samples_cold']; mc=ds[0,:,pn.index('0_log10_mc')].astype(np.float64); h=ds[0,:,pn.index('0_log10_h')].astype(np.float64)
N=len(mc); dl=10**(CONST+(5/3)*mc+(2/3)*np.log10(3.7e-9)-h); del h
out={}
for nt in (0.5,1.0):
    m=np.flatnonzero(np.abs(dl-TARGET)<nt*TARGET)
    ul,se,nb=boot(mc[m],m,20*tau,lambda v: np.quantile(v,Q))
    out['ntol_%d'%int(nt*100)]={'ntol_pc':nt*100,'Nsurv':int(m.size),'UL':float(ul),'ULerr':float(se),'blocks':int(nb)}
    print(out['ntol_%d'%int(nt*100)],flush=True)
json.dump(out,open(PP+'fig2_ntol100_point.json','w'),indent=1)
