import numpy as np, h5py, emcee, time
OUT = 'scratch/projects/IPTA_MDC2/post_processing/bence_ess_check2_output.txt'
L=[]
def log(s):
    L.append(str(s))
    with open(OUT,'w') as f: f.write('\n'.join(L)+'\n')

t0=time.time()
path='/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D1_narrow_UL_4core.h5'
with h5py.File(path,'r') as f:
    pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
    blk = f['samples_cold'][0,:,0:9]           # one chunk-column group: all CW params
    gwbA = f['samples_cold'][0,:,141]
log('read cols 0-8 + gwb_log10_A (%.0f s)'%(time.time()-t0))
names = pn[0:9]
log('cols: '+str(names))
mc  = np.asarray(blk[:, names.index('0_log10_mc')], np.float64)
h   = np.asarray(blk[:, names.index('0_log10_h')],  np.float64)
fg  = np.asarray(blk[:, names.index('0_log10_fgw')],np.float64)
gwbA= np.asarray(gwbA, np.float64)
del blk
N = mc.size

c=299792458.0; Mpc=3.086e22; T_sun=1.327124400e20/c**3
dhat=75.4
def mask_idx(eta):
    dL = 2*c*(10**mc*T_sun)**(5/3)*(np.pi*10**fg)**(2/3)/(10**h)/Mpc
    return np.flatnonzero((dL>=dhat*(1-eta))&(dL<=dhat*(1+eta)))

idx = mask_idx(0.01)
log('=== 1% mask: %d survivors ==='%idx.size)
g=np.diff(idx)
log('gaps mean=%.0f median=%.0f  p10=%.0f p90=%.0f'%(g.mean(),np.median(g),np.percentile(g,10),np.percentile(g,90)))
log('fraction of consecutive-survivor gaps < tau_mc(9385): %.3f'%np.mean(g<9385))
log('fraction < 100: %.3f   clumps (gap>9385)+1 = %d'%(np.mean(g<100), int(np.sum(g>9385))+1))

# ---- block bootstrap ESS, chain-time aware ----
rng=np.random.default_rng(42)
def block_ess(vals, positions, stat, Lblk, nboot=400):
    """vals/positions: survivor values and chain indices. Returns (stat, SE_block, ESS_eff)."""
    nb = int(np.ceil(N/Lblk))
    bid = positions//Lblk
    groups=[vals[bid==b] for b in range(nb)]
    groups=[gg for gg in groups if gg.size>0]
    base=stat(vals)
    boots=np.empty(nboot)
    for k in range(nboot):
        pick=rng.integers(0,len(groups),len(groups))
        boots[k]=stat(np.concatenate([groups[i] for i in pick]))
    se=boots.std(ddof=1)
    sd=vals.std(ddof=1)
    return base, se, (sd/se)**2, len(groups)

for label, arr, tau_ref in (('log10_mc', mc[idx], 9385), ('gwb_log10_A', gwbA[idx], 45501)):
    for mult in (10, 20):
        Lblk = mult*tau_ref
        for sname, sfun in (('mean', np.mean), ('q95', lambda v: np.percentile(v,95))):
            b,se,ess,nb = block_ess(arr, idx, sfun, Lblk)
            log('%-12s %-4s Lblk=%8d (%4d blocks)  %s=%.4f  SE=%.5f  ESS_eff=%7.0f  (iid SE would be %.5f)'
                %(label,sname,Lblk,nb,sname,b,se,ess,arr.std(ddof=1)/np.sqrt(arr.size)))

# naive reindexed estimate for comparison
for label, arr in (('log10_mc', mc[idx]), ('gwb_log10_A', gwbA[idx])):
    try:
        ts=float(emcee.autocorr.integrated_time(arr,quiet=True)[0])
        log('naive reindexed %-12s tau_sub=%.1f -> ESS=%.0f'%(label,ts,arr.size/ts))
    except Exception as e:
        log('naive %s failed %s'%(label,e))

# does the mask change the slow-parameter posterior?
log('gwb_log10_A  full chain mean=%.4f sd=%.4f | masked mean=%.4f sd=%.4f'
    %(gwbA.mean(),gwbA.std(),gwbA[idx].mean(),gwbA[idx].std()))
log('DONE (%.0f s)'%(time.time()-t0))
