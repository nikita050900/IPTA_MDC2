import numpy as np, h5py, emcee
H5='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
RUNS=[('A','G2D1_broad_detect_1e9_fixed_gamma_outfile.h5',53520,143939),
 ('B','G2D1_narrow_detect_fixed_gamma_17_dec_2025_outfile.h5',53438,133668),
 ('C','G2D1_broad_UL_1e9_fixed_gamma_outfile.h5',53427,18582),
 ('D','G2D1_narrow_UL_fixed_gamma_17_dec_2025_outfile.h5',54609,7782),
 ('E','G2D2_broad_detect_tref_09_Jun_2026.h5',47940,204278),
 ('F','G2D2_narrow_detect_tref_09_Jun_2026.h5',47576,239526)]
PARS=['0_log10_mc','0_cos_inc','0_phase0','0_psi','0_log10_fgw','0_log10_h','gwb_log10_A','gwb_gamma']
BLK=2_000_000; TARGET=2_000_000
for run,fn,tw,nmask in RUNS:
    with h5py.File(H5+fn,'r') as f:
        pn=[p.decode() if isinstance(p,bytes) else p for p in f['par_names'][:]]
        d=f['samples_cold']
        N=d.shape[1]
        cols=[(p,pn.index(p)) for p in PARS if p in pn]
        K=max(1,N//TARGET)
        idx=[c for _,c in cols]
        kept={p:[] for p,_ in cols}
        for i0 in range(0,N,BLK):
            b=d[0,i0:min(i0+BLK,N),:][:, idx]
            for j,(p,_) in enumerate(cols):
                kept[p].append(b[::K,j])
        print('Run %s: N_stored=%d thin_extra=%d'%(run,N,K), flush=True)
        taus={}
        for p,_ in cols:
            x=np.concatenate(kept[p])
            try:
                t=K*float(emcee.autocorr.integrated_time(x,quiet=True)[0])
            except Exception as e:
                t=float('nan')
            taus[p]=t
            print('   %-14s tau_stored=%10.1f ESS=%12.0f'%(p,t,N/t if t==t and t>0 else -1), flush=True)
        worst=max([t for t in taus.values() if t==t])
        ess=N/worst
        print('Run %s SUMMARY: tau_max=%.0f ESS=%.0f N_mask=%d N_usable=min=%d R_post=%.3f /s'%(run,worst,ess,nmask,min(ess,nmask),min(ess,nmask)/tw), flush=True)
