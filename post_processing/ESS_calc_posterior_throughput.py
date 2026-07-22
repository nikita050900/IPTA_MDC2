#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ESS via integrated autocorrelation time, per parameter, cold chain in
# sequential order. Quote min-ESS (max tau) as N_usable. Also prints the
# per-parameter breakdown (the formal ESS report that was still pending).
import numpy as np
import h5py
import emcee

ENT_BURN = 3000
THIN = 10          # tau estimated on thinned chain, multiplied back
H5 = '/scratch/na00078/projects/IPTA_MDC2/h5_files/'

RUNS = [  # (run, kind, file, T_wall [s])
 ('H','ent',  H5+'core_single_MDC2_DS1_varyfgw.h5',                    16656),
 ('I','ent',  H5+'core_single_MDC2_DS1_new.h5',                        19143),
 ('J','ent',  H5+'G2D2_varyfgw_core_new.h5',                           10679),
 ('K','ent',  H5+'G2D2_core_fixed_new.h5',                             11784),
 ('L','loki', H5+'G2D1_broad_UL_loki_100M_lastTOA_10_Jul_2026.h5',    231361),
 ('M','loki', H5+'G2D1_fixed_UL_loki_100M_lastTOA_10_Jul_2026.h5',    228908),
 ('N','loki', H5+'G2D2_broad_detect_loki_100M_lastTOA_02_Jun_2026.h5',233341),
 ('O','loki', H5+'G2D2_fixed_detect_loki_100M_lastTOA_02_Jun_2026.h5',231662),
]
P_ENT  = ['log10_mc','cos_inc','phase0','psi','log10_fgw','crn_log10_A','crn_gamma']
P_LOKI = ['0_log10_mc','0_cos_inc','0_phase0','0_psi','0_log10_fgw',
          '0_log10_dist','gwb_log10_A','gwb_gamma']

def tau_of(x):
    thin = THIN if len(x) > 5_000_000 else 1
    try:
        return thin * float(emcee.autocorr.integrated_time(x[::thin], quiet=True)[0])
    except Exception:
        return np.nan

for run, kind, path, twall in RUNS:
    cols = {}
    with h5py.File(path, 'r') as f:
        if kind == 'ent':
            pn = [p.decode() for p in f['params'][:]]
            ch = f['chain'][...][ENT_BURN:]
            for p in P_ENT:
                if p in pn: cols[p] = ch[:, pn.index(p)]
        else:
            pn = [p.decode() for p in f['par_names'][:]]
            for p in P_LOKI:
                if p in pn: cols[p] = f['samples_cold'][0, :, pn.index(p)]
    N = len(next(iter(cols.values())))
    taus = {p: tau_of(x) for p, x in cols.items()}
    tmax = np.nanmax(list(taus.values()))
    worst = [p for p, t in taus.items() if t == tmax][0]
    print(f"Run {run}: N={N:,}  tau_max={tmax:,.0f} ({worst})  "
          f"ESS={N/tmax:,.0f}  R_post={N/tmax/twall:.3f} /s")
    for p, t in sorted(taus.items(), key=lambda kv: -kv[1]):
        print(f"    {p:14s} tau={t:>10,.0f}  ESS={N/t:>10,.0f}")


# In[ ]:




