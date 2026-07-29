#!/usr/bin/env python3
import numpy as np, h5py, emcee, json
H5  = '/scratch/na00078/projects/IPTA_MDC2/h5_files/'
OUT = '/scratch/na00078/projects/IPTA_MDC2/post_processing/'
RUNS = {
 'A': ('G2D1_broad_detect_4core.h5',                       34612, 130506),
 'B': ('G2D1_narrow_detect_4core.h5',                      34950, 132314),
 'C': ('G2D1_broad_UL_4core.h5',                           34860, 13663),
 'D': ('G2D1_narrow_UL_4core.h5',                          34947, 7151),
 'E': ('G2D2_broad_detect_tref_4core.h5',                  34091, 192343),
 'F': ('G2D2_narrow_detect_tref_4core.h5',                 33744, 239483),
 'G': ('G2D2_detect_allsky_4core.h5',                      34094, None),
 'L': ('G2D2_broad_UL_loki_100M_lastTOA_4core.h5',         34701, None),
 'M': ('G2D2_fixed_UL_loki_100M_lastTOA_ntol_10_4core.h5', 35622, None),
 'N': ('G2D2_broad_detect_loki_100M_lastTOA_4core.h5',     34058, None),
 'O': ('G2D2_fixed_detect_loki_100M_lastTOA_4core.h5',     34305, None),
}
CHUNK = 2_000_000
res = {}
for r, (fn, tw, nmask) in RUNS.items():
    print('=' * 20, r, fn, flush=True)
    with h5py.File(H5 + fn, 'r') as h:
        pn = [p.decode() if isinstance(p, bytes) else p for p in h['par_names'][:]]
        cols = [i for i, n in enumerate(pn) if n.startswith('0_') or 'gwb' in n.lower()]
        names = [pn[i] for i in cols]
        print(r, 'params:', names, flush=True)
        d = h['samples_cold']; N = d.shape[1]
        arr = np.empty((N, len(cols)), dtype=np.float32)
        for a in range(0, N, CHUNK):
            b = min(a + CHUNK, N)
            arr[a:b] = d[0, a:b, cols]
    taus = {}
    for j, nm in enumerate(names):
        x = arr[::10, j].astype(np.float64)
        if np.all(x == x[0]):
            print(r, nm, 'constant, skipped', flush=True); continue
        tau = 10 * float(emcee.autocorr.integrated_time(x, quiet=True)[0])
        taus[nm] = tau
        print(r, nm, 'tau %.0f' % tau, flush=True)
    tmax = max(taus.values()); ess = N / tmax
    nus = min(ess, nmask) if nmask else ess
    res[r] = dict(file=fn, N=int(N), tau_max=tmax, tau_by_param=taus, ESS=ess,
                  Nmask=nmask, Nusable=nus, Twall=tw, Rpost=nus / tw)
    print('%s: tau_max %.0f  ESS %.0f  Nusable %.0f  Rpost %.2f per s' % (r, tmax, ess, nus, nus / tw), flush=True)
    json.dump(res, open(OUT + 'sec6_ess_4core.json', 'w'), indent=2)
print(json.dumps({k: dict(ESS=round(v['ESS']), Nusable=round(v['Nusable']), Rpost=round(v['Rpost'], 2)) for k, v in res.items()}, indent=2))
