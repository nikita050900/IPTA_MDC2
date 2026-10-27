#!/usr/bin/env python
# fig2_prior_check.py: does the chirp mass prior upper bound used in the production
# chains (m_max) change anything quoted in the paper? The UL runs use
# LinearExp(m_min, m_max) on log10 Mc, i.e. uniform in Mc, so the 95% quantile of the
# prior alone is log10(0.95 * 10**m_max). Checks Run D unmasked and every sweep
# tolerance, with and without truncating the chain to log10 Mc <= 10 (the bound the
# paper's prior table quotes). Truncation is exact rejection sampling: the prior shape
# below the bound is unchanged, so the conditioned chain is the bound-10 posterior.
import numpy as np, h5py, json, os
H5 = os.environ['PROJECT'] + '/projects/IPTA_MDC2/h5_files/'
PP = os.environ['PROJECT'] + '/projects/IPTA_MDC2/post_processing/'
mpc = 3.086e22; c = 299792458.0; Tsun = 1.327124400e20 / c**3
CONST = np.log10(2 * Tsun**(5/3) * np.pi**(2/3) * c / mpc)
Q = 0.95; TARGET = 75.4; CUT = 10.0
rng = np.random.default_rng(11)
ESSJ = json.load(open(PP + 'sec6_ess_4core.json'))
tau = ESSJ['D']['tau_by_param'].get('0_log10_mc', ESSJ['D']['tau_max'])

def boot(vals, pos, Lblk, stat, nboot=300):
    bid = pos // max(int(Lblk), 1)
    o = np.argsort(bid, kind='stable'); bs = bid[o]; vs = vals[o]
    edges = np.searchsorted(bs, np.unique(bs)); groups = np.split(vs, edges[1:])
    res = np.empty(nboot)
    for k in range(nboot):
        pick = rng.integers(0, len(groups), len(groups))
        res[k] = stat(np.concatenate([groups[i] for i in pick]))
    return stat(vals), res.std(ddof=1)

with h5py.File(H5 + 'G2D1_narrow_UL_4core_UNMASKED_outfile.h5', 'r') as f:
    pn = [p.decode() if isinstance(p, bytes) else str(p) for p in f['par_names'][:]]
    ds = f['samples_cold']
    mc = ds[0, :, pn.index('0_log10_mc')].astype(np.float64)
    h = ds[0, :, pn.index('0_log10_h')].astype(np.float64)
N = len(mc)
print('Run D unmasked: N=%d  log10Mc min=%.4f max=%.4f  frac above 10 = %.4f'
      % (N, mc.min(), mc.max(), (mc > CUT).mean()), flush=True)
print('LinearExp prediction for the 95%% quantile of the prior alone:'
      '  m_max=11 -> %.4f   m_max=10 -> %.4f' % (np.log10(0.95e11), np.log10(0.95e10)), flush=True)
sel = mc <= CUT
u_full, e_full = boot(mc, np.arange(N), 20 * tau, lambda v: np.quantile(v, Q))
pos = np.flatnonzero(sel)
u_cut, e_cut = boot(mc[sel], pos, 20 * tau, lambda v: np.quantile(v, Q))
print('Run D unmasked 95%% UL: as run (m_max=11) %.4f +/- %.4f | truncated to <=10 %.4f +/- %.4f (N=%d)'
      % (u_full, e_full, u_cut, e_cut, sel.sum()), flush=True)

dl = 10**(CONST + (5/3)*mc + (2/3)*np.log10(3.7e-9) - h)
print('\nsweep: eta_tol  N_surv  frac(Mc>10)   UL as run   UL truncated   shift', flush=True)
out = {}
for nt in (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50):
    m = np.abs(dl - TARGET) < nt * TARGET
    v = mc[m]; frac = float((v > CUT).mean())
    q = float(np.quantile(v, Q))
    vt = v[v <= CUT]
    qt = float(np.quantile(vt, Q)) if vt.size else float('nan')
    out['%.3f' % nt] = dict(N=int(m.sum()), frac_above_10=frac, UL=q, UL_cut=qt)
    print('  %5.1f%%  %8d  %10.2e  %9.4f  %12.4f  %7.4f' % (nt*100, m.sum(), frac, q, qt, qt - q), flush=True)
out['unmasked'] = dict(UL_as_run=float(u_full), UL_as_run_err=float(e_full),
                       UL_cut=float(u_cut), UL_cut_err=float(e_cut),
                       N=int(N), N_cut=int(sel.sum()), frac_above_10=float((mc > CUT).mean()))
json.dump(out, open(PP + 'fig2_prior_check.json', 'w'), indent=1)

del mc, h, dl
print('\ndetection runs E and F, log10 Mc max on every 100th cold sample:', flush=True)
for tag, fn in (('E broad', 'G2D2_broad_detect_tref_4core_UNMASKED_outfile.h5'),
                ('F fixed', 'G2D2_narrow_detect_tref_4core_UNMASKED_outfile.h5')):
    with h5py.File(H5 + fn, 'r') as f:
        d = f['samples_cold']
        x = d[0, ::100, 5].astype(np.float64)
    print('  %s: min %.4f max %.4f' % (tag, x.min(), x.max()), flush=True)
print('\nmasked files, log10 Mc max:', flush=True)
for tag, fn in (('A', 'G2D1_broad_detect_4core_dLmasked_75.400Mpc.h5'),
                ('B', 'G2D1_narrow_detect_4core_dLmasked_75.400Mpc.h5'),
                ('C', 'G2D1_broad_UL_4core_dLmasked_75.400Mpc.h5'),
                ('D', 'G2D1_narrow_UL_4core_dLmasked_75.400Mpc.h5'),
                ('E', 'G2D2_broad_detect_tref_4core_dLmasked_75.400Mpc.h5'),
                ('F', 'G2D2_narrow_detect_tref_4core_dLmasked_75.400Mpc.h5')):
    try:
        with h5py.File(H5 + 'dl_masked/' + fn, 'r') as f:
            p = [q.decode() if isinstance(q, bytes) else str(q) for q in f['par_names'][:]]
            v = f['samples_masked'][:, p.index('0_log10_mc')]
        print('  Run %s: N=%d max %.4f  frac above 10 = %.3e' % (tag, len(v), v.max(), (v > CUT).mean()), flush=True)
    except Exception as e:
        print('  Run %s: %r' % (tag, e), flush=True)
