#!/usr/bin/env python
# fig2_nomask_final.py: final Figure 2. Sweep panel byte-identical in style to the
# Overleaf figure written by final_paper_analysis.py (figsize 3.5 x 2.6, font 9, log x,
# xlim 0.4 to 60, default log ticks, same colours, symbols, estimators).
# Adds one black dashed horizontal line: the 95% limit from the same Run D chain with
# no dL mask applied. It is computed on the chain conditioned to log10 Mc <= 10, the
# prior bound quoted in the paper's prior table, because the production chain was run
# with the bound at 11. Conditioning is exact rejection sampling: the UL prior is
# uniform in Mc, so its shape below 10 is unchanged and the likelihood is untouched.
# Drawn as a line, not a point, because the no-mask limit does not depend on eta_tol.
# No text label on the figure; the caption explains the line.
import os, json
import numpy as np, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

H5 = os.environ['PROJECT'] + '/projects/IPTA_MDC2/h5_files/'
PP = os.environ['PROJECT'] + '/projects/IPTA_MDC2/post_processing/'
OUT = PP + 'fig_24sep/'
os.makedirs(OUT, exist_ok=True)
Q = 0.95; CUT = 10.0; NBOOT = 300
rng = np.random.default_rng(11)
R = json.load(open(PP + 'final_paper_analysis.json'))
ESSJ = json.load(open(PP + 'sec6_ess_4core.json'))
tau = ESSJ['D']['tau_by_param'].get('0_log10_mc', ESSJ['D']['tau_max'])

def boot(vals, pos, Lblk, stat, nboot=NBOOT):
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
    mc = f['samples_cold'][0, :, pn.index('0_log10_mc')].astype(np.float64)
sel = mc <= CUT
pos = np.flatnonzero(sel)
ul_cut, err_cut = boot(mc[sel], pos, 20 * tau, lambda v: np.quantile(v, Q))
info = dict(UL=float(ul_cut), ULerr=float(err_cut), N_total=int(len(mc)), N_used=int(sel.sum()),
            prior_bound_used=CUT, prior_bound_in_chain=float(mc.max()),
            note='95% UL with no dL mask, chain conditioned to log10 Mc <= 10 (paper prior bound)')
json.dump(info, open(PP + 'fig2_nomask_point.json', 'w'), indent=1)
print('no-mask 95%% UL (bound 10) = %.4f +/- %.4f, from %d of %d samples'
      % (ul_cut, err_cut, sel.sum(), len(mc)), flush=True)
del mc, sel, pos

C_QCW, C_ENT, C_M, C_C10 = '#4477AA', '#228833', '#CC6677', '#66CCEE'
nt = [r[0] for r in R['sweep']]; ul = [r[2] for r in R['sweep']]; er = [r[3] for r in R['sweep']]
E = R['ent']; M = R['lokiM_1pc']; C10 = R.get('loki_10pc')
plt.rcParams.update({'font.size': 9})
fig, b = plt.subplots(1, 1, figsize=(3.5, 2.6))
b.errorbar(nt, ul, yerr=er, fmt='o-', color=C_QCW, ms=4, lw=1.2, capsize=2)
b.axhline(E['UL'], color=C_ENT, ls='--', lw=1.0)
b.fill_between([0.4, 60], [E['UL'] - E['ULerr']] * 2, [E['UL'] + E['ULerr']] * 2,
               color=C_ENT, alpha=0.18, lw=0)
b.errorbar([1.0], [M['UL']], yerr=[M['ULerr']], fmt='D', color=C_M, ms=5, capsize=2, zorder=5)
if C10:
    b.errorbar([10.0], [C10['UL']], yerr=[C10['ULerr']], fmt='D', color=C_C10, ms=5, capsize=2, zorder=5)
b.axhline(ul_cut, color='k', ls='--', lw=1.2, zorder=4)
b.set_xscale('log'); b.set_xlim(0.4, 60)
b.set_ylim(9.68, 10.00)
b.set_xlabel(r'distance tolerance $\eta_{\rm tol}$ [%]')
b.set_ylabel(r'95% UL on $\log_{10}\mathcal{M}_c$')
fig.tight_layout()
fig.savefig(OUT + 'ntol_sweep_4core.pdf', bbox_inches='tight')
fig.savefig(OUT + 'ntol_sweep_4core.png', dpi=200, bbox_inches='tight')
print('wrote', OUT + 'ntol_sweep_4core.pdf/png', flush=True)

print('\ndetection runs E and F, log10 Mc range on every 100th cold sample:', flush=True)
for tag, fn in (('E broad', 'G2D2_broad_detect_tref_4core_UNMASKED_outfile.h5'),
                ('F fixed', 'G2D2_narrow_detect_tref_4core_UNMASKED_outfile.h5')):
    with h5py.File(H5 + fn, 'r') as f:
        x = f['samples_cold'][0, ::100, 5].astype(np.float64)
    print('  %s: min %.4f max %.4f' % (tag, x.min(), x.max()), flush=True)
