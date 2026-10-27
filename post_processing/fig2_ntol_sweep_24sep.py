#!/usr/bin/env python
# fig2_ntol_sweep_24sep.py: Figure 2 (ntol sweep, Run D) rebuilt from
# final_paper_analysis.json (sweep, Enterprise, QuickGWecc points) plus the
# the fully distance-unconstrained (no mask) Run D limit from fig2_unmasked_point.json
# (Sarah, 10 Sep: show the 100% point). Larger fonts (Sarah, 8 Sep). Same estimators
# and colours as before. The no-mask point is drawn at x = 100 as a black square,
# without an error bar: its block bootstrap error (0.0007) is smaller than the
# symbol and the value sits at the Mc prior edge, so the bar would be misleading.
# Usage: python fig2_ntol_sweep_24sep.py [post_processing_dir] [out_dir]
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

PP = (sys.argv[1] if len(sys.argv) > 1 else '.').rstrip('/') + '/'
OUT = (sys.argv[2] if len(sys.argv) > 2 else PP + 'fig_24sep').rstrip('/') + '/'
os.makedirs(OUT, exist_ok=True)
with open(PP + 'final_paper_analysis.json') as f:
    R = json.load(f)
with open(PP + 'fig2_unmasked_point.json') as f:
    U = json.load(f)                    # no dL mask at all, block bootstrap on the full chain

nt = np.array([r[0] for r in R['sweep']])
ul = np.array([r[2] for r in R['sweep']])
er = np.array([r[3] for r in R['sweep']])
E = R['ent']; M = R['lokiM_1pc']; C10 = R.get('loki_10pc')
C_QCW, C_ENT, C_M, C_C10 = '#4477AA', '#228833', '#CC6677', '#66CCEE'

FS = 12
plt.rcParams.update({'font.size': FS, 'axes.labelsize': FS + 1, 'xtick.labelsize': FS,
                     'ytick.labelsize': FS, 'axes.linewidth': 0.9,
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True})
fig, ax = plt.subplots(1, 1, figsize=(4.9, 4.3))
XMIN, XMAX = 0.35, 160.0
ax.axhline(E['UL'], color=C_ENT, ls='--', lw=1.3, zorder=2)
ax.fill_between([XMIN, XMAX], [E['UL'] - E['ULerr']] * 2, [E['UL'] + E['ULerr']] * 2,
                color=C_ENT, alpha=0.18, lw=0, zorder=1)
ax.errorbar(nt, ul, yerr=er, fmt='o-', color=C_QCW, ms=5.5, lw=1.5, capsize=3, zorder=3)
ax.errorbar([1.0], [M['UL']], yerr=[M['ULerr']], fmt='D', color=C_M, ms=7, capsize=3, zorder=6)
if C10:
    ax.errorbar([10.0], [C10['UL']], yerr=[C10['ULerr']], fmt='D', color=C_C10, ms=7, capsize=3, zorder=6)
ax.plot([100.0], [U['UL']], 's', color='k', ms=7, zorder=7)
ax.set_xscale('log'); ax.set_xlim(XMIN, XMAX)
ax.set_ylim(9.68, 11.05)
ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100]))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: '%g' % v))
ax.set_xlabel(r'distance tolerance $\eta_{\rm tol}$ [%]')
ax.set_ylabel(r'95% UL on $\log_{10}\mathcal{M}_c$')
fig.tight_layout()
fig.savefig(OUT + 'ntol_sweep_4core.pdf', bbox_inches='tight')
fig.savefig(OUT + 'ntol_sweep_4core.png', dpi=200, bbox_inches='tight')
print('no-mask Run D: UL=%.4f +/- %.4f (N=%d, %d blocks); saved %sntol_sweep_4core.pdf/png'
      % (U['UL'], U['ULerr'], U['N'], U['blocks'], OUT))
