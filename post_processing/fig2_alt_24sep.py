#!/usr/bin/env python
# fig2_alt_24sep.py: two further layouts for Figure 2, both keeping the sweep panel
# byte-identical in style to the Overleaf figure (3.5 x 2.6 per panel, font 9,
# log x, xlim 0.4 to 60, same colours and symbols).
#   ntol_sweep_4core_inset2.pdf  inset tidied: smaller, legend out of the way
#   ntol_sweep_4core_2panel.pdf  (a) the sweep at its original y range, untouched,
#                                (b) cumulative log10 Mc with and without the mask
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PP = (sys.argv[1] if len(sys.argv) > 1 else '.').rstrip('/') + '/'
OUT = (sys.argv[2] if len(sys.argv) > 2 else PP + 'fig_24sep').rstrip('/') + '/'
os.makedirs(OUT, exist_ok=True)
R = json.load(open(PP + 'final_paper_analysis.json'))
H = json.load(open(PP + 'fig2_mc_hist.json'))
C_QCW, C_ENT, C_M, C_C10 = '#4477AA', '#228833', '#CC6677', '#66CCEE'
nt = [r[0] for r in R['sweep']]; ul = [r[2] for r in R['sweep']]; er = [r[3] for r in R['sweep']]
E = R['ent']; M = R['lokiM_1pc']; C10 = R.get('loki_10pc')
edges = np.array(H['edges']); ctr = 0.5 * (edges[1:] + edges[:-1]); w = edges[1] - edges[0]

def sweep(b):
    b.errorbar(nt, ul, yerr=er, fmt='o-', color=C_QCW, ms=4, lw=1.2, capsize=2)
    b.axhline(E['UL'], color=C_ENT, ls='--', lw=1.0)
    b.fill_between([0.4, 60], [E['UL'] - E['ULerr']] * 2, [E['UL'] + E['ULerr']] * 2,
                   color=C_ENT, alpha=0.18, lw=0)
    b.errorbar([1.0], [M['UL']], yerr=[M['ULerr']], fmt='D', color=C_M, ms=5, capsize=2, zorder=5)
    if C10:
        b.errorbar([10.0], [C10['UL']], yerr=[C10['ULerr']], fmt='D', color=C_C10, ms=5, capsize=2, zorder=5)
    b.set_xscale('log'); b.set_xlim(0.4, 60)
    b.set_xlabel(r'distance tolerance $\eta_{\rm tol}$ [%]')
    b.set_ylabel(r'95% UL on $\log_{10}\mathcal{M}_c$')

def cdf(key):
    c = np.cumsum(np.array(H[key]) * w); return c / c[-1]

plt.rcParams.update({'font.size': 9})

# --- inset, tidied ---
fig, b = plt.subplots(1, 1, figsize=(3.5, 2.6))
sweep(b)
lo, hi = b.get_ylim(); b.set_ylim(lo, lo + (hi - lo) * 1.38)
ins = b.inset_axes([0.505, 0.615, 0.455, 0.335])
ins.plot(ctr, cdf('masked_1pc'), color=C_QCW, lw=1.1)
ins.plot(ctr, cdf('unmasked'), color='k', lw=1.1)
ins.axhline(0.95, color='0.45', ls=':', lw=0.8)
ins.set_xlim(7.0, 11.0); ins.set_ylim(0, 1.08)
ins.set_xticks([7, 9, 11]); ins.set_yticks([0, 0.95]); ins.set_yticklabels(['0', '.95'])
ins.tick_params(labelsize=6, length=2, pad=1.2)
ins.set_xlabel(r'$\log_{10}\mathcal{M}_c$', fontsize=6.5, labelpad=0.5)
ins.text(0.06, 0.80, r'$1\%$', color=C_QCW, fontsize=6.2, transform=ins.transAxes)
ins.text(0.62, 0.16, 'no mask', color='k', fontsize=6.2, transform=ins.transAxes)
fig.tight_layout()
fig.savefig(OUT + 'ntol_sweep_4core_inset2.pdf', bbox_inches='tight')
fig.savefig(OUT + 'ntol_sweep_4core_inset2.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# --- two panels side by side ---
fig, (a, c) = plt.subplots(1, 2, figsize=(7.0, 2.6))
sweep(a)
a.text(0.04, 0.93, '(a)', transform=a.transAxes, va='top')
c.plot(ctr, cdf('masked_1pc'), color=C_QCW, lw=1.3, label=r'$\eta_{\rm tol}=1\%$')
c.plot(ctr, cdf('unmasked'), color='k', lw=1.3, label='no mask')
c.axhline(0.95, color='0.45', ls=':', lw=1.0)
c.set_xlim(7.0, 11.0); c.set_ylim(0, 1.03)
c.set_yticks([0, 0.25, 0.5, 0.75, 0.95])
c.set_xlabel(r'$\log_{10}\mathcal{M}_c$')
c.set_ylabel('cumulative posterior')
c.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.82), fontsize=8)
c.text(0.04, 0.93, '(b)', transform=c.transAxes, va='top')
fig.tight_layout()
fig.savefig(OUT + 'ntol_sweep_4core_2panel.pdf', bbox_inches='tight')
fig.savefig(OUT + 'ntol_sweep_4core_2panel.png', dpi=200, bbox_inches='tight')
print('wrote ntol_sweep_4core_inset2 and ntol_sweep_4core_2panel to', OUT)
