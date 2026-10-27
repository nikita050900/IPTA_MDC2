#!/usr/bin/env python
# fig2_brk_24sep.py: Figure 2 with the unmasked (no dL mask) 95% limit shown alongside
# the tolerance sweep. The sweep panel is unchanged from the Overleaf figure written by
# final_paper_analysis.py: font 9, log x, xlim 0.4 to 60, default log ticks, same
# colours, symbols and estimators.
# The no-mask limit does not depend on eta_tol, so it is drawn as a horizontal black
# dashed line (the same visual grammar as the green Enterprise line), never as a point
# at a particular tolerance.
#   ntol_sweep_4core_brk.pdf   broken y axis: thin upper strip holds the no-mask line,
#                              the sweep panel keeps its original y range
#   ntol_sweep_4core_full.pdf  single panel stretched to 11.05, no break
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PP = (sys.argv[1] if len(sys.argv) > 1 else '.').rstrip('/') + '/'
OUT = (sys.argv[2] if len(sys.argv) > 2 else PP + 'fig_24sep').rstrip('/') + '/'
os.makedirs(OUT, exist_ok=True)
R = json.load(open(PP + 'final_paper_analysis.json'))
U = json.load(open(PP + 'fig2_unmasked_point.json'))
C_QCW, C_ENT, C_M, C_C10 = '#4477AA', '#228833', '#CC6677', '#66CCEE'
nt = [r[0] for r in R['sweep']]; ul = [r[2] for r in R['sweep']]; er = [r[3] for r in R['sweep']]
E = R['ent']; M = R['lokiM_1pc']; C10 = R.get('loki_10pc')
YL = r'95% UL on $\log_{10}\mathcal{M}_c$'

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

plt.rcParams.update({'font.size': 9})

# ---- broken y axis ----
HR = 5.5
fig, (t, b) = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True,
                           gridspec_kw={'height_ratios': [1, HR], 'hspace': 0.09})
sweep(b)
t.axhline(U['UL'], color='k', ls='--', lw=1.2)
t.set_ylim(10.92, 11.04); t.set_yticks([11.0])
t.text(0.022, 0.60, 'no mask', transform=t.transAxes, fontsize=8, va='bottom')
t.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
t.spines['bottom'].set_visible(False)
b.spines['top'].set_visible(False)
d = 0.012
kw = dict(color='k', clip_on=False, lw=0.9)
t.plot((-d, +d), (-d * HR, +d * HR), transform=t.transAxes, **kw)
t.plot((1 - d, 1 + d), (-d * HR, +d * HR), transform=t.transAxes, **kw)
b.plot((-d, +d), (1 - d, 1 + d), transform=b.transAxes, **kw)
b.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=b.transAxes, **kw)
fig.subplots_adjust(left=0.185, right=0.975, top=0.975, bottom=0.145)
t.set_yticklabels(['11.0'])
b.set_ylabel(YL)
b.yaxis.set_label_coords(-0.155, 0.60)
fig.savefig(OUT + 'ntol_sweep_4core_brk.pdf', bbox_inches='tight')
fig.savefig(OUT + 'ntol_sweep_4core_brk.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ---- single panel, no break ----
fig, b = plt.subplots(1, 1, figsize=(3.5, 2.6))
sweep(b)
b.set_ylabel(YL)
b.axhline(U['UL'], color='k', ls='--', lw=1.2)
b.set_ylim(9.68, 11.06)
b.text(0.55, U['UL'] - 0.03, 'no mask', fontsize=8, va='top')
fig.tight_layout()
fig.savefig(OUT + 'ntol_sweep_4core_full.pdf', bbox_inches='tight')
fig.savefig(OUT + 'ntol_sweep_4core_full.png', dpi=200, bbox_inches='tight')
print('no mask 95%% UL = %.4f; wrote ntol_sweep_4core_{brk,full}.pdf/png to %s' % (U['UL'], OUT))
