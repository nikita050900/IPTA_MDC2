#!/usr/bin/env python
# fig2_final_24sep.py: Figure 2 (ntol sweep, Run D).
# Panel is an exact reproduction of the Overleaf figure written by
# final_paper_analysis.py: figsize 3.5 x 2.6, font size 9, log x, xlim 0.4 to 60,
# default log ticks, same colours, symbols and estimators. Nothing about the sweep,
# the Enterprise band or the QuickGWecc diamonds is changed.
# Two outputs:
#   ntol_sweep_4core_min.pdf    the panel alone (unmasked limit goes in the caption)
#   ntol_sweep_4core_inset.pdf  the same panel plus a small inset showing the
#                               cumulative log10 Mc distribution with and without the
#                               mask, where the 95% level meets the prior edge when no
#                               mask is applied (Sarah, 10 Sep: show the "100%" point)
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PP = (sys.argv[1] if len(sys.argv) > 1 else '.').rstrip('/') + '/'
OUT = (sys.argv[2] if len(sys.argv) > 2 else PP + 'fig_24sep').rstrip('/') + '/'
os.makedirs(OUT, exist_ok=True)
R = json.load(open(PP + 'final_paper_analysis.json'))
U = json.load(open(PP + 'fig2_unmasked_point.json'))
H = json.load(open(PP + 'fig2_mc_hist.json'))

C_QCW, C_ENT, C_M, C_C10 = '#4477AA', '#228833', '#CC6677', '#66CCEE'
nt = [r[0] for r in R['sweep']]; ul = [r[2] for r in R['sweep']]; er = [r[3] for r in R['sweep']]
E = R['ent']; M = R['lokiM_1pc']; C10 = R.get('loki_10pc')

def panel():
    plt.rcParams.update({'font.size': 9})
    fig, b = plt.subplots(1, 1, figsize=(3.5, 2.6))
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
    return fig, b

fig, b = panel()
fig.tight_layout()
fig.savefig(OUT + 'ntol_sweep_4core_min.pdf', bbox_inches='tight')
fig.savefig(OUT + 'ntol_sweep_4core_min.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# inset: cumulative log10 Mc, masked (1%) and unmasked, with the 95% level
fig, b = panel()
b.set_ylim(b.get_ylim()[0], b.get_ylim()[0] + (b.get_ylim()[1] - b.get_ylim()[0]) * 1.55)
edges = np.array(H['edges']); ctr = 0.5 * (edges[1:] + edges[:-1]); w = edges[1] - edges[0]
ins = b.inset_axes([0.40, 0.56, 0.57, 0.40])
for key, col, lab in (('masked_1pc', C_QCW, r'$\eta_{\rm tol}=1\%$'), ('unmasked', 'k', 'no mask')):
    cdf = np.cumsum(np.array(H[key]) * w); cdf /= cdf[-1]
    ins.plot(ctr, cdf, color=col, lw=1.2, label=lab)
ins.axhline(0.95, color='0.45', ls=':', lw=0.9)
ins.set_xlim(7.0, 11.0); ins.set_ylim(0, 1.05)
ins.set_xticks([7, 8, 9, 10, 11]); ins.set_yticks([0, 0.95])
ins.set_yticklabels(['0', '0.95'])
ins.tick_params(labelsize=6.5, length=2.5, pad=1.5)
ins.set_xlabel(r'$\log_{10}\mathcal{M}_c$', fontsize=7, labelpad=1)
ins.legend(fontsize=6.2, frameon=False, loc='upper left', handlelength=1.1,
           borderpad=0.1, labelspacing=0.25, handletextpad=0.5)
fig.tight_layout()
fig.savefig(OUT + 'ntol_sweep_4core_inset.pdf', bbox_inches='tight')
fig.savefig(OUT + 'ntol_sweep_4core_inset.png', dpi=200, bbox_inches='tight')
print('unmasked 95%% = %.4f, masked 1%% = %.4f; wrote ntol_sweep_4core_{min,inset}.pdf/png to %s'
      % (U['UL'], H['q95_masked_1pc'], OUT))
