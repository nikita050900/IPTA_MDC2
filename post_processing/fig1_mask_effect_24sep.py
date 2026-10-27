#!/usr/bin/env python
# fig1_mask_effect_24sep.py: Figure 1 (dL mask effect, Runs E and F) regenerated from
# the 4 core unmasked outfiles with the same convention as fig1.ipynb (Aug 2026):
# pre-mask samples thinned by 100 (blue), post-mask samples (red), injected values
# (black), 0.5/1/1.5/2 sigma contours, 1% dL window. Changes on 24 Sep 2026:
# axis labels without units and in the same font family as the other figures
# (plotting consistency), larger label and tick fonts (Sarah, 8 Sep). Anvil paths.
# Output: post_processing/fig_24sep/IPTA_MDC2_G2D2_{broad,narrow}_detection_dL_75.4_4core_fig1_mask_effect.{pdf,png}
import os, sys, time
import numpy as np, h5py
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import corner

LABEL_FS = int(sys.argv[1]) if len(sys.argv) > 1 else 18
TICK_FS = int(sys.argv[2]) if len(sys.argv) > 2 else 13
H5 = os.environ['PROJECT'] + '/projects/IPTA_MDC2/h5_files/'
OUT = os.environ['PROJECT'] + '/projects/IPTA_MDC2/post_processing/fig_24sep/'
os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.size': LABEL_FS - 2, 'axes.labelsize': LABEL_FS,
                     'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS,
                     'axes.linewidth': 0.9, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True,
                     'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
                     'xtick.major.size': 4.0, 'ytick.major.size': 4.0, 'lines.linewidth': 1.2})
target_d_L = 75.4; eta_tol = 0.01
megaparsec = 3.086e+22; speed_of_light = 299792458.0
T_sun = 1.327124400e20 / speed_of_light**3
COL_F, COL_H, COL_MC = 3, 4, 5
C_PRE, C_POST = '#4477AA', '#CC6677'
RUNS = {
    'broad': dict(unmasked=H5 + 'G2D2_broad_detect_tref_4core_UNMASKED_outfile.h5',
                  base='IPTA_MDC2_G2D2_broad_detection_dL_75.4_4core_', FIXED=False),
    'fixed': dict(unmasked=H5 + 'G2D2_narrow_detect_tref_4core_UNMASKED_outfile.h5',
                  base='IPTA_MDC2_G2D2_narrow_detection_dL_75.4_4core_', FIXED=True),
}
CHUNK = 5_000_000; THIN_PRE = 100

def load(cfg):
    """pre-mask samples thinned by THIN_PRE and all post-mask samples (cols Mc, h0, fGW), chunked."""
    lo, hi = target_d_L * (1 - eta_tol), target_d_L * (1 + eta_tol)
    pre, post, npre = [], [], 0
    with h5py.File(cfg['unmasked'], 'r') as h:
        d = h['samples_cold']; N = d.shape[1]
        for a in range(0, N, CHUNK):
            b = min(a + CHUNK, N)
            sc = d[0, a:b, :8].astype(np.float32)
            h_amp = 10.0 ** sc[:, COL_H].astype(np.float64)
            fff = 10.0 ** sc[:, COL_F].astype(np.float64)
            mmm = 10.0 ** sc[:, COL_MC].astype(np.float64)
            dL = 2 * (mmm * T_sun) ** (5 / 3) * (np.pi * fff) ** (2 / 3) / h_amp * speed_of_light / megaparsec
            m = (dL >= lo) & (dL <= hi)
            post.append(sc[m][:, [COL_MC, COL_H, COL_F]])
            # global stride THIN_PRE across chunks, same rows as sc[::THIN_PRE] on the full array
            first = (-a) % THIN_PRE
            pre.append(sc[first::THIN_PRE][:, [COL_MC, COL_H, COL_F]])
            npre += 1
    pre = np.concatenate(pre); post = np.concatenate(post)
    return pre, post, N, lo, hi

for tag, cfg in RUNS.items():
    t0 = time.time()
    pre, post, N, lo, hi = load(cfg)
    print('%s: N=%d, dL window %.3f to %.3f Mpc, post mask N=%d, pre (thinned) %d, %.0fs'
          % (tag, N, lo, hi, len(post), len(pre), time.time() - t0), flush=True)
    FIXED = cfg['FIXED']
    ncol = 2 if FIXED else 3
    labels = [r'$\log_{10}\mathcal{M}_c$', r'$\log_{10} h_0$']
    truths = [np.log10(4.3e9), -13.67]
    rng = [(7.0, 10.3), (-15.0, -12.5)]
    if not FIXED:
        labels += [r'$\log_{10} f_{\rm GW}$']; truths += [-8.43]; rng += [(-8.8, -7.2)]
    pre = pre[:, :ncol]; post = post[:, :ncol]
    sig = np.array([0.5, 1.0, 1.5, 2.0]); levels = 1.0 - np.exp(-0.5 * sig ** 2)
    side = 2.35 if FIXED else 2.15
    base_kw = dict(labels=labels, range=rng, bins=30, smooth=1.0, levels=levels,
                   plot_datapoints=False, plot_density=False, no_fill_contours=True,
                   fill_contours=False, label_kwargs={'fontsize': LABEL_FS},
                   max_n_ticks=4, use_math_text=True)
    fig = corner.corner(pre, color=C_PRE, hist_kwargs={'density': True, 'color': C_PRE, 'lw': 1.3}, **base_kw)
    fig.set_size_inches(side * ncol, side * ncol)
    corner.corner(post, fig=fig, color=C_POST, truths=truths, truth_color='k',
                  hist_kwargs={'density': True, 'color': C_POST, 'lw': 1.3}, **base_kw)
    for ax in fig.get_axes():
        ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=TICK_FS)
        ax.xaxis.set_label_coords(0.5, -0.36)
        ax.yaxis.set_label_coords(-0.38, 0.5)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)
    stem = OUT + cfg['base'] + 'fig1_mask_effect'
    fig.savefig(stem + '.pdf'); fig.savefig(stem + '.png')
    plt.close(fig)
    print('saved', os.path.basename(stem) + '.pdf/.png', 'label_fs', LABEL_FS, 'tick_fs', TICK_FS, flush=True)
