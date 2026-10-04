#!/usr/bin/env python
# corner_truth_4core_biglabels.py [broad|fixed] [LABEL_FS] [TICK_FS]
# Same data, subsampling (seed 42, same call order) and styling as the publication
# block at the end of run_broad_detect_4core.py / run_fixed_detect_4core.py, with
# larger axis labels and tick labels (Sarah, 8 Sep: "Make bigger labels") and at
# most 3 ticks per axis so the tick labels do not collide. Anvil paths.
# Output: post_processing/fig_24sep/g2d2_{broad,fixed}_truth_4core.{pdf,png}
import sys, os, time
import numpy as np, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

MODE = sys.argv[1] if len(sys.argv) > 1 else 'broad'
LABEL_FS = int(sys.argv[2]) if len(sys.argv) > 2 else 34
TICK_FS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
FIXED = (MODE == 'fixed')
H5 = os.environ['PROJECT'] + '/projects/IPTA_MDC2/h5_files/'
OUT = os.environ['PROJECT'] + '/projects/IPTA_MDC2/post_processing/fig_24sep/'
os.makedirs(OUT, exist_ok=True)
if FIXED:
    QCW_FILE = H5 + 'dl_masked/G2D2_narrow_detect_tref_4core_dLmasked_75.400Mpc.h5'
    LOKI_FILE = H5 + 'G2D2_fixed_detect_loki_100M_lastTOA_4core_outfile.h5'
    ENT_FILE = H5 + 'G2D2_core_fixed_new.h5'
    OUTNAME = 'g2d2_fixed_truth_4core'
else:
    QCW_FILE = H5 + 'dl_masked/G2D2_broad_detect_tref_4core_dLmasked_75.400Mpc.h5'
    LOKI_FILE = H5 + 'G2D2_broad_detect_loki_100M_lastTOA_4core_outfile.h5'
    ENT_FILE = H5 + 'G2D2_varyfgw_core_new.h5'
    OUTNAME = 'g2d2_broad_truth_4core'
ENT_BURN = 3000
np.random.seed(42)

megaparsec = 3.086e+22; speed_of_light = 299792458.0
T_sun = 1.327124400e20 / speed_of_light**3
D_MPC_FIXED = 75.400; F_GW_FIXED = np.log10(3.7e-9)
def derive_log10_h0(log10_mc, log10_fgw, log10_dL_Mpc):
    return (np.log10(2.0) + (5.0/3.0) * (log10_mc + np.log10(T_sun))
            + (2.0/3.0) * (np.log10(np.pi) + log10_fgw)
            - (log10_dL_Mpc + np.log10(megaparsec) - np.log10(speed_of_light)))

t0 = time.time()
with h5py.File(QCW_FILE, 'r') as f:
    qc = f['samples_masked'][...]
    qc_pn = [p.decode() if isinstance(p, bytes) else p for p in f['par_names'][:]]
with h5py.File(ENT_FILE, 'r') as f:
    en_all = f['chain'][...]
    en_pn = [p.decode() if isinstance(p, bytes) else p for p in f['params'][:]]
en = en_all[ENT_BURN:, :]
N_TARGET = len(en)
# same RNG call order as the original scripts: QCW choice first, then Loki choice
qc_idx = np.random.choice(len(qc), min(N_TARGET, len(qc)), replace=False)
qc_sub = qc[qc_idx]
with h5py.File(LOKI_FILE, 'r') as f:
    lk_pn = [p.decode() if isinstance(p, bytes) else p for p in f['par_names'][:]]
    NL = f['samples_cold'].shape[1]
    lk_idx = np.random.choice(NL, N_TARGET, replace=False)
    srt = np.sort(lk_idx)
    lk_sub = f['samples_cold'][0][srt, :]          # sorted fancy read, then restore order
    lk_sub = lk_sub[np.argsort(np.argsort(lk_idx))]
print('loaded in %.0fs: QCW %d, Loki %d of %d, Ent %d; loki pars %s' % (time.time()-t0, len(qc_sub), len(lk_sub), NL, len(en), lk_pn), flush=True)

qcw_mc, qcw_cosi = qc_sub[:, qc_pn.index('0_log10_mc')], qc_sub[:, qc_pn.index('0_cos_inc')]
qcw_h = qc_sub[:, qc_pn.index('0_log10_h')]
qcw_phi, qcw_psi = qc_sub[:, qc_pn.index('0_phase0')], qc_sub[:, qc_pn.index('0_psi')]
loki_mc, loki_cosi = lk_sub[:, lk_pn.index('0_log10_mc')], lk_sub[:, lk_pn.index('0_cos_inc')]
loki_dist = lk_sub[:, lk_pn.index('0_log10_dist')]
loki_phi, loki_psi = lk_sub[:, lk_pn.index('0_phase0')], lk_sub[:, lk_pn.index('0_psi')]
ent_mc, ent_cosi = en[:, en_pn.index('log10_mc')], en[:, en_pn.index('cos_inc')]
ent_phi, ent_psi = en[:, en_pn.index('phase0')], en[:, en_pn.index('psi')]
if FIXED:
    qcw_fgw = np.full(len(qc_sub), F_GW_FIXED); loki_fgw = np.full(len(lk_sub), F_GW_FIXED); ent_fgw = np.full(len(en), F_GW_FIXED)
else:
    qcw_fgw = qc_sub[:, qc_pn.index('0_log10_fgw')]; loki_fgw = lk_sub[:, lk_pn.index('0_log10_fgw')]; ent_fgw = en[:, en_pn.index('log10_fgw')]
loki_h = derive_log10_h0(loki_mc, loki_fgw, loki_dist)
ent_h = derive_log10_h0(ent_mc, ent_fgw, np.full(len(en), np.log10(D_MPC_FIXED)))

T_MC, T_H = np.log10(4.3e9), -13.668773493298787
T_COSI = np.cos(0.8412486994612669)
T_PHI, T_PSI = 0.24434609527920614, 1.1187560505283651
T_FGW = np.log10(3.7e-9)
if FIXED:
    cols = lambda mc, h, fgw, cosi, phi, psi: np.column_stack([mc, h, cosi, phi, psi])
    truths = [T_MC, T_H, T_COSI, T_PHI, T_PSI]
    labels = [r'$\log_{10}\mathcal{M}_c$', r'$\log_{10} h_0$', r'$\cos\iota$', r'$\Phi_0$', r'$\psi$']
    ranges = [(8.5, 10.0), (-15.0, -12.0), (-1, 1), (0, 2*np.pi), (0, np.pi)]
else:
    cols = lambda mc, h, fgw, cosi, phi, psi: np.column_stack([mc, h, fgw, cosi, phi, psi])
    truths = [T_MC, T_H, T_FGW, T_COSI, T_PHI, T_PSI]
    labels = [r'$\log_{10}\mathcal{M}_c$', r'$\log_{10} h_0$', r'$\log_{10} f_{\rm GW}$', r'$\cos\iota$', r'$\Phi_0$', r'$\psi$']
    ranges = [(8.5, 10.0), (-15.0, -12.0), (-9.0, -7.0), (-1, 1), (0, 2*np.pi), (0, np.pi)]
QCW_arr = cols(qcw_mc, qcw_h, qcw_fgw, qcw_cosi, qcw_phi, qcw_psi)
ENT_arr = cols(ent_mc, ent_h, ent_fgw, ent_cosi, ent_phi, ent_psi)
DL_arr = cols(loki_mc, loki_h, loki_fgw, loki_cosi, loki_phi, loki_psi)

plt.rcParams['xtick.labelsize'] = TICK_FS
plt.rcParams['ytick.labelsize'] = TICK_FS
LW = 1.6
sig = np.array([0.5, 1.0, 1.5, 2.0]); levels = 1.0 - np.exp(-0.5*sig**2)
base_kw = dict(labels=labels, bins=30, smooth=1.0, levels=levels, range=ranges,
               plot_datapoints=False, plot_density=False, no_fill_contours=True,
               fill_contours=False, label_kwargs={'fontsize': LABEL_FS},
               max_n_ticks=3, labelpad=0.12)
C_QCW, C_ENT, C_DL = '#4477AA', '#228833', '#CC6677'
fig = corner.corner(QCW_arr, color=C_QCW, contour_kwargs={'linewidths': LW, 'colors': C_QCW}, hist_kwargs={'density': True, 'color': C_QCW, 'lw': LW}, **base_kw)
corner.corner(ENT_arr, fig=fig, color=C_ENT, contour_kwargs={'linewidths': LW, 'colors': C_ENT}, hist_kwargs={'density': True, 'color': C_ENT, 'lw': LW}, **base_kw)
corner.corner(DL_arr, fig=fig, color=C_DL, contour_kwargs={'linewidths': LW, 'colors': C_DL}, hist_kwargs={'density': True, 'color': C_DL, 'lw': LW}, **base_kw)
corner.overplot_lines(fig, truths, color='k', lw=LW)
for ax in fig.axes:
    ax.tick_params(axis='both', which='major', length=6, width=1.2, pad=6)
fig.savefig(OUT + OUTNAME + '.png', dpi=200, bbox_inches='tight')
fig.savefig(OUT + OUTNAME + '.pdf', bbox_inches='tight')
print('saved', OUT + OUTNAME, 'label_fs', LABEL_FS, 'tick_fs', TICK_FS, 'in %.0fs' % (time.time()-t0))
