#!/usr/bin/env python
"""
Side-by-side 2D histogram comparison of frequency-segmented Mc UL.
Left panel:  Run D (QuickCW + dL masking)
Right panel: Loki (direct dL sampling)

Both panels show the (log10 fGW, log10 Mc) sample density on log color scale,
with the per-bin 95% UL on Mc overlaid as a step line with error bars.

Saved to OUTDIR:
    freq_segmented_UL_2dhist_comparison.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
import h5py
from scipy.stats import gaussian_kde

# ------------------------------------------------------------------ #
#  PATHS
# ------------------------------------------------------------------ #
LOKI_H5  = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D1_broad_UL_loki_30_Apr_2026.h5"
RUND_H5  = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D1_broad_UL_1e9_fixed_gamma_outfile.h5"
OUTDIR   = "/scratch/na00078/projects/IPTA_MDC2/loki_postproc"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  CONSTANTS
# ------------------------------------------------------------------ #
TARGET_DL_MPC   = 75.4
DL_TOL          = 0.01
TSPAN_YEARS     = 15.0
N_MIN           = 50
CONF            = 0.95

C_SI            = 299792458.0
T_SUN           = 1.327124400e20 / C_SI**3
MPC_TO_M        = 3.085677581e22

# ------------------------------------------------------------------ #
#  HELPERS
# ------------------------------------------------------------------ #
def find_idx(par_names, candidates):
    for key in candidates:
        for i, n in enumerate(par_names):
            if n == key:
                return i
    raise ValueError(f"None of {candidates} found in {par_names}")

def derive_log10_dL(log10_Mc, log10_fGW, log10_h):
    Mc   = 10.0**log10_Mc * T_SUN
    fGW  = 10.0**log10_fGW
    h    = 10.0**log10_h
    dL_m = 2.0 * Mc**(5.0/3.0) * (np.pi * fGW)**(2.0/3.0) * C_SI / h
    return np.log10(dL_m / MPC_TO_M)

def compute_frequency_bins(fgw_lin, mc_lin, years):
    fmin, fmax = fgw_lin.min(), fgw_lin.max()
    Tspan = years * 365.25 * 24 * 3600
    df    = 1.0 / Tspan
    edges = np.arange(fmin, fmax + df, df)
    inds  = np.digitize(fgw_lin, edges) - 1
    bin_indices = [np.where(inds == i)[0] for i in range(len(edges) - 1)]
    mc_by_bin   = [mc_lin[idx] for idx in bin_indices]
    return edges, mc_by_bin

def compute_UL_per_bin(mc_by_bin, conf=0.95, N_min=50):
    counts, ULs, errs, valid = [], [], [], []
    for samples in mc_by_bin:
        n_k = len(samples)
        counts.append(n_k)
        if n_k < N_min:
            ULs.append(np.nan); errs.append(np.nan); valid.append(False); continue
        ul = np.quantile(samples, conf)
        kde = gaussian_kde(samples)
        f_q = kde.evaluate([ul])[0]
        sigma = np.sqrt(conf * (1 - conf)) / (f_q * np.sqrt(n_k))
        ULs.append(ul); errs.append(sigma); valid.append(True)
    return np.array(counts), np.array(ULs), np.array(errs), np.array(valid)

def linear_to_log10_err(UL, UL_err):
    return UL_err / (UL * np.log(10))

def bin_centers(edges):
    return np.sqrt(edges[:-1] * edges[1:])

# ================================================================== #
#  LOAD LOKI RUN
# ================================================================== #
print("Loading Loki run ...")
with h5py.File(LOKI_H5, "r") as f:
    loki_raw   = f["samples_cold"][0, :, :]
    loki_names = [x.decode("UTF-8") for x in f["par_names"][...]]

i_Mc_L  = find_idx(loki_names, ["0_log10_mc"])
i_fGW_L = find_idx(loki_names, ["0_log10_fgw"])

log10_Mc_L  = loki_raw[:, i_Mc_L]
log10_fGW_L = loki_raw[:, i_fGW_L]
fGW_lin_L = 10.0**log10_fGW_L
Mc_lin_L  = 10.0**log10_Mc_L
print(f"  Loki samples: {len(Mc_lin_L)}")

# ================================================================== #
#  LOAD RUN D + APPLY 1% dL MASK
# ================================================================== #
print("\nLoading Run D ...")
first_n_param = 8
with h5py.File(RUND_H5, "r") as f:
    rund_raw = f["samples_cold"][:, :, :first_n_param]

# Standard QCW order: cos_gwtheta, cos_inc, gwphi, log10_fgw,
#                     log10_h, log10_mc, phase0, psi
log10_fGW_D = rund_raw[0, :, 3]
log10_h_D   = rund_raw[0, :, 4]
log10_Mc_D  = rund_raw[0, :, 5]

log10_dL_D = derive_log10_dL(log10_Mc_D, log10_fGW_D, log10_h_D)
dL_lin_D   = 10.0**log10_dL_D
mask_D     = (dL_lin_D >= TARGET_DL_MPC * (1 - DL_TOL)) & \
             (dL_lin_D <= TARGET_DL_MPC * (1 + DL_TOL))

fGW_lin_D = 10.0**log10_fGW_D[mask_D]
Mc_lin_D  = 10.0**log10_Mc_D[mask_D]
print(f"  Run D pre-mask: {len(log10_Mc_D)},  post-mask: {len(Mc_lin_D)}")

# ================================================================== #
#  PER-BIN ULs
# ================================================================== #
print("\nComputing per-bin ULs ...")
edges_L, mcbin_L = compute_frequency_bins(fGW_lin_L, Mc_lin_L, TSPAN_YEARS)
counts_L, ULs_L, errs_L, valid_L = compute_UL_per_bin(mcbin_L, conf=CONF, N_min=N_MIN)

edges_D, mcbin_D = compute_frequency_bins(fGW_lin_D, Mc_lin_D, TSPAN_YEARS)
counts_D, ULs_D, errs_D, valid_D = compute_UL_per_bin(mcbin_D, conf=CONF, N_min=N_MIN)

centers_L = bin_centers(edges_L)
centers_D = bin_centers(edges_D)

log_fGW_L = np.log10(centers_L[valid_L])
log_UL_L  = np.log10(ULs_L[valid_L])
log_err_L = linear_to_log10_err(ULs_L[valid_L], errs_L[valid_L])

log_fGW_D = np.log10(centers_D[valid_D])
log_UL_D  = np.log10(ULs_D[valid_D])
log_err_D = linear_to_log10_err(ULs_D[valid_D], errs_D[valid_D])

print(f"  Loki valid bins:  {valid_L.sum()} / {len(valid_L)}")
print(f"  Run D valid bins: {valid_D.sum()} / {len(valid_D)}")

# ================================================================== #
#  COMMON PLOT RANGES
# ================================================================== #
all_logf = np.concatenate([np.log10(fGW_lin_L), np.log10(fGW_lin_D)])
all_logmc = np.concatenate([np.log10(Mc_lin_L), np.log10(Mc_lin_D)])

x_lo, x_hi = np.min(all_logf), np.max(all_logf)
y_lo, y_hi = 7.0, 10.0  # match prior bound on log10 Mc

# ================================================================== #
#  SIDE-BY-SIDE PLOT
# ================================================================== #
print("\nMaking side-by-side comparison ...")
plt.rcParams["font.size"] = 14
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

# --- LEFT PANEL: RUN D ---
ax_D = axes[0]
hist_D = ax_D.hist2d(
    np.log10(fGW_lin_D), np.log10(Mc_lin_D),
    bins=50, range=[[x_lo, x_hi], [y_lo, y_hi]],
    norm=colors.LogNorm(), cmap="inferno_r", alpha=0.9,
)

sort_D = np.argsort(log_fGW_D)
ax_D.step(log_fGW_D[sort_D], log_UL_D[sort_D], where="mid",
          color="deepskyblue", lw=2.5, zorder=3,
          label=r"95% UL on $\mathcal{M}_c$")
ax_D.fill_between(log_fGW_D[sort_D],
                  log_UL_D[sort_D] - log_err_D[sort_D],
                  log_UL_D[sort_D] + log_err_D[sort_D],
                  step="mid", color="deepskyblue", alpha=0.25,
                  linewidth=0, zorder=2)
ax_D.errorbar(log_fGW_D[sort_D], log_UL_D[sort_D], yerr=log_err_D[sort_D],
              fmt='o', color='deepskyblue', ecolor='navy',
              elinewidth=1.8, capsize=3, lw=0, zorder=4, markersize=4)

ax_D.set_xlim(x_lo, x_hi)
ax_D.set_ylim(y_lo, y_hi)
ax_D.set_xlabel(r"$\log_{10}(f_{\rm GW}\,/\,{\rm Hz})$", fontsize=18)
ax_D.set_ylabel(r"$\log_{10}(\mathcal{M}_c\,/\,M_\odot)$", fontsize=18)
ax_D.set_title(r"Run D (QuickCW + $d_L$ masking)", fontsize=15)
ax_D.tick_params(direction="in", top=True, right=True, which="both", labelsize=13)
ax_D.legend(loc="lower left", fontsize=12, frameon=True)

# --- RIGHT PANEL: LOKI ---
ax_L = axes[1]
hist_L = ax_L.hist2d(
    np.log10(fGW_lin_L), np.log10(Mc_lin_L),
    bins=50, range=[[x_lo, x_hi], [y_lo, y_hi]],
    norm=colors.LogNorm(), cmap="inferno_r", alpha=0.9,
)

sort_L = np.argsort(log_fGW_L)
ax_L.step(log_fGW_L[sort_L], log_UL_L[sort_L], where="mid",
          color="deepskyblue", lw=2.5, zorder=3,
          label=r"95% UL on $\mathcal{M}_c$")
ax_L.fill_between(log_fGW_L[sort_L],
                  log_UL_L[sort_L] - log_err_L[sort_L],
                  log_UL_L[sort_L] + log_err_L[sort_L],
                  step="mid", color="deepskyblue", alpha=0.25,
                  linewidth=0, zorder=2)
ax_L.errorbar(log_fGW_L[sort_L], log_UL_L[sort_L], yerr=log_err_L[sort_L],
              fmt='o', color='deepskyblue', ecolor='navy',
              elinewidth=1.8, capsize=3, lw=0, zorder=4, markersize=4)

ax_L.set_xlim(x_lo, x_hi)
ax_L.set_ylim(y_lo, y_hi)
ax_L.set_xlabel(r"$\log_{10}(f_{\rm GW}\,/\,{\rm Hz})$", fontsize=18)
ax_L.set_title(r"Loki (direct $d_L$ sampling)", fontsize=15)
ax_L.tick_params(direction="in", top=True, right=True, which="both", labelsize=13)
ax_L.legend(loc="lower left", fontsize=12, frameon=True)

# Shared horizontal colorbar below both panels
cbar = fig.colorbar(hist_L[3], ax=axes.ravel().tolist(),
                    orientation="horizontal", pad=0.12, fraction=0.06)
cbar.set_label("Number of Samples", fontsize=14)
#cbar.ax.xaxis.set_major_formatter(ScalarFormatter())

outpath = os.path.join(OUTDIR, "freq_segmented_UL_2dhist_comparison.png")
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {outpath}")
print("\nDone.")