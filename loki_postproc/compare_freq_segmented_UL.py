#!/usr/bin/env python
"""
Frequency-segmented Mc UL comparison: Loki direct dL sampling vs Run D (QCW + dL masking).
Both are G2D1, broad fGW, UL mode.

Produces (saved to OUTDIR):
    freq_segmented_UL_comparison.png    -- per-bin 95% UL on Mc(f_GW)
    freq_segmented_summary.txt          -- per-bin counts, ULs, errors for both runs
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    """Solve Eq. 2 for dL given Mc, fGW, h0."""
    Mc   = 10.0**log10_Mc * T_SUN
    fGW  = 10.0**log10_fGW
    h    = 10.0**log10_h
    dL_m = 2.0 * Mc**(5.0/3.0) * (np.pi * fGW)**(2.0/3.0) * C_SI / h
    return np.log10(dL_m / MPC_TO_M)

def compute_frequency_bins(fgw_samples, mc_samples, years, mode="step"):
    """fgw_samples and mc_samples are linear (not log)."""
    fmin, fmax = fgw_samples.min(), fgw_samples.max()
    if mode == "step":
        Tspan = years * 365.25 * 24 * 3600
        df    = 1.0 / Tspan
        edges = np.arange(fmin, fmax + df, df)
    else:
        raise ValueError("only 'step' mode supported")

    inds        = np.digitize(fgw_samples, edges) - 1
    bin_indices = [np.where(inds == i)[0] for i in range(len(edges) - 1)]
    mc_by_bin   = [mc_samples[idx] for idx in bin_indices]
    return edges, bin_indices, mc_by_bin

def compute_UL_per_bin(mc_by_bin, conf=0.95, N_min=50):
    counts, ULs, UL_errors, valid = [], [], [], []
    for samples in mc_by_bin:
        n_k = len(samples)
        counts.append(n_k)
        if n_k < N_min:
            ULs.append(np.nan)
            UL_errors.append(np.nan)
            valid.append(False)
            continue
        ul = np.quantile(samples, conf)
        kde = gaussian_kde(samples)
        f_q = kde.evaluate([ul])[0]
        sigma = np.sqrt(conf * (1 - conf)) / (f_q * np.sqrt(n_k))
        ULs.append(ul)
        UL_errors.append(sigma)
        valid.append(True)
    return np.array(counts), np.array(ULs), np.array(UL_errors), np.array(valid)

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
i_dL_L  = find_idx(loki_names, ["0_log10_dist"])
i_fGW_L = find_idx(loki_names, ["0_log10_fgw"])

log10_Mc_L  = loki_raw[:, i_Mc_L]
log10_fGW_L = loki_raw[:, i_fGW_L]

fGW_lin_L = 10.0**log10_fGW_L
Mc_lin_L  = 10.0**log10_Mc_L
print(f"  Loki usable samples: {len(Mc_lin_L)}")

# ================================================================== #
#  LOAD RUN D (existing post-mask QCW)
# ================================================================== #
print("\nLoading Run D ...")
first_n_param = 8
with h5py.File(RUND_H5, "r") as f:
    rund_raw   = f["samples_cold"][:, :, :first_n_param]
    rund_names = [x.decode("UTF-8") for x in f["par_names"][...]]

# Standard QCW parameter order: cos_gwtheta, cos_inc, gwphi, log10_fgw,
#                                log10_h, log10_mc, phase0, psi
log10_fGW_D = rund_raw[0, :, 3]
log10_h_D   = rund_raw[0, :, 4]
log10_Mc_D  = rund_raw[0, :, 5]

# Apply 1% dL mask around 75.4 Mpc
log10_dL_D  = derive_log10_dL(log10_Mc_D, log10_fGW_D, log10_h_D)
dL_lin_D    = 10.0**log10_dL_D
dL_min      = TARGET_DL_MPC * (1.0 - DL_TOL)
dL_max      = TARGET_DL_MPC * (1.0 + DL_TOL)
mask_D      = (dL_lin_D >= dL_min) & (dL_lin_D <= dL_max)

fGW_lin_D = 10.0**log10_fGW_D[mask_D]
Mc_lin_D  = 10.0**log10_Mc_D[mask_D]
print(f"  Run D pre-mask samples:  {len(log10_Mc_D)}")
print(f"  Run D usable post-mask:  {len(Mc_lin_D)}")

# ================================================================== #
#  FREQUENCY-SEGMENTED ULs
# ================================================================== #
print("\nComputing frequency-segmented ULs ...")
edges_L, _, mc_by_bin_L = compute_frequency_bins(fGW_lin_L, Mc_lin_L, TSPAN_YEARS)
counts_L, ULs_L, errs_L, valid_L = compute_UL_per_bin(mc_by_bin_L, conf=CONF, N_min=N_MIN)

edges_D, _, mc_by_bin_D = compute_frequency_bins(fGW_lin_D, Mc_lin_D, TSPAN_YEARS)
counts_D, ULs_D, errs_D, valid_D = compute_UL_per_bin(mc_by_bin_D, conf=CONF, N_min=N_MIN)

fGW_centers_L = bin_centers(edges_L)
fGW_centers_D = bin_centers(edges_D)

log_fGW_L = np.log10(fGW_centers_L[valid_L])
log_UL_L  = np.log10(ULs_L[valid_L])
log_err_L = linear_to_log10_err(ULs_L[valid_L], errs_L[valid_L])

log_fGW_D = np.log10(fGW_centers_D[valid_D])
log_UL_D  = np.log10(ULs_D[valid_D])
log_err_D = linear_to_log10_err(ULs_D[valid_D], errs_D[valid_D])

print(f"  Loki valid bins:  {valid_L.sum()} / {len(valid_L)}")
print(f"  Run D valid bins: {valid_D.sum()} / {len(valid_D)}")

# ================================================================== #
#  PLOT
# ================================================================== #
print("\nMaking comparison plot ...")
plt.rcParams["font.size"] = 14
fig, ax = plt.subplots(figsize=(9, 6.5))

sort_L = np.argsort(log_fGW_L)
ax.errorbar(
    log_fGW_L[sort_L], log_UL_L[sort_L], yerr=log_err_L[sort_L],
    fmt="o", color="xkcd:steel blue", ecolor="xkcd:steel blue",
    elinewidth=1.5, capsize=3, markersize=5,
    label=r"Loki (direct $d_L$ sampling)", zorder=3,
)

sort_D = np.argsort(log_fGW_D)
ax.errorbar(
    log_fGW_D[sort_D], log_UL_D[sort_D], yerr=log_err_D[sort_D],
    fmt="s", color="xkcd:orange", ecolor="xkcd:orange",
    elinewidth=1.5, capsize=3, markersize=5,
    label=r"Run D (QuickCW + $d_L$ masking)", zorder=3,
)

ax.set_xlabel(r"$\log_{10}(f_{\rm GW}\,/\,{\rm Hz})$", fontsize=18)
ax.set_ylabel(r"$\log_{10}(\mathcal{M}_c^{95\%}\,/\,M_\odot)$", fontsize=18)
ax.tick_params(direction="in", top=True, right=True, which="both", labelsize=14)
ax.legend(loc="best", fontsize=13, frameon=True)
ax.grid(False)

plt.tight_layout()
outpath = os.path.join(OUTDIR, "freq_segmented_UL_comparison.png")
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {outpath}")

# ================================================================== #
#  SUMMARY FILE
# ================================================================== #
summary_path = os.path.join(OUTDIR, "freq_segmented_summary.txt")
with open(summary_path, "w") as fout:
    fout.write("Frequency-segmented Mc UL comparison: Loki vs Run D\n")
    fout.write("=" * 60 + "\n\n")
    fout.write(f"Loki total samples:        {len(Mc_lin_L)}\n")
    fout.write(f"Run D pre-mask samples:    {len(log10_Mc_D)}\n")
    fout.write(f"Run D post-mask samples:   {len(Mc_lin_D)}\n\n")
    fout.write(f"Loki valid bins (n_k >= {N_MIN}):  {valid_L.sum()} / {len(valid_L)}\n")
    fout.write(f"Run D valid bins (n_k >= {N_MIN}): {valid_D.sum()} / {len(valid_D)}\n\n")

    fout.write("LOKI per-bin results:\n")
    fout.write(f"  {'log10_fGW':>10}  {'n_k':>8}  {'log10_UL':>10}  {'log10_err':>10}\n")
    for k in range(len(valid_L)):
        if valid_L[k]:
            fout.write(f"  {np.log10(fGW_centers_L[k]):>10.4f}  {counts_L[k]:>8d}  "
                       f"{np.log10(ULs_L[k]):>10.4f}  {linear_to_log10_err(ULs_L[k], errs_L[k]):>10.4f}\n")

    fout.write("\nRUN D per-bin results:\n")
    fout.write(f"  {'log10_fGW':>10}  {'n_k':>8}  {'log10_UL':>10}  {'log10_err':>10}\n")
    for k in range(len(valid_D)):
        if valid_D[k]:
            fout.write(f"  {np.log10(fGW_centers_D[k]):>10.4f}  {counts_D[k]:>8d}  "
                       f"{np.log10(ULs_D[k]):>10.4f}  {linear_to_log10_err(ULs_D[k], errs_D[k]):>10.4f}\n")

print(f"  Saved: {summary_path}")
print("\nDone.")