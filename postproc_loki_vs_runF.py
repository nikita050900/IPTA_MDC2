"""
Apples-to-apples comparison: Loki direct-dL run vs Run F (QuickCW dL masking).
Both are G2D2, fixed fGW, detection mode.

Produces (saved to OUTDIR):
    corner_comparison.png   -- {cos_inc, log10_h0, log10_Mc, Phi0, psi}
                               Loki (blue) vs Run F post-mask (orange)
    summary.txt             -- SD BF for both runs + sample counts
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import corner
from enterprise_extensions import model_utils

# ------------------------------------------------------------------ #
#  PATHS -- set Run F h5 path before running
# ------------------------------------------------------------------ #
LOKI_H5  = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_fixed_detect_loki_22_Apr_2026.h5"
RUNF_H5  = ("/scratch/na00078/projects/IPTA_MDC2/h5_files/dl_masked/"
            "G2D2_narrow_detect_dLmasked_75.400Mpc.h5")
OUTDIR   = "/scratch/na00078/projects/IPTA_MDC2/loki_postproc"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  G2D2 INJECTION (Table 1 of the paper)
# ------------------------------------------------------------------ #
DL_INJ   = 75.4          # Mpc
MC_INJ   = np.log10(4.3e9)
INC_INJ  = np.cos(0.841)
PHI0_INJ = 0.244
PSI_INJ  = 1.119
H0_INJ   = -13.668773493298787  # verified from existing analysis script

# Order: cos_inc, log10_h0, log10_Mc, Phi0, psi
TRUTHS = [INC_INJ, H0_INJ, MC_INJ, PHI0_INJ, PSI_INJ]
LABELS = [
    r"$\cos\iota$",
    r"$\log_{10}h_0$",
    r"$\log_{10}\mathcal{M}$",
    r"$\Phi_0$",
    r"$\psi$",
]
RANGES = [
    (-1, 1),          # cos inc
    (-18, -11),       # log10 h0
    (7.03, 10),       # log10 Mc
    (0, 2 * np.pi),   # Phi0
    (0, np.pi),       # psi
]

# ------------------------------------------------------------------ #
#  PHYSICAL CONSTANTS
# ------------------------------------------------------------------ #
C_SI     = 299792458.0
T_SUN    = 1.327124400e20 / C_SI**3   # seconds
MPC_TO_M = 3.085677581e22

def derive_log10_h(log10_Mc, log10_fGW, log10_dL):
    """Equation 2 of the paper: h0 = 2 Mc^(5/3) (pi fGW)^(2/3) / dL."""
    Mc   = 10.0**log10_Mc  * T_SUN      # solar masses -> seconds
    fGW  = 10.0**log10_fGW              # Hz
    dL_m = 10.0**log10_dL  * MPC_TO_M  # Mpc -> metres
    h0   = 2.0 * Mc**(5.0/3.0) * (np.pi * fGW)**(2.0/3.0) * C_SI / dL_m
    return np.log10(h0)

def find_idx(par_names, candidates):
    for key in candidates:
        for i, n in enumerate(par_names):
            if n == key:
                return i
    raise ValueError(f"None of {candidates} found in {par_names}")

# ================================================================== #
#  LOAD LOKI RUN
#  Parameters sampled: cos_gwtheta, cos_inc, gwphi, log10_dL,
#                      log10_fGW, log10_Mc, phase0, psi
# ================================================================== #
print("Loading Loki run ...")
with h5py.File(LOKI_H5, "r") as f:
    loki_raw   = f["samples_cold"][0, :, :]
    loki_names = [x.decode("UTF-8") for x in f["par_names"][...]]

print(f"  Loki chain shape: {loki_raw.shape}")
print(f"  Loki par_names:   {loki_names}")

i_Mc_L   = find_idx(loki_names, ["0_log10_mc",   "0_log10_Mc"])
i_inc_L  = find_idx(loki_names, ["0_cos_inc",    "cos_inc"])
i_dL_L   = find_idx(loki_names, ["0_log10_dist", "0_log10_dL"])
i_fGW_L  = find_idx(loki_names, ["0_log10_fgw",  "0_log10_f_GW"])
i_phi0_L = find_idx(loki_names, ["0_phase0",     "phase0"])
i_psi_L  = find_idx(loki_names, ["0_psi",        "psi"])

loki_log10h = derive_log10_h(
    loki_raw[:, i_Mc_L],
    loki_raw[:, i_fGW_L],
    loki_raw[:, i_dL_L],
)

# Order: cos_inc, log10_h0, log10_Mc, Phi0, psi
loki_samples = np.column_stack([
    loki_raw[:, i_inc_L],
    loki_log10h,
    loki_raw[:, i_Mc_L],
    loki_raw[:, i_phi0_L],
    loki_raw[:, i_psi_L],
])

finite_mask_loki = np.isfinite(loki_samples).all(axis=1)
loki_samples = loki_samples[finite_mask_loki]
print(f"  Loki usable samples: {loki_samples.shape[0]}")

# ================================================================== #
#  LOAD RUN F (already dL-masked)
#  Parameters sampled by QuickCW: log10_h0, log10_mc, cos_inc,
#                                  phase0, psi, gwphi, cos_gwtheta, (noise params...)
# ================================================================== #
print("\nLoading Run F ...")
with h5py.File(RUNF_H5, "r") as f:
    runf_raw   = f["samples_masked"][...]
    runf_names = [x.decode("utf-8") for x in f["par_names"][...]]

print(f"  Run F shape: {runf_raw.shape}")

i_h0_F   = find_idx(runf_names, ["0_log10_h",  "0_log10_h0"])
i_Mc_F   = find_idx(runf_names, ["0_log10_mc", "0_log10_Mc"])
i_inc_F  = find_idx(runf_names, ["0_cos_inc",  "cos_inc"])
i_phi0_F = find_idx(runf_names, ["0_phase0",   "phase0"])
i_psi_F  = find_idx(runf_names, ["0_psi",      "psi"])

# Order: cos_inc, log10_h0, log10_Mc, Phi0, psi
runf_samples = np.column_stack([
    runf_raw[:, i_inc_F],
    runf_raw[:, i_h0_F],
    runf_raw[:, i_Mc_F],
    runf_raw[:, i_phi0_F],
    runf_raw[:, i_psi_F],
])

finite_mask_runf = np.isfinite(runf_samples).all(axis=1)
runf_samples = runf_samples[finite_mask_runf]
print(f"  Run F usable samples: {runf_samples.shape[0]}")

# ================================================================== #
#  SD BAYES FACTORS
# ================================================================== #
print("\nComputing SD BF for Run F (post-mask, log-uniform prior on h0) ...")
BF_F, BF_F_err = model_utils.bayes_fac(samples=runf_samples[:, 1], logAmax=-11)
print(f"  Run F  B10 = {BF_F:.4f} +/- {BF_F_err:.4f}")

print("Computing SD BF for Loki run (derived log10_h, prior is NOT log-uniform) ...")
BF_L, BF_L_err = model_utils.bayes_fac(samples=loki_samples[:, 1], logAmax=-11)
print(f"  Loki   B10 = {BF_L:.4f} +/- {BF_L_err:.4f}")
print("  WARNING: Loki B10 uses a non-log-uniform prior on h0.")
print("  The two B10 values are not directly comparable.")

# ================================================================== #
#  CORNER PLOT COMPARISON
# ================================================================== #
print("\nMaking corner plot ...")

fig = corner.corner(
    loki_samples,
    labels=LABELS,
    range=RANGES,
    truths=TRUTHS,
    truth_color="xkcd:red",
    color="xkcd:steel blue",
    show_titles=True,
    title_kwargs={"fontsize": 12},
    hist_kwargs={"density": True},
)

corner.corner(
    runf_samples,
    labels=LABELS,
    range=RANGES,
    truths=TRUTHS,
    truth_color="xkcd:red",
    color="xkcd:orange",
    show_titles=False,
    hist_kwargs={"density": True},
    fig=fig,
)

legend_elements = [
    Line2D([0], [0], color="xkcd:steel blue", lw=2,
           label=r"Loki (direct $d_L$ sampling)"),
    Line2D([0], [0], color="xkcd:orange", lw=2,
           label=r"Run F (QuickCW + $d_L$ masking)"),
    Line2D([0], [0], color="xkcd:red", lw=1.5, linestyle="-",
           label="Injected"),
]
fig.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=13,
    frameon=False,
    bbox_to_anchor=(0.98, 0.98),
)

outpath = os.path.join(OUTDIR, "corner_comparison.png")
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {outpath}")

# ================================================================== #
#  SUMMARY FILE
# ================================================================== #
summary_path = os.path.join(OUTDIR, "summary.txt")
with open(summary_path, "w") as fout:
    fout.write("Loki vs Run F comparison summary\n")
    fout.write("=" * 50 + "\n\n")
    fout.write(f"Loki usable samples:   {loki_samples.shape[0]}\n")
    fout.write(f"Run F usable samples:  {runf_samples.shape[0]}\n\n")
    fout.write(f"Run F  B10 = {BF_F:.4f} +/- {BF_F_err:.4f}  (log-uniform prior on h0)\n")
    fout.write(f"Loki   B10 = {BF_L:.4f} +/- {BF_L_err:.4f}  (derived log10_h; prior NOT log-uniform)\n\n")
    fout.write("Injected log10_h: {:.4f}\n".format(H0_INJ))
    fout.write("\nNOTE: Loki B10 is not directly comparable to Run F B10.\n")
    fout.write("The paper should present both with the caveat that the\n")
    fout.write("implicit prior on h0 differs between the two approaches.\n")
print(f"  Saved: {summary_path}")
print("\nDone.")