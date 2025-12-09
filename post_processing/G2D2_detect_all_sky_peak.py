#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import h5py

# ------------------------------------------------------------
# Input H5 file
# ------------------------------------------------------------
infile = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_broad_peak_all_sky_outfile.h5"
first_n_param = 8

print("Reading:", infile)

with h5py.File(infile, 'r') as f:
    samples_cold = f['samples_cold'][:, :, :first_n_param]
    par_names = [x.decode("utf-8") for x in list(f["par_names"])]

print("Loaded samples_cold shape:", samples_cold[-1].shape)

# ------------------------------------------------------------
# True injected values
# ------------------------------------------------------------
xxx0 = {
    "0_cos_inc": 0.8412486994612669,  
    "0_log10_fgw": np.log10(3.7e-9),
    "0_log10_h": -13.668773493298787,
    "0_log10_mc": np.log10(4.3e9),
    "phase0": 0.24434609527920614,
    "psi": 1.1187560505283651
}

# ------------------------------------------------------------
# Parameter order requested:
#   cos(i), fgw, h, mc, phase0, psi
# ------------------------------------------------------------
# INDEX MAPPING IN samples_cold:
#   0 = cos(theta)
#   1 = cos(i)
#   2 = phi
#   3 = log10(fgw)
#   4 = log10(h)
#   5 = log10(mc)
#   6 = phase0
#   7 = psi

corner_mask = [1, 3, 4, 5, 6, 7]

labels = [
    r"$\cos\iota$",
    r"$\log_{10} f_{\rm GW}$",
    r"$\log_{10} A_{\rm e}$",
    r"$\log_{10} {\cal M}$",
    r"$\Phi_0$",
    r"$\psi$",
]

label_to_key = {
    r"$\cos\iota$": "0_cos_inc",
    r"$\log_{10} f_{\rm GW}$": "0_log10_fgw",
    r"$\log_{10} A_{\rm e}$": "0_log10_h",
    r"$\log_{10} {\cal M}$": "0_log10_mc",
    r"$\Phi_0$": "phase0",
    r"$\psi$": "psi",
}

# ------------------------------------------------------------
# Truth values in correct order
# ------------------------------------------------------------
truths = [xxx0[label_to_key[l]] for l in labels]

# ------------------------------------------------------------
# Extract samples from cold chain
# ------------------------------------------------------------
burnin = 0
thin = 1
raw = samples_cold[0][burnin::thin, :]
samples2plot = raw[:, corner_mask]

# ------------------------------------------------------------
# Axis ranges
# ------------------------------------------------------------
ranges = [
    (-1, 1),          # cos(i)
    (-9, -7),         # log10 fGW
    (-18, -11),       # log10 h
    (8, 10),          # log10 Mc
    (0, 2*np.pi),     # phase0
    (0, np.pi)        # psi
]

# ------------------------------------------------------------
# Corner plot
# ------------------------------------------------------------
fig = corner.corner(
    samples2plot,
    labels=labels,
    truths=truths,
    truth_color="red",
    range=ranges,
    show_titles=True,
    title_fmt=".2f",
    hist_kwargs={"density": True},
    label_kwargs={"fontsize": 18}
)

axes = np.array(fig.axes).reshape(len(labels), len(labels))

# ------------------------------------------------------------
# Add PRIOR LINES (green) on diagonals
# ------------------------------------------------------------

# cos(i) prior: uniform [-1,1]
x = np.linspace(-1, 1, 500)
axes[0,0].plot(x, np.ones_like(x)*0.5, color="green")

# log10(fGW) prior: assume uniform
x = np.linspace(-9, -7, 500)
axes[1,1].plot(x, np.ones_like(x)*(1/2), color="green")

# log10(h) prior: assume log-uniform
x = np.linspace(-18, -11, 500)
axes[2,2].plot(x, np.ones_like(x)*(1/7), color="green")

# log10(Mc) prior: uniform
x = np.linspace(8, 10, 500)
axes[3,3].plot(x, np.ones_like(x)*(1/2), color="green")

# phase0 prior: uniform [0,2π]
x = np.linspace(0, 2*np.pi, 500)
axes[4,4].plot(x, np.ones_like(x)*(1/(2*np.pi)), color="green")

# psi prior: uniform [0,π]
x = np.linspace(0, np.pi, 500)
axes[5,5].plot(x, np.ones_like(x)*(1/np.pi), color="green")

# ------------------------------------------------------------
# Compute Single-Dimensional Bayes Factor for log10A
# ------------------------------------------------------------
from enterprise_extensions import model_utils

# Extract log10A samples:
# corner_mask = [0, 2, 1, 4, 5, 6, 7]
# Index 4 corresponds to "0_log10_h"
samples_logA = samples_cold[0][burnin::thin, 4]

# OPTIONAL: thin for independence (common in BF estimation)
samples_logA_thin = samples_logA[::10]

# Compute SD BF (log-uniform noise vs signal evidence)
BF, BF_err = model_utils.bayes_fac(samples=samples_logA_thin, logAmax=-11)

print("\n==============================")
print(f"Single–Dimensional Bayes Factor (log10A): {BF:.4f} ± {BF_err:.4f}")
print("==============================\n")


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------
outfile = "corner_plot_peak.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
print("Saved corner plot to:", outfile)
