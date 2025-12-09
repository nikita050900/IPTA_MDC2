#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import h5py
import scipy.constants as sc

# ------------------------------------------------------------
# Input H5 file
# ------------------------------------------------------------
infile = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_broad_peak_all_sky_outfile.h5"
first_n_param = 8

print("Reading:", infile)

with h5py.File(infile, 'r') as f:
    samples_cold = f['samples_cold'][:, :, :first_n_param]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]

print("Loaded samples_cold shape:", samples_cold[-1].shape)

# ------------------------------------------------------------
# True injected values (dictionary)
# ------------------------------------------------------------
xxx0 = {
    "0_cos_gwtheta": np.cos(0.6387905062299246),    # cos θ
    "0_gwphi": 3.3335788713091694,                 # φ
    "0_cos_inc": 0.8412486994612669,               # cos ι
    "0_log10_h": -13.668773493298787,              # log10 A
    "0_log10_mc": np.log10(4.3e9),                 # log10 Mc
    "phase0": 0.24434609527920614,                 # Φ0
    "psi": 1.1187560505283651                       # ψ
}

# ------------------------------------------------------------
# Parameter order requested:
#   cosθ, φ, cosι, log10A, log10Mc, Φ0, ψ
# ------------------------------------------------------------
corner_mask = [0, 2, 1, 4, 5, 6, 7]

labels = [
    r"$\cos\theta$",
    r"$\phi$",
    r"$\cos\iota$",
    r"$A_{\rm e}$",
    r"$\mathcal{M}$",
    r"$\Phi_0$",
    r"$\psi$"
]

label_to_key = {
    r"$\cos\theta$": "0_cos_gwtheta",
    r"$\phi$": "0_gwphi",
    r"$\cos\iota$": "0_cos_inc",
    r"$A_{\rm e}$": "0_log10_h",
    r"$\mathcal{M}$": "0_log10_mc",
    r"$\Phi_0$": "phase0",
    r"$\psi$": "psi"
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
# Axis ranges matching typical PTA CW corner plots
# ------------------------------------------------------------
ranges = [
    (-1, 1),             # cosθ
    (0, 2*np.pi),        # φ
    (-1, 1),             # cosι
    (-18, -11),          # log10 A
    (8, 10),             # log10 Mc
    (0, 2*np.pi),        # Φ0
    (0, np.pi)           # ψ
]

# ------------------------------------------------------------
# Make corner plot
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

# ------------------------------------------------------------
# Add PRIOR LINES (green) on 1D histograms
# ------------------------------------------------------------
axes = np.array(fig.axes).reshape(len(labels), len(labels))

# cosθ prior: uniform in [-1,1]
x = np.linspace(-1, 1, 500)
prior_cth = np.ones_like(x) * 0.5

ax = axes[0,0]
ax.plot(x, prior_cth, color="green")

# φ prior: uniform on [0,2π]
x = np.linspace(0, 2*np.pi, 500)
prior_phi = np.ones_like(x) * 1/(2*np.pi)
ax = axes[1,1]
ax.plot(x, prior_phi, color="green")

# cosι prior
x = np.linspace(-1, 1, 500)
prior_ci = np.ones_like(x) * 0.5
ax = axes[2,2]
ax.plot(x, prior_ci, color="green")

# A prior (log uniform example)
x = np.linspace(-18, -11, 500)
prior_A = np.ones_like(x) / (7)
axes[3,3].plot(x, prior_A, color="green")

# Mc prior
x = np.linspace(8, 10, 500)
prior_M = np.ones_like(x) * (1/2)
axes[4,4].plot(x, prior_M, color="green")

# Φ0 prior
x = np.linspace(0, 2*np.pi, 500)
prior_ph = np.ones_like(x) * 1/(2*np.pi)
axes[5,5].plot(x, prior_ph, color="green")

# ψ prior
x = np.linspace(0, np.pi, 500)
prior_psi = np.ones_like(x) * 1/np.pi
axes[6,6].plot(x, prior_psi, color="green")

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
