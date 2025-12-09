#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import h5py

# ------------------------------------------------------------
# TRUTH VALUES
# ------------------------------------------------------------
truth_dict = {
    "0_cos_gwtheta": np.cos(0.6387905062299246),
    "0_cos_inc": 0.8412486994612669,
    "0_gwphi": 3.3335788713091694,
    "0_log10_fgw": np.log10(3.7e-9),
    "0_log10_h": -13.668773493298787,
    "0_log10_mc": np.log10(4.3e9),
    "phase0": 0.24434609527920614,
    "psi": 1.1187560505283651,
    "distance": 75.4
}

# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------
infile = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_broad_detect_outfile.h5"
first_n_param = 8

print("Reading:", infile)
with h5py.File(infile, "r") as f:
    samples_cold = f["samples_cold"][:, :, :first_n_param]
    par_names = [x.decode("utf-8") for x in list(f["par_names"])]
print("Loaded cold-chain shape:", samples_cold[0].shape)

# ------------------------------------------------------------
# PARAMETERS TO PLOT (only 5)
# ------------------------------------------------------------
# enterprise index reference:
# 0: cos(theta)
# 1: cos(iota)
# 2: gwphi
# 3: log10(fGW)
# 4: log10(h)
# 5: log10(Mc)
# 6: phi0
# 7: psi

corner_mask = [1, 4, 5, 6, 7]   # cos_i, log10A, log10Mc, phi0, psi

labels = [
    r"$\cos\iota$",
    r"$\log_{10} A_{\rm e}$",
    r"$\log_{10}\mathcal{M}$",
    r"$\Phi_0$",
    r"$\psi$"
]

label_to_key = {
    r"$\cos\iota$": "0_cos_inc",
    r"$\log_{10} A_{\rm e}$": "0_log10_h",
    r"$\log_{10}\mathcal{M}$": "0_log10_mc",
    r"$\Phi_0$": "phase0",
    r"$\psi$": "psi"
}

truths = [truth_dict[label_to_key[lbl]] for lbl in labels]

# ------------------------------------------------------------
# EXTRACT SAMPLES
# ------------------------------------------------------------
burnin = 0
thin = 1
raw = samples_cold[0][burnin::thin, :]
samples = raw[:, corner_mask]

# ------------------------------------------------------------
# RANGES FOR 5 PARAMETERS
# ------------------------------------------------------------
ranges = [
    (-1, 1),          # cos i
    (-18, -11),       # log10 A
    (8, 10),          # log10 Mc
    (0, 2*np.pi),     # phi0
    (0, np.pi)        # psi
]

# ------------------------------------------------------------
# MAKE CORNER PLOT
# ------------------------------------------------------------
fig = corner.corner(
    samples,
    labels=labels,
    truths=truths,
    truth_color="red",
    color="black",
    range=ranges,
    show_titles=True,
    title_fmt=".2f",
    hist_kwargs={"density": True, "color": "black"},
    contour_kwargs={"colors": "black"},
    label_kwargs={"fontsize": 18}
)

axes = np.array(fig.axes).reshape(len(labels), len(labels))

# ------------------------------------------------------------
# ADD PRIORS IN GREEN
# ------------------------------------------------------------

# cos i prior
x = np.linspace(-1, 1, 500)
axes[0, 0].plot(x, np.ones_like(x) * 0.5, color="green")

# log10 A prior
x = np.linspace(-18, -11, 500)
axes[1, 1].plot(x, np.ones_like(x) / 7, color="green")

# log10 Mc prior
x = np.linspace(8, 10, 500)
axes[2, 2].plot(x, np.ones_like(x) / 2, color="green")

# Phi0 prior
x = np.linspace(0, 2*np.pi, 500)
axes[3, 3].plot(x, np.ones_like(x) / (2*np.pi), color="green")

# psi prior
x = np.linspace(0, np.pi, 500)
axes[4, 4].plot(x, np.ones_like(x) / np.pi, color="green")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
outfile = "corner_pre_dL_mask_detect_5params.png"
fig.savefig(outfile, dpi=300, bbox_inches="tight")
print("Saved corner plot to:", outfile)
