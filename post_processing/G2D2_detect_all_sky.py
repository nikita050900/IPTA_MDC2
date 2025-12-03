#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")   # needed for saving PNGs without displaying a window
import matplotlib.pyplot as plt
import corner
import h5py
import scipy.constants as sc

# ------------------------------------------------------------
# Input H5 file
# ------------------------------------------------------------
infile = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_detect_allsky_outfile.h5"
first_n_param = 8

print("Reading:", infile)

with h5py.File(infile, 'r') as f:
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][:, :, :first_n_param]
    log_likelihood = f['log_likelihood'][:1, :]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
    acc_fraction = f['acc_fraction'][...]
    fisher_diag = f['fisher_diag'][...]

print("Loaded samples_cold shape:", samples_cold[-1].shape)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
KPC2S = sc.parsec / sc.c * 1e3
SOLAR2S = sc.G / sc.c ** 3 * 1.98855e30

# ------------------------------------------------------------
# True injected values (dictionary)
# ------------------------------------------------------------
xxx0 = {
    "0_cos_gwtheta": np.cos(0.6387905062299246),
    "0_cos_inc": 0.8412486994612669,
    "0_gwphi": 3.3335788713091694,
    "0_log10_fgw": np.log10(3.7e-09),
    "0_log10_h": -13.668773493298787,
    "0_log10_mc": np.log10(4.3e9),
    "phase0": 0.24434609527920614,
    "psi": 1.1187560505283651,
    "distance": 75.4,
}

# ------------------------------------------------------------
# Corner plot parameter selection (NO fGW)
# ------------------------------------------------------------
corner_mask = [0, 1, 2, 4, 5, 6, 7]  # drop index 3 (fgw)

labels = [
    r"$\alpha\,$(RA)",      
    r"$\delta\,$(Dec)",     
    r"$\cos \iota$",
    r"$\log_{10} A_{\rm e}$",
    r"$\log_{10} {\cal M}$",
    r"$\Phi_0$",
    r"$\psi$"
]

label_to_key = {
    r"$\alpha\,$(RA)": "0_gwphi",
    r"$\delta\,$(Dec)": "0_cos_gwtheta",
    r"$\cos \iota$": "0_cos_inc",
    r"$\log_{10} A_{\rm e}$": "0_log10_h",
    r"$\log_{10} {\cal M}$": "0_log10_mc",
    r"$\Phi_0$": "phase0",
    r"$\psi$": "psi"
}

# ------------------------------------------------------------
# Truth values
# ------------------------------------------------------------
truths = []
for l in labels:
    key = label_to_key[l]
    if l == r"$\delta\,$(Dec)":
        truths.append(np.degrees(np.arcsin(xxx0["0_cos_gwtheta"])))
    elif l == r"$\alpha\,$(RA)":
        truths.append(xxx0["0_gwphi"])
    else:
        truths.append(xxx0[key])

# ------------------------------------------------------------
# Extract samples (no d_L masking)
# ------------------------------------------------------------
burnin = 0
thin = 1
samples_raw = samples_cold[0][burnin::thin, :]

cos_theta = samples_raw[:, 0]
phi = samples_raw[:, 2]

RA = phi
Dec = np.degrees(np.arcsin(cos_theta))

samples2plot = np.vstack([
    RA,
    Dec,
    samples_raw[:, 1],   # cos_inc
    samples_raw[:, 4],   # log10_h
    samples_raw[:, 5],   # log10_mc
    samples_raw[:, 6],   # phase0
    samples_raw[:, 7]    # psi
]).T

# ------------------------------------------------------------
# Ranges
# ------------------------------------------------------------
ranges = [
    (0, 2*np.pi),  
    (-90, 90),      
    (-1, 1),        
    (-18, -11),     
    (8, 10),        
    (0, 2*np.pi),   
    (0, np.pi)      
]

# ------------------------------------------------------------
# Corner plot
# ------------------------------------------------------------
fig = corner.corner(
    samples2plot,
    labels=labels,
    truths=truths,
    truth_color="red",
    show_titles=True,
    range=ranges,
    hist_kwargs={"density": True}
)

for ax in fig.get_axes():
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

fig.suptitle(
    "Corner plot (no $d_L$ masking, RA/Dec included, no $f_{\\mathrm{GW}}$)",
    fontsize=24, y=1.05
)

# ------------------------------------------------------------
# Save to PNG
# ------------------------------------------------------------
outfile = "corner_plot.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
print("Saved corner plot to:", outfile)
