#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import corner
import os
import datetime

import pickle
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise_extensions import deterministic
from scipy.stats import norm
import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP
import glob
import json
import h5py
import healpy as hp
import scipy.constants as sc
import emcee
from numba.typed import List
import sys
from enterprise_extensions import model_utils

# -------------------------------------------------------------------
# INPUT FILE
# -------------------------------------------------------------------
infile = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_narrow_detect_outfile.h5"
first_n_param = 8

print("Loading:", infile)

with h5py.File(infile, 'r') as f:
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][:,:,:first_n_param]
    print("Cold samples shape:", samples_cold[-1].shape)
    log_likelihood = f['log_likelihood'][:1,:]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
    acc_fraction = f['acc_fraction'][...]
    fisher_diag = f['fisher_diag'][...]

# -------------------------------------------------------------------
# PARAMETERS / LABELS
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 1. Setup labels & truth mapping
# -------------------------------------------------------------------
corner_mask = [0, 1, 2, 3, 4, 5, 6, 7]
par_keys = ["0_cos_gwtheta", "0_cos_inc", "0_gwphi", "0_log10_fgw",
            "0_log10_h", "0_log10_mc", "0_phase0", "0_psi", "log10_d_L"]

labels = [
    r"$\cos \iota$",
    r"$\log_{10} A_{\rm e}$",
    r"$\log_{10} {\cal M}$",
    r"$\Phi_0$",
    r"$\psi$"
]

label_to_key = {
    r"$\cos \iota$": "0_cos_inc",
    r"$\log_{10} A_{\rm e}$": "0_log10_h",
    r"$\log_{10} {\cal M}$": "0_log10_mc",
    r"$\Phi_0$": "phase0",
    r"$\psi$": "psi"
}



# -------------------------------------------------------------------
# 2. Burnin / thinning for masked samples
# -------------------------------------------------------------------
burnin = 0
thin = 1   # masked samples: keep your original thin
target_d_L = 75.4#Mpc

unmasked_thin = 100    # << REQUIRED >>

ranges = [(-1,1), (-18,-11), (9,10), (0,2*np.pi), (0,np.pi)]

# -------------------------------------------------------------------
# 3. Compute luminosity distance
# -------------------------------------------------------------------
megaparsec = 3.086e+22
speed_of_light = 299792458.0
T_sun = 1.327124400e20 / speed_of_light**3

h_amp = 10**samples_cold[0][burnin::thin,4]
fff   = 10**samples_cold[0][burnin::thin,3]
mmm   = 10**samples_cold[0][burnin::thin,5]

log10_d_L = np.log10(
    2 * (mmm * T_sun)**(5/3) * (np.pi * fff)**(2/3) / h_amp * speed_of_light / megaparsec
)

# -------------------------------------------------------------------
# 4. Generate masks
# -------------------------------------------------------------------
d_L_percent_tolerance = 1.0
d_L_min = target_d_L*(1 - d_L_percent_tolerance/100)
d_L_max = target_d_L*(1 + d_L_percent_tolerance/100)

d_L_mask = np.where((10**log10_d_L >= d_L_min) & (10**log10_d_L <= d_L_max))[0]


# -------------------------------------------------------------------
# 5. Create sample matrices
# -------------------------------------------------------------------
samples2plot = np.concatenate(
    (samples_cold[0][burnin::thin, corner_mask],
     np.array([log10_d_L]).T),
    axis=1
)

# masked samples
samples2plot_masked = np.vstack((
    samples2plot[d_L_mask,1],
    samples2plot[d_L_mask,4],
    samples2plot[d_L_mask,5],
    samples2plot[d_L_mask,6],
    samples2plot[d_L_mask,7]
)).T



samples2plot_unmasked = np.vstack((
    samples_cold[0][burnin::unmasked_thin, 1],
    samples_cold[0][burnin::unmasked_thin, 4],
    samples_cold[0][burnin::unmasked_thin, 5],
    samples_cold[0][burnin::unmasked_thin, 6],
    samples_cold[0][burnin::unmasked_thin, 7]
)).T

print("Masked samples:", samples2plot_masked.shape)
print("Unmasked samples (thinned):", samples2plot_unmasked.shape)

# -------------------------------------------------------------------
# 6. Colorblind-safe palette (Okabe–Ito)
# -------------------------------------------------------------------
c_masked   = "#0072B2"      # blue
c_unmasked = "#D55E00"      # vermillion

# -------------------------------------------------------------------
# 7. First (masked) corner plot
# -------------------------------------------------------------------
fig = corner.corner(
    samples2plot_masked,
    labels=labels,
    show_titles=True,
    range=ranges,
    color=c_masked,
    hist_kwargs={"density": True, "color": c_masked},
    contour_kwargs={"colors": [c_masked]},
    label_kwargs={"fontsize": 16},
)

# -------------------------------------------------------------------
# 8. Overplot unmasked samples (thinned)
# -------------------------------------------------------------------
corner.corner(
    samples2plot_unmasked,
    fig=fig,
    labels=labels,
    show_titles=False,
    range=ranges,
    color=c_unmasked,
    hist_kwargs={"density": True, "color": c_unmasked},
    contour_kwargs={"colors": [c_unmasked]},
)

# -------------------------------------------------------------------
# 9. Formatting
# -------------------------------------------------------------------
for ax in fig.get_axes():
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

# -------------------------------------------------------------------
# 10. Priors on 1D diagonals (unchanged)
# -------------------------------------------------------------------
for i, ax in enumerate(fig.axes):

    if i == 0:   # cos i
        Xs = np.linspace(-1,1)
        ax.plot(Xs, Xs*0 + 1/2, color="xkcd:green")

    elif i == (len(labels)+1):   # log10 A
        Xs = np.linspace(-18,-11)
        ax.plot(Xs, Xs*0 + 1/7, color="xkcd:green")

    elif i == 2*(len(labels)+1):   # log10 Mc
        Xs = np.linspace(
            np.min(samples2plot[d_L_mask,5]),
            np.max(samples2plot[d_L_mask,5]))
        ax.plot(Xs, Xs*0 + 1/3, color="xkcd:green")

    elif i == 3*(len(labels)+1):   # phi0
        Xs = np.linspace(0,2*np.pi)
        ax.plot(Xs, Xs*0 + 1/(2*np.pi), color="xkcd:green")

    elif i == 4*(len(labels)+1):   # psi
        Xs = np.linspace(0,np.pi)
        ax.plot(Xs, Xs*0 + 1/np.pi, color="xkcd:green")

#plt.suptitle("dL-masked vs non-masked", fontsize=24, y=1.03)

# -------------------------------------------------------------------
# SAVE PNG IN ./plots/
# -------------------------------------------------------------------
os.makedirs("plots", exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"plots/corner_dlmask_{timestamp}.png"

plt.savefig(outfile, dpi=300, bbox_inches="tight")
print("Saved PNG:", outfile)

plt.show()

# -------------------------------------------------------------------
# BAYES FACTORS
# -------------------------------------------------------------------
BF, BF_err = model_utils.bayes_fac(samples=samples2plot[d_L_mask,4], logAmax=-11)
print(f"dL MASKED log10A BF = {BF:.4f} ± {BF_err:.4f}")

BF, BF_err = model_utils.bayes_fac(samples=samples_cold[0][::10,4], logAmax=-11)
print(f"UNMASKED log10A BF = {BF:.4f} ± {BF_err:.4f}")
