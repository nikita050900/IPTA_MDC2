#!/usr/bin/env python3
"""
Simulate IPTA MDC2 Group2 dataset with libstempo:
Inject a continuous wave (CW) + red/white noise.
No Enterprise or PINT dependencies.
"""

import os, json
import numpy as np
import libstempo as T
from astropy import constants as const

# ===============================================================
# PATHS
# ===============================================================
BASE = "/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2"
PAR_DIR = os.path.join(BASE, "par")
TIM_DIR = os.path.join(BASE, "tim")
NOISE_FILE = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/noise_files/fit_psr_noise_dataset2.json"
OUT_DIR = "/scratch/na00078/projects/IPTA_MDC2/sim_libstempo_dataset2"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================================================
# CW PARAMETERS
# ===============================================================
CW = {
    "chirp_mass": 4.3e9,                 # Msun
    "distance": 75.4,                    # Mpc
    "f_gw": 3.7e-9,                      # Hz
    "gw_phi": 3.3335788713091694,        # rad
    "gw_theta": 0.6387905062299246,      # rad
    "inclination": 0.8412486994612669,   # rad
    "log10_A_gwb": -15.070581074285707,
    "log10_h": -13.668773493298787,
    "phase0": 0.24434609527920614,
    "psi": 1.1187560505283651,
}

# ===============================================================
# UTILITIES
# ===============================================================
def load_noise_dict(path):
    with open(path, "r") as f:
        return json.load(f)

def get_noise_params(name, noise_dict):
    """Extract EFAC/EQUAD/red-noise for a single pulsar."""
    out = {}
    for k, v in noise_dict.items():
        if k.startswith(name + "_"):
            out[k.split("_", 1)[1]] = v
    return out

def cw_residuals(psr, CW):
    """Generate simple CW residuals for a libstempo pulsar."""
    toas = psr.toas()  # seconds
    ra, dec = psr.pos   # radians
    gwtheta, gwphi = CW["gw_theta"], CW["gw_phi"]
    inc, psi = CW["inclination"], CW["psi"]
    phase0, fgw = CW["phase0"], CW["f_gw"]
    h0 = 10 ** CW["log10_h"]

    # antenna patterns (simplified)
    fplus = 0.5 * (1 + np.cos(dec) ** 2) * np.cos(2 * (ra - gwphi))
    fcross = np.cos(dec) * np.sin(2 * (ra - gwphi))
    phase = 2 * np.pi * fgw * toas + phase0
    rplus = -0.5 * (1 + np.cos(inc) ** 2) * np.sin(2 * phase)
    rcross = np.cos(inc) * np.cos(2 * phase)
    return h0 / (2 * np.pi * fgw) * (fplus * rplus + fcross * rcross)

def add_white_noise(res, sigma):
    return res + np.random.normal(0, sigma, size=len(res))

def add_red_noise(res, A, gamma, toas):
    """Generate a simple red-noise realization."""
    n = len(toas)
    Tspan = toas.max() - toas.min()
    freqs = np.fft.rfftfreq(n, d=Tspan / n)
    psd = (A**2 / 12.0 / np.pi**2) * (freqs / 1e-8)**(-gamma)
    psd[0] = 0
    wn = np.random.normal(0, 1, len(freqs)) + 1j * np.random.normal(0, 1, len(freqs))
    rn = np.fft.irfft(wn * np.sqrt(psd / 2.0))
    return res + rn[:n]

# ===============================================================
# MAIN
# ===============================================================
def main():
    noise_dict = load_noise_dict(NOISE_FILE)
    par_files = sorted([f for f in os.listdir(PAR_DIR) if f.endswith(".par")])
    print(f"Found {len(par_files)} pulsars")

    for par in par_files:
        name = par.replace(".par", "")
        par_path = os.path.join(PAR_DIR, par)
        tim_path = os.path.join(TIM_DIR, f"{name}.tim")
        if not os.path.isfile(tim_path):
            print(f"Skipping {name}: no .tim file")
            continue

        print(f"Injecting CW + GWB + noise into {name}")
        psr = T.tempopulsar(par_path, tim_path)
        toas = psr.toas()
        res = cw_residuals(psr, CW)

        # Add GWB realization
        A_gwb = 10 ** CW["log10_A_gwb"]
        res += np.random.normal(0, A_gwb, size=len(toas))

        # Add EFAC/EQUAD and red noise
        npars = get_noise_params(name, noise_dict)
        efac = npars.get("efac", 1.0)
        log10_tnequad = npars.get("log10_tnequad", -np.inf)
        equad = 0.0 if not np.isfinite(log10_tnequad) else 10 ** log10_tnequad
        sigma = np.sqrt((efac * psr.toaerrs) ** 2 + equad ** 2)
        res = add_white_noise(res, np.mean(sigma))

        if "red_noise_log10_A" in npars and "red_noise_gamma" in npars:
            res = add_red_noise(res, 10 ** npars["red_noise_log10_A"],
                                npars["red_noise_gamma"], toas)

        # Save residuals and new files
        np.savetxt(os.path.join(OUT_DIR, f"{name}_residuals.txt"), res)
        psr.residuals(res)
        psr.savetim(os.path.join(OUT_DIR, f"{name}_sim.tim"))
        psr.savepar(os.path.join(OUT_DIR, f"{name}_sim.par"))

    print(f"\nSimulation complete. Output saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
