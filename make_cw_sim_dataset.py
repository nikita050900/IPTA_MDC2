#!/usr/bin/env python3
"""
Create simulated PTA datasets (native and rotated) with deterministic CW injection + GWB + pulsar noise.
Purpose: verify QuickCW ψ–Φ₀ phase-offset behavior by replicating IPTA MDC2 dataset 2.
"""

import os, json, pickle
import numpy as np
from astropy import units as u
from pta_replicator import deterministic, red_noise, white_noise

# ================================================================
# USER CONFIG
# ================================================================
NOISE_JSON = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/noise_files/fit_psr_noise_dataset2.json"
PSR_PKL    = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/psr_objects/G2D2_IPTA_MDC2_all_pulsars.pkl"

BASE_OUT = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/sim_dataset2"
os.makedirs(BASE_OUT, exist_ok=True)

# ------------------------------------------------
# CW PARAMETERS (MDC2 G2D2)
# ------------------------------------------------
CW_native = {
    "chirp_mass": 4.3e9,                     # Msun
    "distance": 75.4,                        # Mpc
    "f_gw": 3.7e-9,                          # Hz
    "gw_phi": 3.3335788713091694,            # rad
    "gw_theta": 0.6387905062299246,          # colatitude [rad]
    "inclination": 0.8412486994612669,       # rad
    "phase0": 0.24434609527920614,           # rad
    "psi": 1.1187560505283651,               # rad
    "log10_A_gwb": -15.070581074285707,      # stochastic background amplitude
}

# Rotated equivalent: ψ→ψ+π/2, Φ₀→Φ₀+π
CW_rot = dict(CW_native)
CW_rot["phase0"] = (CW_native["phase0"] + np.pi) % (2 * np.pi)
CW_rot["psi"]    = (CW_native["psi"] + 0.5 * np.pi) % np.pi  # ψ periodic in π

# Toggle via environment (0=native, 1=rotated)
USE_ROTATED = bool(int(os.getenv("USE_ROTATED", "0")))
CW = CW_rot if USE_ROTATED else CW_native
OUTDIR = os.path.join(BASE_OUT, "rotated" if USE_ROTATED else "native")
os.makedirs(OUTDIR, exist_ok=True)
PKL_OUT = os.path.join(OUTDIR, "G2D2_IPTA_MDC2_all_pulsars.pkl")

# ================================================================
# HELPERS
# ================================================================
def load_noise_dict(path):
    with open(path, "r") as f:
        return json.load(f)

def get_noise_params(name, noise_dict):
    """Extract per-pulsar EFAC/EQUAD/red-noise parameters."""
    out = {}
    for k, v in noise_dict.items():
        if k.startswith(name + "_"):
            out[k.split("_", 1)[1]] = v
    return out

# ================================================================
# LOAD PULSARS (from TEMPO2-built pickle)
# ================================================================
with open(PSR_PKL, "rb") as f:
    psrs = pickle.load(f)
print(f"Loaded {len(psrs)} pulsars from {PSR_PKL}")

noise_dict = load_noise_dict(NOISE_JSON)

# ================================================================
# INJECT CW + GWB + NOISE
# ================================================================
for psr in psrs:
    pname = psr.name
    print(f"Injecting CW + GWB + noise into {pname}")

    # Start from zero residuals
    if hasattr(psr, "residuals") and hasattr(psr.residuals, "value"):
        psr.residuals.value[:] = 0.0

    # --- Continuous GW injection (amplitude computed internally) ---
    cw_kwargs = dict(
        gwtheta=CW["gw_theta"],
        gwphi=CW["gw_phi"],
        mc=CW["chirp_mass"],
        dist=CW["distance"],
        fgw=CW["f_gw"],
        phase0=CW["phase0"],
        psi=CW["psi"],
        inc=CW["inclination"],
        psrTerm=True,
        evolve=True,
    )

    if hasattr(deterministic, "add_cgw"):
        deterministic.add_cgw(psr, **cw_kwargs)
    else:
        raise RuntimeError("pta_replicator version lacks add_cgw(); cannot inject CW.")

    # --- Stochastic GWB injection (to match dataset2) ---
    LOG10_A_GWB = CW["log10_A_gwb"]
    GWB_GAMMA = 13 / 3
    if hasattr(deterministic, "add_gwb"):
        try:
            deterministic.add_gwb(psr, A=10**LOG10_A_GWB, gamma=GWB_GAMMA)
        except TypeError:
            deterministic.add_gwb(psr, 10**LOG10_A_GWB, GWB_GAMMA)

    # --- White noise (EFAC/EQUAD) ---
    npars = get_noise_params(pname, noise_dict)
    efac = npars.get("efac", 1.0)
    log10_tnequad = npars.get("log10_tnequad", -np.inf)

    if hasattr(white_noise, "apply_white_noise"):
        white_noise.apply_white_noise(psr, efac=efac, log10_tnequad=log10_tnequad)
    else:
        errs_s = psr.toas.get_errors().to(u.s).value
        equad_s = 0.0 if not np.isfinite(log10_tnequad) else 10.0 ** log10_tnequad
        new_errs_s = np.sqrt((efac * errs_s)**2 + equad_s**2)
        psr.toas.table["error"] = (new_errs_s * u.s)

    # --- Red noise realization (if defined) ---
    if "red_noise_log10_A" in npars and "red_noise_gamma" in npars and hasattr(red_noise, "add_red_noise"):
        psr = red_noise.add_red_noise(
            psr,
            A=10.0 ** npars["red_noise_log10_A"],
            gamma=npars["red_noise_gamma"],
        )

    # Save diagnostic residuals per pulsar
    np.savetxt(os.path.join(OUTDIR, f"{pname}_residuals.txt"), psr.residuals.value)

# ================================================================
# SAVE OUTPUTS
# ================================================================
with open(PKL_OUT, "wb") as f:
    pickle.dump(psrs, f, protocol=pickle.HIGHEST_PROTOCOL)

meta = dict(
    CW_params=CW,
    USE_ROTATED=USE_ROTATED,
    NOISE_JSON=NOISE_JSON,
    n_pulsars=len(psrs),
    source_pickle=PSR_PKL,
)
with open(os.path.join(OUTDIR, "injection_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nSimulation complete.")
print(f"  USE_ROTATED = {USE_ROTATED}")
print(f"  ψ = {CW['psi']:.4f} rad, Φ₀ = {CW['phase0']:.4f} rad")
print(f"  Output pickle: {PKL_OUT}")
