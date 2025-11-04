#!/usr/bin/env python3
"""
Create simulated PTA datasets (native and rotated) with deterministic CW injection + GWB + pulsar noise.
Purpose: verify QuickCW ψ–Φ₀ phase-offset behavior by replicating IPTA MDC2 dataset 2,
using existing Tempo2-built Pulsar pickles with a shim for pta_replicator compatibility.
"""

import os, json, pickle, types, sys, traceback
import numpy as np
from astropy import units as u
from pta_replicator import deterministic, red_noise, white_noise

# ================================================================
# USER CONFIG
# ================================================================
NOISE_JSON = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/noise_files/fit_psr_noise_dataset2.json"
PSR_PKL    = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/psr_objects/G2D2_IPTA_MDC2_all_pulsars.pkl"
BASE_OUT   = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/sim_dataset2"
os.makedirs(BASE_OUT, exist_ok=True)

# ------------------------------------------------
# CW PARAMETERS (MDC2 G2D2)
# ------------------------------------------------
CW_native = {
    "chirp_mass": 4.3e9,  # Msun
    "distance": 75.4,  # Mpc
    "f_gw": 3.7e-9,  # Hz
    "gw_phi": 3.3335788713091694,  # rad
    "gw_theta": 0.6387905062299246,  # colatitude [rad]
    "inclination": 0.8412486994612669,  # rad
    "phase0": 0.24434609527920614,  # rad
    "psi": 1.1187560505283651,  # rad
    "log10_A_gwb": -15.070581074285707,  # stochastic background amplitude
}

# Rotated equivalent: ψ→ψ+π/2, Φ₀→Φ₀+π
CW_rot = dict(CW_native)
CW_rot["phase0"] = (CW_native["phase0"] + np.pi) % (2 * np.pi)
CW_rot["psi"] = (CW_native["psi"] + 0.5 * np.pi) % np.pi  # ψ periodic in π


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


def install_replicator_shim(psr):
    """Adds update/remove/get_added_signals methods expected by pta_replicator."""
    if hasattr(psr, "update_added_signals"):
        return  # already patched
    psr._added_signals = {}

    def _as_numeric(arr):
        """Convert residual container to a plain float array."""
        if hasattr(arr, "value"):
            return arr.value
        arr = np.asarray(arr)
        if arr.dtype == object:
            arr = arr.astype(float)
        return arr

    def update_added_signals(self, key, residual, **kwargs):
        # Handle dicts returned by pta_replicator.add_cgw
        if isinstance(residual, dict):
            if "residuals" in residual:
                residual = residual["residuals"]
            elif "signal" in residual:
                residual = residual["signal"]
            else:
                raise TypeError(f"Unexpected dict keys in residual: {residual.keys()}")

        r = np.asarray(residual, dtype=float)
        base = _as_numeric(self.residuals)
        base += r

        if hasattr(self.residuals, "value"):
            self.residuals.value[:] = base
        else:
            self.residuals[:] = base

        self._added_signals[key] = r


    def remove_added_signals(self, key):
        r = self._added_signals.pop(key, None)
        if r is not None:
            base = _as_numeric(self.residuals)
            base -= r
            if hasattr(self.residuals, "value"):
                self.residuals.value[:] = base
            else:
                self.residuals[:] = base

    def get_added_signals(self):
        return dict(self._added_signals)

    psr.update_added_signals = types.MethodType(update_added_signals, psr)
    psr.remove_added_signals = types.MethodType(remove_added_signals, psr)
    psr.get_added_signals = types.MethodType(get_added_signals, psr)


# ================================================================
# MAIN
# ================================================================
def main():
    use_rotated_env = os.getenv("USE_ROTATED", "0")
    print(f"USE_ROTATED={use_rotated_env}")
    sys.stdout.flush()

    try:
        USE_ROTATED = bool(int(use_rotated_env))
    except Exception:
        print("Invalid USE_ROTATED. Expected 0 or 1.", file=sys.stderr)
        sys.exit(2)

    if not os.path.isfile(PSR_PKL):
        print(f"ERROR: PSR_PKL not found: {PSR_PKL}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(NOISE_JSON):
        print(f"ERROR: NOISE_JSON not found: {NOISE_JSON}", file=sys.stderr)
        sys.exit(2)

    CW = CW_rot if USE_ROTATED else CW_native
    OUTDIR = os.path.join(BASE_OUT, "rotated" if USE_ROTATED else "native")
    os.makedirs(OUTDIR, exist_ok=True)
    PKL_OUT = os.path.join(OUTDIR, "G2D2_IPTA_MDC2_all_pulsars.pkl")

    with open(PSR_PKL, "rb") as f:
        psrs = pickle.load(f)
    print(f"Loaded {len(psrs)} pulsars from {PSR_PKL}")
    sys.stdout.flush()

    noise_dict = load_noise_dict(NOISE_JSON)

    # Inject signals into each pulsar
    for psr in psrs:
        pname = psr.name
        print(f"Injecting CW + GWB + noise into {pname}")
        sys.stdout.flush()

        install_replicator_shim(psr)

        # Reset residuals
        if hasattr(psr, "residuals") and hasattr(psr.residuals, "value"):
            psr.residuals.value[:] = 0.0
        else:
            psr.residuals[:] = 0.0

        # --- CW injection ---
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
        deterministic.add_cgw(psr, **cw_kwargs)

        # --- GWB injection ---
        LOG10_A_GWB = CW["log10_A_gwb"]
        GWB_GAMMA = 13 / 3
        if hasattr(deterministic, "add_gwb"):
            try:
                deterministic.add_gwb(psr, A=10**LOG10_A_GWB, gamma=GWB_GAMMA)
            except TypeError:
                deterministic.add_gwb(psr, 10**LOG10_A_GWB, GWB_GAMMA)

        # --- White noise ---
        npars = get_noise_params(pname, noise_dict)
        efac = npars.get("efac", 1.0)
        log10_tnequad = npars.get("log10_tnequad", -np.inf)

        if hasattr(white_noise, "apply_white_noise"):
            white_noise.apply_white_noise(psr, efac=efac, log10_tnequad=log10_tnequad)
        else:
            errs_s = psr.toas.get_errors().to(u.s).value
            equad_s = 0.0 if not np.isfinite(log10_tnequad) else 10.0 ** log10_tnequad
            new_errs_s = np.sqrt((efac * errs_s) ** 2 + equad_s ** 2)
            psr.toas.table["error"] = (new_errs_s * u.s)

        # --- Red noise ---
        if (
            "red_noise_log10_A" in npars
            and "red_noise_gamma" in npars
            and hasattr(red_noise, "add_red_noise")
        ):
            psr = red_noise.add_red_noise(
                psr,
                A=10.0 ** npars["red_noise_log10_A"],
                gamma=npars["red_noise_gamma"],
            )

        # Save residuals
        np.savetxt(os.path.join(OUTDIR, f"{pname}_residuals.txt"), np.asarray(psr.residuals, dtype=float))

    # Save dataset pickle
    with open(PKL_OUT, "wb") as f:
        pickle.dump(psrs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
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
    sys.stdout.flush()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
