#!/usr/bin/env python3
"""
Simulate CW-injected libstempo dataset for IPTA-MDC2 (dataset_2)
matching QuickCW conventions, with rotation switch for psi/phase0.

Outputs:
  /sim_dataset2_normal/*.tim
  /sim_dataset2_rotated/*.tim
  plus corresponding Enterprise-compatible PKLs
"""
import os
os.environ["ASTROPY_USE_SYSTEM_IERS"] = "1"
os.environ["IERS_AUTO_URL"] = ""

from pathlib import Path
import numpy as np, pickle, libstempo as T
from enterprise.pulsar import Pulsar
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------- Paths ----------
BASE = Path("/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2")
PAR_DIR, TIM_DIR = BASE / "par", BASE / "tim"

OUT_BASE = Path("/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data")
OUT_NORMAL = OUT_BASE / "sim_dataset2_normal"
OUT_ROT = OUT_BASE / "sim_dataset2_rotated"
for d in [OUT_NORMAL, OUT_ROT]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- Base CW parameters (dataset2) ----------
CW_BASE = dict(
    chirp_mass=4.3e9,
    distance=75.4,
    fgw=3.7e-9,
    gw_phi=3.3335788713091694,
    gw_theta=0.6387905062299246,
    inclination=0.8412486994612669,
    log10_h=-13.668773493298787,
    phase0=0.24434609527920614,
    psi=1.1187560505283651,
)

# ---------- Rotation switch ----------
def rotate_params(CW, rotate=False):
    """Return new CW dict with psi→psi+π/2 and phase0→phase0+π if rotate=True."""
    new = CW.copy()
    if rotate:
        new["psi"] = (new["psi"] + np.pi/2.0) % np.pi
        new["phase0"] = (new["phase0"] + np.pi) % (2*np.pi)
    return new

# ---------- Helpers ----------
def get_radec(psr):
    """Return RA, DEC [rad] handling RAJ/DECJ or ELONG/ELAT."""
    pars_upper = [p.upper() for p in psr.pars()]
    if "ELONG" in pars_upper and "ELAT" in pars_upper:
        elon = float(psr["ELONG"].val)
        elat = float(psr["ELAT"].val)
        c = SkyCoord(elon*u.rad, elat*u.rad, frame="barycentrictrueecliptic")
        return float(c.icrs.ra.rad), float(c.icrs.dec.rad)
    for racand, deccand in [("RAJ", "DECJ"), ("RA", "DEC")]:
        if racand in pars_upper and deccand in pars_upper:
            pra = float(psr[psr.pars()[pars_upper.index(racand)]].val)
            pdec = float(psr[psr.pars()[pars_upper.index(deccand)]].val)
            return pra, pdec
    raise KeyError(f"No RA/DEC or ELONG/ELAT in {psr.name}")

def antenna_factors(p_ra, p_dec, g_ra, g_dec, psi):
    """Compute F+ and Fx antenna patterns."""
    ph = np.array([np.cos(p_dec)*np.cos(p_ra),
                   np.cos(p_dec)*np.sin(p_ra),
                   np.sin(p_dec)])
    kh = np.array([np.cos(g_dec)*np.cos(g_ra),
                   np.cos(g_dec)*np.sin(g_ra),
                   np.sin(g_dec)])
    x = np.array([-np.sin(g_ra), np.cos(g_ra), 0.0])
    y = np.cross(kh, x)
    c2, s2 = np.cos(2*psi), np.sin(2*psi)
    eplus  = np.outer(x, x) - np.outer(y, y)
    ecross = np.outer(x, y) + np.outer(y, x)
    eplus, ecross = c2*eplus + s2*ecross, -s2*eplus + c2*ecross
    denom = 1.0 - ph.dot(kh)
    Fp = 0.5 * (ph @ eplus @ ph) / denom
    Fx = 0.5 * (ph @ ecross @ ph) / denom
    return Fp, Fx

# ---------- Injection function ----------
def cw_residuals(psr, CW):
    """Generate CW residuals (Earth + pulsar term)."""
    fgw = CW["fgw"]
    iota, psi, phi0 = CW["inclination"], CW["psi"], CW["phase0"]
    g_ra, g_dec = CW["gw_phi"], np.pi/2.0 - CW["gw_theta"]
    pra, pdec = get_radec(psr)
    Fp, Fx = antenna_factors(pra, pdec, g_ra, g_dec, psi)

    toas = psr.stoas.copy().astype(float) * 86400.0  # s
    tref = psr["PEPOCH"].val * 86400.0               # QuickCW uses PEPOCH as tref

    cth = np.cos(iota)
    # Earth term
    phiE = 2*np.pi*fgw*(toas - tref) + phi0
    splus_E  = 0.5*(1 + cth**2)*np.cos(phiE)
    scross_E = cth*np.sin(phiE)

    # Pulsar term (depends on distance via PX)
    try:
        pdist = 1.0 / psr["PX"].val  # kpc
    except Exception:
        pdist = 1.0  # fallback
    tau = pdist * 3.086e19 / (3e8)  # kpc→m→s
    phiP = 2*np.pi*fgw*(toas - tref - tau) + phi0
    splus_P  = 0.5*(1 + cth**2)*np.cos(phiP)
    scross_P = cth*np.sin(phiP)

    h0 = 10**CW["log10_h"]
    res = (h0 / (2*np.pi*fgw)) * (Fp*(splus_E - splus_P) + Fx*(scross_E - scross_P))
    return res

# ---------- Main loop ----------
def build_dataset(rotate=False):
    CW = rotate_params(CW_BASE, rotate)
    outdir = OUT_ROT if rotate else OUT_NORMAL
    pkl_path = outdir / ("G2D2_sim_injected_rot.pkl" if rotate else "G2D2_sim_injected.pkl")

    pulsars = []
    for par in sorted(PAR_DIR.glob("*.par")):
        tim = TIM_DIR / (par.stem + ".tim")
        psr = T.tempopulsar(str(par), str(tim))
        res = cw_residuals(psr, CW)
        psr.stoas[:] += res / 86400.0
        psr.savetim(str(outdir / f"{psr.name}_CW_new{'_rot' if rotate else ''}.tim"))
        rms = np.std(res)*1e6
        print(f"{psr.name:12s} [{'ROT' if rotate else 'NORM'}] RMS={rms:.3f} µs")
        # build Enterprise object
        psr_ent = Pulsar(str(par), str(outdir / f"{psr.name}_CW_new{'_rot' if rotate else ''}.tim"), timing_package="tempo2")
        pulsars.append(psr_ent)

    with open(pkl_path, "wb") as f:
        pickle.dump(pulsars, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved {len(pulsars)} {'rotated' if rotate else 'normal'} pulsars → {pkl_path}")

# ---------- Execute ----------
print("Building normal (unrotated) dataset...")
build_dataset(rotate=False)
print("\nBuilding rotated dataset (ψ→ψ+π/2, Φ₀→Φ₀+π)...")
build_dataset(rotate=True)
print("\n✅ Done.")
