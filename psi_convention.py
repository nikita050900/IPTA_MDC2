#!/usr/bin/env python3
"""
QUICKCW SIDE
Check the polarization-basis offset δ between libstempo and QuickCW
and verify ψ–Φ₀ consistency using the identified combination χ.
"""

import numpy as np
from math import pi

# --------------------------------------------------------------------
# USER INPUTS — edit only these values
# --------------------------------------------------------------------
gwtheta = 0.6387905062299246
gwphi   = 3.3335788713091694
psi_inj = 1.1187560505283651          # injected ψ (libstempo)
phi0_inj = 0.24434609527920614        # injected Φ₀ (libstempo)
phi0_rec_mean = 4.476207              # recovered Φ₀ mean from your chain
psi_rec_mean  = 1.390576              # recovered ψ mean from your chain

# paths to saved libstempo data
F_lib_file = "F_libstempo.npy"
phat_file  = "phat_libstempo.npy"
# --------------------------------------------------------------------


# ---- helper functions ----
def triad_quickcw(theta, phi):
    sin_gwtheta = np.sqrt(1 - np.cos(theta)**2)
    sin_gwphi = np.sin(phi)
    cos_gwphi = np.cos(phi)
    m  = np.array([ sin_gwphi, -cos_gwphi, 0.0 ])
    n  = np.array([-np.cos(theta)*cos_gwphi,
                   -np.cos(theta)*sin_gwphi,
                    sin_gwtheta ])
    Om = np.array([-sin_gwtheta*cos_gwphi,
                   -sin_gwtheta*sin_gwphi,
                   -np.cos(theta)])
    return m, n, Om

def antenna_patterns(m, n, Om, phat):
    cosMu = -np.dot(Om, phat)
    denom = 1.0 - cosMu
    Fp = 0.5 * ((m@phat)**2 - (n@phat)**2) / denom
    Fx =      ((m@phat) * (n@phat)) / denom
    return Fp, Fx

def estimate_delta(Fp_ls, Fx_ls, Fp_qc, Fx_qc):
    num = Fp_qc*Fx_ls - Fx_qc*Fp_ls
    den = Fp_qc*Fp_ls + Fx_qc*Fx_ls
    return 0.5*np.arctan2(num, den)

def wrap_pi(x):  return x % np.pi
def wrap_2pi(x): return x % (2*np.pi)
def circ_diff(a,b):
    d = (a-b+pi) % (2*pi) - pi
    return abs(d)


# ---- load libstempo data ----
Fp_ls, Fx_ls = np.load(F_lib_file)
phat = np.load(phat_file)

# ---- compute QuickCW antenna patterns ----
m_qc, n_qc, Om_qc = triad_quickcw(gwtheta, gwphi)
Fp_qc, Fx_qc = antenna_patterns(m_qc, n_qc, Om_qc, phat)

# ---- measure δ ----
delta = estimate_delta(Fp_ls, Fx_ls, Fp_qc, Fx_qc)
print(f"\nPolarization basis offset δ = {delta:.6f} rad = {delta*180/pi:.6f} deg")

# ---- translate injection (libstempo → QuickCW) ----
psi_inj_qc = wrap_pi(psi_inj + delta)
phi0A = wrap_2pi(phi0_inj)
phi0B = wrap_2pi(phi0_inj + np.pi)
phi0_inj_qc = phi0A if circ_diff(phi0A, phi0_rec_mean) < circ_diff(phi0B, phi0_rec_mean) else phi0B
choice = "A (no +π)" if phi0_inj_qc==phi0A else "B (+π)"

print(f"\nTranslated injection parameters (QuickCW/Enterprise convention):")
print(f"  ψ_inj_qc  = {psi_inj_qc:.6f} rad")
print(f"  Φ₀_inj_qc = {phi0_inj_qc:.6f} rad   [{choice}]")

# ====================================================================
# === χ CHECK: (2ψ - φ_F_eff) - 2Φ₀  =================================
# ====================================================================
phiF_eff = float(np.arctan2(Fx_qc, -Fp_qc))  # antenna phase for one representative pulsar

chi_inj = (2*psi_inj_qc - phiF_eff - 2*phi0_inj_qc) % (2*np.pi)
chi_rec = (2*psi_rec_mean - phiF_eff - 2*phi0_rec_mean) % (2*np.pi)
dchi = ((chi_rec - chi_inj + np.pi) % (2*np.pi)) - np.pi

print("\n--- χ consistency check ---")
print(f"φF_eff = {phiF_eff:.6f} rad")
print(f"χ_injected  = {chi_inj:.6f} rad")
print(f"χ_recovered = {chi_rec:.6f} rad")
print(f"Δχ = {dchi:.6f} rad")

if abs(dchi) < 0.3:
    print("✅ χ matches → ψ–Φ₀ combination consistent with injected waveform.\n")
else:
    print("⚠️  χ differs → check tref or pulsar-term conventions.\n")
# ====================================================================

print("Use ψ_inj_qc and Φ₀_inj_qc as the true injection values in your corner plots.")
