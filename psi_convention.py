#!/usr/bin/env python3
"""
QUICKCW SIDE
Measure the fixed polarization-basis offset δ between libstempo and QuickCW,
and translate the libstempo injection parameters into the QuickCW/Enterprise convention.

Run this in your normal QuickCW environment (not inside Singularity).
"""

import numpy as np
from math import pi

# --------------------------------------------------------------------
# USER CONFIGURATION — fill in your values here
# --------------------------------------------------------------------
gwtheta = 0.6387905062299246
gwphi   = 3.3335788713091694
psi_inj = 1.1187560505283651          # ψ injected in libstempo
phi0_inj = 0.24434609527920614        # Φ₀ injected in libstempo
phi0_rec_mean = 4.476207              # recovered Φ₀ mean from QuickCW run (you provided this)
# paths to files copied from libstempo container
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

# ---- compute QuickCW patterns ----
m_qc, n_qc, Om_qc = triad_quickcw(gwtheta, gwphi)
Fp_qc, Fx_qc = antenna_patterns(m_qc, n_qc, Om_qc, phat)

# ---- measure delta ----
delta = estimate_delta(Fp_ls, Fx_ls, Fp_qc, Fx_qc)
print(f"\nPolarization basis offset δ = {delta:.6f} rad = {delta*180/pi:.6f} deg")

# ---- translate injection ----
psi_inj_qc = wrap_pi(psi_inj + delta)
phi0A = wrap_2pi(phi0_inj)
phi0B = wrap_2pi(phi0_inj + np.pi)
phi0_inj_qc = phi0A if circ_diff(phi0A, phi0_rec_mean) < circ_diff(phi0B, phi0_rec_mean) else phi0B
choice = "A (no +π)" if phi0_inj_qc==phi0A else "B (+π)"

print(f"\nTranslated injection parameters (QuickCW/Enterprise convention):")
print(f"  ψ_inj_qc  = {psi_inj_qc:.6f} rad")
print(f"  Φ₀_inj_qc = {phi0_inj_qc:.6f} rad   [{choice}]")

print("\nUse these translated values as the 'true' injection for overlaying in your QuickCW/Enterprise corner plots.")
print("If you run this for another pulsar and get the same δ, it is a fixed convention offset between libstempo and QuickCW.")
