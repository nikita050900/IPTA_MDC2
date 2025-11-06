import numpy as np
from enterprise_extensions import deterministic
from QuickCW import const_mcmc as cm

# --- Injected CW parameters --- #
cos_gwtheta = np.cos(0.6387905062299246)
gwphi = 3.3335788713091694
cos_inc = 0.8412486994612669
log10_fgw = np.log10(3.7e-09)
log10_mc = np.log10(4.3e9)
log10_h = -13.668773493298787
phase0 = 0.24434609527920614
psi = 1.1187560505283651
tref = int(round(55443.93364609394 * 86400))
cm.tref = tref

# --- Define times --- #
t = np.linspace(0, 5 * 365.25 * 86400, 2000)  # 5 years in seconds
fgw = 10 ** log10_fgw
omega = 2 * np.pi * fgw

# --- Earth-term polarization factors (simplified) --- #
# h_plus and h_cross at Earth
h_plus = np.cos(2 * (omega * (t - tref) + phase0)) * np.cos(2 * psi)
h_cross = np.cos(2 * (omega * (t - tref) + phase0)) * np.sin(2 * psi)

# --- Compare ψ and ψ - π/2 --- #
psi_rot = psi - np.pi / 2
h_plus_rot = np.cos(2 * (omega * (t - tref) + phase0)) * np.cos(2 * psi_rot)
h_cross_rot = np.cos(2 * (omega * (t - tref) + phase0)) * np.sin(2 * psi_rot)

corr = np.corrcoef(h_plus + h_cross, h_plus_rot + h_cross_rot)[0, 1]
print(f"Correlation between ψ and ψ−π/2 signals = {corr:.3f}")

if corr > 0.9:
    print("ψ and ψ−π/2 produce identical waveforms → rotate ψ by −π/2 in QuickCW.")
elif corr < -0.9:
    print("ψ and ψ−π/2 are opposite sign → same orientation modulo phase flip.")
else:
    print("ψ rotation changes waveform as expected → QuickCW ψ convention likely correct.")
