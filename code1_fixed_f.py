#!/usr/bin/env python3
import os
import glob
import numpy as np
import libstempo as lt

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
SOLAR2S = 4.925490947e-6
MPC2S   = 1.02927125e14        # 1 Mpc in seconds
KPC2S   = 3.085677581e19 / 2.99792458e8  # kpc in seconds
EPS     = np.deg2rad(23.439291111)

def ecl_to_equ(elong, elat):
    lam, beta = elong, elat
    sin_delta = np.sin(beta)*np.cos(EPS) + np.cos(beta)*np.sin(EPS)*np.sin(lam)
    delta = np.arcsin(sin_delta)
    y = np.sin(lam)*np.cos(EPS) - np.tan(beta)*np.sin(EPS)
    x = np.cos(lam)
    alpha = np.arctan2(y, x) % (2*np.pi)
    return alpha, delta

# ---------------------------------------------------------
# PULSAR DISTANCE FROM PAR FILE (PX)
# ---------------------------------------------------------
def get_pulsar_distance_kpc(psr):
    if "PX" in psr.pars():
        px = psr["PX"].val
        if px != 0:
            return 1.0 / px          # kpc
    # fallback if no PX
    return 1.0                       # assume 1 kpc fallback

# ---------------------------------------------------------
# FIXED-FREQUENCY CW MODEL WITH EARTH + PULSAR TERM
# ---------------------------------------------------------
def cw_residuals_fixedfreq(psr, gwtheta, gwphi, mc, dist_mpc, fgw,
                           phase0, psi, inc, tref):

    # convert units
    mc_seconds   = mc * SOLAR2S
    dist_seconds = dist_mpc * MPC2S
    omega = 2 * np.pi * fgw

    # GW sky geometry
    cosT, sinT = np.cos(gwtheta), np.sin(gwtheta)
    cosP, sinP = np.cos(gwphi),   np.sin(gwphi)

    m = np.array([ sinP,        -cosP,        0.0 ])
    n = np.array([-cosT*cosP,   -cosT*sinP,   sinT])
    omhat = np.array([-sinT*cosP, -sinT*sinP, -cosT])

    # pulsar sky position
    if ("RAJ" in psr.pars()) and ("DECJ" in psr.pars()):
        ra, dec = psr["RAJ"].val, psr["DECJ"].val
    else:
        ra, dec = ecl_to_equ(psr["ELONG"].val, psr["ELAT"].val)

    ptheta = np.pi/2 - dec
    pphi   = ra
    phat = np.array([
        np.sin(ptheta)*np.cos(pphi),
        np.sin(ptheta)*np.sin(pphi),
        np.cos(ptheta)
    ])

    # antenna patterns
    denom = (1 + np.dot(omhat, phat))
    fplus  = 0.5 * ((m@phat)**2 - (n@phat)**2) / denom
    fcross =      ((m@phat) * (n@phat))        / denom

    cosMu = -np.dot(omhat, phat)

    # pulsar distance → time delay
    dp_kpc = get_pulsar_distance_kpc(psr)
    dp_sec = dp_kpc * KPC2S
    tau_p = dp_sec * (1 - cosMu)      # pulsar time delay

    # TOAs
    toas_sec = psr.toas() * 86400.0
    tE = toas_sec - tref
    tP = tE - tau_p

    # Earth and pulsar term phases
    phiE = phase0 + omega * tE
    phiP = phase0 + omega * tP

    sin2i = 0.5*(3 + np.cos(2*inc))
    cosi  = 2*np.cos(inc)

    # strain amplitude (non-evolving)
    amp = mc_seconds**(5/3) / dist_seconds / (omega)**(1/3)

    h_plus_E  = amp * ( sin2i*np.sin(2*phiE)*np.cos(2*psi) + cosi*np.cos(2*phiE)*np.sin(2*psi) )
    h_cross_E = amp * (-sin2i*np.sin(2*phiE)*np.sin(2*psi) + cosi*np.cos(2*phiE)*np.cos(2*psi) )

    h_plus_P  = amp * ( sin2i*np.sin(2*phiP)*np.cos(2*psi) + cosi*np.cos(2*phiP)*np.sin(2*psi) )
    h_cross_P = amp * (-sin2i*np.sin(2*phiP)*np.sin(2*psi) + cosi*np.cos(2*phiP)*np.cos(2*psi) )

    # TOTAL RESIDUAL = projection × (pulsar term − Earth term)
    return fplus*(h_plus_P - h_plus_E) + fcross*(h_cross_P - h_cross_E)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
par_dir = "/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2/par"
tim_dir = "/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2/tim"

out_clean = "/scratch/na00078/projects/IPTA_MDC2/clean_noCW"
out_reinj = "/scratch/na00078/projects/IPTA_MDC2/reinj_correctedCW"
os.makedirs(out_clean, exist_ok=True)
os.makedirs(out_reinj, exist_ok=True)

# ---------------------------------------------------------
# ORIGINAL SIGNAL PARAMETERS
# ---------------------------------------------------------
gwtheta = 0.6387905062299246
gwphi   = 3.3335788713091694
mc      = 4.3e9            # solar masses
dist    = 75.4             # Mpc
fgw     = 3.7e-9
phase0  = 0.24434609527920614
psi_old = 1.1187560505283651
psi_new = 1.1187560505283651   # replace here
inc     = 0.8412486994612669
tref    = 55443.93364609394 * 86400   # seconds

# ---------------------------------------------------------
# STEP 1: REMOVE ORIGINAL CW
# ---------------------------------------------------------
pars = sorted(glob.glob(os.path.join(par_dir, "*.par")))
for p in pars:
    name = os.path.basename(p)[:-4]
    t = os.path.join(tim_dir, f"{name}.tim")

    psr = lt.tempopulsar(p, t)

    r = cw_residuals_fixedfreq(psr, gwtheta, gwphi, mc, dist,
                               fgw, phase0, psi_old, inc, tref)

    # Remove from residuals, NOT stoas
    psr.stoas[:] -= r / 86400.0


    psr.savetim(os.path.join(out_clean, f"{name}.tim"))
    os.system(f"cp {p} {os.path.join(out_clean, name+'.par')}")

print("Saved CW-cleaned files.")

# ---------------------------------------------------------
# STEP 2: REINJECT CORRECTED CW
# ---------------------------------------------------------
clean_pars = sorted(glob.glob(os.path.join(out_clean, "*.par")))
for p in clean_pars:
    name = os.path.basename(p)[:-4]
    t = os.path.join(out_clean, f"{name}.tim")

    psr = lt.tempopulsar(p, t)

    rnew = cw_residuals_fixedfreq(psr, gwtheta, gwphi, mc, dist,
                                  fgw, phase0, psi_new, inc, tref)

    psr.stoas[:] += rnew / 86400.0


    psr.savetim(os.path.join(out_reinj, f"{name}.tim"))
    os.system(f"cp {p} {os.path.join(out_reinj, name+'.par')}")

print("Saved dataset with corrected ψ.")
