#!/usr/bin/env python
"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)
[adapted with robust par/tim handling and offline PINT settings]
"""

import os
import glob
import json
import pickle
import argparse
import subprocess
import shutil

import numpy as np
np.seterr(all='raise')

import matplotlib.pyplot as plt  # noqa: F401

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter  # noqa: F401
from enterprise.signals import utils  # noqa: F401
from enterprise.signals import signal_base  # noqa: F401
from enterprise.signals import selections  # noqa: F401
from enterprise.signals.selections import Selection  # noqa: F401
from enterprise.signals import white_signals  # noqa: F401
from enterprise.signals import gp_signals  # noqa: F401
from enterprise.signals import deterministic_signals  # noqa: F401
import enterprise.constants as const  # noqa: F401

from enterprise_extensions import deterministic  # noqa: F401

import QuickCW.QuickCW_narrow as QuickCW
from QuickCW.QuickMCMCUtils import ChainParams

# Optional astronomy imports (kept to match your original script)
from astropy import units as u  # noqa: F401
from astropy.coordinates import SkyCoord  # noqa: F401
import healpy as hp  # noqa: F401
from healpy.newvisufunc import projview, newprojplot  # noqa: F401
from matplotlib.pyplot import figure  # noqa: F401
from matplotlib import patheffects  # noqa: F401
from matplotlib import text  # noqa: F401


from pint.models.timing_model import UnknownBinaryModel

# --------------------------------------------------------------------------------------
# Offline cache/env defaults so PINT uses your pre-warmed cache + DE436 (harmless no-ops
# if already set in your SLURM job script)
os.environ.setdefault("XDG_CACHE_HOME", "/gpfs20/scratch/na00078/.cache")
os.environ.setdefault("ASTROPY_CACHE_DIR", "/gpfs20/scratch/na00078/astropy_cache")
os.environ.setdefault("PINT_EPHEM", "DE436")

# --------------------------------------------------------------------------------------
# Data locations (dataset_2)
timdir2 = '/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2/tim/'
pardir2 = '/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2/par/'

# --------------------------------------------------------------------------------------
# Helpers for robust par handling

def run(cmd: str) -> bool:
    """Run a shell command; print stderr on failure."""
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 and r.stderr.strip():
        print(r.stderr.strip())
    return r.returncode == 0

def parse_parfile_fast(par_path: str) -> dict:
    """Minimal parser: key -> [values...] (first token per line as value)."""
    out = {}
    with open(par_path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            k = parts[0]
            v = parts[1] if len(parts) > 1 else ""
            out.setdefault(k, []).append(v)
    return out

def dedup_key_in_par(par_path: str, key="NE_SW"):
    """Keep the first line that starts with `key`, drop subsequent ones."""
    with open(par_path, "r") as f:
        lines = f.read().splitlines()
    seen = False
    new_lines = []
    for ln in lines:
        if ln.startswith(key):
            if seen:
                continue  # drop duplicate
            seen = True
        new_lines.append(ln)
    with open(par_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

# --------------------------------------------------------------------------------------
# Build pulsars from par/tim, with T2-aware backend selection and safe TCB->TDB conversion

# Collect and align by basename (avoid ordering mismatches)
pars = sorted(glob.glob(os.path.join(pardir2, "*.par")))
tims = sorted(glob.glob(os.path.join(timdir2, "*.tim")))

def base(p): return os.path.splitext(os.path.basename(p))[0]
par_by_name = {base(p): p for p in pars}
pairs = []
for t in tims:
    name = base(t)
    if name in par_by_name:
        pairs.append((par_by_name[name], t))
pairs.sort(key=lambda x: base(x[0]))  # stable order

psrs = []
for par, tim in pairs:
    print(par)
    print(tim, "\n")

    pf = parse_parfile_fast(par)
    if pf.get("NE_SW") and len(pf["NE_SW"]) > 1:
        print(f"removing extra NE_SW in {par}")
        dedup_key_in_par(par)
        pf = parse_parfile_fast(par)

    units  = (pf.get("UNITS", ["TDB"])[0] or "TDB").upper()
    binary = (pf.get("BINARY", [""])[0] or "").upper()

    # --- Tempo2-only backends we know PINT can't handle here ---
    if binary in {"T2", "DDH"}:
        why = "T2" if binary == "T2" else "DDH (PINT does not provide this component in your build)"
        print(f"BINARY {why} -> using tempo2 backend; skipping tcb2tdb / conversion.")
        try:
            epsr = Pulsar(par, tim, timing_package="tempo2", ephem="DE436", planets=False)
            ntoa = getattr(epsr, "ntoa", None) or len(getattr(epsr, "toas", []))
            if ntoa == 0:
                print(f"WARNING: Tempo2 returned 0 TOAs for {os.path.basename(par)} — skipping this pulsar.")
                continue
            psrs.append(epsr)
        except Exception as e:
            print(f"WARNING: Tempo2 failed for {os.path.basename(par)}: {e}\nSkipping this pulsar.")
        continue

    # --- For everything else, try PINT first (convert TCB→TDB only if we use PINT) ---
    par_for_pint = par
    if units == "TCB":
        print(f"Convert TCB → TDB for {par} (PINT path)")
        tmp_out = par[:-4] + "_tdb.par" if par.endswith(".par") else par + "_tdb"
        bak = par + ".tcb.bak"
        if not os.path.exists(bak):
            shutil.copy2(par, bak)
        if run(f"tcb2tdb {par} {tmp_out}"):
            par_for_pint = tmp_out
        else:
            print(f"WARNING: tcb2tdb failed for {par}; will try Tempo2 fallback.")

    try:
        epsr = Pulsar(par_for_pint, tim, timing_package="pint", ephem="DE436",
                      include_bipm=True, bipm_version="bipm2015")
        psrs.append(epsr)
    except UnknownBinaryModel as e:
        # Cleanly fall back to Tempo2 if PINT can't handle this binary model
        print(f"PINT cannot handle {os.path.basename(par)} ({e}); using Tempo2 backend.")
        try:
            epsr = Pulsar(par, tim, timing_package="tempo2", ephem="DE436", planets=False)
            ntoa = getattr(epsr, "ntoa", None) or len(getattr(epsr, "toas", []))
            if ntoa == 0:
                print(f"WARNING: Tempo2 returned 0 TOAs for {os.path.basename(par)} — skipping this pulsar.")
                continue
            psrs.append(epsr)
        except Exception as e2:
            print(f"WARNING: Tempo2 also failed for {os.path.basename(par)}: {e2}\nSkipping this pulsar.")

print(f"Loaded {len(psrs)} pulsars.")

####################################################################
#number of iterations (increase to 100 million - 1 billion for actual analysis)
N = 1e9

n_int_block = 10000 #number of iterations in a block (which has one shape update and the rest are projection updates)
save_every_n = 100000 #number of iterations between saving intermediate results (needs to be intiger multiple of n_int_block)
N_blocks = np.int64(N//n_int_block) #number of blocks to do
fisher_eig_downsample = 2000 #multiplier for how much less to do more expensive updates to fisher eigendirections for red noise and common parameters compared to diagonal elements

n_status_update = 100 #number of status update printouts (N/n_status_update needs to be an intiger multiple of n_int_block)
n_block_status_update = np.int64(N_blocks//n_status_update) #number of bllocks between status updates

assert N_blocks%n_status_update ==0 #or we won't print status updates
assert N%save_every_n == 0 #or we won't save a complete block
assert N%n_int_block == 0 #or we won't execute the right number of blocks

#Parallel tempering prameters
T_max = 3.
n_chain = 4

#make sure this points to your white noise dictionary
noisefile = '/scratch/na00078/projects/IPTA_MDC2/noise_files/fit_psr_noise_dataset2.json'

#make sure this points to the RN empirical distribution file you plan to use (or set to None to not use empirical distributions)
#rn_emp_dist_file = '/scratch/na00078/15yr_data/15yr_v1_1/rn_distr_v1p1.pkl'
rn_emp_dist_file = None

#file containing information about pulsar distances - None means use pulsar distances present in psr objects
#if not None psr objects must have zero distance and unit variance
#psr_dist_file = '/scratch/na00078/15yr_data/15yrCW/pulsar_distances_15yr.pkl'
psr_dist_file = None


##################################################################
'''
# Fixed directory path
save_dir = "/scratch/na00078/projects/IPTA_MDC2/h5_files"

parser = argparse.ArgumentParser(description="Run QuickMCMC narrow targeted search .")
parser.add_argument(
    "--save_filename",
    type=str,
    default="redo.h5",
    help="Name of the .h5 file to save (default: %(default)s)"
)
parser.add_argument(
    "--amplitude_prior",
    type=str,
    choices=["detection", "UL"],
    default="detection",
    help="Amplitude prior type: detection or UL (default: %(default)s)"
)
args = parser.parse_args()

# Combine fixed path with filename
savefile = os.path.join(save_dir, args.save_filename)
amplitude_prior = args.amplitude_prior

print(f"Saving to: {savefile}")
print(f"Using amplitude_prior: {amplitude_prior}")
'''
#####################################################################
#this is where results will be saved
savefile = '/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_detect.h5'
#savefile = None

###############
#targeted search params-LondonAdd
cos_gwtheta = np.cos(0.6387905062299246)
gwphi = 3.3335788713091694

#targeted freq
TargFreq = 3.7e-09
##############

#Setup and start MCMC
#object containing common parameters for the mcmc chain
chain_params = ChainParams(T_max,n_chain, n_block_status_update,
                           freq_bounds=np.array([TargFreq-.5e-21, TargFreq+.5e-21]), #prior bounds used on the GW frequency (a lower bound of np.nan is interpreted as 1/T_obs)
                           n_int_block=n_int_block, #number of iterations in a block (which has one shape update and the rest are projection updates)
                           save_every_n=save_every_n, #number of iterations between saving intermediate results (needs to be intiger multiple of n_int_block)
                           fisher_eig_downsample=fisher_eig_downsample, #multiplier for how much less to do more expensive updates to fisher eigendirections for red noise and common parameters compared to diagonal elements
                           rn_emp_dist_file=rn_emp_dist_file, #RN empirical distribution file to use (no empirical distribution jumps attempted if set to None)
                           savefile = savefile,#hdf5 file to save to, will not save at all if None
                           thin=10,  #thinning, i.e. save every `thin`th sample to file (increase to higher than one to keep file sizes small)
                           prior_draw_prob=0.2, de_prob=0.6, fisher_prob=0.3, #probability of different jump types
                           dist_jump_weight=0.2, rn_jump_weight=0.3, gwb_jump_weight=0.1, common_jump_weight=0.2, all_jump_weight=0.2, #probability of updating different groups of parameters
                           fix_rn=False, zero_rn=False, fix_gwb=False, zero_gwb=False, cos_gwtheta_bounds= [cos_gwtheta-1e-8,cos_gwtheta+1e-8], gwphi_bounds =[gwphi-1e-8,gwphi+1e-8]) #switches to turn off GWB or RN jumps and keep them fixed and to set them to practically zero (gamma=0.0, log10_A=-20)


pta,mcc = QuickCW.QuickCW(chain_params, psrs,
                                  amplitude_prior='detection', #specify amplitude prior to use - 'detection':uniform in log-amplitude, 'UL': uniform in amplitude
                                  psr_distance_file=psr_dist_file, #file to specify advanced (parallax+DM) pulsar distance priors, if None use regular Gaussian priors based on pulsar distances in pulsar objects
                                  noise_json=noisefile)

#Some parameters in chain_params can be updated later if needed
#mcc.chain_params.thin = 10

#Do the main MCMC iteration
mcc.advance_N_blocks(N_blocks)
