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
import sys
from pathlib import Path

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

from pint.models.timing_model import UnknownBinaryModel  # noqa: F401
from pint.models.model_builder import parse_parfile
from pathlib import Path

# --------------------------------------------------------------------------------------
# PINT-compatible PSR loading (no external CLI dependencies in PATH)
# --------------------------------------------------------------------------------------

def _run_pint_module(module_name: str, args: list, logpath: Path) -> bool:
    """Run a PINT script as a Python module to avoid PATH issues."""
    logpath.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", module_name] + args
    with open(logpath, "w") as lg:
        try:
            subprocess.run(cmd, stdout=lg, stderr=lg, check=True, text=True)
            return True
        except subprocess.CalledProcessError:
            return False

def _dedupe_ne_sw(par_in: Path, par_out: Path) -> None:
    """Copy par_in -> par_out, removing duplicate NE_SW lines (keep first)."""
    lines = par_in.read_text().splitlines()
    seen = False
    out = []
    for ln in lines:
        if ln.strip().startswith("NE_SW"):
            if seen:
                continue
            seen = True
        out.append(ln)
    par_out.write_text("\n".join(out) + "\n")

def build_pint_psrs(base_dir: Path,
                    ephem: str = "DE436",
                    bipm_version: str = "BIPM2015"):
    """
    Returns a list of enterprise.Pulsar objects built with PINT.
    Writes intermediate/converted .par files to <base_dir>/par_pint
    and logs to <base_dir>/pint_fix_logs.
    """
    pardir = base_dir / "par"
    timdir = base_dir / "tim"
    outdir = base_dir / "par_pint"
    logs   = base_dir / "pint_fix_logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    psrs = []
    parfiles = sorted(pardir.glob("*.par"))
    print(f"Found {len(parfiles)} .par files in {pardir}")

    for par in parfiles:
        name = par.stem
        tim  = timdir / f"{name}.tim"
        if not tim.exists():
            print(f"❌ No .tim for {name}, skipping")
            continue

        # Start from a deduped copy (handles duplicate NE_SW)
        work = outdir / f"{name}.par"
        _dedupe_ne_sw(par, work)

        # Convert Binary model first (PINT can't parse T2)
        meta   = parse_parfile(str(work))
        binary = (meta.get("BINARY") or [""])[0].upper()
        if binary == "T2":
            pint_par = outdir / f"{name}_pint.par"
            if not pint_par.exists():
                ok = _run_pint_module(
                    "pint.scripts.t2binary2pint",
                    [str(work), str(pint_par)],
                    logs / f"{name}_t2binary2pint.log",
                )
                if not ok or not pint_par.exists():
                    print(f"❌ t2binary2pint failed for {name} (see {logs})")
                    continue
            work = pint_par
            meta = parse_parfile(str(work))  # refresh

        # Now handle timescale: TCB -> TDB (after binary conversion)
        units = (meta.get("UNITS") or [""])[0].upper()
        if units == "TCB":
            tdb = outdir / f"{Path(work).stem}_tdb.par"
            if not tdb.exists():
                ok = _run_pint_module(
                    "pint.scripts.tcb2tdb",
                    [str(work), str(tdb)],
                    logs / f"{name}_tcb2tdb.log",
                )
                if not ok or not tdb.exists():
                    print(f"❌ tcb2tdb failed for {name} (see {logs})")
                    continue
            work = tdb

        # Build Pulsar with PINT
        try:
            epsr = Pulsar(
                str(work), str(tim),
                timing_package="pint",
                ephem=ephem,
                include_bipm=True,
                bipm_version=bipm_version,
            )
            ntoa = len(epsr.toas)
            if ntoa > 0:
                psrs.append(epsr)
                print(f"✅ {name}: {ntoa} TOAs [par={Path(work).name}]")
            else:
                print(f"⚠️  {name}: 0 TOAs after fixes (skipping)")
        except Exception as e:
            print(f"❌ {name}: Pulsar build failed: {e}")

    print(f"\nSuccessfully loaded {len(psrs)} pulsars with PINT")
    return psrs

# ------------------------------------------------------------------------------
# Build pulsars (dataset_2 as requested)
dataset_dir = Path("/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2")

psrs = build_pint_psrs(dataset_dir, ephem="DE436", bipm_version="BIPM2015")
if len(psrs) == 0:
    raise RuntimeError(
        "No pulsars were loaded. Check logs in "
        f"{dataset_dir/'pint_fix_logs'} and confirm tim/par paths."
    )
print(f"Using {len(psrs)} pulsars\n")

# --------------------------------------------------------------------------------------
# MCMC / QuickCW config (unchanged)
# --------------------------------------------------------------------------------------

# number of iterations (increase to 100 million - 1 billion for actual analysis)
N = 1e9

n_int_block = 10000  # iterations per block
save_every_n = 100000
N_blocks = np.int64(N // n_int_block)
fisher_eig_downsample = 2000

n_status_update = 100
n_block_status_update = np.int64(N_blocks // n_status_update)

assert N_blocks % n_status_update == 0
assert N % save_every_n == 0
assert N % n_int_block == 0

# Parallel tempering
T_max = 3.0
n_chain = 4

# white noise dictionary
noisefile = '/scratch/na00078/projects/IPTA_MDC2/noise_files/fit_psr_noise_dataset2.json'

# RN empirical distribution (None to disable)
rn_emp_dist_file = None

# pulsar distances file (None → use distances in psr objects)
psr_dist_file = None

# results will be saved here
savefile = '/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_detect_new.h5'

# targeted search params
cos_gwtheta = np.cos(0.6387905062299246)
gwphi = 3.3335788713091694
TargFreq = 3.7e-09

# Setup and start MCMC
chain_params = ChainParams(
    T_max, n_chain, n_block_status_update,
    freq_bounds=np.array([TargFreq - .5e-21, TargFreq + .5e-21]),
    n_int_block=n_int_block,
    save_every_n=save_every_n,
    fisher_eig_downsample=fisher_eig_downsample,
    rn_emp_dist_file=rn_emp_dist_file,
    savefile=savefile,
    thin=10,
    prior_draw_prob=0.2, de_prob=0.6, fisher_prob=0.3,
    dist_jump_weight=0.2, rn_jump_weight=0.3, gwb_jump_weight=0.1,
    common_jump_weight=0.2, all_jump_weight=0.2,
    fix_rn=False, zero_rn=False, fix_gwb=False, zero_gwb=False,
    cos_gwtheta_bounds=[cos_gwtheta - 1e-8, cos_gwtheta + 1e-8],
    gwphi_bounds=[gwphi - 1e-8, gwphi + 1e-8],
)

pta, mcc = QuickCW.QuickCW(
    chain_params, psrs,
    amplitude_prior='detection',
    psr_distance_file=psr_dist_file,
    noise_json=noisefile,
)

# Do the main MCMC iteration
mcc.advance_N_blocks(N_blocks)
