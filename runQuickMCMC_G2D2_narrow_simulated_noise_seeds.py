#!/usr/bin/env python

import numpy as np
np.seterr(all='raise')
import pickle, os, argparse
import enterprise
from enterprise_extensions import deterministic
from enterprise.pulsar import Pulsar
import QuickCW.QuickCW_G2D2 as QuickCW
from QuickCW.QuickMCMCUtils import ChainParams

# ---------------------------------------------------------
# Load pulsars
# ---------------------------------------------------------
data_pkl = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/psr_objects/G2D2_simulated_all_pulsars.pkl"

with open(data_pkl, "rb") as f:
    psrs = pickle.load(f)

print(f"Loaded {len(psrs)} pulsars from {data_pkl}")

# ---------------------------------------------------------
# OPTIONAL: Add extra noise using NOISE_SEED
# ---------------------------------------------------------
noise_seed_str = os.getenv("NOISE_SEED")

if noise_seed_str is not None:
    seed = int(noise_seed_str)
    rng = np.random.default_rng(seed)
    for psr in psrs:
        # add Gaussian white noise consistent with TOA uncertainties
        psr.residuals += psr.toaerrs * rng.standard_normal(psr.toas.size)
    print(f"Injected synthetic white noise with NOISE_SEED={seed}")
else:
    print("NOISE_SEED not set — running with original residuals only.")

# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Run QuickMCMC narrow detection search on G2D2 simulated data.")
parser.add_argument("--save_filename", type=str, required=True)
args = parser.parse_args()

save_dir = "/scratch/na00078/projects/IPTA_MDC2/h5_files"
savefile = os.path.join(save_dir, args.save_filename)

print(f"Saving output to {savefile}")
print("Running detection prior only.")

# ---------------------------------------------------------
# Injection constants (from MDC2)
# ---------------------------------------------------------
cos_gwtheta = np.cos(0.6387905062299246)
gwphi       = 3.3335788713091694
TargFreq    = 3.7e-09

# ---------------------------------------------------------
# MCMC config
# ---------------------------------------------------------
N = int(1e9)         # total iterations
n_int_block = 10000
save_every_n = 100000
N_blocks = N // n_int_block
n_status_update = 100
n_block_status_update = N_blocks // n_status_update

chain_params = ChainParams(
    T_max=3.,
    n_chain=4,
    n_block_status_update=n_block_status_update,
    freq_bounds=np.array([TargFreq - 1e-21, TargFreq + 1e-21]),
    n_int_block=n_int_block,
    save_every_n=save_every_n,
    fisher_eig_downsample=2000,
    rn_emp_dist_file=None,
    savefile=savefile,
    thin=10,
    prior_draw_prob=0.2,
    de_prob=0.6,
    fisher_prob=0.3,
    dist_jump_weight=0.2,
    rn_jump_weight=0.3,
    gwb_jump_weight=0.1,
    common_jump_weight=0.2,
    all_jump_weight=0.2,
    cos_gwtheta_bounds=[cos_gwtheta - 1e-8, cos_gwtheta + 1e-8],
    gwphi_bounds=[gwphi - 1e-8,   gwphi + 1e-8],
)

noise_json = "/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/noise_files/fit_psr_noise_dataset2.json"

pta, mcc = QuickCW.QuickCW(
    chain_params,
    psrs,
    amplitude_prior="detection",
    psr_distance_file=None,
    noise_json=noise_json,
)

mcc.advance_N_blocks(N_blocks)
