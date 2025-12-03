#!/usr/bin/env python3
import numpy as np
import h5py
import os
import sys

# ---------------------- CLI arguments ---------------------- #
if len(sys.argv) < 3:
    print("Usage: python dL_masked_h5_file_generator_multiple_files.py <infile> <target_dL>")
    sys.exit(1)

infile = sys.argv[1]
target_d_L = float(sys.argv[2])

print(f"Input file: {infile}")
print(f"Target d_L: {target_d_L} Mpc")

# ---------------------- Constants ---------------------- #
megaparsec = 3.086e+22   # m
speed_of_light = 299792458.0  # m/s
T_sun = 1.327124400e20 / speed_of_light**3
dL_percent_tolerance = 1.0
burnin = 0
thin = 1

# ---------------------- Load ONLY metadata ---------------------- #
with h5py.File(infile, 'r') as f:
    Ts = f['T-ladder'][...]
    par_names = [x.decode('utf-8') for x in f['par_names'][...]]
    nchain, nsamp, npar = f['samples_cold'].shape

print(f"File contains {nsamp} samples × {npar} parameters")
print("Streaming in chunks...\n")

# ---------------------- Derived target range ---------------------- #
d_L_min = target_d_L * (1 - dL_percent_tolerance / 100)
d_L_max = target_d_L * (1 + dL_percent_tolerance / 100)

# ---------------------- Prepare outputs ---------------------- #
masked_samples = []
masked_logL = []
masked_dL = []
masked_indices = []

chunk = 1_000_000  # safely fits in RAM

with h5py.File(infile, "r") as f:

    dset_samples = f["samples_cold"]
    dset_logL = f["log_likelihood"]

    out_index_counter = 0

    for start in range(0, nsamp, chunk):
        end = min(start + chunk, nsamp)
        print(f"Processing samples {start:,} → {end:,}")

        # read chunk for cold chain only
        samp_block = dset_samples[0, start:end, :]     # shape: (chunk, npar)
        logL_block = dset_logL[0, start:end]           # shape: (chunk,)

        # extract needed parameters
        h_amp = 10 ** samp_block[:, 4]
        fgw   = 10 ** samp_block[:, 3]
        mc    = 10 ** samp_block[:, 5]

        # compute d_L for chunk
        log10_d_L = np.log10(
            2 * (mc * T_sun)**(5/3) *
            (np.pi * fgw)**(2/3) / h_amp *
            speed_of_light / megaparsec
        )
        d_L = 10 ** log10_d_L

        # mask this chunk
        mask = (d_L >= d_L_min) & (d_L <= d_L_max)
        idx = np.where(mask)[0]

        masked_samples.append(samp_block[idx])
        masked_logL.append(logL_block[idx])
        masked_dL.append(d_L[idx])
        masked_indices.append(start + idx)

# ---------------------- Concatenate all kept samples ---------------------- #
masked_samples = np.concatenate(masked_samples, axis=0)
masked_logL = np.concatenate(masked_logL, axis=0)
masked_dL = np.concatenate(masked_dL, axis=0)
masked_indices = np.concatenate(masked_indices, axis=0)

print(f"\nMask keeps {len(masked_indices)} / {nsamp} samples")

# ---------------------- Save output ---------------------- #
outdir = "/scratch/na00078/projects/IPTA_MDC2/h5_files/dl_masked"
os.makedirs(outdir, exist_ok=True)

base = os.path.basename(infile).replace(".h5", f"_dLmasked_{target_d_L:.3f}Mpc.h5")
outfile = os.path.join(outdir, base)

with h5py.File(outfile, 'w') as f_out:
    f_out.create_dataset('samples_masked', data=masked_samples)
    f_out.create_dataset('logL_masked', data=masked_logL)
    f_out.create_dataset('dL_masked', data=masked_dL)
    f_out.create_dataset('mask_indices', data=masked_indices)
    f_out.create_dataset('T_ladder', data=Ts)

    f_out.create_dataset('par_names', data=np.array([x.encode("utf-8") for x in par_names]))

    f_out.attrs.update({
        'target_dL_Mpc': target_d_L,
        'dL_percent_tolerance': dL_percent_tolerance,
        'dL_min_Mpc': d_L_min,
        'dL_max_Mpc': d_L_max,
        'burnin': burnin,
        'thin': thin,
        'n_params': npar,
    })

print(f"\nSaved masked file → {outfile}")
