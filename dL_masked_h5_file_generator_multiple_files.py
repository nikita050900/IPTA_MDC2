import numpy as np
import h5py
import os
import sys

# ---------------------- CLI arguments ---------------------- #
if len(sys.argv) < 3:
    print("Usage: python dL_masked_h5_file_generator.py <infile> <target_dL>")
    sys.exit(1)

infile = sys.argv[1]
target_d_L = float(sys.argv[2])

print(f"Input file: {infile}")
print(f"Target d_L: {target_d_L} Mpc")

# ---------------------- Load data ---------------------- #
with h5py.File(infile, 'r') as f:
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][...]
    log_likelihood = f['log_likelihood'][...]
    par_names = [x.decode('utf-8') for x in f['par_names'][...]]

first_n_param = len(par_names)
print(f"Loaded {first_n_param} parameters from file.")

# ---------------------- Constants ---------------------- #
megaparsec = 3.086e+22   # m
speed_of_light = 299792458.0  # m/s
T_sun = 1.327124400e20 / speed_of_light**3  # G*M_sun/c^3
burnin = 0
thin = 1
d_L_percent_tolerance = 1.0

# ---------------------- Compute derived dL ---------------------- #
chain = samples_cold[0][burnin::thin, :first_n_param]
logL_chain = log_likelihood[0, burnin::thin]

h_amp = 10 ** chain[:, 4]
fgw   = 10 ** chain[:, 3]
mc    = 10 ** chain[:, 5]

log10_d_L = np.log10(
    2 * (mc * T_sun)**(5/3) * (np.pi * fgw)**(2/3) / h_amp * speed_of_light / megaparsec
)
d_L = 10**log10_d_L  # Mpc

d_L_min = target_d_L * (1 - d_L_percent_tolerance / 100)
d_L_max = target_d_L * (1 + d_L_percent_tolerance / 100)

mask = (d_L >= d_L_min) & (d_L <= d_L_max)
masked_samples = chain[mask]
masked_logL = logL_chain[mask]
masked_dL = d_L[mask]
masked_indices = np.where(mask)[0]

print(f"\nMask keeps {len(masked_indices)} / {len(d_L)} samples")
print(f"dL range allowed: [{d_L_min:.3f}, {d_L_max:.3f}] Mpc")

# ---------------------- Save masked HDF5 ---------------------- #
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

    parname_bytes = np.array([s.encode('utf-8') for s in par_names])
    f_out.create_dataset('par_names', data=parname_bytes)

    f_out.attrs.update({
        'target_dL_Mpc': target_d_L,
        'dL_percent_tolerance': d_L_percent_tolerance,
        'dL_min_Mpc': d_L_min,
        'dL_max_Mpc': d_L_max,
        'burnin': burnin,
        'thin': thin,
        'n_params': first_n_param,
    })

print(f"\n💾 Saved masked file → {outfile}")

# ---------------------- Verify saved file ---------------------- #
with h5py.File(outfile, 'r') as fcheck:
    print("\nSaved contents:")
    for k, v in fcheck.items():
        print(f"  {k}: {v.shape}")
    print("Attributes:")
    for a, v in fcheck.attrs.items():
        print(f"  {a}: {v}")
